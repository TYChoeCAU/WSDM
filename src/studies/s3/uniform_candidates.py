"""Uniform candidates on the exact frozen matched A/B cohort.

Only U's weights differ from the saved W prepool. This module never trains a
backbone, rebuilds a split, scores the entire catalog, or writes MATCHED files.
Production preparation requires the completion runner's explicit authorization.
"""
from __future__ import annotations
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

from .cross_common import RUN, MATCHED, ROOT, FEATURES, atomic_json, identity, load, save, sha, deadline, coord, cache_dir
from .cross_backbone import load_adapter
from .cross_candidates import records_digest
from .cross_features import (_A_graph, _pair_statistics, _phase_features, external_applied,
                            ndarray_digest, _save_array, eval_batches, tensor_digest, exclusive_lock, _old)
from studies.corrected.corrected_sampling import sample_negatives, group_rng, _record, rank_record

PHASES = ('train','valid','test')
VERSION = 'matched-AB-U-weight-only-v1'


def _read(path):
    return json.loads(Path(path).read_text())


def resample_record(record, item_num):
    """Reuse W entropy, B group boundaries, exclusions, history and full truth."""
    key = json.loads(record['group_key'])
    if key['protocol'] != 'W' or key['seed'] != 2020:
        raise ValueError('Expected the fixed matched W candidate key')
    entropy = {k:key[k] for k in ('dataset','seed','phase','user','slate_index','protocol')}
    targets = torch.as_tensor(record['positive_items']).long().tolist()
    excluded = set(torch.as_tensor(record['excluded_items']).long().tolist()) | set(targets)
    negative, meta = sample_negatives(group_rng(**entropy), item_num=item_num, excluded=excluded,
                                     count=100*len(targets), protocol='U')
    # The sampler's protocol label is U. Entropy deliberately stays W so the
    # only sampling intervention is the probability vector, not the RNG seed.
    meta.update(rng_key=record['group_key'], rng_protocol='W', rng_policy='inherit exact matched W entropy',
                intervention='uniform weights only', weight='uniform')
    result = _record(record['user'], targets, negative, torch.as_tensor(record['history']).tolist(),
                     record['group_key'], meta,
                     torch.as_tensor(record['original_target_items']).tolist() if 'original_target_items' in record else None)
    for name in ('excluded_items','phase','slate_index','history_feedback'):
        result[name] = copy.deepcopy(record[name])
    result['candidate_protocol'] = 'U'
    return result


def _phase_summary(records):
    return dict(groups=len(records), unique_users=len({r['user'] for r in records}),
                prepool_pairs=sum(len(r['items']) for r in records),
                full_relevant_targets=sum(r['positive_count'] for r in records),
                zero_relevant_groups=sum(r['positive_count']==0 for r in records),
                digest=records_digest(records))


def get_uniform_pools(dataset):
    # Both GPUs may request the same fixed pool for different backbone seeds.
    # Serialize publication only inside NEW, never touch the matched W lock.
    with exclusive_lock(RUN/'master'/dataset/'uniform_pool.lock'):
        return _get_uniform_pools_locked(dataset)


def _get_uniform_pools_locked(dataset):
    """Read fixed W records and publish a dataset-level U pool in NEW only."""
    deadline()
    master_path=MATCHED/'master'/dataset/'bundle.pt'
    w_path=MATCHED/'master'/dataset/'fixed_W_pools.pt'
    if not master_path.exists() or not w_path.exists():
        raise FileNotFoundError('The matched frozen master/W pool must already exist')
    manifest=_read(master_path.with_name('manifest.json'))
    if sha(master_path)!=manifest['bundle_sha256']:
        raise ValueError('Matched master file changed')
    w=load(w_path);master=load(master_path)
    if w['signature']['master_digest']!=master['signature']['sha256']:
        raise ValueError('Matched pool/master identity differs')
    inputs=dict(version=VERSION, dataset=dataset, master_digest=master['signature']['sha256'],
                matched_W_pool_signature=w['signature'], matched_W_pool_sha256=sha(w_path),
                sampler_source_sha256=sha(Path(sys.modules['studies.corrected.corrected_sampling'].__file__)),
                source_sha256=sha(Path(__file__)))
    path=RUN/'master'/dataset/'fixed_U_pools.pt';meta_path=path.with_suffix('.json')
    if path.exists():
        meta=_read(meta_path)
        if meta['input_identity']!=identity(inputs) or sha(path)!=meta['pool_sha256']:
            raise ValueError('Existing U pool is incompatible or changed')
        return load(path)
    result={p:[] for p in PHASES};stats={}
    for phase in PHASES:
        for index,record in enumerate(w[phase]):
            if index%2048==0:deadline()
            result[phase].append(resample_record(record,int(master['item_num'])))
        stats[phase]=_phase_summary(result[phase])
    signature=copy.deepcopy(w['signature']);signature.pop('sha256',None)
    signature.update(version=VERSION,protocol='U',weight='uniform',rng_protocol='W',
                     rng_policy='exact matched W group entropy, only sampling weights change',
                     matched_W_pool_sha256=sha(w_path),matched_W_pool_signature_sha256=w['signature']['sha256'],
                     phase_digests={p:stats[p]['digest'] for p in PHASES})
    full_truth={p:identity([(r['user'],r['positive_items'].tolist()) for r in result[p]]) for p in PHASES}
    if full_truth!=w['signature']['full_truth_digests']:
        raise ValueError('Uniform pool changed full relevant targets')
    from .cross_common import CONFIG
    expected=json.loads((CONFIG/'master'/(dataset+'_slate.json')).read_text())
    if signature['phase_digests']!=expected['fixed_U_phase_digests']:
        raise ValueError('Fixed U candidate records differ from recorded S3 evidence')
    signature['sha256']=identity(signature)
    result.update(signature=signature,summary=stats,cache_path=str(path),histories=w['histories'],
                  model_config=w['model_config'],input_identity=identity(inputs))
    save(path,result)
    atomic_json(meta_path,dict(status='COMPLETED',signature=signature,summary=stats,
                             input_identity=identity(inputs),pool_sha256=sha(path),original_W_unchanged=sha(w_path)==inputs['matched_W_pool_sha256']))
    return result


class SparseScoreAdapter:
    """Cached sampled pairs plus lazily memoized exact graph-reference pairs."""
    def __init__(self, adapter, keys, values, item_num):
        self.adapter=adapter;self.keys=np.asarray(keys);self.values=values;self.item_num=int(item_num)
        self.extra={};self.reference_requests=0;self.reference_unique_scored=0
        self.native_parity_predict_rows=0;self.native_parity_predict_calls=0
        if hasattr(adapter,'model'):
            owner=self
            class CountingProbabilityModel:
                def __getattr__(self,name):return getattr(adapter.model,name)
                def predict(self,interaction):
                    owner.native_parity_predict_rows+=len(interaction)
                    owner.native_parity_predict_calls+=1
                    return adapter.model.predict(interaction)
            # The unchanged native GraphReranker performs direct reference
            # checks outside score_candidates. Count those additional rows.
            self.model=CountingProbabilityModel()

    def __getattr__(self,name):
        return getattr(self.adapter,name)

    def score_candidates(self,users,items,supported_features=None):
        del supported_features
        u=torch.as_tensor(users).cpu().numpy();i=torch.as_tensor(items).cpu().numpy()
        if u.shape!=i.shape:raise ValueError('User/item score shape mismatch')
        shape=u.shape;encoded=(u.reshape(-1)*self.item_num+i.reshape(-1)).astype(np.int64)
        unique,inverse=np.unique(encoded,return_inverse=True);out=np.empty(len(unique),dtype=np.float32)
        indices=np.searchsorted(self.keys,unique);hit=indices<len(self.keys)
        hit[hit] &= self.keys[indices[hit]]==unique[hit]
        out[hit]=np.asarray(self.values[indices[hit]])
        missing=[];positions=[]
        for pos in np.flatnonzero(~hit):
            key=int(unique[pos]);self.reference_requests+=1
            if key in self.extra:out[pos]=self.extra[key]
            else:missing.append(key);positions.append(pos)
        if missing:
            missing=np.asarray(missing,dtype=np.int64)
            scores=self.adapter.score_candidates(missing//self.item_num,missing%self.item_num).reshape(-1).numpy()
            if len(scores)!=len(missing) or not np.isfinite(scores).all():raise ValueError('Invalid graph reference scores')
            out[np.asarray(positions)]=scores
            self.extra.update(zip(missing.tolist(),scores.tolist()));self.reference_unique_scored+=len(missing)
        return torch.from_numpy(out[inverse].reshape(shape).copy())


def _sampled_score_registry(adapter,pools,item_num,directory,signature):
    directory.mkdir(parents=True,exist_ok=True)
    key_parts=[np.int64(r['user'])*item_num+torch.as_tensor(r['items']).long().numpy() for p in PHASES for r in pools[p]]
    keys=np.unique(np.concatenate(key_parts));del key_parts
    record_identity={**signature,'sampled_pair_keys_sha256':ndarray_digest(keys),'sampled_unique_pairs':len(keys)}
    key_path=directory/'sampled_pair_keys.npy';value_path=directory/'sampled_pair_scores.npy';meta_path=directory/'sampled_score_progress.json'
    started=time.monotonic();score_identity=identity(record_identity)
    if meta_path.exists():
        meta=_read(meta_path)
        if meta['identity_hash']!=score_identity or not key_path.exists() or not value_path.exists():
            raise ValueError('Existing sampled score progress lacks matching inputs/files')
        if not np.array_equal(np.load(key_path,mmap_mode='r'),keys):raise ValueError('Sampled pair keys changed')
        values=np.load(value_path,mmap_mode='r+')
        if values.shape!=(len(keys),):raise ValueError('Sampled score array shape differs')
        if meta['status']=='COMPLETED':
            if sha(value_path)!=meta['scores_sha256']:raise ValueError('Sampled scores changed')
            return keys,values,{**meta,'sampled_pairs_scored_this_resume':0,'reused':True}
        cursor=int(meta['next_pair'])
        if not np.isfinite(values[:cursor]).all():raise ValueError('Sampled score resume has incomplete prefix')
    else:
        _save_array(key_path,keys)
        values=np.lib.format.open_memmap(value_path,mode='w+',dtype='float32',shape=(len(keys),))
        values[:]=np.nan;values.flush();cursor=0
    for first in range(cursor,len(keys),262144):
        deadline();last=min(first+262144,len(keys));part=keys[first:last]
        score=adapter.score_candidates(part//item_num,part%item_num).reshape(-1).numpy()
        if len(score)!=len(part) or not np.isfinite(score).all() or np.any((score < -1e-3)|(score > 1+1e-3)):
            raise ValueError('Frozen backbone score is not a finite probability')
        values[first:last]=score;values.flush()
        atomic_json(meta_path,dict(status='PREPARING',identity_hash=score_identity,identity=record_identity,next_pair=last))
    meta=dict(status='COMPLETED',identity_hash=score_identity,identity=record_identity,next_pair=len(keys),
              scores_sha256=sha(value_path),sampled_pairs_scored_this_resume=len(keys)-cursor,
              seconds_this_resume=time.monotonic()-started,full_catalog_scoring=False)
    atomic_json(meta_path,meta)
    return keys,values,meta


def _rank_pools(pools,adapter):
    result={};summary={}
    for phase in PHASES:
        result[phase]=[]
        for index,record in enumerate(pools[phase]):
            if index%2048==0:deadline()
            items=torch.as_tensor(record['items']).long()
            scores=adapter.score_candidates(torch.full_like(items,int(record['user'])),items)
            result[phase].append(rank_record(record,scores,50))
        rows=result[phase]
        summary[phase]=dict(groups=len(rows),candidate_pairs=sum(len(r['items']) for r in rows),
                           full_positive_count=sum(r['positive_count'] for r in rows),
                           zero_candidate_positive_groups=sum(int(r['labels'].sum())==0 for r in rows),
                           candidate_digest=records_digest(rows))
    return result,summary


def _top_pairs(records,item_num):
    keys=np.concatenate([int(r['user'])*item_num+torch.as_tensor(r['items']).long().numpy() for p in PHASES for r in records[p]])
    base=np.concatenate([r['base_score_raw'].numpy() for p in PHASES for r in records[p]])
    unique,first=np.unique(keys,return_index=True)
    return unique,base[first],dict(total_candidate_pairs=len(keys),unique_pairs=len(unique),
                                  graph_pair_duplicates_avoided=len(keys)-len(unique))


def _statistics_with_reuse(adapter,keys,base,item_num,tables,alpha,directory,common,matched_directory):
    """Reuse only exact matched-checkpoint pair statistics, query missing pairs."""
    statistics=np.empty((len(keys),5),dtype=np.float32);reused=np.zeros(len(keys),dtype=bool)
    reuse={};parent_meta=matched_directory/'progress.json'
    if parent_meta.exists():
        parent=_read(parent_meta);pid=parent['identity']
        for field in ('checkpoint_sha256','master_digest','graph_fingerprint','neighbor_k','alpha','minimal_neighbors_sha256','native_neighbors_sha256'):
            if pid[field]!=common[field]:raise ValueError('Existing matched graph-pair lineage differs: '+field)
        if parent['status']!='COMPLETED':raise ValueError('Matched graph pair cache is incomplete')
        parent_keys=np.load(parent['keys_path'],mmap_mode='r');parent_base=np.load(parent['base_scores_path'],mmap_mode='r')
        parent_stats=np.load(parent['statistics_path'],mmap_mode='r')
        if sha(parent['statistics_path'])!=parent['statistics_sha256'] or ndarray_digest(parent_keys)!=pid['keys_sha256'] or ndarray_digest(parent_base)!=pid['base_scores_sha256']:
            raise ValueError('Matched graph pair source changed')
        index=np.searchsorted(parent_keys,keys);reused=index<len(parent_keys);reused[reused]&=parent_keys[index[reused]]==keys[reused]
        error=float(np.max(np.abs(parent_base[index[reused]]-base[reused]),initial=0))
        if error>1e-3:raise ValueError('U and matched frozen pair scores differ')
        statistics[reused]=parent_stats[index[reused]]
        reuse=dict(path=str(parent_meta),sha256=sha(parent_meta),reused_pairs=int(reused.sum()),
                   shared_score_max_absolute_error=error,native_direct_parity_max_error=parent['native_direct_parity_max_error'])
    new_meta={};missing=~reused
    if missing.any():
        fresh,new_meta=_pair_statistics(adapter,keys[missing],base[missing],item_num,tables,alpha,directory/'new_graph_pairs',common)
        statistics[missing]=fresh
    if not np.isfinite(statistics).all():raise ValueError('Incomplete U graph features')
    merged_path=directory/'top50_graph_statistics.npy';_save_array(merged_path,statistics)
    return statistics,dict(reused_matched_pairs=reuse,new_pairs=int(missing.sum()),new_graph_metadata=new_meta,
                           native_direct_parity_max_error=max(float(reuse.get('native_direct_parity_max_error',0)),
                                                             float(new_meta.get('native_direct_parity_max_error',0))),
                           statistics_path=str(merged_path),statistics_sha256=sha(merged_path),identity_hash=identity(common))


def prepare_uniform(dataset,backbone,seed,device='cuda'):
    """Return standard cache paths. Called after the fixed W/F pools and A graph tables exist."""
    if os.environ.get('S3_ALLOW_PREPARATION')!='1':
        raise RuntimeError('U production preparation requires explicit candidate preparation invocation')
    deadline();started=time.monotonic();directory=cache_dir(dataset,backbone,seed,'U')
    pools=get_uniform_pools(dataset);master_path=MATCHED/'master'/dataset/'bundle.pt';master=load(master_path)
    bb_result=_read(MATCHED/'backbones'/dataset/backbone/f'seed_{seed}'/'result.json')
    if bb_result['status']!='COMPLETE':raise ValueError('Matched backbone is not complete')
    common_input=dict(version=VERSION,dataset=dataset,backbone=backbone,training_seed=int(seed),
                      master_digest=master['signature']['sha256'],master_bundle_sha256=sha(master_path),
                      checkpoint_sha256=bb_result['checkpoint_sha256'],pool_signature=pools['signature'],
                      source_hashes={str(Path(__file__)):sha(Path(__file__)),str(Path(__file__).with_name('cross_features.py')):sha(Path(__file__).with_name('cross_features.py'))})
    marker=directory/'complete.json'
    if marker.exists():
        proof=_read(marker)
        if proof['input_identity_hash']!=identity(common_input):raise ValueError('Existing U cache inputs differ')
        for name in ('feature_cache','slate_cache'):
            if sha(directory/(name+'.pt'))!=proof[name+'_sha256']:raise ValueError('Existing U cache changed')
        return dict(feature_cache_path=str(directory/'feature_cache.pt'),slate_cache_path=str(directory/'slate_cache.pt'),
                    summary=proof['summary'],identity_hash=proof['identity_hash'],reused=True)
    adapter=load_adapter(dataset,backbone,int(seed),device=device)
    if adapter.checkpoint_sha256!=bb_result['checkpoint_sha256']:raise ValueError('Selected backbone changed')
    before=adapter.state_digest();item_num=int(master['item_num']);work=coord(dataset,backbone,seed,area='uniform_preparation')
    keys,values,score_meta=_sampled_score_registry(adapter,pools,item_num,work,common_input)
    sparse=SparseScoreAdapter(adapter,keys,values,item_num)
    records,phase_summary=_rank_pools(pools,sparse)
    graph,fingerprint,user_degree,item_degree=_A_graph(master,adapter)
    tables={};table_meta={}
    for family in ('minimal','nonparamgc'):
        path=MATCHED/'master'/dataset/'neighbors'/(family+'_k2.npz')
        tables[family],table_meta[family]=_old.tables(path,fingerprint,2,family)
    settings=external_applied(dataset,backbone,int(seed),'U',method='original_nonparamgc');alpha=float(settings['alpha'])
    common={**common_input,'version':'matched-A-only-joint7-features-U-v1','split_digests':master['split_digests'],
            'candidate_signature':dict(pool=pools['signature'],master_digest=master['signature']['sha256'],checkpoint_sha256=adapter.checkpoint_sha256,
                                       sampled_score_identity=score_meta['identity_hash'],fixed_data_seed=2020),
            'graph_fingerprint':fingerprint,'graph_source':'A observed binary edges only',
            'minimal_neighbors_sha256':table_meta['minimal']['sha256'],'native_neighbors_sha256':table_meta['nonparamgc']['sha256'],
            'neighbor_k':2,'alpha':alpha,'feature_order':list(FEATURES),'candidate_top_m':50,
            'score_domain':'sigmoid(raw model logit), shared by all paths','tie_rule':'score descending then mapped item ID ascending',
            'protocol':'U','train_protocol':'U','validation_protocol':'U','test_protocol':'U',
            'parent_nonparamgc_settings_sha256':settings['parent_actual_settings_sha256']}
    top_keys,top_base,pair_summary=_top_pairs(records,item_num)
    statistics,pair_meta=_statistics_with_reuse(sparse,top_keys,top_base,item_num,tables,alpha,work,common,
        MATCHED/'pair_statistics'/dataset/backbone/f'seed_{seed}')
    cache={'summary':{},'item_degree':torch.from_numpy(item_degree),'user_degree':torch.from_numpy(user_degree)}
    for phase in PHASES:
        cache[phase]=_phase_features(records[phase],top_keys,statistics,item_num,user_degree,item_degree)
        cache['summary'][phase]={**phase_summary[phase],'native_direct_parity_max_error':pair_meta['native_direct_parity_max_error']}
    train=cache['train'];features=train['features'][train['valid_mask']]
    scaler={'mean':features.mean(0),'std':features.std(0,unbiased=False).clamp_min(1e-6),
            'score_scale':max(float(train['s'][train['valid_mask']].std(unbiased=False)),1e-6),'feature_names':list(FEATURES)}
    cache['scaler']=scaler
    signature={**common,'phase_candidate_digests':{p:cache[p]['candidate_digest'] for p in PHASES},
               'train_scaler_digest':tensor_digest({'mean':scaler['mean'],'std':scaler['std']}),
               'residual_scale_t':scaler['score_scale'],'pair_statistics_identity':pair_meta['identity_hash']}
    cache['signature']=signature
    eligible=(train['labels'].sum(1)>0)&(train['labels'].sum(1)<train['valid_mask'].sum(1))
    cache['summary'].update(pairwise_eligible_groups=int(eligible.sum()),pairwise_excluded_groups=int((~eligible).sum()),
        graph_source='A only',histories='A only for train/valid/test',train_predictions_out_of_sample=True,
        all_known_relevant_outside_top50_retained=True,graph_fingerprint=fingerprint,shared_pair_preparation=pair_summary,
        sampled_unique_pair_scores=len(keys),graph_reference_unique_pair_scores=sparse.reference_unique_scored,
        graph_reference_requests=sparse.reference_requests,native_parity_predict_rows=sparse.native_parity_predict_rows,
        native_parity_predict_calls=sparse.native_parity_predict_calls,
        sampled_pairs_scored_this_resume=score_meta.get('sampled_pairs_scored_this_resume',0),
        full_catalog_scoring=False,graph_pair_reuse=pair_meta,
        preparation_seconds=time.monotonic()-started)
    if not eligible.any():raise ValueError('Uniform pool has no pairwise-eligible training group')
    histories=pools['histories'];config={**pools['model_config'],'candidate_capacity':50,'user_num':int(master['user_num']),'item_num':item_num}
    slate=dict(signature=signature,preparation_signature=signature,preparation_summary=cache['summary'],backbone_identity=adapter.identity,
               model_config=config,user_num=int(master['user_num']),item_num=item_num,train_samples=records['train'],train_histories=histories,
               valid_batches=eval_batches(records['valid'],'valid'),test_batches=eval_batches(records['test'],'test'),
               evaluation_histories=histories,evaluation_histories_by_split={'valid':histories,'test':histories})
    adapter.assert_frozen_and_unchanged()
    if before!=adapter.state_digest():raise ValueError('Frozen backbone changed during U preparation')
    save(directory/'feature_cache.pt',cache);save(directory/'slate_cache.pt',slate)
    proof=dict(status='COMPLETED',input_identity_hash=identity(common_input),identity_hash=identity(signature),signature=signature,
               summary=cache['summary'],feature_cache_sha256=sha(directory/'feature_cache.pt'),slate_cache_sha256=sha(directory/'slate_cache.pt'),
               minimal_graph=table_meta['minimal'],native_graph=table_meta['nonparamgc'],backbone_state_unchanged=True,
               numerical_validation_atol=1e-3,score_registry=score_meta,graph_statistics=pair_meta,
               backbone_selection='existing fixed validation-W NDCG@20 selected checkpoint, not U-optimized')
    atomic_json(directory/'cache_lineage.json',proof);atomic_json(marker,proof)
    del adapter,sparse,cache,slate;gc.collect()
    return dict(feature_cache_path=str(directory/'feature_cache.pt'),slate_cache_path=str(directory/'slate_cache.pt'),
                summary=proof['summary'],identity_hash=proof['identity_hash'],reused=False)


