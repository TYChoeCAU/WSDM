"""A-only graph features shared by the matched W/F candidate slates.

Existing graph tables, sufficient-statistic computation and native NonParamGC
remain unchanged. A sorted (u,i) registry prevents repeated graph computation
across phases/protocols for one frozen backbone checkpoint.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from .cross_common import RUN, ROOT, CONFIG, FEATURES, coord, identity, sha, load, save, atomic_json, read_csv, deadline
import numpy as np
import torch
from diagnostics.graph_fusion.feature_cache import _compute_records
from graph_rerank.graph import training_graph, graph_fingerprint, GraphScorerConfig, load_or_build_tables

from studies.corrected import native as _old
native_scores = _old.native_scores
eval_batches = _old.eval_batches
tensor_digest = _old.tensor_digest
PHASES = ('train','valid','test')
PROTOCOLS = ('W','F')
PAIR_COLUMNS = ('h','Q','V','N_eff','npgc_scores')
NUMERICAL_ATOL = 1e-3


def external_applied(dataset, backbone, seed, protocol, method='prm_principled'):
    path=CONFIG/'master'/(dataset+'_external.json')
    original=json.loads(path.read_text())[method]
    if method=='original_nonparamgc' and int(original['neighbor_k'])!=2:
        raise ValueError('S3 requires the recorded k=2 graph setting')
    return {**original,'dataset':dataset,'backbone':backbone,'seed':int(seed),'training_seed':int(seed),
        'protocol':protocol,'train_protocol':protocol,'validation_protocol':protocol,'test_protocol':protocol,
        'parent_setting_dataset':dataset,'parent_setting_backbone':'NFM','parent_setting_seed':2020,
        'parent_settings_source':str(path),'parent_settings_source_sha256':sha(path),
        'parent_actual_settings_sha256':identity(original)}


@contextmanager
def exclusive_lock(path):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as handle:
        while True:
            deadline()
            try:
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(.2)
        try:yield
        finally:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def ndarray_digest(array):
    array=np.ascontiguousarray(array)
    digest=hashlib.sha256();digest.update(str(array.dtype).encode());digest.update(str(array.shape).encode())
    digest.update(memoryview(array).cast('B'));return digest.hexdigest()


def _save_array(path, value):
    path=Path(path);temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('wb') as handle:np.save(handle,value,allow_pickle=False)
    temporary.replace(path)


def _A_graph(master, adapter):
    data=master['train'];dataset=adapter.dataset
    if (int(dataset.user_num),int(dataset.item_num))!=(int(master['user_num']),int(master['item_num'])):
        raise ValueError('Adapter and frozen master ID spaces differ')
    supplied=SimpleNamespace(inter_feat={dataset.uid_field:data['user_ids'],dataset.iid_field:data['item_ids']},
        uid_field=dataset.uid_field,iid_field=dataset.iid_field,user_num=master['user_num'],item_num=master['item_num'])
    graph=training_graph(supplied);fingerprint=graph_fingerprint(graph)
    # The model's join data must also be the A-only dataset, never outer A+B.
    if graph_fingerprint(training_graph(dataset))!=fingerprint:
        raise ValueError('Adapter graph contains edges beyond frozen backbone-fit A')
    item=np.asarray(graph.sum(0)).ravel().astype(np.int64)
    user=np.asarray(graph.sum(1)).ravel().astype(np.int64)
    if not np.array_equal(item,np.asarray(master['item_degree'])) or not np.array_equal(user,np.asarray(master['user_degree'])):
        raise ValueError('Frozen A degrees differ from graph reconstruction')
    return graph,fingerprint,user,item


def _neighbors(dataset,graph,fingerprint):
    directory=RUN/'master'/dataset/'neighbors';directory.mkdir(parents=True,exist_ok=True)
    result={};metadata={};started=time.monotonic()
    with exclusive_lock(directory/'tables.lock'):
        for family in ('minimal','nonparamgc'):
            deadline();path=directory/(family+'_k2.npz')
            config=GraphScorerConfig.preset(family,k=2)
            result[family]=load_or_build_tables(graph,path,scorer_config=config)
            with np.load(path,allow_pickle=False) as archive:meta=json.loads(str(archive['metadata'].item()))
            if meta['fingerprint']!=fingerprint or meta['scorer_config']['k']!=2 or meta['scorer_config']['family']!=family:
                raise ValueError('Graph table lineage mismatch')
            metadata[family]={'path':str(path),'sha256':sha(path),'metadata':meta}
    return result,metadata,time.monotonic()-started


def _unique_pairs(candidates,item_num):
    key_parts=[];base_parts=[];truth_digests={};groups={}
    for protocol in PROTOCOLS:
        for phase in PHASES:
            records=candidates[protocol][phase];keys=[];scores=[];truth=[]
            if not records:raise ValueError('Empty candidate phase: '+protocol+'/'+phase)
            for record in records:
                ids=torch.as_tensor(record['items']).long().cpu().numpy()
                raw=torch.as_tensor(record['base_score_raw']).float().cpu().numpy()
                positives=torch.as_tensor(record['positive_items']).long().cpu().numpy()
                if len(ids)>50 or len(ids)!=len(raw) or len(np.unique(ids))!=len(ids) or np.any((ids<=0)|(ids>=item_num)):
                    raise ValueError('Invalid Top-50 IDs')
                if not np.isfinite(raw).all() or np.any((raw < -NUMERICAL_ATOL)|(raw > 1+NUMERICAL_ATOL)):
                    raise ValueError('Every graph/reranker input must share the probability score domain')
                if len(np.unique(positives))!=len(positives) or int(record['positive_count'])!=len(positives):
                    raise ValueError('Full positive denominator is inconsistent')
                if not np.array_equal(np.isin(ids,positives),torch.as_tensor(record['labels']).bool().cpu().numpy()):
                    raise ValueError('Candidate labels differ from full relevant targets')
                keys.append(np.int64(record['user'])*item_num+ids);scores.append(raw)
                truth.append((int(record['user']),positives.tolist()))
            key_parts.append(np.concatenate(keys));base_parts.append(np.concatenate(scores))
            truth_digests[protocol+'/'+phase]=identity(truth);groups[protocol+'/'+phase]=len(records)
            expected=candidates['signature'].get('pool',{}).get('full_truth_digests',{}).get(phase)
            if expected is not None and truth_digests[protocol+'/'+phase]!=expected:
                raise ValueError('Candidate full truth changed from fixed shared supervision')
    keys=np.concatenate(key_parts);base=np.concatenate(base_parts)
    unique,first=np.unique(keys,return_index=True);unique_base=base[first]
    maximum=0.
    for start in range(0,len(keys),262144):
        positions=np.searchsorted(unique,keys[start:start+262144])
        maximum=max(maximum,float(np.max(np.abs(unique_base[positions]-base[start:start+262144]),initial=0.)))
    if maximum>NUMERICAL_ATOL:raise ValueError('Same frozen pair has inconsistent scores across candidate phases')
    return unique,unique_base,dict(total_candidate_pairs=len(keys),unique_pairs=len(unique),
        graph_pair_duplicates_avoided=len(keys)-len(unique),repeated_pair_score_max_error=maximum,
        full_truth_digests=truth_digests,groups=groups)


def _union_records(keys,base,item_num):
    """Pack pair-independent graph queries into the original width-50 API."""
    records=[];cursor=0
    users=keys//item_num
    while cursor<len(keys):
        end=min(cursor+50,int(np.searchsorted(users,users[cursor],side='right')))
        records.append({'user':int(users[cursor]),'items':torch.from_numpy((keys[cursor:end]%item_num).astype(np.int32)),
                        'base_score_raw':torch.from_numpy(base[cursor:end].copy())})
        cursor=end
    return records


def _pair_statistics(adapter,keys,base,item_num,neighbors,alpha,directory,signature):
    directory.mkdir(parents=True,exist_ok=True)
    meta_path=directory/'progress.json';keys_path=directory/'pair_keys.npy';base_path=directory/'base_scores.npy';stats_path=directory/'statistics.npy'
    pair_identity={**signature,'keys_sha256':ndarray_digest(keys),'base_scores_sha256':ndarray_digest(base),
                   'pair_columns':list(PAIR_COLUMNS),'pairs':len(keys),'alpha':float(alpha)}
    signature_hash=identity(pair_identity);started=time.monotonic();maximum=0.
    with exclusive_lock(directory/'preparation.lock'):
        if meta_path.exists():
            progress=json.loads(meta_path.read_text())
            if progress['identity_hash']!=signature_hash:raise ValueError('Existing pair cache belongs to another checkpoint/graph/pool')
            if not np.array_equal(np.load(keys_path,mmap_mode='r'),keys) or not np.array_equal(np.load(base_path,mmap_mode='r'),base):
                raise ValueError('Existing pair-key registry changed')
            stats=np.load(stats_path,mmap_mode='r+')
            if stats.shape!=(len(keys),len(PAIR_COLUMNS)):raise ValueError('Pair statistic shape differs')
            maximum=float(progress.get('native_direct_parity_max_error',0.))
            if progress['status']=='COMPLETED':
                if sha(stats_path)!=progress['statistics_sha256']:raise ValueError('Stored pair statistics changed')
                return stats,progress
            start=int(progress['next_pair'])
        else:
            _save_array(keys_path,keys);_save_array(base_path,base)
            stats=np.lib.format.open_memmap(stats_path,mode='w+',dtype='float32',shape=(len(keys),len(PAIR_COLUMNS)))
            stats[:]=np.nan;stats.flush();start=0
            atomic_json(meta_path,dict(status='PREPARING',identity_hash=signature_hash,identity=pair_identity,next_pair=0))
        for first in range(start,len(keys),8192):
            deadline();last=min(first+8192,len(keys));records=_union_records(keys[first:last],base[first:last],item_num)
            computed=_compute_records(adapter,*neighbors['minimal'],records,positive_items=[torch.empty(0,dtype=torch.long) for _ in records])
            native,error=native_scores(adapter,neighbors['nonparamgc'],records,float(alpha));maximum=max(maximum,error)
            mask=computed['valid_mask']
            values=torch.stack([computed[name][mask] for name in ('h','Q','V','N_eff')]+[native[mask]],dim=1).numpy()
            if values.shape!=(last-first,len(PAIR_COLUMNS)) or not np.isfinite(values).all():raise ValueError('Invalid pair statistics')
            stats[first:last]=values;stats.flush()
            atomic_json(meta_path,dict(status='PREPARING',identity_hash=signature_hash,identity=pair_identity,next_pair=last,
                native_direct_parity_max_error=maximum,seconds_this_resume=time.monotonic()-started))
            print(json.dumps({'stage':'shared_pair_graph_statistics','pairs_done':last,'pairs_total':len(keys),'seconds':time.monotonic()-started}),flush=True)
        progress=dict(status='COMPLETED',identity_hash=signature_hash,identity=pair_identity,next_pair=len(keys),
            native_direct_parity_max_error=maximum,seconds_this_resume=time.monotonic()-started,
            keys_path=str(keys_path),base_scores_path=str(base_path),statistics_path=str(stats_path),statistics_sha256=sha(stats_path))
        atomic_json(meta_path,progress)
        return stats,progress


def _phase_features(records,keys,statistics,item_num,user_degree,item_degree):
    size=len(records);width=50
    fields={name:torch.zeros((size,width),dtype=torch.float32) for name in ('s','h','Q','V','N_eff','npgc_scores')}
    ids=torch.zeros((size,width),dtype=torch.int32);mask=torch.zeros((size,width),dtype=torch.bool);labels=torch.zeros_like(mask)
    users=torch.tensor([int(r['user']) for r in records],dtype=torch.int32)
    positives=[];counts=[]
    for row,record in enumerate(records):
        if row%2048==0:deadline()
        items=torch.as_tensor(record['items']).int();n=len(items);raw=torch.as_tensor(record['base_score_raw']).float()
        target=torch.as_tensor(record['positive_items']).long();positives.append(target);counts.append(len(target))
        ids[row,:n]=items;mask[row,:n]=True;fields['s'][row,:n]=raw
        labels[row,:n]=torch.isin(items.long(),target)
        if n:
            pair=np.int64(record['user'])*item_num+items.long().numpy();index=np.searchsorted(keys,pair)
            if not np.array_equal(keys[index],pair):raise ValueError('Final slate pair missing from registry')
            values=torch.from_numpy(np.asarray(statistics[index]).copy())
            for column,name in enumerate(PAIR_COLUMNS):fields[name][row,:n]=values[:,column]
    fields['h']=torch.where(fields['Q']>0,fields['h'],fields['s'])
    pi=torch.from_numpy(np.log1p(item_degree)).float()[ids.long()]
    pu=torch.from_numpy(np.log1p(user_degree)).float()[users.long()]
    features=torch.stack((fields['s'],pi,torch.log1p(fields['Q']),fields['N_eff'],fields['h'],fields['h']-fields['s'],fields['V']),dim=-1)
    result={**fields,'candidate_ids':ids,'valid_mask':mask,'labels':labels,'positive_count':torch.tensor(counts,dtype=torch.long),
        'user_ids':users,'item_num':int(item_num),'features':features,'joint_features':features,
        'gamma':fields['Q'].clamp(max=1.),'has_graph_evidence':fields['Q']>0,
        'item_log_degree':pi,'user_log_degree':pu[:,None].expand_as(pi),
        'item_degree':torch.as_tensor(item_degree)[ids.long()],'user_degree':torch.as_tensor(user_degree)[users.long()],
        'positive_items':positives}
    result['candidate_digest']=tensor_digest({k:result[k] for k in ('user_ids','candidate_ids','labels','valid_mask','positive_count')})
    return result


def _complete(path,input_hash):
    marker=path/'complete.json'
    if not marker.exists():return None
    value=json.loads(marker.read_text())
    if value['input_identity_hash']!=input_hash:raise ValueError('Completed feature cache inputs changed')
    for name in ('feature_cache','slate_cache'):
        target=path/(name+'.pt')
        if not target.is_file() or sha(target)!=value[name+'_sha256']:raise ValueError('Completed cache file changed: '+name)
    return {'feature_cache_path':str(path/'feature_cache.pt'),'slate_cache_path':str(path/'slate_cache.pt'),
            'summary':value['summary'],'identity_hash':value['identity_hash'],'reused':True}


def prepare_features(adapter,dataset,backbone,seed,candidates):
    """Publish both protocol caches without fitting any residual/backbone."""
    deadline();started=time.monotonic();adapter=candidates.get('adapter',adapter)
    master_path=RUN/'master'/dataset/'bundle.pt';master=load(master_path)
    if candidates['signature']['master_digest']!=master['signature']['sha256']:
        raise ValueError('Candidate/master lineage differs')
    if candidates['signature']['checkpoint_sha256']!=adapter.checkpoint_sha256:
        raise ValueError('Candidate backbone checkpoint differs')
    graph,fingerprint,user_degree,item_degree=_A_graph(master,adapter)
    tables,table_meta,table_seconds=_neighbors(dataset,graph,fingerprint)
    settings=external_applied(dataset,backbone,seed,'W',method='original_nonparamgc');alpha=float(settings['alpha'])
    sources=[Path(__file__),ROOT/'src/diagnostics/graph_fusion/feature_cache.py',ROOT/'src/graph_rerank/graph.py',
             ROOT/'src/graph_rerank/reranker.py',ROOT/'src/studies/corrected/native.py']
    common=dict(version='matched-A-only-joint7-features-v1',dataset=dataset,backbone=backbone,training_seed=int(seed),
        master_digest=master['signature']['sha256'],master_bundle_sha256=sha(master_path),
        split_digests=master['split_digests'],checkpoint_sha256=adapter.checkpoint_sha256,
        candidate_signature=candidates['signature'],graph_fingerprint=fingerprint,graph_source='A observed binary edges only',
        minimal_neighbors_sha256=table_meta['minimal']['sha256'],native_neighbors_sha256=table_meta['nonparamgc']['sha256'],
        neighbor_k=2,alpha=alpha,feature_order=list(FEATURES),score_domain='sigmoid(raw model logit), shared by all paths',
        tie_rule='score descending then mapped item ID ascending',candidate_top_m=50,source_hashes={str(p):sha(p) for p in sources},
        parent_nonparamgc_settings_sha256=settings['parent_actual_settings_sha256'])
    inputs={protocol:{**common,'protocol':protocol,'train_protocol':protocol,'validation_protocol':protocol,'test_protocol':protocol} for protocol in PROTOCOLS}
    outputs={};directories={p:coord(dataset,backbone,seed,p,area='cache') for p in PROTOCOLS}
    for protocol in PROTOCOLS:
        previous=_complete(directories[protocol],identity(inputs[protocol]))
        if previous is not None:outputs[protocol]=previous
    if len(outputs)==2:return outputs
    before=adapter.state_digest()
    keys,base,pair_summary=_unique_pairs(candidates,int(master['item_num']))
    if not len(keys):raise ValueError('No candidate pairs exist for either protocol')
    statistics,pair_meta=_pair_statistics(adapter,keys,base,int(master['item_num']),tables,alpha,
        coord(dataset,backbone,seed,area='pair_statistics'),common)
    histories={}
    for phase in PHASES:
        for record in candidates['W'][phase]:
            u=int(record['user']);history=torch.as_tensor(record['history']).long().tolist()
            if u in histories and histories[u]!=history:raise ValueError('PRM history changes across phase, expected A only')
            histories[u]=history
    expected_histories={}
    for u,i in zip(master['train']['user_ids'].tolist(),master['train']['item_ids'].tolist()):
        expected_histories.setdefault(int(u),set()).add(int(i))
    for user,history in histories.items():
        if set(history)!=expected_histories.get(user,set()):raise ValueError('External history is not exactly A observed items')
    for protocol in PROTOCOLS:
        if protocol in outputs:continue
        deadline();directory=directories[protocol];directory.mkdir(parents=True,exist_ok=True)
        cache={'summary':{},'item_degree':torch.from_numpy(item_degree),'user_degree':torch.from_numpy(user_degree)}
        for phase in PHASES:
            phase_start=time.monotonic();records=candidates[protocol][phase]
            cache[phase]=_phase_features(records,keys,statistics,int(master['item_num']),user_degree,item_degree)
            cache['summary'][phase]={**candidates['summary'].get(protocol+'/'+phase,{}),
                'native_direct_parity_max_error':pair_meta['native_direct_parity_max_error'],
                'seconds_scatter_shared_statistics':time.monotonic()-phase_start}
        train=cache['train'];values=train['features'][train['valid_mask']]
        if not len(values):raise ValueError('Train scaler has no valid candidates')
        scaler={'mean':values.mean(0),'std':values.std(0,unbiased=False).clamp_min(1e-6),
            'score_scale':max(float(train['s'][train['valid_mask']].std(unbiased=False)),1e-6),'feature_names':list(FEATURES)}
        cache['scaler']=scaler
        candidate_digests={phase:cache[phase]['candidate_digest'] for phase in PHASES}
        signature={**inputs[protocol],'phase_candidate_digests':candidate_digests,
            'train_scaler_digest':tensor_digest({'mean':scaler['mean'],'std':scaler['std']}),
            'residual_scale_t':scaler['score_scale'],'pair_statistics_identity':pair_meta['identity_hash']}
        cache['signature']=signature
        eligible=(train['labels'].sum(1)>0)&(train['labels'].sum(1)<train['valid_mask'].sum(1))
        cache['summary'].update(pairwise_eligible_groups=int(eligible.sum()),pairwise_excluded_groups=int((~eligible).sum()),
            shared_pair_preparation=pair_summary,shared_graph_table_seconds=table_seconds,
            shared_graph_query_seconds=pair_meta['seconds_this_resume'],feature_preparation_elapsed_seconds=time.monotonic()-started,
            graph_fingerprint=fingerprint,train_predictions_out_of_sample=True,graph_source='A only',
            histories='A only for train/valid/test',all_known_relevant_outside_top50_retained=True)
        if not eligible.any():raise ValueError('No pairwise-eligible B refinement group')
        config={**candidates['model_config'],'candidate_capacity':50,'user_num':int(master['user_num']),'item_num':int(master['item_num'])}
        slate={'signature':signature,'preparation_signature':signature,'preparation_summary':cache['summary'],
            'backbone_identity':adapter.identity,'model_config':config,'user_num':int(master['user_num']),'item_num':int(master['item_num']),
            'train_samples':candidates[protocol]['train'],'train_histories':histories,
            'valid_batches':eval_batches(candidates[protocol]['valid'],'valid'),
            'test_batches':eval_batches(candidates[protocol]['test'],'test'),
            'evaluation_histories':histories,'evaluation_histories_by_split':{'valid':histories,'test':histories}}
        save(directory/'feature_cache.pt',cache);save(directory/'slate_cache.pt',slate)
        adapter.assert_frozen_and_unchanged()
        if adapter.state_digest()!=before:raise ValueError('Backbone changed during graph feature preparation')
        proof=dict(status='COMPLETED',input_identity_hash=identity(inputs[protocol]),identity_hash=identity(signature),
            signature=signature,summary=cache['summary'],feature_cache_sha256=sha(directory/'feature_cache.pt'),
            slate_cache_sha256=sha(directory/'slate_cache.pt'),minimal_graph=table_meta['minimal'],native_graph=table_meta['nonparamgc'],
            shared_pair_metadata_path=str(coord(dataset,backbone,seed,area='pair_statistics')/'progress.json'),
            numerical_validation_atol=NUMERICAL_ATOL,backbone_state_unchanged=True,
            candidate_score_parity_checks=candidates.get('checks',{}))
        atomic_json(directory/'cache_lineage.json',proof);atomic_json(directory/'complete.json',proof)
        outputs[protocol]={'feature_cache_path':str(directory/'feature_cache.pt'),'slate_cache_path':str(directory/'slate_cache.pt'),
            'summary':cache['summary'],'identity_hash':identity(signature),'reused':False}
        del cache,slate,train,values;gc.collect()
    return outputs
