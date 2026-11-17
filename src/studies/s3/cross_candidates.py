"""Fixed A/B supervision pools and shared, exhaustive catalog scoring.

The temporary disk matrix contains each frozen (u,i) probability exactly once.
All W/F phases and exact graph queries share it. Only this run's temporary
matrix may be removed after both feature caches have completed.
"""
from __future__ import annotations
import copy
import hashlib
import json
import time
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from .cross_common import *
from .master import prepare_master, tensor_digest

from studies.corrected.corrected_sampling import _read_interactions, sample_negatives, group_rng, group_key, _record, rank_record


def historical_slate(dataset):
    from .cross_common import CONFIG
    path=CONFIG/'master'/(dataset+'_slate.json')
    recorded=json.loads(path.read_text())
    return int(recorded['targets_per_slate_cap']),recorded['model_config'],str(path)


def records_digest(records):
    h=hashlib.sha256()
    for r in records:
        h.update(str(r['user']).encode())
        for k in ('items','labels','positive_items','excluded_items','history'):
            v=torch.as_tensor(r[k]).contiguous();h.update(k.encode());h.update(v.numpy().tobytes())
    return h.hexdigest()


@lru_cache(maxsize=2)
def get_pools(dataset):
    """Common candidate RNG and B supervision for all backbone/training seeds."""
    master=prepare_master(dataset);path=RUN/'master'/dataset/'fixed_W_pools.pt'
    if path.exists():
        result=load(path)
        if result['signature']['master_digest']!=master['signature']['sha256']:raise RuntimeError('Pool/master mismatch')
        return result
    cap,model_config,cap_source=historical_slate(dataset)
    item_num=int(master['item_num']);degree=master['item_degree'].numpy()
    source={p:_read_interactions(master[p],item_num)[0] for p in ('train','refinement','valid','test')}
    history={u:list(items) for u,items in source['train'].items()}
    result={'train':[],'valid':[],'test':[],'histories':history,'model_config':model_config}
    stats={}
    for phase in ('train','valid','test'):
        current=source['refinement' if phase=='train' else phase]
        for user,items in sorted(current.items()):
            known=set(source['train'].get(user,{}))|set(source['refinement'].get(user,{}))
            if phase=='test':known|={i for i,y in source['valid'].get(user,{}).items() if y}
            # B grouping is fixed before training and retains low-rated groups
            # for exclusion/accounting. No second target split occurs here.
            ordered=list(items)
            if phase=='train':
                rng=group_rng(dataset=dataset,seed=2020,phase='B_slate_order',user=user,protocol='W')
                ordered=np.asarray(ordered,dtype=np.int64);rng.shuffle(ordered);ordered=ordered.tolist()
                groups=[ordered[n:n+cap] for n in range(0,len(ordered),cap)]
            else:groups=[[i for i,y in items.items() if y]]
            for index,original in enumerate(groups):
                targets=[i for i in original if items[i]]
                if phase!='train' and not targets:continue
                if phase=='train' and set(targets)&set(source['train'].get(user,{})):
                    raise RuntimeError('Refinement target entered backbone A')
                # Current truth stays eligible. Other known B observations,
                # including low-rated targets, never become synthetic negatives.
                exclusion=known-set(targets)
                key=dict(dataset=dataset,seed=2020,phase=phase,user=user,slate_index=index,protocol='W')
                negative,meta=sample_negatives(group_rng(**key),item_num=item_num,
                    excluded=known|set(targets),count=100*len(targets),protocol='W',item_degree=degree)
                r=_record(user,targets,negative,history.get(user,[]),group_key(**key),meta,original if phase=='train' else None)
                r['excluded_items']=torch.tensor(sorted(exclusion),dtype=torch.int32)
                r['phase']=phase;r['slate_index']=index
                result[phase].append(r)
            if user%5000==0:deadline()
        stats[phase]={'groups':len(result[phase]),'unique_users':len({r['user'] for r in result[phase]}),
                      'prepool_pairs':sum(len(r['items']) for r in result[phase]),
                      'full_relevant_targets':sum(r['positive_count'] for r in result[phase]),
                      'zero_relevant_groups':sum(r['positive_count']==0 for r in result[phase]),
                      'digest':records_digest(result[phase])}
        print(json.dumps({'stage':'fixed_pool','dataset':dataset,'phase':phase,**stats[phase]}),flush=True)
    signature={'master_digest':master['signature']['sha256'],'data_seed':2020,'training_seed_independent':True,
        'target_slate_cap':cap,'target_cap_source':cap_source,'negative_per_relevant':100,'weight':'1+A binary item degree',
        'B_slate_rule':'fixed shuffled B observations, legacy target cap, threshold-positive targets',
        'history_source':'A only, all observed edges','phase_digests':{p:stats[p]['digest'] for p in stats},
        'full_truth_digests':{p:identity([(r['user'],r['positive_items'].tolist()) for r in result[p]]) for p in stats},
        'exclusions':{'train':'A plus other known B observations','valid':'A+B','test':'A+B+validation relevant items'},
        'no_test_labels_in_training_candidates':True,'tie_rule':'score descending, mapped item ID ascending',
        'score_domain':'sigmoid(raw logit), all backbones and methods'}
    from .cross_common import CONFIG
    expected=json.loads((CONFIG/'master'/(dataset+'_slate.json')).read_text())
    if signature['phase_digests']!=expected['fixed_W_phase_digests']:
        raise ValueError('Fixed W candidate records differ from recorded S3 evidence')
    if signature['full_truth_digests']!=expected['fixed_W_full_truth_digests']:
        raise ValueError('Fixed candidate relevance differs from recorded S3 evidence')
    signature['sha256']=identity(signature)
    result.update(signature=signature,summary=stats,cache_path=str(path))
    save(path,result);atomic_json(path.with_suffix('.json'),{'signature':signature,'summary':stats})
    return result


def top_ids(scores,eligible,limit=50):
    eligible=np.asarray(eligible,dtype=np.int64)
    if len(eligible)>limit:
        values=scores[eligible];threshold=np.partition(values,len(values)-limit)[len(values)-limit]
        above=eligible[values>threshold];tied=eligible[values==threshold]
        eligible=np.concatenate((above,np.sort(tied)[:limit-len(above)]))
    return eligible[np.lexsort((eligible,-scores[eligible]))]


class MatrixAdapter:
    def __init__(self,adapter,path,shape):
        self.adapter=adapter;self.matrix=np.memmap(path,dtype='float32',mode='r',shape=shape)
        self.dataset=adapter.dataset;self.device=adapter.device;self.model=adapter.model
        self.config=adapter.config;self.uid_field=adapter.uid_field;self.iid_field=adapter.iid_field
        self.identity=adapter.identity;self.checkpoint_sha256=adapter.checkpoint_sha256
    def score_candidates(self,users,items):
        u=torch.as_tensor(users).cpu().numpy();i=torch.as_tensor(items).cpu().numpy()
        return torch.from_numpy(np.asarray(self.matrix[u,i]).copy())
    def state_digest(self):return self.adapter.state_digest()
    def assert_frozen_and_unchanged(self):return self.adapter.assert_frozen_and_unchanged()


def score_catalog(adapter,dataset,backbone,seed):
    out=coord(dataset,backbone,seed,area='catalog_scores');out.mkdir(parents=True,exist_ok=True)
    shape=(int(adapter.dataset.user_num),int(adapter.dataset.item_num));path=out/'probabilities.float32'
    signature={'checkpoint_sha256':adapter.checkpoint_sha256,'shape':shape,'dtype':'float32',
        'score_domain':'sigmoid(raw logit)','master_digest':prepare_master(dataset)['signature']['sha256']}
    signature['sha256']=identity(signature)
    metadata=out/'progress.json'
    if metadata.exists():
        progress=json.loads(metadata.read_text())
        if progress['signature']!=json.loads(json.dumps(signature)):raise RuntimeError('Catalog matrix lineage differs')
        start=int(progress['next_user'])
        if start>1 and not path.exists():
            raise RuntimeError('Catalog progress exists but its score matrix is absent. Completed W/F feature caches must be reused, never an empty replacement matrix.')
    else:
        start=1;atomic_json(metadata,{'signature':signature,'next_user':1,'status':'SCORING'})
    if path.exists() and path.stat().st_size!=shape[0]*shape[1]*4:
        raise RuntimeError('Catalog score matrix is truncated or has another shape')
    matrix=np.memmap(path,dtype='float32',mode='r+' if path.exists() else 'w+',shape=shape)
    matrix[0]=0;started=time.monotonic()
    # Same checkpoint and each nonpadding pair are scored once. Fixed inference
    # batches affect neither loss nor optimization. Output is disk-backed.
    for first in range(start,shape[0],32):
        deadline();users=np.arange(first,min(shape[0],first+32),dtype=np.int64)
        matrix[users,0]=0
        for left in range(1,shape[1],2048):
            items=np.arange(left,min(shape[1],left+2048),dtype=np.int64)
            if hasattr(adapter,'score_block'):
                value=adapter.score_block(users,items).cpu().numpy()
            else:
                value=adapter.score_candidates(np.repeat(users,len(items)),np.tile(items,len(users))).reshape(len(users),len(items)).numpy()
            if not np.isfinite(value).all():raise FloatingPointError('Nonfinite catalog probability')
            matrix[np.ix_(users,items)]=value
        if first%512==1 or first+32>=shape[0]:
            matrix.flush();atomic_json(metadata,{'signature':signature,'next_user':int(users[-1])+1,'status':'SCORING',
                'elapsed_this_resume_seconds':time.monotonic()-started})
            print(json.dumps({'stage':'full_catalog_scoring','dataset':dataset,'backbone':backbone,'seed':seed,
                             'users_done':int(users[-1]),'users_total':shape[0]-1,'seconds':time.monotonic()-started}),flush=True)
    matrix.flush();del matrix
    atomic_json(metadata,{'signature':signature,'next_user':shape[0],'status':'COMPLETE','bytes':path.stat().st_size})
    return MatrixAdapter(adapter,path,shape),signature


def prepare_candidates(adapter,dataset,backbone,seed):
    pools=get_pools(dataset);mapped,matrix_signature=score_catalog(adapter,dataset,backbone,seed)
    master=prepare_master(dataset);item_num=int(master['item_num']);catalog=np.arange(1,item_num,dtype=np.int64)
    result={'W':{},'F':{},'adapter':mapped,'model_config':pools['model_config'],'summary':{}}
    checks={'sampled_full_max_absolute_error':0.0,'direct_F_top20_checks':0,'direct_F_top20_matches':True}
    for phase in ('train','valid','test'):
        wr=[];fr=[]
        for index,record in enumerate(pools[phase]):
            row=mapped.matrix[record['user']]
            score=torch.from_numpy(row[record['items'].long().numpy()].copy())
            wr.append(rank_record(record,score,top_m=50))
            eligible=np.ones(item_num,dtype=bool);eligible[0]=False;eligible[record['excluded_items'].long().numpy()]=False
            selected=top_ids(row,np.flatnonzero(eligible),50)
            r={**record,'items':torch.tensor(selected,dtype=torch.int32),
                'labels':torch.tensor(np.isin(selected,record['positive_items'].numpy()),dtype=torch.float32),
                'base_score_raw':torch.tensor(row[selected].copy()),'sampling_meta':{'protocol':'F','eligible_count':int(eligible.sum()),
                    'all_eligible_scored':True,'forced_truth_insertion':False}}
            fr.append(r)
            if index<3:
                direct=adapter.score_candidates(np.full(len(selected),record['user']),selected)
                error=float((direct-r['base_score_raw']).abs().max()) if len(selected) else 0.0
                checks['sampled_full_max_absolute_error']=max(checks['sampled_full_max_absolute_error'],error)
                if error>1e-3:raise RuntimeError('Sampled/full scorer mismatch')
                brute=np.flatnonzero(eligible);brute=brute[np.lexsort((brute,-row[brute]))][:20]
                if not np.array_equal(selected[:20],brute):raise RuntimeError('Direct F Top20 differs from candidate Base Top20')
                checks['direct_F_top20_checks']+=1
            if index%1000==0:deadline()
        result['W'][phase]=wr;result['F'][phase]=fr
        for protocol,records in [('W',wr),('F',fr)]:
            result['summary'][protocol+'/'+phase]={'groups':len(records),'candidate_pairs':sum(len(r['items']) for r in records),
                'full_positive_count':sum(r['positive_count'] for r in records),
                'zero_candidate_positive_groups':sum(not bool(r['labels'].sum()) for r in records),
                'candidate_digest':records_digest(records)}
        print(json.dumps({'stage':'W_F_top50_complete','dataset':dataset,'backbone':backbone,'seed':seed,'phase':phase}),flush=True)
    result['checks']=checks
    result['signature']={'pool':pools['signature'],'catalog':matrix_signature,'master_digest':master['signature']['sha256'],
                        'split_digests':master['split_digests'],'checkpoint_sha256':adapter.checkpoint_sha256,
                        'fixed_data_seed':2020,'training_seed':seed,'score_parity_atol':1e-3,'score_parity_passed':True,
                        'direct_F_top20_matches':True}
    return result
