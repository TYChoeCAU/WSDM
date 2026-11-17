"""Frozen S3 DF core using local selected models and exact recorded caches."""
from pathlib import Path
import csv,json,os,time,copy,gc
import numpy as np,torch
from .paths import ROOT,OUT,S3,resolve
MATCHED=PRIOR=S3
from . import runtime as rt
from studies.s3.cross_common import sha,identity,load,save,atomic_json,read_csv,write_csv,MASKS,FEATURES
from diagnostics.graph_fusion.evaluate import ranking_metrics
from diagnostics.revision_evidence.common import per_group_metrics

def evidence(dataset,backbone,seed,protocol,method,family='matched'):
    path=rt.result_path(dataset,backbone,seed,protocol,method,family)
    result=json.loads(path.read_text())
    if result.get('status') not in ('COMPLETE','COMPLETED'):raise ValueError('Selected source result is incomplete')
    actual=(result['dataset'],result['backbone'],int(result.get('training_seed',result.get('seed',-1))),result['protocol'],result['method'])
    if actual!=(dataset,backbone,int(seed),protocol,method):raise ValueError('Selected source result coordinate differs')
    for key in ('checkpoint_path','trained_only_checkpoint_path'):
        if result.get(key):result[key]=str(resolve(result[key]))
    return path,result

def selected_adapter(dataset,backbone,seed,modules,device='cuda'):
    _,base=evidence(dataset,backbone,seed,'F','base')
    resultpath=rt.backbone_result(dataset,backbone,seed)
    result=json.loads(resultpath.read_text())
    expected=base['backbone_checkpoint_sha256']
    if result['checkpoint_sha256']!=expected:raise ValueError('Selected backbone/checkpoint lineage differs')
    checkpoint=resolve(result['checkpoint'])
    if sha(checkpoint)!=expected:raise ValueError('Selected checkpoint SHA differs')
    bb=modules['backbone'];bundle=rt.common.read_master(dataset);cp=load(checkpoint)
    if cp['identity']['master_signature']!=bundle['signature']['sha256'] or cp['identity']!=result['identity']:
        raise ValueError('Backbone master/checkpoint/run identity differs')
    data=load(MATCHED/'master'/dataset/'dataset.pt');config=copy.deepcopy(cp['config'])
    config['device']=torch.device(device);config['use_gpu']=torch.device(device).type=='cuda'
    model=bb.get_model(config['model'])(config,data).to(device)
    model.load_state_dict(cp['state_dict']);model.load_other_parameter(cp.get('other_parameter'))
    adapter=bb.CrossAdapter(model,data,config,backbone,checkpoint,cp['identity'],predict_batch_size=262144)
    adapter.df_selected_result_path=str(resultpath)
    return adapter
def runtime():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return rt.install()[0]


def digest_split(split):
    from studies.s3.master import tensor_digest
    return tensor_digest({k:split[k] for k in ('user_ids','candidate_ids','labels','valid_mask','positive_count')})


def records_from_split(split):
    records=[]
    for i,user in enumerate(split['user_ids']):
        mask=split['valid_mask'][i]
        records.append(dict(user=int(user),items=split['candidate_ids'][i][mask].clone(),
            base_score_raw=split['s'][i][mask].clone(),positive_items=torch.as_tensor(split['positive_items'][i]).long(),
            positive_count=int(split['positive_count'][i]),labels=split['labels'][i][mask].clone()))
    return records


def read_f_cache(dataset,seed,modules):
    paths=rt.cache_paths(dataset,'MF_BPR',seed,'F')
    proof=json.loads(paths['complete_path'].read_text())
    if sha(paths['feature_cache_path'])!=proof['feature_cache_sha256']:raise ValueError('MF F feature cache byte hash differs')
    cache=load(paths['feature_cache_path']);split=cache['test'];sig=cache['signature']
    if identity(sig)!=proof['identity_hash'] or digest_split(split)!=split['candidate_digest']:raise ValueError('MF F cache identity/candidates differ')
    _,base=evidence(dataset,'MF_BPR',seed,'F','base')
    if base['backbone_checkpoint_sha256']!=sig['checkpoint_sha256'] or base['candidate_digest']!=split['candidate_digest']:raise ValueError('MF F evidence does not match cache')
    expected=base['model_identity']['cache_signature']
    if identity(expected)!=identity(sig):raise ValueError('MF F model cache identity differs')
    # Existing fixed pool is read directly. Never invoke its fallback builder.
    poolpath=MATCHED/'master'/dataset/'fixed_W_pools.pt';pool=load(poolpath)
    master=rt.common.read_master(dataset)
    if pool['signature']['master_digest']!=master['signature']['sha256']:raise ValueError('Frozen pool master differs')
    records=pool['test'];truth=[(int(r['user']),torch.as_tensor(r['positive_items']).long().tolist()) for r in records]
    if identity(truth)!=sig['candidate_signature']['pool']['full_truth_digests']['test']:raise ValueError('MF F full truth differs from frozen supervision')
    if len(records)!=len(split['user_ids']):raise ValueError('DF user count differs from existing F pool')
    violations=0
    for j,r in enumerate(records):
        ids=split['candidate_ids'][j][split['valid_mask'][j]].long();targets=torch.as_tensor(r['positive_items']).long()
        if int(split['user_ids'][j])!=int(r['user']):raise ValueError('User order differs')
        if not torch.equal(targets,torch.as_tensor(split['positive_items'][j]).long()):raise ValueError('Relevant item identities differ')
        if int(split['positive_count'][j])!=len(targets):raise ValueError('Full denominator differs')
        if not torch.equal(torch.isin(ids,targets),split['labels'][j][split['valid_mask'][j]]):raise ValueError('Labels differ')
        violations+=int(torch.isin(ids,torch.as_tensor(r['excluded_items']).long()).sum())
        if len(ids)!=len(torch.unique(ids)) or bool((ids<=0).any()) or bool((ids>=int(master['item_num'])).any()):raise ValueError('Invalid candidate IDs')
    if violations:raise ValueError('F candidate exclusion violations')
    checks=dict(cached_F_signature_verified=True,existing_full_catalog_top20_check=sig['candidate_signature']['direct_F_top20_matches'],
        full_truth_exact=True,candidate_label_digest_exact=True,exclusion_violations=violations,
        all_methods_same_candidate_digest=split['candidate_digest'],feature_cache_path=str(paths['feature_cache_path']),
        feature_cache_sha256=proof['feature_cache_sha256'],complete_path=str(paths['complete_path']),
        retriever_checkpoint_sha256=sig['checkpoint_sha256'],fixed_pool_path=str(poolpath),fixed_pool_sha256=sha(poolpath))
    return cache,paths,master,checks


def score_split(adapter,split):
    valid=split['valid_mask'];users=split['user_ids'][:,None].expand_as(split['candidate_ids'])
    output=torch.zeros_like(split['s'])
    output[valid]=adapter.score_candidates(users[valid].numpy().copy(),split['candidate_ids'][valid].numpy().copy())
    return output


def prepared_features(dataset,seed,modules,cache,master,adapter,directory,compute=True):
    features=modules['features'];old=cache['test']
    signature=cache['signature'];nfm_path,nfm=evidence(dataset,'NFM',seed,'F','base')
    nfm_sig=nfm['model_identity']['cache_signature']
    if adapter.checkpoint_sha256!=nfm_sig['checkpoint_sha256']:raise ValueError('S3 NFM checkpoint differs')
    if nfm_sig['master_digest']!=signature['master_digest'] or nfm_sig['graph_fingerprint']!=signature['graph_fingerprint']:raise ValueError('Retriever/scorer split or graph differs')
    graph,fingerprint,userdeg,itemdeg=features._A_graph(master,adapter)
    if fingerprint!=signature['graph_fingerprint']:raise ValueError('A graph differs from prior')
    neighbors,metadata,table_seconds=features._neighbors(dataset,graph,fingerprint)
    for family,key in [('minimal','minimal_neighbors_sha256'),('nonparamgc','native_neighbors_sha256')]:
        if metadata[family]['sha256']!=nfm_sig[key]:raise ValueError('Frozen NFM neighbor table differs')
    if int(nfm_sig['neighbor_k'])!=2:raise ValueError('Unexpected neighborhood K')
    phase={**old,'s':score_split(adapter,old)} if compute else {**old,'s':load(directory/'nfm_scores.pt')}
    if compute:save(directory/'nfm_scores.pt',phase['s'])
    records=records_from_split(phase);item_num=int(master['item_num'])
    packed=np.concatenate([np.int64(r['user'])*item_num+r['items'].long().numpy() for r in records])
    raw=np.concatenate([r['base_score_raw'].numpy() for r in records]);keys,first=np.unique(packed,return_index=True);base=raw[first]
    registry=dict(dataset=dataset,seed=seed,protocol='DF',retriever_checkpoint_sha256=signature['checkpoint_sha256'],
        scorer_checkpoint_sha256=adapter.checkpoint_sha256,candidate_digest=old['candidate_digest'],master_digest=signature['master_digest'],
        graph_fingerprint=fingerprint,neighbor_metadata=metadata,feature_order=list(FEATURES),code_sha256=sha(Path(__file__)))
    if (directory/'pairs/progress.json').exists():
        registry['code_sha256']=json.loads((directory/'pairs/progress.json').read_text())['identity']['code_sha256']
    if not compute and not (directory/'pairs/progress.json').exists():raise ValueError('Prepared DF statistics missing, no graph regeneration in learned phase')
    stats,progress=features._pair_statistics(adapter,keys,base,item_num,neighbors,float(nfm_sig['alpha']),directory/'pairs',registry)
    target=features._phase_features(records,keys,stats,item_num,userdeg,itemdeg)
    if digest_split(target)!=old['candidate_digest']:raise ValueError('DF features changed candidate identity/full truth')
    df_sig={**nfm_sig,'protocol':'DF','test_protocol':'DF','source_protocols':['U','W'],
        'retriever_backbone':'MF_BPR','retriever_checkpoint_sha256':signature['checkpoint_sha256'],
        'phase_candidate_digests':{'test':target['candidate_digest']},'pair_statistics_identity':progress['identity_hash'],
        'scaler_policy':'source checkpoint only, no DF fit','candidate_origin_cache':str(rt.cache_paths(dataset,'MF_BPR',seed,'F')['feature_cache_path'])}
    # No scaler is computed from DF. Source checkpoints supply all scaler/t.
    return target,records,df_sig,progress,registry


def save_prediction(directory,name,scores,split,source=None):
    per=per_group_metrics(scores,split);metrics=ranking_metrics(scores,split)
    if any(abs(float(per[k].mean())-metrics[k])>1e-3 for k in metrics):raise ValueError('Per-user vs common evaluator differs')
    dest=directory/(name+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
    # Compact numeric predictions only. Candidate IDs remain in referenced MF F cache.
    np.savez_compressed(dest,scores=scores.cpu().numpy().astype(np.float32),user_ids=split['user_ids'].numpy(),
        **{k:v.numpy().astype(np.float64) for k,v in per.items()})
    full=split['positive_count'].double();hit=(split['labels']&split['valid_mask']).sum(1).double()
    return dict(metrics=metrics,candidate_recall50=float(torch.where(full>0,hit/full,torch.zeros_like(full)).mean()),
        oracle_ndcg20=ranking_metrics(split['labels'].float(),split)['ndcg@20'],n_users=len(full),
        candidate_digest=split['candidate_digest'],predictions_path=str(dest),predictions_sha256=sha(dest),**(source or {}))
