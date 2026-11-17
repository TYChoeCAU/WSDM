"""Public orchestration of the recorded S3 numerical functions."""
from pathlib import Path
import gc,json,os,time
from .s3_common import *

def initialize():
    verify_configuration()
    RUN.mkdir(parents=True,exist_ok=True)
    plan=json.loads((CONFIG/'analysis_plan.json').read_text())
    path=RUN/'analysis_plan_w2w3.json'
    if path.exists()and json.loads(path.read_text())!=plan:raise ValueError('Analysis plan changed in an existing run')
    if not path.exists():atomic_json(path,plan)

def verify_cache(dataset,backbone,seed,protocol):
    paths=cache_paths(dataset,backbone,seed,protocol);proof=json.loads(paths['complete_path'].read_text())
    for name in ('feature_cache','slate_cache'):
        if sha(paths[name+'_path'])!=proof[name+'_sha256']:raise ValueError('Cache bytes changed: '+name)
    sig=proof['signature'];master=read_master(dataset)
    for k,v in dict(dataset=dataset,backbone=backbone,training_seed=int(seed),protocol=protocol).items():
        if sig[k]!=v:raise ValueError('Candidate coordinate mismatch: '+k)
    if sig['master_digest']!=master['signature']['sha256']:raise ValueError('Candidate master mismatch')
    bb=json.loads((coord(dataset,backbone,seed,area='backbones')/'result.json').read_text())
    if bb['checkpoint_sha256']!=sig['checkpoint_sha256']or sha(bb['checkpoint'])!=bb['checkpoint_sha256']:
        raise ValueError('Candidate backbone mismatch')
    return paths

def split(dataset):
    from .master import prepare_master
    initialize();return prepare_master(dataset)

def pools(dataset):
    from .cross_candidates import get_pools
    from .uniform_candidates import get_uniform_pools
    split(dataset);get_pools(dataset);get_uniform_pools(dataset)

def backbone(dataset,model,seed,device):
    initialize();pools(dataset);modules=configure_runtime()
    os.environ['S3_ALLOW_TRAINING']='1'
    return modules['backbone'].train_backbone(dataset,model,int(seed),device)

def candidates(dataset,model,seed,device):
    initialize();read_master(dataset);modules=configure_runtime()
    if not(coord(dataset,model,seed,area='backbones')/'result.json').is_file():
        raise FileNotFoundError('Run the backbone stage first')
    # W/F are prepared together; their exact pair registry is then reused by U.
    if not all(cache_paths(dataset,model,seed,p)['complete_path'].is_file()for p in ('W','F')):
        adapter=modules['backbone'].load_adapter(dataset,model,int(seed),device)
        prepared=modules['candidates'].prepare_candidates(adapter,dataset,model,int(seed))
        modules['features'].prepare_features(adapter,dataset,model,int(seed),prepared)
        del adapter,prepared;gc.collect()
    os.environ['S3_ALLOW_PREPARATION']='1'
    modules['uniform'].prepare_uniform(dataset,model,int(seed),device)
    for p in PROTOCOLS:verify_cache(dataset,model,seed,p)

def matched(dataset,model,seed,protocol,device,methods=None):
    import torch
    initialize();modules=configure_runtime();paths=verify_cache(dataset,model,seed,protocol)
    cache=load(paths['feature_cache_path']);slates=load(paths['slate_cache_path']);master=read_master(dataset)
    if cache['signature']!=slates['signature']or identity(cache['signature'])!=json.loads(paths['complete_path'].read_text())['identity_hash']:
        raise ValueError('Feature/slate/cache identity differs')
    for phase in ('train','valid','test'):
        part=cache[phase]
        digest=modules['features'].tensor_digest({k:part[k]for k in ('user_ids','candidate_ids','labels','valid_mask','positive_count')})
        if digest!=part['candidate_digest']:raise ValueError('Cached candidate tensor identity differs')
    for method in (METHODS if methods is None else methods):
        if method not in METHODS:raise ValueError('Outside the released 12-method scope')
        path=result_path(dataset,model,seed,protocol,method)
        if valid_result(path,identity(cache['signature'])):continue
        if path.exists():raise ValueError('Existing result failed lineage checks; preserve it and use a new run directory')
        app=applied_configuration(dataset,model,seed,protocol,method,master)
        directory=path.parent;directory.mkdir(parents=True,exist_ok=True);started=time.monotonic()
        fit=modules['models'].train_and_score(method,cache,slates,app,directory,device)
        modules['publisher'].publish(dataset,model,int(seed),protocol,method,cache,paths['feature_cache_path'],master,fit,directory,app,time.monotonic()-started)
        result=json.loads(path.read_text());result.update(study_id='S3',regime='R0',backbone_batch=4096,
            reranker_batch=app.get('batch_size'),backbone_training_seed=int(seed),reranker_training_seed=int(seed),
            outer_split_seed=2020,ab_split_seed=2020,candidate_seed=2020,method_id=method,
            mask_id=method if method in MASKS else None,selection_policy=result['training_summary']['selection_policy'],
            predictions_sha256=sha(result['predictions_path']),
            actual_new_fits=0 if method not in LEARNED or fit.get('reused_compatible_fit')else 1,
            actual_new_inference_passes=4 if method in MASKS else 2 if method=='prm_principled'else 0)
        if method in MASKS:result['parameter_count']=801
        atomic_json(path,result)
        if not valid_result(path,identity(cache['signature'])):raise ValueError('Published S3 result failed lineage checks')
        del fit;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()

def frozen(dataset,model,seed,source,device,methods=None):
    from .transfer_worker import execute
    initialize();verify_cache(dataset,model,seed,'F')
    return execute(dataset,model,int(seed),source,device,methods)

def index_results():
    from .transfer_worker import completed_transfer,transfer_path
    rows=[]
    for d in DATASETS:
      for b in BACKBONES:
       for s in SEEDS:
        for p in PROTOCOLS:
         for m in METHODS:
            path=result_path(d,b,s,p,m,require_complete=True)
            rows.append(dict(dataset=d,backbone=b,training_seed=s,seed=s,protocol=p,method=m,
                analysis_family='matched',source_protocol=p,regime='R0',result_path=str(path),
                backbone_batch=4096,reranker_batch=32 if m=='prm_principled'else 256 if m in MASKS else '',
                source_result_sha256=sha(path)))
        for p in ('U','W'):
         for m in LEARNED:
            path=transfer_path(d,b,s,p,m)/'result.json'
            if not completed_transfer(d,b,s,p,m):raise ValueError('Frozen result missing or incompatible: '+str(path))
            r=json.loads(path.read_text())
            if sha(r['predictions_path'])!=r['predictions_sha256']:raise ValueError('Frozen predictions changed')
            rows.append(dict(dataset=d,backbone=b,training_seed=s,seed=s,protocol=p+'_to_F',method=m,
                analysis_family='frozen',source_protocol=p,regime='R0',result_path=str(path),
                backbone_batch=4096,reranker_batch=32 if m=='prm_principled'else 256,source_result_sha256=sha(path)))
    write_csv(RUN/'evidence_index.csv',rows)
    return rows

def missing_statistics_inputs():
    """List absent cohort cells; present artifacts still undergo strict validation."""
    from .transfer_worker import transfer_path
    missing=[]
    for d in DATASETS:
      for b in BACKBONES:
       for seed in SEEDS:
        for protocol in PROTOCOLS:
         for method in METHODS:
            path=result_path(d,b,seed,protocol,method)
            if not path.is_file():missing.append(str(path.relative_to(RUN)))
        for protocol in ('U','W'):
         for method in LEARNED:
            path=transfer_path(d,b,seed,protocol,method)/'result.json'
            if not path.is_file():missing.append(str(path.relative_to(RUN)))
    return missing


def statistics():
    initialize();index_results()
    from .w2w3_extract import extract_existing,audit_all_users
    from .w2w3_analysis import analyze
    extract_existing();audit_all_users();analyze()
