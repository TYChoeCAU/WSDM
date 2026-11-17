"""Label-free Stage C2 inference on saved source-selected Res(p) checkpoints.

Only OUT is written. No model selection, fitting, labels, or ranking metrics.
Each module uses its own stored train scaler and residual scale t. Quantile
interventions replace raw log1p(item-degree) before the original transform.
"""
from pathlib import Path
import argparse, gc, hashlib, json, sys, time
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

OUT = PRIOR = None
def configure(s3_run, out):
    global PRIOR, OUT
    PRIOR, OUT = Path(s3_run), Path(out)
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'audit').mkdir(exist_ok=True)
from diagnostics.revision_evidence.common import JointResidualMLP

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def tensor_sha(*values):
    h=hashlib.sha256()
    for v in values:
        v=torch.as_tensor(v).detach().cpu().contiguous()
        h.update(str(tuple(v.shape)).encode());h.update(str(v.dtype).encode());h.update(v.numpy().tobytes())
    return h.hexdigest()

def load(path):
    return torch.load(path,map_location='cpu',weights_only=False,mmap=True)

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def find_result(index,ds,bb,seed,fit,family='matched'):
    q=index[(index.dataset==ds)&(index.backbone==bb)&(index.training_seed==seed)&(index.method=='P')&(index.analysis_family==family)]
    q=q[q.protocol.eq(fit)] if family=='matched' else q[q.source_protocol.eq(fit)]
    assert len(q)==1
    row=q.iloc[0];path=Path(row.result_path)
    assert sha(path)==row.source_result_sha256
    z=json.loads(path.read_text());z['_result_path']=str(path)
    return z

def label_free_test(cache_path):
    cache=load(cache_path)
    # Explicit whitelist: no relevance/labels are passed to the diagnostic.
    test=cache['test']
    rows={key:test[key] for key in ('features','s','valid_mask','user_ids','candidate_ids','candidate_digest')}
    signature=cache['signature']
    return rows,signature

@torch.inference_mode()
def outputs(model,z,batch):
    out=torch.empty(len(z),dtype=z.dtype)
    for start in range(0,len(z),batch):out[start:start+batch]=model(z[start:start+batch])
    return out

def list_spearman(p,adj,mask):
    """Equal user weight, average ranks for ties; omit undefined correlations."""
    rp=rankdata(np.where(mask,p,np.nan),axis=1,method='average',nan_policy='omit')
    ra=rankdata(np.where(mask,adj,np.nan),axis=1,method='average',nan_policy='omit')
    n=mask.sum(axis=1)
    rp=np.where(mask,rp-(n[:,None]+1)/2,0)
    ra=np.where(mask,ra-(n[:,None]+1)/2,0)
    numer=(rp*ra).sum(axis=1);denom=np.sqrt((rp*rp).sum(axis=1)*(ra*ra).sum(axis=1))
    defined=(denom>0)&(n>=2)
    rho=np.full(len(n),np.nan);rho[defined]=numer[defined]/denom[defined]
    return rho,defined

@torch.inference_mode()
def evaluate(cp,model,rows,result,eval_role,eval_protocol,batch):
    valid=rows['valid_mask'].bool();raw=rows['features'];n_users=len(valid)
    # Valid candidate rows only: padding does not enter quantiles or means.
    raw_valid=raw[valid]
    assert raw_valid.shape[1]==7 and torch.isfinite(raw_valid).all()
    mean=cp['train_scaler']['mean'];std=cp['train_scaler']['std'];mask=torch.tensor(cp['feature_mask'],dtype=raw.dtype)
    t=float(cp['residual_scale_t']);is_identity=cp['epoch']==-1
    z=((raw_valid-mean)/std)*mask
    observed_g=outputs(model,z,batch)
    adjustment=t*observed_g
    observed_scores=rows['s'][valid]+adjustment
    prediction=load(result['predictions_path'])
    assert torch.equal(prediction['user_ids'],rows['user_ids'])
    assert prediction['candidate_digest']==rows['candidate_digest']
    score_diff=float((observed_scores-prediction['scores'][valid]).abs().max())
    score_status='PASS' if score_diff<=1e-3 else 'FAIL'
    # Stop this module on a scientific mismatch; do not change or rerun it.
    if score_status!='PASS':raise ValueError(f'saved score mismatch {score_diff}')
    if is_identity:
        assert torch.equal(adjustment,torch.zeros_like(adjustment))
        assert torch.equal(observed_scores,rows['s'][valid])
        assert torch.equal(prediction['scores'][valid],rows['s'][valid])
    p=raw_valid[:,1].numpy()
    quantiles=np.arange(5,100,5)
    grid=np.percentile(p,quantiles,method='linear')
    pd_rows=[]
    for percentile,value in zip(quantiles,grid):
        # Set raw p first, then apply SOURCE train standardization and mask.
        raw_p=torch.tensor(float(value),dtype=raw.dtype)
        z[:,1]=((raw_p-mean[1])/std[1])*mask[1]
        predicted=outputs(model,z,batch)
        if is_identity:assert torch.equal(predicted,torch.zeros_like(predicted))
        mean_g=float(predicted.double().mean());mean_adjustment=float((predicted*t).double().mean())
        pd_rows.append(dict(percentile=int(percentile),p_grid=float(value),p_grid_applied_float32=float(raw_p),mean_adjustment_score_units=mean_adjustment,mean_adjustment_t_units=mean_g))
    del raw_valid,z,observed_scores,prediction
    pmat=raw[:,:,1].numpy();amat=np.full(valid.shape,np.nan,dtype=np.float32);amat[valid.numpy()]=adjustment.numpy()
    rho,defined=list_spearman(pmat,amat,valid.numpy())
    # Spearman is undefined for the zero/constant identity; do not call it 0.
    rho_mean=float(rho[defined].mean()) if defined.any() else None
    p10=next(x for x in pd_rows if x['percentile']==10);p90=next(x for x in pd_rows if x['percentile']==90)
    delta=p90['mean_adjustment_score_units']-p10['mean_adjustment_score_units']
    summary=dict(eval_role=eval_role,eval_protocol=eval_protocol,n_users=n_users,n_valid_rows=int(valid.sum()),p_feature='log1p_item_degree',quantile_method='numpy linear over all valid candidate rows',residual_scale_t=t,
      pd90_minus_pd10_score_units=delta,pd90_minus_pd10_t_units=p90['mean_adjustment_t_units']-p10['mean_adjustment_t_units'],pd90_minus_pd10_sign='zero' if delta==0 else 'positive' if delta>0 else 'negative',
      within_list_spearman_user_mean=rho_mean,spearman_defined_users=int(defined.sum()),spearman_undefined_users=int((~defined).sum()),spearman_constant_policy='undefined, excluded from mean; identity has no defined users',
      observed_mean_adjustment_score_units=float(adjustment.double().mean()),observed_mean_adjustment_t_units=float(observed_g.double().mean()),identity_selected=is_identity,identity_adjustment_exact_zero=is_identity and bool(torch.equal(adjustment,torch.zeros_like(adjustment))),
      saved_prediction_max_abs_diff=score_diff,score_validation_status=score_status,candidate_digest=rows['candidate_digest'],labels_used=False)
    return pd_rows,summary

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--threads',type=int,default=4);parser.add_argument('--batch',type=int,default=262144);a=parser.parse_args(argv)
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    index=pd.read_csv(PRIOR/'evidence_index.csv');index=index[index.regime.eq('R0')]
    folder=OUT/'c_pd_modules';folder.mkdir(exist_ok=True)
    full_rows=[];summaries=[];module_audits=[];failures=[];start=time.time()
    for ds in ('yelp2018','ml-1m'):
     for bb in ('MF_BPR','NFM'):
      for seed in range(2020,2025):
       for fit in ('U','W','F'):
        key=dict(dataset=ds,backbone=bb,training_seed=seed,fit_protocol=fit,regime='R0')
        slug=f'{ds}_{bb}_{seed}_{fit}';beg=time.time()
        try:
            source=find_result(index,ds,bb,seed,fit)
            cp_path=source['checkpoint_path'];cp_sha=sha(cp_path)
            assert cp_sha==source['checkpoint_sha256']
            cp=load(cp_path)
            assert cp['feature_names']==['s','log1p_item_degree','log1p_q','n_eff','h','h_minus_s','variance']
            assert cp['feature_mask']==[1,1,0,0,0,0,0]
            assert cp['cross_model_identity_hash']==source['model_identity_hash']
            assert cp['residual_scale_t']==source['residual_scale_t']
            assert tensor_sha(cp['train_scaler']['mean'],cp['train_scaler']['std'])==source['scaler_digest']
            assert (cp['epoch']==-1)==source['identity_selected']
            model=JointResidualMLP();model.load_state_dict(cp['state_dict']);model.eval()
            if cp['epoch']==-1:
                assert torch.count_nonzero(cp['state_dict']['network.4.weight'])==0
                assert torch.count_nonzero(cp['state_dict']['network.4.bias'])==0
            pd_local=[];summary_local=[];cache_paths=[]
            # F-fit has the same two requested row sets; compute once, emit both roles.
            roles=[('fit_test',fit,source)]
            if fit!='F':roles.append(('F_test','F',find_result(index,ds,bb,seed,fit,'frozen')))
            for role,ep,record in roles:
                rows,signature=label_free_test(record['cache_path'])
                assert signature['protocol']==ep
                assert signature['checkpoint_sha256']==source['backbone_checkpoint_sha256']
                assert signature['graph_fingerprint']==source['graph_digest']
                assert signature['master_digest']==source['master_split_digest']
                expected_cache=record['target_cache_identity'] if role=='F_test' and fit!='F' else source['cache_identity']
                assert identity(signature)==expected_cache
                pd0,s0=evaluate(cp,model,rows,record,role,ep,a.batch)
                lineage=dict(checkpoint_path=cp_path,checkpoint_sha256=cp_sha,train_scaler_digest=source['scaler_digest'],evaluation_cache_path=record['cache_path'],source_result_path=source['_result_path'],prediction_result_path=record['_result_path'])
                for q in pd0:pd_local.append(dict(**key,eval_role=role,eval_protocol=ep,identity_selected=cp['epoch']==-1,**q,residual_scale_t=float(cp['residual_scale_t']),n_valid_rows=s0['n_valid_rows'],labels_used=False,**lineage))
                summary_local.append(dict(**key,**s0,**lineage));cache_paths.append(record['cache_path'])
                del rows,signature;gc.collect()
            if fit=='F':
                pd_local.extend([dict(x,eval_role='F_test') for x in list(pd_local)])
                summary_local.append(dict(summary_local[0],eval_role='F_test'))
            assert sha(cp_path)==cp_sha
            pd.DataFrame(pd_local).to_csv(folder/(slug+'_grid.csv'),index=False)
            pd.DataFrame(summary_local).to_csv(folder/(slug+'_summary.csv'),index=False)
            audit=dict(**key,status='PASS',checkpoint_path=cp_path,checkpoint_sha256=cp_sha,checkpoint_unchanged=True,train_scaler_digest=source['scaler_digest'],selected_epoch=int(cp['epoch']),identity_selected=cp['epoch']==-1,cache_paths=cache_paths,labels_used=False,device='cpu',seconds=time.time()-beg)
            write_json(folder/(slug+'_audit.json'),audit)
            full_rows.extend(pd_local);summaries.extend(summary_local);module_audits.append(audit)
            print(json.dumps(dict(**key,status='PASS',seconds=round(time.time()-beg,2),completed_modules=len(module_audits),total_modules=60,elapsed_minutes=round((time.time()-start)/60,2))),flush=True)
        except Exception as exc:
            failure=dict(**key,status='FAIL_NO_RETRY',error=repr(exc),seconds=time.time()-beg)
            failures.append(failure);write_json(folder/(slug+'_failure.json'),failure)
            print(json.dumps(failure),flush=True)
        pd.DataFrame(full_rows).to_csv(OUT/'C_partial_dependence.csv',index=False)
        pd.DataFrame(summaries).to_csv(OUT/'C_partial_dependence_summary.csv',index=False)
        write_json(OUT/'audit/c_pd_progress.json',dict(completed_modules=len(module_audits),failed_modules=len(failures),expected_modules=60,elapsed_minutes=(time.time()-start)/60,failures=failures))
        gc.collect()
    pd.DataFrame(module_audits).to_csv(OUT/'audit/c_checkpoint_audit.csv',index=False)
    write_json(OUT/'audit/c_pd_status.json',dict(status='PASS' if len(module_audits)==60 and not failures else 'FAIL',completed_modules=len(module_audits),expected_modules=60,grid_rows=len(full_rows),expected_grid_rows=2280,evaluation_summaries=len(summaries),expected_evaluation_summaries=120,labels_used=False,new_training=0,device='cpu',seconds=time.time()-start,failures=failures))


