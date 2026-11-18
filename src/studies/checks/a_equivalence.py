"""Original direct official NPGC numerical equivalence check, on generated artifacts."""
import json,hashlib,time,copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from baselines.dnr.backbone_adapter import FrozenBackboneAdapter,load_checkpoint,checkpoint_for
from recbole.utils import get_model
from recbole.data.interaction import Interaction
from diagnostics.graph_fusion.evaluate import ranking_metrics
from graph_rerank.graph import training_graph,graph_fingerprint
import recbole_tag as upstream
OUT = S1 = S3 = DATA = None

def configure(s1_run,s3_run,data_root,out):
    global OUT,S1,S3,DATA
    OUT,S1,S3,DATA=Path(out),Path(s1_run),Path(s3_run),Path(data_root)
    (OUT/'audit/a_equivalence').mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)

def get_inputs(study):
    if study=='S1_fresh_U':
        directory=S1/'ml-1m/NFM/seed_2020';source=directory/'U_predictions.pt'
        saved=load(source);cp=checkpoint_for(S1/'backbones','ml-1m','NFM',2020)
        if cp is None and (directory/'completion.json').is_file():
            cp=Path(json.loads((directory/'completion.json').read_text())['checkpoint_path'])
        if cp is None:raise FileNotFoundError('S1 ML-1M/NFM/2020 checkpoint is required')
        adapter=FrozenBackboneAdapter(cp,device=torch.device('cpu'),sample_num=100,predict_batch_size=8192,data_root=DATA)
        assert adapter.checkpoint_sha256==saved['checkpoint_sha256']
        neighbor=directory/'native_neighbors.npz'
        return adapter.model,adapter.dataset,saved['split'],saved['scores']['original_nonparamgc'],cp,neighbor,source,adapter.config
    if study!='S3_W':raise ValueError(study)
    source=S3/'cache/ml-1m/MF_BPR/seed_2020/W/feature_cache.pt';cache=load(source)
    cp=S3/'backbones/ml-1m/MF_BPR/seed_2020/best_checkpoint.pt';payload=load(cp)
    assert sha(cp)==cache['signature']['checkpoint_sha256']
    cfg=copy.deepcopy(payload['config']);cfg['use_gpu']=False;cfg['device']=torch.device('cpu')
    ds=load(S3/'master/ml-1m/dataset.pt')
    model=get_model(cfg['model'])(cfg,ds);model.load_state_dict(payload['state_dict']);model.load_other_parameter(payload.get('other_parameter'))
    model=ProbabilityModel(model);neighbor=S3/'master/ml-1m/neighbors/nonparamgc_k2.npz'
    assert sha(neighbor)==cache['signature']['native_neighbors_sha256']
    return model,ds,cache['test'],cache['test']['npgc_scores'],cp,neighbor,source,cfg

def load(p):return torch.load(p,map_location='cpu',weights_only=False,mmap=True)


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def log(x):print(json.dumps(x,allow_nan=False),flush=True)


class ProbabilityModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__();self.native=model;self.device=torch.device('cpu')
    def predict(self,interaction):return torch.sigmoid(self.native.predict(interaction))


@torch.inference_mode()
def check(study):
    start=time.time();model,ds,split,expected,cp,neighbor,source,cfg=get_inputs(study)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    state_before={k:v.clone() for k,v in model.state_dict().items()}
    original_hashes={str(p):sha(p) for p in (cp,neighbor,source,Path(upstream.__file__))}
    args=SimpleNamespace(k=2,m=-.5,n=-.5,alpha=1.)
    log(dict(stage='direct_upstream_get_topk_index_weight',study=study,graph_rows=len(ds.inter_feat)))
    graph_infos=upstream.get_topk_index_weight(ds,args)
    table_report={}
    with np.load(neighbor) as z:
        for side,idx,wgt in [('user',0,1),('item',2,3)]:
            a,b=graph_infos[idx].numpy(),z[side+'_ids']; wa,wb=graph_infos[wgt].numpy(),z[side+'_weights']
            valid=np.isfinite(wa)&(a>=0)&(b>=0)
            table_report[side]={'exact_ids':bool(np.array_equal(a,b)),'different_id_rows':int(np.any(a!=b,axis=1).sum()),
                'max_abs_finite_weight_diff':float(np.max(np.abs(wa[valid]-wb[valid]))) if valid.any() else None,
                'upstream_nonfinite_weight_values':int((~np.isfinite(wa)).sum()),'padding_id0_nonfinite':bool((~np.isfinite(wa[0])).any())}
    upstream.args=args;upstream.model=model
    mask=split['valid_mask']; flat_positions=mask.nonzero();users=split['user_ids'][:,None].expand_as(mask)[mask].long();items=split['candidate_ids'][mask].long()
    output=torch.zeros_like(expected);values=[]
    used_u=graph_infos[0][users];used_i=graph_infos[2][items]
    if (used_u<0).any() or (used_i<0).any(): raise ValueError('Upstream selected padding in evaluated candidates')
    log(dict(stage='frozen_scoring',study=study,candidate_pairs=len(users),neighbor_pairs=len(users)*4,table_report=table_report))
    for first in range(0,len(users),8192):
        it=Interaction({'user_id':users[first:first+8192],'item_id':items[first:first+8192]})
        interaction,uw,iw,_,_=upstream.construct_new_interaction(graph_infos,it,ds)
        weight=uw.repeat(1,args.k)*torch.repeat_interleave(iw,args.k,dim=-1)
        weight[:,0]=(1-weight[:,1:]).sum(-1)
        values.append(upstream.TTA_pred(model.predict,interaction,weight,args).cpu())
        if first%81920==0:log(dict(stage='progress',study=study,done=min(first+8192,len(users)),seconds=time.time()-start))
    output[mask]=torch.cat(values);diff=(output[mask]-expected[mask]).abs()
    metrics=ranking_metrics(output,split);ref=ranking_metrics(expected,split)
    rows_unique=len(set(zip(ds.inter_feat[ds.uid_field].tolist(),ds.inter_feat[ds.iid_field].tolist())))
    result=dict(study=study,dataset='ml-1m',backbone='NFM' if study.startswith('S1') else 'MF_BPR',seed=2020,
        protocol='U_fresh' if study.startswith('S1') else 'W',source=str(source),checkpoint=str(cp),neighbor=str(neighbor),
        upstream_function='recbole_tag.get_topk_index_weight (direct call, unmodified)',
        upstream_aggregation='recbole_tag.construct_new_interaction + TTA_pred (direct calls, unmodified); active predict weight line106 copied exactly to avoid timing-only CUDA synchronize',
        evaluated_users=len(mask),evaluated_pairs=int(mask.sum()),graph_input_rows=len(ds.inter_feat),graph_unique_pairs=rows_unique,
        max_abs_score_diff=float(diff.max()),mean_abs_score_diff=float(diff.mean()),score_pairs_above_1e3=int((diff>1e-3).sum()),
        upstream_ndcg20=metrics['ndcg@20'],saved_ndcg20=ref['ndcg@20'],ndcg20_diff=metrics['ndcg@20']-ref['ndcg@20'],
        upstream_recall20=metrics['recall@20'],saved_recall20=ref['recall@20'],recall20_diff=metrics['recall@20']-ref['recall@20'],
        numerical_tolerance=1e-3,numeric_status='PASS' if max(float(diff.max()),abs(metrics['ndcg@20']-ref['ndcg@20']),abs(metrics['recall@20']-ref['recall@20']))<=1e-3 else 'FAIL',
        neighbor_identity_status='PASS' if all(v['exact_ids'] for v in table_report.values()) else 'FAIL',
        table_report=table_report,training_performed=False,experimental_mutation=False,seconds=time.time()-start,
        original_file_hashes=original_hashes,
        settings={k:cfg[k] for k in ['embedding_size','mlp_hidden_size','epochs','stopping_step','learning_rate','train_batch_size','eval_batch_size','eval_args','train_neg_sample_args','valid_metric','eval_step','dropout_prob','weight_decay']})
    assert all(torch.equal(state_before[k],v) for k,v in model.state_dict().items())
    assert all(sha(p)==h for p,h in original_hashes.items())
    result['frozen_state_exactly_unchanged']=True;result['original_file_hashes_unchanged']=True
    torch.save({'upstream_scores':output,'saved_scores':expected,'candidate_ids':split['candidate_ids'],'valid_mask':mask},OUT/'audit/a_equivalence'/f'{study}_scores.pt')
    (OUT/'audit/a_equivalence'/f'{study}.json').write_text(json.dumps(result,indent=2,default=str,allow_nan=False)+'\n')
    log({k:v for k,v in result.items() if k not in ['settings','original_file_hashes']})
    return result

