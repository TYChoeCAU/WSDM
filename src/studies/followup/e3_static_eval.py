"""E3 static U/W/F evaluation with exact existing scorer/evaluator semantics.

The original E1/E2 streamed catalog routine retains only Top-50 in RAM.
Lossless compressed compact predictions replace redundant tensor copies on disk.
"""
from __future__ import annotations
import json,time
import numpy as np,torch
from .e3_worker import ROOT,OUT,MATCHED,PRIOR,load,sha,write,write_csv,emit,identity,relative,safe_path,disk_check
from . import e_static_eval as original
from graph_rerank.graph import NeighborTable,training_graph,graph_fingerprint
from diagnostics.graph_fusion.evaluate import ranking_metrics
pack=original.pack
sampled=original.sampled
full_catalog=original.full_catalog
sha_tensor=original.sha_tensor

def save_npz(path,arrays):
    path=safe_path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    # np.savez receives a handle so it cannot append an untracked extension.
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)
def array(value):return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)

def evaluate(adapter,backbone_result,out,deadline):
    dataset=backbone_result['dataset'];architecture=backbone_result['architecture'];seed=backbone_result['seed'];deadline()
    bundle=load(MATCHED/'master'/dataset/'bundle.pt');degree=bundle['item_degree'];fingerprint=graph_fingerprint(training_graph(adapter.dataset))
    if bundle['signature']['sha256']!=backbone_result['identity']['master_digest']:raise RuntimeError('E3 master changed before evaluation')
    path=MATCHED/'master'/dataset/'neighbors/nonparamgc_k2.npz'
    with np.load(path,allow_pickle=False) as z:
        metadata=json.loads(str(z['metadata'].item()))
        if metadata['fingerprint']!=fingerprint or metadata['scorer_config']['family']!='nonparamgc' or metadata['scorer_config']['k']!=2:raise RuntimeError('A native neighbors differ')
        tables=(NeighborTable(z['user_ids'].copy(),z['user_weights'].copy()),NeighborTable(z['item_ids'].copy(),z['item_weights'].copy()))
    from studies.corrected import native
    from studies.s1 import static as aux
    native.deadline=deadline
    poolpaths={'W':MATCHED/'master'/dataset/'fixed_W_pools.pt','U':PRIOR/'master'/dataset/'fixed_U_pools.pt'}
    inputs={relative(path):sha(path),relative(adapter.checkpoint_path):adapter.checkpoint_sha256}
    for protocol,p in poolpaths.items():
        if not p.is_file():raise FileNotFoundError('Frozen '+protocol+' pool missing; rebuilding forbidden')
        inputs[relative(p)]=sha(p)
    for protocol in ('U','W'):
        if inputs[relative(poolpaths[protocol])]!=backbone_result['identity'][protocol+'_pool_sha256']:raise RuntimeError('Frozen '+protocol+' pool changed')
    rows=[]
    for protocol in ('U','W','F'):
        deadline();disk_check();dest=out/'evaluation'/protocol;marker=dest/'result.json'
        if marker.exists():
            r=json.loads(marker.read_text())
            if r['checkpoint_sha256']!=adapter.checkpoint_sha256 or r['input_hashes']!=inputs:raise RuntimeError('Existing E3 evaluation identity differs')
            for name,digest in r['output_hashes'].items():
                if sha(ROOT/name)!=digest:raise RuntimeError('Completed compact prediction changed')
            rows.extend(r['rows']);continue
        started=time.monotonic();pools=load(poolpaths['W' if protocol=='F' else protocol]);records=pools['test']
        if pools['signature']['master_digest']!=bundle['signature']['sha256']:raise RuntimeError('Frozen pool master differs')
        if protocol=='F':split,ranked=full_catalog(adapter,records,degree,deadline)
        else:split,ranked=sampled(adapter,records,degree,deadline)
        native_score,error=native.native_scores(adapter,tables,ranked,alpha=1.)
        if error>1e-3:raise RuntimeError('Native batching parity failed')
        scores={'base':split['s'],'item_degree_ranking':split['item_degree'],'original_nonparamgc':native_score}
        oracle,recall=aux.oracle_vectors(split);rho=aux.spearman_rows(split['s'],split['item_degree'],split['valid_mask']);defined=np.isfinite(rho)
        vectors={};current=[]
        for method,values in scores.items():
            deadline();metric=ranking_metrics(values,split);v=aux.metric_vectors(values,split)
            if max(abs(float(v[k].mean())-metric[k]) for k in metric)>1e-3:raise RuntimeError('Per-user/common evaluator parity differs')
            vectors[method]=v
            row={'analysis_family':'negative_exposure_BCE_intervention','phase':'E3','architecture':architecture,'dataset':dataset,
                'backbone':architecture+'_BCE_augmented','training_seed':seed,'protocol':protocol,'method':method,**metric,
                'candidate_recall50':float(recall.mean()),'oracle_ndcg20':float(oracle.mean()),'n_users':len(recall),
                'spearman_s_p_mean':float(rho[defined].mean()) if defined.any() else None,'spearman_defined_users':int(defined.sum()),
                'selected_backbone_epoch':backbone_result['selected_epoch'],'checkpoint_sha256':adapter.checkpoint_sha256,
                'graph_fingerprint':fingerprint,'neighbor_k':2,'alpha':1.,'full_truth_denominator':True,'regime':'E3_frozen_baseline_settings',
                'raw_metric_unit':'proportion','reported_metric_unit':'percent','master_digest':bundle['signature']['sha256']}
            for key in (*metric.keys(),'candidate_recall50','oracle_ndcg20'):row[key+'_pct']=100*row[key]
            current.append(row)
        candidate_digest=identity({'ids':sha_tensor(split['candidate_ids']),'users':sha_tensor(split['user_ids']),'mask':sha_tensor(split['valid_mask'])})
        compact={k:array(split[k]) for k in ('user_ids','candidate_ids','s','valid_mask','labels','positive_count','item_degree')}
        compact['original_nonparamgc']=array(native_score)
        prediction_path=dest/'top50_predictions.npz';save_npz(prediction_path,compact)
        per_user={'oracle_ndcg20':oracle,'candidate_recall50':recall,'spearman_s_p':rho,'user_ids':array(split['user_ids'])}
        for method,mv in vectors.items():
            for metric,value in mv.items():per_user[method+'__'+metric]=value
        vector_path=dest/'per_user_metrics.npz';save_npz(vector_path,per_user)
        write(marker,{'status':'COMPLETE','phase':'E3','architecture':architecture,'dataset':dataset,'seed':seed,'protocol':protocol,'rows':current,
            'checkpoint_sha256':adapter.checkpoint_sha256,'candidate_digest':candidate_digest,'input_hashes':inputs,'master_digest':bundle['signature']['sha256'],
            'output_hashes':{relative(prediction_path):sha(prediction_path),relative(vector_path):sha(vector_path)},
            'native_parity_error':error,'elapsed_seconds':time.monotonic()-started,'catalog_matrix_saved':False,
            'compression':'lossless np.savez_compressed; float scores retain original float32 dtype',
            'reranker_training_performed':False,'test_tuning_performed':False,'full_truth_denominator':True,
            'pool_rule':'existing U/W fixed pools; F all eligible catalog excluding record.excluded_items; item ID ascending ties'})
        rows.extend(current);del pools,records,ranked,split,scores,native_score,compact,per_user,vectors
        emit({'stage':'E3_evaluation','architecture':architecture,'dataset':dataset,'seed':seed,'protocol':protocol,'status':'COMPLETE','elapsed_seconds':time.monotonic()-started})
    adapter.assert_frozen_and_unchanged()
    if any(sha(ROOT/p)!=h for p,h in inputs.items()):raise RuntimeError('Frozen input changed during evaluation')
    write_csv(out/'seed_results.csv',rows)
    write(out/'complete.json',{'status':'COMPLETE','phase':'E3','architecture':architecture,'dataset':dataset,'seed':seed,'all_protocols':['U','W','F'],
        'original_files_unchanged':True,'new_output_only':True,'full_catalog_matrix_saved':False,'checkpoint':relative(out/'best_checkpoint.pt'),
        'checkpoint_sha256':adapter.checkpoint_sha256,'progress_cleanup':'remove run-local redundant optimizer progress only after this completion marker'})
    progress=out/'progress.pt'
    if progress.exists():safe_path(progress).unlink()
    emit({'status':'COMPLETE','phase':'E3','architecture':architecture,'dataset':dataset,'seed':seed,'output':relative(out)})
    return rows
