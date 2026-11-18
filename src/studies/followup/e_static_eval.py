"""Original E U/W/F inference and exact native score calculation."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np,torch
from .e_worker import ROOT,OUT,MATCHED,PRIOR,load,sha,write,save,write_csv,emit,identity
from graph_rerank.graph import NeighborTable,training_graph,graph_fingerprint
from diagnostics.graph_fusion.evaluate import ranking_metrics
from diagnostics.score_degree_essentials.full_catalog_eval import merge_top50
from studies.corrected.corrected_sampling import rank_record
from studies.s1.static import oracle_vectors,spearman_rows
def pack(records,degree):
    n=len(records);ids=torch.zeros((n,50),dtype=torch.int32);s=torch.zeros((n,50));mask=torch.zeros((n,50),dtype=torch.bool);labels=torch.zeros_like(mask)
    for j,r in enumerate(records):
        i=torch.as_tensor(r['items']).long();k=len(i);ids[j,:k]=i.int();s[j,:k]=torch.as_tensor(r['base_score_raw']);mask[j,:k]=True
        labels[j,:k]=torch.isin(i,torch.as_tensor(r['positive_items']).long())
    return {'user_ids':torch.tensor([r['user'] for r in records],dtype=torch.int32),'candidate_ids':ids,'s':s,'valid_mask':mask,'labels':labels,
            'positive_count':torch.tensor([r['positive_count'] for r in records]),'item_degree':torch.log1p(torch.as_tensor(degree).float())[ids.long()]}


def sampled(adapter,records,degree,deadline):
    ranked=[]
    for first in range(0,len(records),256):
        deadline();group=records[first:first+256];lengths=[len(r['items']) for r in group]
        users=np.concatenate([np.full(n,r['user'],np.int64) for r,n in zip(group,lengths)])
        items=torch.cat([torch.as_tensor(r['items']).long() for r in group]);values=adapter.score_candidates(users,items);cursor=0
        for r,n in zip(group,lengths):ranked.append(rank_record(r,values[cursor:cursor+n],top_m=50));cursor+=n
    return pack(ranked,degree),ranked


def full_catalog(adapter,records,degree,deadline):
    ranked=[];item_num=len(degree);start=time.monotonic()
    for first in range(0,len(records),64):
        deadline();group=records[first:first+64];users=torch.tensor([r['user'] for r in group],dtype=torch.long)
        best_s=torch.full((len(group),50),-torch.inf);best_i=torch.full((len(group),50),item_num,dtype=torch.long)
        for left in range(1,item_num,8192):
            deadline();ids=torch.arange(left,min(left+8192,item_num));score=adapter.score_block(users,ids).clone()
            for j,r in enumerate(group):
                # Exact existing S3 test exclusions: A+B+validation relevant,
                # minus current truth as already encoded by fixed W record.
                excluded=torch.as_tensor(r['excluded_items']).long()
                score[j,torch.isin(ids,excluded)]=-torch.inf
            best_s,best_i=merge_top50(best_s,best_i,score,ids.expand(len(group),-1))
        for j,r in enumerate(group):
            valid=torch.isfinite(best_s[j]);selected=best_i[j,valid]
            ranked.append({**r,'items':selected.int(),'base_score_raw':best_s[j,valid].float(),
                           'labels':torch.isin(selected,torch.as_tensor(r['positive_items']).long()).float(),
                           'sampling_meta':{'protocol':'F','all_eligible_scored':True,'forced_truth_insertion':False}})
        if first%640==0:emit({'stage':'E_stream_F','users_done':min(first+64,len(records)),'users_total':len(records),'seconds':time.monotonic()-start})
    return pack(ranked,degree),ranked


def evaluate(adapter,backbone_result,out,deadline):
    dataset=backbone_result['dataset'];phase=backbone_result['phase'];seed=backbone_result['seed'];deadline()
    bundle=load(MATCHED/'master'/dataset/'bundle.pt');degree=bundle['item_degree'];fingerprint=graph_fingerprint(training_graph(adapter.dataset))
    if bundle['signature']['sha256']!=backbone_result['identity']['master_digest']:raise RuntimeError('E checkpoint master split changed before evaluation')
    path=MATCHED/'master'/dataset/'neighbors/nonparamgc_k2.npz'
    with np.load(path,allow_pickle=False) as z:
        metadata=json.loads(str(z['metadata'].item()))
        if metadata['fingerprint']!=fingerprint or metadata['scorer_config']['family']!='nonparamgc' or metadata['scorer_config']['k']!=2:
            raise RuntimeError('Stored A native neighbors do not match E backbone training rows')
        tables=(NeighborTable(z['user_ids'].copy(),z['user_weights'].copy()),NeighborTable(z['item_ids'].copy(),z['item_weights'].copy()))
    from studies.corrected import native
    native.deadline=deadline
    poolpaths={'W':MATCHED/'master'/dataset/'fixed_W_pools.pt','U':PRIOR/'master'/dataset/'fixed_U_pools.pt'}
    inputs={str(path):sha(path),str(adapter.checkpoint_path):adapter.checkpoint_sha256}
    for protocol,p in poolpaths.items():
        if not p.is_file():raise FileNotFoundError('Frozen '+protocol+' pool missing; rebuilding forbidden')
        inputs[str(p)]=sha(p)
    if inputs[str(poolpaths['W'])]!=backbone_result['identity']['W_pool_sha256']:raise RuntimeError('Fixed W pool changed since E training')
    rows=[]
    for protocol in ('U','W','F'):
        deadline();dest=out/'evaluation'/protocol;marker=dest/'result.json'
        if marker.exists():
            r=json.loads(marker.read_text())
            if r['checkpoint_sha256']!=adapter.checkpoint_sha256 or r['input_hashes']!=inputs:raise RuntimeError('Existing E evaluation identity differs')
            rows.extend(r['rows']);continue
        started=time.monotonic();pools=load(poolpaths['W' if protocol=='F' else protocol]);records=pools['test']
        if pools['signature']['master_digest']!=bundle['signature']['sha256']:raise RuntimeError('Frozen pool master differs')
        if protocol=='F':split,ranked=full_catalog(adapter,records,degree,deadline)
        else:split,ranked=sampled(adapter,records,degree,deadline)
        native_score,error=native.native_scores(adapter,tables,ranked,alpha=1.)
        if error>1e-3:raise RuntimeError('Inherited native batching parity failed; no automatic repair')
        scores={'base':split['s'],'item_degree_ranking':split['item_degree'],'original_nonparamgc':native_score}
        oracle,recall=oracle_vectors(split);rho=spearman_rows(split['s'],split['item_degree'],split['valid_mask'])
        defined=np.isfinite(rho);current=[]
        for method,values in scores.items():
            metric=ranking_metrics(values,split)
            row={'analysis_family':'optional_backbone_loss_intervention','phase':phase,'dataset':dataset,
                 'backbone':'NFM_BPR' if phase=='E1' else 'MF_BCE','training_seed':seed,'protocol':protocol,'method':method,
                 **metric,'candidate_recall50':float(recall.mean()),'oracle_ndcg20':float(oracle.mean()),'n_users':len(recall),
                 'spearman_s_p_mean':float(rho[defined].mean()) if defined.any() else None,'spearman_defined_users':int(defined.sum()),
                 'selected_backbone_epoch':backbone_result['selected_epoch'],'checkpoint_sha256':adapter.checkpoint_sha256,
                 'graph_fingerprint':fingerprint,'neighbor_k':2,'alpha':1.,'full_truth_denominator':True,'regime':'E1_fixed_R0_batch' if phase=='E1' else 'E2_fixed_R0_batch'}
            current.append(row)
        candidate_digest=identity({
            'ids':sha_tensor(split['candidate_ids']),'users':sha_tensor(split['user_ids']),'mask':sha_tensor(split['valid_mask'])})
        save(dest/'top50_predictions.pt',{'split':split,'scores':scores,'oracle_ndcg20':oracle,'candidate_recall50':recall,'spearman_s_p':rho,
                                       'checkpoint_sha256':adapter.checkpoint_sha256,'candidate_digest':candidate_digest})
        write(marker,{'status':'COMPLETE','rows':current,'checkpoint_sha256':adapter.checkpoint_sha256,'candidate_digest':candidate_digest,
                      'input_hashes':inputs,'native_parity_error':error,'elapsed_seconds':time.monotonic()-started,'catalog_matrix_saved':False,
                      'reranker_training_performed':False,'test_tuning_performed':False})
        rows.extend(current);del pools,records,ranked,split,scores,native_score
    adapter.assert_frozen_and_unchanged()
    if any(sha(p)!=h for p,h in inputs.items()):raise RuntimeError('Frozen input changed during E evaluation')
    write_csv(out/'seed_results.csv',rows)
    write(out/'complete.json',{'status':'COMPLETE','phase':phase,'dataset':dataset,'seed':seed,'all_protocols':['U','W','F'],
                             'original_files_unchanged':True,'new_output_only':True,'full_catalog_matrix_saved':False})
    emit({'status':'COMPLETE','phase':phase,'dataset':dataset,'seed':seed,'output':str(out)})
    return rows


def sha_tensor(t):
    import hashlib
    t=t.detach().cpu().contiguous();h=hashlib.sha256();h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes());return h.hexdigest()
