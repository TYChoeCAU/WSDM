"""S1 execution: actual CTR training, historical benchmark, fresh controls and MostPop."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from common.runtime import REPO, bootstrap

DATASETS=('yelp2018','amazon-books','ml-1m')
BACKBONES=('NFM','xDeepFM','DCNV2','AutoInt','EulerNet','QNN_alpha')

def configurations(args):
    paths=sorted((REPO/'configs/s1').glob('*/*/*.json'))
    rows=[json.loads(p.read_text()) for p in paths]
    return [r for r in rows if (args.dataset is None or r['dataset']==args.dataset)
            and (args.backbone is None or r['backbone']==args.backbone)
            and (args.seed is None or r['seed']==args.seed)]

def prepare_inputs(spec, checkpoint, data_root, output, device):
    """Original independent validation/test loaders and candidate grouping."""
    import hashlib
    import numpy as np
    import torch
    from baselines.dnr.backbone_adapter import FrozenBackboneAdapter
    from baselines.dnr.data_adapter import build_eval_batches
    from diagnostics.revision_evidence.pipeline import positive_records
    from studies.corrected.native import phase_data, tensor_digest
    from min_path_mean.graph import training_graph
    target=output/spec['dataset']/spec['backbone']/f"seed_{spec['seed']}"
    target.mkdir(parents=True,exist_ok=True)
    adapter=FrozenBackboneAdapter(checkpoint,device=device,data_root=data_root,sample_num=100,predict_batch_size=262144)
    valid_batches,_=build_eval_batches(adapter,adapter.valid_data,'valid')
    test_adapter=FrozenBackboneAdapter(checkpoint,device=device,data_root=data_root,sample_num=100,predict_batch_size=262144)
    test_batches,_=build_eval_batches(test_adapter,test_adapter.test_data,'test')
    actual=(str(adapter.config['model']),int(adapter.config['seed']),str(adapter.config['dataset']))
    expected=(spec['backbone'],spec['seed'],spec['backbone_config']['dataset'])
    if actual!=expected:raise ValueError('Supplied checkpoint coordinate differs from selected configuration')
    graph=training_graph(adapter.dataset)
    history={int(u):set(map(int,graph.indices[graph.indptr[u]:graph.indptr[u+1]])) for u in range(graph.shape[0])}
    valid={}
    for u,_,positive in positive_records(valid_batches):valid.setdefault(u,set()).update(positive.tolist())
    records=positive_records(test_batches);degree=np.asarray(graph.sum(0)).ravel()
    bundle={p:phase_data(adapter,p) for p in ('train','valid','test')}
    bundle.update(user_num=adapter.dataset.user_num,item_num=adapter.dataset.item_num,
                  uid_field=adapter.uid_field,iid_field=adapter.iid_field,
                  split_digests={p:tensor_digest(bundle[p]) for p in ('train','valid','test')},
                  mapping_digests={name:hashlib.sha256(json.dumps(adapter.dataset.field2id_token[field].tolist(),sort_keys=True).encode()).hexdigest()
                                   for name,field in [('user',adapter.uid_field),('item',adapter.iid_field)]})
    torch.save(bundle,target/'split_bundle.pt')
    # This contains only candidate construction, without a DNR model or fit.
    torch.save({'checkpoint_sha256':adapter.checkpoint_sha256,'valid_batches':valid_batches,'test_batches':test_batches,
                'histories':history,'valid_positives':valid,'item_degree':degree},target/'benchmark_candidates.pt')
    test_adapter.assert_frozen_and_unchanged()
    return adapter,graph,records,history,valid,degree,target

def evaluate(spec,checkpoint,args):
    import numpy as np
    import torch
    from baselines.dnr.data_adapter import truncate_ranked_records
    from diagnostics.revision_evidence.pipeline import top50_from_records,protocol_split,sample_unique_negatives,group_rng
    from diagnostics.revision_evidence.common import atomic_json,write_csv,ranking_metrics
    from min_path_mean.graph import GraphScorerConfig,load_or_build_tables
    from studies.corrected.native import native_scores
    from .static import pack,_catalog_split,oracle_vectors,spearman_rows,summarize_spearman
    from .mostpop import Metrics,full_catalog,top_ids
    adapter,graph,records,history,valid,degree,target=prepare_inputs(spec,checkpoint,args.data_root,args.out,torch.device(args.device))
    identity={k:spec[k] for k in ('dataset','backbone','seed')}
    k=int(spec['npgc']['neighbor_k'])
    native=load_or_build_tables(graph,target/'native_neighbors.npz',scorer_config=GraphScorerConfig.preset('nonparamgc',k=k))
    minimal=load_or_build_tables(graph,target/'minimal_neighbors.npz',scorer_config=GraphScorerConfig.preset('minimal',k=k))
    ut,it=minimal
    factors={'user_sum':ut.weights.sum(1),'user_sq':np.square(ut.weights).sum(1),'item_sum':it.weights.sum(1),'item_sq':np.square(it.weights).sum(1)}
    rows=[];diagnostics=[]
    protocols=['benchmark_U'] if args.stage=='benchmark' else ([] if args.stage=='mostpop' else ['benchmark_U','U','W','F'])
    if args.stage=='fresh':protocols=['U','W','F']
    if spec['backbone'] not in ('NFM','DCNV2'):protocols=[p for p in protocols if p=='benchmark_U']
    for protocol in protocols:
        if protocol=='benchmark_U':
            enriched=[dict(g,positive_items=p,positive_count=len(p)) for _,g,p in records]
            ranked=truncate_ranked_records(enriched,50)
            split=pack(ranked,degree)
        elif protocol=='F':split=_catalog_split(adapter,records,history,valid,degree)
        else:
            generated,_=top50_from_records(adapter,records,spec['dataset'],spec['seed'],'U_control' if protocol=='U' else 'P_control',history,valid,degree,factors)
            split=protocol_split(generated,np.log1p(degree).astype(np.float32))
        ranked=[dict(user=int(u),items=split['candidate_ids'][i][split['valid_mask'][i]].long(),base_score_raw=split['s'][i][split['valid_mask'][i]]) for i,u in enumerate(split['user_ids'])]
        npgc,error=native_scores(adapter,native,ranked,float(spec['npgc']['alpha']))
        score_sets={'base':split['s'],'original_nonparamgc':npgc,'item_degree_ranking':split['item_degree']}
        oracle,recall=oracle_vectors(split)
        for method,scores in score_sets.items():
            rows.append(dict(**identity,protocol=protocol,method=method,**ranking_metrics(scores,split),candidate_recall50=float(recall.mean()),oracle_ndcg20=float(oracle.mean()),native_parity_max_abs_error=error))
        rho=spearman_rows(split['s'],split['item_degree'],split['valid_mask'])
        diagnostics.append(summarize_spearman(rho,dict(**identity,protocol=protocol)))
        torch.save({'split':split,'scores':score_sets,'checkpoint_sha256':adapter.checkpoint_sha256},target/f'{protocol}_predictions.pt')
    if args.stage in ('all','mostpop'):
        most=[]
        for protocol in ('benchmark_U','U','W','F'):
            if protocol=='F':
                result=full_catalog([(u,p.tolist(),len(p),history.get(u,set())|valid.get(u,set())) for u,_,p in records],degree)
            else:
                m=Metrics();weights=degree[1:].astype(np.float64)+1.;weights/=weights.sum()
                for u,g,positive in records:
                    p=set(map(int,positive.tolist()))
                    if protocol=='benchmark_U':pool=np.unique(g['items'].numpy())
                    else:
                        count=len(set(map(int,g['items'].tolist()))-p)
                        negatives,_=sample_unique_negatives(group_rng(spec['dataset'],spec['seed'],u,'U_control' if protocol=='U' else 'P_control'),len(degree),count,history.get(u,set())|valid.get(u,set())|p|{0},None if protocol=='U' else weights)
                        pool=np.asarray(sorted(p|set(negatives.tolist())),dtype=np.int64)
                    m.add(u,top_ids(pool,degree),positive.tolist(),len(positive))
                result=m.finish(degree)
            most.append(dict(**identity,protocol=protocol,**result))
        atomic_json(target/'mostpop.json',most)
    if rows:
        import csv
        previous=list(csv.DictReader((target/'seed_results.csv').open())) if (target/'seed_results.csv').is_file() else []
        current={(r['protocol'],r['method']) for r in rows}
        kept=[r for r in previous if (r['protocol'],r['method']) not in current]
        write_csv(target/'seed_results.csv',kept+rows)
    if diagnostics:atomic_json(target/'list_spearman.json',diagnostics)
    adapter.assert_frozen_and_unchanged()
    atomic_json(target/'completion.json',dict(**identity,stage=args.stage,checkpoint_path=str(adapter.checkpoint_path),checkpoint_sha256=adapter.checkpoint_sha256,status='COMPLETE',historical_candidate_identity='Fresh reconstruction; exact historical pre-pool IDs cannot be certified without archived private candidates.'))
    return rows

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['all','train','benchmark','fresh','mostpop','summarize'])
    p.add_argument('--dataset',choices=DATASETS);p.add_argument('--backbone',choices=BACKBONES);p.add_argument('--seed',type=int,choices=[2020,2021,2022])
    p.add_argument('--data-root',type=Path,default=Path(os.environ.get('DATA_ROOT','data')))
    p.add_argument('--out',type=Path,default=Path(os.environ.get('RESULTS_ROOT','runs'))/'s1')
    p.add_argument('--device',default='cuda:0');p.add_argument('--checkpoint',type=Path)
    p.add_argument('--dry-run',action='store_true');args=p.parse_args(argv)
    specs=configurations(args)
    if args.checkpoint and len(specs)!=1:p.error('--checkpoint requires one dataset/backbone/seed')
    plan={'study':'S1','stage':args.stage,'coordinates':len(specs),'selected':[{k:r[k] for k in ('dataset','backbone','seed')} for r in specs], 'data_root':str(args.data_root),'out':str(args.out),'fresh_backbones':['NFM','DCNV2'],'protocols':['benchmark_U','U','W','F']}
    if args.dry_run:print(json.dumps(plan,indent=2));return
    if args.stage=='summarize':
        from common.summaries import summarize
        print(json.dumps(summarize(args.out),indent=2));return
    bootstrap(True)
    from backbones.train import train_backbone
    from baselines.dnr.backbone_adapter import checkpoint_for
    from diagnostics.revision_evidence.common import write_csv
    all_rows=[]
    for spec in specs:
        checkpoint=args.checkpoint or checkpoint_for(args.out/'backbones',spec['dataset'],spec['backbone'],spec['seed'])
        if checkpoint is None and args.stage in ('all','train'):
            checkpoint=train_backbone(spec,args.data_root,args.out/'backbones',args.device)
        if checkpoint is None:raise FileNotFoundError('Checkpoint missing; run S1 train/all first or supply --checkpoint for one coordinate.')
        if args.stage!='train':all_rows.extend(evaluate(spec,checkpoint,args))
    if all_rows:
        from common.summaries import summarize
        print(json.dumps(summarize(args.out),indent=2))
if __name__=='__main__':main()
