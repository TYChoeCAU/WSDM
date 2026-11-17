"""Execute S2's published MF-BPR fit and frozen U/W/F evaluation."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from common.runtime import REPO,bootstrap,restore_config

def configs(args):
    return [r for p in sorted((REPO/'configs/s2').glob('*/*.json'))
            if (r:=json.loads(p.read_text())) and (args.dataset is None or r['dataset']==args.dataset)
            and (args.seed is None or r['seed']==args.seed)]

def prepare_split(spec,args):
    import hashlib
    import torch
    from recbole.data import create_dataset,data_preparation
    from recbole.utils import init_seed
    from studies.corrected.native import tensor_digest
    path=args.out/'private/splits'/spec['dataset']/f"seed_{spec['seed']}.pt"
    if path.is_file():return path
    s1=json.loads((REPO/spec['split_source']).read_text())
    config=restore_config(s1['backbone_config'],args.data_root,args.out/'unused_checkpoints',args.device)
    init_seed(config['seed'],config['reproducibility'])
    dataset=create_dataset(config);train,valid,test=data_preparation(config,dataset)
    bundle={}
    for phase,loader in [('train',train),('valid',valid),('test',test)]:
        data=loader._dataset;f=data.inter_feat
        bundle[phase]={'user_ids':f[data.uid_field].long().cpu(),'item_ids':f[data.iid_field].long().cpu(),'labels':f[data.label_field].float().cpu()}
    data=train._dataset
    bundle.update(user_num=data.user_num,item_num=data.item_num,uid_field=data.uid_field,iid_field=data.iid_field,
                  split_digests={p:tensor_digest(bundle[p]) for p in ('train','valid','test')},
                  mapping_digests={name:hashlib.sha256(json.dumps(data.field2id_token[field].tolist(),sort_keys=True).encode()).hexdigest()
                                   for name,field in [('user',data.uid_field),('item',data.iid_field)]})
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(bundle,path)
    return path

def evaluate(spec,args,profile):
    import torch
    from . import cf_experiment as cf
    from .sampled import _s2_uw
    from studies.s1.static import oracle_vectors
    from diagnostics.revision_evidence.common import ranking_metrics,write_csv,atomic_json
    dataset,seed=spec['dataset'],spec['seed']
    out=cf.coord_dir(dataset,seed)/'selected';out.mkdir(parents=True,exist_ok=True)
    checkpoint=cf.coord_dir(dataset,seed)/f"lr_{profile['learning_rate']:g}_wd_{profile['weight_decay']:g}"/'best_checkpoint.pt'
    if not checkpoint.is_file():raise FileNotFoundError('S2 checkpoint missing; run train/all first: '+str(checkpoint))
    bundle=cf.split_bundle(dataset,seed);_,_,_,degree=cf.sets_and_degree(bundle)
    cp=cf.load(checkpoint);model=cf.model_for(bundle,torch.device(args.device));model.load_state_dict(cp['state_dict']);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    rows=[]
    for protocol in ('U','W','F'):
        if protocol=='F':_,split,_=cf.eval_catalog(model,bundle,'test');split['item_degree']=torch.log1p(degree)[split['candidate_ids'].long()]
        else:split=_s2_uw(model,cf,bundle,degree,seed,protocol,dataset)
        scores={'base':split['s'],'item_degree_ranking':split['item_degree']}
        oracle,recall=oracle_vectors(split)
        for method,values in scores.items():rows.append(dict(dataset=dataset,backbone='MF_BPR',seed=seed,protocol=protocol,method=method,**ranking_metrics(values,split),candidate_recall50=float(recall.mean()),oracle_ndcg20=float(oracle.mean())))
        torch.save({'split':split,'scores':scores},out/f'{protocol}_predictions.pt')
    write_csv(out/'seed_results.csv',rows);atomic_json(out/'completion.json',{'status':'COMPLETE','dataset':dataset,'seed':seed,'selected_epoch':cp['selected_epoch'],'protocols':['U','W','F']})
    return rows

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['all','prepare','train','evaluate','summarize'])
    p.add_argument('--dataset',choices=['yelp2018','ml-1m']);p.add_argument('--seed',type=int,choices=[2020,2021,2022])
    p.add_argument('--data-root',type=Path,default=Path(os.environ.get('DATA_ROOT','data')))
    p.add_argument('--out',type=Path,default=Path(os.environ.get('RESULTS_ROOT','runs'))/'s2')
    p.add_argument('--device',default='cuda:0');p.add_argument('--search',action='store_true',help='Reproduce the original seed-2020 validation-only four-profile search before final repeats.')
    p.add_argument('--dry-run',action='store_true');args=p.parse_args(argv);specs=configs(args)
    if args.search and args.seed not in (None,2020):p.error('--search requires seed 2020 or all seeds')
    if args.dry_run:
        print(json.dumps({'study':'S2','stage':args.stage,'coordinates':len(specs),'selected':specs,'search':args.search,'out':str(args.out)},indent=2));return
    if args.stage=='summarize':
        from common.summaries import summarize
        print(json.dumps(summarize(args.out),indent=2));return
    bootstrap(True)
    import torch
    from . import cf_experiment as cf
    from diagnostics.revision_evidence.common import write_csv
    cf.RUN=args.out;cf.SCOPE={'cf_profile_search':specs[0]['profile_search']}
    for spec in specs:prepare_split(spec,args)
    if args.stage=='prepare':return
    if args.search and args.stage in ('all','train'):
        for dataset in sorted({s['dataset'] for s in specs}):
            for lr in (0.001,0.003):
                for wd in (0.,0.0001):cf.trial(dataset,2020,lr,wd,torch.device(args.device))
            cf.choose(dataset)
    rows=[]
    for spec in specs:
        profile=spec['selected_profile'];selected=args.out/'cf'/spec['dataset']/'selected_profile.json'
        if args.search and selected.is_file():profile=json.loads(selected.read_text())
        if args.stage in ('all','train'):cf.trial(spec['dataset'],spec['seed'],profile['learning_rate'],profile['weight_decay'],torch.device(args.device))
        if args.stage in ('all','evaluate'):rows.extend(evaluate(spec,args,profile))
    if rows:
        from common.summaries import summarize
        print(json.dumps(summarize(args.out),indent=2))
if __name__=='__main__':main()
