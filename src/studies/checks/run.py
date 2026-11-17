"""Regenerate direct NPGC, frozen residual, MostPop, and rank diagnostics."""
from __future__ import annotations
import argparse,csv,json,os
from pathlib import Path
from common.runtime import bootstrap,REPO


def write_csv(path,rows):
    rows=list(rows);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def identity_selections(s3_run,out):
    from common.hashing import sha256
    masks=('S','P','R','PR','H','RH','PH','PRH');rows=[]
    registry=s3_run/'evidence_index.csv'
    if not registry.is_file():raise FileNotFoundError('Run S3 statistics/indexing before the identity diagnostic.')
    for entry in csv.DictReader(registry.open()):
        if entry.get('method') not in masks or entry.get('regime')!='R0' or entry.get('analysis_family')!='matched':continue
        path=Path(entry['result_path'])
        if sha256(path)!=entry['source_result_sha256']:raise ValueError('Registered residual result changed: '+str(path))
        r=json.loads(path.read_text())
        cp=Path(r['checkpoint_path'])
        if sha256(cp)!=r['checkpoint_sha256']:raise ValueError('Selected residual checkpoint changed: '+str(cp))
        rows.append({k:r.get(k) for k in ('dataset','backbone','training_seed','protocol','method','identity_selected','selected_epoch','checkpoint_sha256')})
    if len(rows)!=480:raise RuntimeError(f'Complete S3 identity audit needs 480 residual selections; found {len(rows)}')
    write_csv(out/'identity_all_residuals.csv',rows)
    summary=[]
    for backbone in ('NFM','MF_BPR'):
        selected=[r for r in rows if r['backbone']==backbone]
        summary.append(dict(backbone=backbone,identity_count=sum(bool(r['identity_selected']) for r in selected),total_modules=len(selected)))
    write_csv(out/'identity_summary.csv',summary)

def spearman(s1_run,s2_run,s3_run,out):
    import torch
    from studies.s1.static import spearman_rows,summarize_spearman
    rows=[]
    def emit(path,meta,split,popularity):
        rho=spearman_rows(split['s'],popularity,split['valid_mask'])
        rows.append(summarize_spearman(rho,dict(**meta,source_cache=str(path))))
    for path in sorted(s1_run.glob('*/*/seed_*/*_predictions.pt')):
        d,b,seed,filename=path.relative_to(s1_run).parts
        protocol=filename.removesuffix('_predictions.pt');saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True);split=saved['split']
        emit(path,dict(study='S1_historical' if protocol=='benchmark_U' else 'S1_fresh',dataset=d,backbone=b,training_seed=int(seed.split('_')[-1]),protocol=protocol),split,split['item_degree'])
    for path in sorted((s2_run/'cf').glob('*/seed_*/selected/*_predictions.pt')):
        d,seed,_,filename=path.relative_to(s2_run/'cf').parts
        saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True);split=saved['split']
        emit(path,dict(study='S2',dataset=d,backbone='MF_BPR',training_seed=int(seed.split('_')[-1]),protocol=filename.removesuffix('_predictions.pt')),split,split['item_degree'])
    for path in sorted((s3_run/'cache').glob('*/*/seed_*/*/feature_cache.pt')):
        d,b,seed,protocol,_=path.relative_to(s3_run/'cache').parts
        cache=torch.load(path,map_location='cpu',weights_only=False,mmap=True);split=cache['test']
        emit(path,dict(study='S3',dataset=d,backbone=b,training_seed=int(seed.split('_')[-1]),protocol=protocol,regime='R0'),split,split['features'][...,1])
    if not rows:raise FileNotFoundError('No generated prediction/feature caches found for list-Spearman diagnostics.')
    write_csv(out/'list_spearman_summary.csv',rows)

def mostpop(s1_run,s3_run,out):
    import numpy as np
    import torch
    from studies.s1.mostpop import Metrics,top_ids,full_catalog
    from diagnostics.revision_evidence.pipeline import positive_records,sample_unique_negatives,group_rng
    from common.statistics import repeat_interval
    rows=[]
    for path in sorted(s1_run.glob('*/*/seed_*/benchmark_candidates.pt')):
        d,b,seed,_=path.relative_to(s1_run).parts;seed=int(seed.split('_')[-1])
        saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        degree=saved['item_degree'];history=saved['histories'];valid=saved['valid_positives'];records=positive_records(saved['test_batches'])
        for protocol in ('benchmark_U','U','W','F'):
            if protocol=='F':result=full_catalog([(u,p.tolist(),len(p),history.get(u,set())|valid.get(u,set())) for u,_,p in records],degree)
            else:
                m=Metrics();weights=degree[1:].astype(np.float64)+1.;weights/=weights.sum()
                for u,g,positive in records:
                    pos=set(map(int,positive.tolist()))
                    if protocol=='benchmark_U':pool=np.unique(g['items'].numpy())
                    else:
                        n=len(set(map(int,g['items'].tolist()))-pos)
                        neg,_=sample_unique_negatives(group_rng(d,seed,u,'U_control' if protocol=='U' else 'P_control'),len(degree),n,history.get(u,set())|valid.get(u,set())|pos|{0},None if protocol=='U' else weights)
                        pool=np.asarray(sorted(pos|set(neg.tolist())),dtype=np.int64)
                    m.add(u,top_ids(pool,degree),positive.tolist(),len(positive))
                result=m.finish(degree)
            rows.append(dict(study='S1',dataset=d,backbone=b,seed=seed,protocol=protocol,status='PASS',candidate_identity='generated same-run candidate pool',**result))
    for d in ('yelp2018','ml-1m'):
        directory=s3_run/'master'/d
        if not (directory/'bundle.pt').exists():continue
        bundle=torch.load(directory/'bundle.pt',map_location='cpu',weights_only=False,mmap=True);degree=bundle['item_degree'].numpy().astype(np.int64);reference=None
        for protocol in ('U','W'):
            pools=torch.load(directory/f'fixed_{protocol}_pools.pt',map_location='cpu',weights_only=False,mmap=True)
            if pools['signature']['master_digest']!=bundle['signature']['sha256']:raise ValueError('S3 MostPop master digest differs')
            m=Metrics();identity=[]
            for r in pools['test']:
                p=r['positive_items'].numpy();ids=np.unique(r['items'].numpy());excluded=set(map(int,r['excluded_items'].tolist()))
                if set(ids)&excluded:raise ValueError('S3 excluded item in pool')
                m.add(r['user'],top_ids(ids,degree),p,r['positive_count']);identity.append((r['user'],tuple(p),r['positive_count'],tuple(sorted(excluded))))
            if reference is not None and identity!=reference:raise ValueError('S3 U/W full truth/exclusions differ')
            reference=identity;result=m.finish(degree)
            for b in ('NFM','MF_BPR'):rows.append(dict(study='S3',dataset=d,backbone=b,seed=2020,protocol=protocol,status='PASS',candidate_identity='fixed S3 pool',**result))
        result=full_catalog([(u,p,n,set(ex)) for u,p,n,ex in reference],degree)
        for b in ('NFM','MF_BPR'):rows.append(dict(study='S3',dataset=d,backbone=b,seed=2020,protocol='F',status='PASS',candidate_identity='fixed S3 eligible catalog',**result))
    if not rows:raise FileNotFoundError('MostPop requires generated S1 candidates or S3 fixed pools.')
    write_csv(out/'D_mostpop.csv',rows)
    summary=[]
    for study in ('S1','S3'):
      for dataset in sorted({r['dataset'] for r in rows if r['study']==study}):
       for protocol in ('benchmark_U','U','W','F'):
        group=[r for r in rows if (r['study'],r['dataset'],r['protocol'],r['backbone'])==(study,dataset,protocol,'NFM')]
        if not group:continue
        row={'Study':study,'Dataset':dataset,'Protocol':protocol}
        for label,key in [('NDCG@20','ndcg20'),('Recall@20','recall20'),('Cand R@50','candidate_recall50'),('Random F Cand R@50','random_top50_expected_candidate_recall50')]:
            if key not in group[0]:row[label]='';continue
            stat=repeat_interval([float(r[key]) for r in group])
            row[label]=f"{stat['mean_x100']:.4f}"+(f" ± {stat['sample_sd_x100']:.4f}" if len(group)>1 else '')
        summary.append(row)
    write_csv(out/'D_mostpop_summary.csv',summary)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['all','equivalence','tables','partial-dependence','identity','spearman','mostpop'])
    p.add_argument('--s1-run',type=Path,default=Path('runs/s1'));p.add_argument('--s2-run',type=Path,default=Path('runs/s2'))
    p.add_argument('--s3-run',type=Path,default=Path('artifacts/s3'));p.add_argument('--out',type=Path,default=Path('runs/checks'))
    p.add_argument('--data-root',type=Path,default=Path(os.environ.get('DATA_ROOT','data')))
    p.add_argument('--threads',type=int,default=4);p.add_argument('--batch',type=int,default=262144)
    p.add_argument('--studies',nargs='+',choices=['S1_fresh_U','S3_W'],default=['S1_fresh_U','S3_W'])
    p.add_argument('--dry-run',action='store_true');a=p.parse_args(argv)
    if a.dry_run:
        print(json.dumps({'stage':a.stage,'S1':str(a.s1_run),'S2':str(a.s2_run),'S3':str(a.s3_run),'out':str(a.out),'equivalence_studies':a.studies,'required':'generated candidate caches and S3 selected checkpoints; run S3 statistics before tables/partial-dependence','training':False,'inference_stages':['equivalence','partial-dependence'],'read_only_stages':['tables','identity','spearman','mostpop']},indent=2));return
    bootstrap(True);a.out.mkdir(parents=True,exist_ok=True);(a.out/'audit').mkdir(exist_ok=True)
    if a.stage in ('all','tables'):
        from . import c_tables
        c_tables.configure(a.s3_run,a.out);c_tables.main()
    if a.stage in ('all','identity'):identity_selections(a.s3_run,a.out)
    if a.stage in ('all','spearman'):spearman(a.s1_run,a.s2_run,a.s3_run,a.out)
    if a.stage in ('all','mostpop'):mostpop(a.s1_run,a.s3_run,a.out)
    if a.stage in ('all','partial-dependence'):
        from . import c_partial_dependence
        c_partial_dependence.configure(a.s3_run,a.out);c_partial_dependence.main(['--threads',str(a.threads),'--batch',str(a.batch)])
        status=json.loads((a.out/'audit/c_pd_status.json').read_text())
        if status['status']!='PASS':raise RuntimeError('Partial-dependence diagnostic failed; inspect audit/c_pd_status.json')
    if a.stage in ('all','equivalence'):
        from . import a_equivalence
        a_equivalence.configure(a.s1_run,a.s3_run,a.data_root,a.out)
        values=[a_equivalence.check(study) for study in a.studies]
        write_csv(a.out/'A_npgc_equivalence.csv',({k:v for k,v in r.items() if k not in ('table_report','original_file_hashes','settings')} for r in values))
if __name__=='__main__':main()
