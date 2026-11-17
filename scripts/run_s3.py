#!/usr/bin/env python3
"""Reproduce S3 from public data using the recorded fixed R0 configuration."""
from pathlib import Path
import argparse,itertools,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('dry-run','split','pools','backbones','candidates','matched','frozen','statistics','all'),default='dry-run')
    p.add_argument('--data-root',type=Path,default=Path(os.environ.get('DATA_ROOT',ROOT/'dataset')))
    p.add_argument('--run',type=Path,default=ROOT/'artifacts/s3')
    p.add_argument('--datasets',nargs='+',choices=('yelp2018','ml-1m'),default=['yelp2018','ml-1m'])
    p.add_argument('--backbones',nargs='+',choices=('NFM','MF_BPR'),default=['NFM','MF_BPR'])
    p.add_argument('--seeds',nargs='+',type=int,choices=range(2020,2025),default=list(range(2020,2025)))
    p.add_argument('--protocols',nargs='+',choices=('U','W','F'),default=['U','W','F'])
    p.add_argument('--methods',nargs='+',choices=('base','item_degree_ranking','original_nonparamgc','prm_principled','S','P','R','PR','H','RH','PH','PRH'))
    p.add_argument('--device',default='cuda');p.add_argument('--import-check',action='store_true')
    a=p.parse_args();os.environ['DATA_ROOT']=str(a.data_root.resolve());os.environ['S3_RUN']=str(a.run.resolve())
    from common.runtime import bootstrap
    bootstrap(require_upstream=a.stage not in ('dry-run','statistics'))
    from studies.s3.cross_common import verify_configuration,CONFIG,METHODS,MASKS
    manifest=verify_configuration()
    if a.stage=='dry-run':
        count=0
        for d,b,s,proto in itertools.product(a.datasets,a.backbones,a.seeds,a.protocols):
            applied=json.loads((CONFIG/'applied'/d/b/f'seed_{s}'/(proto+'.json')).read_text())
            assert set(applied)==set(METHODS)
            fixed=json.loads((CONFIG/'backbones'/d/b/f'seed_{s}.json').read_text())
            assert fixed['profile']['train_batch_size']==4096
            for m,v in applied.items():
                if m in MASKS or m=='prm_principled':assert v['batch_size']==(32 if m=='prm_principled'else 256)
            count+=len(applied)
        if a.import_check:
            bootstrap(True)
            from studies.s3.s3_common import configure_runtime
            configure_runtime()
            from studies.s3 import worker,w2w3_analysis,w2w3_extract
        print(json.dumps(dict(status='DRY_RUN_PASS',coordinates=count,configuration_files=len(manifest['files']),
            imports_checked=a.import_check,training_or_inference_launched=False,backbone_batch=4096,residual_batch=256,prm_batch=32)))
        return
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    from studies.s3 import worker
    if a.stage=='statistics':worker.statistics();return
    if a.stage in ('split','pools','all'):
        for d in a.datasets:
            if a.stage=='split':worker.split(d)
            else:worker.pools(d)
        if a.stage!='all':return
    for d,b,s in itertools.product(a.datasets,a.backbones,a.seeds):
        if a.stage in ('backbones','all'):worker.backbone(d,b,s,a.device)
        if a.stage in ('candidates','all'):worker.candidates(d,b,s,a.device)
        if a.stage in ('matched','all'):
            for proto in a.protocols:worker.matched(d,b,s,proto,a.device,a.methods)
        if a.stage in ('frozen','all'):
            methods=None if a.methods is None else[m for m in a.methods if m in MASKS or m=='prm_principled']
            for proto in a.protocols:
                if proto!='F':worker.frozen(d,b,s,proto,a.device,methods)
    if a.stage=='all':
        missing=worker.missing_statistics_inputs()
        if missing:
            report=dict(status='PARTIAL_COHORT',requested_stages_completed=True,statistics_run=False,
                missing_cohort_cells=len(missing),required_backbone_repeats=20,
                reason='Primary statistics require the complete five-seed R0 cohort')
            worker.atomic_json(worker.RUN/'partial_cohort.json',{**report,'missing_results':missing})
            print(json.dumps(report))
        else:worker.statistics()

if __name__=='__main__':main()
