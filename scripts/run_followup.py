#!/usr/bin/env python3
"""Completed follow-up implementations for the submitted manuscript.

Training/inference requires the prepared data, selected checkpoints and cached
candidate artifacts described by --manifest. These large inputs are not bundled.
Use --import-check to import the code without fitting, inference or data reads.
Existing S3 artifacts use --s3-run. S1/S2 inputs are explicit manifest entries.
E3 train/all needs prepared S3 data and pools; evaluate needs its saved checkpoint.
N1 needs saved S3 native predictions and feature caches, not a live model. Missing
S1 native scores remain UNAVAILABLE. No data, checkpoints or large caches ship here.
"""
from pathlib import Path
import argparse,importlib,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
TASKS=('e1','e2','e3','n1','df-s3','df-s1','fullpool','p0','p4')

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('task',choices=TASKS)
    p.add_argument('--stage',choices=('all','train','evaluate','aggregate'),default='all',help='E1/E2/E3 train/all fit new backbones when explicitly requested. N1 reads saved native scores only; aggregate uses completed outputs.')
    p.add_argument('--phase',choices=('fixed','frozen','all'),default='all',help='df-s3 fixed must precede frozen learned inference.')
    p.add_argument('--dataset',choices=('yelp2018','ml-1m'))
    p.add_argument('--backbone',choices=('NFM','MF_BPR','xDeepFM','DCNV2','AutoInt','EulerNet','QNN_alpha'))
    p.add_argument('--architecture',choices=('MF','NFM'),help='E3 architecture. Requires prepared S3 master data, fixed U/W pools, native neighbors; evaluate also requires an E3 checkpoint.')
    p.add_argument('--seed',type=int,choices=range(2020,2025),default=2020)
    p.add_argument('--draw',type=int,choices=range(8702031,8702036),help='p4 candidate draw only. Model training seed remains 2020.')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=32768,help='fullpool inference batch only, not a training setting.')
    p.add_argument('--data-root',type=Path,default=Path(os.environ.get('DATA_ROOT',ROOT/'dataset')))
    p.add_argument('--s3-run',type=Path,default=ROOT/'artifacts/s3',help='Outputs of scripts/run_s3.py, including master, checkpoints and caches.')
    p.add_argument('--s2-run',type=Path,default=ROOT/'artifacts/s2',help='Outputs of scripts/run_s2.py.')
    p.add_argument('--evidence-root',type=Path,default=ROOT/'results',help='Directory containing s3/ and checks/ numeric evidence CSVs.')
    p.add_argument('--manifest',type=Path,help='Real local artifact paths and optional SHA bindings. See configs/followup/manifest.example.json.')
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/followup')
    p.add_argument('--import-check',action='store_true',help='Import all follow-up core modules only. Never train, infer, regenerate caches or aggregate.')
    a=p.parse_args()
    for key,path in [('DATA_ROOT',a.data_root),('S3_RUN',a.s3_run),('S2_RUN',a.s2_run),('FOLLOWUP_EVIDENCE',a.evidence_root),('FOLLOWUP_OUT',a.out)]:os.environ[key]=str(path.expanduser().resolve())
    if a.manifest:os.environ['FOLLOWUP_MANIFEST']=str(a.manifest.expanduser().resolve())
    from common.runtime import bootstrap
    bootstrap(require_upstream=a.import_check or (a.task not in ('p0','n1') and a.stage!='aggregate'))
    if a.import_check:
        modules=('paths','runtime','e_worker','e_static_eval','e_summarize','e_analysis','e3_worker','e3_static_eval','e3_summarize','n1_popularity','candidate_helpers','metric_helpers','df_common','df_worker','df_summary','s1_inputs','s1_df','fullpool_native','fullpool','p0p1_analysis','p4_draws')
        for name in modules:importlib.import_module('studies.followup.'+name)
        print(json.dumps(dict(status='IMPORT_PASS',modules=len(modules),training_or_inference_launched=False)))
        return
    if a.stage=='train' and a.task not in ('e1','e2','e3'):p.error('Only e1/e2/e3 train models. Use all/evaluate for frozen inference.')
    if a.task=='p4' and a.seed!=2020:p.error('p4 fixes the model training seed at 2020. --draw changes only candidate RNG.')
    if a.task=='p0':
        from studies.followup.p0p1_analysis import run
        print(json.dumps(run()));return
    if a.task=='n1':
        if a.stage=='train':p.error('n1 never trains or runs inference; it reads saved candidate scores')
        from studies.followup.n1_popularity import main as run_n1
        run_n1();return
    if a.task=='e3' and a.seed>2022:p.error('E3 uses the completed training seeds 2020,2021,2022')
    if a.stage!='aggregate' and not a.dataset:p.error('--dataset is required for execution')
    if a.task in ('df-s1','fullpool') and a.seed>2022:p.error('S1 uses split seeds 2020,2021,2022')
    a.out.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.task in ('e1','e2'):
        from studies.followup import e_worker,e_static_eval,e_summarize
        phase=a.task.upper();declaration=ROOT/'configs/followup/declaration.json'
        if a.stage=='aggregate':
            for name in ('E1','E2'):e_summarize.summarize(declaration,name)
            from studies.followup.e_analysis import run_p1
            from studies.followup.p0p1_analysis import s3_index,read
            from studies.followup.paths import input_file,EVIDENCE
            print(json.dumps(run_p1(s3_index(read(input_file('s3_seed_results',EVIDENCE/'s3/extracted_seed_results.csv'))))))
            return
        out=a.out/'e'/phase/a.dataset/f'seed_{a.seed}'
        if a.stage in ('train','all'):out=e_worker.train(a.dataset,phase,a.seed,a.device,declaration)
        if a.stage in ('evaluate','all'):
            adapter,result=e_worker.load_adapter(out,a.device);e_static_eval.evaluate(adapter,result,out,e_worker.deadline)
    elif a.task=='e3':
        if a.stage=='aggregate':
            from studies.followup import e3_summarize
            e3_summarize.main();return
        from studies.followup import e3_worker,e3_static_eval
        if not a.architecture:p.error('e3 requires --architecture MF or NFM')
        declaration=ROOT/'configs/followup/e3_declaration.json'
        out=a.out/'e3'/a.architecture/a.dataset/f'seed_{a.seed}'
        if a.stage in ('train','all'):out=e3_worker.train(a.dataset,a.architecture,a.seed,a.device,declaration)
        if a.stage in ('evaluate','all'):
            adapter,result=e3_worker.load_adapter(out,a.device)
            e3_static_eval.evaluate(adapter,result,out,e3_worker.deadline)
    elif a.task=='df-s3':
        from studies.followup import df_worker,df_summary
        if a.stage=='aggregate':df_summary.main();return
        if a.phase in ('fixed','all'):df_worker.fixed(a.dataset,a.seed,a.device)
        if a.phase in ('frozen','all'):df_worker.frozen(a.dataset,a.seed,a.device)
    elif a.task=='df-s1':
        from studies.followup import s1_df
        if a.stage=='aggregate':print(json.dumps(s1_df._aggregate()));return
        if not a.backbone:p.error('--backbone is required')
        s1_df.run(a.dataset,a.backbone,a.seed,a.device)
    elif a.task=='fullpool':
        from studies.followup import fullpool
        if a.stage=='aggregate':fullpool.aggregate();return
        if not a.backbone:p.error('--backbone is required')
        fullpool.run(a.dataset,a.backbone,a.seed,a.device,a.batch_size)
    elif a.task=='p4':
        from studies.followup import p4_draws
        if a.stage=='aggregate':print(json.dumps(p4_draws._aggregate_unlocked()));return
        if a.backbone not in ('NFM','MF_BPR'):p.error('p4 requires --backbone NFM or MF_BPR')
        p4_draws.run_coordinate(a.dataset,a.backbone,a.device,[a.draw] if a.draw else None)

if __name__=='__main__':main()
