"""E3 observed-label BCE plus one uniform unobserved negative per A positive.

Only run-local data augmentation changes. Models, native Trainer, optimizer,
validation W, strict checkpoint selection, and early-stop budgets are reused.
Run through scripts/run_followup.py with prepared S3 master data and fixed pools.
"""
from __future__ import annotations
import copy,csv,hashlib,json,os,shutil,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
from .paths import ROOT,OUT,S3
MATCHED=PRIOR=S3
import numpy as np,torch
from studies.s3 import cross_backbone as bb
from . import e_worker as old
from recbole.data.interaction import Interaction
from recbole.data.dataloader import TrainDataLoader
from recbole.trainer import Trainer
from recbole.utils import init_seed,InputType
from recbole.model.context_aware_recommender.nfm import NFM
MFBCE=old.MFBCE
VERSION='E3-observed-plus-unobserved-BCE-v1'

load=old.load;sha=old.sha;identity=old.identity

def relative(p):
    p=Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
def report_value(x):
    if isinstance(x,dict):return {report_value(str(k)):report_value(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [report_value(v) for v in x]
    if isinstance(x,Path):return relative(x)
    if isinstance(x,str):return x.replace(str(ROOT)+'/', '')
    return x
def emit(x):print(json.dumps(report_value(x),default=str,allow_nan=False),flush=True)
def safe_path(p):
    p=Path(p).resolve()
    if not p.is_relative_to(OUT):raise PermissionError('E3 output outside new run')
    return p
def save(p,x):
    p=safe_path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');torch.save(x,q);q.replace(p)
def write(p,x):
    p=safe_path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(report_value(x),indent=2,default=str,allow_nan=False)+'\n');q.replace(p)
def write_csv(p,rows):
    p=safe_path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');rows=[report_value(r) for r in rows]
    with q.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    q.replace(p)
def deadline():
    # The submitted runs used a wall-clock submission deadline. A local rerun
    # keeps the same epoch/early-stop budgets without that obsolete calendar gate.
    return None

def disk_check(required=500*1024**2):
    if shutil.disk_usage(OUT).free<required:raise OSError('E3 space reserve reached; preserve completed outputs and stop')

def tensor_digest(t):
    t=t.detach().cpu().contiguous();h=hashlib.sha256();h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes());return h.hexdigest()

def profile(dataset,architecture):
    """Read the executed E3 baseline profile, with numerical values unchanged."""
    source=ROOT/'configs/followup/e3'/dataset/(architecture+'.json')
    saved=json.loads(source.read_text())
    if (saved['dataset'],saved['architecture'])!=(dataset,architecture):
        raise ValueError('Frozen E3 profile coordinate differs')
    return copy.deepcopy(saved['profile'])


def config_for(dataset,architecture,seed,device,out):
    c=bb.config_for(dataset,'MF_BPR' if architecture=='MF' else 'NFM',seed,device)
    c['checkpoint_dir']='.';c['save_dataset']=False;c['save_dataloaders']=False;c['log_wandb']=False
    c['MODEL_INPUT_TYPE']=InputType.POINTWISE
    c['train_neg_sample_args']={'distribution':'none','sample_num':'none','alpha':'none','dynamic':False,'candidate_num':0}
    if isinstance(c['data_path'],str):c['data_path']=report_value(c['data_path'])
    return c

def inputs(dataset):
    bundle,data,pools=old.inputs(dataset)
    labels=bundle['train']['labels']
    if not torch.equal(data.inter_feat[data.config['LABEL_FIELD']],labels):raise RuntimeError('A labels do not match frozen master')
    if not torch.all((labels==0)|(labels==1)):raise RuntimeError('A labels are not binary')
    if data.config['threshold']!={'rating':4}:raise RuntimeError('Frozen label threshold differs from rating>=4')
    return bundle,data,pools

def draw_epoch_negatives(bundle,seed,epoch,device='cpu'):
    """Same E1 uniform-rejection algorithm, with a dedicated sampler generator.

    Returns users in positive-row order, one negative item for each positive.
    No valid/test labels, positives, or candidate pools are read here.
    """
    a=bundle['train'];positive=a['labels']>0
    users=a['user_ids'][positive].to(device);ni=int(bundle['item_num']);observed=bundle['outer_train']
    known=(observed['user_ids']*ni+observed['item_ids']).unique(sorted=True).to(device)
    known=torch.cat((known,torch.tensor([int(bundle['user_num'])*ni+ni],device=device)))
    g=torch.Generator(device=device);g.manual_seed(int(seed)*10000+int(epoch))
    order=torch.randperm(len(users),generator=g,device=device);negatives=torch.empty_like(users)
    for start in range(0,len(order),4096):
        deadline();ix=order[start:start+4096];u=users[ix]
        negative=torch.randint(1,ni,(len(ix),),generator=g,device=device)
        for _ in range(10000):
            encoded=u*ni+negative;bad=known[torch.searchsorted(known,encoded)]==encoded
            if not bad.any():break
            negative[bad]=torch.randint(1,ni,(int(bad.sum()),),generator=g,device=device)
        else:raise RuntimeError('No eligible E3 negative after excluding A+B')
        negatives[ix]=negative
    enc=users*ni+negatives
    if torch.any(known[torch.searchsorted(known,enc)]==enc):raise AssertionError('E3 negative is observed in A+B')
    return users.cpu(),negatives.cpu()

def augmented_rows(data,bundle,seed,epoch,device='cpu'):
    users,items=draw_epoch_negatives(bundle,seed,epoch,device)
    fields=data.inter_feat.interaction
    if set(fields)!={data.uid_field,data.iid_field,data.config['LABEL_FIELD']}:raise RuntimeError('Unexpected interaction fields require explicit row adapter')
    extras={data.uid_field:users,data.iid_field:items,data.config['LABEL_FIELD']:torch.zeros(len(users),dtype=fields[data.config['LABEL_FIELD']].dtype)}
    rows=Interaction({k:torch.cat((v,extras[k])) for k,v in fields.items()})
    audit={'epoch':int(epoch),'observed_rows':len(data),'observed_positive_rows':int((bundle['train']['labels']>0).sum()),
        'observed_negative_rows':int((bundle['train']['labels']==0).sum()),'synthetic_negative_rows':len(users),'total_rows':len(rows),
        'synthetic_users_sha256':tensor_digest(users),'synthetic_items_sha256':tensor_digest(items),
        'observed_A_unchanged':all(torch.equal(rows[k][:len(data)],v) for k,v in fields.items()),
        'all_negatives_unobserved_in_A_plus_B':True,'heldout_labels_used':False}
    if not audit['observed_A_unchanged'] or len(users)!=audit['observed_positive_rows']:raise AssertionError('E3 row accounting differs')
    return rows,audit

def train(dataset,architecture,seed,device,gate_path):
    deadline();out=OUT/'e3'/architecture/dataset/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
    bundle,data,pools=inputs(dataset);p=profile(dataset,architecture);p['training_seed_budget']=list(json.loads(Path(gate_path).read_text())['seeds'])
    import inspect
    sources=[Path(__file__),Path(__file__).with_name('e3_static_eval.py'),Path(old.__file__),Path(__file__).with_name('e_static_eval.py'),Path(bb.__file__),Path(inspect.getsourcefile(NFM)),Path(inspect.getsourcefile(MFBCE)),Path(inspect.getsourcefile(Trainer)),Path(inspect.getsourcefile(TrainDataLoader))]
    signature={'version':VERSION,'phase':'E3','architecture':architecture,'dataset':dataset,'seed':seed,'profile':p,
        'master_digest':bundle['signature']['sha256'],'A_dataset_sha256':sha(MATCHED/'master'/dataset/'dataset.pt'),
        'validation_W_digest':bb.validation_digest(pools['valid']),'W_pool_sha256':sha(MATCHED/'master'/dataset/'fixed_W_pools.pt'),
        'U_pool_sha256':sha(PRIOR/'master'/dataset/'fixed_U_pools.pt'),'gate_sha256':sha(gate_path),
        'source_hashes':{relative(f):sha(f) for f in sources},'no_test_selection':True}
    signature['sha256']=identity(signature);result_path=out/'backbone_result.json'
    if result_path.exists():
        r=json.loads(result_path.read_text())
        if r['identity']!=signature or sha(out/'best_checkpoint.pt')!=r['checkpoint_sha256']:raise RuntimeError('Completed coordinate differs; no overwrite')
        return out
    fixed=out/'fixed_settings.json'
    if fixed.exists() and json.loads(fixed.read_text())!=signature:raise RuntimeError('E3 settings changed after start')
    write(fixed,signature);config=config_for(dataset,architecture,seed,device,out);data.config=config
    if any(config[k]!=p[v] for k,v in [('train_batch_size','train_batch_size'),('learning_rate','learning_rate'),('weight_decay','weight_decay'),('epochs','maximum_epochs'),('eval_step','validation_interval'),('stopping_step','patience_validation_checks')]):raise RuntimeError('Actual config differs from baseline profile')
    init_seed(seed,config['reproducibility']);model=(MFBCE if architecture=='MF' else NFM)(config,data).to(device)
    rows,row_audit=augmented_rows(data,bundle,seed,1,device);training_data=data.copy(rows)
    loader=TrainDataLoader(config,training_data,sampler=None,shuffle=config['shuffle'])
    previous=Path.cwd()
    try:os.chdir(out);trainer=Trainer(config,model)
    finally:os.chdir(previous)
    optimizer=trainer.optimizer
    adapter=bb.CrossAdapter(model,data,config,'MF_BPR' if architecture=='MF' else 'NFM',freeze=False,predict_batch_size=65536)
    best=-float('inf');best_epoch=-1;stale=0;curves=[];start_epoch=1;prior_seconds=0.;progress=out/'progress.pt'
    if progress.exists():
        cp=load(progress)
        if cp['identity']!=signature:raise RuntimeError('E3 resume signature mismatch')
        model.load_state_dict(cp['state_dict']);optimizer.load_state_dict(cp['optimizer'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.to(device)
        best=cp['best'];best_epoch=cp['best_epoch'];stale=cp['stale'];curves=cp['curves'];start_epoch=cp['epoch']+1;prior_seconds=cp['elapsed_seconds'];bb._restore_rng(cp['rng'],loader)
    started=time.monotonic();reason='MAXIMUM_EPOCH_BUDGET';epoch=start_epoch-1
    def stopped(e):return e>=p['minimum_epochs'] and (stale>p['patience_validation_checks'] if architecture=='NFM' else stale>=p['patience_validation_checks'])
    already=stopped(epoch)
    try:
        for epoch in ([] if already else range(start_epoch,p['maximum_epochs']+1)):
            deadline();disk_check();mark=time.monotonic()
            if epoch!=1:rows,row_audit=augmented_rows(data,bundle,seed,epoch,device)
            training_data.inter_feat=rows
            prepare_seconds=time.monotonic()-mark;mark=time.monotonic()
            loss=trainer._train_epoch(bb.DeadlineLoader(loader),epoch-1,show_progress=False)
            if isinstance(loss,tuple):loss=sum(loss)
            row={'dataset':dataset,'phase':'E3','architecture':architecture,'seed':seed,'epoch':epoch,'training_loss':float(loss),
                'training_seconds':time.monotonic()-mark,'augmentation_seconds':prepare_seconds,'validation_seconds':0.,'validation_ndcg@20':None,'selected':False,**row_audit}
            if epoch%p['validation_interval']==0:
                mark=time.monotonic();model.eval();metrics=bb.evaluate_fixed_W(adapter,pools['valid'],group_batch=256);score=metrics['ndcg@20'];row.update(validation_seconds=time.monotonic()-mark,**{'validation_ndcg@20':score})
                if score>best:
                    best=score;best_epoch=epoch;stale=0;row['selected']=True
                    save(out/'best_checkpoint.pt',{'identity':signature,'config':config,'state_dict':model.state_dict(),'other_parameter':model.other_parameter(),'selected_epoch':epoch,'validation_metrics':metrics,'training_parameters':sum(q.numel() for q in model.parameters())})
                else:stale+=1
            curves.append(row);write_csv(out/'training_curves.csv',curves)
            save(progress,{'identity':signature,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best,'best_epoch':best_epoch,'stale':stale,'curves':curves,'rng':bb._rng_state(loader),'elapsed_seconds':prior_seconds+time.monotonic()-started})
            emit(row)
            if stopped(epoch):reason='EARLY_STOPPED';break
        if already:reason='EARLY_STOPPED'
    except BaseException as e:
        write(out/'run_status.json',{'status':'INCOMPLETE','error':type(e).__name__+': '+str(e),'last_complete_epoch':curves[-1]['epoch'] if curves else 0,'partial_fit_not_published':True});raise
    finally:trainer.tensorboard.close()
    cp=load(out/'best_checkpoint.pt');result={'status':'COMPLETE','phase':'E3','architecture':architecture,'dataset':dataset,'seed':seed,'identity':signature,'selected_epoch':best_epoch,'epochs_ran':epoch,'completion_reason':reason,
        'checkpoint':relative(out/'best_checkpoint.pt'),'checkpoint_sha256':sha(out/'best_checkpoint.pt'),'validation_metrics':cp['validation_metrics'],
        'wall_seconds':prior_seconds+time.monotonic()-started,'test_evaluated_during_training':False,
        'training_seconds':sum(r['training_seconds'] for r in curves),'validation_seconds':sum(r['validation_seconds'] for r in curves),
        'augmentation_seconds':sum(r['augmentation_seconds'] for r in curves),'progress_removed_after_full_evaluation':True}
    write(result_path,result);write(out/'run_status.json',result)
    # Keep resumable state until all three protocol evaluations complete.
    del cp
    return out

def load_adapter(out,device):
    r=json.loads((out/'backbone_result.json').read_text());cp=load(out/'best_checkpoint.pt')
    if r['status']!='COMPLETE' or r['identity']!=cp['identity'] or sha(out/'best_checkpoint.pt')!=r['checkpoint_sha256']:raise RuntimeError('Unverified E3 checkpoint')
    path=MATCHED/'master'/r['dataset']/'dataset.pt'
    if sha(path)!=cp['identity']['A_dataset_sha256']:raise RuntimeError('A dataset changed')
    data=load(path);c=copy.deepcopy(cp['config']);c['device']=torch.device(device);c['use_gpu']=torch.device(device).type=='cuda';data.config=c
    model=(MFBCE if r['architecture']=='MF' else NFM)(c,data).to(device);model.load_state_dict(cp['state_dict']);model.load_other_parameter(cp.get('other_parameter'))
    return bb.CrossAdapter(model,data,c,'MF_BPR' if r['architecture']=='MF' else 'NFM',out/'best_checkpoint.pt',cp['identity'],predict_batch_size=65536),r

