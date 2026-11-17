"""Original E1 NFM-BPR and E2 MF-BCE loss/epoch/checkpoint implementation.
Requires prepared S3 master data and fixed U/W pools. Training is explicit.
Only paths, provenance labels, and the obsolete scheduling gate were adapted.
"""
from __future__ import annotations
import os,sys,json,csv,time,copy,hashlib,inspect
from pathlib import Path
import numpy as np,torch
from .paths import ROOT,OUT,S3
MATCHED=PRIOR=S3
from studies.s3 import cross_backbone as bb
from recbole.data.interaction import Interaction,cat_interactions
from recbole.model.context_aware_recommender.nfm import NFM
from recbole.model.general_recommender.bpr import BPR
from recbole.model.loss import BPRLoss
from recbole.utils import init_seed,InputType
from recbole.data.dataloader import TrainDataLoader
from recbole.trainer import Trainer
VERSION='fixed-loss-intervention-v1'
def deadline():return None
def load(p):return torch.load(p,map_location='cpu',weights_only=False,mmap=True)


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()


def emit(x):print(json.dumps(x,default=str,allow_nan=False),flush=True)


def safe_path(p):
    p=Path(p).resolve()
    if not p.is_relative_to(OUT):raise PermissionError('Stage E output must be inside new review run: '+str(p))
    return p


def save(p,x):
    p=safe_path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');torch.save(x,q);q.replace(p)


def write(p,x):
    p=safe_path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,default=str,allow_nan=False)+'\n');q.replace(p)


def write_csv(p,rows):
    p=safe_path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    with q.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    q.replace(p)


class NFMBPR(NFM):
    """Native NFM layers unchanged; pairwise loss sees correct side features.

    Positives and negatives are concatenated in one forward call, so native
    batch normalization receives the combined 2B rows. Existing dropout and
    normalization layers are retained. No additional learned layer is added.
    """
    def __init__(self,config,dataset):
        super().__init__(config,dataset)
        self.USER_ID=config['USER_ID_FIELD'];self.ITEM_ID=config['ITEM_ID_FIELD'];self.NEG_ITEM_ID=config['NEG_PREFIX']+self.ITEM_ID
        self._join_dataset=dataset;self.loss=BPRLoss()
    def calculate_loss(self,interaction):
        u=interaction[self.USER_ID].detach().cpu();p=interaction[self.ITEM_ID].detach().cpu();n=interaction[self.NEG_ITEM_ID].detach().cpu()
        pair=Interaction({self.USER_ID:torch.cat((u,u)),self.ITEM_ID:torch.cat((p,n))})
        joined=self._join_dataset.join(pair).to(self.device)
        logits=self.forward(joined);positive,negative=logits.split(len(u))
        return self.loss(positive,negative)


class MFBCE(BPR):
    """Native MF embedding/dot product unchanged; observed-label BCE only."""
    def __init__(self,config,dataset):
        super().__init__(config,dataset);self._label_field=config['LABEL_FIELD'];self.loss=torch.nn.BCEWithLogitsLoss()
    def calculate_loss(self,interaction):
        return self.loss(super().predict(interaction),interaction[self._label_field].float())


def profile(dataset,phase):
    p=copy.deepcopy(bb.frozen_profile(dataset,'NFM' if phase=='E1' else 'MF_BPR'))
    p.update(version=VERSION,phase=phase,embedding_size=64,train_batch_size=4096,selector='fixed validation-W NDCG@20',
             training_seed_budget=[2020,2021,2022,2023,2024],candidate_seed=2020,graph='frozen original partition A binary observed pairs',
             negative_exclusion='all observed A+B pairs, without held-out labels' if phase=='E1' else 'not applicable: observed A rows only',
             new_loss='original BPRLoss on NFM positive/negative logits' if phase=='E1' else 'BCEWithLogitsLoss on native MF dot product',
             batch_normalization='one combined positive+negative 2B forward; native NFM layers unchanged' if phase=='E1' else 'native MF has no BN',
             native_epoch='cross_backbone._bpr_epoch' if phase=='E1' else 'recbole.trainer.Trainer._train_epoch',
             architecture='NFM' if phase=='E1' else 'MF',score_domain='sigmoid(raw logit)',
             profile_source_for_this_intervention='artifact:configuration/s3/backbone-profile')
    p['loss']=p['new_loss'];p['model']='NFM' if phase=='E1' else 'BPR';p['reported_backbone']='NFM_BPR' if phase=='E1' else 'MF_BCE'
    return p


def config_for(dataset,phase,seed,device,out):
    c=bb.config_for(dataset,'NFM' if phase=='E1' else 'MF_BPR',seed,device)
    c['checkpoint_dir']=str(out);c['save_dataset']=False;c['save_dataloaders']=False;c['log_wandb']=False
    c['train_batch_size']=4096;c['embedding_size']=64;c['NEG_PREFIX']='neg_'
    if phase=='E2':
        c['MODEL_INPUT_TYPE']=InputType.POINTWISE
        c['train_neg_sample_args']={'distribution':'none','sample_num':'none','alpha':'none','dynamic':False,'candidate_num':0}
    return c


def inputs(dataset):
    directory=MATCHED/'master'/dataset
    manifest=json.loads((directory/'manifest.json').read_text())
    for filename,key in [('bundle.pt','bundle_sha256'),('dataset.pt','dataset_sha256')]:
        if sha(directory/filename)!=manifest[key]:raise RuntimeError('Frozen master lineage mismatch: '+filename)
    bundle=load(directory/'bundle.pt');data=load(directory/'dataset.pt');pools=load(directory/'fixed_W_pools.pt')
    if pools['signature']['master_digest']!=bundle['signature']['sha256']:raise RuntimeError('W pool/master mismatch')
    if not torch.equal(data.inter_feat[data.uid_field],bundle['train']['user_ids']) or not torch.equal(data.inter_feat[data.iid_field],bundle['train']['item_ids']):
        raise RuntimeError('A-only dataset row identity mismatch')
    return bundle,data,pools


def train(dataset,phase,seed,device,gate_path):
    deadline();out=OUT/'e'/phase/dataset/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
    bundle,data,pools=inputs(dataset);p=profile(dataset,phase)
    p['training_seed_budget']=list(range(2020,2025))
    source_files=[Path(__file__),Path(__file__).with_name('e_static_eval.py'),Path(bb.__file__),Path(inspect.getfile(NFM)),Path(inspect.getfile(BPR)),Path(inspect.getfile(Trainer))]
    signature={'version':VERSION,'dataset':dataset,'phase':phase,'seed':seed,'profile':p,'master_digest':bundle['signature']['sha256'],
        'A_dataset_sha256':sha(MATCHED/'master'/dataset/'dataset.pt'),'validation_W_digest':bb.validation_digest(pools['valid']),
        'W_pool_sha256':sha(MATCHED/'master'/dataset/'fixed_W_pools.pt'),
        'declaration_sha256':sha(gate_path),'source_hashes':{'artifact:source/'+f.name:sha(f) for f in source_files},'no_test_selection':True}
    signature['sha256']=identity(signature)
    result_path=out/'backbone_result.json'
    if result_path.exists():
        r=json.loads(result_path.read_text())
        if r['identity']!=signature or sha(out/'best_checkpoint.pt')!=r['checkpoint_sha256']:raise RuntimeError('Completed E coordinate differs; no overwrite')
        return out
    fixed=out/'fixed_settings.json'
    if fixed.exists() and json.loads(fixed.read_text())!=signature:raise RuntimeError('E settings changed after start')
    write(fixed,signature);config=config_for(dataset,phase,seed,device,out);data.config=config;init_seed(seed,True)
    if phase=='E1' and (str(p['optimizer']).lower()!='adam' or config['clip_grad_norm'] is not None):
        raise RuntimeError('Unchanged BPR epoch is authorized only for stored NFM Adam/no-clipping profile')
    model=(NFMBPR if phase=='E1' else MFBCE)(config,data).to(device)
    loader=trainer=None
    if phase=='E1':optimizer=torch.optim.Adam(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    else:
        loader=TrainDataLoader(config,data,sampler=None,shuffle=config['shuffle'])
        cwd=Path.cwd()
        try:os.chdir(out);trainer=Trainer(config,model)
        finally:os.chdir(cwd)
        optimizer=trainer.optimizer
    adapter=bb.CrossAdapter(model,data,config,'NFM' if phase=='E1' else 'MF_BPR',freeze=False,predict_batch_size=65536)
    best=-float('inf');best_epoch=-1;stale=0;curves=[];start_epoch=1;prior_seconds=0.;progress=out/'progress.pt'
    if progress.exists():
        cp=load(progress)
        if cp['identity']!=signature:raise RuntimeError('E resume signature mismatch')
        model.load_state_dict(cp['state_dict']);optimizer.load_state_dict(cp['optimizer'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.to(device)
        best=cp['best'];best_epoch=cp['best_epoch'];stale=cp['stale'];curves=cp['curves'];start_epoch=cp['epoch']+1;prior_seconds=cp['elapsed_seconds']
        bb._restore_rng(cp['rng'],loader)
    started=time.monotonic();reason='MAXIMUM_EPOCH_BUDGET';epoch=start_epoch-1
    # E1 inherits original NFM patience (>5 checks). E2 inherits native MF
    # profile's >=10 checks/min50/validation every10. No test feedback.
    def stopped(e):return e>=p['minimum_epochs'] and (stale>p['patience_validation_checks'] if phase=='E1' else stale>=p['patience_validation_checks'])
    already=stopped(epoch)
    try:
        for epoch in ([] if already else range(start_epoch,p['maximum_epochs']+1)):
            deadline();mark=time.monotonic()
            if phase=='E1':loss=bb._bpr_epoch(model,bundle,optimizer,device,seed,epoch)
            else:
                loss=trainer._train_epoch(bb.DeadlineLoader(loader),epoch-1,show_progress=False)
                if isinstance(loss,tuple):loss=sum(loss)
            row={'dataset':dataset,'phase':phase,'seed':seed,'epoch':epoch,'training_loss':float(loss),'training_seconds':time.monotonic()-mark,
                 'validation_seconds':0.,'validation_ndcg@20':None,'selected':False}
            if epoch%p['validation_interval']==0:
                mark=time.monotonic();model.eval();metrics=bb.evaluate_fixed_W(adapter,pools['valid'],group_batch=256);score=metrics['ndcg@20']
                row.update(validation_seconds=time.monotonic()-mark,**{'validation_ndcg@20':score})
                if score>best:
                    best=score;best_epoch=epoch;stale=0;row['selected']=True
                    save(out/'best_checkpoint.pt',{'identity':signature,'config':config,'state_dict':model.state_dict(),'other_parameter':model.other_parameter(),
                         'selected_epoch':epoch,'validation_metrics':metrics,'training_parameters':sum(q.numel() for q in model.parameters())})
                else:stale+=1
            curves.append(row);write_csv(out/'training_curves.csv',curves)
            save(progress,{'identity':signature,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best,'best_epoch':best_epoch,
                'stale':stale,'curves':curves,'rng':bb._rng_state(loader),'elapsed_seconds':prior_seconds+time.monotonic()-started})
            emit(row)
            if stopped(epoch):reason='EARLY_STOPPED';break
        if already:reason='EARLY_STOPPED'
    except BaseException as e:
        write(out/'run_status.json',{'status':'INCOMPLETE','error':type(e).__name__+': '+str(e),'last_complete_epoch':curves[-1]['epoch'] if curves else 0,
                                   'partial_fit_not_published':True});raise
    finally:
        if trainer is not None:trainer.tensorboard.close()
    cp=load(out/'best_checkpoint.pt')
    result={'status':'COMPLETE','phase':phase,'dataset':dataset,'seed':seed,'identity':signature,'selected_epoch':best_epoch,'epochs_ran':epoch,
        'completion_reason':reason,'checkpoint':str(out/'best_checkpoint.pt'),'checkpoint_sha256':sha(out/'best_checkpoint.pt'),
        'validation_metrics':cp['validation_metrics'],'wall_seconds':prior_seconds+time.monotonic()-started,'test_evaluated_during_training':False,
        'training_seconds':sum(r['training_seconds'] for r in curves),'validation_seconds':sum(r['validation_seconds'] for r in curves)}
    write(result_path,result);write(out/'run_status.json',result)
    return out


def load_adapter(out,device):
    r=json.loads((out/'backbone_result.json').read_text());cp=load(out/'best_checkpoint.pt')
    if r['status']!='COMPLETE' or r['identity']!=cp['identity'] or sha(out/'best_checkpoint.pt')!=r['checkpoint_sha256']:raise RuntimeError('Unverified E checkpoint')
    data_path=MATCHED/'master'/r['dataset']/'dataset.pt'
    if sha(data_path)!=cp['identity']['A_dataset_sha256']:raise RuntimeError('E checkpoint A dataset changed before evaluation')
    data=load(data_path);c=copy.deepcopy(cp['config']);c['device']=torch.device(device);c['use_gpu']=True
    model=(NFMBPR if r['phase']=='E1' else MFBCE)(c,data).to(device);model.load_state_dict(cp['state_dict']);model.load_other_parameter(cp.get('other_parameter'))
    return bb.CrossAdapter(model,data,c,'NFM' if r['phase']=='E1' else 'MF_BPR',out/'best_checkpoint.pt',cp['identity'],predict_batch_size=65536),r
