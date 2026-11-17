"""Local generated-artifact lineage and the corrected R0 runtime."""
import copy,functools,json,math
from pathlib import Path
from .cross_common import *
_INSTALLED=None

def read_master(dataset):
    p=RUN/'master'/dataset/'bundle.pt';manifest=json.loads(p.with_name('manifest.json').read_text())
    if sha(p)!=manifest['bundle_sha256']or sha(p.with_name('dataset.pt'))!=manifest['dataset_sha256']:
        raise ValueError('Frozen S3 master changed')
    return load(p)
def ensure_references(dataset,backbone=None,seed=None):
    # Every artifact is generated in this reproduction; no historic path fallback.
    read_master(dataset)

def configure_runtime():
    global _INSTALLED
    if _INSTALLED is not None:return _INSTALLED
    from . import cross_backbone as backbone,cross_candidates as candidates,cross_features as features
    from . import uniform_candidates as uniform,uniform_models as models,publisher
    from . import evaluation_batches
    modules=dict(backbone=backbone,candidates=candidates,features=features,uniform=uniform,models=models,publisher=publisher)
    if models.RESIDUAL_SETTINGS['batch_size']!=256:raise ValueError('Non-R0 residual module')
    changes=[];evaluation_batches._patch_candidates(modules,changes)
    original_init=backbone.CrossAdapter.__init__;original_eval=backbone.evaluate_fixed_W
    @functools.wraps(original_init)
    def init(self,*args,**kwargs):
        if len(args)>=8:args=list(args);args[7]=262144;args=tuple(args);kwargs.pop('predict_batch_size',None)
        else:kwargs['predict_batch_size']=262144
        return original_init(self,*args,**kwargs)
    backbone.CrossAdapter.__init__=init
    backbone.evaluate_fixed_W=lambda adapter,records,group_batch=128:original_eval(adapter,records,group_batch=2048)
    predict=models.predict_split
    @functools.wraps(predict)
    def inference(*args,**kwargs):
        if len(args)>5:args=list(args);args[5]=8192;args=tuple(args);kwargs.pop('batch_groups',None)
        else:kwargs['batch_groups']=8192
        return predict(*args,**kwargs)
    models.predict_split=inference
    uniform._pair_statistics=features._pair_statistics
    _INSTALLED=modules
    return modules

def cache_paths(dataset,backbone,seed,protocol):
    p=coord(dataset,backbone,seed,protocol,area='cache')
    return dict(directory=p,feature_cache_path=p/'feature_cache.pt',slate_cache_path=p/'slate_cache.pt',complete_path=p/'complete.json')

def applied_configuration(dataset,backbone,seed,protocol,method,master):
    path=CONFIG/'applied'/dataset/backbone/f'seed_{int(seed)}'/(protocol+'.json')
    app=copy.deepcopy(json.loads(path.read_text())[method])
    app.update(dataset=dataset,backbone=backbone,seed=int(seed),training_seed=int(seed),protocol=protocol,
        train_protocol=protocol,validation_protocol=protocol,test_protocol=protocol,deadline_kst=None,
        master_split_digest=master['signature']['sha256'],prm_lr_selection_scope=master['signature']['sha256'])
    if method in LEARNED and app['batch_size']!=(32 if method=='prm_principled'else 256):
        raise ValueError('Only recorded R0 training batches are permitted')
    if method=='prm_principled':
        choice_path=CONFIG/'prm_selection'/dataset/backbone/(protocol+'.json')
        choice=json.loads(choice_path.read_text())['recorded_selection']
        if choice['selection_seed']!=2020 or choice['test_consulted'] is not False:
            raise ValueError('PRM choice must have validation-only seed2020 lineage')
        if float(app['learning_rate'])!=float(choice['selected_learning_rate']):
            raise ValueError('PRM rate differs from the recorded validation selection')
        for key in ('embedding_dim','d_model','d_inner','n_head','dropout','weight_decay','maximum_epochs',
                    'minimum_epochs','patience','gradient_clip_norm','selection_min_delta','score_transform','source_commits'):
            if choice['scope']['settings'].get(key)!=app.get(key):
                raise ValueError('PRM non-batch configuration differs from its selection scope: '+key)
    app['release_configuration']=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),
        source_result_sha256=app.pop('recorded_result_sha256'),
        source_hashes={p.name:sha(p)for p in sorted(Path(__file__).parent.glob('*.py'))},
        learning_rate_policy='recorded seed2020 validation choice, no new tuning')
    return app

def valid_result(path,expected_cache=None):
    path=Path(path)
    if not path.is_file()or not path.resolve().is_relative_to(RUN):return False
    try:
        r=json.loads(path.read_text());sig=r['model_identity']['cache_signature']
        if r['status']!='COMPLETED'or r['regime']!='R0'or r['method']not in METHODS:return False
        if any(r[k]!=r['protocol']for k in ('train_protocol','validation_protocol','test_protocol')):return False
        if identity(r['model_identity'])!=r['model_identity_hash']or identity(sig)!=r['cache_identity']:return False
        if expected_cache is not None and r['cache_identity']!=expected_cache:return False
        bb=json.loads((coord(r['dataset'],r['backbone'],r['training_seed'],area='backbones')/'result.json').read_text())
        if bb['identity']['profile']['train_batch_size']!=4096:return False
        if bb['checkpoint_sha256']!=sig['checkpoint_sha256']or sha(bb['checkpoint'])!=bb['checkpoint_sha256']:return False
        proof=json.loads(cache_paths(r['dataset'],r['backbone'],r['training_seed'],r['protocol'])['complete_path'].read_text())
        if proof['identity_hash']!=r['cache_identity']or sig['phase_candidate_digests']['test']!=r['candidate_digest']:return False
        if sha(r['predictions_path'])!=r['predictions_sha256']or sha(r['resolved_config_path'])!=r['resolved_config_sha256']:return False
        for key in ('checkpoint','trained_only_checkpoint'):
            if r.get(key+'_path')and sha(r[key+'_path'])!=r[key+'_sha256']:return False
        app=r['model_identity']['applied'];proof=app['release_configuration']
        if sha(ROOT/proof['path'])!=proof['sha256']:return False
        if proof['source_hashes']!={p.name:sha(p)for p in sorted(Path(__file__).parent.glob('*.py'))}:return False
        if r['method']in LEARNED and app['batch_size']!=(32 if r['method']=='prm_principled'else 256):return False
        return all(math.isfinite(float(r['metrics'][m]))for m in ('ndcg@10','ndcg@20','recall@10','recall@20'))
    except (KeyError,ValueError,TypeError,OSError):return False

def result_path(dataset,backbone,seed,protocol,method,require_complete=False):
    p=coord(dataset,backbone,seed,protocol,method)/'result.json'
    if require_complete and not valid_result(p):raise ValueError('Missing or incompatible S3 result: '+str(p))
    return p
