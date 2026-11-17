"""Frozen U/W-selected models on actual F features. No fitting or selection."""
from __future__ import annotations
import argparse
import copy
import gc
import json
from pathlib import Path
import time
import torch
from .s3_common import *
from diagnostics.graph_fusion.evaluate import ranking_metrics
from diagnostics.revision_evidence.common import per_group_metrics

LEARNED=('prm_principled',*MASKS)


def transfer_path(dataset,backbone,seed,source_protocol,method):
    return RUN/'transfer'/dataset/backbone/f'seed_{int(seed)}'/(source_protocol+'_to_F')/method


def completed_transfer(dataset,backbone,seed,source_protocol,method):
    path=transfer_path(dataset,backbone,seed,source_protocol,method)/'result.json'
    if not path.exists():return False
    try:
        row=json.loads(path.read_text());source=result_path(dataset,backbone,seed,source_protocol,method,require_complete=True)
        target=cache_paths(dataset,backbone,seed,'F')['complete_path']
        return (row['status']=='COMPLETED'and row['source_result_sha256']==sha(source)and
                row['target_cache_identity']==json.loads(target.read_text())['identity_hash']and
                Path(row['predictions_path']).is_file()and sha(row['checkpoint_path'])==row['checkpoint_sha256']and
                (method not in MASKS or Path(row['trained_only_checkpoint_path']).exists()))
    except (OSError,KeyError,ValueError):return False


def predict_frozen(source_path,target_features,target_slates,device='cuda',trained_only=False):
    """Source model/scaler/t stay in its checkpoint, target supplies raw rows."""
    modules=configure_runtime();source=json.loads(Path(source_path).read_text());method=source['method']
    signature=target_features['signature']
    if source['protocol']not in ('U','W')or signature['protocol']!='F':raise ValueError('Expected U/W -> F')
    for expected,actual in ((source['backbone_checkpoint_sha256'],signature['checkpoint_sha256']),
                            (source['master_split_digest'],signature['master_digest']),
                            (source['graph_digest'],signature['graph_fingerprint'])):
        if expected!=actual:raise ValueError('Frozen transfer backbone/split/graph differs')
    path=source['trained_only_checkpoint_path']if trained_only else source['checkpoint_path']
    checksum=source['trained_only_checkpoint_sha256']if trained_only else source['checkpoint_sha256']
    cp=load(path)
    if sha(path)!=checksum or cp['cross_model_identity_hash']!=source['model_identity_hash']:
        raise ValueError('Source checkpoint identity differs')
    applied=copy.deepcopy(cp['applied'])
    applied.update(deadline_kst=None,expected_model_identity_hash=source['model_identity_hash'],expected_checkpoint_sha256=checksum)
    applied.pop('deadline_utc',None)
    scores=modules['models'].score_selected(method,target_features,'test',path,applied,
                                          torch.device(device),slate_cache=target_slates)
    if method in MASKS:
        if cp['feature_mask']!=list(MASKS[method])or cp['feature_names']!=list(FEATURES):raise ValueError('Source feature mask changed')
        if cp['residual_scale_t']!=source['residual_scale_t']:raise ValueError('Source residual scale differs')
    return scores


def execute(dataset,backbone,seed,source_protocol,device='cuda',methods=None):
    if source_protocol not in ('U','W'):raise ValueError('Source protocol must be U or W')
    methods=LEARNED if methods is None else tuple(methods)
    if any(m not in LEARNED for m in methods):raise ValueError('Transfer covers the nine learned methods')
    ensure_references(dataset,backbone,seed);configure_runtime()
    paths=cache_paths(dataset,backbone,seed,'F')
    if not paths['complete_path'].exists():raise FileNotFoundError('F cache is not yet prepared, transfer cannot construct or refit it')
    proof=json.loads(paths['complete_path'].read_text())
    for name in ('feature_cache','slate_cache'):
        if sha(paths[name+'_path'])!=proof[name+'_sha256']:raise ValueError('F cache bytes differ')
    target=load(paths['feature_cache_path']);slates=load(paths['slate_cache_path']);test=target['test'];output=[]
    for method in methods:
        if completed_transfer(dataset,backbone,seed,source_protocol,method):
            output.append(dict(method=method,status='REUSED'));continue
        source_path=result_path(dataset,backbone,seed,source_protocol,method,require_complete=True)
        source=json.loads(source_path.read_text());source_predictions=load(source['predictions_path'])
        if not torch.equal(source_predictions['user_ids'],test['user_ids']):
            raise ValueError('Source/F user population or order differs, refusing an inner join')
        source_cache=load(source['cache_path'])
        if not torch.equal(source_cache['test']['positive_count'],test['positive_count']):
            raise ValueError('Source/F full relevance denominator differs')
        if not all(torch.equal(torch.as_tensor(a),torch.as_tensor(b))for a,b in zip(source_cache['test']['positive_items'],test['positive_items'])):
            raise ValueError('Source/F relevant item identity differs')
        before=sha(source['checkpoint_path']);started=time.monotonic()
        scores=predict_frozen(source_path,target,slates,device=device)
        per=per_group_metrics(scores,test);metrics=ranking_metrics(scores,test)
        if any(abs(float(per[k].mean())-metrics[k])>1e-3 for k in metrics):raise ValueError('Per-user/common metric differs')
        prediction=dict(scores=scores,per_user_metrics=per,user_ids=test['user_ids'],candidate_digest=test['candidate_digest'],
                        source_model_identity_hash=source['model_identity_hash'],source_protocol=source_protocol,test_protocol='F')
        extra={}
        if method in MASKS:
            same_checkpoint=source['trained_only_checkpoint_sha256']==source['checkpoint_sha256']
            trained=scores.clone()if same_checkpoint else predict_frozen(source_path,target,slates,device=device,trained_only=True)
            trained_per=per_group_metrics(trained,test)
            prediction.update(trained_only_scores=trained,trained_only_per_user_metrics=trained_per)
            extra=dict(trained_only_metrics=ranking_metrics(trained,test),trained_only_checkpoint_path=source['trained_only_checkpoint_path'],
                       trained_only_checkpoint_sha256=source['trained_only_checkpoint_sha256'],trained_selected_epoch=source['trained_selected_epoch'],
                       source_trained_only_metrics=source['trained_only_metrics'])
            extra['selected_trained_only_inference_shared']=same_checkpoint
        if sha(source['checkpoint_path'])!=before:raise ValueError('Source checkpoint changed during inference')
        directory=transfer_path(dataset,backbone,seed,source_protocol,method);directory.mkdir(parents=True,exist_ok=True)
        predictions_path=directory/'test_predictions.pt';save(predictions_path,prediction)
        hits=test['labels'].sum(1).double();full=test['positive_count'].double()
        result=dict(regime='R0',predictions_sha256=sha(predictions_path),status='COMPLETED',study_id='S3',cohort='matched_AB_fixed2020_frozen_transfer',dataset=dataset,backbone=backbone,
                    training_seed=int(seed),seed=int(seed),backbone_training_seed=int(seed),reranker_training_seed=int(seed),
                    outer_split_seed=2020,ab_split_seed=2020,candidate_seed=2020,method=method,method_id=method,
                    mask_id=method if method in MASKS else None,source_protocol=source_protocol,protocol='F',
                    train_protocol=source_protocol,validation_protocol=source_protocol,test_protocol='F',eval_protocol='F',
                    metrics=metrics,source_metrics=source['metrics'],predictions_path=str(predictions_path),
                    source_result_path=str(source_path),source_result_sha256=sha(source_path),
                    source_cache_identity=source['cache_identity'],target_cache_identity=identity(target['signature']),
                    cache_path=str(paths['feature_cache_path']),candidate_digest=test['candidate_digest'],
                    checkpoint_path=source['checkpoint_path'],checkpoint_sha256=source['checkpoint_sha256'],
                    source_model_identity_hash=source['model_identity_hash'],selected_epoch=source['selected_epoch'],
                    selection_policy=source['training_summary']['selection_policy'],identity_selected=source.get('identity_selected',False),
                    source_scaler_digest=source.get('scaler_digest'),source_residual_scale_t=source.get('residual_scale_t'),
                    backbone_checkpoint_sha256=source['backbone_checkpoint_sha256'],master_split_digest=source['master_split_digest'],
                    A_digest=source['A_digest'],B_digest=source['B_digest'],graph_digest=source['graph_digest'],
                    n_users=len(test['user_ids']),all_relevant_positives=int(full.sum()),candidate_hits=int(hits.sum()),
                    zero_hit_users=int((hits==0).sum()),candidate_recall50_macro=float(torch.where(full>0,hits/full,torch.zeros_like(full)).mean()),
                    full_truth_denominator=True,feature_names=list(FEATURES),new_model_fits=0,
                    source_selection_preserved=True,F_statistics_fit=False,F_validation_selection=False,
                    inference_seconds=time.monotonic()-started,**extra)
        atomic_json(directory/'result.json',result);output.append(dict(method=method,status='COMPLETED',result_path=str(directory/'result.json')))
        del source_cache,source_predictions,scores,prediction;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    return dict(status='COMPLETED',dataset=dataset,backbone=backbone,seed=int(seed),source_protocol=source_protocol,
                test_protocol='F',new_model_fits=0,methods=output)


