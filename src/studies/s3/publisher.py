"""Unchanged common result publisher, with the twelve-method release scope."""
import json
import torch
from .cross_common import *
from .uniform_models import tensor_hash
from diagnostics.revision_evidence.common import per_group_metrics
NUMERICAL_ATOL=1e-3

def publish(dataset, backbone, seed, protocol, method, cache, cache_path, master, fit, path, applied, seconds):
    test = cache['test']; per = per_group_metrics(fit['scores'], test)
    metrics = {key: float(value.mean()) for key, value in per.items()}
    if max(abs(metrics[k] - fit['metrics'][k]) for k in metrics) > NUMERICAL_ATOL:
        raise ValueError('Published per-user metrics disagree with evaluator')
    prediction = dict(scores=fit['scores'], per_user_metrics=per, user_ids=test['user_ids'],
        candidate_digest=test['candidate_digest'], cache_identity=identity(cache['signature']), model_identity_hash=fit['model_identity_hash'])
    if 'trained_only_scores' in fit:
        prediction.update(trained_only_scores=fit['trained_only_scores'], trained_only_per_user_metrics=per_group_metrics(fit['trained_only_scores'], test))
    prediction_path = path / 'test_predictions.pt'; save(prediction_path, prediction)
    hits = test['labels'].sum(1).double(); total = test['positive_count'].double()
    recall = torch.where(total > 0, hits / total, torch.zeros_like(total))
    result = {k: v for k, v in fit.items() if not torch.is_tensor(v)}
    config_path = path / 'applied_config.json'
    if not config_path.exists(): atomic_json(config_path, applied)
    signature = cache['signature']; summary = fit['training_summary']
    trials = 1
    result.update(status='COMPLETED', cohort='matched_AB_fixed2020', dataset=dataset, backbone=backbone,
        training_seed=seed, seed=seed, protocol=protocol, method=method, train_protocol=protocol,
        validation_protocol=protocol, test_protocol=protocol, metrics=metrics, predictions_path=str(prediction_path),
        cache_path=str(cache_path), candidate_digest=test['candidate_digest'],
        scaler_digest=tensor_hash(cache['scaler']['mean'], cache['scaler']['std']), residual_scale_t=float(cache['scaler']['score_scale']),
        backbone_checkpoint_sha256=signature['checkpoint_sha256'], master_split_digest=master['signature']['sha256'],
        A_digest=master['split_digests']['train'], B_digest=master['split_digests']['refinement'], graph_digest=signature['graph_fingerprint'],
        full_truth_digest=signature['candidate_signature']['pool']['full_truth_digests']['test'],
        source_applied_config=str(config_path), resolved_config_path=str(config_path), resolved_config_sha256=sha(config_path),
        backbone_selector='fixed validation-W NDCG@20', reranker_selector=f'validation-{protocol} NDCG@20' if method not in ('base', 'item_degree_ranking', 'original_nonparamgc') else 'none',
        n_users=len(hits), candidate_recall50_macro=float(recall.mean()), candidate_recall50_micro=float(hits.sum() / total.sum()),
        candidate_hits=int(hits.sum()), all_relevant_positives=int(total.sum()), zero_hit_users=int((hits == 0).sum()),
        full_truth_denominator=True, feature_names=list(FEATURES), execution_kind='reused_fit_inference' if fit.get('reused_compatible_fit') else 'new_fit_and_inference' if method not in ('base', 'item_degree_ranking', 'original_nonparamgc') else 'frozen_score_evaluation',
        actual_new_fits=0 if fit.get('reused_compatible_fit') or method in ('base', 'item_degree_ranking', 'original_nonparamgc') else trials,
        actual_new_inference_passes=4 if method in MASKS else 2 if method == 'prm_principled' else 0,
        inference_count_definition='selected-model validation/test passes, plus trained-only validation/test for S/P; epoch validation and candidate-score generation counted separately',
        selected_epoch=summary.get('selected_epoch'), identity_selected=bool(fit.get('identity_selected', False)),
        wall_seconds=seconds, parameter_count=801 if method in MASKS else summary.get('trainable_parameters'))
    atomic_json(path / 'result.json', result)
    print(json.dumps(dict(event='METHOD_COMPLETED', dataset=dataset, backbone=backbone, seed=seed, protocol=protocol, method=method, metrics=metrics)), flush=True)
