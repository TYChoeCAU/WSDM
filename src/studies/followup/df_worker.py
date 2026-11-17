"""Fixed and frozen learned DF inference. Run fixed before frozen."""
from .df_common import *
from diagnostics.revision_evidence.common import stable_order
import datetime
def directory(dataset,seed):return OUT/'artifacts/df_s3'/dataset/f'seed_{seed}'


def fixed(dataset,seed,device='cuda'):
    dest=directory(dataset,seed);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'fixed_result.json').exists():
        old_result=json.loads((dest/'fixed_result.json').read_text())
        _,mf_source=evidence(dataset,'MF_BPR',seed,'F','base');_,nfm_source=evidence(dataset,'NFM',seed,'F','base')
        if old_result.get('status')!='COMPLETE' or (old_result.get('dataset'),old_result.get('seed'))!=(dataset,seed):raise ValueError('Existing DF output coordinate differs')
        if old_result['signature']['checkpoint_sha256']!=nfm_source['backbone_checkpoint_sha256'] or old_result['signature']['retriever_checkpoint_sha256']!=mf_source['backbone_checkpoint_sha256'] or old_result['metrics']['base']['candidate_digest']!=mf_source['candidate_digest']:
            raise ValueError('Existing DF output has different source/candidate bindings. Use a separate output directory.')
        return old_result
    start=time.monotonic();modules=runtime();cache,paths,master,checks=read_f_cache(dataset,seed,modules)
    old=cache['test'];mf=selected_adapter(dataset,'MF_BPR',seed,modules,device=device)
    if mf.checkpoint_sha256!=cache['signature']['checkpoint_sha256']:raise ValueError('MF checkpoint changed')
    checked=score_split(mf,old);recomputed=ranking_metrics(checked,old);cached=ranking_metrics(old['s'],old)
    source_path,source=evidence(dataset,'MF_BPR',seed,'F','base')
    maximum=max(abs(recomputed[k]-source['metrics'][k]) for k in recomputed)
    checks.update(mf_scoring_check_max_metric_error=maximum,mf_pair_score_max_error=float((checked-old['s']).abs().max()),
        mf_F_base_original=source['metrics'],mf_F_base_recomputed=recomputed,mf_F_base_result=str(source_path))
    mf.assert_frozen_and_unchanged();del mf;torch.cuda.empty_cache()
    if maximum>1e-3:raise ValueError('MF scorer fails F Base parity')
    nfm=selected_adapter(dataset,'NFM',seed,modules,device=device)
    target,records,signature,progress,registry=prepared_features(dataset,seed,modules,cache,master,nfm,dest,compute=True)
    metrics={}
    for name,scores in [('base',target['s']),('item_degree_ranking',target['item_log_degree']),
                        ('original_nonparamgc',target['npgc_scores']),('retriever',checked),('oracle',target['labels'].float())]:
        value=save_prediction(dest/'predictions',name,scores,target)
        metrics[name]=dict(dataset=dataset,backbone='NFM',retriever='MF_BPR',training_seed=seed,method=name,
            protocol='DF',analysis_family='decoupled_fixed',status='COMPLETE',**value)
    ref=dict(candidate_recall50=float(((old['labels']&old['valid_mask']).sum(1).double()/old['positive_count'].double()).mean()),oracle_ndcg20=ranking_metrics(old['labels'].float(),old)['ndcg@20'])
    cr_error=abs(metrics['base']['candidate_recall50']-float(ref['candidate_recall50']))
    oracle_error=abs(metrics['base']['oracle_ndcg20']-float(ref['oracle_ndcg20']))
    checks.update(candidate_recall50_error=cr_error,oracle_ndcg20_error=oracle_error,all_methods_same_candidate_ids=True,
        reference_candidate_recall50=float(ref['candidate_recall50']),reference_oracle_ndcg20=float(ref['oracle_ndcg20']),
        native_direct_parity_max_error=progress['native_direct_parity_max_error'],source_scorer_checkpoint_sha256=nfm.checkpoint_sha256,
        all_mandatory_checks_pass=maximum<=1e-3 and cr_error<=1e-3 and oracle_error<=1e-3,
        manuscript_eligible=maximum<=1e-3 and cr_error<=1e-3 and oracle_error<=1e-3)
    nfm.assert_frozen_and_unchanged()
    result=dict(status='COMPLETE',dataset=dataset,seed=seed,metrics=metrics,checks=checks,signature=signature,
        registry=registry,seconds=time.monotonic()-start,new_model_fits=0,source_checkpoint_reselected=False,
        source_files=rt.sources(),worker_sha256=sha(Path(__file__)),finished_kst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat())
    atomic_json(dest/'fixed_result.json',result)
    print(json.dumps(dict(stage='df_fixed_complete',dataset=dataset,seed=seed,seconds=result['seconds'],metrics={k:v['metrics'] for k,v in metrics.items()},checks=checks)),flush=True)
    return result


def frozen(dataset,seed,device='cuda'):
    dest=directory(dataset,seed);fixed_result=json.loads((dest/'fixed_result.json').read_text())
    if not fixed_result['checks']['manuscript_eligible']:raise ValueError('DF fixed checks failed')
    modules=runtime();cache,paths,master,checks=read_f_cache(dataset,seed,modules)
    nfm=selected_adapter(dataset,'NFM',seed,modules,device=device)
    target,records,signature,progress,registry=prepared_features(dataset,seed,modules,cache,master,nfm,dest,compute=False)
    nfm.assert_frozen_and_unchanged();del nfm;torch.cuda.empty_cache()
    completed=[]
    for protocol in ('U','W'):
        # Valid/train are the unchanged source cache, required by the legacy PRM
        # constructor. No fitting/evaluation/selection is performed on them.
        source_paths=rt.cache_paths(dataset,'NFM',seed,protocol)
        source_cache=load(source_paths['feature_cache_path']);source_slates=load(source_paths['slate_cache_path'])
        if not torch.equal(source_cache['test']['user_ids'],target['user_ids']):raise ValueError('Frozen source population differs')
        if not torch.equal(source_cache['test']['positive_count'],target['positive_count']):raise ValueError('Frozen source denominator differs')
        if not all(torch.equal(a,b) for a,b in zip(source_cache['test']['positive_items'],target['positive_items'])):raise ValueError('Frozen source truth differs')
        target_cache={**source_cache,'signature':signature,'test':target}
        target_slates={**source_slates,'test_batches':modules['features'].eval_batches(records,'test')}
        for method in (*MASKS,'prm_principled'):
            out=dest/'frozen'/protocol/method/'result.json'
            start=time.monotonic();path,source=evidence(dataset,'NFM',seed,protocol,method)
            if out.exists():
                old_result=json.loads(out.read_text())
                if old_result.get('status')!='COMPLETE' or old_result['checkpoint_sha256']!=source['checkpoint_sha256'] or old_result['candidate_digest']!=target['candidate_digest']:
                    raise ValueError('Existing frozen DF output has different checkpoint/candidate bindings. Use a separate output directory.')
                completed.append(old_result);continue
            for k,target_key in [('backbone_checkpoint_sha256','checkpoint_sha256'),('master_split_digest','master_digest'),('graph_digest','graph_fingerprint')]:
                if source[k]!=signature[target_key]:raise ValueError('Frozen source identity differs '+k)
            if identity(source_cache['signature'])!=source['cache_identity']:raise ValueError('Source feature cache identity differs from selected checkpoint evidence')
            if source_slates['signature']!=source_cache['signature']:raise ValueError('Source slate/feature signature mismatch')
            cp_path=Path(source['checkpoint_path']);checksum=sha(cp_path)
            if checksum!=source['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
            cp=load(cp_path)
            if cp['cross_model_identity_hash']!=source['model_identity_hash']:raise ValueError('Source model identity mismatch')
            app=copy.deepcopy(cp['applied']);app.pop('deadline_utc',None)
            app.update(deadline_kst=None,expected_model_identity_hash=source['model_identity_hash'],expected_checkpoint_sha256=checksum)
            score_cache=target_cache;score_slates=target_slates;permutation=None
            if method=='prm_principled':
                # PRM has rank-position embeddings. Feed the actual NFM DF Base
                # order, then scatter predictions back to shared MF ID storage.
                permutation=stable_order(target['s'].double().masked_fill(~target['valid_mask'],-torch.inf),target['candidate_ids'])
                ordered={}
                for key,value in target.items():
                    if torch.is_tensor(value) and value.ndim>=2 and tuple(value.shape[:2])==tuple(target['s'].shape):
                        ix=permutation
                        while ix.ndim<value.ndim:ix=ix.unsqueeze(-1)
                        ordered[key]=torch.gather(value,1,ix.expand_as(value))
                    else:ordered[key]=value
                ordered['candidate_digest']=digest_split(ordered)
                if ranking_metrics(ordered['s'],ordered)!=ranking_metrics(target['s'],target):raise ValueError('NFM rank permutation changed Base metrics')
                score_cache={**target_cache,'test':ordered}
                score_slates={**target_slates,'test_batches':modules['features'].eval_batches(records_from_split(ordered),'test')}
                if not torch.equal(torch.sort(ordered['candidate_ids'],dim=1).values,torch.sort(target['candidate_ids'],dim=1).values):raise ValueError('PRM ordering changed candidate set')
            with torch.inference_mode():
                predicted=modules['models'].score_selected(method,score_cache,'test',cp_path,app,torch.device(device),slate_cache=score_slates)
                scores=predicted if permutation is None else torch.zeros_like(predicted).scatter(1,permutation,predicted)
            if sha(cp_path)!=checksum:raise ValueError('Checkpoint modified during inference')
            extra={}
            if method in MASKS:
                if cp['feature_names']!=list(FEATURES) or cp['feature_mask']!=list(MASKS[method]):raise ValueError('Frozen feature mask differs')
                if cp['residual_scale_t']!=source['residual_scale_t']:raise ValueError('Source residual scale differs')
                extra=dict(source_scaler_digest=source['scaler_digest'],source_residual_scale_t=cp['residual_scale_t'],feature_mask=cp['feature_mask'])
            value=save_prediction(out.parent,'predictions',scores,target)
            result=dict(status='COMPLETE',dataset=dataset,backbone='NFM',retriever='MF_BPR',training_seed=seed,method=method,
                protocol=protocol+'_to_DF',source_protocol=protocol,target_protocol='DF',analysis_family='frozen',selection_policy='identity_inclusive' if method in MASKS else 'native',
                source_checkpoint_selection_rule=source['training_summary'].get('selection_policy',source.get('selection_policy','recorded_source')),
                **value,**extra,source_result_path=str(path),source_result_sha256=sha(path),checkpoint_path=str(cp_path),checkpoint_sha256=checksum,
                source_model_identity_hash=source['model_identity_hash'],identity_selected=source.get('identity_selected',False),
                selected_epoch=source['selected_epoch'],new_model_fits=0,target_statistics_fit=False,target_validation_selection=False,
                source_selection_preserved=True,prm_position_policy='NFM score descending, item ID ascending; predictions scattered to common candidate storage' if method=='prm_principled' else None,seconds=time.monotonic()-start)
            atomic_json(out,result);completed.append(result)
            print(json.dumps(dict(stage='df_frozen_complete',dataset=dataset,seed=seed,protocol=protocol,method=method,seconds=result['seconds'],metrics=result['metrics'])),flush=True)
            del scores,cp;gc.collect();torch.cuda.empty_cache()
        del source_cache,source_slates,target_cache,target_slates;gc.collect()
    atomic_json(dest/'frozen_complete.json',dict(status='COMPLETE',results=len(completed),expected=18,new_model_fits=0))
    return completed
