"""Original candidate RNG-only intervention at a fixed selected checkpoint."""
from __future__ import annotations
import copy,gc,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from .df_common import OUT,ROOT,PRIOR,MATCHED,rt,runtime,evidence,selected_adapter,sha,identity,load,atomic_json,read_csv,write_csv
from .paths import resolve,input_file,EVIDENCE
from studies.corrected.corrected_sampling import sample_negatives,group_rng,group_key,_record
from .candidate_helpers import sampled_top50,split_from_records,candidate_summary
from diagnostics.graph_fusion.evaluate import ranking_metrics
from diagnostics.revision_evidence.common import per_group_metrics
from studies.s3.cross_candidates import records_digest
from .p0p1_analysis import transfer,s3_index,summarize,SEEDS
DRAW_SEEDS=(8702031,8702032,8702033,8702034,8702035)
DATASETS=('ml-1m','yelp2018');BACKBONES=('NFM','MF_BPR');PROTOCOLS=('U','W')
METHODS=('base','item_degree_ranking','original_nonparamgc')
METRICS=('ndcg@10','ndcg@20','recall@10','recall@20','candidate_recall50','oracle_ndcg20')
def resample(record,draw_seed,protocol,item_num,degree):
    key=json.loads(record['group_key'])
    if key['protocol']!='W' or key['seed']!=2020 or key['phase']!='test':
        raise ValueError('Expected original S3 test W group entropy')
    entropy={k:key[k] for k in ('dataset','seed','phase','user','slate_index','protocol')}
    entropy['seed']=int(draw_seed)
    targets=torch.as_tensor(record['positive_items']).long().tolist()
    excluded=set(torch.as_tensor(record['excluded_items']).long().tolist())|set(targets)
    negatives,meta=sample_negatives(group_rng(**entropy),item_num=item_num,excluded=excluded,
                                  count=100*len(targets),protocol=protocol,item_degree=degree)
    meta.update(rng_protocol='W',rng_policy='original S3 W entropy with candidate seed changed only',
                original_candidate_seed=2020,candidate_draw_seed=int(draw_seed))
    result=_record(record['user'],targets,negatives,torch.as_tensor(record['history']).tolist(),
                   group_key(**entropy),meta)
    for name in ('excluded_items','phase','slate_index','history_feedback'):
        result[name]=copy.copy(record[name])
    result['candidate_protocol']=protocol
    if not torch.equal(result['positive_items'],record['positive_items']):
        raise ValueError('Resampling altered full relevant truth')
    return result


def pool_inputs(dataset):
    master=rt.common.read_master(dataset)
    path=MATCHED/'master'/dataset/'fixed_W_pools.pt'
    pools=load(path)
    if pools['signature']['master_digest']!=master['signature']['sha256']:
        raise ValueError('Existing W pool/master mismatch')
    if identity([(r['user'],r['positive_items'].tolist()) for r in pools['test']])!=pools['signature']['full_truth_digests']['test']:
        raise ValueError('Original test full-truth digest mismatch')
    return master,pools,path


def cell_dir(dataset,backbone,draw,protocol):
    return OUT/'artifacts/p4_draws'/dataset/backbone/f'draw_{draw}'/protocol


def run_coordinate(dataset,backbone,device,draws=None):
    mods=runtime()
    master,pools,poolpath=pool_inputs(dataset)
    _,source=evidence(dataset,backbone,2020,'W','base')
    signature=source['model_identity']['cache_signature']
    selected_path=rt.backbone_result(dataset,backbone,2020);selected=json.loads(selected_path.read_text())
    graph_path=input_file(dataset+'_nonparamgc_neighbors',MATCHED/'master'/dataset/'neighbors/nonparamgc_k2.npz')
    a=dict(checkpoint_path=str(resolve(selected['checkpoint'])),checkpoint_sha256=selected['checkpoint_sha256'],pool_sha256=sha(poolpath),
        graph_fingerprint=signature['graph_fingerprint'],native_neighbor_path=str(graph_path),native_neighbor_sha256=signature['native_neighbors_sha256'],
        neighbor_k=signature['neighbor_k'],alpha=signature['alpha'],source_test_truth_digest=pools['signature']['full_truth_digests']['test'])
    if sha(poolpath)!=a['pool_sha256']:raise ValueError('Original W pool changed after audit')
    adapter=selected_adapter(dataset,backbone,2020,mods,device=device)
    if adapter.checkpoint_sha256!=a['checkpoint_sha256']:raise ValueError('Checkpoint changed after audit')
    graph,fingerprint,_,_=mods['features']._A_graph(master,adapter)
    if fingerprint!=a['graph_fingerprint']:raise ValueError('Frozen scorer A graph differs')
    neighbors,table=mods['features']._old.tables(a['native_neighbor_path'],fingerprint,int(a['neighbor_k']),'nonparamgc')
    if table['sha256']!=a['native_neighbor_sha256']:raise ValueError('Native neighbors changed after audit')
    del graph
    item_num=int(master['item_num']);degree=np.asarray(master['item_degree'])
    source_hashes=rt.sources()
    for draw in (DRAW_SEEDS if draws is None else draws):
      if draw not in DRAW_SEEDS:raise ValueError('Unregistered draw seed')
      for protocol in PROTOCOLS:
        dest=cell_dir(dataset,backbone,draw,protocol);done=dest/'result.json'
        if done.exists():
            old=json.loads(done.read_text())
            if old['checkpoint_sha256']!=a['checkpoint_sha256'] or old['draw_seed']!=draw or old['status']!='COMPLETE' or old['pool_sha256']!=a['pool_sha256'] or old['native_k']!=int(a['neighbor_k']) or old['alpha']!=float(a['alpha']):raise ValueError('Existing draw result is incompatible')
            continue
        start=time.monotonic();ranked=[];prepool=hashlib.sha256();sampling_seconds=scoring_seconds=0.
        for first in range(0,len(pools['test']),256):
            t=time.monotonic()
            group=[resample(r,draw,protocol,item_num,degree) for r in pools['test'][first:first+256]]
            prepool.update(records_digest(group).encode());sampling_seconds+=time.monotonic()-t
            t=time.monotonic();ranked.extend(sampled_top50(adapter,group,group_batch=256));scoring_seconds+=time.monotonic()-t
            if first%4096==0:
                atomic_json(dest/'progress.json',dict(status='RUNNING',phase='sampling_and_scoring',users_done=first+len(group),users_total=len(pools['test']),
                    dataset=dataset,backbone=backbone,draw_seed=draw,protocol=protocol,seconds=time.monotonic()-start,
                    sampling_seconds=sampling_seconds,scoring_seconds=scoring_seconds))
                print(json.dumps(dict(event='P4_TOP50',dataset=dataset,backbone=backbone,draw_seed=draw,protocol=protocol,
                    users_done=first+len(group),users_total=len(pools['test']),seconds=time.monotonic()-start)),flush=True)
        split=split_from_records(ranked);cstats,_,_=candidate_summary(split,ranked,item_num)
        truth=identity([(r['user'],r['positive_items'].tolist()) for r in ranked])
        if truth!=a['source_test_truth_digest']:raise ValueError('Draw changed user/full truth identity')
        t=time.monotonic();npgc,parity=mods['features'].native_scores(adapter,neighbors,ranked,float(a['alpha']))
        native_seconds=time.monotonic()-t
        if parity>1e-3:raise ValueError('Original native NonParamGC parity failed')
        methods={'base':split['s'],'item_degree_ranking':torch.log1p(master['item_degree'].float())[split['candidate_ids'].long()],
                 'original_nonparamgc':npgc}
        rows=[]
        for method,scores in methods.items():
            metrics=ranking_metrics(scores,split);per=per_group_metrics(scores,split)
            if any(abs(float(per[k].mean())-metrics[k])>1e-3 for k in metrics):raise ValueError('Common/per-user evaluator mismatch')
            rows.append(dict(dataset=dataset,backbone=backbone,training_seed=2020,draw_seed=draw,protocol=protocol,method=method,
                status='COMPLETE',inference_device=device,cpu_threads=torch.get_num_threads(),**metrics,**cstats,candidate_digest=split['candidate_digest'],checkpoint_sha256=a['checkpoint_sha256']))
        adapter.assert_frozen_and_unchanged()
        result=dict(status='COMPLETE',dataset=dataset,backbone=backbone,training_seed=2020,draw_seed=draw,protocol=protocol,
            checkpoint_sha256=a['checkpoint_sha256'],checkpoint_path=a['checkpoint_path'],source_inventory=a,rows=rows,
            pool_sha256=a['pool_sha256'],source_full_truth_digest=truth,new_prepool_chunked_digest=prepool.hexdigest(),
            prepool_digest_algorithm='SHA256 of 256-user records_digest chunks in original order',candidate_digest=split['candidate_digest'],
            original_sampler=True,rng_seed_only_intervention=True,negative_per_relevant=100,rng_protocol='W',native_k=int(a['neighbor_k']),alpha=float(a['alpha']),
            native_direct_parity_max_error=parity,source_hashes=source_hashes,new_model_fits=0,checkpoint_reselection=False,
            inference_device=device,cpu_threads=torch.get_num_threads(),
            sampling_seconds=sampling_seconds,scoring_seconds=scoring_seconds,native_seconds=native_seconds,seconds=time.monotonic()-start)
        atomic_json(done,result);atomic_json(dest/'progress.json',dict(status='COMPLETE',seconds=result['seconds']))
        print(json.dumps(dict(event='P4_CELL_COMPLETE',dataset=dataset,backbone=backbone,draw_seed=draw,protocol=protocol,
            seconds=result['seconds'],sampling_seconds=sampling_seconds,scoring_seconds=scoring_seconds,native_seconds=native_seconds,rows=rows)),flush=True)
        del ranked,split,npgc,methods;gc.collect();torch.cuda.empty_cache()
    if sha(a['checkpoint_path'])!=a['checkpoint_sha256'] or sha(poolpath)!=a['pool_sha256']:
        raise ValueError('Original input changed during P4 inference')


def _aggregate_unlocked():
    all_rows=[];inventory=[];runtime_rows=[]
    for dataset in DATASETS:
      for backbone in BACKBONES:
       for draw in DRAW_SEEDS:
        for protocol in PROTOCOLS:
            path=cell_dir(dataset,backbone,draw,protocol)/'result.json'
            row=dict(dataset=dataset,backbone=backbone,training_seed=2020,draw_seed=draw,protocol=protocol,result_path=str(path))
            if path.exists():
                data=json.loads(path.read_text());row['status']=data['status']
                device=data.get('inference_device')
                device_evidence='result.json'
                row.update(inference_device=device or 'UNRECORDED',device_evidence=device_evidence)
                all_rows.extend({**r,'inference_device':device or 'UNRECORDED','device_evidence':device_evidence} for r in data['rows'])
                runtime_rows.append({**row,**{k:data[k] for k in ('seconds','sampling_seconds','scoring_seconds','native_seconds')}})
            else:
                progress=path.with_name('progress.json')
                row['status']=json.loads(progress.read_text()).get('status','NOT_STARTED') if progress.exists() else 'NOT_STARTED'
            inventory.append(row)
    write_csv(OUT/'P4_run_inventory.csv',inventory)
    if all_rows:write_csv(OUT/'P4_seed_results.csv',all_rows)
    if runtime_rows:write_csv(OUT/'P4_runtime.csv',runtime_rows)
    source=s3_index(read_csv(input_file('s3_seed_results',EVIDENCE/'s3/extracted_seed_results.csv')));idx={(r['dataset'],r['backbone'],int(r['draw_seed']),r['protocol'],r['method']):r for r in all_rows}
    summary=[];contrasts=[]
    for d in DATASETS:
      for b in BACKBONES:
       for protocol in PROTOCOLS:
        for method in METHODS:
         complete=[seed for seed in DRAW_SEEDS if (d,b,seed,protocol,method) in idx]
         if not complete:continue
         for metric in METRICS:
            values=[float(idx[d,b,seed,protocol,method][metric]) for seed in complete]
            baseline=[float(source[d,b,seed,'matched',protocol,method][metric]) for seed in SEEDS]
            s=summarize(values);old=summarize(baseline)
            summary.append(dict(dataset=d,backbone=b,protocol=protocol,method=method,metric=metric,status='COMPLETE' if len(complete)==5 else f'PARTIAL(n={len(complete)})',
                n_candidate_draws=len(complete),fixed_training_seed=2020,draw_seeds=';'.join(map(str,complete)),
                draw_mean_x100=100*s['mean_raw'],draw_sample_sd_x100=100*s['sample_sd_raw'] if len(complete)>1 else None,
                training_mean_x100=100*old['mean_raw'],training_sample_sd_x100=100*old['sample_sd_raw'],n_training_seeds=5,
                draw_uncertainty='candidate RNG repeats at fixed checkpoint',training_uncertainty='checkpoint training repeats at fixed candidate pool',
                source_evidence='artifact:results/s3/extracted_seed_results.csv'))
        for contrast,method in [('Pop-Base','item_degree_ranking'),('NPGC-Base','original_nonparamgc')]:
         complete=[seed for seed in DRAW_SEEDS if all((d,b,seed,protocol,m) in idx for m in (method,'base'))]
         if not complete:continue
         for metric in ('ndcg@10','ndcg@20','recall@10','recall@20'):
            values=[float(idx[d,b,seed,protocol,method][metric])-float(idx[d,b,seed,protocol,'base'][metric]) for seed in complete]
            baseline=[float(source[d,b,seed,'matched',protocol,method][metric])-float(source[d,b,seed,'matched',protocol,'base'][metric]) for seed in SEEDS]
            s=summarize(values);old=summarize(baseline)
            summary.append(dict(dataset=d,backbone=b,protocol=protocol,method=contrast,metric=metric,status='COMPLETE' if len(complete)==5 else f'PARTIAL(n={len(complete)})',
                n_candidate_draws=len(complete),fixed_training_seed=2020,draw_seeds=';'.join(map(str,complete)),
                draw_mean_x100=100*s['mean_raw'],draw_sample_sd_x100=100*s['sample_sd_raw'] if len(complete)>1 else None,
                training_mean_x100=100*old['mean_raw'],training_sample_sd_x100=100*old['sample_sd_raw'],n_training_seeds=5,
                draw_uncertainty='paired candidate RNG differences at fixed checkpoint',training_uncertainty='paired checkpoint training repeats at fixed candidate pool',
                source_evidence='artifact:results/s3/extracted_seed_results.csv'))
       for name,method in [('Pop-Base','item_degree_ranking'),('NPGC-Base','original_nonparamgc')]:
        complete=[seed for seed in DRAW_SEEDS if all((d,b,seed,p,m) in idx for p in PROTOCOLS for m in (method,'base'))]
        if len(complete)<2:continue
        draw_values={p:[float(idx[d,b,seed,p,method]['ndcg@20'])-float(idx[d,b,seed,p,'base']['ndcg@20']) for seed in complete] for p in PROTOCOLS}
        training_values={p:[float(source[d,b,seed,'matched',p,method]['ndcg@20'])-float(source[d,b,seed,'matched',p,'base']['ndcg@20']) for seed in SEEDS] for p in PROTOCOLS}
        meta=dict(dataset=d,backbone=b,contrast=name,source_protocol='U',target_protocol='W')
        old=transfer(training_values['U'],training_values['W'],meta)
        new=transfer(draw_values['U'],draw_values['W'],meta,seeds=complete)
        new['n_candidate_draws']=new.pop('n_training_seeds');new['draw_seeds']=new.pop('training_seeds')
        new.update(fixed_training_seed=2020,status='COMPLETE' if len(complete)==5 else f'PARTIAL(n={len(complete)})',
            uncertainty_unit='paired_candidate_draws_fixed_checkpoint',training_repeat_classification=old['classification'],
            classification_changed=new['classification']!=old['classification'],training_source_sd_x100=old['source_sample_sd_x100'],
            training_target_sd_x100=old['target_sample_sd_x100'],source_effects_by_draw_raw=new.pop('source_effects_by_seed_raw'),
            target_effects_by_draw_raw=new.pop('target_effects_by_seed_raw'))
        contrasts.append(new)
    if summary:write_csv(OUT/'P4_draw_variance.csv',summary)
    if contrasts:write_csv(OUT/'P4_transfer.csv',contrasts)
    return dict(completed_draw_cells=sum(r['status']=='COMPLETE' for r in inventory),expected_draw_cells=40,metric_rows=len(all_rows))
