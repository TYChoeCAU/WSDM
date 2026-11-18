"""S1 frozen scoring of an explicitly supplied selected S2 MF catalog Top-50."""
from pathlib import Path
import collections,gc,json,time
import numpy as np,torch
from .paths import OUT
from .s1_inputs import coordinate_input,scorer,neighbors,s2_input
from studies.corrected import native
from studies.s3.cross_common import load,sha,atomic_json,read_csv,write_csv
from diagnostics.revision_evidence.pipeline import train_degrees
from diagnostics.graph_fusion.evaluate import ranking_metrics
from studies.s1.static import oracle_vectors,metric_vectors
from graph_rerank.graph import training_graph,graph_fingerprint
SEEDS=(2020,2021,2022)
METRICS=('ndcg@20','recall@20','candidate_recall50','oracle_ndcg20')
def read(path):return json.loads(Path(path).read_text())
def by_user(split, positive=False):
    out = collections.defaultdict(set)
    for u, i, y in zip(split['user_ids'].tolist(), split['item_ids'].tolist(), split['labels'].tolist()):
        if u > 0 and i > 0 and (not positive or y > 0): out[u].add(i)
    return dict(out)


def run(dataset, backbone, seed, device):
    dest = OUT/'artifacts/df_s1'/dataset/backbone/f'seed_{seed}'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'result.json').is_file():
        prior=read(dest/'result.json')
        _,_,checkpoint,_,_=coordinate_input(dataset,backbone,seed)
        if prior.get('status')!='COMPLETE' or (prior.get('dataset'),prior.get('backbone'),prior.get('seed'))!=(dataset,backbone,seed) or prior['source']['scorer_checkpoint_sha256']!=sha(checkpoint):
            raise ValueError('Existing S1 DF output has different selected scorer bindings. Use a separate output directory.')
        return prior
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    row,original,cp_path,spec,source_path=coordinate_input(dataset,backbone,seed)
    predict_batch_size=32768 if str(device).startswith('cuda') else 262144
    adapter=scorer(cp_path,device,predict_batch_size)
    bundle,bundle_path,saved,score_path,retriever_cp,parity,checks=s2_input(dataset,seed,adapter)
    split = {k:saved['split'][k] for k in ('user_ids','candidate_ids','labels','valid_mask','positive_count','s')}
    split['candidate_ids'] = split['candidate_ids'].long()
    history, _, degree = train_degrees(original)
    if history != by_user(bundle['train']): raise ValueError('S1/S2 all-observed train history differs')
    valid, truth = by_user(bundle['valid'], True), by_user(bundle['test'], True)
    if split['user_ids'].tolist() != sorted(truth): raise ValueError('S2 test user population differs')
    for j, u0 in enumerate(split['user_ids']):
        u = int(u0); mask = split['valid_mask'][j]; ids = split['candidate_ids'][j][mask]
        positives = torch.tensor(sorted(truth[u]), dtype=torch.long)
        if int(split['positive_count'][j]) != len(positives): raise ValueError('Full truth denominator differs')
        if not torch.equal(torch.isin(ids, positives), split['labels'][j][mask]): raise ValueError('Candidate labels differ')
        if len(ids) != len(torch.unique(ids)) or bool((ids <= 0).any()) or bool((ids >= bundle['item_num']).any()): raise ValueError('Invalid candidate IDs')
        excluded = torch.tensor(sorted(history.get(u,set()) | valid.get(u,set())), dtype=torch.long)
        if bool(torch.isin(ids, excluded).any()): raise ValueError('S1/S2 F exclusion differs')
    digest = native.tensor_digest({k:split[k] for k in ('user_ids','candidate_ids','labels','valid_mask','positive_count')})
    graph = training_graph(adapter.dataset); fingerprint = graph_fingerprint(graph)
    if not np.array_equal(np.asarray(graph.sum(0)).ravel(), np.asarray(degree)):
        raise ValueError('Popularity graph differs from actual S1 train history')
    actual = spec['npgc']
    k = int(actual['neighbor_k']); alpha = float(actual['alpha'])
    if k != 2: raise ValueError('Requested S1 dataset K=2 differs from recorded setting')
    neighbor_tables, neighbor_meta = neighbors(row,dataset,seed,fingerprint,k)
    mask = split['valid_mask']; users = split['user_ids'][:,None].expand_as(split['candidate_ids'])
    base = torch.zeros_like(split['s'])
    base[mask] = adapter.score_candidates(users[mask].numpy().copy(), split['candidate_ids'][mask].numpy().copy())
    records = [dict(user=int(u), items=split['candidate_ids'][i][mask[i]], base_score_raw=base[i][mask[i]])
               for i,u in enumerate(split['user_ids'])]
    npgc, native_error = native.native_scores(adapter,neighbor_tables,records,alpha)
    pop = torch.log1p(torch.as_tensor(degree))[split['candidate_ids']]
    oracle, recall = oracle_vectors(split)
    outputs = {'Base':base, 'Pop':pop, 'NPGC':npgc, 'Retriever':split['s'], 'Oracle':split['labels'].float()}
    rows=[]; numeric = dict(user_ids=split['user_ids'].numpy(), oracle_ndcg20=oracle, candidate_recall50=recall)
    for method, scores in outputs.items():
        if not torch.isfinite(scores[mask]).all(): raise ValueError('Nonfinite score in valid candidates')
        metrics = ranking_metrics(scores, split); per = metric_vectors(scores, split)
        if any(abs(float(per[m].mean())-metrics[m])>1e-3 for m in metrics):
            raise ValueError('Shared ranking/per-user evaluation differs')
        for metric,value in per.items(): numeric[method+'_'+metric] = value
        rows.append(dict(study='S1',dataset=dataset,backbone=backbone,seed=seed,protocol='DF',method=method,
             status='COMPLETE', **metrics, candidate_recall50=float(recall.mean()),oracle_ndcg20=float(oracle.mean()),
             candidate_digest=digest,n_users=len(base)))
    adapter.assert_frozen_and_unchanged()
    checks.update(full_truth_exact=True, exclusion_violations=0, all_methods_same_candidates=True,
                  existing_full_catalog_prefix_check=parity, native_parity_max_error=native_error,
                  manuscript_eligible=native_error<=1e-3, no_checkpoint_reselection=True)
    np.savez_compressed(dest/'per_user.npz', **numeric)
    result = dict(status='COMPLETE',dataset=dataset,backbone=backbone,seed=seed, rows=rows,checks=checks,
        source=dict(scorer_checkpoint=str(cp_path),scorer_checkpoint_sha256=adapter.checkpoint_sha256,
                    retriever_checkpoint=str(retriever_cp),retriever_checkpoint_sha256=saved['identity']['cf_checkpoint_sha256'],
                    retriever_scores_path=str(score_path),retriever_scores_sha256=sha(score_path),
                    split_bundle_path=str(bundle_path),split_bundle_sha256=sha(bundle_path),
                    neighbor_metadata=neighbor_meta,actual_npgc_settings=actual,inference_batch_size=predict_batch_size),
        seconds=time.monotonic()-started,no_training=True)
    atomic_json(dest/'result.json',result)
    print(json.dumps(dict(stage='s1_df_complete',dataset=dataset,backbone=backbone,seed=seed,seconds=result['seconds'])),flush=True)
    del adapter, original, saved, bundle
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def _aggregate():
    from scipy.stats import t
    results = [read(p) for p in sorted((OUT/'artifacts/df_s1').glob('*/*/seed_*/result.json'))]
    seed_rows = [r for result in results for r in result['rows']]
    write_csv(OUT/'DF_s1_seed_results.csv',seed_rows)
    summary=[]; paired=[]
    for dataset in ('ml-1m','yelp2018'):
      for bb in sorted(set(r['backbone'] for r in seed_rows)):
        selected=[r for r in seed_rows if r['dataset']==dataset and r['backbone']==bb]
        available=sorted(set(r['seed'] for r in selected)); n=len(available)
        if not n: continue
        status='COMPLETE' if available==list(SEEDS) else f'PARTIAL(n={n})'
        for method in ('Base','Pop','NPGC','Retriever','Oracle'):
          for metric in METRICS:
            values=np.array([r[metric] for r in selected if r['method']==method])*100
            summary.append(dict(dataset=dataset,backbone=bb,protocol='DF',method=method,metric=metric,
                n_seeds=n,seeds=';'.join(map(str,available)),status=status,mean_x100=float(values.mean()),
                sample_sd_x100=float(values.std(ddof=1)) if n>1 else None))
        for a,b in [('Pop','Base'),('NPGC','Base'),('Pop','NPGC')]:
          for metric in ('ndcg@20','recall@20'):
            lookup={(r['seed'],r['method']):r[metric] for r in selected}
            v=np.array([lookup[(s,a)]-lookup[(s,b)] for s in available])*100
            sd=float(v.std(ddof=1)) if n>1 else None; half=t.ppf(.975,n-1)*sd/np.sqrt(n) if n>1 else None
            paired.append(dict(dataset=dataset,backbone=bb,contrast=a+'-'+b,metric=metric,n_seeds=n,status=status,
                mean_x100=float(v.mean()),sample_sd_x100=sd,ci95_low_x100=float(v.mean()-half) if n>1 else None,
                ci95_high_x100=float(v.mean()+half) if n>1 else None))
    write_csv(OUT/'DF_s1_summary.csv',summary)
    write_csv(OUT/'DF_s1_paired.csv',paired)
    atomic_json(OUT/'DF_s1_checks.json',dict(completed_coordinates=len(results),checks=[dict(dataset=r['dataset'],backbone=r['backbone'],seed=r['seed'],**r['checks']) for r in results]))
    return dict(completed_coordinates=len(results),seed_rows=len(seed_rows))
