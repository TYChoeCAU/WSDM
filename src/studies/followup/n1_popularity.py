"""Label-free CPU diagnosis of stored NPGC candidate scores, never model inference."""
from pathlib import Path
import csv, json, hashlib, time, gc, os, resource
from collections import Counter, defaultdict
import numpy as np
import torch
from scipy.stats import rankdata

from .paths import ROOT, OUT, EVIDENCE, MANIFEST, manifest, coordinate, input_file, resolve
from .runtime import result_path as native_result_path
INDEX = input_file('n1_evidence_index', None) if manifest().get('inputs', {}).get('n1_evidence_index') else None
S1_METRICS = input_file('s1_seed_metrics', EVIDENCE/'s1/benchmark_seed_metrics.csv')
DATASETS = ('yelp2018', 'amazon-books', 'ml-1m')
BACKBONES = ('NFM', 'xDeepFM', 'DCNV2', 'AutoInt', 'EulerNet', 'QNN_alpha')
METRICS = ('rho_delta_score_popularity', 'rho_rank_improvement_popularity',
           'top20_entrant_minus_exit_popularity', 'top20_entrant_minus_exit_popularity_zero_nochange')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def write(name, values):
    values = list(values); path = OUT/name
    keys = list(dict.fromkeys(key for row in values for key in row)) or ['status']
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(values)
    temporary.replace(path)


def dump(name, value):
    path = OUT/name; temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temporary.replace(path)


def load(path):
    # Tensor storage stays memory mapped. No label tensor is indexed or copied.
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def correlations(left, right, mask):
    a = rankdata(np.where(mask, left, np.nan), axis=1, method='average', nan_policy='omit')
    b = rankdata(np.where(mask, right, np.nan), axis=1, method='average', nan_policy='omit')
    n = mask.sum(1)
    a = np.where(mask, a-np.nansum(a, axis=1)[:, None]/n[:, None], 0)
    b = np.where(mask, b-np.nansum(b, axis=1)[:, None]/n[:, None], 0)
    denominator = np.sqrt(np.sum(a*a, axis=1)*np.sum(b*b, axis=1))
    result = np.full(len(n), np.nan)
    np.divide(np.sum(a*b, axis=1), denominator, out=result, where=(n>=2)&(denominator>0))
    return result


def diagnostics(base, native, pop, ids, mask):
    """All ranks are score-descending, item-ID-ascending within a user's Top50."""
    n, width = ids.shape
    base_order = np.lexsort((ids, np.where(mask, -base, np.inf)), axis=1)
    native_order = np.lexsort((ids, np.where(mask, -native, np.inf)), axis=1)
    base_rank = np.empty_like(base_order); native_rank = np.empty_like(native_order)
    positions = np.broadcast_to(np.arange(width), (n, width))
    np.put_along_axis(base_rank, base_order, positions, axis=1)
    np.put_along_axis(native_rank, native_order, positions, axis=1)
    # Positive means the item moves toward rank one under NPGC.
    improvement = base_rank-native_rank
    base20 = np.zeros_like(mask); native20 = np.zeros_like(mask)
    np.put_along_axis(base20, base_order[:, :20], True, axis=1)
    np.put_along_axis(native20, native_order[:, :20], True, axis=1)
    base20 &= mask; native20 &= mask
    entering = native20 & ~base20; leaving = base20 & ~native20
    count_in, count_out = entering.sum(1), leaving.sum(1)
    if not np.array_equal(count_in, count_out):
        raise ValueError('Top20 entrant/exit counts differ on the fixed candidate set')
    difference = np.full(n, np.nan); changed = count_in>0
    difference[changed] = (pop*entering).sum(1)[changed]/count_in[changed]-(pop*leaving).sum(1)[changed]/count_out[changed]
    return dict(rho_delta_score_popularity=correlations(native-base, pop, mask),
                rho_rank_improvement_popularity=correlations(improvement, pop, mask),
                top20_entrant_minus_exit_popularity=difference,
                top20_entrant_minus_exit_popularity_zero_nochange=np.where(changed, difference, 0)), count_in


def s1_inventory():
    """Audit explicitly supplied S1 Top50 storage without regenerating scores.

    The released scalar metrics cannot supply per-candidate native scores.
    Optional manifest s1.n1_top50_cache entries may provide the original
    feature-cache layout. A newly discovered native array requires an explicit
    benchmark-lineage review before it can be admitted to this diagnostic.
    """
    metric = {(r['dataset'], r['backbone'], int(r['seed'])): r for r in rows(S1_METRICS)
              if r['method']=='original_nonparamgc'} if S1_METRICS.is_file() else {}
    inventory = []
    for dataset in DATASETS:
      for backbone in BACKBONES:
       for seed in (2020, 2021, 2022):
        row = dict(study='S1', dataset=dataset, backbone=backbone, seed=seed, protocol='benchmark_U',
                   status='UNAVAILABLE', reason='missing_native_candidate_scores', top50_available=False,
                   native_candidate_scores_available=False, labels_used=False)
        supplied = coordinate('s1', dataset, backbone, seed, False)
        value = supplied.get('n1_top50_cache') if supplied else None
        if value is not None:
            path = resolve(value)
            row['candidate_cache_path'] = relative(path)
            if path.is_file():
                cache = load(path); part = cache['test']
                row.update(top50_available=all(k in part for k in ('candidate_ids','valid_mask','s')),
                           users=int(part['s'].shape[0]), candidate_width=int(part['s'].shape[1]),
                           candidate_cache_bytes=path.stat().st_size,
                           saved_test_field_names=';'.join(sorted(part)),
                           candidate_ids_dtype=str(part['candidate_ids'].dtype))
                # h is the separate graph descriptor, not the native score.
                if 'npgc_scores' in part:
                    row.update(status='REQUIRES_LINEAGE_REVIEW', reason='unexpected_stored_native_scores_require_benchmark_identity_check', native_candidate_scores_available=True)
                del cache, part; gc.collect()
            else:
                row['input_note'] = 'supplied_top50_cache_does_not_exist'
        else:
            row['input_note'] = 'no_local_top50_cache_supplied; published scalar evidence alone is insufficient'
        if (dataset, backbone, seed) in metric:
            row['scalar_evidence_csv'] = relative(S1_METRICS)
            row['scalar_evidence_available'] = True
        inventory.append(row)
    return inventory


def selected_evidence():
    """Use an explicit index or the public S3 result-path manifest adapter."""
    if INDEX is not None:
        return [r for r in rows(INDEX) if r['regime']=='R0' and r['analysis_family']=='matched'
                and r['method']=='original_nonparamgc']
    return [dict(dataset=d, backbone=b, training_seed=s, protocol=p,
                 result_path=str(native_result_path(d,b,s,p,'original_nonparamgc')))
            for d in ('yelp2018','ml-1m') for b in ('NFM','MF_BPR')
            for s in range(2020,2025) for p in ('U','W','F')]


def s3_coordinate(entry):
    started = time.monotonic()
    dataset, backbone, seed, protocol = entry['dataset'], entry['backbone'], int(entry['training_seed']), entry['protocol']
    result_path = resolve(entry['result_path']); result = read(result_path)
    cache_path = resolve(result['cache_path']); prediction_path = resolve(result['predictions_path'])
    cache = load(cache_path); part = cache['test']; predictions = load(prediction_path); sig = cache['signature']
    if (sig['dataset'], sig['backbone'], int(sig['training_seed']), sig['protocol']) != (dataset,backbone,seed,protocol):
        raise ValueError('Feature-cache coordinate/protocol differs from evidence index')
    if any(result[k]!=protocol for k in ('train_protocol','validation_protocol','test_protocol')):
        raise ValueError('Result is not a matched protocol coordinate')
    # JSON result metadata represents tuple-valued tensor shapes as lists.
    # Canonical JSON preserves every value while normalizing that storage type.
    if json.dumps(result['model_identity']['cache_signature'],sort_keys=True) != json.dumps(sig,sort_keys=True):
        raise ValueError('Stored model/cache signature differs')
    if predictions['cache_identity'] != result['cache_identity'] or predictions['model_identity_hash'] != result['model_identity_hash']:
        raise ValueError('Stored prediction/cache identity differs')
    if predictions['candidate_digest'] != part['candidate_digest'] or predictions['candidate_digest'] != result['candidate_digest']:
        raise ValueError('Stored candidate digests differ')
    if not torch.equal(predictions['user_ids'], part['user_ids']):
        raise ValueError('Stored prediction user order differs')
    total = len(part['user_ids'])
    if len(torch.unique(part['user_ids'])) != total:
        raise ValueError('Multiple test slates per user need a separate aggregation policy')
    sums=Counter(); sumsquares=Counter(); counts=Counter(); entrants=0; changed=0
    used=hashlib.sha256(); maximum_score_error=0.; maximum_pop_error=0.
    for first in range(0,total,512):
        last=min(first+512,total); selection=slice(first,last)
        ids=part['candidate_ids'][selection].numpy().astype(np.int64)
        mask=part['valid_mask'][selection].numpy().astype(bool)
        base=part['s'][selection].numpy().astype(np.float64)
        native=predictions['scores'][selection].numpy().astype(np.float64)
        pop=part['item_log_degree'][selection].numpy().astype(np.float64)
        if not mask.any(1).all() or not all(np.isfinite(x[mask]).all() for x in (base,native,pop)):
            raise ValueError('Empty user list or nonfinite stored score')
        if mask.sum(1).max()>50:raise ValueError('Candidate list exceeds Top50')
        if any(len(np.unique(i[m]))!=int(m.sum()) for i,m in zip(ids,mask)):
            raise ValueError('Duplicate candidate IDs')
        expected_pop=torch.log1p(cache['item_degree'][part['candidate_ids'][selection].long()].float()).numpy()
        maximum_pop_error=max(maximum_pop_error,float(np.max(np.abs(pop[mask]-expected_pop[mask]))))
        maximum_score_error=max(maximum_score_error,float(np.max(np.abs(native[mask]-part['npgc_scores'][selection].numpy()[mask]))))
        if maximum_pop_error>1e-3 or maximum_score_error>1e-3:
            raise ValueError('Stored native score or popularity feature differs')
        for name,array in [('ids',ids),('valid_mask',mask),('s',base),('native',native),('p',pop)]:
            used.update(name.encode());used.update(array.tobytes())
        values, entered=diagnostics(base,native,pop,ids,mask)
        entrants+=int(entered.sum());changed+=int((entered>0).sum())
        for name,value in values.items():
            finite=value[np.isfinite(value)];sums[name]+=float(finite.sum());sumsquares[name]+=float(np.square(finite).sum());counts[name]+=len(finite)
    row=dict(study='S3',dataset=dataset,backbone=backbone,seed=seed,protocol=protocol,status='COMPLETE',users=total,
             users_top20_changed=changed,users_top20_unchanged=total-changed,top20_entrant_items=entrants,
             popularity_definition='p_i=log(1+degree_A(i)); binary all-observed partition-A user-item edges',
             labels_used=False,checkpoint_loaded=False,inference_performed=False,source_result=relative(result_path),
             source_cache=relative(cache_path),source_predictions=relative(prediction_path),
             result_sha256=sha(result_path),predictions_sha256=sha(prediction_path),used_arrays_sha256=used.hexdigest(),
             maximum_saved_native_score_difference=maximum_score_error,maximum_popularity_feature_difference=maximum_pop_error,
             stored_candidate_digest=part['candidate_digest'],stored_cache_identity=result['cache_identity'],
             seconds=time.monotonic()-started)
    for name in METRICS:
        n=counts[name];mean=sums[name]/n if n else None
        row[name+'_user_mean']=mean;row[name+'_defined_users']=n;row[name+'_undefined_users']=total-n
        row[name+'_user_sd']=np.sqrt(max(0.,(sumsquares[name]-sums[name]**2/n)/(n-1))) if n>1 else None
    del cache,part,predictions;gc.collect()
    return row


def summarize(seed_rows):
    output=[];grouped=defaultdict(list)
    for row in seed_rows:
        grouped[row['study'],row['dataset'],row['backbone'],row['protocol']].append(row)
    for (study,dataset,backbone,protocol),group in sorted(grouped.items()):
        complete=[r for r in group if r['status']=='COMPLETE'];expected=3 if study=='S1' else 5
        row=dict(study=study,dataset=dataset,backbone=backbone,protocol=protocol,status='COMPLETE' if len(complete)==expected else f'PARTIAL(n={len(complete)})' if complete else 'UNAVAILABLE',
                 n_seeds=len(complete),expected_seeds=expected,seeds=';'.join(str(r['seed']) for r in sorted(complete,key=lambda r:r['seed'])),
                 aggregation='equal seed means of equal user means; sample SD across seeds (ddof=1)',
                 undefined_spearman='exclude constant/undersized lists and report exact counts',
                 entrants_primary='user mean among changed Top20 lists; all-user zero-nochange version reported separately')
        for name in METRICS:
            values=[r[name+'_user_mean'] for r in complete if r.get(name+'_user_mean') is not None]
            row[name+'_mean']=float(np.mean(values)) if values else None
            row[name+'_seed_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
            row[name+'_seeds_with_defined_mean']=len(values)
        output.append(row)
    return output


def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();inventory=s1_inventory()
    seed_rows=[{k:r[k] for k in ('study','dataset','backbone','seed','protocol','status','reason')} for r in inventory]
    write('N1_coordinate_inventory.csv',inventory);write('N1_npgc_popularity_seed.csv',seed_rows)
    selected=selected_evidence()
    expected={(d,b,s,p) for d in ('yelp2018','ml-1m') for b in ('NFM','MF_BPR') for s in range(2020,2025) for p in ('U','W','F')}
    observed=[(r['dataset'],r['backbone'],int(r['training_seed']),r['protocol']) for r in selected]
    if len(observed)!=len(set(observed)) or set(observed)!=expected:
        raise ValueError('S3 stored native-score evidence coverage differs from 60 requested coordinates')
    for entry in sorted(selected,key=lambda r:(r['dataset'],r['backbone'],int(r['training_seed']),r['protocol'])):
        try:
            row=s3_coordinate(entry)
        except Exception as error:
            row=dict(study='S3',dataset=entry['dataset'],backbone=entry['backbone'],seed=int(entry['training_seed']),protocol=entry['protocol'],status='INPUT_CHECK_FAILED',reason=str(error))
        seed_rows.append(row);inventory.append({k:v for k,v in row.items() if not k.startswith(METRICS)})
        write('N1_npgc_popularity_seed.csv',seed_rows);write('N1_coordinate_inventory.csv',inventory)
        status=dict(status='RUNNING',completed=sum(r['status']=='COMPLETE' for r in seed_rows),unavailable=sum(r['status']=='UNAVAILABLE' for r in seed_rows),examined=len(seed_rows),expected_coordinates=114,seconds=time.monotonic()-started)
        dump('N1_status.json',status);print(json.dumps({k:row.get(k) for k in ('study','dataset','backbone','seed','protocol','status','seconds','reason')}),flush=True)
    write('N1_npgc_popularity_summary.csv',summarize(seed_rows))
    counts=Counter(r['status'] for r in seed_rows)
    status=dict(status='COMPLETE_WITH_UNAVAILABLE_INPUTS' if counts.get('COMPLETE')==60 and counts.get('UNAVAILABLE')==54 else 'PARTIAL',coordinate_counts=dict(counts),expected_coordinates=114,
                S1_native_scores_missing=54 if counts.get('UNAVAILABLE')==54 else None,labels_used=False,checkpoint_loaded=False,new_inference=False,new_training=False,
                cpu_threads=torch.get_num_threads(),seconds=time.monotonic()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                source_sha256=sha(__file__),evidence_index_sha256=sha(INDEX) if INDEX else None,
                manifest_sha256=sha(MANIFEST) if MANIFEST else None,S1_metrics_sha256=sha(S1_METRICS) if S1_METRICS.is_file() else None)
    dump('N1_status.json',status);print(json.dumps(status),flush=True)


if __name__=='__main__':main()
