"""Evaluate Base/Pop/native NPGC on actual saved benchmark pre-Top50 pools.
This is a benchmark sampled pool, not the whole eligible catalog (protocol F).
No candidate reconstruction or model fitting occurs.
"""
from pathlib import Path
import json,time,hashlib,gc
import numpy as np,torch
from .paths import OUT
from .s1_inputs import coordinate_input,scorer,neighbors
from .fullpool_native import native_scores
from .metric_helpers import metric_vectors
from studies.s3.cross_common import atomic_json,write_csv,sha
from diagnostics.revision_evidence.pipeline import positive_records,train_degrees
from graph_rerank.graph import training_graph,graph_fingerprint
def evaluate(records,scores,degree=None):
 acc={};users=[];topsha=hashlib.sha256()
 for first in range(0,len(records),128):
  sub=records[first:first+128];width=max(20,max(len(g['items']) for _,g,_ in sub));n=len(sub)
  ids=np.zeros((n,width),np.int64);mask=np.zeros_like(ids,bool);labels=np.zeros_like(ids,bool);vals=np.zeros_like(ids,np.float64);positive=np.empty(n,np.int64)
  for j,(u,g,p) in enumerate(sub):
   a=g['items'].numpy().astype(np.int64);m=len(a);ids[j,:m]=a;mask[j,:m]=True;labels[j,:m]=np.isin(a,p.numpy());positive[j]=len(p);users.append(u)
   if degree is not None:vals[j,:m]=degree[a]
   else:vals[j,:m]=scores[first+j].numpy()
  v,order=metric_vectors(vals,ids,mask,labels,positive)
  selected=np.take_along_axis(ids,order[:,:50],axis=1);valid=np.take_along_axis(mask,order[:,:50],axis=1)
  # digest matches historical MostPop fixed 50-column selected ID matrix.
  if selected.shape[1]<50:selected=np.pad(selected,((0,0),(0,50-selected.shape[1])))
  topsha.update(selected.astype(np.int64).tobytes())
  hit=np.take_along_axis(labels,order[:,:50],axis=1).sum(1);prefix=np.r_[0,np.cumsum(1/np.log2(np.arange(2,22)))];den=prefix[np.minimum(20,positive)]
  v['candidate_recall50']=hit/positive;v['oracle_ndcg20']=prefix[np.minimum(20,hit)]/den
  for k,a in v.items():acc.setdefault(k,[]).append(a)
 acc={k:np.concatenate(v) for k,v in acc.items()}
 return {k:float(v.mean()) for k,v in acc.items()},acc,topsha.hexdigest()

def run(dataset,backbone,seed,device='cuda',batch_size=32768):
    """Compute full-pool and Base-selected Top50 metrics from the same saved input.

    No historical metric row is imported into this run. Its Top50 comparison
    restricts the already computed native score vector to the Base Top50 IDs.
    Historical MostPop/checkpoint audit rows remain separate release evidence.
    """
    from studies.corrected.corrected_sampling import rank_record
    started=time.monotonic();dest=OUT/'artifacts/fullpool'/dataset/backbone/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
    marker=dest/'result.json'
    row,cache,checkpoint,spec,source_path=coordinate_input(dataset,backbone,seed,fullpool=True)
    source_sha=sha(source_path);checkpoint_sha=sha(checkpoint)
    if marker.is_file():
        old=json.loads(marker.read_text())
        if old.get('status')!='COMPLETE' or (old.get('dataset'),old.get('backbone'),old.get('seed'))!=(dataset,backbone,seed):
            raise ValueError('Existing full-pool result is incomplete or belongs to another coordinate')
        if old.get('source_pool_sha256')!=source_sha or old.get('checkpoint_sha256')!=checkpoint_sha or 'top50_metrics' not in old:
            raise ValueError('Existing full-pool result has different input bindings or lacks its Top50 comparison. Use a separate output directory.')
        return old
    records=positive_records(cache['test_batches'])
    _,_,degree=train_degrees(cache)
    top_records=[];top_indices=[]
    for user,group,positive in records:
        ids=group['items'];scores=group['base_score_raw']
        if ids.dtype not in (torch.int32,torch.int64) or len(ids)!=len(scores):raise ValueError('Invalid candidate ID/score storage')
        if len(ids)!=len(ids.unique()) or not len(positive) or len(positive)!=len(positive.unique()):raise ValueError('Duplicate candidates or empty/duplicate full truth')
        if bool((ids<=0).any()) or bool((ids>=len(degree)).any()):raise ValueError('Invalid candidate IDs')
        # Existing label-blind rank_record keeps full truth and resolves ties by
        # item ID. Full-pool records themselves are never regrouped or deduplicated.
        ranked=rank_record(dict(group,user=user,labels=torch.isin(ids,positive),positive_items=positive,positive_count=len(positive)),scores,top_m=50)
        index={int(item):i for i,item in enumerate(ids)}
        top_indices.append(torch.tensor([index[int(item)] for item in ranked['items']],dtype=torch.long))
        top_records.append((user,ranked,positive))
    baseline,bvec,_=evaluate(records,[g['base_score_raw'] for _,g,_ in records])
    pop,pvec,_=evaluate(records,None,degree)
    top_base,_,_=evaluate(top_records,[g['base_score_raw'] for _,g,_ in top_records])
    top_pop,_,_=evaluate(top_records,None,degree)
    checks=[]
    for metric in ('ndcg@10','ndcg@20','recall@10','recall@20'):
        delta=abs(baseline[metric]-top_base[metric])
        checks.append(dict(check='fullpool_Base_equals_same_input_Top50_Base_'+metric.replace('@',''),actual=baseline[metric],expected=top_base[metric],abs_difference=delta,tolerance=1e-3,status='PASS' if delta<=1e-3 else 'FAIL'))
    if any(c['status']!='PASS' for c in checks):raise ValueError('Full-pool/Base Top50 prefix invariance failed')
    adapter=scorer(checkpoint,device,batch_size)
    graph=training_graph(adapter.dataset);fingerprint=graph_fingerprint(graph)
    if not np.array_equal(np.asarray(graph.sum(0)).ravel().astype(np.int64),degree):raise ValueError('Train graph degree differs from saved histories')
    k=int(spec['npgc']['neighbor_k']);alpha=float(spec['npgc']['alpha'])
    tables,metadata=neighbors(row,dataset,seed,fingerprint,k)
    score_start=time.monotonic();values,parity=native_scores(adapter,tables,[dict(g,user=u) for u,g,p in records],alpha);score_seconds=time.monotonic()-score_start
    adapter.assert_frozen_and_unchanged();metric,nvec,_=evaluate(records,values)
    top_native=[value[index] for value,index in zip(values,top_indices)]
    top_npgc,_,_=evaluate(top_records,top_native)
    checks.append(dict(check='batched_native_score_matches_existing_GraphReranker_eq12',actual=parity,expected=0.0,abs_difference=parity,tolerance=1e-3,status='PASS' if parity<=1e-3 else 'FAIL'))
    np.savez_compressed(dest/'user_metrics.npz',user_ids=np.asarray([u for u,g,p in records],np.int64),**{m+'__'+key:v for m,vector in [('Base',bvec),('NPGC',nvec),('Pop',pvec)] for key,v in vector.items()})
    result=dict(dataset=dataset,backbone=backbone,seed=seed,status='COMPLETE' if all(c['status']=='PASS' for c in checks) else 'CHECK_FAILED',users=len(records),pool_items=sum(len(g['items']) for _,g,_ in records),
        metrics={'Base':baseline,'NPGC':metric,'Pop':pop},top50_metrics={'Base':top_base,'NPGC':top_npgc,'Pop':top_pop},checks=checks,
        native_parity_max_error=parity,checkpoint_sha256=adapter.checkpoint_sha256,source_pool_sha256=source_sha,
        source_pool='artifact:s1/'+dataset+'/'+backbone+'/seed_'+str(seed)+'/pre-Top50',graph_fingerprint=fingerprint,neighbor_metadata=metadata,k=k,alpha=alpha,
        elapsed_seconds=time.monotonic()-started,native_score_seconds=score_seconds,native_candidate_items_per_second=sum(len(g['items']) for _,g,_ in records)/score_seconds,
        no_training=True,no_pool_reconstruction=True,top50_comparison_source='same saved pool, Base score-desc/item-ID-asc Top50; native scores restricted without a second inference',
        historical_mostpop_comparison='NOT_PERFORMED: historical reference metrics are not mixed into a fresh run')
    atomic_json(marker,result)
    return result

def aggregate():
    """Original seed/macro/ratio formulas, with explicit local input coordinates.

    ALL_FIVE_MACRO retains the five backbones of the measured full-pool study.
    It averages those five within each seed before the three-seed sample SD.
    Missing inputs/results remain missing rather than borrowing release metrics.
    """
    from .paths import manifest
    from scipy.stats import t
    backbone_order=('NFM','xDeepFM','DCNV2','AutoInt','EulerNet')
    seeds=(2020,2021,2022)
    coordinates={(r['dataset'],r['backbone'],int(r['seed'])) for r in manifest().get('s1',[]) if any(key in r for key in ('fullpool_cache','benchmark_cache'))}
    results={}
    for path in sorted((OUT/'artifacts/fullpool').glob('*/*/seed_*/result.json')):
        value=json.loads(path.read_text());key=(value['dataset'],value['backbone'],int(value['seed']))
        if key in results:raise ValueError('Duplicate full-pool output coordinate')
        if key[2] not in seeds:raise ValueError('Full-pool summary requires the registered three S1 split seeds')
        results[key]=(path,value);coordinates.add(key)
    seed_rows=[];inventory=[];comparison=[];checks=[]
    for dataset,backbone,seed in sorted(coordinates):
        path,value=results.get((dataset,backbone,seed),(None,{}));status=value.get('status','PENDING')
        identity=dict(dataset=dataset,backbone=backbone,seed=seed)
        inventory.append(dict(**identity,status=status,reason=value.get('reason',''),result_path=str(path) if path else ''))
        for method in ('Base','NPGC','Pop'):
            row=dict(**identity,method=method,status=status,ndcg20='',recall20='')
            if value.get('metrics'):
                row.update({key.replace('@',''):number for key,number in value['metrics'][method].items()})
                row.update(users=value['users'],pool_items=value['pool_items'],elapsed_seconds=value['elapsed_seconds'],source=str(path))
            seed_rows.append(row)
        checks.extend(dict(**identity,**check) for check in value.get('checks',[]))
        if status=='COMPLETE':
            if 'top50_metrics' not in value:raise ValueError('Completed output lacks its same-input Top50 comparison')
            for condition,metrics in [('full_pool',value['metrics']),('backbone_Top50',value['top50_metrics'])]:
                base=metrics['Base']['ndcg@20'];native=metrics['NPGC']['ndcg@20'];pop=metrics['Pop']['ndcg@20']
                comparison.append(dict(**identity,condition=condition,base_ndcg20=base,np_gc_ndcg20=native,pop_ndcg20=pop,np_gc_minus_base_relative_percent=100*(native/base-1),pop_minus_np_gc_points=100*(pop-native)))
    write_csv(OUT/'S1_fullpool_seed_results.csv',seed_rows)
    write_csv(OUT/'S1_fullpool_inventory.csv',inventory)
    write_csv(OUT/'S1_fullpool_top50_comparison.csv',comparison)
    write_csv(OUT/'S1_fullpool_checks.csv',checks)
    summary=[];comparisons=[]
    completed=[value for path,value in results.values() if value['status']=='COMPLETE']
    for dataset in ('yelp2018','ml-1m'):
      for backbone in (*backbone_order,'ALL_FIVE_MACRO'):
        for method in ('Base','NPGC','Pop'):
            selected=[row for row in seed_rows if row['dataset']==dataset and (row['backbone'] in backbone_order if backbone=='ALL_FIVE_MACRO' else row['backbone']==backbone) and row['method']==method and row['status']=='COMPLETE']
            seed_values=[]
            for seed in seeds:
                subset=[row for row in selected if row['seed']==seed]
                if len(subset)==(5 if backbone=='ALL_FIVE_MACRO' else 1):seed_values.append({key:np.mean([row[key] for row in subset]) for key in ('ndcg20','recall20','candidate_recall50','oracle_ndcg20')})
            n=len(seed_values);row=dict(dataset=dataset,backbone=backbone,method=method,status='COMPLETE' if n==3 else f'PARTIAL(n={n})',completed_seeds=n,expected_seeds=3,unit='raw [0,1]; sample SD across3seed means, equal5backbone per seed when macro')
            for key in ('ndcg20','recall20','candidate_recall50','oracle_ndcg20'):
                values=[r[key] for r in seed_values];row[key+'_mean']=float(np.mean(values)) if n else '';row[key+'_sd']=float(np.std(values,ddof=1)) if n>=2 else ''
            summary.append(row)
        subset=[r for r in completed if r['dataset']==dataset and (r['backbone'] in backbone_order if backbone=='ALL_FIVE_MACRO' else r['backbone']==backbone)]
        for condition,field in [('full_pool','metrics'),('backbone_Top50','top50_metrics')]:
            values=[]
            for seed in seeds:
                group=[r for r in subset if r['seed']==seed]
                if len(group)==(5 if backbone=='ALL_FIVE_MACRO' else 1):values.append({method:np.mean([r[field][method]['ndcg@20'] for r in group]) for method in ('Base','NPGC','Pop')})
            n=len(values);row=dict(dataset=dataset,backbone=backbone,condition=condition,status='COMPLETE' if n==3 else f'PARTIAL(n={n})',n_seeds=n)
            if n:
                for method in ('Base','NPGC','Pop'):
                    points=[100*v[method] for v in values];row[method+'_ndcg20_points_mean']=float(np.mean(points));row[method+'_ndcg20_points_sd']=float(np.std(points,ddof=1)) if n>1 else ''
                relative=[100*(v['NPGC']/v['Base']-1) for v in values];delta=[100*(v['Pop']-v['NPGC']) for v in values]
                row['NPGC_minus_Base_relative_percent_ratio_of_means']=100*(np.mean([v['NPGC'] for v in values])/np.mean([v['Base'] for v in values])-1)
                row['NPGC_minus_Base_relative_percent_seed_mean']=float(np.mean(relative));row['NPGC_minus_Base_relative_percent_seed_sd']=float(np.std(relative,ddof=1)) if n>1 else ''
                row['Pop_minus_NPGC_points_mean']=float(np.mean(delta));row['Pop_minus_NPGC_points_sd']=float(np.std(delta,ddof=1)) if n>1 else ''
                if n>1:
                    half=float(t.ppf(.975,n-1)*np.std(delta,ddof=1)/np.sqrt(n));row['Pop_minus_NPGC_repeat95_low']=float(np.mean(delta)-half);row['Pop_minus_NPGC_repeat95_high']=float(np.mean(delta)+half)
            comparisons.append(row)
    write_csv(OUT/'S1_fullpool_summary.csv',summary)
    write_csv(OUT/'S1_fullpool_comparison_summary.csv',comparisons)
    counts={status:sum(row['status']==status for row in inventory) for status in sorted({row['status'] for row in inventory})}
    atomic_json(OUT/'S1_fullpool_status.json',dict(coordinate_counts=counts,expected_manifest_coordinates=len(coordinates),status='COMPLETE' if coordinates and counts.get('COMPLETE')==len(coordinates) else 'PARTIAL',
        units='raw seed metrics, comparisons percentage points where explicitly named',historical_metrics_imported=False,macro_backbones=list(backbone_order),macro_requires_all_five_in_each_seed=True))
    return dict(coordinate_counts=counts,summary_rows=len(summary),comparison_rows=len(comparisons))
