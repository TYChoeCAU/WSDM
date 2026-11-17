"""A2/A5 read-only extraction of recorded scores/metrics into new Parquet files.

No model is instantiated. Original caches remain on the server and are mmap-read
where supported. Source result JSONs determine exact cache/prediction locations.
"""
from __future__ import annotations
import argparse, csv, gc, hashlib, json, os, shutil, sys, time
from collections import defaultdict
from pathlib import Path
from .w2w3_statistics import RUN, ROOT, METHODS, FIXED, RESIDUALS, METRICS, input_digest
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)
def write_csv(path,rows,columns=None):
    rows=list(rows);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=columns or list(dict.fromkeys(k for r in rows for k in r))
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    tmp.replace(path)
def array(v,dtype=None):return np.asarray(v.detach().cpu().numpy()if hasattr(v,'detach')else v,dtype=dtype)
def load(path):
    try:return torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    except RuntimeError as e:
        if 'mmap'not in str(e):raise
        return torch.load(path,map_location='cpu',weights_only=False)

def evidence_rows():
    """Parent-owned registry is the sole coordinate mapping authority."""
    registry=RUN/'evidence_index.csv'
    if not registry.is_file():raise FileNotFoundError('Parent evidence_index.csv has not been published')
    rows=[];seen={}
    for raw in csv.DictReader(registry.open()):
        if raw.get('method')not in METHODS or raw.get('regime')!='R0':continue
        if raw.get('status','COMPLETED')not in ('COMPLETED','COMPLETE','REUSED','REUSE_VERIFIED'):continue
        path=raw.get('result_path')or raw.get('source_result')or raw.get('path')
        if not path or not Path(path).is_file():continue
        if sha(path)!=raw['source_result_sha256']:raise ValueError('Registered result bytes changed')
        r=read(path)
        if r.get('status')!='COMPLETED':continue
        family=raw.get('analysis_family')or ('frozen'if 'source_result_path'in r else'matched')
        family='frozen'if family in ('frozen','frozen_transfer','frozen_model_transfer')else'matched'
        source=raw.get('source_protocol')or r.get('source_protocol')or r.get('train_protocol',r['protocol'])
        protocol=(source+'_to_F')if family=='frozen'else r['protocol']
        row=dict(dataset=r['dataset'],backbone=r['backbone'],training_seed=int(r.get('seed',r.get('training_seed'))),
            regime=raw['regime'],analysis_family=family,protocol=protocol,method=r['method'],
            result_path=str(Path(path).resolve()),source_protocol=source)
        key=tuple(row[k]for k in ('dataset','backbone','training_seed','regime','analysis_family','protocol','method'))
        if key in seen and seen[key]!=row['result_path']:raise ValueError('Ambiguous registry result for '+str(key))
        if key not in seen:rows.append(row);seen[key]=row['result_path']
    return rows

def partition_path(row):
    return RUN/'private/per_user_parts'/row['regime']/row['analysis_family']/row['dataset']/row['backbone']/f"seed_{row['training_seed']}"/row['protocol']/'metrics.parquet'

def metric_vectors(scores,ids,mask,labels,positive):
    scores=array(scores,np.float64)
    if not np.isfinite(scores[mask]).all():raise ValueError('Nonfinite candidate score')
    order=np.lexsort((ids,np.where(mask,-scores,np.inf)),axis=1)
    ranked=np.take_along_axis(labels&mask,order,axis=1)
    discount=1/np.log2(np.arange(2,22));prefix=np.r_[0,np.cumsum(discount)];out={}
    for k in (10,20):
        out[f'recall@{k}']=np.divide(ranked[:,:k].sum(1),positive,out=np.zeros(len(positive)),where=positive>0)
        denom=prefix[np.minimum(k,positive)]
        out[f'ndcg@{k}']=np.divide((ranked[:,:k]*discount[:k]).sum(1),denom,out=np.zeros(len(positive)),where=denom>0)
    return out

def extract_partition(rows):
    first=read(rows[0]['result_path']);part=partition_path(rows[0]);meta=part.with_name('extraction.json')
    result_hashes={r['method']:sha(r['result_path'])for r in rows}
    digest=input_digest(dict(results=result_hashes,version='w2w3-saved-metrics-v1'))
    if part.is_file()and meta.is_file()and read(meta).get('input_digest')==digest:return read(meta)
    part.parent.mkdir(parents=True,exist_ok=True)
    print('EXTRACT',rows[0]['regime'],rows[0]['analysis_family'],rows[0]['dataset'],rows[0]['backbone'],rows[0]['training_seed'],rows[0]['protocol'],len(rows),flush=True)
    cache=load(first['cache_path']);phase=cache['test']
    users=array(phase['user_ids'],np.int64);positive=array(phase['positive_count'],np.int64)
    ids=array(phase['candidate_ids'],np.int64);mask=array(phase['valid_mask'],bool);labels=array(phase['labels'],bool)
    if len(np.unique(users))!=len(users):raise ValueError('Repeated evaluation users need explicit slate aggregation')
    if np.any(ids[mask]<=0)or np.any(labels&~mask):raise ValueError('Invalid padding/candidate labels')
    truth_hash=[]
    for i,t in enumerate(phase['positive_items']):
        values=np.unique(array(t,np.int64))
        if len(values)!=positive[i]:raise ValueError('Full relevant-count mismatch')
        truth_hash.append(hashlib.sha256(values.tobytes()).hexdigest())
    hit=(labels&mask).sum(1);candrec=np.divide(hit,positive,out=np.zeros(len(users)),where=positive>0)
    prefix=np.r_[0,np.cumsum(1/np.log2(np.arange(2,22)))];den=prefix[np.minimum(20,positive)]
    oracle=np.divide(prefix[np.minimum(20,hit)],den,out=np.zeros(len(users)),where=den>0)
    checks=[];method_summaries=[];writer=None;tmp=part.with_suffix('.tmp.parquet')
    for row in rows:
        result=read(row['result_path']);pred=load(result['predictions_path']);family=row['analysis_family']
        source=read(result['source_result_path'])if family=='frozen'else result
        if result['candidate_digest']!=first['candidate_digest'] or pred['candidate_digest']!=phase['candidate_digest']:raise ValueError('Candidate identity mismatch')
        if not np.array_equal(array(pred['user_ids']),users):raise ValueError('Prediction user order mismatch')
        for field in ('backbone_checkpoint_sha256','A_digest','B_digest','master_split_digest','graph_digest'):
            if result[field]!=first[field]:raise ValueError('Within-partition lineage mismatch: '+field)
        if family=='frozen':
            if result['new_model_fits']!=0 or result['F_statistics_fit']or result['F_validation_selection']:raise ValueError('Frozen evaluation refits or selects on F')
            if result['source_model_identity_hash']!=source['model_identity_hash']:raise ValueError('Frozen model identity mismatch')
            if pred.get('source_model_identity_hash')!=source['model_identity_hash']:raise ValueError('Frozen saved-prediction model identity mismatch')
        else:
            if pred.get('model_identity_hash')!=result['model_identity_hash']:raise ValueError('Saved model identity mismatch')
        policies=[('identity_inclusive'if row['method']in RESIDUALS else'native','per_user_metrics','metrics','scores')]
        if row['method']in RESIDUALS:policies.append(('trained_only','trained_only_per_user_metrics','trained_only_metrics','trained_only_scores'))
        for policy,key,macrokey,scorekey in policies:
            available=pred.get(key)
            metrics={m:array(available[m],np.float64)for m in METRICS}if available is not None else metric_vectors(pred[scorekey],ids,mask,labels,positive)
            if any(len(v)!=len(users)or not np.isfinite(v).all()for v in metrics.values()):raise ValueError('Invalid per-user metrics')
            error=max(abs(float(metrics[m].mean())-result[macrokey][m])for m in METRICS)
            if error>1e-3:raise ValueError('Stored aggregate/per-user difference exceeds 1e-3')
            ident=bool(source.get('identity_selected'))and policy=='identity_inclusive'
            identity_score_error=None
            if ident:
                identity_score_error=float(np.max(np.abs(array(pred[scorekey])[mask]-array(phase['s'])[mask])))
                if identity_score_error>1e-3:raise ValueError('Identity-selected prediction differs from frozen Base')
            n=len(users);columns={k:[v]*n for k,v in dict(study='S3',regime=row['regime'],analysis_family=family,
                dataset=row['dataset'],backbone=row['backbone'],training_seed=row['training_seed'],protocol=row['protocol'],
                source_protocol=row['source_protocol'],method=row['method'],selection_policy=policy,identity_selected=ident,
                candidate_digest=result['candidate_digest'],source_result_sha256=result_hashes[row['method']]).items()}
            columns.update(user_id=users,full_relevant_count=positive,full_truth_user_digest=truth_hash,
                candidate_relevant_count=hit,candidate_recall50=candrec,oracle_ndcg20=oracle,
                **{m.replace('@',''):metrics[m]for m in METRICS})
            table=pa.table(columns)
            if writer is None:writer=pq.ParquetWriter(tmp,table.schema,compression='zstd')
            writer.write_table(table)
            checks.append(dict(**row,selection_policy=policy,status='PASS',n_users=n,maximum_aggregate_error=error,
                identity_score_error=identity_score_error,
                saved_per_user_present=available is not None,model_inference=False,result_sha256=result_hashes[row['method']]))
            method_summaries.append(dict(**row,selection_policy=policy,identity_selected=ident,**{m:float(metrics[m].mean())for m in METRICS},
                candidate_recall50=float(candrec.mean()),oracle_ndcg20=float(oracle.mean()),n_users=n))
        del pred
    writer.close();tmp.replace(part)
    metadata=dict(input_digest=digest,partition_path=str(part),coordinate={k:rows[0][k]for k in ('regime','analysis_family','dataset','backbone','training_seed','protocol')},
        checks=checks,method_summaries=method_summaries,n_users=len(users),
        lineage={k:first[k]for k in ('backbone_checkpoint_sha256','A_digest','B_digest','master_split_digest','graph_digest')},
        cache_path=first['cache_path'],cache_copied=False,training=False,model_inference=False)
    atomic_json(meta,metadata);del cache,phase;gc.collect();return metadata

def merge_parts():
    parts=sorted((RUN/'private/per_user_parts').glob('*/*/*/*/seed_*/*/metrics.parquet'))
    # A Parquet directory is readable through the same pandas/Arrow APIs. Each
    # leaf is a hard link to the immutable extraction partition, so rebuilding
    # the public analysis view never duplicates its multi-gigabyte payload.
    target=RUN/'per_user_metrics.parquet';tmp=RUN/f'.per_user_metrics.dataset.{os.getpid()}.tmp'
    backup=RUN/f'.per_user_metrics.dataset.{os.getpid()}.previous'
    if tmp.exists():shutil.rmtree(tmp)
    tmp.mkdir();total=0;checks=[];summaries=[];logical_bytes=0;schema=None
    for index,path in enumerate(parts):
        meta=read(path.with_name('extraction.json'));checks.extend(meta['checks']);summaries.extend(meta['method_summaries'])
        parquet=pq.ParquetFile(path)
        if schema is None:schema=parquet.schema_arrow
        elif not parquet.schema_arrow.equals(schema):raise ValueError('Parquet dataset schema mismatch')
        leaf=tmp/f'part-{index:04d}.parquet';os.link(path,leaf)
        if os.stat(path).st_ino!=os.stat(leaf).st_ino:raise ValueError('Dataset leaf is not the same inode')
        total+=parquet.metadata.num_rows;logical_bytes+=path.stat().st_size
    if parts:
        import pyarrow.dataset as ds
        if ds.dataset(tmp,format='parquet').count_rows()!=total:raise ValueError('Parquet dataset row count mismatch')
        if target.exists():target.rename(backup)
        try:tmp.rename(target)
        except Exception:
            if backup.exists():backup.rename(target)
            raise
        if backup.is_dir():shutil.rmtree(backup)
        elif backup.exists():backup.unlink()
    else:tmp.rmdir()
    write_csv(RUN/'audit/saved_prediction_inventory.csv',checks);write_csv(RUN/'extracted_seed_results.csv',summaries)
    atomic_json(RUN/'audit/extraction_status.json',dict(partitions=len(parts),per_user_rows=total,method_policy_checks=len(checks),
        source_models_inferred=False,original_caches_copied=False,DNR_excluded=True,
        parquet_storage='directory_dataset_with_hardlinked_partition_leaves',extra_payload_bytes=0))
    atomic_json(RUN/'per_user_metrics_format.json',dict(path=str(target),format='parquet_directory_dataset',
        pandas_api='pandas.read_parquet(path, engine="pyarrow", columns=..., filters=...)',
        arrow_api='pyarrow.parquet.read_table(path, columns=..., filters=...)',
        source_partitions='private/per_user_parts',n_leaves=len(parts),n_rows=total,
        logical_bytes=logical_bytes,extra_payload_bytes=0,hardlink_inode_checks='PASS',
        note='The .parquet path is a directory. Its leaves share file bytes with the source partitions. Original caches are not copied.'))

def audit_all_users():
    """A2 covers every extracted original/new regime, independent of analysis readiness."""
    groups=defaultdict(list);outputs=[]
    for path in sorted((RUN/'private/per_user_parts').glob('*/*/*/*/seed_*/*/metrics.parquet')):
        meta=read(path.with_name('extraction.json'));c=meta['coordinate']
        groups[(c['dataset'],c['backbone'],c['training_seed'])].append((path,meta))
    for key,parts in groups.items():
        views={}
        for path,meta in parts:
            # Each write_table is a complete method-policy vector. Its first row
            # group contains the complete test-user vector for current datasets.
            parquet=pq.ParquetFile(path);frame=parquet.read_row_group(0,columns=['user_id','full_relevant_count','full_truth_user_digest']).to_pandas()
            if len(frame)!=meta['n_users']:raise ValueError('First row group does not cover the user vector')
            frame=frame.sort_values('user_id',kind='stable');c=meta['coordinate'];name='/'.join((c['regime'],c['analysis_family'],c['protocol']))
            views[name]=frame
        for a,b in __import__('itertools').combinations(sorted(views),2):
            x,y=views[a],views[b];u=np.intersect1d(x['user_id'],y['user_id']);xx=x[x['user_id'].isin(u)];yy=y[y['user_id'].isin(u)]
            valid=np.array_equal(xx['full_relevant_count'],yy['full_relevant_count'])and np.array_equal(xx['full_truth_user_digest'],yy['full_truth_user_digest'])
            outputs.append(dict(dataset=key[0],backbone=key[1],training_seed=key[2],source=a,target=b,
                source_users=len(x),target_users=len(y),common_users=len(u),source_excluded=len(x)-len(u),target_excluded=len(y)-len(u),
                identical_user_sets=len(u)==len(x)==len(y),full_truth_equal_on_common_users=valid,
                status='PASS'if valid else'REVIEW_REQUIRED_FULL_TRUTH_MISMATCH'))
    write_csv(RUN/'audit/A2_user_set_checks.csv',outputs)
    atomic_json(RUN/'audit/A2_status.json',dict(coordinate_count=len(groups),pair_checks=len(outputs),
        full_truth_mismatches=sum(r['status']!='PASS'for r in outputs),
        unequal_user_pairs=sum(not r['identical_user_sets']for r in outputs),
        no_model_inference=True,no_training=True))

def extract_existing(a5=False,merge_only=False):
    torch.set_num_threads(min(4,int(os.environ.get('OMP_NUM_THREADS','2'))));pa.set_cpu_count(2)
    if not(RUN/'analysis_plan_w2w3.json').is_file():raise RuntimeError('Analysis plan must be fixed before result analysis')
    if not merge_only:
        groups=defaultdict(list)
        for row in evidence_rows():
            if a5 and not(row['regime']=='R0'and row['analysis_family']=='matched'and row['training_seed']<=2022 and row['method']in FIXED):continue
            groups[tuple(row[k]for k in ('regime','analysis_family','dataset','backbone','training_seed','protocol'))].append(row)
        for _,rows in sorted(groups.items()):extract_partition(rows)
    merge_parts();audit_all_users()
