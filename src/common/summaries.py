"""Collect regenerated S1/S2 rows and call the original repeat-interval function."""
import csv,json
from collections import defaultdict
from pathlib import Path
from .statistics import repeat_interval
from .hashing import write_csv
METRICS=('ndcg@10','ndcg@20','recall@10','recall@20')

def summarize(root):
    root=Path(root);lookup={}
    for path in sorted(root.glob('*/**/seed_results.csv')):
        for row in csv.DictReader(path.open()):
            key=tuple(row[k] for k in ('dataset','backbone','seed','protocol','method'))
            if key in lookup and lookup[key]!=row:raise RuntimeError('Conflicting regenerated row: '+str(key))
            lookup[key]=row
    rows=list(lookup.values())
    if not rows:raise FileNotFoundError('No regenerated coordinate seed_results.csv under '+str(root))
    write_csv(root/'seed_results.csv',rows)
    groups=defaultdict(list)
    for row in rows:
        for metric in (*METRICS,'candidate_recall50','oracle_ndcg20'):
            if metric in row:groups[(row['dataset'],row['backbone'],row['protocol'],row['method'],metric)].append((int(row['seed']),float(row[metric])))
    summary=[]
    for key,values in sorted(groups.items()):
        values.sort();summary.append(dict(zip(('dataset','backbone','protocol','method','metric'),key),training_seeds=json.dumps([s for s,_ in values]),**repeat_interval([v for _,v in values])))
    write_csv(root/'method_summary.csv',summary)
    pairs=defaultdict(list)
    for row in rows:
        for name,left,right in [('Pop-Base','item_degree_ranking','base'),('NPGC-Base','original_nonparamgc','base'),('Pop-NPGC','item_degree_ranking','original_nonparamgc')]:
            if row['method']!=left:continue
            key=tuple(row[k] for k in ('dataset','backbone','seed','protocol'))+(right,)
            if key not in lookup:continue
            control=lookup[key]
            for metric in METRICS:pairs[(row['dataset'],row['backbone'],row['protocol'],name,metric)].append((int(row['seed']),float(row[metric])-float(control[metric])))
    intervals=[]
    for key,values in sorted(pairs.items()):
        values.sort();intervals.append(dict(zip(('dataset','backbone','protocol','comparison','metric'),key),training_seeds=json.dumps([s for s,_ in values]),**repeat_interval([v for _,v in values])))
    if intervals:write_csv(root/'paired_repeat_intervals.csv',intervals)
    most=[]
    for path in sorted(root.glob('*/**/mostpop.json')):most.extend(json.loads(path.read_text()))
    if most:write_csv(root/'mostpop_seed_results.csv',most)
    return dict(seed_rows=len(rows),method_summary_rows=len(summary),paired_interval_rows=len(intervals),mostpop_rows=len(most))
