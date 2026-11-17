"""Small CSV-only computations shared by table generation and numerical checks."""
from pathlib import Path
from statistics import mean,stdev
import csv
from scipy.stats import t
from transfer_counts import load,classify,summarize,CELLS,PAIRS,fmt
DATASETS=('yelp2018','amazon-books','ml-1m')
BACKBONES=('NFM','xDeepFM','DCNV2','AutoInt','EulerNet','QNN_alpha')
DISPLAY={'yelp2018':'Yelp2018','amazon-books':'Amazon-Books','ml-1m':'ML-1M','DCNV2':'DCNv2','MF_BPR':'MF-BPR','QNN_alpha':r'QNN-$\alpha$'}
def read(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def choose(rows,**kw):return [r for r in rows if all(str(r[k])==str(v) for k,v in kw.items())]
def one(rows,**kw):
 rr=choose(rows,**kw)
 if len(rr)!=1:raise ValueError(f'Expected one row for {kw}, got {len(rr)}')
 return rr[0]
def stats(values):
 vv=list(values); m=mean(vv);s=stdev(vv);h=float(t.ppf(.975,len(vv)-1))*s/len(vv)**.5
 return m,s,m-h,m+h
def cell(kind,*values,dp=2,signed=False):return {'kind':kind,'values':list(values),'dp':dp,'signed':signed}
def metric_cell(values,dp=2,signed=False):return cell('score',*stats(values)[:2],dp=dp,signed=signed)
def build(results):
 results=Path(results);tables={}
 # Presentation metadata follows the submitted manuscript. Numeric evidence stays unchanged.
 studies=[]
 for row in read(results/'studies.csv'):
  row=dict(row)
  if row['field']=='Training':row['S3']=r'BCE, BPR, $2\times2$'
  if row['field']=='Re-rankers':
   studies.extend([{'field':'Rules','S1':'Pop, NPGC','S2':'Pop','S3':'Pop, NPGC'},
                   {'field':'Learned','S1':'--','S2':'--','S3':r'PRM+PL, Res($\cdot$)'}])
  else:studies.append(row)
 tables[1]=[{'label':r['field'],'cells':[cell('text',r[x]) for x in ('S1','S2','S3')]} for r in studies]
 b=read(results/'s1/benchmark_seed_metrics.csv');cr=read(results/'s1/s1_candidate_recall.csv');rows=[]
 for bb in BACKBONES:
  cells=[]
  for ds in DATASETS:
   for method in ('base','original_nonparamgc','item_degree_ranking'):
    cells.append(metric_cell(100*float(r['ndcg@20']) for r in choose(b,dataset=ds,backbone=bb,method=method)))
   for field in ('candidate_recall50_macro','oracle_ndcg20'):
    cells.append(cell('plain',mean(100*float(r[field]) for r in choose(cr,dataset=ds,backbone=bb))))
  rows.append({'label':DISPLAY.get(bb,bb),'cells':cells})
 tables[2]=rows
 raw=read(results/'s1/fresh_seed_metrics.csv')+read(results/'s2/seed_metrics.csv');rows=[]
 for study,ds,bb in [('S1',d,b) for d in DATASETS for b in ('NFM','DCNV2')]+[('S2',d,'MF_BPR') for d in ('yelp2018','ml-1m')]:
  diffs={};base={}
  for p in ('U','W','F'):
   rr=choose(raw,study=study,dataset=ds,backbone=bb,protocol=p)
   base[p]=[100*float(one(rr,seed=s,method='base')['ndcg@20']) for s in (2020,2021,2022)]
   pop=[100*float(one(rr,seed=s,method='item_degree_ranking')['ndcg@20']) for s in (2020,2021,2022)]
   diffs[p]=stats(x-y for x,y in zip(pop,base[p]))
  cells=[cell('score',*diffs['U'][:2],dp=3,signed=True),cell('score',*diffs['W'][:2],dp=3,signed=True),cell('ci',*diffs['W'][2:],dp=3,signed=True),cell('plain',mean(base['F']),dp=3),cell('score',*diffs['F'][:2],dp=3,signed=True),cell('ci',*diffs['F'][2:],dp=3,signed=True)]
  rows.append({'label':' & '.join((study,DISPLAY[ds],DISPLAY.get(bb,bb))),'cells':cells})
 tables[3]=rows
 summ=read(results/'s3/homogeneous_method_summary.csv');summ=[r for r in summ if r['analysis_family']=='matched' and r['selection_policy'] in ('native','identity_inclusive')];rows=[]
 for ds,bb in [(d,b) for d in ('yelp2018','ml-1m') for b in ('NFM','MF_BPR')]:
  for p in ('U','W','F'):
   cells=[]
   for method,metric in [(m,'ndcg@20') for m in ('base','item_degree_ranking','original_nonparamgc','prm_principled','P','H')]+[('base','candidate_recall50'),('base','oracle_ndcg20')]:
    r=one(summ,dataset=ds,backbone=bb,protocol=p,method=method,metric=metric)
    cells.append(cell('score',float(r['mean_x100']),float(r['sample_sd_x100'])))
   rows.append({'label':' & '.join((DISPLAY[ds],DISPLAY[bb] if bb in DISPLAY else bb,p)),'cells':cells})
 tables[4]=rows
 counts=load(str(results/'s3'),'ndcg@20','identity_inclusive','0.0025');_,cc,_=summarize(counts,lambda r:classify(r,.0025,.0025))
 pair_labels={'U-W':'U--W','U-F':'U--F','W-F':'W--F','U-U_to_F':r'U$\to$F','W-W_to_F':r'W$\to$F'}
 rows=[{'label':pair_labels[pair],'cells':[cell('count',fmt(cc[(pair,c)])) for c in CELLS]} for pair in PAIRS]
 tables[5]=rows
 return tables

def render(c):
 kind=c['kind'];v=c['values'];dp=c['dp'];sig='+' if c['signed'] else ''
 if kind in ('text','count'):return v[0]
 if kind=='plain':return f'{v[0]:.{dp}f}'
 if kind=='score':return r'\Score{'+f'{v[0]:{sig}.{dp}f}'+r'}{'+f'{v[1]:.{dp}f}'+'}'
 if kind=='ci':
  ss=[f'{x:{sig}.{4 if abs(x)<.0005 and x!=0 else dp}f}' for x in v]
  return '$['+r',\,'.join(ss)+']$'
 raise ValueError(kind)
def numeric_tokens(row):
 import re
 return [x for c in row['cells'] for x in (re.findall(r'[-+]?\d+(?:\.\d+)?',render(c)) if c['kind']!='text' else [])]
