"""Recalculate submitted manuscript numbers with display-rounding tolerances; never change evidence."""
from pathlib import Path
from statistics import mean
from collections import defaultdict
import argparse,json,math,sys
from paper_data import read,choose,one,build,numeric_tokens
from transfer_counts import load,classify,summarize,headroom,SRC

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',type=Path,default=Path('results'));ap.add_argument('--out',type=Path,default=Path('outputs/PAPER_NUMBER_CHECK.md'));args=ap.parse_args();r=args.results;checks=[]
 manuscript_expected=json.loads((Path(__file__).resolve().parents[1]/'configs/analysis_expected.json').read_text())
 def check(name,actual,expected,note=''):
  ex=str(expected);digits=len(ex.split('.')[1]) if '.' in ex else 0;tol=.5*10**(-digits) if digits else 0
  ok=math.isfinite(float(actual)) and abs(float(actual)-float(expected))<=tol+1e-12
  checks.append(dict(item=name,expected=ex,actual=f'{actual:.12g}' if isinstance(actual,(float,int)) else str(actual),status='PASS' if ok else 'MISMATCH',note=note if ok else note or 'Saved evidence differs at manuscript precision; neither evidence nor expectations were changed.'))
 b=read(r/'s1/benchmark_seed_metrics.csv');groups=defaultdict(dict)
 for x in b:groups[x['dataset'],x['backbone'],x['seed']][x['method']]=x
 check('S1 benchmark coordinates',len(groups),54)
 for m in ('ndcg@20','recall@20'):check('S1 Pop > Base and NPGC: '+m,sum(float(g['item_degree_ranking'][m])>max(float(g[z][m]) for z in ('base','original_nonparamgc')) for g in groups.values()),54)
 gains=defaultdict(list);ng=defaultdict(list);relative=defaultdict(list)
 for (ds,bb,seed),g in groups.items():
  a=float(g['base']['ndcg@20']);n=float(g['original_nonparamgc']['ndcg@20']);p=float(g['item_degree_ranking']['ndcg@20']);gains[ds,bb].append(100*(p-a));ng[ds,bb].append(100*(n-a));relative[ds,bb].append((a,n))
 check('S1 backbone Pop-Base minimum',min(map(mean,gains.values())),'7.5');check('S1 backbone Pop-Base maximum',max(map(mean,gains.values())),'14.0');check('S1 backbone NPGC-Base maximum',max(map(mean,ng.values())),'1.59')
 rel=[100*(mean(y for x,y in v)/mean(x for x,y in v)-1) for v in relative.values()]
 check('S1 NPGC relative gain minimum (%)',min(rel),'3.7');check('S1 NPGC relative gain maximum (%)',max(rel),manuscript_expected['body']['s1_npgc_relative_gain_max_percent'])
 cap=read(r/'s1/s1_oracle_headroom.csv')
 for method,lo,hi in [('item_degree_ranking','28.2','35.7'),('original_nonparamgc','1.2','3.4')]:
  vals=[mean(100*float(x['capture']) for x in choose(cap,dataset=ds,method=method)) for ds in ('yelp2018','amazon-books','ml-1m')];check('S1 oracle gap recovery minimum '+method,min(vals),lo);check('S1 oracle gap recovery maximum '+method,max(vals),hi)
 mp=read(r/'checks/D_mostpop_summary.csv')
 for ds,p,ex in [('yelp2018','F','1.00'),('amazon-books','F','0.66'),('ml-1m','F','12.8'),('yelp2018','benchmark_U','27.9'),('ml-1m','benchmark_U','30.8')]:
  x=one(mp,Study='S1',Dataset=ds,Protocol=p);check(f'MostPop {ds} {p}',float(x['NDCG@20'].split()[0]),ex)
 ri=read(r/'s3/repeat_intervals.csv')
 for bb,ex in [('NFM','13.2'),('MF_BPR','-18.7')]:
  x=one(ri,dataset='yelp2018',backbone=bb,selection_policy='identity_inclusive',metric='ndcg@20',analysis_family='matched',comparison='U',contrast='Pop-Base');check('S3 Yelp2018 U Pop-Base '+bb,float(x['mean_x100']),ex)
 sp=read(r/'checks/list_spearman_seed_results.csv')
 for bb,ex in [('NFM','-0.15'),('MF_BPR','0.30')]:check('S3 Yelp2018 U Spearman '+bb,mean(float(x['mean']) for x in choose(sp,study='S3',dataset='yelp2018',backbone=bb,protocol='U',regime='R0')),ex)
 sg=defaultdict(list)
 for x in choose(sp,study='S1_historical'):sg[x['dataset'],x['backbone']].append(float(x['mean']))
 check('S1 within-list Spearman minimum',min(map(mean,sg.values())),'-0.13');check('S1 within-list Spearman maximum',max(map(mean,sg.values())),'0.16')
 run=str(r/'s3');rows=load(run,'ndcg@20','identity_inclusive','0.0025');counts,_,ft=summarize(rows,lambda x:classify(x,.0025,.0025))
 check('Transfer unique comparisons',len(rows),136)
 for label,ex in zip('TREVU',(56,1,28,34,17)):check('Transfer outcome '+label,counts[label],ex)
 for fam,num in [('matched',96),('frozen',40)]:check('Unique '+fam+' comparisons',sum(x['analysis_family']==fam for x in rows),num)
 check('Matched NDCG reversals',sum(x['analysis_family']=='matched' and classify(x,.0025,.0025)=='R' for x in rows),0)
 for ds,num in [('ml-1m',37),('yelp2018',0)]:check('F-target '+ds+' count',sum(ft[ds].values()),52);check('F-target '+ds+' T',ft[ds]['T'],num)
 nofloor=[x for x in rows if (x['dataset'],x['backbone'])!=('yelp2018','NFM')]
 check('Outside Yelp NFM comparisons',len(nofloor),102);check('Outside Yelp NFM T',sum(classify(x,.0025,.0025)=='T' for x in nofloor),55);check('Vanish on Yelp2018',sum(classify(x,.0025,.0025)=='V' and x['dataset']=='yelp2018' for x in rows),34)
 for d in ('0.001','0.005'):
  xx=load(run,'ndcg@20','identity_inclusive',d);check('Reversals at delta='+d,sum(classify(x,float(d),float(d))=='R' for x in xx),1)
  if d=='0.001':check('Yelp F-target transfer at delta=0.1 points',sum(x['dataset']=='yelp2018' and x['comparison']!='U-W' and classify(x,.001,.001)=='T' for x in xx),5)
 tr=load(run,'ndcg@20','trained_only','0.0025');check('Trained-only reversals',sum(classify(x,.0025,.0025)=='R' for x in tr),1)
 h=headroom(run);check('Headroom 1% reversals',sum(classify(x,.01*h(x['dataset'],x['backbone'],SRC[x['comparison']][0]),.01*h(x['dataset'],x['backbone'],SRC[x['comparison']][1]))=='R' for x in rows),3)
 check('All five user-bootstrap decisions agree',sum(x['user_seed_agrees_with_D1']=='5' for x in rows),117,'Agreement with original D1 classes; V is a later subdivision.')
 rc=load(run,'recall@20','identity_inclusive','0.0025');check('Matched Recall reversals',sum(x['analysis_family']=='matched' and classify(x,.0025,.0025)=='R' for x in rc),4)
 tost=[x for x in read(r/'s3/tost_holm_primary.csv') if x['registered_primary_family']=='True'];check('TOST primary hypotheses',len(tost),16);check('TOST Holm equivalents',sum(x['equivalent_holm']=='True' for x in tost),1)
 changes=[abs(float(x['target_effect_mean_raw'])-float(x['source_effect_mean_raw']))*100 for x in rows if x['comparison']=='U-F' and x['contrast']=='Pop-Base'];check('U-to-F Pop-Base change minimum',min(changes),'7.05');check('U-to-F Pop-Base change maximum',max(changes),'18.75')
 reg=read(r/'checks/C_regret_summary.csv')
 for bb,ex in [('NFM','0.033'),('MF_BPR','0.068')]:check('Yelp matched W-F REF4 regret '+bb,float(one(reg,dataset='yelp2018',backbone=bb,analysis_family='matched',method_set='REF4')['mean_x100']),ex)
 paired=read(r/'checks/C_paired_repeat_ci.csv')
 for con,fit,ev,ex in [('Res(p)-Base','W','W','1.64'),('Res(p)-Base','W','F','-2.06'),('Res(p)-Base','F','F','0.03'),('Res(p)-Pop','W','W','28.70'),('Res(p)-Pop','W','F','-2.09')]:
  x=one(paired,dataset='yelp2018',backbone='MF_BPR',contrast=con,fit_protocol=fit,eval_protocol=ev);check(f'Yelp MF-BPR {con} {fit}->{ev}',float(x['mean_x100']),ex)
  if con=='Res(p)-Pop' and ev=='F':check('Yelp MF-BPR frozen Res(p)-Pop CI lower',float(x['ci95_low_x100']),'-2.21');check('Yelp MF-BPR frozen Res(p)-Pop CI upper',float(x['ci95_high_x100']),'-1.98')
 raw=read(r/'s3/extracted_seed_results.csv')
 for bb,num in [('MF_BPR',137),('NFM',22)]:
  xx=[x for x in raw if x['backbone']==bb and x['analysis_family']=='matched' and x['selection_policy']=='identity_inclusive' and x['method'] in ('S','P','R','PR','H','RH','PH','PRH')];check('Residual fit total '+bb,len(xx),240);check('Identity selected '+bb,sum(x['identity_selected']=='True' for x in xx),num)
 eq=read(r/'checks/A_npgc_equivalence.csv');check('NPGC official equivalence audited cells',len(eq),2);err=max(float(x['max_abs_score_diff']) for x in eq)
 checks.append(dict(item='NPGC official maximum absolute score difference',expected='<= 4e-7',actual=str(err),status='PASS' if err<=4e-7 else 'MISMATCH',note='Saved two-cell upstream audit.'))
 expected=manuscript_expected['tables'];tables=build(r)
 for n,tab in expected.items():
  actual=tables[int(n)];check(f'Table {n} rows',len(actual),len(tab))
  for i,(er,ar) in enumerate(zip(tab,actual),1):
   et=er['expected_numeric_tokens'];vals=[]
   for c in ar['cells']:vals.extend(map(int,c['values'][0].split('/')) if c['kind']=='count' else c['values'] if c['kind']!='text' else [])
   check(f'Table {n} row {i} numeric cell count',len(vals),len(et))
   for j,(av,ev) in enumerate(zip(vals,et),1):check(f'Table {n} row {i} value {j}',av,ev)
 bad=[x for x in checks if x['status']!='PASS'];args.out.parent.mkdir(parents=True,exist_ok=True)
 lines=['# Submitted manuscript number check','',f'{len(checks)-len(bad)}/{len(checks)} checks passed; {len(bad)} mismatches.','','Only released scalar evidence is read. No training, inference, setting selection, or evidence alteration is performed. Numeric tolerances are half a unit at manuscript display precision; integer counts must agree exactly. Table 1 is design metadata. Every numeric cell of Tables 2–5 is compared with the submitted manuscript.','','Historical limit: saved Yelp2018 S1 F diagnostics say historical candidate IDs were not saved; reconstructed candidate identity is not promoted to historical identity.','']
 if bad:
  lines+=['## Mismatches','','| Item | Expected | Recomputed | Reason / scope |','|---|---:|---:|---|']+[f"| {x['item']} | {x['expected']} | {x['actual']} | {x['note']} |" for x in bad]
 lines+=['','## All checks','','| Item | Expected | Recomputed | Status |','|---|---:|---:|---|']+[f"| {x['item']} | {x['expected']} | {x['actual']} | {x['status']} |" for x in checks]
 args.out.write_text('\n'.join(lines)+'\n');args.out.with_suffix('.json').write_text(json.dumps(checks,indent=2)+'\n')
 for x in checks:
  if not x['item'].startswith('Table '):print(f"{x['status']:8s} {x['item']}: {x['actual']} (expected {x['expected']})")
 print(f'{len(checks)-len(bad)}/{len(checks)} passed. Report: {args.out}');return 1 if bad else 0
if __name__=='__main__':sys.exit(main())
