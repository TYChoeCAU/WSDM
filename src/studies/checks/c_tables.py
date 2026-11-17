"""Stage C1/C3: read-only extraction of locked R0 results; writes only review OUT."""
from pathlib import Path
import hashlib, json, sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import t

OUT = PRIOR = None
METHODS = ('base','item_degree_ranking','original_nonparamgc','prm_principled','S','P','R','PR','H','RH','PH','PRH')
SETS = {'REF4':METHODS[:4], 'ALL12':METHODS}
NAMES = {'base':'Base','item_degree_ranking':'Pop','original_nonparamgc':'NPGC','prm_principled':'PRM+PL','P':'Res(p)'}
EVIDENCE = EXTRACT = None
RESULTS, AUDIT = {}, []
def configure(s3_run, out):
    global PRIOR, OUT, EVIDENCE, EXTRACT, RESULTS, AUDIT
    PRIOR, OUT = Path(s3_run), Path(out)
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'audit').mkdir(exist_ok=True)
    EVIDENCE = pd.read_csv(PRIOR/'evidence_index.csv')
    EVIDENCE['evidence_csv_line'] = np.arange(2,len(EVIDENCE)+2)
    EVIDENCE = EVIDENCE[EVIDENCE.regime.eq('R0')]
    EXTRACT = pd.read_csv(PRIOR/'extracted_seed_results.csv')
    RESULTS, AUDIT = {}, []

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def result(ds,bb,seed,protocol,method,family='matched'):
    if family=='frozen' and method in METHODS[:3]:
        return result(ds,bb,seed,'F',method,'matched')
    key=(ds,bb,seed,protocol,method,family)
    if key in RESULTS:return RESULTS[key]
    lookup_protocol='F' if family=='frozen' else protocol
    r=EVIDENCE[(EVIDENCE.dataset==ds)&(EVIDENCE.backbone==bb)&(EVIDENCE.training_seed==seed)&EVIDENCE.protocol.isin([lookup_protocol,protocol])&(EVIDENCE.method==method)&(EVIDENCE.analysis_family==family)]
    if family=='frozen':r=r[r.source_protocol.eq(protocol.split('_')[0])]
    assert len(r)==1,(key,len(r))
    r=r.iloc[0]; p=Path(r.result_path); z=json.loads(p.read_text())
    assert sha(p)==r.source_result_sha256,(key,'result hash mismatch')
    ex=EXTRACT[(EXTRACT.regime=='R0')&(EXTRACT.dataset==ds)&(EXTRACT.backbone==bb)&(EXTRACT.training_seed==seed)&(EXTRACT.protocol==protocol)&(EXTRACT.method==method)&(EXTRACT.analysis_family==family)&EXTRACT.selection_policy.isin(['native','identity_inclusive'])]
    assert len(ex)==1
    delta=abs(float(ex.iloc[0]['ndcg@20'])-z['metrics']['ndcg@20'])
    assert delta<=1e-3
    z['_path']=str(p);z['_line']=int(r.evidence_csv_line)
    RESULTS[key]=z
    AUDIT.append(dict(dataset=ds,backbone=bb,training_seed=seed,protocol=protocol,method=method,analysis_family=family,result_path=str(p),result_sha256=sha(p),evidence_csv_line=int(r.evidence_csv_line),extracted_ndcg_difference=delta,status='PASS'))
    return z

def stats(values):
    a=np.asarray(values,dtype=float); assert len(a)==5
    m=float(a.mean()); sd=float(a.std(ddof=1)); h=float(t.ppf(.975,4)*sd/np.sqrt(5))
    return dict(n_seeds=5,seeds='2020;2021;2022;2023;2024',mean_x100=m,sample_sd_x100=sd,ci95_low_x100=m-h,ci95_high_x100=m+h,df=4,t_critical=float(t.ppf(.975,4)))

def main():
    native=[];paired=[];ident=[];wide=[]
    modes=[('W','W','matched'),('W','F','frozen'),('F','F','matched'),('U','U','matched'),('U','F','frozen')]
    for ds in ('yelp2018','ml-1m'):
      for bb in ('MF_BPR','NFM'):
       for seed in range(2020,2025):
        basekey=dict(dataset=ds,backbone=bb,training_seed=seed,primary_focus=ds=='yelp2018' and bb=='MF_BPR',regime='R0')
        w=dict(basekey)
        for fit,ev,fam in modes:
            proto=fit+'_to_F' if fam=='frozen' else ev
            z=result(ds,bb,seed,proto,'P',fam);v=z['metrics']['ndcg@20']*100
            label=f'Res(p) {fit}->{ev} {fam}'
            native.append(dict(**basekey,method='Res(p)',fit_protocol=fit,eval_protocol=ev,analysis_family=fam,condition=label,ndcg20_x100=v,identity_selected=z['identity_selected'],result_path=z['_path'],evidence_csv_line=z['_line']))
            w[label]=v;w[f'identity_fit_{fit}']=z['identity_selected']
            for control in ('item_degree_ranking','base'):
                c=result(ds,bb,seed,ev,control);cv=c['metrics']['ndcg@20']*100
                paired.append(dict(**basekey,fit_protocol=fit,eval_protocol=ev,analysis_family=fam,contrast='Res(p)-'+NAMES[control],res_p_x100=v,control_x100=cv,difference_x100=v-cv,identity_selected=z['identity_selected'],res_p_result_path=z['_path'],control_result_path=c['_path']))
        for fit in ('U','W','F'):
            z=result(ds,bb,seed,fit,'P')
            ident.append(dict(**basekey,fit_protocol=fit,identity_selected=z['identity_selected'],selected_epoch=z['selected_epoch'],checkpoint_path=z['checkpoint_path'],checkpoint_sha256=z['checkpoint_sha256'],result_path=z['_path']))
            for m in ('item_degree_ranking','base'):
                z=result(ds,bb,seed,fit,m);v=z['metrics']['ndcg@20']*100;label=f'{NAMES[m]} {fit}'
                native.append(dict(**basekey,method=NAMES[m],fit_protocol='none',eval_protocol=fit,analysis_family='fixed',condition=label,ndcg20_x100=v,identity_selected='',result_path=z['_path'],evidence_csv_line=z['_line']));w[label]=v
        wide.append(w)
    n=pd.DataFrame(native);p=pd.DataFrame(paired)
    n.to_csv(OUT/'C_frozen_resp_seed_long.csv',index=False)
    pd.DataFrame(wide).to_csv(OUT/'C_frozen_resp_seed.csv',index=False)
    p.to_csv(OUT/'C_paired_seed_differences.csv',index=False)
    pd.DataFrame(ident).to_csv(OUT/'C_identity_selection.csv',index=False)
    summaries=[]
    group=['dataset','backbone','primary_focus','method','fit_protocol','eval_protocol','analysis_family','condition']
    for key,g in n.groupby(group,sort=False):
        summaries.append(dict(zip(group,key),**stats(g.ndcg20_x100),identity_count=int(g.identity_selected.eq(True).sum()) if key[3]=='Res(p)' else '',identity_seeds=';'.join(str(x) for x in g.loc[g.identity_selected.eq(True),'training_seed'])))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'C_frozen_resp_table.csv',index=False)
    group=['dataset','backbone','primary_focus','fit_protocol','eval_protocol','analysis_family','contrast']
    ps=[]
    for key,g in p.groupby(group,sort=False):ps.append(dict(zip(group,key),**stats(g.difference_x100)))
    pd.DataFrame(ps).to_csv(OUT/'C_paired_repeat_ci.csv',index=False)
    tex=[r'\begin{tabular}{llrrrrr}',r'\toprule',r'Dataset & Backbone & $W\!\to\!W$ & $W\!\to\!F$ & $F\!\to\!F$ & $U\!\to\!U$ & $U\!\to\!F$ \\',r'\midrule']
    for (ds,bb),g in summary[summary.method=='Res(p)'].groupby(['dataset','backbone'],sort=False):
        cells=[]
        for fit,ev,fam in modes:
            x=g[(g.fit_protocol==fit)&(g.eval_protocol==ev)&(g.analysis_family==fam)].iloc[0]
            cells.append(f'${x.mean_x100:.3f} \\pm {x.sample_sd_x100:.3f}$')
        tex.append(('Yelp2018' if ds=='yelp2018' else 'ML-1M')+' & '+bb.replace('_','-')+' & '+' & '.join(cells)+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'% NDCG@20 x100, mean +/- sample SD over five R0 training seeds 2020--2024.',r'% U/W->F use frozen source-selected checkpoints. F->F is protocol-matched training.',r'% Yelp2018 MF-BPR is the primary diagnostic cell; all four cells retained.']
    (OUT/'C_frozen_resp_table.tex').write_text('\n'.join(tex)+'\n')

    original=pd.read_csv(PRIOR/'selection_regret.csv');original['source_csv_line']=np.arange(2,len(original)+2)
    select=original[(original.regime=='R0')&(original.dataset=='yelp2018')&(original.selection_policy=='identity_inclusive')&(((original.analysis_family=='frozen')&(original.comparison=='W-W_to_F')&(original.backbone=='MF_BPR')&original.method_set.isin(['ALL12','REF4']))|((original.analysis_family=='matched')&(original.comparison=='W-F')&(original.method_set=='REF4')))]
    details=[];allmethods=[]
    for _,r in select.iterrows():
        ds,bb,seed=r.dataset,r.backbone,int(r.training_seed);methods=SETS[r.method_set]
        source=[result(ds,bb,seed,'W',m) for m in methods]
        target=[result(ds,bb,seed,'W_to_F' if r.analysis_family=='frozen' else 'F',m,r.analysis_family) for m in methods]
        xs=np.array([z['metrics']['ndcg@20'] for z in source]);ys=np.array([z['metrics']['ndcg@20'] for z in target]);i=int(xs.argmax());j=int(ys.argmax());regret=float(ys[j]-ys[i])
        assert methods[i]==r.source_selected_method and methods[j]==r.target_selected_method
        assert abs(regret-r.regret_raw)<=1e-3
        details.append(dict(dataset=ds,backbone=bb,training_seed=seed,analysis_family=r.analysis_family,comparison=r.comparison,method_set=r.method_set,source_selected_method=methods[i],target_selected_method=methods[j],source_winner_W_x100=xs[i]*100,target_at_source_winner_F_x100=ys[i]*100,target_optimal_F_x100=ys[j]*100,regret_x100=regret*100,saved_regret_x100=r.regret_raw*100,regret_difference_raw=regret-r.regret_raw,source_result_path=source[i]['_path'],target_at_source_result_path=target[i]['_path'],target_optimal_result_path=target[j]['_path'],selection_regret_csv_line=int(r.source_csv_line),status='PASS'))
        for k,m in enumerate(methods):allmethods.append(dict(dataset=ds,backbone=bb,training_seed=seed,analysis_family=r.analysis_family,method_set=r.method_set,method=m,registered_order=k,W_x100=xs[k]*100,F_x100=ys[k]*100,source_selected=k==i,target_selected=k==j,source_result_path=source[k]['_path'],target_result_path=target[k]['_path']))
    detail=pd.DataFrame(details);detail.to_csv(OUT/'C_regret_detail.csv',index=False)
    pd.DataFrame(allmethods).to_csv(OUT/'C_regret_all_method_scores.csv',index=False)
    rr=[]
    for key,g in detail.groupby(['dataset','backbone','analysis_family','comparison','method_set'],sort=False):rr.append(dict(zip(['dataset','backbone','analysis_family','comparison','method_set'],key),**stats(g.regret_x100)))
    pd.DataFrame(rr).to_csv(OUT/'C_regret_summary.csv',index=False)
    pd.DataFrame(AUDIT).to_csv(OUT/'audit/c_result_audit.csv',index=False)
    (OUT/'audit/c_tables_status.json').write_text(json.dumps(dict(status='PASS',metric_rows=len(n),repeat_summary_rows=len(summary),paired_seed_rows=len(p),paired_repeat_rows=len(ps),identity_modules=len(ident),regret_details=len(detail),result_artifacts_verified=len(AUDIT),missing_metrics=0,new_training=0,metric_reevaluation=0,confidence_interval='mean +/- t_0.975,4 * sample_sd / sqrt(5)',regret_selection='descriptive saved test metrics; no model tuning',regret_0033_dataset_backbone='yelp2018/NFM',regret_0068_dataset_backbone='yelp2018/MF_BPR'),indent=2)+'\n')
    print(summary.to_string(index=False));print(detail.to_string(index=False))


