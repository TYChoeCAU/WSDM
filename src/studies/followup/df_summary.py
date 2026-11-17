"""Original DF summaries and paired source-to-DF/F transfer contrasts."""
from .df_common import *
from .paths import input_file,EVIDENCE
from .p0p1_analysis import transfer,summarize,SEEDS,s3_index
from collections import defaultdict
FIXED={'Pop-Base':('item_degree_ranking','base'),'NPGC-Base':('original_nonparamgc','base'),'Pop-NPGC':('item_degree_ranking','original_nonparamgc')}
LEARNED_CONTRASTS={**{m+'-Base':(m,'base') for m in (*MASKS,'prm_principled')},'PRM+PL-NPGC':('prm_principled','original_nonparamgc'),'PRM+PL-Res(p)':('prm_principled','P'),'Res(p)-Pop':('P','item_degree_ranking'),'Res(H)-NPGC':('H','original_nonparamgc')}
FIXED_METHODS=('base','item_degree_ranking','original_nonparamgc')
def flatten(row):
    value={**{k:v for k,v in row.items() if k!='metrics'},**row['metrics']}
    value.update({k+'_x100':100*float(value[k]) for k in (*row['metrics'],'candidate_recall50','oracle_ndcg20')})
    value['metric_units']='raw fractions plus explicitly labelled x100 columns'
    return value


def main():
    prior=s3_index(read_csv(input_file('s3_seed_results',EVIDENCE/'s3/extracted_seed_results.csv')));fixed=[];frozen=[];checks=[];runtime_rows=[]
    for path in sorted((OUT/'artifacts/df_s3').glob('*/seed_*/fixed_result.json')):
        result=json.loads(path.read_text());checks.append(dict(dataset=result['dataset'],seed=result['seed'],**result['checks']))
        runtime_rows.append(dict(dataset=result['dataset'],seed=result['seed'],phase='fixed',seconds=result['seconds']))
        fixed.extend(flatten(r) for r in result['metrics'].values())
    for path in sorted((OUT/'artifacts/df_s3').glob('*/seed_*/frozen/*/*/result.json')):
        r=json.loads(path.read_text());frozen.append(flatten(r));runtime_rows.append(dict(dataset=r['dataset'],seed=r['training_seed'],phase=r['protocol']+'/'+r['method'],seconds=r['seconds']))
    write_csv(OUT/'DF_s3_seed_results.csv',fixed);write_csv(OUT/'DF_frozen_learned_seed_results.csv',frozen)
    summaries=[]
    for family,rows in [('fixed',fixed),('frozen',frozen)]:
        groups=defaultdict(list)
        for r in rows:groups[r['dataset'],r['protocol'],r['method']].append(r)
        for (dataset,protocol,method),rr in groups.items():
            seeds=sorted(r['training_seed'] for r in rr)
            entry=dict(dataset=dataset,backbone='NFM',retriever='MF_BPR',analysis_family=family,protocol=protocol,method=method,
                n_seeds=len(seeds),seeds=';'.join(map(str,seeds)),status='COMPLETE' if tuple(seeds)==SEEDS else f'PARTIAL(n={len(seeds)})')
            for metric in ['ndcg@10','ndcg@20','recall@10','recall@20','candidate_recall50','oracle_ndcg20']:
                stat=summarize([r[metric] for r in rr]);entry.update({metric+'_'+key:val for key,val in stat.items() if key in ['mean_raw','sample_sd_raw','ci_low_raw','ci_high_raw','mean_x100','sample_sd_x100','ci_low_x100','ci_high_x100']})
            summaries.append(entry)
    write_csv(OUT/'DF_s3_summary.csv',[r for r in summaries if r['analysis_family']=='fixed'])
    write_csv(OUT/'DF_frozen_learned_summary.csv',[r for r in summaries if r['analysis_family']=='frozen'])
    fi={(r['dataset'],r['training_seed'],r['method']):r for r in fixed}
    fr={(r['dataset'],r['training_seed'],r['source_protocol'],r['method']):r for r in frozen}
    transfers=[];paired=[]
    for dataset in ['ml-1m','yelp2018']:
      for family,contrasts in [('fixed',FIXED),('frozen',LEARNED_CONTRASTS)]:
       for source in ['U','W']:
        for contrast,(a,b) in contrasts.items():
            def exists(s,m):return (dataset,s,m)in fi if m in FIXED_METHODS else (dataset,s,source,m)in fr
            def value(s,m):return fi[dataset,s,m] if m in FIXED_METHODS else fr[dataset,s,source,m]
            def old_value(s,m):return prior[dataset,'NFM',s,'matched','F',m] if m in FIXED_METHODS else prior[dataset,'NFM',s,'frozen',source+'_to_F',m]
            seeds=[s for s in SEEDS if exists(s,a) and exists(s,b)]
            if not seeds:continue
            sv=[];tv=[];fv=[]
            for s in seeds:
                sv.append(float(prior[dataset,'NFM',s,'matched',source,a]['ndcg@20'])-float(prior[dataset,'NFM',s,'matched',source,b]['ndcg@20']))
                tv.append(float(value(s,a)['ndcg@20'])-float(value(s,b)['ndcg@20']))
                fa=old_value(s,a);fb=old_value(s,b)
                fv.append(float(fa['ndcg@20'])-float(fb['ndcg@20']))
            meta=dict(dataset=dataset,backbone='NFM',retriever='MF_BPR',analysis_family=family,source_protocol=source,
                target_protocol='DF',comparison=source+('-DF' if family=='fixed'else'_to_DF'),contrast=contrast,method_a=a,method_b=b)
            row=transfer(sv,tv,meta,seeds=tuple(seeds));old=transfer(sv,fv,{**meta,'target_protocol':'F'},seeds=tuple(seeds))
            row.update(existing_F_classification=old['classification'],existing_F_target_mean_raw=old['target_effect_mean_raw'],
                existing_F_target_mean_x100=100*old['target_effect_mean_raw'],existing_F_target_ci95_low=old['target_ci95_low'],
                existing_F_target_ci95_high=old['target_ci95_high'],classification_changed=row['classification']!=old['classification'])
            transfers.append(row)
            if family=='fixed'and source=='U':paired.append(dict(dataset=dataset,contrast=contrast,n_seeds=len(seeds),**summarize(tv)))
    write_csv(OUT/'DF_s3_transfer.csv',[r for r in transfers if r['analysis_family']=='fixed'])
    write_csv(OUT/'DF_frozen_learned.csv',[r for r in transfers if r['analysis_family']=='frozen'])
    write_csv(OUT/'DF_s3_paired.csv',paired);write_csv(OUT/'DF_runtime.csv',runtime_rows)
    checkpath=OUT/'DF_checks.json';allchecks=json.loads(checkpath.read_text())if checkpath.exists()else{}
    allchecks['S3']=dict(status='COMPLETE'if len(checks)==10 else f'PARTIAL(n={len(checks)})',expected_coordinates=10,
        completed_coordinates=len(checks),all_mandatory_checks_pass=all(r['all_mandatory_checks_pass'] for r in checks),
        raw_metric_atol=1e-3,identifier_matching='exact',coordinates=checks,
        frozen_completed=len(frozen),frozen_expected=180,original_sources_read_only=True,new_training_runs=0)
    atomic_json(checkpath,allchecks)
    print(json.dumps(dict(fixed=len(fixed),expected_fixed=50,frozen=len(frozen),expected_frozen=180,checks=len(checks))))
