"""Original 2x2 output aggregation, using local completed E artifacts."""
from .p0p1_analysis import *
def run_p1(index):
    rho_rows=read(input_file('s3_spearman',EVIDENCE/'checks/list_spearman_seed_results.csv'))
    rho={}
    for r in rho_rows:
        if r['study']=='S3'and r['regime']=='R0':
            key=(r['dataset'],r['backbone'],int(r['training_seed']),r['protocol'])
            if key in rho:raise ValueError('Duplicate rho '+str(key))
            rho[key]=r
    records=[];completions=[]
    specs=[('S3_NFM_BCE','NFM','observed_threshold_BCE','existing_S3'),
           ('E1_NFM_BPR','NFM','BPR_unobserved_negative','E1'),
           ('E2_MF_BCE','MF','observed_threshold_BCE','E2'),
           ('S3_MF_BPR','MF','BPR_unobserved_negative','existing_S3')]
    for model,architecture,objective,phase in specs:
      if phase=='existing_S3':
        backbone='NFM'if architecture=='NFM'else'MF_BPR'
        for d,s,p,m in itertools.product(DATASETS,SEEDS,PROTOCOLS,FIXED):
            original=index[d,backbone,s,'matched',p,m];rr=rho[d,backbone,s,p]
            records.append(dict(cohort=model,architecture=architecture,objective=objective,phase=phase,dataset=d,
                training_seed=s,protocol=p,method=m,method_label=LABELS[m],**{k:float(original[k])for k in('ndcg@10','ndcg@20','recall@10','recall@20','candidate_recall50','oracle_ndcg20')},
                backbone_score_popularity_spearman_mean=float(rr['mean']),spearman_defined_users=int(rr['defined_users']),
                n_users=int(original['n_users']),source_evidence=f"{original['_source_path']}:{original['_source_line']}",
                spearman_evidence=f"{rr['_source_path']}:{rr['_source_line']}",status='COMPLETE'))
      else:
        aggregate=read(OUT/'e'/f'{phase}_seed_results.csv')
        for d,s in itertools.product(DATASETS,SEEDS):
            directory=OUT/'e'/phase/d/f'seed_{s}'
            status_path=directory/'run_status.json';seed_path=directory/'seed_results.csv'
            if not status_path.exists() or not seed_path.exists():
                completions.append(dict(phase=phase,dataset=d,training_seed=s,status='NOT_STARTED_OR_INCOMPLETE'));continue
            status=json.loads(status_path.read_text())
            if status['status']!='COMPLETE':
                completions.append(dict(phase=phase,dataset=d,training_seed=s,status=status['status']));continue
            own=read(seed_path)
            if len(own)!=9 or {(r['protocol'],r['method'])for r in own}!=set(itertools.product(PROTOCOLS,FIXED)):
                raise ValueError('Incomplete method/protocol coordinate '+str(seed_path))
            agg=[r for r in aggregate if r['dataset']==d and int(r['training_seed'])==s]
            for r in own:
                ar=[v for v in agg if(v['protocol'],v['method'])==(r['protocol'],r['method'])]
                if len(ar)!=1:raise ValueError('Phase seed summary mismatch')
                for k in ('ndcg@20','recall@20','candidate_recall50','oracle_ndcg20','spearman_s_p_mean'):
                    if float(r[k])!=float(ar[0][k]):raise ValueError('Phase aggregate differs from its seed file')
                if r['checkpoint_sha256']!=status['checkpoint_sha256']:
                    raise ValueError('Selected checkpoint identity differs')
                if r['full_truth_denominator']!='True':raise ValueError('Full truth denominator not recorded')
                records.append(dict(cohort=model,architecture=architecture,objective=objective,phase=phase,dataset=d,
                    training_seed=s,protocol=r['protocol'],method=r['method'],method_label=LABELS[r['method']],
                    **{k:float(r[k])for k in('ndcg@10','ndcg@20','recall@10','recall@20','candidate_recall50','oracle_ndcg20')},
                    backbone_score_popularity_spearman_mean=float(r['spearman_s_p_mean']),spearman_defined_users=int(r['spearman_defined_users']),
                    n_users=int(r['n_users']),checkpoint_sha256=r['checkpoint_sha256'],selected_backbone_epoch=int(r['selected_backbone_epoch']),
                    source_evidence=f"{r['_source_path']}:{r['_source_line']}",spearman_evidence=f"{r['_source_path']}:{r['_source_line']}",status='COMPLETE'))
            completions.append(dict(phase=phase,dataset=d,training_seed=s,status='COMPLETE',checkpoint_sha256=status['checkpoint_sha256'],
                                    status_evidence=str(status_path)+':2',no_new_training_or_restart=True))
    write('E_seed_results.csv',records)
    write('E_completion_inventory.csv',completions)
    mapping={(r['cohort'],r['dataset'],r['training_seed'],r['protocol'],r['method']):r for r in records}
    if len(mapping)!=len(records):raise ValueError('Duplicate E analysis coordinate')
    summary=[];paired=[];transfers=[]
    for model,architecture,objective,phase in specs:
      for d in DATASETS:
        present=sorted({r['training_seed']for r in records if(r['cohort'],r['dataset'])==(model,d)})
        status='COMPLETE'if tuple(present)==SEEDS else f'PARTIAL(n={len(present)})'if present else'NOT_STARTED'
        if not present:
            summary.append(dict(cohort=model,dataset=d,status=status,n_training_seeds=0));continue
        for p,m in itertools.product(PROTOCOLS,FIXED):
            selected=[mapping[model,d,s,p,m]for s in present]
            row=dict(cohort=model,architecture=architecture,objective=objective,phase=phase,dataset=d,protocol=p,method=m,method_label=LABELS[m],
                     status=status,n_training_seeds=len(present),training_seeds=';'.join(map(str,present)))
            for metric in('ndcg@10','ndcg@20','recall@10','recall@20','candidate_recall50','oracle_ndcg20'):
                stat=summarize([r[metric]for r in selected]);row[metric+'_mean_x100']=stat['mean_x100'];row[metric+'_sample_sd_x100']=stat['sample_sd_x100']
            stat=summarize([r['backbone_score_popularity_spearman_mean']for r in selected]);row['spearman_user_mean_then_seed_mean']=stat['mean_raw'];row['spearman_seed_sample_sd']=stat['sample_sd_raw']
            row['n_users_by_seed']=';'.join(str(r['n_users'])for r in selected);row['spearman_defined_users_by_seed']=';'.join(str(r['spearman_defined_users'])for r in selected)
            summary.append(row)
        contrasts={k:v for k,v in STAT.CONTRASTS.items()if k in('Pop-Base','NPGC-Base','Pop-NPGC')}
        effects={}
        for p,(name,(left,right)),metric in itertools.product(PROTOCOLS,contrasts.items(),('ndcg@20','recall@20')):
            values=[mapping[model,d,s,p,left][metric]-mapping[model,d,s,p,right][metric]for s in present]
            if metric=='ndcg@20':effects[p,name]=values
            paired.append(dict(cohort=model,architecture=architecture,objective=objective,phase=phase,dataset=d,protocol=p,contrast=name,metric=metric,
                status=status,training_seeds=';'.join(map(str,present)),**summarize(values),differences_by_seed_raw=json.dumps(dict(zip(map(str,present),values)))))
        for source,target in [('U','W'),('U','F'),('W','F')]:
          for name in contrasts:
            transfers.append(transfer(effects[source,name],effects[target,name],dict(cohort=model,architecture=architecture,objective=objective,phase=phase,
                dataset=d,analysis_family='matched_objective_comparison',comparison=source+'-'+target,source_protocol=source,target_protocol=target,contrast=name),tuple(present)))
    write('E_summary.csv',summary);write('E_pop_minus_base_paired.csv',paired);write('E_transfer.csv',transfers)
    return dict(completed_stage_E_coordinates=sum(r['status']=='COMPLETE' for r in completions),expected_stage_E_coordinates=20,seed_rows=len(records),summary_rows=len(summary),paired_rows=len(paired),transfer_rows=len(transfers))
