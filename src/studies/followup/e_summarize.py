"""Original E per-seed means, sample SDs and paired repeat intervals."""
from pathlib import Path
import json,csv,statistics,math
from scipy.stats import t
from .e_worker import OUT,write,write_csv
def summarize(gate_path,phase):
    gate=json.loads(Path(gate_path).read_text());seeds=gate['seeds'];datasets=gate['datasets']
    rows=[];missing=[]
    for d in datasets:
        for seed in seeds:
            p=OUT/'e'/phase/d/f'seed_{seed}'/'seed_results.csv';marker=p.with_name('complete.json')
            if not p.is_file() or not marker.is_file():missing.append([d,seed]);continue
            r=list(csv.DictReader(p.open()))
            if len(r)!=9:raise ValueError('Expected exactly U/W/F x Base/Pop/NPGC')
            if any(int(x['training_seed'])!=seed or x['dataset']!=d or x['phase']!=phase for x in r):raise ValueError('E result identity differs')
            rows.extend(r)
    write_csv(OUT/'e'/f'{phase}_seed_results.csv',rows or [{'status':'NO_COMPLETED_COORDINATES'}])
    if missing:
        write(OUT/'e'/f'{phase}_summary_status.json',{'status':'INCOMPLETE_PREDECLARED_SEED_SET','missing':missing,'predeclared_seeds':seeds,
              'no_reduced_seed_claim':True});return
    summary=[];paired=[]
    metrics=['ndcg@20','recall@20','candidate_recall50','oracle_ndcg20','spearman_s_p_mean']
    for d in datasets:
        for protocol in ['U','W','F']:
            for method in ['base','item_degree_ranking','original_nonparamgc']:
                rr=[r for r in rows if r['dataset']==d and r['protocol']==protocol and r['method']==method]
                if sorted(int(r['training_seed']) for r in rr)!=seeds:raise ValueError('Duplicate/missing seed')
                for metric in metrics:
                    undefined=[int(r['training_seed']) for r in rr if r.get(metric) in (None,'','None')]
                    if undefined:
                        summary.append({'phase':phase,'dataset':d,'protocol':protocol,'method':method,'metric':metric,'n_seeds':len(rr),
                                        'defined_seeds':len(rr)-len(undefined),'undefined_seeds':json.dumps(undefined),'status':'UNDEFINED_SEED_VALUES',
                                        'mean':None,'sd':None,'repeat_ci95_low':None,'repeat_ci95_high':None})
                        continue
                    values=[float(r[metric]) for r in rr];sd=statistics.stdev(values);half=float(t.ppf(.975,len(values)-1))*sd/math.sqrt(len(values))
                    summary.append({'phase':phase,'dataset':d,'protocol':protocol,'method':method,'metric':metric,'n_seeds':len(values),
                                    'defined_seeds':len(values),'undefined_seeds':'[]','status':'DEFINED',
                                    'mean':statistics.mean(values),'sd':sd,'repeat_ci95_low':statistics.mean(values)-half,'repeat_ci95_high':statistics.mean(values)+half})
            for metric in ['ndcg@20','recall@20']:
                lookup={(int(r['training_seed']),r['method']):float(r[metric]) for r in rows if r['dataset']==d and r['protocol']==protocol}
                differences=[lookup[(s,'item_degree_ranking')]-lookup[(s,'base')] for s in seeds]
                mean=statistics.mean(differences);sd=statistics.stdev(differences);half=float(t.ppf(.975,len(seeds)-1))*sd/math.sqrt(len(seeds))
                paired.append({'phase':phase,'dataset':d,'protocol':protocol,'contrast':'Pop-Base','metric':metric,'n_seeds':len(seeds),
                               'mean_difference':mean,'sd':sd,'repeat_ci95_low':mean-half,'repeat_ci95_high':mean+half,
                               'seed_differences':json.dumps(dict(zip(seeds,differences)))})
    write_csv(OUT/'e'/f'{phase}_mean_sd_ci.csv',summary);write_csv(OUT/'e'/f'{phase}_pop_minus_base_paired.csv',paired)
    write(OUT/'e'/f'{phase}_summary_status.json',{'status':'COMPLETE','predeclared_seeds':seeds,'seed_rows':len(rows),'new_experiment_kept_separate':True})
