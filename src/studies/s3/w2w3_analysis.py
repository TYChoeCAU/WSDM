"""D1--D7 from immutable saved per-user predictions, with regimes kept separate."""
from __future__ import annotations
import argparse, csv, hashlib, itertools, json, os, sys
from collections import defaultdict, Counter
from pathlib import Path
from .w2w3_statistics import *
from .w2w3_extract import read, write_csv, atomic_json, evidence_rows, sha
import pyarrow.parquet as pq

CELLS=tuple(itertools.product(('yelp2018','ml-1m'),('NFM','MF_BPR')))
ANALYSIS_METRICS=('ndcg20','recall20')
POLICIES=('identity_inclusive','trained_only')
PAIRS=(('U','W','matched'),('U','F','matched'),('W','F','matched'),('U','U_to_F','frozen'),('W','W_to_F','frozen'))

def ci_tuple(values,level=.95):
    r=interval(values,level)
    return r['ci_low'],r['ci_high']

def scope_ready(regime,seeds,available):
    for ds,bb,seed,p,m in itertools.product(('yelp2018','ml-1m'),('NFM','MF_BPR'),seeds,('U','W','F'),METHODS):
        if (regime,ds,bb,seed,p,m)not in available:return False
    for ds,bb,seed,p,m in itertools.product(('yelp2018','ml-1m'),('NFM','MF_BPR'),seeds,('U_to_F','W_to_F'),('prm_principled',*RESIDUALS)):
        if (regime,ds,bb,seed,p,m)not in available:return False
    return True

def scopes_from_index(rows):
    available={(r['regime'],r['dataset'],r['backbone'],r['training_seed'],r['protocol'],r['method'])for r in rows}
    complete=scope_ready('R0',(2020,2021,2022,2023,2024),available)
    if not complete:raise ValueError('Analysis requires all 20 R0 backbone repeats and matched/frozen method cells')
    return [dict(scope='main_R0',regime='R0',seeds=(2020,2021,2022,2023,2024),
        method_sets=('REF4','ALL12'),allowed=METHODS,primary=True)],True,True

def load_coordinate(paths):
    protocols={};all_columns={};proofs={};methods_by_protocol={}
    for path in paths:
        meta=read(path.with_name('extraction.json'));protocol=meta['coordinate']['protocol'];proofs[protocol]=meta
        frame=pq.read_table(path,columns=['method','selection_policy','user_id','full_relevant_count','full_truth_user_digest',*ANALYSIS_METRICS]).to_pandas()
        users=truth=None;vectors={}
        for (method,policy),part in frame.groupby(['method','selection_policy'],sort=True):
            part=part.sort_values('user_id',kind='stable')
            uu=part['user_id'].to_numpy();tt=np.asarray([str(a)+':'+str(b)for a,b in zip(part['full_relevant_count'],part['full_truth_user_digest'])])
            if users is None:users,truth=uu,tt
            elif not np.array_equal(users,uu)or not np.array_equal(truth,tt):raise ValueError('Within-protocol user/full-truth mismatch')
            for metric in ANALYSIS_METRICS:vectors[(protocol,method,policy,metric)]=part[metric].to_numpy(dtype=np.float64)
        cols=sorted(vectors);all_columns[protocol]=cols
        protocols[protocol]=(users,truth,np.column_stack([vectors[c]for c in cols]))
        methods_by_protocol[protocol]=set(frame['method'])
        del frame
    for field in ('backbone_checkpoint_sha256','A_digest','B_digest','master_split_digest','graph_digest'):
        if len({p['lineage'][field]for p in proofs.values()})!=1:raise ValueError('Recorded cross-protocol lineage mismatch '+field)
    users,matrices,audit=aligned_users(protocols)
    vectors={c:matrices[p][:,j]for p,cols in all_columns.items()for j,c in enumerate(cols)}
    for p in list(proofs):
        if p.endswith('_to_F'):
            for m,metric in itertools.product(FIXED,ANALYSIS_METRICS):
                col=('F',m,'native',metric)
                if col in vectors:vectors[(p,m,'native',metric)]=vectors[col]
    columns=sorted(vectors);matrix=np.column_stack([vectors[c]for c in columns])
    lineage={field:next(iter(proofs.values()))['lineage'][field]for field in ('backbone_checkpoint_sha256','A_digest','B_digest','master_split_digest','graph_digest')}
    return users,columns,matrix,dict(**audit,**lineage),proofs

def user_outputs(scope,key,users,columns,matrix,boot):
    regime,dataset,bb,seed=key;lookup={c:i for i,c in enumerate(columns)};means=matrix.mean(0)
    effects=[];classifications=[];ranks=[];reversals=[];regrets=[];within=[];missing=[]
    common=dict(scope=scope['scope'],regime=regime,dataset=dataset,backbone=bb,training_seed=seed,
        n_common_users=len(users),bootstrap_replicates=len(boot),bootstrap_seed=8702026)
    for policy in POLICIES:
      def col(p,m,metric):
        # Frozen fixed rules have no fitted parameters or protocol selector.
        # Their actual F predictions therefore supply the frozen target values.
        if p.endswith('_to_F')and m in FIXED:p='F'
        return(p,m,policy if m in RESIDUALS else'native',metric)
      for metric in ANALYSIS_METRICS:
        # D1 includes contrasts inside each protocol as well as changes between protocols.
        for p in ('U','W','F'):
          for name,(left,right)in CONTRASTS.items():
            if left not in scope['allowed']or right not in scope['allowed']:continue
            if not all(col(p,m,metric)in lookup for m in (left,right)):continue
            a,b=(lookup[col(p,m,metric)]for m in (left,right));draws=boot[:,a]-boot[:,b]
            within.append(dict(**common,selection_policy=policy,metric=metric.replace('20','@20'),analysis_family='matched',
                comparison=p,contrast=name,effect_source_raw=float(means[a]-means[b]),effect_target_raw=float(means[a]-means[b]),
                delta_raw=0.,point_effect_raw=float(means[a]-means[b]),user_ci_low=interval(draws)['ci_low'],user_ci_high=interval(draws)['ci_high']))
        for p,q,family in PAIRS:
          base=dict(**common,selection_policy=policy,metric=metric.replace('20','@20'),analysis_family=family,comparison=p+'-'+q)
          for name,(left,right)in CONTRASTS.items():
            if left not in scope['allowed']or right not in scope['allowed']:continue
            cc=[col(t,m,metric)for t in (p,q)for m in (left,right)]
            if not all(c in lookup for c in cc):
                missing.append(dict(**base,kind='contrast',name=name,status='MISSING_SAVED_METHODS'));continue
            ix=[lookup[c]for c in cc];a=boot[:,ix[0]]-boot[:,ix[1]];b=boot[:,ix[2]]-boot[:,ix[3]];change=b-a
            point_a=float(means[ix[0]]-means[ix[1]]);point_b=float(means[ix[2]]-means[ix[3]])
            row=dict(**base,contrast=name,effect_source_raw=point_a,effect_target_raw=point_b,
                delta_raw=point_b-point_a,delta_x100=(point_b-point_a)*100,
                source_ci95_low=ci_tuple(a)[0],source_ci95_high=ci_tuple(a)[1],target_ci95_low=ci_tuple(b)[0],target_ci95_high=ci_tuple(b)[1],
                source_ci90_low=ci_tuple(a,.90)[0],source_ci90_high=ci_tuple(a,.90)[1],target_ci90_low=ci_tuple(b,.90)[0],target_ci90_high=ci_tuple(b,.90)[1],
                delta_ci95_low=ci_tuple(change)[0],delta_ci95_high=ci_tuple(change)[1],
                uncertainty_unit='paired_user_clusters_conditional_on_fixed_trained_models')
            effects.append(row)
            for delta in DELTAS:
                classifications.append(dict(**base,contrast=name,practical_delta_raw=delta,decision_basis='D2_seed_user_bootstrap',
                    classification=classify(ci_tuple(a),ci_tuple(b),ci_tuple(a,.9),ci_tuple(b,.9),delta),
                    source_effect_raw=point_a,target_effect_raw=point_b))
          for method_set in scope['method_sets']:
            methods=METHOD_SETS[method_set];required=[col(t,m,metric)for t in (p,q)for m in methods]
            if not all(c in lookup for c in required):
                missing.append(dict(**base,kind='method_set',name=method_set,status='MISSING_SAVED_METHODS'));continue
            ia=[lookup[col(p,m,metric)]for m in methods];ib=[lookup[col(q,m,metric)]for m in methods]
            x,y=means[ia],means[ib];bx,by=boot[:,ia],boot[:,ib]
            rank_draws=tau_b(bx,by);tau=finite(tau_b(x,y))
            ranks.append(dict(**base,method_set=method_set,n_methods=len(methods),kendall_tau_b=tau,
                status='DEFINED'if tau is not None else'UNDEFINED_CONSTANT_ORDER',**interval(rank_draws),
                uncertainty_unit='paired_user_clusters_conditional_on_fixed_trained_models'))
            for delta in DELTAS:
                actual=clear_reversal(x,y,delta);draws=clear_reversal(bx,by,delta)
                reversals.append(dict(**base,method_set=method_set,practical_delta_raw=delta,
                    **{k:finite(v)for k,v in actual.items()},
                    reversal_fraction_total_ci_low=ci_tuple(draws['reversal_fraction_total'])[0],
                    reversal_fraction_total_ci_high=ci_tuple(draws['reversal_fraction_total'])[1],
                    informative_reversal_ci_low=ci_tuple(draws['reversal_fraction_informative'])[0],
                    informative_reversal_ci_high=ci_tuple(draws['reversal_fraction_informative'])[1]))
            if q not in ('F','U_to_F','W_to_F')or metric!='ndcg20':continue
            actual=decision_regret(x,y);draws=decision_regret(bx,by)
            rr=dict(**base,method_set=method_set,source_selected_method=methods[int(actual['source_selected_index'])],
                target_selected_method=methods[int(actual['target_selected_index'])],
                tie_policy='registered method order for single choice; fractional top-set agreement is Jaccard',
                selection_scope='descriptive choice from saved test metrics, not validation tuning or a deployment selection rule')
            for name,value in actual.items():
                rr[name]=finite(value)
                if name not in ('source_selected_index','target_selected_index','source_tied_maxima','target_tied_maxima'):
                    lo,hi=ci_tuple(draws[name]);rr[name+'_ci_low']=lo;rr[name+'_ci_high']=hi
            regrets.append(rr)
    return effects,classifications,ranks,reversals,regrets,within,missing

def group_rows(rows,fields):
    grouped=defaultdict(list)
    for r in rows:grouped[tuple(r[k]for k in fields)].append(r)
    return [(dict(zip(fields,k)),v)for k,v in grouped.items()]

def tost_family_role(key,delta):
    selected=(key['selection_policy']=='identity_inclusive'and key['metric']=='ndcg@20'and delta==.0025)
    if selected and key['analysis_family']=='matched':
        if key['scope']=='main_R0':return True,'registered_primary'
        if key['scope']=='A5_R0':return False,'registered_partial_A5'
    return False,'exploratory_or_sensitivity'

def repeat_outputs(scopes,effects,userclasses,within):
    repeated=[];classified=[];tost=[]
    scope_map={x['scope']:x for x in scopes}
    fields=('scope','regime','dataset','backbone','selection_policy','metric','analysis_family','comparison','contrast')
    for key,rows in group_rows([*effects,*within],fields):
        expected=scope_map[key['scope']]['seeds'];rows=sorted(rows,key=lambda r:r['training_seed'])
        actual=[r['training_seed']for r in rows]
        if tuple(actual)!=tuple(expected):continue
        a=[r['effect_source_raw']for r in rows];b=[r['effect_target_raw']for r in rows]
        change=np.asarray(b)-np.asarray(a)
        point=a if '-'not in key['comparison']else change
        summary=repeat_interval(point);sa=repeat_interval(a);sb=repeat_interval(b);a90=repeat_interval(a,.9);b90=repeat_interval(b,.9)
        base=dict(**key,training_seeds=';'.join(map(str,expected)),status='COMPLETE_HOMOGENEOUS',**summary,
            source_effect_mean_raw=sa['mean_raw'],target_effect_mean_raw=sb['mean_raw'],
            source_ci95_low=sa['ci_low_raw'],source_ci95_high=sa['ci_high_raw'],target_ci95_low=sb['ci_low_raw'],target_ci95_high=sb['ci_high_raw'],
            source_ci90_low=a90['ci_low_raw'],source_ci90_high=a90['ci_high_raw'],target_ci90_low=b90['ci_low_raw'],target_ci90_high=b90['ci_high_raw'],
            uncertainty_unit='training_repeats_fixed_split_within_one_recorded_batch_regime')
        repeated.append(base)
        if '-'not in key['comparison']:continue
        for delta in DELTAS:
            cls=classify((sa['ci_low_raw'],sa['ci_high_raw']),(sb['ci_low_raw'],sb['ci_high_raw']),
                (a90['ci_low_raw'],a90['ci_high_raw']),(b90['ci_low_raw'],b90['ci_high_raw']),delta)
            matching=[r for r in userclasses if all(r[k]==v for k,v in key.items())and r['practical_delta_raw']==delta]
            cc=Counter(r['classification']for r in matching)
            classified.append(dict(**base,practical_delta_raw=delta,decision_basis='D1_paired_training_repeat_t_intervals',
                classification=cls,user_seed_decisions=json.dumps(dict(cc),sort_keys=True),user_seed_agrees_with_D1=cc.get(cls,0),
                user_seed_decision_count=len(matching),zero_variance_structural_review=bool(np.std(a)==0 and np.std(b)==0)))
            if key['contrast']not in ('Pop-Base','PRM+PL-Res(p)')or key['comparison']not in ('U-F','W-F','U-U_to_F','W-W_to_F'):
                continue
            stat=paired_tost(b,a,delta)
            primary,family_role=tost_family_role(key,delta)
            tost.append(dict(**key,training_seeds=';'.join(map(str,expected)),status='COMPLETE_HOMOGENEOUS',
                **stat,registered_primary_family=primary,planned_family_size=16,
                tested_quantity='target-minus-source change of a paired method contrast',
                family_role=family_role))
    familyfields=('scope','regime','selection_policy','metric','analysis_family','delta_raw')
    for _,rows in group_rows(tost,familyfields):
        # Every family retains its planned sixteen slots, including missing contrasts.
        adjusted=holm_planned([r.get('p_equivalence')for r in rows],16)
        available=sum(r.get('p_equivalence')is not None for r in rows)
        for row,p in zip(rows,adjusted):row.update(p_equivalence_holm=float(p),computed_family_size=available,
            family_complete=available==16,equivalent_holm=bool(p<.05)if row.get('p_equivalence')is not None else None)
    return repeated,classified,tost

def scalar_repeat_summary(rows,scopes,kind,metric_field):
    scopes={s['scope']:s for s in scopes};out=[]
    fields=('scope','regime','dataset','backbone','selection_policy','metric','analysis_family','comparison','method_set')
    for key,values in group_rows(rows,fields):
        expected=scopes[key['scope']]['seeds'];selected=sorted(values,key=lambda x:x['training_seed'])
        if tuple(x['training_seed']for x in selected)!=tuple(expected):continue
        finite_values=[x[metric_field]for x in selected]
        if any(x is None for x in finite_values):
            out.append(dict(**key,statistic=kind,status='UNDEFINED_IN_ONE_OR_MORE_TRAINING_REPEATS',n_training_seeds=len(expected),training_seeds=';'.join(map(str,expected))))
        else:out.append(dict(**key,statistic=kind,status='COMPLETE_HOMOGENEOUS',training_seeds=';'.join(map(str,expected)),**repeat_interval(finite_values)))
    return out

def method_summaries(scopes):
    path=RUN/'extracted_seed_results.csv';rows=list(csv.DictReader(path.open()))if path.exists()else[];out=[]
    fields=('regime','analysis_family','dataset','backbone','protocol','method','selection_policy')
    for scope in scopes:
        eligible=[r for r in rows if r['regime']==scope['regime']and int(r['training_seed'])in scope['seeds']and r['method']in scope['allowed']]
        for key,values in group_rows(eligible,fields):
            values=sorted(values,key=lambda x:int(x['training_seed']))
            if tuple(int(x['training_seed'])for x in values)!=tuple(scope['seeds']):continue
            for metric in (*METRICS,'candidate_recall50','oracle_ndcg20'):
                out.append(dict(scope=scope['scope'],**key,metric=metric,training_seeds=';'.join(map(str,scope['seeds'])),
                    status='COMPLETE_HOMOGENEOUS',**repeat_interval([float(r[metric])for r in values])))
    write_csv(RUN/'homogeneous_method_summary.csv',out)

def analyze(a5=False):
    plan=read(RUN/'analysis_plan_w2w3.json');scopes,b,c=scopes_from_index(evidence_rows())
    if a5:scopes=[s for s in scopes if s['scope']=='A5_R0']
    paths=sorted((RUN/'private/per_user_parts').glob('*/*/*/*/seed_*/*/metrics.parquet'));groups=defaultdict(list)
    for path in paths:
        m=read(path.with_name('extraction.json'))['coordinate'];groups[(m['regime'],m['dataset'],m['backbone'],m['training_seed'])].append(path)
    effects=[];userclasses=[];ranks=[];reversals=[];regrets=[];within=[];missing=[];pairing=[];boot_inventory=[]
    for scope in scopes:
        for dataset,bb,seed in itertools.product(('yelp2018','ml-1m'),('NFM','MF_BPR'),scope['seeds']):
            if (scope['regime'],dataset,bb,seed)not in groups:
                missing.append(dict(scope=scope['scope'],regime=scope['regime'],dataset=dataset,backbone=bb,training_seed=seed,
                    status='MISSING_EXTRACTED_COORDINATE'))
    for key,parts in sorted(groups.items()):
        active=[s for s in scopes if s['regime']==key[0]and key[3]in s['seeds']]
        if not active:continue
        print('BOOTSTRAP',key,flush=True)
        try:users,columns,matrix,audit,proofs=load_coordinate(parts)
        except ValueError as error:
            missing.append(dict(regime=key[0],dataset=key[1],backbone=key[2],training_seed=key[3],status='PAIRING_REVIEW_REQUIRED',reason=str(error)));continue
        pairing.append(dict(regime=key[0],dataset=key[1],backbone=key[2],training_seed=key[3],**{k:json.dumps(v,sort_keys=True)if isinstance(v,dict)else v for k,v in audit.items()}))
        digest=input_digest(dict(parts={p:m['input_digest']for p,m in proofs.items()},columns=columns,plan_sha256=sha(RUN/'analysis_plan_w2w3.json'),users_sha256=hashlib.sha256(users.tobytes()).hexdigest()))
        output=RUN/'private/bootstrap_macros'/key[0]/key[1]/key[2]/f'seed_{key[3]}.npz';metadata=output.with_suffix('.json');output.parent.mkdir(parents=True,exist_ok=True)
        reused=output.is_file()and metadata.is_file()and read(metadata).get('input_digest')==digest
        if reused:boot=np.load(output)['macros']
        else:
            boot=bootstrap_macros(matrix,users,plan['bootstrap_replicates'],plan['bootstrap_seed'],
                identity=f'{key[1]}|{key[3]}|'+hashlib.sha256(users.tobytes()).hexdigest())
            tmp=output.with_suffix('.tmp.npz');np.savez_compressed(tmp,macros=boot);tmp.replace(output)
            atomic_json(metadata,dict(input_digest=digest,columns=columns,n_users=len(users),bootstrap_seed=plan['bootstrap_seed'],replicates=plan['bootstrap_replicates']))
        boot_inventory.append(dict(regime=key[0],dataset=key[1],backbone=key[2],training_seed=key[3],input_digest=digest,reused=reused,n_users=len(users),replicates=len(boot)))
        for scope in active:
            values=user_outputs(scope,key,users,columns,matrix,boot)
            for dest,new in zip((effects,userclasses,ranks,reversals,regrets,within,missing),values):dest.extend(new)
        del matrix,boot
    repeats,classes,tost=repeat_outputs(scopes,effects,userclasses,within)
    method_summaries(scopes)
    rank_summary=scalar_repeat_summary(ranks,scopes,'kendall_tau_b','kendall_tau_b')
    regret_summary=[]
    for name in ('regret_raw','top1_exact_match','fractional_top_set_agreement','source_top_target_optimal_fraction'):
        regret_summary.extend(scalar_repeat_summary(regrets,scopes,name,name))
    for name,rows in [('repeat_intervals',repeats),('user_bootstrap',[*effects,*within]),('transfer_classification',[*classes,*userclasses]),
        ('tost_holm_primary',tost),('rank_agreement',ranks),('clear_reversal',reversals),('selection_regret',regrets),
        ('rank_repeat_summary',rank_summary),('regret_repeat_summary',regret_summary),
        ('audit/user_pairing_checks',pairing),('audit/bootstrap_inventory',boot_inventory),('audit/statistical_missing_cells',missing)]:
        write_csv(RUN/(name+'.csv'),rows)
    status=dict(scope='S3 R0 five seeds',stage_B_registry_ready=b,stage_C_registry_ready=c,primary_training_seeds=(list(range(2020,2025))if c else[2020,2021,2022]if b else[]),
        active_scopes=[{**s,'seeds':list(s['seeds'])}for s in scopes],bootstrap_coordinates=len(boot_inventory),
        missing_entries=len(missing),paired_user_checks=len(pairing),effect_rows=len(effects),rank_rows=len(ranks),regret_rows=len(regrets),
        all_primary_statistics_complete=bool((b or c)and not[x for x in missing if x.get('scope')=='main_R0'or x.get('status')=='PAIRING_REVIEW_REQUIRED']),
        statistical_scope='Complete R0 five-seed cohort only',model_training=False,model_inference=False)
    atomic_json(RUN/'w2w3_analysis_status.json',status);print(json.dumps(status),flush=True)

