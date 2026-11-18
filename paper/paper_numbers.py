"""Check submitted-manuscript claims from released scalar evidence, without fitting models."""
from pathlib import Path
from statistics import mean
from collections import Counter, defaultdict
from decimal import Decimal
import argparse
import ast
import csv
import json
import math
import sys

from paper_data import read, choose, one, build
from transfer_counts import load, classify, summarize, headroom, headroom_summary, SRC

REPO = Path(__file__).resolve().parents[1]


def collect_settings():
    """Inspect JSON and Python AST only. Do not import or instantiate a model."""
    vals={}
    def uniform(key, observed):
        observed=list(observed)
        if not observed:raise ValueError('No setting evidence '+key)
        unique={json.dumps(x,sort_keys=True) for x in observed}
        vals['setting.'+key]=observed[0] if len(unique)==1 else 'INCONSISTENT:'+str(sorted(unique))
    s1=[json.loads(p.read_text()) for p in sorted((REPO/'configs/s1').glob('*/*/seed_*.json'))]
    for i,label in enumerate(('train_pct','valid_pct','test_pct')):
        uniform('s1_split_'+label,(100*x['backbone_config']['eval_args']['split']['RS'][i] for x in s1))
    uniform('rating_threshold',(x['backbone_config']['threshold']['rating'] for x in s1))
    # The saved backbone config selects labeled AUC. The evaluation adapter
    # supplies the benchmark negative count explicitly at the real entrypoint.
    entry=ast.parse((REPO/'src/studies/s1/run.py').read_text())
    uniform('s1_negatives',(ast.literal_eval(k.value) for node in ast.walk(entry) if isinstance(node,ast.Call) for k in node.keywords if k.arg=='sample_num'))
    uniform('top_m',(x['npgc']['candidate_top_m'] for x in s1))
    uniform('metric_cutoff',(x['npgc']['cutoff_k'] for x in s1))
    uniform('npgc_k_yelp_ml',(x['npgc']['neighbor_k'] for x in s1 if x['dataset']!='amazon-books'))
    uniform('npgc_k_amazon',(x['npgc']['neighbor_k'] for x in s1 if x['dataset']=='amazon-books'))
    vals['setting.s1_seed_start']=min(x['seed'] for x in s1)
    vals['setting.s1_seed_end']=max(x['seed'] for x in s1)
    s2=[json.loads(p.read_text()) for p in (REPO/'configs/s2').glob('*/seed_*.json')]
    uniform('s2_dimensions',(x['profile_search']['embedding_dim'] for x in s2))
    vals['setting.s2_split_count']=len({x['seed'] for x in s2})
    s3=[json.loads(p.read_text()) for p in (REPO/'configs/s3/backbones').glob('*/*/seed_*.json')]
    uniform('s3_dimensions',(x['profile']['embedding_size'] for x in s3))
    uniform('s3_split_seed',(x['validation_pool_signature']['data_seed'] for x in s3))
    vals['setting.s3_split_count']=len({x['validation_pool_signature']['data_seed'] for x in s3})
    uniform('s3_shared_checkpoint',(int(x['W_F_checkpoint_shared']) for x in s3))
    uniform('fixed_validation_pools_per_dataset',(len({x['validation_W_candidate_sha256'] for x in s3 if x['dataset']==ds}) for ds in {x['dataset'] for x in s3}))
    uniform('s3_negatives',(x['validation_pool_signature']['negative_per_relevant'] for x in s3))
    uniform('s3_bpr_training_negatives',(1 if 'one eligible negative' in x['profile']['loss'] else -1 for x in s3 if x['backbone']=='MF_BPR'))
    plan=json.loads((REPO/'configs/s3/analysis_plan.json').read_text())
    for name,value in [('repeat_seeds',len(plan['training_seeds_final'])),('seed_start',min(plan['training_seeds_final'])),('seed_end',max(plan['training_seeds_final']))]:vals['setting.s3_'+name]=value
    vals['setting.repeat_ci_pct']=100*plan['effect_ci']
    vals['setting.equivalence_ci_pct']=100*plan['tost_unadjusted_ci']
    vals['setting.equivalence_alpha_pct']=100*(1-plan['tost_unadjusted_ci'])/2
    vals['setting.margin_points']=100*plan['primary_practical_delta_raw']
    vals['setting.sensitivity_margin_min']=100*min(plan['sensitivity_delta_raw'])
    vals['setting.sensitivity_margin_max']=100*max(plan['sensitivity_delta_raw'])
    vals['setting.reference_methods']=len(plan['method_sets']['REF4'])-1
    uniform('contrast_methods',(len(c.split('-')) for c in plan['contrasts']))
    uniform('contrast_protocols',(len(c) for c in SRC.values()))
    vals['setting.protocol_pairs']=len(set(SRC.values()))
    analysis_tree=ast.parse((REPO/'paper/transfer_counts.py').read_text())
    fractions=next(ast.literal_eval(n.iter) for n in ast.walk(analysis_tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='frac')
    vals['setting.headroom_margin_pct']=100*max(fractions)
    applied=[json.loads(p.read_text()) for p in (REPO/'configs/s3/applied').glob('*/*/seed_*/*.json')]
    residual=[v for x in applied for k,v in x.items() if k in ('S','P','R','PR','H','RH','PH','PRH')]
    for key,field in [('residual_lr','learning_rate'),('residual_weight_decay','weight_decay'),('residual_batch','batch_size'),('residual_epochs','maximum_epochs')]:
        uniform(key,(x[field] for x in residual))
    prm=[x['recorded_selection'] for x in (json.loads(p.read_text()) for p in (REPO/'configs/s3/prm_selection').glob('*/*/*.json'))]
    uniform('prm_batch',(x['prm_principled']['batch_size'] for x in applied))
    uniform('prm_selection_batch',(x['prm_principled']['batch_size'] for x in applied))
    uniform('prm_epochs',(x['prm_principled']['maximum_epochs'] for x in applied))
    uniform('prm_grid_min',(min(x['scope']['grid']) for x in prm))
    uniform('prm_grid_max',(max(x['scope']['grid']) for x in prm))
    uniform('prm_selection_seed',(x['selection_seed'] for x in prm))
    common=ast.parse((REPO/'src/studies/s3/cross_common.py').read_text())
    literals={node.targets[0].id:ast.literal_eval(node.value) for node in common.body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('MASKS','FEATURES','DATASETS','BACKBONES','PROTOCOLS')}
    vals['setting.mask_count']=len(literals['MASKS'])
    vals['setting.feature_groups']=len({'P','R','H'} & set(literals['MASKS']))
    vals['setting.graph_feature_groups']=len({'R','H'} & set(literals['MASKS']))
    vals['setting.s3_datasets']=len(literals['DATASETS'])
    vals['setting.s3_backbones']=len(literals['BACKBONES'])
    vals['setting.protocols']=len(literals['PROTOCOLS'])
    tree=ast.parse((REPO/'src/diagnostics/revision_evidence/common.py').read_text())
    model=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='JointResidualMLP')
    linear=[tuple(ast.literal_eval(a) for a in n.args[:2]) for n in ast.walk(model) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Linear']
    dims=[linear[0][0]]+[pair[1] for pair in linear]
    for i,value in enumerate(dims):vals[f'setting.mlp_dim_{i}']=value
    vals['setting.mlp_parameters']=sum(i*o+o for i,o in linear)
    vals['setting.mlp_layers']=len(linear)
    vals['setting.zero_output_initialization']=0 if sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='zeros_' for n in ast.walk(model))==2 else -1
    trainer=ast.parse((REPO/'src/studies/s3/uniform_models.py').read_text())
    vals['setting.residual_optimizer']='AdamW' if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='AdamW' for n in ast.walk(trainer)) else 'NOT_FOUND'
    master=ast.parse((REPO/'src/studies/s3/master.py').read_text())
    partition=next(n for n in master.body if isinstance(n,ast.FunctionDef) and n.name=='partition_indices')
    nb=next(n.value for n in ast.walk(partition) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='n_B' for t in n.targets))
    if not isinstance(nb,ast.BinOp) or not isinstance(nb.op,ast.FloorDiv):raise ValueError('Unexpected A/B partition expression')
    divisor=ast.literal_eval(nb.right)
    vals['setting.reranker_partition_pct']=100/divisor
    vals['setting.backbone_partition_pct']=100-100/divisor
    e3=read(REPO/'results/e3/E3_seed_results.csv')
    vals['setting.e3_repeat_seeds']=len({int(x['training_seed']) for x in e3})
    vals['setting.training_setups']=len({x['cohort'] for x in read(REPO/'results/e3/E3_all_conditions_seed_results.csv') if x['architecture']=='NFM'})
    vals['setting.e3_training_negatives']=1 if 'one uniform unobserved item per A positive per epoch' in json.loads((REPO/'configs/e3/fixed_settings.json').read_text()).get('fixed_rules',{}).get('added_negatives','') else -1
    return vals


def mostpop_accuracy_rule(results):
    """Recalculate the 36-cell rule from paired raw repeats and fixed-pool MostPop.

    Existing conditions use five training seeds. Added-negative BCE uses three.
    Whole-pool MostPop is one fixed candidate draw, not Top-50 Pop or a repeat.
    """
    raw=read(results/'e3/E3_all_conditions_seed_results.csv')
    popular=read(results/'checks/D_mostpop.csv');groups=defaultdict(list)
    for x in raw:
        if x['method'] in ('base','item_degree_ranking'):
            groups[x['cohort'],x['dataset'],x['protocol']].append(x)
    rows=[]
    for (cohort,ds,protocol),rr in sorted(groups.items()):
        seeds=sorted({int(x['training_seed']) for x in rr})
        expected=list(range(2020,2023 if cohort.startswith('E3_') else 2025))
        if seeds!=expected:raise ValueError('Incomplete declared training seeds '+cohort)
        architecture=rr[0]['architecture'];bb='MF_BPR' if architecture=='MF' else architecture
        pool=one(popular,study='S3',dataset=ds,backbone=bb,protocol=protocol,seed=2020)
        if pool['status']!='PASS':raise ValueError('MostPop source check did not pass')
        base=[100*float(one(rr,training_seed=seed,method='base')['ndcg@20']) for seed in seeds]
        pop=[100*float(one(rr,training_seed=seed,method='item_degree_ranking')['ndcg@20']) for seed in seeds]
        b,p,m=mean(base),mean(pop),100*float(pool['ndcg20'])
        rows.append(dict(cohort=cohort,dataset=ds,protocol=protocol,architecture=architecture,
            n_training_seeds=len(seeds),mostpop_candidate_seed=2020,mostpop_pool_repeats=1,
            base_points=b,same_pool_mostpop_points=m,top50_pop_points=p,pop_minus_base_points=p-b,
            base_below_mostpop=b<m,pop_helps=p>b,rule_agrees=(b<m)==(p>b),
            mostpop_source='checks/D_mostpop.csv',metric_source='e3/E3_all_conditions_seed_results.csv'))
    return rows


def matched_seed_records(results):
    """Use the formal main-regime values, retaining the original selection policy."""
    rows=[x for x in read(results/'s3/extracted_seed_results.csv')
          if x['regime']=='R0' and x['analysis_family']=='matched'
          and x['selection_policy'] in ('native','identity_inclusive')]
    indexed={}
    for x in rows:
        key=(x['dataset'],x['backbone'],x['protocol'],x['method'],int(x['training_seed']))
        if key in indexed:raise ValueError('Duplicate matched seed coordinate '+str(key))
        indexed[key]=x
    return indexed


def headroom_capture_ratios(results):
    """Mean of paired per-seed (Pop - Base) / (Oracle - Base), in percent.

    This is deliberately not the ratio of two already-averaged effects.
    Oracle, Base and Pop refer to the identical backbone candidate list.
    """
    rows=matched_seed_records(results);out=[]
    coords=sorted({key[:3] for key in rows})
    for dataset,backbone,protocol in coords:
        ratios=[]
        for seed in range(2020,2025):
            base=rows[dataset,backbone,protocol,'base',seed]
            pop=rows[dataset,backbone,protocol,'item_degree_ranking',seed]
            denominator=float(base['oracle_ndcg20'])-float(base['ndcg@20'])
            if denominator<=0:raise ValueError('Nonpositive oracle headroom')
            if abs(float(base['oracle_ndcg20'])-float(pop['oracle_ndcg20']))>1e-12:
                raise ValueError('Pop and Base oracle values differ')
            ratios.append(100*(float(pop['ndcg@20'])-float(base['ndcg@20']))/denominator)
        out.append(dict(dataset=dataset,backbone=backbone,protocol=protocol,
                        mean_per_seed_ratio_percent=mean(ratios),n_training_seeds=len(ratios),
                        ratios_by_seed_percent=dict(zip(range(2020,2025),ratios))))
    return out


def collect_claims(results):
    """Recalculate observations. Expected values reside only in the separate config."""
    r = results
    values = {}
    def put(key, value):
        if key in values:
            raise ValueError('Duplicate calculation ' + key)
        values[key] = value
    b = read(r/'s1/benchmark_seed_metrics.csv')
    groups = defaultdict(dict)
    for x in b:
        groups[x['dataset'], x['backbone'], x['seed']][x['method']] = x
    put('s1.coordinates', len(groups))
    put('s1.orderings',len({x['method'] for x in b}))
    for metric, short in [('ndcg@20', 'ndcg20'), ('recall@20', 'recall20')]:
        put('s1.pop_wins.'+short, sum(float(g['item_degree_ranking'][metric]) > max(float(g[z][metric]) for z in ('base', 'original_nonparamgc')) for g in groups.values()))
    for field, pos in [('datasets', 0), ('backbones', 1), ('splits', 2)]:
        put('s1.'+field, len({key[pos] for key in groups}))
    gains, ng, relative = defaultdict(list), defaultdict(list), defaultdict(list)
    for (ds, bb, seed), g in groups.items():
        a, n, p = [float(g[z]['ndcg@20']) for z in ('base', 'original_nonparamgc', 'item_degree_ranking')]
        gains[ds, bb].append(100*(p-a)); ng[ds, bb].append(100*(n-a)); relative[ds, bb].append((a,n))
    put('s1.pop_gain_min', min(map(mean, gains.values())))
    put('s1.pop_gain_max', max(map(mean, gains.values())))
    put('s1.npgc_gain_max', max(map(mean, ng.values())))
    ratios = [100*(mean(y for x,y in z)/mean(x for x,y in z)-1) for z in relative.values()]
    put('s1.npgc_relative_min', min(ratios)); put('s1.npgc_relative_max', max(ratios))

    fp = choose(read(r/'followup/S1_fullpool_seed_results.csv'), status='COMPLETE')
    fg = defaultdict(dict)
    for x in fp:
        key = (x['dataset'], x['backbone'], int(x['seed']))
        if x['method'] in fg[key]: raise ValueError('Duplicate full-pool method')
        fg[key][x['method']] = float(x['ndcg20'])
    if any(set(x) != {'Base','NPGC','Pop'} for x in fg.values()):
        raise ValueError('Incomplete full-pool methods')
    put('fullpool.runs', len(fg)); put('fullpool.pop_wins', sum(x['Pop']>x['NPGC'] for x in fg.values()))
    put('fullpool.backbones', len({k[1] for k in fg}))
    fm = defaultdict(lambda: defaultdict(list))
    for (ds,bb,seed), x in fg.items():
        for m,v in x.items():fm[ds,bb][m].append(v)
    rel = [100*(mean(x['NPGC'])/mean(x['Base'])-1) for x in fm.values()]
    diff = [100*(mean(x['Pop'])-mean(x['NPGC'])) for x in fm.values()]
    put('fullpool.npgc_relative_min', min(rel)); put('fullpool.npgc_relative_max', max(rel))
    put('fullpool.pop_npgc_min', min(diff)); put('fullpool.pop_npgc_max', max(diff))
    for ds in ('yelp2018','amazon-books','ml-1m'):
        x=one(read(r/'checks/D_mostpop_summary.csv'), Study='S1', Dataset=ds, Protocol='F')
        put('mostpop.catalog.'+ds, float(x['NDCG@20'].split()[0]))

    fresh=read(r/'s1/fresh_seed_metrics.csv'); fd={}; fb={}
    for ds in ('yelp2018','amazon-books','ml-1m'):
        for bb in ('NFM','DCNV2'):
            for protocol in ('U','W','F'):
                rr=choose(fresh, dataset=ds, backbone=bb, protocol=protocol)
                aa=mean(float(one(rr, seed=seed, method='base')['ndcg@20']) for seed in (2020,2021,2022))
                pp=mean(float(one(rr, seed=seed, method='item_degree_ranking')['ndcg@20']) for seed in (2020,2021,2022))
                fd[ds,bb,protocol]=100*(pp-aa);fb[ds,bb,protocol]=100*aa
    put('fresh.yelp_u_mean', mean(fd['yelp2018',bb,'U'] for bb in ('NFM','DCNV2')))
    put('fresh.amazon_w_min', min(fd['amazon-books',bb,'W'] for bb in ('NFM','DCNV2')))
    put('fresh.amazon_w_max', max(fd['amazon-books',bb,'W'] for bb in ('NFM','DCNV2')))
    put('fresh.yelp_amazon_f_base_max', max(fb[ds,bb,'F'] for ds in ('yelp2018','amazon-books') for bb in ('NFM','DCNV2')))
    put('fresh.ml_f_pop_gain', mean(fd['ml-1m',bb,'F'] for bb in ('NFM','DCNV2')))
    put('fresh.ml_f_fraction_max', max(fd['ml-1m',bb,'F']/fd['ml-1m',bb,'U'] for bb in ('NFM','DCNV2')))
    descriptive={'ml_w_over_u_gain_ratio': {bb:fd['ml-1m',bb,'W']/fd['ml-1m',bb,'U'] for bb in ('NFM','DCNV2')}}

    dsp=read(r/'followup/DF_s1_paired.csv')
    for ds in ('yelp2018','ml-1m'):
        xx=choose(dsp, dataset=ds, contrast='Pop-Base', metric='ndcg@20')
        put('df.s1.'+ds+'.min', min(float(x['mean_x100']) for x in xx))
        put('df.s1.'+ds+'.max', max(float(x['mean_x100']) for x in xx))
        put('df.s1.'+ds+'.backbones',len({x['backbone'] for x in xx}))
    put('df.s1.npgc_abs_max', max(abs(float(x['mean_x100'])) for x in choose(dsp, contrast='NPGC-Base', metric='ndcg@20')))
    ep=read(r/'e3/E3_pop_minus_base_paired.csv')
    put('setting.metric_scale',max(float(x['mean_x100'])/float(x['mean_raw']) for x in ep if float(x['mean_raw'])!=0))
    put('setting.objective_types',len({x['objective'] for x in read(r/'followup/E_summary.csv')}))
    for x in choose(ep, contrast='Pop-Base', metric='ndcg@20', protocol='U'):
        put('objective.'+x['dataset']+'.'+x['cohort'], float(x['mean_x100']))
    put('abstract.nfm_bce_pop', values['objective.yelp2018.S3_NFM_BCE'])
    put('abstract.nfm_bce_negative_pop', values['objective.yelp2018.E3_NFM_BCE_augmented'])
    rule=mostpop_accuracy_rule(r); exceptions=[x for x in rule if not x['rule_agrees']]
    put('accuracy_rule.combinations',len(rule));put('accuracy_rule.exceptions',len(exceptions))
    put('accuracy_rule.agreements',sum(x['rule_agrees'] for x in rule))
    y=one(rule,dataset='yelp2018',cohort='E2_MF_BCE',protocol='W')
    for key,field in [('effect','pop_minus_base_points'),('base','base_points'),('mostpop','same_pool_mostpop_points')]:
        put('objective.yelp_mf_bce_w.'+key,y[field])
    put('accuracy_rule.exceptions_inside_margin',sum(abs(x['pop_minus_base_points'])<.25 for x in exceptions))
    put('accuracy_rule.exceptions_outside_margin',sum(abs(x['pop_minus_base_points'])>=.25 for x in exceptions))
    put('accuracy_rule.exceptions_yelp_f',sum(x['dataset']=='yelp2018' and x['protocol']=='F' for x in exceptions))
    outside=[x['pop_minus_base_points'] for x in exceptions if abs(x['pop_minus_base_points'])>=.25]
    put('accuracy_rule.exception_outside_effect',outside[0] if len(outside)==1 else float('nan'))

    run=str(r/'s3');rows=load(run,'ndcg@20','identity_inclusive','0.0025')
    cls=lambda x:classify(x,.0025,.0025)
    put('transfer.matched',sum(x['analysis_family']=='matched' for x in rows))
    put('transfer.frozen',sum(x['analysis_family']=='frozen' for x in rows))
    put('transfer.matched_reversals',sum(x['analysis_family']=='matched' and cls(x)=='R' for x in rows))
    put('transfer.matched_transfers',sum(x['analysis_family']=='matched' and cls(x)=='T' for x in rows))
    put('transfer.comparisons',len(rows));put('transfer.cells',len({(x['dataset'],x['backbone']) for x in rows}))
    put('transfer.registered_contrasts',len({x['contrast'] for x in rows}))
    put('transfer.frozen_contrasts',len({x['contrast'] for x in rows if x['analysis_family']=='frozen'}))
    ft=[x for x in rows if x['comparison']!='U-W']
    put('transfer.f_total',len(ft))
    put('transfer.f_transfers',sum(cls(x)=='T' for x in ft))
    put('transfer.yelp_nfm_f_total',sum((x['dataset'],x['backbone'])==('yelp2018','NFM') for x in ft))
    put('transfer.yelp_mf_f_vanish',sum((x['dataset'],x['backbone'])==('yelp2018','MF_BPR') and cls(x)=='V' for x in ft))
    put('transfer.ml_f_total',sum(x['dataset']=='ml-1m' for x in ft))
    put('transfer.ml_f_transfer',sum(x['dataset']=='ml-1m' and cls(x)=='T' for x in ft))
    put('transfer.yelp_f_vanish',sum(x['dataset']=='yelp2018' and cls(x)=='V' for x in ft))
    put('transfer.yelp_uw_vanish',sum(x['dataset']=='yelp2018' and x['comparison']=='U-W' and cls(x)=='V' for x in rows))
    nf=[x for x in ft if (x['dataset'],x['backbone'])!=('yelp2018','NFM')]
    put('transfer.nofloor_f_total',len(nf));put('transfer.nofloor_f_transfer',sum(cls(x)=='T' for x in nf))
    put('transfer.reversals',sum(cls(x)=='R' for x in rows))
    put('transfer.user_bootstrap_agreement',sum(x['user_seed_agrees_with_D1']=='5' for x in rows))
    disagreement=[x for x in rows if x['user_seed_agrees_with_D1']!='5']
    put('transfer.user_disagreements',len(disagreement))
    for label in ('E','X'):
        put('transfer.user_disagreements_'+label,sum(cls(x)==label for x in disagreement))
    put('transfer.user_disagreements_other',sum(cls(x) not in ('E','X') for x in disagreement))
    xx=load(run,'ndcg@20','identity_inclusive','0.001')
    put('transfer.yelp_delta01_f_transfer',sum(x['dataset']=='yelp2018' and x['comparison']!='U-W' and classify(x,.001,.001)=='T' for x in xx))
    put('sensitivity.delta01_reversals',sum(classify(x,.001,.001)=='R' for x in xx))
    xx=load(run,'ndcg@20','identity_inclusive','0.005')
    put('sensitivity.delta05_reversals',sum(classify(x,.005,.005)=='R' for x in xx))
    xx=load(run,'ndcg@20','trained_only','0.0025')
    put('sensitivity.trained_only_reversals',sum(classify(x,.0025,.0025)=='R' for x in xx))
    h=headroom(run)
    _,_,hft=headroom_summary(run,rows,.02)
    for ds in ('ml-1m','yelp2018'):put('headroom02.'+ds+'.f_transfer',hft[ds]['T'])
    for fraction,name in [(.01,'01'),(.02,'02')]:
        rr=[x for x in rows if classify(x,fraction*h(x['dataset'],x['backbone'],SRC[x['comparison']][0]),fraction*h(x['dataset'],x['backbone'],SRC[x['comparison']][1]))=='R']
        put('sensitivity.headroom'+name+'_reversals',len(rr))
        if name=='01':put('sensitivity.headroom01_additional_yelp_nfm',sum((x['dataset'],x['backbone'])==('yelp2018','NFM') for x in rr))
    rc=load(run,'recall@20','identity_inclusive','0.0025')
    rr=[x for x in rc if x['analysis_family']=='matched' and cls(x)=='R']
    put('recall.matched_reversals',len(rr));put('recall.yelp_reversals',sum(x['dataset']=='yelp2018' for x in rr))
    put('recall.yelp_pop_reversals',sum(x['dataset']=='yelp2018' and 'Pop' in x['contrast'].split('-') for x in rr))
    changes=[100*abs(float(x['target_effect_mean_raw'])-float(x['source_effect_mean_raw'])) for x in rows if x['comparison']=='U-F' and x['contrast']=='Pop-Base']
    put('transfer.pop_uf_change_min',min(changes));put('transfer.pop_uf_change_max',max(changes))
    tost=choose(read(r/'s3/tost_holm_primary.csv'),registered_primary_family='True')
    put('tost.tests',len(tost));put('tost.equivalent',sum(x['equivalent_holm']=='True' for x in tost))
    put('tost.beyond_margin',sum(float(x['ci95_low_raw'])>.0025 or float(x['ci95_high_raw'])<-.0025 for x in tost))

    summ=[x for x in read(r/'s3/homogeneous_method_summary.csv') if x['scope']=='main_R0' and x['selection_policy'] in ('native','identity_inclusive')]
    def sm(ds,bb,protocol,method,metric='ndcg@20',family='matched'):
        return float(one(summ,dataset=ds,backbone=bb,protocol=protocol,method=method,metric=metric,analysis_family=family)['mean_x100'])
    reference_methods=('base','item_degree_ranking','original_nonparamgc','prm_principled')
    leader_agreement=[]
    for ds in ('yelp2018','ml-1m'):
        for bb in ('NFM','MF_BPR'):
            leaders=[max(reference_methods,key=lambda m:sm(ds,bb,protocol,m)) for protocol in ('U','F')]
            leader_agreement.append(leaders[0]==leaders[1])
    put('leaders.uf_same_cells',sum(leader_agreement))
    capture=headroom_capture_ratios(r)
    for x in capture:put('headroom_ratio.'+'.'.join(x[k] for k in ('dataset','backbone','protocol')),x['mean_per_seed_ratio_percent'])
    descriptive['headroom_capture_ratios']=capture
    matched=matched_seed_records(r)
    near_ties=[abs(float(matched['yelp2018','NFM','U','P',seed]['ndcg@20'])-float(matched['yelp2018','NFM','U','item_degree_ranking',seed]['ndcg@20'])) for seed in range(2020,2025)]
    put('caption.yelp_nfm_u_resp_pop_max_abs_raw',max(near_ties))
    put('caption.yelp_nfm_u_resp_pop_seeds',len(near_ties))
    descriptive['caption_near_tie']=[dict(training_seed=seed,absolute_ndcg20_difference_raw=value,absolute_ndcg20_difference_points=100*value) for seed,value in zip(range(2020,2025),near_ties)]
    put('s3.yelp_nfm_f_coverage',sm('yelp2018','NFM','F','base','candidate_recall50'))
    base=sm('yelp2018','MF_BPR','F','base');gap=sm('yelp2018','MF_BPR','F','base','oracle_ndcg20')-base
    put('s3.yelp_mf_f_headroom',gap)
    best=max(float(x['mean_x100']) for x in choose(summ,dataset='yelp2018',backbone='MF_BPR',protocol='F',metric='ndcg@20',analysis_family='matched'))
    put('s3.yelp_mf_f_max_capture',100*(best-base)/gap)
    put('frozen.yelp_mf_scoreonly_gap',sm('yelp2018','MF_BPR','W_to_F','S',family='frozen')-base)
    put('frozen.yelp_mf_prm_abs_gap',abs(sm('yelp2018','MF_BPR','W_to_F','prm_principled',family='frozen')-base))
    dp=one(read(r/'followup/DF_s3_paired.csv'),dataset='yelp2018',contrast='Pop-Base')
    put('df.s3.yelp_pop_gain',float(dp['mean_x100']))
    for contrast in ('Pop-Base','Pop-NPGC','NPGC-Base'):
        x=one(read(r/'followup/DF_s3_transfer.csv'),dataset='yelp2018',backbone='NFM',source_protocol='U',contrast=contrast)
        put('df.s3.yelp_u.'+contrast,x['classification'])
    pp=one(read(r/'followup/P0_extended_counts.csv'),contrast_set='all_pairs15',subset='all')
    displayed_methods={'base','item_degree_ranking','original_nonparamgc','prm_principled','P','H'}
    available={x['method'] for x in summ} & displayed_methods
    put('allpairs.method_pairs',math.comb(len(available),2));put('allpairs.reversals',int(pp['R']))
    rev=read(r/'followup/P0_all_pairs_reversals.csv')
    put('allpairs.matched_reversals',sum(x['analysis_family']=='matched' for x in rev))
    put('allpairs.yelp_mf_wf_residual_reversals',sum(x['dataset']=='yelp2018' and x['backbone']=='MF_BPR' and x['analysis_family']=='frozen' and x['comparison']=='W-W_to_F' and 'Res(p)' in x['contrast'] for x in rev))
    x=one(read(r/'followup/DF_frozen_learned.csv'),dataset='yelp2018',backbone='NFM',source_protocol='W',contrast='P-Base')
    put('frozen.yelp_nfm_wdf_resp',float(x['target_effect_mean_x100']))
    x=one(read(r/'followup/DF_frozen_learned.csv'),dataset='yelp2018',backbone='NFM',source_protocol='W',contrast='Res(p)-Pop')
    for name,field in [('source','source_effect_mean_x100'),('target','target_effect_mean_x100'),('low','target_ci95_low_x100'),('high','target_ci95_high_x100')]:
        put('frozen.yelp_nfm_wdf_resp_pop.'+name,float(x[field]))
    put('frozen.yelp_nfm_wdf_resp_pop.classification',x['classification'])
    x=one(read(r/'followup/DF_frozen_learned.csv'),dataset='yelp2018',backbone='NFM',source_protocol='U',contrast='Res(p)-Pop')
    put('frozen.yelp_nfm_udf_resp_pop',float(x['target_effect_mean_x100']))
    paired=read(r/'checks/C_paired_repeat_ci.csv')
    for key,con,fit,ev,field in [('resp_ww_base','Res(p)-Base','W','W','mean_x100'),('resp_wf_base','Res(p)-Base','W','F','mean_x100'),('resp_ww_pop','Res(p)-Pop','W','W','mean_x100'),('resp_wf_pop','Res(p)-Pop','W','F','mean_x100'),('resp_wf_pop_ci_low','Res(p)-Pop','W','F','ci95_low_x100'),('resp_wf_pop_ci_high','Res(p)-Pop','W','F','ci95_high_x100')]:
        put('frozen.'+key,float(one(paired,dataset='yelp2018',backbone='MF_BPR',contrast=con,fit_protocol=fit,eval_protocol=ev)[field]))
    raw=read(r/'s3/extracted_seed_results.csv')
    fits=[x for x in raw if x['backbone']=='MF_BPR' and x['regime']=='R0' and x['analysis_family']=='matched' and x['selection_policy']=='identity_inclusive' and x['method'] in ('S','P','R','PR','H','RH','PH','PRH')]
    put('residual.mf_fits',len(fits));put('residual.mf_identity',sum(x['identity_selected']=='True' for x in fits))
    pd=one(read(r/'checks/C_partial_dependence_repeat_summary.csv'),dataset='yelp2018',backbone='MF_BPR',fit_protocol='W',eval_role='fit_test',eval_protocol='W')
    put('residual.yelp_mf_w_negative_pd',int(pd['negative_PD_seeds']))
    draw=read(r/'followup/P4_seed_results.csv');dt=read(r/'followup/P4_transfer.csv')
    put('p4.draws',len({x['draw_seed'] for x in draw}));put('p4.decisions',len(dt))
    put('p4.unchanged',sum(x['classification']==x['training_repeat_classification'] and x['classification_changed']=='False' for x in dt))
    eq=read(r/'checks/A_npgc_equivalence.csv');put('official.audit_cells',len(eq))
    put('official.max_error',max(float(x['max_abs_score_diff']) for x in eq))
    # Fresh-pool contrasts use means of three paired split differences.
    fresh_effects=defaultdict(list)
    for ds in ('yelp2018','amazon-books','ml-1m'):
        for bb in ('NFM','DCNV2'):
            for protocol in ('U','W'):
                rr=choose(fresh,dataset=ds,backbone=bb,protocol=protocol)
                fresh_effects[protocol].append(mean(100*(float(one(rr,seed=seed,method='original_nonparamgc')['ndcg@20'])-float(one(rr,seed=seed,method='base')['ndcg@20'])) for seed in (2020,2021,2022)))
    for protocol in ('U','W'):
        put('fresh.npgc_'+protocol.lower()+'_min',min(fresh_effects[protocol]));put('fresh.npgc_'+protocol.lower()+'_max',max(fresh_effects[protocol]))
    put('fresh.ml_w_min',min(fd['ml-1m',bb,'W'] for bb in ('NFM','DCNV2')))
    put('fresh.ml_w_max',max(fd['ml-1m',bb,'W'] for bb in ('NFM','DCNV2')))
    put('fresh.yelp_w_abs_max',max(abs(fd['yelp2018',bb,'W']) for bb in ('NFM','DCNV2')))
    put('s3.yelp_nfm_f_headroom',sm('yelp2018','NFM','F','base','oracle_ndcg20')-sm('yelp2018','NFM','F','base'))
    put('s3.yelp_mf_f_coverage',sm('yelp2018','MF_BPR','F','base','candidate_recall50'))
    put('s3.mf_npgc_below_base_cells',sum(sm(ds,'MF_BPR',protocol,'original_nonparamgc')<sm(ds,'MF_BPR',protocol,'base') for ds in ('yelp2018','ml-1m') for protocol in ('U','W','F')))
    put('transfer.f_matched',sum(x['analysis_family']=='matched' for x in ft))
    put('transfer.f_frozen',sum(x['analysis_family']=='frozen' for x in ft))
    put('transfer.yelp_f_transfer',sum(x['dataset']=='yelp2018' and cls(x)=='T' for x in ft))
    dfraw=choose(read(r/'followup/DF_s3_seed_results.csv'),dataset='yelp2018',backbone='NFM')
    base_df=mean(float(x['ndcg@20']) for x in choose(dfraw,method='base'))
    put('df.s3.yelp_npgc_abs_gap',100*abs(mean(float(x['ndcg@20']) for x in choose(dfraw,method='original_nonparamgc'))-base_df))
    put('df.s3.yelp_base_below_retriever',100*(mean(float(x['ndcg@20']) for x in choose(dfraw,method='retriever'))-base_df))
    for name,field in [('resp_wf_base_ci_low','ci95_low_x100'),('resp_wf_base_ci_high','ci95_high_x100')]:
        put('frozen.'+name,float(one(paired,dataset='yelp2018',backbone='MF_BPR',contrast='Res(p)-Base',fit_protocol='W',eval_protocol='F')[field]))
    put('frozen.resp_ff_base',sm('yelp2018','MF_BPR','F','P')-base)
    n1=choose(read(r/'e3/N1_npgc_popularity_summary.csv'),study='S3',status='COMPLETE')
    rho=[float(x['rho_delta_score_popularity_mean']) for x in n1]
    put('n1.s3_cells',len(n1));put('n1.positive_adjustment_popularity_cells',sum(x>0 for x in rho))
    put('n1.rho_min',min(rho));put('n1.rho_max',max(rho))
    put('figure2.max_objective_ci_halfwidth',max((float(x['ci95_high_x100'])-float(x['ci95_low_x100']))/2 for x in choose(ep,contrast='Pop-Base',metric='ndcg@20')))
    e3u=[x for x in choose(ep,contrast='Pop-Base',metric='ndcg@20',protocol='U') if x['cohort'].startswith('E3_')]
    put('e3.u_negative_mean_cells',sum(float(x['mean_raw'])<0 for x in e3u))
    put('e3.u_negative_ci_cells',sum(float(x['ci95_high_raw'])<0 for x in e3u))
    return values,rule,descriptive


def comparison(actual, expected, kind, operator):
    if kind=='label':return str(actual)==str(expected),0
    try:a=float(actual);e=float(expected)
    except (ValueError,TypeError):return False,0
    # Decimal('13') has exponent0, so rounded whole-point metrics allow0.5.
    tolerance=0 if kind in ('count','setting') else float(Decimal(5).scaleb(Decimal(str(expected)).as_tuple().exponent-1))
    if not math.isfinite(a):return False,tolerance
    if operator=='eq':ok=abs(a-e)<=tolerance+1e-12
    elif operator=='le':ok=a<=e+tolerance+1e-12
    elif operator=='lt':ok=a<e  # Strictly stated bounds are not weakened by display rounding.
    else:raise ValueError('Unknown comparison '+operator)
    return ok,tolerance


def numeric_coverage(expected, checks, out):
    """Account for every recorded prose/caption token without inventing checks."""
    inventory=json.loads((REPO/'configs/manuscript_numeric_inventory.json').read_text())
    if inventory['manuscript_sha256']!=expected['manuscript_sha256']:
        raise ValueError('Numeric inventory and expectations refer to different manuscripts')
    indexed={x['item']:x for x in checks};rows=[]
    for token in inventory['entries']:
        category=token['category'];ids=token['claim_ids']
        if category in ('CHECKED_LOCAL_CLAIM','CHECKED_TABLE_VALUE','CHECKED_TABLE_TEXT'):
            if not ids or any(i not in indexed for i in ids):
                raise ValueError('Missing computation for numeric token '+token['token_id'])
            status='PASS' if all(indexed[i]['status']=='PASS' for i in ids) else 'MISMATCH'
        elif category=='UNMAPPED':
            raise ValueError('Unmapped manuscript numeric token '+token['token_id'])
        else:
            status=category  # A definition/reference is not a computational PASS.
        rows.append(dict(token_id=token['token_id'],location=token['location'],section=token['section'],
                         document_part=token.get('document_part','table_value'),token=token['token'],
                         category=category,status=status,claim_ids=';'.join(ids),evidence=';'.join(token['evidence'])))
    if len({x['token_id'] for x in rows})!=len(rows):raise ValueError('Duplicate inventory token ID')
    counts=dict(Counter(x['category'] for x in rows))
    if len(rows)!=inventory['token_count'] or counts!=inventory['category_counts']:
        raise ValueError('Numeric inventory totals differ')
    with (out/'MANUSCRIPT_NUMERIC_COVERAGE.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(token_count=len(rows),category_counts=counts,status_counts=dict(Counter(x['status'] for x in rows)),
                 unmapped_tokens=0,manuscript_text_included=False,
                 scope=inventory['scope'],
                 external_reference_claims=expected['external_reference_claims'],
                 qualitative_approximations=expected['descriptive_approximations'])
    (out/'MANUSCRIPT_NUMERIC_COVERAGE.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,default=Path('results'))
    ap.add_argument('--out',type=Path,default=Path('outputs/PAPER_NUMBER_CHECK.md'))
    args=ap.parse_args()
    expected=json.loads((REPO/'configs/analysis_expected.json').read_text())
    actual,rule,descriptive=collect_claims(args.results)
    actual.update(collect_settings())
    checks=[]
    def check(name,a,e,kind='display_metric',op='eq',note='',source=''):
        ok,tol=comparison(a,e,kind,op)
        checks.append(dict(item=name,expected=str(e),actual=a,status='PASS' if ok else 'MISMATCH',kind=kind,operator=op,tolerance=tol,evidence=source,note=note))
    for claim in expected['body_claims']:
        if claim['id'] not in actual:raise KeyError('No calculation for manuscript claim '+claim['id'])
        check(claim['id'],actual[claim['id']],claim['expected'],claim['kind'],claim['operator'],claim['note'],claim['evidence'])
    tables=build(args.results)
    for n,tab in expected['tables'].items():
        observed=tables[int(n)];check(f'Table {n} rows',len(observed),len(tab),'count')
        for i,(er,ar) in enumerate(zip(tab,observed),1):
            if 'expected_text_cells' in er:
                check(f'Table {n} row {i} label',ar['label'],er['label'],'label',source=f'table:{n}/row:{i}')
                check(f'Table {n} row {i} text cells',len(ar['cells']),len(er['expected_text_cells']),'count')
                for j,(ac,ec) in enumerate(zip(ar['cells'],er['expected_text_cells']),1):
                    check(f'Table {n} row {i} text {j}',ac['values'][0],ec,'label',source=f'table:{n}/row:{i}')
            vals=[]
            for cell in ar['cells']:
                vals.extend(map(int,cell['values'][0].split('/')) if cell['kind']=='count' else cell['values'] if cell['kind']!='text' else [])
            ee=er['expected_numeric_tokens'];check(f'Table {n} row {i} numeric cells',len(vals),len(ee),'count')
            for j,(av,ev) in enumerate(zip(vals,ee),1):
                check(f'Table {n} row {i} value {j}',av,ev,'display_metric',source=f'table:{n}/row:{i}')
    bad=[x for x in checks if x['status']!='PASS']
    args.out.parent.mkdir(parents=True,exist_ok=True)
    coverage=numeric_coverage(expected,checks,args.out.parent)
    args.out.with_suffix('.json').write_text(json.dumps(checks,indent=2)+'\n')
    (args.out.parent/'NUMERICAL_SUPPORTING_DETAILS.json').write_text(json.dumps(descriptive,indent=2)+'\n')
    rule_path=args.out.parent/'MOSTPOP_ACCURACY_RULE.csv'
    with rule_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rule[0]));writer.writeheader();writer.writerows(rule)
    lines=['# Submitted manuscript number check','',f'{len(checks)-len(bad)}/{len(checks)} computational checks passed, with {len(bad)} mismatches.','',
           'Only released scalar evidence and existing settings are read. No training, inference, model selection, or evidence alteration is performed. Displayed metric values use half a unit at the final displayed decimal place. Integer counts and category labels are exact.','',
           'Expectations were transcribed from the submitted manuscript. All numeric cells in Tables 1–4 are checked. The 36-cell accuracy rule uses five training seeds for the four historical conditions and three for the two added-negative BCE conditions. It compares Base to MostPop over the same pre-Top-50 pool, separately from the popularity reorder of the backbone Top-50.','',
           '## Explicitly separate verification limits','',
           'Externally cited original-paper values are catalogued as EXTERNAL_REFERENCE_NOT_RECALCULATED. They are not counted as computational PASS results. Qualitative and structural statements are not counted as numerical PASS results.','',
           'Exact ML-1M W/U popularity-gain ratios: '+json.dumps(descriptive['ml_w_over_u_gain_ratio'])+'.','',
           'The Table 4 caption diagnostic compares all five paired seed metrics. Its threshold is 0.0003 raw NDCG@20, equivalent to 0.03 points on the 100-point scale. Near-equal scalar metrics do not establish identical item ordering. Candidate-level ordering equality is not checked. Individual seed differences and the per-seed headroom ratios are saved in NUMERICAL_SUPPORTING_DETAILS.json.','',
           'The submitted manuscript itself is intentionally not included. The numerical-token inventory records locations, categories, and evidence links without copying the manuscript.','']
    lines += [f"Numeric inventory: {coverage['token_count']} tokens, zero unmapped. Categories: "+json.dumps(coverage['category_counts'])+'.',
              'Coverage is not a second set of independent statistical tests. Identifiers, structural statements, mathematical definitions, cited external numbers and qualitative approximations are labelled separately. See MANUSCRIPT_NUMERIC_COVERAGE.csv and .json.','']
    if bad:
        lines += ['## Mismatches','','| Item | Expected | Recomputed | Evidence |','|---|---:|---:|---|']
        lines += [f"| {x['item']} | {x['expected']} | {x['actual']} | {x['evidence']} |" for x in bad]
    lines += ['','## All computational checks','','| Item | Expected | Recomputed | Status |','|---|---:|---:|---|']
    lines += [f"| {x['item']} | {x['expected']} | {x['actual']:.12g} | {x['status']} |" if isinstance(x['actual'],(int,float)) else f"| {x['item']} | {x['expected']} | {x['actual']} | {x['status']} |" for x in checks]
    args.out.write_text('\n'.join(lines)+'\n')
    for x in bad:print('MISMATCH',x['item'],x['actual'],'expected',x['expected'])
    print(f'{len(checks)-len(bad)}/{len(checks)} passed. Report: {args.out}')
    print(f"Numeric inventory: {coverage['token_count']} tokens, {coverage['unmapped_tokens']} unmapped.")
    print('External-reference and qualitative-approximation entries are disclosed separately and are not counted as PASS.')
    return 1 if bad else 0


if __name__=='__main__':
    sys.exit(main())
