"""Original paired training-repeat intervals and expanded contrast aggregation."""
import csv,json,itertools,importlib.util
from pathlib import Path
from collections import Counter
import numpy as np
from .paths import OUT,ROOT,EVIDENCE,input_file
from studies.s3 import w2w3_statistics as STAT
_spec=importlib.util.spec_from_file_location('_public_transfer_counts',ROOT/'paper/transfer_counts.py')
PAPER=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(PAPER)
SEEDS=(2020,2021,2022,2023,2024)
DATASETS=('yelp2018','ml-1m');BACKBONES=('NFM','MF_BPR');PROTOCOLS=('U','W','F')
FIXED=('base','item_degree_ranking','original_nonparamgc')
LABELS=dict(base='Base',item_degree_ranking='Pop',original_nonparamgc='NPGC',prm_principled='PRM+PL',P='Res(p)',H='Res(H)')
DELTA=.0025

def read(path):
    with Path(path).open(newline='') as f:
        return [dict(r,_source_path='artifact:results/'+Path(path).name,_source_line=i) for i,r in enumerate(csv.DictReader(f),2)]

def write(name,rows,empty_fields=None):
    rows=list(rows);path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r)) or empty_fields or ['status']
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def run():
    path=input_file('s3_seed_results',EVIDENCE/'s3/extracted_seed_results.csv')
    return run_p0(s3_index(read(path)))
def summarize(values, level=.95):
    return STAT.repeat_interval(values, level)


def transfer(source, target, meta, seeds=SEEDS):
    sa, ta = summarize(source), summarize(target)
    s9, t9 = summarize(source, .9), summarize(target, .9)
    row = dict(**meta, metric='ndcg@20', n_training_seeds=len(seeds),
               training_seeds=';'.join(map(str, seeds)), status='COMPLETE' if tuple(seeds)==SEEDS else f'PARTIAL(n={len(seeds)})',
               practical_delta_raw=DELTA, practical_delta_points=100*DELTA,
               source_effect_mean_raw=sa['mean_raw'], target_effect_mean_raw=ta['mean_raw'],
               source_sample_sd_raw=sa['sample_sd_raw'], target_sample_sd_raw=ta['sample_sd_raw'],
               source_ci95_low=sa['ci_low_raw'], source_ci95_high=sa['ci_high_raw'],
               target_ci95_low=ta['ci_low_raw'], target_ci95_high=ta['ci_high_raw'],
               source_ci90_low=s9['ci_low_raw'], source_ci90_high=s9['ci_high_raw'],
               target_ci90_low=t9['ci_low_raw'], target_ci90_high=t9['ci_high_raw'],
               uncertainty_unit='paired_training_repeats_fixed_split_and_candidate_draw',
               source_effects_by_seed_raw=json.dumps(dict(zip(map(str,seeds),map(float,source)))),
               target_effects_by_seed_raw=json.dumps(dict(zip(map(str,seeds),map(float,target)))))
    for key, value in list(row.items()):
        if key.endswith('_raw') and isinstance(value,(int,float)):
            row[key[:-4]+'_x100'] = 100*value
        elif ('_ci95_' in key or '_ci90_' in key) and value is not None:
            row[key+'_x100'] = 100*value
    if len(seeds)<2:
        row.update(classification='NOT_ESTIMABLE', classification_before_vanish='NOT_ESTIMABLE')
        return row
    original = STAT.classify((row['source_ci95_low'],row['source_ci95_high']),
                            (row['target_ci95_low'],row['target_ci95_high']),
                            (row['source_ci90_low'],row['source_ci90_high']),
                            (row['target_ci90_low'],row['target_ci90_high']), DELTA)
    displayed = PAPER.classify(row,DELTA,DELTA).replace('U','X')
    assert (original=='unresolved' and displayed in ('V','X')) or displayed==dict(transfer='T',reverse='R',equivalent='E').get(original)
    row.update(classification=displayed, classification_before_vanish=original,
               point_mean_sign_flip=bool(sa['mean_raw']*ta['mean_raw']<0))
    return row


def s3_index(rows):
    result = {}
    for row in rows:
        if row['regime']!='R0' or int(row['training_seed']) not in SEEDS:
            continue
        policy = 'identity_inclusive' if row['method'] in STAT.RESIDUALS else 'native'
        if row['selection_policy']!=policy:
            continue
        key = (row['dataset'],row['backbone'],int(row['training_seed']),row['analysis_family'],row['protocol'],row['method'])
        if key in result:
            raise ValueError('Duplicate recorded coordinate '+str(key))
        result[key] = row
    if len(result)!=1080:
        raise ValueError(f'Expected 720 matched plus360 learned frozen coordinates, found{len(result)}')
    return result


def run_p0(index):
    registered = dict(STAT.CONTRASTS)
    extended = {**registered, 'Res(p)-Base':('P','base'),'Res(H)-Base':('H','base')}
    all_pairs = {}
    methods = ('base','item_degree_ranking','original_nonparamgc','prm_principled','P','H')
    for first,second in itertools.combinations(methods,2):
        existing = [(name,pair) for name,pair in extended.items() if set(pair)=={first,second}]
        if existing:
            name,pair=existing[0]
        else:
            pair=(second,first); name=LABELS[second]+'-'+LABELS[first]
        all_pairs[name]=pair
    assert len(all_pairs)==15

    def get(d,b,s,family,protocol,method):
        if family=='frozen' and method in FIXED:
            family,protocol='matched','F'
        return index[d,b,s,family,protocol,method]

    detailed=[]
    for cohort,contrasts in [('registered8',registered),('extended10',extended),('all_pairs15',all_pairs)]:
      for d,b in itertools.product(DATASETS,BACKBONES):
       for family,pairs in [('matched',[('U','W'),('U','F'),('W','F')]),('frozen',[('U','U_to_F'),('W','W_to_F')])]:
        for source,target in pairs:
         for name,(left,right) in contrasts.items():
            if family=='frozen' and left in FIXED and right in FIXED:
                continue
            a=[];t=[];source_refs=[];target_refs=[]
            for seed in SEEDS:
                sl=get(d,b,seed,'matched',source,left);sr=get(d,b,seed,'matched',source,right)
                tl=get(d,b,seed,family,target,left);tr=get(d,b,seed,family,target,right)
                a.append(float(sl['ndcg@20'])-float(sr['ndcg@20']))
                t.append(float(tl['ndcg@20'])-float(tr['ndcg@20']))
                source_refs.extend(f"{x['_source_path']}:{x['_source_line']}"for x in(sl,sr))
                target_refs.extend(f"{x['_source_path']}:{x['_source_line']}"for x in(tl,tr))
            row=transfer(a,t,dict(contrast_set=cohort,dataset=d,backbone=b,analysis_family=family,
                comparison=source+'-'+target,source_protocol=source,target_protocol='F'if family=='frozen'else target,
                contrast=name,left_method=left,right_method=right,selection_policy='identity_inclusive',
                source_evidence=';'.join(source_refs),target_evidence=';'.join(target_refs)))
            detailed.append(row)
    write('P0_contrast_details.csv',detailed)
    counts=[]
    for cohort in ('registered8','extended10','all_pairs15'):
        base=[r for r in detailed if r['contrast_set']==cohort]
        groups=[('all',base),('matched',[r for r in base if r['analysis_family']=='matched']),
                ('frozen',[r for r in base if r['analysis_family']=='frozen']),
                ('F_target',[r for r in base if r['target_protocol']=='F']),
                ('F_target_without_Yelp_NFM',[r for r in base if r['target_protocol']=='F'and(r['dataset'],r['backbone'])!=('yelp2018','NFM')])]
        for d,b in itertools.product(DATASETS,BACKBONES):
            groups.append((d+'/'+b,[r for r in base if(r['dataset'],r['backbone'])==(d,b)]))
        for name,rr in groups:
            c=Counter(r['classification']for r in rr)
            counts.append(dict(contrast_set=cohort,subset=name,n_comparisons=len(rr),**{k:c[k]for k in 'TREVX'},
                               transfer_fraction=c['T']/len(rr),frozen_training_free_duplicates_removed=True,status='COMPLETE'))
    write('P0_extended_counts.csv',counts)
    reversals=[r for r in detailed if r['contrast_set']=='all_pairs15' and r['classification']=='R']
    write('P0_all_pairs_reversals.csv',reversals, list(detailed[0]))
    flips=[r for r in detailed if r['point_mean_sign_flip'] and r['classification']!='R']
    write('P0_sign_flips.csv',flips,list(detailed[0]))

    return dict(contrast_rows=len(detailed),count_rows=len(counts),reversal_rows=len(reversals),sign_flip_rows=len(flips))
