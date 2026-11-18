"""Refresh scalar-only E3 evidence and six-condition summaries, without inference.

Existing E evidence is read only. A new coordinate contributes metrics only after
all U/W/F evaluations, its seed CSV and its COMPLETE marker agree.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import sys
import fcntl
sys.dont_write_bytecode = True
from scipy.stats import t

from .paths import ROOT,OUT,EVIDENCE,input_file
OLD_E = input_file('objective_seed_results',EVIDENCE/'followup/E_seed_results.csv')
MOSTPOP = input_file('mostpop',EVIDENCE/'checks/D_mostpop.csv')
DATASETS = ('yelp2018', 'ml-1m')
ARCHITECTURES = ('MF', 'NFM')
PROTOCOLS = ('U', 'W', 'F')
METHODS = ('base', 'item_degree_ranking', 'original_nonparamgc')
METHOD_LABEL = dict(zip(METHODS, ('Base', 'Pop', 'NPGC')))
METRICS = ('ndcg@10', 'ndcg@20', 'recall@10', 'recall@20', 'candidate_recall50', 'oracle_ndcg20')
REQUIRED = (2020, 2021, 2022)
OPTIONAL = (2023, 2024)
COHORTS = {
    ('MF', 'observed BCE'): 'E2_MF_BCE',
    ('NFM', 'observed BCE'): 'S3_NFM_BCE',
    ('MF', 'BCE + unobserved negatives'): 'E3_MF_BCE_augmented',
    ('NFM', 'BCE + unobserved negatives'): 'E3_NFM_BCE_augmented',
    ('MF', 'BPR'): 'S3_MF_BPR',
    ('NFM', 'BPR'): 'E1_NFM_BPR',
}
CONDITIONS = ('observed BCE', 'BCE + unobserved negatives', 'BPR')


def relative(path):
    """Use project-relative references in every generated report/CSV."""
    path=Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def clean_source(value):
    return str(value).replace(str(ROOT) + '/', '')


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    if not path.resolve().is_relative_to(OUT):
        raise PermissionError('Output must stay within this new E3 run')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(text)
    temp.replace(path)


def write_json(path, obj):
    atomic_text(path, json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows, empty_fields=('status',)):
    import io
    fields = list(dict.fromkeys(key for row in rows for key in row)) or list(empty_fields)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def stats(values):
    values = list(values)
    n = len(values)
    if not n:
        return dict(n=0, mean=None, sample_sd=None, ci95_low=None, ci95_high=None, df=None)
    mean = statistics.mean(values)
    if n == 1:
        return dict(n=n, mean=mean, sample_sd=None, ci95_low=None, ci95_high=None, df=0)
    sd = statistics.stdev(values)
    half = float(t.ppf(.975, n - 1)) * sd / math.sqrt(n)
    return dict(n=n, mean=mean, sample_sd=sd, ci95_low=mean-half, ci95_high=mean+half, df=n-1)


def completion_status(seeds, existing=False):
    seeds = sorted(set(seeds))
    if not seeds:
        return 'NOT_STARTED'
    if existing and seeds == list(REQUIRED + OPTIONAL):
        return 'COMPLETE(n=5)'
    if not existing and seeds in (list(REQUIRED), list(REQUIRED + OPTIONAL)):
        return f'COMPLETE(n={len(seeds)})'
    return f'PARTIAL(n={len(seeds)})'


def load_existing():
    rows = read_csv(OLD_E)
    expected_cohorts = set(COHORTS.values()) - {COHORTS[a, CONDITIONS[1]] for a in ARCHITECTURES}
    result = []
    for number, source in enumerate(rows, 2):
        if source['cohort'] not in expected_cohorts or source['status'] != 'COMPLETE':
            raise ValueError(f'Unexpected historical E row at {number}')
        row = {key: clean_source(value) for key, value in source.items()}
        row['input_csv_evidence'] = relative(OLD_E) + ':' + str(number)
        row['existing_or_new'] = 'existing_5_seed'
        result.append(row)
    return result


def load_e3():
    rows, inventory = [], []
    popular = read_csv(MOSTPOP)
    for architecture in ARCHITECTURES:
        for dataset in DATASETS:
            for seed in REQUIRED + OPTIONAL:
                directory = OUT / 'e3' / architecture / dataset / f'seed_{seed}'
                marker = directory / 'complete.json'
                seed_csv = directory / 'seed_results.csv'
                backbone = directory / 'backbone_result.json'
                status_path = directory / 'run_status.json'
                running = json.loads(status_path.read_text()) if status_path.exists() else {}
                inv = dict(architecture=architecture, dataset=dataset, training_seed=seed,
                           required=seed in REQUIRED, status='NOT_STARTED',
                           source=relative(directory), completed_protocols=0,
                           checkpoint_sha256='', last_complete_epoch=running.get('last_complete_epoch', ''),
                           error=running.get('error', ''))
                eval_paths = [directory/'evaluation'/p/'result.json' for p in PROTOCOLS]
                inv['completed_protocols'] = sum(p.exists() and json.loads(p.read_text()).get('status') == 'COMPLETE' for p in eval_paths)
                if not marker.exists():
                    if directory.exists() and any(directory.iterdir()):
                        inv['status'] = ('BACKBONE_COMPLETE_EVALUATION_PENDING' if running.get('status') == 'COMPLETE' else running.get('status', 'RUNNING_OR_PENDING_EVALUATION'))
                    inventory.append(inv)
                    continue
                complete = json.loads(marker.read_text())
                if complete.get('status') != 'COMPLETE' or not seed_csv.exists() or not backbone.exists() or inv['completed_protocols'] != 3:
                    inv['status'] = 'INCOMPLETE_ARTIFACTS'
                    inventory.append(inv)
                    continue
                if (complete.get('dataset'),int(complete.get('seed',-1)),complete.get('architecture')) != (dataset,seed,architecture):
                    raise ValueError('Completion marker identity mismatch')
                model = json.loads(backbone.read_text())
                metrics = read_csv(seed_csv)
                expected_keys = {(p, m) for p in PROTOCOLS for m in METHODS}
                keys = {(r['protocol'], r['method']) for r in metrics}
                if len(metrics) != 9 or keys != expected_keys:
                    raise ValueError('Completed E3 coordinate is missing or duplicates methods/protocols: ' + relative(directory))
                if any(int(r['training_seed']) != seed or r['dataset'] != dataset for r in metrics):
                    raise ValueError('Completed E3 coordinate identity mismatch')
                if any(r.get('architecture', architecture) != architecture for r in metrics):
                    raise ValueError('E3 architecture mismatch')
                checkpoint = model['checkpoint_sha256']
                digest = model['identity']['master_digest']
                for path in eval_paths:
                    evaluation = json.loads(path.read_text())
                    if evaluation['checkpoint_sha256'] != checkpoint:
                        raise ValueError('Evaluation uses another checkpoint')
                    protocol = path.parent.name
                    reference = [x for x in popular if x['study'] == 'S3' and x['dataset'] == dataset and x['backbone'] == ('MF_BPR' if architecture == 'MF' else 'NFM') and x['protocol'] == protocol]
                    if len(reference) != 1 or reference[0]['master_digest'] != digest:
                        raise ValueError('E3/MostPop master lineage differs or is unverifiable')
                    expected_rows = {x['method']: x for x in evaluation['rows']}
                    for row in [x for x in metrics if x['protocol'] == protocol]:
                        if row.get('checkpoint_sha256', checkpoint) != checkpoint:
                            raise ValueError('Seed CSV checkpoint differs')
                        for field in METRICS:
                            value = float(row[field])
                            if not math.isfinite(value) or not 0 <= value <= 1:
                                raise ValueError('Invalid completed scalar metric')
                            if abs(value-float(expected_rows[row['method']][field])) > 1e-3:
                                raise ValueError('CSV/evaluation metric mismatch beyond1e-3')
                    for field in ('candidate_recall50', 'oracle_ndcg20'):
                        values = [float(x[field]) for x in metrics if x['protocol'] == protocol]
                        if max(values)-min(values) > 1e-3:
                            raise ValueError('Methods do not share candidate coverage/oracle')
                for number, source in enumerate(metrics, 2):
                    row = {key: clean_source(value) for key, value in source.items()}
                    row.update(cohort=COHORTS[architecture, CONDITIONS[1]], architecture=architecture,
                               objective='observed_BCE_plus_uniform_unobserved_negatives', phase='E3',
                               method_label=METHOD_LABEL[row['method']], status='COMPLETE',
                               input_csv_evidence=relative(seed_csv)+':'+str(number),
                               existing_or_new='new_E3', master_digest=digest,
                               checkpoint_sha256=checkpoint, source_evidence=relative(seed_csv)+':'+str(number))
                    rows.append(row)
                inv.update(status='COMPLETE', checkpoint_sha256=checkpoint, master_digest=digest,
                           selected_backbone_epoch=model.get('selected_epoch', ''),
                           maximum_epochs=model.get('identity', {}).get('profile', {}).get('maximum_epochs', ''),
                           training_seconds=model.get('elapsed_seconds', model.get('training_seconds', '')))
                inventory.append(inv)
    return rows, inventory


def summaries(rows):
    indexed = {}
    for row in rows:
        key = tuple(row[k] for k in ('cohort', 'dataset', 'protocol', 'method')) + (int(row['training_seed']),)
        if key in indexed:
            raise ValueError('Duplicate complete seed metric key')
        indexed[key] = row
    output, paired, table = [], [], []
    for architecture in ARCHITECTURES:
        for condition in CONDITIONS:
            cohort = COHORTS[architecture, condition]
            existing = not cohort.startswith('E3_')
            for dataset in DATASETS:
                for protocol in PROTOCOLS:
                    seed_sets = [{k[-1] for k in indexed if k[:4] == (cohort, dataset, protocol, method)} for method in METHODS]
                    if not all(seed_sets[0] == s for s in seed_sets):
                        raise ValueError('Method seed coverage differs')
                    seeds = sorted(seed_sets[0])
                    state = completion_status(seeds, existing)
                    metadata = dict(cohort=cohort, architecture=architecture, training_condition=condition,
                                    dataset=dataset, protocol=protocol, status=state, n_training_seeds=len(seeds),
                                    training_seeds=';'.join(map(str, seeds)), repetition_unit='training_seed_fixed_split_fixed_candidate_draw',
                                    required_seed_set_complete=existing or set(REQUIRED).issubset(seeds),
                                    existing_or_new='existing_5_seed' if existing else 'new_E3')
                    for method in METHODS:
                        row = dict(metadata, method=method, method_label=METHOD_LABEL[method])
                        for metric in METRICS:
                            result = stats(float(indexed[cohort, dataset, protocol, method, seed][metric])*100 for seed in seeds)
                            for name in ('mean', 'sample_sd', 'ci95_low', 'ci95_high'):
                                row[metric+'_'+name+'_x100'] = result[name]
                        row['df'] = len(seeds)-1 if seeds else ''
                        rho=[];undefined=[]
                        for seed in seeds:
                            record=indexed[cohort,dataset,protocol,method,seed]
                            value=record.get('backbone_score_popularity_spearman_mean',record.get('spearman_s_p_mean'))
                            if value in (None,'','None'):
                                undefined.append(seed)
                            else:
                                rho.append(float(value))
                        rho_stats=stats(rho)
                        row['spearman_s_p_user_mean_then_seed_mean']=rho_stats['mean']
                        row['spearman_s_p_sample_sd']=rho_stats['sample_sd']
                        row['spearman_defined_training_seeds']=len(rho)
                        row['spearman_undefined_training_seeds']=';'.join(map(str,undefined))
                        output.append(row)
                    for contrast, method in [('Pop-Base', 'item_degree_ranking'), ('NPGC-Base', 'original_nonparamgc')]:
                        for metric in ('ndcg@20', 'recall@20'):
                            differences = {seed: float(indexed[cohort, dataset, protocol, method, seed][metric])-float(indexed[cohort, dataset, protocol, 'base', seed][metric]) for seed in seeds}
                            result = stats(differences.values())
                            row = dict(metadata, contrast=contrast, metric=metric, df=result['df'], ci_level=.95,
                                       differences_by_seed_raw=json.dumps(differences, sort_keys=True))
                            for name in ('mean', 'sample_sd', 'ci95_low', 'ci95_high'):
                                row[name+'_raw'] = result[name]
                                row[name+'_x100'] = None if result[name] is None else 100*result[name]
                            paired.append(row)
                    pop = paired[-4]
                    table.append({**metadata, **{key: pop[key] for key in ('mean_x100', 'sample_sd_x100', 'ci95_low_x100', 'ci95_high_x100')}})
    return output, paired, table


def accuracy_rule(summary):
    popular = read_csv(MOSTPOP)
    indexed = {(x['cohort'], x['dataset'], x['protocol'], x['method']): x for x in summary}
    output = []
    for base in [x for x in summary if x['method'] == 'base']:
        key = (base['cohort'], base['dataset'], base['protocol'])
        pop = indexed[key+('item_degree_ranking',)]
        matches = [x for x in popular if x['study'] == 'S3' and x['dataset'] == base['dataset'] and x['backbone'] == ('MF_BPR' if base['architecture'] == 'MF' else 'NFM') and x['protocol'] == base['protocol']]
        if len(matches) != 1 or matches[0]['status'] != 'PASS' or int(matches[0]['seed']) != 2020:
            raise ValueError('MostPop fixed-pool reference is missing or not audited')
        reference = matches[0]
        row = {k: base[k] for k in ('cohort', 'architecture', 'training_condition', 'dataset', 'protocol', 'status', 'n_training_seeds', 'training_seeds', 'existing_or_new')}
        row.update(same_pool_mostpop_points=100*float(reference['ndcg20']),
                   mostpop_candidate_seed=2020, mostpop_repetitions=1,
                   mostpop_source=relative(MOSTPOP), mostpop_master_digest=reference['master_digest'],
                   practical_margin_points=.25, included_in_complete_rule_count=False,
                   base_points=None, top50_pop_points=None, pop_minus_base_points=None,
                   base_below_mostpop=None, pop_helps=None, rule_agrees=None,
                   exception_inside_margin=None)
        if base['n_training_seeds']:
            b, p = base['ndcg@20_mean_x100'], pop['ndcg@20_mean_x100']
            effect = p-b
            agree = (b < row['same_pool_mostpop_points']) == (effect > 0)
            row.update(base_points=b, top50_pop_points=p, pop_minus_base_points=effect,
                       base_below_mostpop=b < row['same_pool_mostpop_points'], pop_helps=effect > 0,
                       rule_agrees=agree, exception_inside_margin=not agree and abs(effect) < .25,
                       included_in_complete_rule_count=base['status'].startswith('COMPLETE'))
        output.append(row)
    if len(output) != 36:
        raise ValueError('Expected36 planned architecture/training/dataset/protocol cells')
    old = [x for x in output if x['existing_or_new'] == 'existing_5_seed']
    if len(old) != 24 or sum(x['rule_agrees'] for x in old) != 23:
        raise ValueError('Historical23/24 rule reproduction failed')
    return output


def format_effect(row):
    if not row['n_training_seeds']:
        return 'NOT_STARTED'
    mean, low, high = row['mean_x100'], row['ci95_low_x100'], row['ci95_high_x100']
    interval = 'CI unavailable' if low is None else f'[{low:+.3f}, {high:+.3f}]'
    return f'{mean:+.3f} {interval}, n={row["n_training_seeds"]}' + ('' if row['status'].startswith('COMPLETE') else ' PARTIAL')


def source_line(path, needle):
    if not path.exists():
        return relative(path) + ' (준비 중)'
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if needle in line:
            return relative(path) + ':' + str(n)
    return relative(path) + ':1'



def stage2_report():
    status_path=OUT/'N1_status.json'
    seed_path=OUT/'N1_npgc_popularity_seed.csv'
    summary_path=OUT/'N1_npgc_popularity_summary.csv'
    state=json.loads(status_path.read_text()) if status_path.exists() else {'status':'NOT_STARTED'}
    lines=['', '## Stage 2: NPGC 보정과 인기도의 연관', '',
           '이 분석은 저장된 Top-50 점수와 인기도만 읽고 label을 사용하지 않습니다. 상관관계는 보정의 인기도 연관을 기술하며 NPGC 이득의 원인을 확정하지 않습니다. p는 binary training-graph degree의 log(1+degree)입니다.', '',
           '상태: `'+json.dumps(state,ensure_ascii=False)+'`.']
    rows=read_csv(seed_path) if seed_path.exists() else []
    for study,expected in [('S1',54),('S3',60)]:
        rr=[x for x in rows if x.get('study')==study and x.get('status')=='COMPLETE']
        lines.append(f'- {study}: 완료 {len(rr)}/{expected} 좌표. 완전한 저장 점수가 없는 좌표는 추정하거나 재추론하여 채우지 않습니다.')
    lines += ['', 'Top-20 entrant−exit의 주 평균은 실제 Top-20 교체가 발생하여 두 집합이 정의된 사용자만 포함합니다. 별도 zero-nochange 열은 교체가 없는 사용자에게0을 부여한 전체 사용자 평균입니다. 상관계수가 정의되지 않은 상수 점수·인기도 리스트 수는 각 metric의 undefined count로 남깁니다. Spearman과 log-popularity 차이는 ×100을 적용하지 않습니다.']
    if summary_path.exists():
        summary=read_csv(summary_path)
        metrics=('rho_delta_score_popularity','rho_rank_improvement_popularity','top20_entrant_minus_exit_popularity')
        lines += ['', '| Study | Dataset | Backbone | Protocol | n | 상태 | ρ(보정량,p) | ρ(순위상승,p) | Top20 entrant−exit p |', '|---|---|---|---|---:|---|---:|---:|---:|']
        for row in summary:
            def val(metric):
                mean=row.get(metric+'_mean','');sd=row.get(metric+'_seed_sd','')
                if mean in ('','None'):return 'undefined'
                return f'{float(mean):+.4f}' + (f' ± {float(sd):.4f}' if sd not in ('','None') else ' (SD undefined)')
            lines.append('| '+' | '.join(str(row.get(k,'')) for k in ('study','dataset','backbone','protocol','n_seeds','status'))+' | '+' | '.join(val(m) for m in metrics)+' |')
    lines += ['', '- 진단 원시값: `'+relative(seed_path)+':1`.',
              '- summary: `'+relative(summary_path)+':1`.',
              '- 미가용 좌표·사유: `'+relative(OUT/'N1_coordinate_inventory.csv')+':1`.',
              '- 점수·순위 보정 계산: `'+source_line(Path(__file__).with_name('n1_popularity.py'),'def diagnostics(')+'`.',
              '- 상관 계산: `'+source_line(Path(__file__).with_name('n1_popularity.py'),'def correlations(')+'`.']
    return lines


def build_report(summary, paired, tables, rules, inventory):
    complete = [x for x in inventory if x['status'] == 'COMPLETE']
    nrequired = sum(x['required'] for x in complete)
    noptional = sum(not x['required'] for x in complete)
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec='seconds')
    sections = ['# E3: observed BCE에 미관측 negative를 추가한 비교', '', f'갱신: {now}', '',
                f'필수 완료 **{nrequired}/12 좌표**, 선택 확장 **{noptional}/8 좌표**. 한 좌표는 architecture×dataset×training seed이며 U/W/F의 Base·Pop·NPGC 평가9행이 모두 완료되어야 완료로 셉니다.', '',
                '기존4조건은 각5-seed(2020–2024) 결과를 재사용합니다. E3는 우선3-seed(2020–2022)를 수행하고 여유가 있을 때만2023–2024를 추가합니다. n=3 또는 n=5의 사전 지정 seed 집합만 COMPLETE로 표시합니다. PARTIAL(n)은 완료된 개별 실행만의 중간 집계이며 최종 비교 결론에 포함하지 않습니다.', '',
                '학습이 중단된 checkpoint나 아직 한 protocol만 완료된 좌표는 발표용 seed CSV에 들어가지 않습니다. 기존 결과·학습 코드·원고는 수정하지 않습니다.', '',
                '## 진행 상태', '', '| Architecture | Dataset | 필수 완료 | 선택 완료 | 완료 seeds |', '|---|---|---:|---:|---|']
    for architecture in ARCHITECTURES:
        for dataset in DATASETS:
            rr=[x for x in complete if x['architecture']==architecture and x['dataset']==dataset]
            sections.append(f'| {architecture} | {dataset} | {sum(x["required"] for x in rr)}/3 | {sum(not x["required"] for x in rr)}/2 | '+', '.join(str(x['training_seed']) for x in rr)+' |')
    sections += ['', '## 통계와 조건', '',
                 '모든 정확도·coverage·oracle 수치는 ×100 단위입니다. 구간은 같은 training seed의 paired 차이에 대한 Student-t95% 구간이며 df=n−1입니다. 표준편차는 ddof=1입니다. Candidate Recall@50와 oracle NDCG@20은 각 backbone이 제공한 동일 Top-50 후보의 full-relevance 분모를 사용합니다.', '',
                 'BCE에 미관측 negative를 추가하면 negative 노출과 학습 row 구성이 함께 달라집니다. 같은 BCE 목적의 비교는 이 개입의 관측 효과를 보여주지만, 인기 편향의 단일 원인이나 negative 노출만의 독립적 인과 효과를 증명하지 않습니다.', '',
                 '## Dataset·protocol별3×2 비교', '', '각 셀은 Pop−Base NDCG@20 평균 [95% training-repeat CI], n입니다. 기존5-seed와 신규3/5-seed의 반복 수를 셀마다 표기합니다.']
    table_lines=['# E3 six-condition Pop−Base tables', '', 'Unit: NDCG@20 points (×100). Paired Student-t95% CI over completed training seeds. Existing conditions use n=5. New conditions explicitly show n=3 or n=5, otherwise PARTIAL.']
    indexed={(x['dataset'],x['protocol'],x['architecture'],x['training_condition']):x for x in tables}
    for dataset in DATASETS:
        for protocol in PROTOCOLS:
            block=['', f'### {dataset} / {protocol}', '', '| Training condition | NFM | MF |', '|---|---|---|']
            for condition in CONDITIONS:
                block.append('| '+condition+' | '+' | '.join(format_effect(indexed[dataset,protocol,a,condition]) for a in ('NFM','MF'))+' |')
            sections += block; table_lines += block
    complete_rules=[x for x in rules if x['included_in_complete_rule_count']]
    new_rules=[x for x in complete_rules if x['existing_or_new']=='new_E3']
    exceptions=[x for x in complete_rules if not x['rule_agrees']]
    sections += ['', '## Same-pool MostPop 규칙', '',
                 '규칙은 Base < whole-pool MostPop와 Top-50 Pop−Base > 0의 일치 여부입니다. Whole-pool MostPop는 학습 seed별 반복이 아니라 기존 fixed candidate draw2020의 한 결과입니다. Top-50 Pop와 혼동하지 않습니다.', '',
                 f'기존 규칙은 **23/24 일치**로 재현됩니다. 새 E3 완료 셀은 **{sum(x["rule_agrees"] for x in new_rules)}/{len(new_rules)} 일치**, 전체 완료 셀은 **{sum(x["rule_agrees"] for x in complete_rules)}/{len(complete_rules)} 일치**입니다. 예정 전체는36셀입니다. 미완료 E3 셀은 분모에 넣지 않습니다.', '',
                 '| Cohort | Dataset | Protocol | Pop−Base | Base | Whole-pool MostPop | 절대 효과<0.25 |', '|---|---|---|---:|---:|---:|---|']
    for row in exceptions:
        sections.append(f'| {row["cohort"]} | {row["dataset"]} | {row["protocol"]} | {row["pop_minus_base_points"]:+.5f} | {row["base_points"]:.5f} | {row["same_pool_mostpop_points"]:.5f} | {row["exception_inside_margin"]} |')
    sections += stage2_report()
    sections += ['', '## 원고용 영어 문장', '']
    if nrequired == 12 and len(new_rules) == 12:
        nsets=sorted({x['n_training_seeds'] for x in new_rules})
        sizes='/'.join(map(str,nsets))
        signs=[]
        for a in ARCHITECTURES:
            for ds in DATASETS:
                x=indexed[ds,'U',a,CONDITIONS[1]]
                if x['status'].startswith('COMPLETE'):signs.append('positive' if x['mean_x100']>0 else 'negative' if x['mean_x100']<0 else 'zero')
        if len(signs)==4 and len(set(signs))==1:
            sections.append(f'With {sizes} completed training seeds per E3 condition, adding uniformly sampled unobserved negatives to observed-label BCE yields a {signs[0]} Pop−Base contrast under U for MF and NFM on both datasets.')
        else:
            sections.append(f'The E3 comparison uses {sizes} completed training seeds per condition to measure the Pop−Base contrast after observed-label BCE receives uniformly sampled unobserved negatives.')
        sections.append('This intervention changes negative exposure together with the composition of training rows, so the observed contrasts do not isolate a single causal mechanism.')
    else:
        sections.append(f'E3 currently completes {nrequired} of 12 required architecture–dataset–training-seed coordinates, while {len(new_rules)} of 12 E3 protocol cells have complete prespecified seed sets. The comparison retains the BCE objective while it changes negative exposure together with the composition of training rows.')
    sections += ['', '## 근거와 산출물', '',
                 '- `E3_seed_results.csv`: 완료된 E3 raw metric·seed·checkpoint·출처.',
                 '- `E3_all_conditions_seed_results.csv`: 기존4조건과 E3 두 조건의 raw metric.',
                 '- `E3_summary.csv`, `E3_pop_minus_base_paired.csv` (`E3_paired.csv`와 동일): 평균·표본SD·95% paired training-repeat CI. Paired 파일에는 Pop−Base와 NPGC−Base를 모두 저장합니다.',
                 '- `E3_accuracy_rule_36.csv`: 예정36셀의 same-pool 규칙, 완료 여부·예외.',
                 '- `E3_run_inventory.csv`: 필수12·선택8좌표 상태와 미완료 좌표.',
                 '- `E3_table.md`, `E3_training_3x2_table.csv`:6개dataset–protocol표.',
                 '- 기존 증거: `'+relative(OLD_E)+':1`, `'+relative(MOSTPOP)+':1`.',
                 '- 통계·반복 단위: `'+source_line(Path(__file__),'def stats(')+'`.',
                 '- COMPLETE marker·좌표 검증: `'+source_line(Path(__file__),'def load_e3(')+'`.',
                 '- paired seed 집계: `'+source_line(Path(__file__),'def summaries(')+'`.',
                 '- whole-pool MostPop 대조: `'+source_line(Path(__file__),'def accuracy_rule(')+'`.',
                 '- 학습 연결: `'+source_line(Path(__file__).with_name('e3_worker.py'),'def train(')+'`.',
                 '- 평가 연결: `'+source_line(Path(__file__).with_name('e3_static_eval.py'),'def evaluate(')+'`.']
    return '\n'.join(sections)+'\n', '\n'.join(table_lines)+'\n'


def refresh():
    old_hash=sha(OLD_E); pop_hash=sha(MOSTPOP)
    historical=load_existing();new,inventory=load_e3()
    rows=historical+new
    summary,paired,tables=summaries(rows)
    rules=accuracy_rule(summary)
    report,table_md=build_report(summary,paired,tables,rules,inventory)
    write_csv(OUT/'E3_seed_results.csv',new,('cohort','dataset','training_seed','protocol','method','status'))
    write_csv(OUT/'E3_all_conditions_seed_results.csv',rows)
    write_csv(OUT/'E3_summary.csv',summary)
    write_csv(OUT/'E3_paired.csv',paired)
    write_csv(OUT/'E3_pop_minus_base_paired.csv',paired)
    write_csv(OUT/'E3_training_3x2_table.csv',tables)
    write_csv(OUT/'E3_accuracy_rule_36.csv',rules)
    write_csv(OUT/'E3_run_inventory.csv',inventory)
    atomic_text(OUT/'E3_REPORT_KO.md',report)
    atomic_text(OUT/'E3_table.md',table_md)
    if sha(OLD_E)!=old_hash or sha(MOSTPOP)!=pop_hash:
        raise RuntimeError('Existing evidence changed during read-only aggregation')
    completed=[x for x in inventory if x['status']=='COMPLETE']
    audit=dict(status='COMPLETE' if sum(x['required'] for x in completed)==12 else 'IN_PROGRESS',
               required_complete=sum(x['required'] for x in completed),required_total=12,
               optional_complete=sum(not x['required'] for x in completed),optional_total=8,
               existing_seed_rows=len(historical),e3_seed_rows=len(new),planned_rule_cells=len(rules),
               complete_rule_cells=sum(x['included_in_complete_rule_count'] for x in rules),
               inputs=[dict(path=relative(OLD_E),sha256=old_hash),dict(path=relative(MOSTPOP),sha256=pop_hash)],
               new_training_started_by_aggregator=False,source_artifacts_modified=False,
               stats='Paired training-seed differences; sample SD ddof1; Student-t95% CI df=n-1. All accuracy metrics ×100.',
               incomplete_coordinate_metrics_published=False)
    write_json(OUT/'E3_summary_status.json',audit)
    print(json.dumps(audit,ensure_ascii=False))


def main():
    # Dataset lanes can finish together. Serialize only this inexpensive refresh.
    with (OUT/'.e3_summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        refresh()


if __name__=='__main__':
    main()
