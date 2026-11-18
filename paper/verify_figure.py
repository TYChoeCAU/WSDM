#!/usr/bin/env python3
"""Verify all Figure 2 means and intervals against seed evidence and manuscript values."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t

REPO = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(2020, 2025))
DATASETS = ('yelp2018', 'ml-1m')
BACKBONES = ('NFM', 'MF_BPR')
PROTOCOLS = ('U', 'W', 'F')
NUMERIC_ATOL_POINTS = 1e-3


def read(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def choose(rows, **keys):
    return [row for row in rows if all(str(row.get(k)) == str(v) for k, v in keys.items())]


def one(rows, **keys):
    selected = choose(rows, **keys)
    if len(selected) != 1:
        raise ValueError(f'Expected exactly one row for {keys}, found {len(selected)}')
    return selected[0]


def seed_values(rows, dataset, backbone, protocol, method, family='matched'):
    policy = 'native' if method in ('base', 'item_degree_ranking') else 'identity_inclusive'
    selected = choose(rows, regime='R0', dataset=dataset, backbone=backbone,
                      protocol=protocol, method=method, analysis_family=family,
                      selection_policy=policy)
    seeds = [int(row['training_seed']) for row in selected]
    if sorted(seeds) != list(SEEDS):
        raise ValueError(f'Expected each of five seeds once: {(dataset, backbone, protocol, method, family)}, got {seeds}')
    values = {int(row['training_seed']): 100 * float(row['ndcg@20']) for row in selected}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError('Nonfinite seed metric')
    return values


def paired_statistics(left, right):
    if set(left) != set(right):
        raise ValueError('Mismatched paired seed identities')
    seeds = sorted(left)
    differences = [left[seed] - right[seed] for seed in seeds]
    center = mean(differences)
    half = float(t.ppf(.975, len(seeds) - 1)) * stdev(differences) / math.sqrt(len(seeds))
    return (center, center - half, center + half)


def rounding_tolerance(token):
    token = str(token)
    places = len(token.partition('.')[2]) if '.' in token else 0
    return .5 * 10 ** (-places)


def table4_cells(expected):
    """Expand omitted manuscript row labels without assuming a CSV row order."""
    cells = {}
    dataset = backbone = None
    for row in expected['tables']['4']:
        fields = [part.strip() for part in row['label'].split('&')]
        if len(fields) != 3:
            raise ValueError('Unexpected Table 4 label: ' + row['label'])
        dataset = fields[0] or dataset
        backbone = fields[1] or backbone
        d = {'Yelp2018': 'yelp2018', 'ML-1M': 'ml-1m'}[dataset]
        b = {'NFM': 'NFM', 'MF-BPR': 'MF_BPR'}[backbone]
        key = (d, b, fields[2])
        if key in cells or len(row['expected_numeric_tokens']) != 14:
            raise ValueError('Duplicate or malformed Table 4 expectation: ' + str(key))
        cells[key] = row['expected_numeric_tokens']
    if set(cells) != {(d, b, p) for d in DATASETS for b in BACKBONES for p in PROTOCOLS}:
        raise ValueError('Table 4 expectations do not cover all twelve figure coordinates')
    return cells


SERIES = {
    'NFM-BCE': ('S3_NFM_BCE', 'NFM', 'observed_threshold_BCE'),
    'NFM-BPR': ('E1_NFM_BPR', 'NFM', 'BPR_unobserved_negative'),
    'MF-BCE': ('E2_MF_BCE', 'MF', 'observed_threshold_BCE'),
    'MF-BPR': ('S3_MF_BPR', 'MF', 'BPR_unobserved_negative'),
    'NFM-BCE+neg.': ('E3_NFM_BCE_augmented', 'NFM', 'observed_BCE_plus_uniform_unobserved_negatives'),
    'MF-BCE+neg.': ('E3_MF_BCE_augmented', 'MF', 'observed_BCE_plus_uniform_unobserved_negatives'),
}


def objective_seed_values(rows, dataset, series, protocol, method):
    cohort, architecture, objective = SERIES[series]
    selected = choose(rows, dataset=dataset, cohort=cohort, architecture=architecture,
                      objective=objective, protocol=protocol, method=method, status='COMPLETE')
    seeds = [int(row['training_seed']) for row in selected]
    required = list(range(2020, 2023 if cohort.startswith('E3_') else 2025))
    if sorted(seeds) != required:
        raise ValueError(f'Wrong training repeats for {dataset}/{series}/{protocol}/{method}: {seeds}')
    values = {int(row['training_seed']): 100 * float(row['ndcg@20']) for row in selected}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError('Nonfinite objective-ablation seed metric')
    return values


def verify(results, expected_path):
    results = Path(results)
    expected = json.loads(Path(expected_path).read_text())
    objective_raw = read(results/'e3/E3_all_conditions_seed_results.csv')
    objective_intervals = read(results/'e3/E3_pop_minus_base_paired.csv')
    raw = read(results/'s3/extracted_seed_results.csv')
    residual_intervals = read(results/'checks/C_paired_repeat_ci.csv')
    table4 = table4_cells(expected)
    checks, points = [], []
    figure_expectations = json.loads((Path(expected_path).parent/'figure_expected.json').read_text())
    seen_expectations = set()

    def compare(name, observed, target, tolerance, source):
        observed, target = float(observed), float(target)
        difference = abs(observed - target)
        passed = math.isfinite(observed) and math.isfinite(target) and difference <= tolerance + 1e-10
        checks.append(dict(name=name, source=source, observed=observed, expected=target,
                           absolute_difference=difference, tolerance=tolerance,
                           status='PASS' if passed else 'FAIL'))

    def record_point(panel, label, row, fields, recomputed, n_seeds):
        observed = tuple(float(row[field]) for field in fields)
        if not all(math.isfinite(x) for x in observed) or not observed[1] <= observed[0] <= observed[2]:
            raise ValueError('Invalid plotted interval: ' + label)
        for field, value, target in zip(('mean', 'ci95_low', 'ci95_high'), observed, recomputed):
            compare(label + ':' + field, value, target, NUMERIC_ATOL_POINTS, 'independent_paired_seed_recomputation')
        points.append(dict(panel=panel, coordinate=label, mean_points=observed[0],
                           ci95_low_points=observed[1], ci95_high_points=observed[2],
                           n_training_seeds=n_seeds))
        return observed

    def compare_manuscript(section, coordinate, observed, **keys):
        # Only printed manuscript values belong here. Unprinted interval values
        # remain independently checked against raw seeds above, never invented.
        subset = choose(figure_expectations[section], **keys)
        if len(subset) > 1:
            raise ValueError('Duplicate manuscript figure expectation: ' + coordinate)
        if not subset:
            return
        entry = subset[0]
        seen_expectations.add((section, tuple(sorted((key, str(value)) for key, value in keys.items()))))
        for field, actual in zip(('mean_points', 'ci95_low_points', 'ci95_high_points'), observed):
            if field in entry:
                token = entry[field]
                compare(coordinate + ':manuscript_' + field, actual, token,
                        rounding_tolerance(token), 'expected_values_from_the_manuscript')

    for panel, dataset in zip(('a', 'b'), DATASETS):
        for series, (cohort, architecture, objective) in SERIES.items():
            for protocol in PROTOCOLS:
                label = '/'.join((dataset, series, protocol)) + ':Pop-Base'
                row = one(objective_intervals, dataset=dataset, cohort=cohort,
                          architecture=architecture, protocol=protocol,
                          contrast='Pop-Base', metric='ndcg@20')
                required = tuple(range(2020, 2023 if cohort.startswith('E3_') else 2025))
                if (int(row['n_training_seeds']) != len(required) or int(row['df']) != len(required)-1
                        or float(row['ci_level']) != .95
                        or row['training_seeds'] != ';'.join(map(str, required))
                        or row['status'] != f'COMPLETE(n={len(required)})'):
                    raise ValueError('Wrong plotted uncertainty cohort: ' + label)
                base = objective_seed_values(objective_raw, dataset, series, protocol, 'base')
                pop = objective_seed_values(objective_raw, dataset, series, protocol, 'item_degree_ranking')
                values = record_point(panel, label, row,
                                      ('mean_x100', 'ci95_low_x100', 'ci95_high_x100'),
                                      paired_statistics(pop, base), len(required))
                compare_manuscript('objective_pop_minus_base', label, values,
                                   dataset=dataset, series=series, protocol=protocol)
                # Table 4 remains unchanged and covers the two original S3 cohorts.
                if series in ('NFM-BCE', 'MF-BPR'):
                    backbone = 'NFM' if series == 'NFM-BCE' else 'MF_BPR'
                    base_token, pop_token = table4[(dataset, backbone, protocol)][0:3:2]
                    for method, seed_metrics, token in (('Base', base, base_token), ('Pop', pop, pop_token)):
                        compare(label + ':Table4_' + method, mean(seed_metrics.values()), token,
                                rounding_tolerance(token), 'manuscript_Table4')
                    compare(label + ':Table4_difference', values[0], float(pop_token) - float(base_token),
                            rounding_tolerance(pop_token) + rounding_tolerance(base_token),
                            'difference_of_manuscript_rounded_cells')

    conditions = (('W', 'W', 'matched'), ('W', 'F', 'frozen'), ('F', 'F', 'matched'))
    for fit, evaluation, family in conditions:
        label = 'yelp2018/MF_BPR/' + fit + '->' + evaluation + ':Res(p)-Base'
        row = one(residual_intervals, dataset='yelp2018', backbone='MF_BPR',
                  contrast='Res(p)-Base', fit_protocol=fit, eval_protocol=evaluation,
                  analysis_family=family)
        if (int(row['n_seeds']) != 5 or int(row['df']) != 4
                or row['seeds'] != ';'.join(map(str, SEEDS))):
            raise ValueError('Wrong residual plotted uncertainty cohort: ' + label)
        protocol = fit + '_to_' + evaluation if family == 'frozen' else fit
        residual = seed_values(raw, 'yelp2018', 'MF_BPR', protocol, 'P', family)
        base = seed_values(raw, 'yelp2018', 'MF_BPR', evaluation, 'base')
        values = record_point('c', label, row, ('mean_x100', 'ci95_low_x100', 'ci95_high_x100'),
                              paired_statistics(residual, base), len(SEEDS))
        compare_manuscript('residual_minus_base', label, values,
                           fit_protocol=fit, eval_protocol=evaluation)

    expected_count = sum(len(figure_expectations[key]) for key in ('objective_pop_minus_base', 'residual_minus_base'))
    if len(seen_expectations) != expected_count:
        raise ValueError('Some manuscript figure expectation has no plotted coordinate')
    if len(points) != 39 or sum(point['panel'] in ('a', 'b') for point in points) != 36:
        raise ValueError('Expected 36 objective-ablation points plus three residual points')
    half_width = max(max(p['mean_points']-p['ci95_low_points'],
                         p['ci95_high_points']-p['mean_points'])
                     for p in points if p['panel'] in ('a', 'b'))
    limit = float(figure_expectations['panel_ab_max_half_width_points'])
    checks.append(dict(name='panel_ab_max_half_width', observed=half_width,
                       expected_upper_bound=limit, source='manuscript_Figure2_caption',
                       status='PASS' if half_width <= limit else 'FAIL'))
    failed = [check for check in checks if check['status'] != 'PASS']
    return dict(status='PASS' if not failed else 'FAIL', plotted_points=len(points),
                objective_ablation_points=36, residual_points=3,
                plotted_values_checked=3*len(points), total_checks=len(checks),
                passed_checks=len(checks)-len(failed), failed_checks=len(failed),
                manuscript_coordinate_entries=expected_count,
                manuscript_printed_expectation_coordinates=sum(
                    any(field in entry for field in ('mean_points', 'ci95_low_points', 'ci95_high_points'))
                    for section in ('objective_pop_minus_base', 'residual_minus_base')
                    for entry in figure_expectations[section]),
                raw_numeric_tolerance_points=NUMERIC_ATOL_POINTS,
                interval_rule='paired Student-t 95% CI: three E3 training seeds (df=2), five existing training seeds (df=4)',
                raw_evidence='E3_all_conditions_seed_results.csv for panels a,b; extracted_seed_results.csv for panel c',
                manuscript_rule='half a printed unit; summed rounding bounds for differences of rounded cells',
                unprinted_values_rule='all means and interval endpoints independently recomputed from the prespecified three or five seed results',
                pixel_comparison_performed=False, points=points, checks=checks)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--expected', type=Path, default=REPO/'configs/analysis_expected.json')
    parser.add_argument('--out', type=Path, default=Path('outputs/figure_value_check.json'))
    args = parser.parse_args()
    try:
        report = verify(args.results, args.expected)
    except (ValueError, KeyError, IndexError, OSError) as exc:
        report = dict(status='FAIL_INPUT', error=str(exc), pixel_comparison_performed=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('points', 'checks')}))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

