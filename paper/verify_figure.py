#!/usr/bin/env python3
"""Verify all Figure 1 means and intervals against seed evidence and manuscript values."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
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
    differences = [left[seed] - right[seed] for seed in SEEDS]
    center = mean(differences)
    half = float(t.ppf(.975, len(SEEDS) - 1)) * stdev(differences) / math.sqrt(len(SEEDS))
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
        if key in cells or len(row['expected_numeric_tokens']) != 16:
            raise ValueError('Duplicate or malformed Table 4 expectation: ' + str(key))
        cells[key] = row['expected_numeric_tokens']
    if set(cells) != {(d, b, p) for d in DATASETS for b in BACKBONES for p in PROTOCOLS}:
        raise ValueError('Table 4 expectations do not cover all twelve figure coordinates')
    return cells


def verify(results, expected_path):
    results = Path(results)
    expected = json.loads(Path(expected_path).read_text())
    raw = read(results/'s3/extracted_seed_results.csv')
    intervals = read(results/'s3/repeat_intervals.csv')
    residual_intervals = read(results/'checks/C_paired_repeat_ci.csv')
    table4 = table4_cells(expected)
    checks, points = [], []

    def compare(name, observed, target, tolerance, source):
        observed, target = float(observed), float(target)
        difference = abs(observed - target)
        passed = math.isfinite(observed) and math.isfinite(target) and difference <= tolerance + 1e-10
        checks.append(dict(name=name, source=source, observed=observed, expected=target,
                           absolute_difference=difference, tolerance=tolerance,
                           status='PASS' if passed else 'FAIL'))

    def record_point(panel, label, row, fields, recomputed):
        observed = tuple(float(row[field]) for field in fields)
        if not all(math.isfinite(x) for x in observed) or not observed[1] <= observed[0] <= observed[2]:
            raise ValueError('Invalid plotted interval: ' + label)
        for field, value, target in zip(('mean', 'ci95_low', 'ci95_high'), observed, recomputed):
            compare(label + ':' + field, value, target, NUMERIC_ATOL_POINTS, 'paired_seed_recomputation')
        points.append(dict(panel=panel, coordinate=label, mean_points=observed[0],
                           ci95_low_points=observed[1], ci95_high_points=observed[2],
                           n_training_seeds=len(SEEDS)))
        return observed

    for panel, dataset in zip(('a', 'b'), DATASETS):
        for backbone in BACKBONES:
            for protocol in PROTOCOLS:
                key = (dataset, backbone, protocol)
                label = '/'.join(key) + ':Pop-Base'
                row = one(intervals, scope='main_R0', regime='R0', metric='ndcg@20',
                          selection_policy='identity_inclusive', contrast='Pop-Base',
                          analysis_family='matched', dataset=dataset, backbone=backbone,
                          comparison=protocol)
                if (int(row['n_training_seeds']) != 5 or int(row['df']) != 4
                        or float(row['ci_level']) != .95
                        or row['training_seeds'] != ';'.join(map(str, SEEDS))):
                    raise ValueError('Wrong plotted uncertainty cohort: ' + label)
                base = seed_values(raw, dataset, backbone, protocol, 'base')
                pop = seed_values(raw, dataset, backbone, protocol, 'item_degree_ranking')
                values = record_point(panel, label, row,
                                      ('mean_x100', 'ci_low_x100', 'ci_high_x100'),
                                      paired_statistics(pop, base))
                base_token, pop_token = table4[key][0], table4[key][2]
                for method, values_by_seed, token in (('Base', base, base_token), ('Pop', pop, pop_token)):
                    compare(label + ':Table4_' + method, mean(values_by_seed.values()), token,
                            rounding_tolerance(token), 'manuscript_Table4')
                compare(label + ':Table4_difference', values[0], float(pop_token) - float(base_token),
                        rounding_tolerance(pop_token) + rounding_tolerance(base_token),
                        'difference_of_manuscript_rounded_cells')

    conditions = (('W', 'W', 'matched'), ('W', 'F', 'frozen'), ('F', 'F', 'matched'))
    body = expected['body']['figure1_panel_c_residual_minus_base']
    for offset, (fit, evaluation, family) in enumerate(conditions, 5):
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
                              paired_statistics(residual, base))
        table5 = expected['tables']['5'][offset]
        clean_label = re.sub('[^A-Za-z]', '', table5['label'])
        if not clean_label.endswith(fit + 'to' + evaluation) or len(table5['expected_numeric_tokens']) != 8:
            raise ValueError('Unexpected Table 5 residual expectation: ' + table5['label'])
        residual_token = table5['expected_numeric_tokens'][6]  # Yelp2018 MF-BPR, fourth column pair.
        base_token = table4[('yelp2018', 'MF_BPR', evaluation)][0]
        compare(label + ':Table5_Residual', mean(residual.values()), residual_token,
                rounding_tolerance(residual_token), 'manuscript_Table5')
        compare(label + ':Table5_minus_Table4', values[0], float(residual_token) - float(base_token),
                rounding_tolerance(residual_token) + rounding_tolerance(base_token),
                'difference_of_manuscript_rounded_cells')
        token = body[fit + '->' + evaluation]
        compare(label + ':manuscript_body', values[0], token, rounding_tolerance(token), 'manuscript_body')

    failed = [check for check in checks if check['status'] != 'PASS']
    return dict(status='PASS' if not failed else 'FAIL', plotted_points=len(points),
                plotted_values_checked=3*len(points), total_checks=len(checks),
                passed_checks=len(checks)-len(failed), failed_checks=len(failed),
                raw_numeric_tolerance_points=NUMERIC_ATOL_POINTS,
                interval_rule='paired five-training-seed Student-t 95% CI, df=4',
                manuscript_rule='half a printed unit; summed rounding bounds for differences of rounded cells',
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

