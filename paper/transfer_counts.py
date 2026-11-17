"""Recompute the S3 transfer outcomes reported in the paper (Table 5 and Sec. 4.3).

Usage:
  python transfer_counts.py --run <w2w3_run_dir>

Inputs (from the W2/W3 run directory):
  transfer_classification.csv, homogeneous_method_summary.csv

Rules (NDCG@20 or Recall@20, R0 main scope, D1 paired training-repeat intervals,
identity-inclusive selection unless --policy is given):
  transfer   : both 95% intervals beyond +delta or both beyond -delta
  reverse    : one 95% interval beyond +delta, the other beyond -delta
  equivalent : both 90% intervals inside [-delta, +delta]   (checked first)
  vanish     : source 95% interval beyond delta, target 90% interval inside  (post hoc split of unresolved)
  unresolved : everything else (includes 'emerge', the converse of vanish)
Frozen pairs (U->F, W->F) skip the three training-free contrasts, which coincide with matched pairs.
"""
import argparse, collections, csv, os

FIXED = {'Pop-Base', 'NPGC-Base', 'Pop-NPGC'}
SRC = {'U-W': ('U', 'W'), 'U-F': ('U', 'F'), 'W-F': ('W', 'F'),
       'U-U_to_F': ('U', 'F'), 'W-W_to_F': ('W', 'F')}
PAIRS = ['U-W', 'U-F', 'W-F', 'U-U_to_F', 'W-W_to_F']
CELLS = [('ml-1m', 'NFM'), ('ml-1m', 'MF_BPR'), ('yelp2018', 'NFM'), ('yelp2018', 'MF_BPR')]


def classify(r, ds, dt, split_vanish=True):
    sl, sh, tl, th = (float(r[k]) for k in ('source_ci95_low', 'source_ci95_high', 'target_ci95_low', 'target_ci95_high'))
    s9l, s9h, t9l, t9h = (float(r[k]) for k in ('source_ci90_low', 'source_ci90_high', 'target_ci90_low', 'target_ci90_high'))
    side = lambda lo, hi, d: 1 if lo > d else (-1 if hi < -d else 0)
    a, b = side(sl, sh, ds), side(tl, th, dt)
    s_in, t_in = (s9l > -ds and s9h < ds), (t9l > -dt and t9h < dt)
    if s_in and t_in:
        return 'E'
    if a and a == b:
        return 'T'
    if a and b:
        return 'R'
    if split_vanish and a and t_in:
        return 'V'
    return 'U'


def load(run, metric, policy, delta):
    rows = [r for r in csv.DictReader(open(os.path.join(run, 'transfer_classification.csv')))
            if r['scope'] == 'main_R0' and r['decision_basis'].startswith('D1') and r['metric'] == metric
            and r['selection_policy'] == policy and r['practical_delta_raw'] == delta]
    return [r for r in rows if not (r['analysis_family'] == 'frozen' and r['contrast'] in FIXED)]


def headroom(run):
    val = {}
    for r in csv.DictReader(open(os.path.join(run, 'homogeneous_method_summary.csv'))):
        if r['scope'] == 'main_R0' and r['analysis_family'] == 'matched' and r['selection_policy'] in ('identity_inclusive', 'native'):
            val[(r['dataset'], r['backbone'], r['protocol'], r['method'], r['metric'])] = float(r['mean_raw'])
    return lambda ds, bb, p: val[(ds, bb, p, 'base', 'oracle_ndcg20')] - val[(ds, bb, p, 'base', 'ndcg@20')]


def summarize(rows, ds_fn):
    tot, cell, ftarget = collections.Counter(), collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    for r in rows:
        k = ds_fn(r)
        tot[k] += 1
        cell[(r['comparison'], (r['dataset'], r['backbone']))][k] += 1
        if r['comparison'] != 'U-W':
            ftarget[r['dataset']][k] += 1
    return tot, cell, ftarget


def fmt(c):
    return '/'.join(str(c[x]) for x in 'TREVU')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    args = ap.parse_args()
    d = 0.0025
    rows = load(args.run, 'ndcg@20', 'identity_inclusive', '0.0025')
    tot, cell, ft = summarize(rows, lambda r: classify(r, d, d))
    print('Table 5 (T/R/E/V/U), columns:', CELLS)
    for p in PAIRS:
        print(f'  {p:10s}', *[fmt(cell[(p, c)]) for c in CELLS])
    print('All comparisons:', len(rows), dict(tot))
    print('Matched only:', dict(collections.Counter(classify(r, d, d) for r in rows if r['analysis_family'] == 'matched')))
    print('F-target per dataset:', {k: fmt(v) for k, v in ft.items()})
    no_floor = [r for r in rows if not (r['dataset'] == 'yelp2018' and r['backbone'] == 'NFM')]
    print('Outside Yelp2018 NFM:', len(no_floor), dict(collections.Counter(classify(r, d, d) for r in no_floor)))
    for dd in ('0.001', '0.005'):
        rr = load(args.run, 'ndcg@20', 'identity_inclusive', dd)
        c = collections.Counter(classify(r, float(dd), float(dd)) for r in rr)
        yf = collections.Counter(classify(r, float(dd), float(dd)) for r in rr if r['dataset'] == 'yelp2018' and r['comparison'] != 'U-W')
        print(f'delta={dd}:', dict(c), 'Yelp F-target:', dict(yf))
    tr = load(args.run, 'ndcg@20', 'trained_only', '0.0025')
    print('trained_only:', dict(collections.Counter(classify(r, d, d) for r in tr)))
    rc = load(args.run, 'recall@20', 'identity_inclusive', '0.0025')
    print('recall@20 matched:', dict(collections.Counter(classify(r, d, d) for r in rc if r['analysis_family'] == 'matched')))
    print('user-bootstrap agreement (all five repeats):', sum(r['user_seed_agrees_with_D1'] == '5' for r in rows), 'of', len(rows))
    h = headroom(args.run)
    for frac in (0.01, 0.02):
        c = collections.Counter()
        for r in rows:
            s, t = SRC[r['comparison']]
            c[classify(r, frac * h(r['dataset'], r['backbone'], s), frac * h(r['dataset'], r['backbone'], t))] += 1
        print(f'headroom margin {frac:.0%}:', dict(c))


if __name__ == '__main__':
    main()
