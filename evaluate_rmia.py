#!/usr/bin/env python3
"""Paired, stratified bootstrap over fixed candidate IDs; no attack tuning."""
import argparse
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from rmia_common import save_json


def write_csv(path, rows):
    if not rows:
        raise ValueError('Empty results')
    tmp = Path(str(path) + '.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def metrics(labels, score):
    if not np.isfinite(score).all() or len(np.unique(labels)) != 2:
        raise ValueError('Invalid attack scores or labels')
    fpr, tpr, thresholds = roc_curve(labels, score, drop_intermediate=False)
    values = [float(roc_auc_score(labels, score))]
    for budget in (.01, .05):
        values.append(float(tpr[fpr <= budget].max()))
    return np.array(values), (fpr, tpr, thresholds)


def evaluate(scores, labels, ds, out, bootstraps=1000):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if not len(pos) or not len(neg):
        raise ValueError('Missing member or nonmember samples')
    names = list(scores)
    point, curves = {}, []
    for name, score in scores.items():
        point[name], (f, t, th) = metrics(labels, score)
        curves.extend({'dataset': ds, 'target_method': name, 'fpr': float(x), 'tpr': float(y),
                       'threshold': float(z) if np.isfinite(z) else 'inf'} for x, y, z in zip(f,t,th))
    rng = np.random.default_rng(72027)
    samples = {name: np.empty((bootstraps, 3)) for name in names}
    for b in range(bootstraps):
        # Exactly the same resampled image IDs for every target method.
        ids = np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
        for name in names:
            samples[name][b] = metrics(labels[ids], scores[name][ids])[0]
    rows = []
    for name in names:
        lo, hi = np.quantile(samples[name], [.025, .975], axis=0)
        rows.append({'dataset': ds, 'target_method': name, 'auc': point[name][0],
                     'tpr_at_1pct_fpr': point[name][1], 'tpr_at_5pct_fpr': point[name][2],
                     'auc_ci_low': lo[0], 'auc_ci_high': hi[0],
                     'tpr_at_1pct_fpr_ci_low': lo[1], 'tpr_at_1pct_fpr_ci_high': hi[1],
                     'tpr_at_5pct_fpr_ci_low': lo[2], 'tpr_at_5pct_fpr_ci_high': hi[2],
                     'n_member': len(pos), 'n_nonmember': len(neg)})
    paired = []
    for i, name in enumerate(names):
        for other in names[i+1:]:
            lo, hi = np.quantile(samples[name] - samples[other], [.025, .975], axis=0)
            paired.append({'method_a': name, 'method_b': other,
                           'auc_difference': point[name][0] - point[other][0],
                           'auc_difference_ci_low': lo[0], 'auc_difference_ci_high': hi[0]})
    write_csv(out / 'metrics.csv', rows)
    write_csv(out / 'roc_curves.csv', curves)
    if paired:
        write_csv(out / 'paired_differences.csv', paired)
    save_json(out / 'bootstrap_config.json', {'replicates': bootstraps, 'seed': 72027,
              'scheme': 'stratified by membership; same candidate resamples paired across methods',
              'interval': 'percentile 95%', 'units': 'fractions (0 to 1)'})
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scores', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--bootstraps', type=int, default=1000)
    a = p.parse_args()
    rows = list(csv.DictReader(open(a.scores)))
    names = sorted({r['target_method'] for r in rows})
    grouped = {n: sorted([r for r in rows if r['target_method'] == n], key=lambda r:r['image_id']) for n in names}
    first = grouped[names[0]]
    if any([(r['image_id'], r['membership']) for r in grouped[n]] != [(r['image_id'], r['membership']) for r in first] for n in names):
        raise ValueError('Unpaired candidate IDs')
    if len({r['image_id'] for r in first}) != len(first):
        raise ValueError('Duplicate candidate IDs')
    evaluate({n: np.array([float(r['rmia_score']) for r in grouped[n]]) for n in names},
             np.array([int(r['membership']) for r in first]), first[0]['dataset'], a.output_dir, a.bootstraps)
