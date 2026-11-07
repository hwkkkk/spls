#!/usr/bin/env python3
"""Deterministic softmax RMIA using candidate-specific OUT references."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from rmia_common import CFG, SOURCE, dataset, digest, load_model, ref_path, save_json, targets
from evaluate_rmia import evaluate, metrics, write_csv


def calibrated_probability(probability, a):
    if not 0 <= a <= 1:
        raise ValueError('a must be in [0,1]')
    return (1 + a) / 2 * probability + (1 - a) / 2


def attack_score(log_prob, log_bar, candidates, population, gamma=2):
    if gamma < 1 or len(set(candidates) & set(population)):
        raise ValueError('Invalid gamma or overlapping population')
    ratio = log_prob - log_bar
    if not np.isfinite(ratio).all():
        raise ValueError('Nonfinite log likelihood ratio')
    z = np.sort(ratio[population])
    # searchsorted computes exactly mean[LR(x,z)>=gamma], including ties.
    result = np.searchsorted(z, ratio[candidates] - np.log(gamma), side='right') / len(z)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite RMIA scores')
    return result


@torch.no_grad()
def collect(path, ds, out, args):
    signature = {'checkpoint_sha256': digest(path), 'dataset': ds,
                 'transform': str(dataset(ds, False).transform), 'version': 1}
    cache = out / 'logits' / (signature['checkpoint_sha256'] + '.npz')
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        with np.load(cache) as f:
            if json.loads(str(f['signature'])) != signature:
                raise ValueError('Logit cache signature mismatch')
            logits, labels = f['logits'].copy(), f['labels'].copy()
    else:
        model = load_model(path, CFG[ds][0], args.device)
        logits_list, label_list = [], []
        for train in (True, False):
            loader = DataLoader(dataset(ds, train), batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
            for step, (x, y) in enumerate(loader):
                logits_list.append(model(x.to(args.device, non_blocking=True)).float().cpu().numpy())
                label_list.append(y.numpy())
                if step % 100 == 0:
                    print(f'[logits] {ds} {path.name} train={train} {step}/{len(loader)}', flush=True)
        logits, labels = np.concatenate(logits_list), np.concatenate(label_list)
        tmp = cache.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, logits=logits, labels=labels, signature=json.dumps(signature))
        tmp.replace(cache)
        del model
        torch.cuda.empty_cache()
    if logits.shape != (60000, CFG[ds][0]) or not np.isfinite(logits).all():
        raise ValueError('Invalid checkpoint output')
    expected = np.r_[dataset(ds, True).targets, dataset(ds, False).targets]
    if not np.array_equal(labels, expected):
        raise ValueError('Dataset order mismatch')
    l = torch.from_numpy(logits).double().log_softmax(1).numpy()
    log_prob = l[np.arange(len(labels)), labels]
    return log_prob, logits.argmax(1), labels, signature


def run(args):
    torch.set_num_threads(4)
    out = Path(args.output_dir)
    manifest_path = Path(args.ref_membership_index or out / 'reference_membership.json')
    membership = json.loads(manifest_path.read_text())
    if membership['dataset'] != args.dataset:
        raise ValueError('Reference dataset mismatch')
    populations = json.loads(Path(args.population_index or out / 'population_ids.json').read_text())
    refs = [int(x) for x in args.ref_ids.split(',')]
    ref_paths = ([Path(x) for x in args.ref_checkpoints.split(',')] if args.ref_checkpoints
                 else [ref_path(out, ref) for ref in refs])
    if len(ref_paths) != len(refs) or len(set(refs)) != len(refs):
        raise ValueError('Invalid reference list')
    probs, masks, signatures = [], [], {}
    labels = None
    for ref, path in zip(refs, ref_paths):
        status = json.loads((out / 'references' / f'ref{ref}.json').read_text())
        if status['status'] != 'completed' or status['checkpoint_sha256'] != digest(path):
            raise ValueError('Incomplete or changed reference checkpoint')
        if status['manifest_sha256'] != digest(manifest_path):
            raise ValueError('Reference membership manifest mismatch')
        lp, pred, labels, sig = collect(path, args.dataset, out, args)
        ids = membership['training_indices'][str(ref)]
        if len(ids) != len(set(ids)) or min(ids) < 0 or max(ids) >= 50000:
            raise ValueError('Invalid or duplicate membership IDs')
        mask = np.ones(60000, dtype=bool)
        mask[ids] = False
        masks.append(mask)
        probs.append(np.exp(lp))
        signatures[f'ref{ref}'] = dict(sig, path=str(path.resolve()))
    masks, probs = np.stack(masks), np.stack(probs)
    counts = masks.sum(0)
    forget = args.forget_class if args.forget_class is not None else CFG[args.dataset][1]
    if forget != CFG[args.dataset][1]:
        raise ValueError('Forget class does not match the fixed target protocol')
    candidates = np.flatnonzero(labels == forget)
    nmember = int((candidates < 50000).sum())
    if len(candidates) != (6000 if args.dataset == 'cifar10' else 600):
        raise ValueError('Incorrect candidate count')
    y = (candidates < 50000).astype(int)
    population_ids = {k: np.array(v, dtype=int) + 50000 for k,v in populations.items()}
    for ids in population_ids.values():
        if len(ids) != len(set(ids)) or len(set(ids) & set(candidates)) or np.any((ids < 50000) | (ids >= 60000)):
            raise ValueError('Invalid population IDs')
    if args.population_size > len(population_ids['retain_test']):
        raise ValueError('Population size exceeds manifest')
    population = population_ids['retain_test'][:args.population_size]
    needed = np.unique(np.r_[candidates, *population_ids.values()])
    if np.any(counts[needed] == 0):
        raise ValueError('Candidate or population sample has no OUT reference')
    pbar = calibrated_probability((probs * masks).sum(0) / np.maximum(counts,1), args.a)
    logbar = np.log(np.clip(pbar, 1e-300, 1))
    paths = (json.loads(Path(args.target_checkpoints).read_text()) if args.target_checkpoints
             else targets(args.dataset, args.target_seed))
    methods = args.methods.split(',')
    results, rows, accuracies, target_logp = {}, [], {}, {}
    for method in methods:
        path = Path(paths[method])
        lp, pred, labs, sig = collect(path, args.dataset, out, args)
        if not np.array_equal(labs, labels):
            raise ValueError('Unpaired target labels')
        target_logp[method] = lp
        signatures[method] = dict(sig, path=str(path.resolve()))
        results[method] = attack_score(lp, logbar, candidates, population, args.gamma)
        accuracies[method] = {}
        for split, mask_split in [('train', np.arange(60000) < 50000), ('test', np.arange(60000) >= 50000)]:
            for group, mask_cls in [('forget', labels == forget), ('retain', labels != forget)]:
                mask = mask_split & mask_cls
                accuracies[method][f'{group}_{split}_accuracy'] = float((pred[mask] == labels[mask]).mean())
        for i, idx in enumerate(candidates):
            split = 'train' if idx < 50000 else 'test'
            rows.append({'dataset': args.dataset, 'target_method': method, 'target_seed': args.target_seed,
                         'image_id': f'{args.dataset}:{split}:{int(idx % 50000)}', 'split': split,
                         'membership': int(y[i]), 'rmia_score': results[method][i],
                         'out_ref_count': int(counts[idx])})
    destination = out / ('smoke' if args.smoke else 'evaluation')
    destination.mkdir(exist_ok=True)
    write_csv(destination / 'scores.csv', rows)
    metric_rows = evaluate(results, y, args.dataset, destination, args.bootstraps)
    coverage = {'min': int(counts[candidates].min()), 'median': float(np.median(counts[candidates])),
                'max': int(counts[candidates].max())}
    sanity = {'status': 'completed', 'reference_coverage': coverage, 'accuracies': accuracies,
              'original_auc_above_chance': bool(next((r['auc'] > .5 for r in metric_rows if r['target_method'] == 'original'), False)),
              'notes': ['AUC near 0.5 does not prove information removal.',
                        'CIFAR100 has only 100 nonmembers; low-FPR TPR is unstable.',
                        'Retain-class test population differs from the forget-class candidate distribution.',
                        'Half-size references differ from full-data targets.',
                        'Sample identity, rather than content uniqueness, defines membership; see data_audit.json.']}
    save_json(destination / 'sanity_report.json', sanity)
    config = dict(vars(args), calibration='p_bar=((1+a)*p_OUT+(1-a))/2',
                  formula_source=SOURCE, signals='ordinary softmax, temperature=1; mean OUT probability; deterministic queries',
                  difference_from_official_demo='No Taylor/soft-margin signal or query augmentation; follows rmia.md probability definition. >= comparison follows rmia.md.',
                  defaults_source='official rmia_offline_2_ref_models.yaml: a=0.3, gamma=2.0; values fixed before evaluation',
                  checkpoints=signatures, reference_coverage=coverage, n_member=nmember)
    save_json(destination / 'config.json', config)
    if not args.smoke:
        sensitivity = []
        # 1 is sample-adaptive first OUT; 2 uses the first complementary pair;
        # 4 uses all refs. No candidate is silently removed.
        for count in (1,2,4):
            selected = masks.copy()
            if count == 1:
                selected &= np.cumsum(selected, axis=0) == 1
                description = 'first available OUT per sample from four trained refs'
            else:
                selected[count:] = False
                description = f'first {count} trained refs; sample-specific OUT subset'
            n = selected.sum(0)
            if np.any(n[needed] == 0):
                raise ValueError('Sensitivity reference coverage failure')
            bar = np.log(np.clip(calibrated_probability((probs * selected).sum(0)/np.maximum(n,1), args.a),1e-300,1))
            for name, lp in target_logp.items():
                value, _ = metrics(y, attack_score(lp, bar, candidates, population, args.gamma))
                sensitivity.append({'dataset': args.dataset, 'method': name, 'scenario': f'references_{count}',
                                    'description': description, 'auc': value[0], 'tpr_at_1pct_fpr': value[1],
                                    'out_refs_min': int(n[candidates].min()), 'out_refs_max': int(n[candidates].max())})
        for pop_name, pop in population_ids.items():
            for name, lp in target_logp.items():
                value, _ = metrics(y, attack_score(lp, logbar, candidates, pop, args.gamma))
                sensitivity.append({'dataset': args.dataset, 'method': name, 'scenario': f'population_{pop_name}',
                                    'description': f'M={len(pop)}; fixed disjoint pool', 'auc': value[0], 'tpr_at_1pct_fpr': value[1],
                                    'out_refs_min': int(counts[candidates].min()), 'out_refs_max': int(counts[candidates].max())})
        write_csv(destination / 'sensitivity.csv', sensitivity)
    print(json.dumps({'dataset': args.dataset, 'metrics': metric_rows, 'coverage': coverage}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=CFG, required=True)
    p.add_argument('--target-seed', type=int, default=2027)
    p.add_argument('--forget-class', type=int)
    p.add_argument('--target-checkpoints', help='JSON mapping method names to paths')
    p.add_argument('--ref-checkpoints', help='comma-separated paths in ref-id order')
    p.add_argument('--ref-ids', default='0,1,2,3')
    p.add_argument('--ref-membership-index')
    p.add_argument('--population-index')
    p.add_argument('--gamma', type=float, default=2.)
    p.add_argument('--a', type=float, default=.3)
    p.add_argument('--population-size', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--device', default='cuda')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--methods', default='original,retrain,delete,spls')
    p.add_argument('--bootstraps', type=int, default=1000)
    p.add_argument('--smoke', action='store_true')
    run(p.parse_args())
