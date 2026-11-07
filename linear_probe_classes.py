#!/usr/bin/env python3
"""Evaluate each seed's forget class separately; no averaging across classes."""
import csv
import hashlib
import json
import os
import random
import traceback
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import datasets

from linear_probe import ROOT, REVISION, accuracy, extract, load_model, save_json, train_probe
from utils import get_transforms

OUT = ROOT / 'linear_probe_results_classes_2027_2031'


def write_rows(rows):
    if not rows:
        return
    path = OUT / 'lp_by_class.csv'
    with path.with_suffix('.tmp').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix('.tmp').replace(path)
    lines = ['# Linear probing by forget class', '',
             'Each seed identifies its own one-class unlearning checkpoint. No cross-class averaging.',
             'Output metrics are re-evaluated on the full test set. All train images are used for probing.',
             'Deterministic test preprocessing includes the checkpoint-required 224x224 resize.', '',
             '| Dataset | Seed | Forget class | Method | Output Retain | Output Forget | LP Retain | LP Forget | Epochs |',
             '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| {dataset} | {seed} | {forget_class} | {method} | {output_retain:.2f} | {output_forget:.2f} | {lp_retain:.2f} | {lp_forget:.2f} | {epochs} |'.format(**r))
    (OUT / 'results.md').write_text('\n'.join(lines) + '\n')


def main():
    OUT.mkdir(exist_ok=True)
    for folder in ('features', 'probes'):
        (OUT / folder).mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    jobs = []
    for ds, classes in [('cifar10', 10), ('cifar100', 100)]:
        cfg = OmegaConf.load(ROOT / f'config/{ds}_resnet18.yaml')
        for seed in range(2027, 2032):
            perm = list(cfg.permutation_map)
            random.Random(seed).shuffle(perm)
            for method, suffix in [('original', 'original_model'), ('retrain', 'retrain_forget1_model'),
                                   ('delete', 'forget1_delete_*'), ('spls', 'forget1_ldmu_*')]:
                matches = sorted(REVISION.glob(f'{seed}_{ds}_resnet18_{suffix}.pth'))
                if len(matches) != 1:
                    raise ValueError((seed, ds, method, matches))
                jobs.append(dict(dataset=ds, seed=seed, forget_class=int(perm[0]), method=method,
                                 checkpoint=str(matches[0]), classes=classes))
    manifest = dict(gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), device=torch.cuda.get_device_name(0),
                    probe_seed=0, optimizer='Adam', lr=0.001, batch_size=1024,
                    scheduler='CosineAnnealingLR', models=jobs, aggregation='none')
    save_json(OUT / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    for ds, factory in [('cifar10', datasets.CIFAR10), ('cifar100', datasets.CIFAR100)]:
        _, transform = get_transforms(ds, 'resnet18', wo_dataaug=True)
        train = factory(ROOT / 'data', train=True, download=False, transform=transform)
        test = factory(ROOT / 'data', train=False, download=False, transform=transform)
        assert len(train) == 50000 and len(test) == 10000
        loaders = [DataLoader(d, batch_size=128, shuffle=False, num_workers=4,
                              pin_memory=True, persistent_workers=True) for d in (train, test)]
        for job in [j for j in jobs if j['dataset'] == ds]:
            key = f"{ds}_{job['seed']}_{job['method']}"
            cache_path = OUT / 'features' / f'{key}.pt'
            job['cache'] = str(cache_path)
            stat = Path(job['checkpoint']).stat()
            signature = hashlib.sha256(f"{job['checkpoint']}:{stat.st_size}:{stat.st_mtime_ns}:{transform}:v1".encode()).hexdigest()
            if cache_path.exists():
                cache = torch.load(cache_path, weights_only=True)
                valid = cache.get('signature') == signature
                del cache
                if valid:
                    print(f'[cached] {key}', flush=True)
                    continue
            save_json(OUT / 'status.json', dict(status='extracting', current=key,
                                               completed_features=len(list((OUT / 'features').glob('*.pt'))), total=40))
            print(f'[extract] {key} forget_class={job["forget_class"]}', flush=True)
            model = load_model(job['checkpoint'], job['classes'], device)
            Xtr, ytr, _ = extract(model, loaders[0], device)
            Xte, yte, pred = extract(model, loaders[1], device)
            assert torch.equal(ytr, torch.tensor(train.targets))
            assert torch.equal(yte, torch.tensor(test.targets))
            metrics = accuracy(pred, yte, job['forget_class'])
            cache = dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, output=metrics,
                         forget_class=job['forget_class'], signature=signature)
            torch.save(cache, cache_path.with_suffix('.tmp'))
            cache_path.with_suffix('.tmp').replace(cache_path)
            print(f'[output] {key}: {metrics}', flush=True)
            del model, cache, Xtr, Xte, ytr, yte, pred
            torch.cuda.empty_cache()
        del loaders
    # Calibrate all CIFAR-100 originals before choosing one epoch count for every model.
    epochs, calibration = 100, {}
    for job in [j for j in jobs if j['dataset'] == 'cifar100' and j['method'] == 'original']:
        key = f"{job['dataset']}_{job['seed']}_{job['method']}"
        save_json(OUT / 'status.json', dict(status='calibrating', current=key))
        cache = torch.load(job['cache'], weights_only=True)
        lp = train_probe(cache, 100, 100, device, OUT / 'probes' / f'{key}_e100.pt')
        calibration[key] = lp
        if cache['output']['overall'] - lp['overall'] > 1:
            epochs = 200
        del cache
    manifest.update(epochs=epochs, calibration=calibration)
    save_json(OUT / 'manifest.json', manifest)
    rows = []
    for job in jobs:
        key = f"{job['dataset']}_{job['seed']}_{job['method']}"
        save_json(OUT / 'status.json', dict(status='probing', current=key, completed=len(rows), total=len(jobs), epochs=epochs))
        cache = torch.load(job['cache'], weights_only=True)
        print(f'[probe] {key} epochs={epochs}', flush=True)
        lp = calibration[key] if key in calibration and epochs == 100 else train_probe(
            cache, job['classes'], epochs, device, OUT / 'probes' / f'{key}_e{epochs}.pt')
        row = {k: job[k] for k in ('dataset', 'seed', 'forget_class', 'method', 'checkpoint')}
        row.update({f'output_{k}': v for k, v in cache['output'].items()})
        row.update({f'lp_{k}': v for k, v in lp.items()})
        row.update(epochs=epochs, original_sanity=(abs(lp['overall'] - cache['output']['overall']) <= 1) if job['method'] == 'original' else '')
        rows.append(row)
        write_rows(rows)
        print(f'[result] {json.dumps(row)}', flush=True)
        del cache
    save_json(OUT / 'status.json', dict(status='completed', rows=len(rows), epochs=epochs,
                                       sanity_failures=[r for r in rows if r['original_sanity'] is False]))
    print('[done]', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        OUT.mkdir(exist_ok=True)
        save_json(OUT / 'status.json', dict(status='failed', traceback=traceback.format_exc()))
        raise
