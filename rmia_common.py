"""Shared, fixed-index CIFAR protocol for rmia.md."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10, CIFAR100
from common import ROOT, REVISION, load_model, save_json
from utils import get_transforms

CFG = {'cifar10': (10, 6), 'cifar100': (100, 73)}
SOURCE = 'https://github.com/privacytrustlab/ml_privacy_meter/blob/d32734161a3395211fe5f3cd461932290b1fafbe/research/2024_rmia/main_ood.py#L453-L455'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dataset(ds, train, augment=False):
    tr, te = get_transforms(ds, 'resnet18', False)
    return (CIFAR10 if ds == 'cifar10' else CIFAR100)(
        ROOT / 'data', train=train, download=False, transform=tr if augment else te)


def targets(ds, seed=2027):
    paths = {'original': REVISION / f'{seed}_{ds}_resnet18_original_model.pth',
             'retrain': REVISION / f'{seed}_{ds}_resnet18_retrain_forget1_model.pth'}
    for name, token in [('delete', 'delete'), ('spls', 'ldmu')]:
        found = sorted(REVISION.glob(f'{seed}_{ds}_resnet18_forget1_{token}_*.pth'))
        if len(found) != 1:
            raise ValueError(f'Ambiguous checkpoint {ds}/{name}: {found}')
        paths[name] = found[0]
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(p)
    return paths


def prepare(ds, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train, test = dataset(ds, True), dataset(ds, False)
    labels, test_labels = np.asarray(train.targets), np.asarray(test.targets)
    classes, forget = CFG[ds]
    refs = {}
    for pair in range(2):
        rng = np.random.default_rng(5100 + pair)
        halves = [[], []]
        for cls in range(classes):
            ids = rng.permutation(np.flatnonzero(labels == cls))
            for side, indices in enumerate(np.array_split(ids, 2)):
                halves[side].extend(indices.tolist())
        for side in range(2):
            refs[str(2 * pair + side)] = sorted(halves[side])
    hashes_train = [hashlib.sha256(x.tobytes()).hexdigest() for x in train.data]
    hashes_test = [hashlib.sha256(x.tobytes()).hexdigest() for x in test.data]
    eval_hashes = {hashes_train[i] for i in np.flatnonzero(labels == forget)}
    eval_hashes.update(hashes_test[i] for i in np.flatnonzero(test_labels == forget))
    # A population image must be genuinely OUT by identity as well as sample ID.
    train_hash_set = set(hashes_train)
    eligible = np.array([i for i, h in enumerate(hashes_test)
                         if h not in train_hash_set and h not in eval_hashes])
    rng = np.random.default_rng(62027)
    population = rng.choice(eligible[test_labels[eligible] != forget], 2000, replace=False)
    balanced = []
    for cls in range(classes):
        pool = eligible[test_labels[eligible] == cls]
        balanced.extend(rng.choice(pool, min(len(pool), 2000 // classes), replace=False).tolist())
    populations = {'retain_test': sorted(population.tolist()),
                   'all_test_disjoint': sorted(eligible.tolist()),
                   'balanced_test_disjoint': sorted(balanced)}
    record = {'dataset': ds, 'split_seeds': [5100, 5101], 'training_indices': refs,
              'index_namespace': 'original CIFAR train index',
              'train_data_sha256': hashlib.sha256(train.data.tobytes()).hexdigest(),
              'test_data_sha256': hashlib.sha256(test.data.tobytes()).hexdigest()}
    for name, obj in [('reference_membership.json', record), ('population_ids.json', populations)]:
        path = out / name
        if path.exists() and json.loads(path.read_text()) != obj:
            raise ValueError(f'Incompatible existing manifest: {path}')
        save_json(path, obj)
    for pair in range(2):
        a, b = map(set, (refs[str(2 * pair)], refs[str(2 * pair + 1)]))
        if a & b or len(a | b) != 50000:
            raise ValueError('Invalid complementary partition')
    save_json(out / 'data_audit.json', {
        'unique_train_ids': 50000, 'unique_test_ids': 10000,
        'duplicate_train_image_count': 50000 - len(set(hashes_train)),
        'duplicate_test_image_count': 10000 - len(set(hashes_test)),
        'train_test_identical_image_hashes': len(train_hash_set & set(hashes_test)),
        'forget_train_test_identical_image_hashes': len(
            {hashes_train[i] for i in np.flatnonzero(labels == forget)} &
            {hashes_test[i] for i in np.flatnonzero(test_labels == forget)}),
        'population_excludes_train_and_evaluation_content': True,
        'note': 'Image-content duplicates are reported; sample-ID membership labels are preserved.'})
    return record, populations


def ref_path(out, ref):
    return Path(out) / 'references' / f'ref{ref}_best.pth'
