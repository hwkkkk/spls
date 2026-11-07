#!/usr/bin/env python3
"""Train one 50% stratified reference; never use retain-only LiRA shadows."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18
from trainer import train, test, optimizer_picker
from utils import seed_torch
from shadow_model import atomic_save
from rmia_common import CFG, dataset, digest, ref_path, save_json


def run(args):
    out = Path(args.output_dir)
    manifest = out / 'reference_membership.json'
    record = json.loads(manifest.read_text())
    assert record['dataset'] == args.dataset
    indices = record['training_indices'][str(args.ref_id)]
    if len(indices) != 25000 or len(set(indices)) != 25000:
        raise ValueError('Bad reference sample IDs')
    folder = out / 'references'
    folder.mkdir(exist_ok=True)
    best = ref_path(out, args.ref_id)
    last = folder / f'ref{args.ref_id}_last.pt'
    status_path = folder / f'ref{args.ref_id}.json'
    protocol = {'dataset': args.dataset, 'ref_id': args.ref_id, 'seed': 5200 + args.ref_id,
                'epochs': 150, 'batch_size': 128, 'lr': 0.1, 'momentum': 0.9,
                'weight_decay': 0.0005, 'scheduler': 'StepLR(40,0.1)',
                'manifest_sha256': digest(manifest), 'train_count': len(indices),
                'selection': 'best full test accuracy; last tie, matching original trainer',
                'note': 'Half-size training subset; original recipe otherwise unchanged. Test labels are used only for checkpoint selection, never attack tuning.'}
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if any(status.get(k) != v for k, v in protocol.items()):
            raise ValueError('Reference protocol mismatch')
        if status['status'] == 'completed':
            if digest(best) != status['checkpoint_sha256']:
                raise ValueError('Reference checkpoint hash mismatch')
            print(f'[cached] {best}', flush=True)
            return
    seed_torch(protocol['seed'])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    trainset = dataset(args.dataset, True, True)
    testset = dataset(args.dataset, False)
    loader = DataLoader(Subset(trainset, indices), batch_size=128, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)
    te = DataLoader(testset, batch_size=128, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    model = resnet18(num_classes=CFG[args.dataset][0]).cuda()
    opt = optimizer_picker('sgd', model.parameters(), 0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, 40, 0.1)
    start, best_acc, history = 0, -1., []
    if last.exists():
        ck = torch.load(last, map_location='cpu', weights_only=False)
        if ck['protocol'] != protocol:
            raise ValueError('Resume protocol mismatch')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        sched.load_state_dict(ck['scheduler'])
        start, best_acc, history = ck['epoch'], ck['best_acc'], ck['history']
        random.setstate(ck['python_rng'])
        np.random.set_state(ck['numpy_rng'])
        torch.set_rng_state(ck['torch_rng'])
        torch.cuda.set_rng_state_all(ck['cuda_rng'])
    print(f'[start] {args.dataset} ref={args.ref_id} epoch={start + 1} gpu={torch.cuda.get_device_name()}', flush=True)
    for epoch in range(start, 150):
        begin = time.monotonic()
        train(model, loader, opt, epoch, tqdm_on=False)
        train_loss, train_acc = test(model, loader)
        test_loss, test_acc = test(model, te)
        if test_acc >= best_acc:
            best_acc = test_acc
            atomic_save(model.state_dict(), best)
        sched.step()
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc,
                        'test_loss': test_loss, 'test_acc': test_acc,
                        'seconds': time.monotonic() - begin})
        atomic_save({'protocol': protocol, 'epoch': epoch + 1, 'model': model.state_dict(),
                     'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
                     'best_acc': best_acc, 'history': history,
                     'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                     'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, last)
        save_json(status_path, dict(protocol, status='training', epoch=epoch + 1,
                                   best_acc=best_acc, history=history))
        print(f'[epoch] {args.dataset} ref={args.ref_id} {epoch+1}/150 train={train_acc:.4f} test={test_acc:.4f} seconds={history[-1]["seconds"]:.1f}', flush=True)
    save_json(status_path, dict(protocol, status='completed', epoch=150, best_acc=best_acc,
                               checkpoint_sha256=digest(best), history=history))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=CFG, required=True)
    p.add_argument('--ref-id', type=int, choices=range(4), required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--num-workers', type=int, default=4)
    run(p.parse_args())
