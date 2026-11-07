#!/usr/bin/env python3
"""Persistent GPU 1/2/5 queue with smoke-test gate and resumable references."""
import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from rmia_common import ROOT, CFG, digest, load_model, prepare, save_json, targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', default='rmia_results_seed2027')
    p.add_argument('--gpus', default='1,2,5')
    a = p.parse_args()
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'launcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.chdir(ROOT)
    gpus = a.gpus.split(',')
    if len(gpus) != len(set(gpus)):
        raise ValueError('Duplicate GPU assignments')
    active, completed = {}, []
    state = {'status': 'preflight', 'gpus': gpus, 'pid': os.getpid(),
             'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             'target_seed': 2027, 'reference_epochs': 150, 'target_retraining': False}

    def status():
        save_json(out / 'status.json', dict(state, completed=completed,
                  active={gpu: {'job': job, 'pid': proc.pid} for gpu,(job,proc,log) in active.items()}))

    def spawn(gpu, job, cmd):
        log = (out / f'{job}.log').open('a', buffering=1)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
        proc = subprocess.Popen([sys.executable, '-u', *cmd], cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        active[gpu] = (job, proc, log)
        print(f'[launch] GPU={gpu} job={job} pid={proc.pid}', flush=True)
        status()

    def train_cmd(ds, ref):
        return ['train_reference.py', '--dataset', ds, '--ref-id', str(ref), '--output-dir', str(out/ds)]

    def score_cmd(ds, smoke=False):
        cmd = ['score_offline_rmia.py', '--dataset', ds, '--output-dir', str(out/ds)]
        if smoke:
            cmd += ['--ref-ids', '0,1', '--methods', 'original', '--smoke']
        return cmd

    try:
        import torch
        torch.set_num_threads(4)
        # Fail before any expensive training if targets or input datasets are incompatible.
        audit = {}
        for ds in CFG:
            prepare(ds, out/ds)
            audit[ds] = {}
            for name, path in targets(ds).items():
                model = load_model(path, CFG[ds][0], 'cpu')
                if model.fc.out_features != CFG[ds][0]:
                    raise ValueError('Invalid target classifier dimension')
                audit[ds][name] = {'path': str(path.resolve()), 'sha256': digest(path)}
                del model
        save_json(out / 'target_manifest.json', audit)
        # Three independent reference jobs use all requested GPUs with the exact
        # single-GPU training batch/BN recipe. The first complementary pair gates
        # scoring and the CIFAR100 expansion; a third ref overlaps this initial stage.
        pending = [(f'cifar10_ref{i}', train_cmd('cifar10', i)) for i in range(3)]
        smoke_scheduled = False
        expanded = False
        eval_scheduled = set()
        state['status'] = 'running'
        while True:
            for gpu, (job, proc, log) in list(active.items()):
                code = proc.poll()
                if code is not None:
                    log.close()
                    del active[gpu]
                    if code:
                        raise RuntimeError(f'{job} failed (exit {code}); see {out / (job + ".log")}')
                    completed.append(job)
                    print(f'[done] {job}', flush=True)
            if not smoke_scheduled and all(f'cifar10_ref{i}' in completed for i in (0,1)):
                pending.insert(0, ('cifar10_smoke', score_cmd('cifar10', True)))
                smoke_scheduled = True
            if 'cifar10_smoke' in completed and not expanded:
                report = json.loads((out/'cifar10/smoke/sanity_report.json').read_text())
                if report['reference_coverage']['min'] < 1:
                    raise ValueError('Smoke OUT coverage failed')
                if not report['original_auc_above_chance']:
                    state['status'] = 'smoke_requires_review'
                    raise ValueError('Original smoke AUC <= 0.5: inspect split/recipe before expansion')
                pending += [('cifar10_ref3', train_cmd('cifar10',3))]
                pending += [(f'cifar100_ref{i}', train_cmd('cifar100',i)) for i in range(4)]
                expanded = True
            for ds in CFG:
                if expanded and ds not in eval_scheduled and all(f'{ds}_ref{i}' in completed for i in range(4)):
                    pending.insert(0, (f'{ds}_evaluate', score_cmd(ds)))
                    eval_scheduled.add(ds)
            for gpu in gpus:
                if gpu not in active and pending:
                    job, cmd = pending.pop(0)
                    spawn(gpu, job, cmd)
            status()
            if all(f'{ds}_evaluate' in completed for ds in CFG):
                break
            if not active and not pending:
                raise RuntimeError('Queue stalled before completion')
            time.sleep(10)
        rows = []
        for ds in CFG:
            with (out/ds/'evaluation/metrics.csv').open() as f:
                rows.extend(csv.DictReader(f))
        from evaluate_rmia import write_csv
        write_csv(out/'metrics.csv', rows)
        lines = ['# Offline RMIA, target seed 2027', '',
                 'Metrics are fractions. Fixed a=0.3, gamma=2, M=2000. 95% paired/stratified bootstrap intervals.', '',
                 '|Dataset|Method|AUC (95% CI)|TPR at 1% FPR|TPR at 5% FPR|',
                 '|---|---|---|---|---|']
        for r in rows:
            lines.append(f'|{r["dataset"]}|{r["target_method"]}|{float(r["auc"]):.4f} ({float(r["auc_ci_low"]):.4f}, {float(r["auc_ci_high"]):.4f})|{float(r["tpr_at_1pct_fpr"]):.4f}|{float(r["tpr_at_5pct_fpr"]):.4f}|')
        lines += ['', 'This measures sample-ID membership leakage for one forget class per dataset. It does not establish complete information removal or equivalence to retraining.',
                  'Half-data references, different-class population, existing image duplicates, and only 100 CIFAR100 nonmembers limit interpretation. See dataset data_audit.json and evaluation/config.json.']
        (out/'results.md').write_text('\n'.join(lines)+'\n')
        state['status'] = 'completed'
        status()
    except BaseException as exc:
        for job, proc, log in active.values():
            proc.terminate()
        for job, proc, log in active.values():
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()
        active.clear()
        state['status'] = 'failed'
        state['error'] = repr(exc)
        status()
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
