"""Resumable GPU queue for additional ablation seeds, followed by mean/SD plots."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from spls_ablation.collect import ABLATION_DIR, SETTINGS, LOSS_FORGET_COUNTS, PROJECT_ROOT
from spls_ablation.targets import ABLATION_VARIANTS, LOSS_VARIANTS
from spls_ablation.run_variant import PRETRAINED_DIR


def checkpoint(seed, dataset, model, shared_swin=False):
    name = f"{seed}_{dataset}_{model}_original_model.pth"
    for folder in (PRETRAINED_DIR, PROJECT_ROOT / "revision"):
        path = folder / name
        if path.is_file():
            return path
    if shared_swin and model == "swin-t":
        path = PRETRAINED_DIR / f"2027_{dataset}_{model}_original_model.pth"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No original checkpoint for seed={seed}, {dataset}/{model}")


def jobs(seeds, models, shared_swin):
    result = []
    for seed in seeds:
        for ds, model, fc in SETTINGS:
            if model not in models:
                continue
            for variant in ["min_lam0", *ABLATION_VARIANTS]:
                result.append(dict(seed=seed, dataset=ds, model=model, forget_class=fc,
                                   variant=variant, checkpoint=str(checkpoint(seed, ds, model, shared_swin))))
        for model in models:
            for fc in LOSS_FORGET_COUNTS:
                variants = LOSS_VARIANTS + (["min_lam0"] if fc == 2 else [])
                for variant in variants:
                    result.append(dict(seed=seed, dataset="cifar100", model=model, forget_class=fc,
                                       variant=variant, checkpoint=str(checkpoint(seed, "cifar100", model, shared_swin))))
    return result


def name(job):
    return f"{job['seed']}_{job['dataset']}_{job['model']}_forget{job['forget_class']}_{job['variant']}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[2028, 2029])
    p.add_argument("--models", nargs="+", choices=["resnet18", "swin-t"], default=["resnet18"])
    p.add_argument("--gpu", default="1")
    p.add_argument("--shared-swin-original", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--wait-pid", type=int, help="Preserve a running job before resuming the filtered queue")
    args = p.parse_args()
    queue = jobs(args.seeds, args.models, args.shared_swin_original)
    out = ABLATION_DIR / ("multiseed_2027_2029" if len(args.models) == 2 else "multiseed_2027_2029_" + "_".join(args.models))
    out.mkdir(exist_ok=True)
    manifest = {"gpu": args.gpu, "seeds": [2027, *args.seeds], "models": args.models,
                "shared_swin_original": args.shared_swin_original, "jobs": queue}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    lock = (out / "launcher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, MPLBACKEND="Agg", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    failures = []
    print(f"{datetime.now().isoformat()} queue={len(queue)} gpu={args.gpu}", flush=True)
    if args.wait_pid:
        proc = Path(f"/proc/{args.wait_pid}")
        print(f"Waiting for existing job PID {args.wait_pid}", flush=True)
        while proc.exists():
            try:
                if (proc / "stat").read_text().split(") ", 1)[1].startswith("Z "):
                    break
            except FileNotFoundError:
                break
            time.sleep(5)
        print("Existing job finished; resuming ResNet18 queue", flush=True)
    for i, job in enumerate(queue, 1):
        run_name = name(job)
        status = ABLATION_DIR / "ablation_logs" / f"{run_name}.status.json"
        if status.exists() and json.loads(status.read_text()).get("status") == "ok":
            print(f"SKIP {run_name}", flush=True)
            continue
        cmd = [sys.executable, "spls_ablation/run_variant.py", "--seed", str(job["seed"]),
               "--dataset", job["dataset"], "--model", job["model"],
               "--forget_class", str(job["forget_class"]), "--variant", job["variant"],
               "--original_checkpoint", job["checkpoint"]]
        print(f"{datetime.now().isoformat()} START {i}/{len(queue)} {run_name}", flush=True)
        with (ABLATION_DIR / "ablation_logs" / f"{run_name}.out").open("a") as log:
            code = subprocess.call(cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        state = json.loads(status.read_text()) if status.exists() else {}
        if code or state.get("status") != "ok":
            failures.append({"run": run_name, "exit_code": code, "status": state.get("status")})
        progress = {"finished_jobs": i, "total_jobs": len(queue), "failures": failures,
                    "last_run": run_name, "last_exit_code": code, "updated": datetime.now().isoformat()}
        (out / "progress.json").write_text(json.dumps(progress, indent=2))
        print(f"{datetime.now().isoformat()} END {run_name} exit={code}", flush=True)
        # Ordinary setup failures should not consume the entire queue.
        if code and state.get("status") != "failed_nonfinite":
            raise RuntimeError(f"Run failed; inspect {run_name}.out before resuming")
    if failures:
        raise RuntimeError(f"{len(failures)} runs failed; no final mean plots generated")
    from spls_ablation.collect_multiseed import collect
    summary = collect(ABLATION_DIR / "ablation_logs", out, [2027, *args.seeds], args.models)
    if not summary.complete.all():
        raise RuntimeError("Some conditions lack all three seeds; inspect ablation_summary.csv")
    expected = len(queue) // len(args.seeds)
    if len(summary) != expected:
        raise RuntimeError(f"Expected {expected} conditions, found {len(summary)}")
    (out / "DONE.json").write_text(json.dumps({"finished": datetime.now().isoformat(),
                                               "conditions": expected, "seeds": [2027, *args.seeds]}, indent=2))
    print("ALL DONE: mean/SD CSVs and PNG/PDF plots generated", flush=True)


if __name__ == "__main__":
    main()
