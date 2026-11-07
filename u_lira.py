#!/usr/bin/env python3
"""U-LiRA pipeline: concurrent IN shadows, method-specific unlearning and held-out/LOO attacks."""
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

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100

from linear_probe import ROOT, REVISION, load_model, save_json
from lira import shadow_stats
from shadow_model import train_shadow, atomic_save
from utils import get_transforms, get_unlearn_loader, seed_torch

CFG = {"cifar10": (10, 6, 5000), "cifar100": (100, 73, 500)}
IN_SEEDS = (4001, 4002, 4003, 4004)
OUT_SEEDS = (3001, 3002, 3003, 3004)
METHODS = ("none", "delete", "spls")


def target_paths(ds):
    result = {"none": REVISION / f"2027_{ds}_resnet18_original_model.pth",
              "retrain": REVISION / f"2027_{ds}_resnet18_retrain_forget1_model.pth"}
    for method, token in (("delete", "delete"), ("spls", "ldmu")):
        candidates = sorted(REVISION.glob(f"2027_{ds}_resnet18_forget1_{token}_*.pth"))
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous target {ds}/{method}: {candidates}")
        result[method] = candidates[0]
    for path in result.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return result


def forget_loader(ds, workers):
    _, forget, count = CFG[ds]
    _, transform = get_transforms(ds, "resnet18", True)
    factory = CIFAR10 if ds == "cifar10" else CIFAR100
    data = factory(ROOT / "data", train=True, download=False, transform=transform)
    indices = np.flatnonzero(np.asarray(data.targets) == forget)
    assert len(indices) == count
    return DataLoader(Subset(data, indices.tolist()), batch_size=128, shuffle=False,
                      num_workers=workers, pin_memory=True), indices


@torch.no_grad()
def collect(checkpoint, ds, out_file, workers):
    stat = checkpoint.stat()
    signature = f"{checkpoint.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:v1"
    if out_file.exists():
        with np.load(out_file) as saved:
            if str(saved["signature"]) == signature:
                return {key: saved[key].copy() for key in saved.files}
    classes, forget, count = CFG[ds]
    loader, indices = forget_loader(ds, workers)
    model = load_model(checkpoint, classes, "cuda")
    phis, correct = [], 0
    for x, labels in loader:
        assert (labels == forget).all()
        logits = model(x.to("cuda", non_blocking=True)).float()
        assert logits.shape[1] == classes
        correct += int((logits.argmax(1).cpu() == labels).sum())
        rest = torch.cat((logits[:, :forget], logits[:, forget + 1:]), 1)
        phis.append((logits[:, forget] - torch.logsumexp(rest, 1)).cpu())
    phi = torch.cat(phis).numpy().astype(np.float64)
    assert phi.shape == (count,) and np.isfinite(phi).all()
    result = {"phi": phi, "indices": indices, "accuracy": np.array(100 * correct / count),
              "signature": np.array(signature)}
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **result)
    tmp.replace(out_file)
    del model, loader
    torch.cuda.empty_cache()
    return result


def preflight(out, workers):
    checks = {}
    for ds in CFG:
        checks[ds] = {}
        for method, path in target_paths(ds).items():
            record = collect(path, ds, out / ds / "phi" / f"target_{method}.npz", workers)
            acc = float(record["accuracy"])
            checks[ds][method] = {"forget_train_acc": acc, "checkpoint": str(path)}
            print(f"[class check] {ds} {method} forget_train_acc={acc:.2f}%", flush=True)
            if method != "none" and acc != 0:
                raise ValueError(f"Target forget class check failed: {ds} {method} {acc}%")
            if method == "none" and acc < 80:
                raise ValueError(f"Original forget class accuracy unexpectedly low: {ds} {acc}%")
        save_json(out / "preflight.json", checks)
    return checks


def unlearn(ds, method, original, seed, root, workers, spls_soft_label="lowest"):
    import log_utils
    from method.ldmu import ldmu
    from method.delete import delete
    if spls_soft_label not in ("lowest", "lowest-std"):
        raise ValueError(f"Unsupported SPLS objective: {spls_soft_label}")
    soft_label = spls_soft_label if method == "spls" else "inf"
    destination = root / ds / method / f"seed{seed}.pth"
    experiment = destination.parent / f"seed{seed}_logs"
    if destination.exists():
        if method == "spls":
            config_path = experiment / "config.json"
            if not config_path.exists() or json.loads(config_path.read_text()).get("soft_label") != soft_label:
                raise ValueError(f"Cached SPLS objective does not match {soft_label}: {destination}. Use a separate output directory.")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    experiment.mkdir(exist_ok=True)
    seed_torch(seed)
    _, forget, _ = CFG[ds]
    factory = CIFAR10 if ds == "cifar10" else CIFAR100
    train_transform, test_transform = get_transforms(ds, "resnet18", False)
    trainset = factory(ROOT / "data", train=True, download=False, transform=train_transform)
    testset = factory(ROOT / "data", train=False, download=False, transform=test_transform)
    loaders = get_unlearn_loader(trainset, testset, [forget], 128, float("inf"), workers)
    loader_dict = dict(zip(("train_forget", "train_remain", "test_forget", "test_remain"), loaders[:4]))
    model = load_model(original, CFG[ds][0], "cuda")
    model.requires_grad_(True)
    logger, handler = log_utils.setup_logger(experiment, logger_name=f"{ds}_{method}_{seed}")
    save_json(experiment / "config.json", {"dataset": ds, "seed": seed, "forget": forget,
              "original": str(original), "method": method, "epochs": 20,
              "optimizer": "SGD", "lr": 0.003 if method == "spls" else 0.001,
              "momentum": 0.9, "weight_decay": 0, "batch_size": 128,
              "soft_label": soft_label, "k": 1.0 if soft_label == "lowest-std" else None,
              "target_rule": "min" if soft_label == "lowest" else ("min-std" if soft_label == "lowest-std" else "inf"),
              "freeze_linear": False, "disable_bn": False,
              "implementation": "method.ldmu.ldmu" if method == "spls" else "method.delete.delete",
              "epoch_evaluations": "all four loaders, matching main.py"})
    kwargs = dict(ori_model=model, train_forget_loader=loaders[0], unlearn_epoch=20,
                  unlearn_rate=0.003 if method == "spls" else 0.001,
                  logger=logger, console_handler=handler, loader_dict=loader_dict,
                  experiment_path=experiment, disable_bn=False,
                  soft_label=soft_label)
    result = ldmu(k=1.0, **kwargs) if method == "spls" else delete(**kwargs)
    atomic_save(result.state_dict(), destination)
    del result, model, loaders
    torch.cuda.empty_cache()
    return destination


def worker(ds, seed, root, workers):
    # One unique worker owns each dataset/seed directory; original training resumes by epoch.
    torch.cuda.set_per_process_memory_fraction(0.25)
    original = train_shadow(seed, root / ds / "original", workers, ds, mode="original")
    paths = {"none": original}
    for method in ("spls", "delete"):
        print(f"[unlearning] {ds} seed={seed} {method}", flush=True)
        paths[method] = unlearn(ds, method, original, seed, root, workers)
    metrics = {}
    for method, path in paths.items():
        rec = collect(path, ds, root / ds / "phi" / f"{method}_{seed}.npz", workers)
        metrics[method] = float(rec["accuracy"])
    save_json(root / ds / f"worker_{seed}.json", {"status": "completed", "dataset": ds,
               "seed": seed, "forget_train_acc": metrics,
               "unlearning_zero_accuracy": all(metrics[m] == 0 for m in ("spls", "delete"))})
    print(f"[worker done] {ds} seed={seed}: {metrics}", flush=True)


def ulira_scores(phi, P_in, P_out):
    mu_i, sig_i = shadow_stats(P_in)
    mu_o, sig_o = shadow_stats(P_out)
    return (-0.5 * ((phi - mu_i) / sig_i) ** 2 - np.log(sig_i)
            + 0.5 * ((phi - mu_o) / sig_o) ** 2 + np.log(sig_o))


def evaluate_pair(lam_in, lam_out):
    accuracy = 50 * ((lam_in > 0).mean() + (lam_out <= 0).mean())
    labels = np.r_[np.ones(len(lam_in)), np.zeros(len(lam_out))]
    auc = roc_auc_score(labels, np.r_[lam_in, lam_out]) * 100
    return float(accuracy), float(auc)


def low_fpr_point(lam_positive, lam_negative, max_fpr=0.01):
    """Best empirical ROC point with FPR <= budget; no interpolation or tie splitting."""
    labels = np.r_[np.ones(len(lam_positive)), np.zeros(len(lam_negative))]
    fpr, tpr, thresholds = roc_curve(labels, np.r_[lam_positive, lam_negative], drop_intermediate=False)
    eligible = np.flatnonzero(fpr <= max_fpr)
    index = eligible[np.argmax(tpr[eligible])]
    threshold = float(thresholds[index])
    return {"tpr": float(tpr[index] * 100), "fpr": float(fpr[index] * 100),
            "threshold": threshold if np.isfinite(threshold) else None,
            "reject_all": bool(not np.isfinite(threshold))}


def retrain_control(target_phi, P_in, P_out):
    """Compare two independent OUT models under the same method-specific attack.

    Retrain-2027 is arbitrarily positive; OUT[k] is negative. Remove OUT[k]
    from calibration for BOTH targets, retaining all four method-specific INs.
    """
    folds, scores = [], {}
    for k, seed in enumerate(OUT_SEEDS):
        calibration_out = np.delete(P_out, k, axis=0)
        positive = ulira_scores(target_phi, P_in, calibration_out)
        negative = ulira_scores(P_out[k], P_in, calibration_out)
        accuracy, auc = evaluate_pair(positive, negative)
        point = low_fpr_point(positive, negative)
        folds.append({"positive_seed": 2027, "negative_seed": seed,
                      "calibration_out_seeds": [s for s in OUT_SEEDS if s != seed],
                      "accuracy": accuracy, "auc": auc, "tpr_at_1pct_fpr": point["tpr"],
                      "achieved_fpr": point["fpr"], "roc_threshold": point["threshold"],
                      "roc_reject_all": point["reject_all"]})
        scores[f"positive_{seed}"] = positive
        scores[f"negative_{seed}"] = negative
    summary = {}
    for key in ("accuracy", "auc", "tpr_at_1pct_fpr"):
        values = np.array([fold[key] for fold in folds])
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=0))
    return summary, folds, scores


def out_root(ds):
    return ROOT / ("lira_results_seed2027" if ds == "cifar10" else "lira_results_cifar100_seed2027")


def out_ready(ds):
    folder = out_root(ds) / "shadows"
    return all((folder / f"shadow_seed{s}.json").is_file()
               and json.loads((folder / f"shadow_seed{s}.json").read_text()).get("status") == "completed"
               and (folder / f"shadow_seed{s}_best.pth").is_file() for s in OUT_SEEDS)


def evaluate_dataset(ds, root, workers):
    _, _, n = CFG[ds]
    target_retrain = np.load(root / ds / "phi/target_retrain.npz")
    expected_indices = target_retrain["indices"]
    old = out_root(ds)
    if (old / "shadow_statistics.npz").is_file() and (old / "sample_indices.npz").is_file():
        with np.load(old / "sample_indices.npz") as saved:
            assert np.array_equal(saved["member_indices"], expected_indices)
        with np.load(old / "shadow_statistics.npz") as saved:
            P_out = saved["phi"][:, :n].copy()
    else:
        records = [collect(old / "shadows" / f"shadow_seed{s}_best.pth", ds,
                           root / ds / "phi" / f"out_{s}.npz", workers) for s in OUT_SEEDS]
        assert all(np.array_equal(r["indices"], expected_indices) for r in records)
        P_out = np.stack([r["phi"] for r in records])
    assert P_out.shape == (4, n)
    out_checks = [collect(old / "shadows" / f"shadow_seed{s}_best.pth", ds,
                         root / ds / "phi" / f"out_{s}.npz", workers) for s in OUT_SEEDS]
    assert all(np.array_equal(r["indices"], expected_indices) for r in out_checks)
    assert np.allclose(P_out, np.stack([r["phi"] for r in out_checks]), rtol=1e-5, atol=1e-5)
    rows, diagnostics, control_rows = [], {}, []
    for method in METHODS:
        recs = [np.load(root / ds / "phi" / f"{method}_{s}.npz") for s in IN_SEEDS]
        assert all(np.array_equal(rec["indices"], expected_indices) for rec in recs)
        P_in = np.stack([rec["phi"] for rec in recs])
        target = np.load(root / ds / "phi" / f"target_{method}.npz")
        assert np.array_equal(target["indices"], expected_indices)
        lam_t = ulira_scores(target["phi"], P_in, P_out)
        lam_r = ulira_scores(target_retrain["phi"], P_in, P_out)
        acc, auc = evaluate_pair(lam_t, lam_r)
        target_point = low_fpr_point(lam_t, lam_r)
        folds, scores = [], {}
        loo_points = []
        for k in range(4):
            # Match u-lira.md: exclude the target only from its own distribution.
            a = ulira_scores(P_in[k], np.delete(P_in, k, 0), P_out)
            b = ulira_scores(P_out[k], P_in, np.delete(P_out, k, 0))
            folds.append(evaluate_pair(a, b))
            loo_points.append(low_fpr_point(a, b))
            scores[f"loo_in_{k}"] = a
            scores[f"loo_out_{k}"] = b
        values = np.asarray(folds)
        rows.append({"dataset": ds, "method": method, "accuracy_target": acc, "auc_target": auc,
                     "accuracy_loo_mean": float(values[:, 0].mean()),
                     "accuracy_loo_std": float(values[:, 0].std()),
                     "auc_loo_mean": float(values[:, 1].mean()), "auc_loo_std": float(values[:, 1].std()),
                     "tpr_at_1pct_fpr_target": target_point["tpr"],
                     "tpr_at_1pct_fpr_loo_mean": float(np.mean([p["tpr"] for p in loo_points])),
                     "tpr_at_1pct_fpr_loo_std": float(np.std([p["tpr"] for p in loo_points], ddof=0))})
        control_summary, control_folds, control_scores = retrain_control(target_retrain["phi"], P_in, P_out)
        control_rows.append({"dataset": ds, "attacker": method, **control_summary})
        np.savez_compressed(root / ds / f"{method}_retrain_control.npz", indices=expected_indices, **control_scores)
        mu_i, sig_i = shadow_stats(P_in)
        mu_o, sig_o = shadow_stats(P_out)
        diagnostics[method] = {"mean_phi_in": float(P_in.mean()), "mean_phi_out": float(P_out.mean()),
                               "mean_phi_target": float(target["phi"].mean()),
                               "mean_phi_retrain": float(target_retrain["phi"].mean()),
                               "sigma_in": sig_i, "sigma_out": sig_o, "folds": folds,
                               "shadow_forget_train_acc": [float(rec["accuracy"]) for rec in recs],
                               "target_low_fpr": target_point, "loo_low_fpr": loo_points,
                               "retrain_control": {"summary": control_summary, "folds": control_folds}}
        np.savez_compressed(root / ds / f"{method}_attack.npz", P_in=P_in, P_out=P_out,
                            mu_in=mu_i, mu_out=mu_o, sigma_in=sig_i, sigma_out=sig_o,
                            lambda_target=lam_t, lambda_retrain=lam_r, indices=expected_indices, **scores)
    diagnostics["control_pass"] = rows[0]["accuracy_target"] >= 95
    diagnostics["out_shadow_forget_train_acc"] = [float(r["accuracy"]) for r in out_checks]
    diagnostics["original_test_accuracy"] = {str(s): json.loads((root / ds / "original" /
        f"shadow_seed{s}.json").read_text())["best_acc"] * 100 for s in IN_SEEDS}
    lp_csv = ROOT / "linear_probe_results_seed2027/lp_seed2027.csv"
    if lp_csv.exists():
        with lp_csv.open() as f:
            original_row = next(r for r in csv.DictReader(f)
                                if r["dataset"] == ds and r["method"] == "original")
        diagnostics["target_original_test_accuracy"] = float(original_row["output_overall"])
    save_json(root / ds / "diagnostics.json", diagnostics)
    save_json(root / ds / "rows.json", rows)
    save_json(root / ds / "retrain_control_rows.json", control_rows)
    return rows


def report(root):
    rows = []
    for ds in CFG:
        if (root / ds / "rows.json").exists():
            rows += json.loads((root / ds / "rows.json").read_text())
    if not rows:
        return
    with (root / "u_lira_seed2027.csv.tmp").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "u_lira_seed2027.csv.tmp").replace(root / "u_lira_seed2027.csv")
    lines = ["# U-LiRA: ResNet-18, target seed 2027", "",
             "Four IN seeds 4001–4004 and four OUT seeds 3001–3004 per dataset.",
             "CIFAR-10: class 6, 5,000 forget train samples; CIFAR-100: class 73, 500 samples.",
             "Test preprocessing, no augmentation/flip during queries. Decision threshold: log likelihood ratio > 0.",
             "LOO excludes each target from its own distribution and uses all four opposite shadows.",
             "LOO standard deviations use ddof=0 and reflect four model pairs, not sample confidence intervals.", "",
             "| Dataset | Method | Accuracy target | AUC target | TPR@1% FPR target | Accuracy LOO | AUC LOO | TPR@1% FPR LOO |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("spls_target_rule"):
            lines.insert(2, f"SPLS IN objective: {manifest['spls_target_rule']} (soft_label={manifest['spls_soft_label']}).")
    for r in rows:
        target_tpr = f"{r['tpr_at_1pct_fpr_target']:.2f}" if 'tpr_at_1pct_fpr_target' in r else 'pending'
        loo_tpr = (f"{r['tpr_at_1pct_fpr_loo_mean']:.2f} ± {r['tpr_at_1pct_fpr_loo_std']:.2f}"
                   if 'tpr_at_1pct_fpr_loo_mean' in r else 'pending')
        lines.append(("| {dataset} | {method} | {accuracy_target:.2f} | {auc_target:.2f} | " + target_tpr + " | "
                     "{accuracy_loo_mean:.2f} ± {accuracy_loo_std:.2f} | {auc_loo_mean:.2f} ± {auc_loo_std:.2f} | " + loo_tpr + " |").format(**r))
    lines += ["", "All metrics are percentages. Target positives are method-2027 scores and negatives are Retrain-2027 scores on the SAME forget-train samples; forget-test samples are not used.",
              "TPR@1% FPR is the maximum empirical ROC TPR at FPR <= 1%, without interpolation or splitting ties. ROC prediction uses score >= threshold.",
              "Its ROC threshold varies per evaluated pair and is separate from the fixed LLR > 0 accuracy threshold; it is a descriptive ROC metric, not a separately calibrated deployment threshold.",
              "Negative counts are 5,000 (CIFAR-10) and 500 (CIFAR-100); FPR <= 1% allows at most 50 and 5 false positives respectively."]
    control_rows = []
    for ds in CFG:
        control_path = root / ds / "retrain_control_rows.json"
        if control_path.exists():
            control_rows += json.loads(control_path.read_text())
    if control_rows:
        with (root / "retrain_control.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(control_rows[0]))
            writer.writeheader()
            writer.writerows(control_rows)
        lines += ["", "## Retrain versus Retrain control", "",
                  "Independent models: Retrain-2027 (arbitrarily positive) versus retrain shadow k=3001–3004 (negative).",
                  "For each pair, use the SAME existing method-specific LLR for both targets, calibrated on four IN shadows and the OTHER three OUT shadows. Neither compared retrain model is in calibration.",
                  "This tests the existing attack's behavior on two retrains; it does not fit a separate attack with retrain models as IN shadows.",
                  "Report mean ± population std (ddof=0) across four pairs. Pairs share Retrain-2027 and calibration models; these are not confidence intervals.",
                  "Chance reference: AUC 50%, balanced accuracy 50%, TPR approximately 1% at 1% FPR; finite model/sample effects can cause deviations.", "",
                  "| Dataset | Existing attacker | Accuracy at LLR > 0 | AUC | TPR@1% FPR |",
                  "|---|---|---:|---:|---:|"]
        for r in control_rows:
            lines.append(("| {dataset} | {attacker} | {accuracy_mean:.2f} ± {accuracy_std:.2f} | "
                          "{auc_mean:.2f} ± {auc_std:.2f} | {tpr_at_1pct_fpr_mean:.2f} ± {tpr_at_1pct_fpr_std:.2f} |").format(**r))
    for ds in CFG:
        path = root / ds / "diagnostics.json"
        if path.exists():
            diag = json.loads(path.read_text())
            lines += ["", f"{ds}: NONE positive control (target accuracy >=95%): {diag['control_pass']}"]
            bad = {m: v["shadow_forget_train_acc"] for m, v in diag.items()
                   if m in ("spls", "delete") and any(a != 0 for a in v["shadow_forget_train_acc"])}
            if bad:
                lines += [f"Review needed: nonzero IN-shadow forget accuracy: {bad}"]
    (root / "results.md").write_text("\n".join(lines) + "\n")


def supervise(args, root):
    jobs = [(ds, seed) for seed in IN_SEEDS for ds in CFG]
    running, attempts = {}, {}
    limit = args.concurrency
    completed = []
    while jobs or running:
        while jobs and len(running) < limit:
            ds, seed = jobs.pop(0)
            done_path = root / ds / f"worker_{seed}.json"
            if done_path.exists() and json.loads(done_path.read_text()).get("status") == "completed":
                completed.append((ds, seed))
                continue
            log_path = root / "logs" / f"{ds}_{seed}.log"
            log = log_path.open("a")
            command = [sys.executable, "-u", str(ROOT / "u_lira.py"), "--worker", "--dataset", ds,
                       "--seed", str(seed), "--output-dir", str(root), "--num-workers", str(args.num_workers)]
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
            log.close()
            running[(ds, seed)] = (proc, log_path)
            attempts[(ds, seed)] = attempts.get((ds, seed), 0) + 1
            print(f"[launch] {ds} seed={seed} pid={proc.pid} concurrency={limit}", flush=True)
        for key, (proc, log_path) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[key]
            if rc == 0:
                completed.append(key)
            elif attempts[key] < 4:
                tail = log_path.read_text()[-5000:]
                if "out of memory" in tail.lower() or rc == -9:
                    limit = 1
                jobs.insert(0, key)
                print(f"[retry] {key} rc={rc} attempt={attempts[key]} concurrency={limit}\n{tail[-1500:]}", flush=True)
            else:
                raise RuntimeError(f"Worker {key} failed four times; inspect {log_path}")
        save_json(root / "status.json", {"status": "training_in_shadows", "pid": os.getpid(),
                  "completed": completed, "running": [{"dataset": ds, "seed": seed, "pid": p.pid}
                    for (ds, seed), (p, _) in running.items()], "pending": jobs, "concurrency": limit})
        for ds in CFG:
            if not (root / ds / "rows.json").exists() and out_ready(ds) and all(
                (root / ds / f"worker_{seed}.json").exists() for seed in IN_SEEDS):
                print(f"[evaluate ready dataset] {ds}", flush=True)
                evaluate_dataset(ds, root, args.num_workers)
                report(root)
        time.sleep(10)
    for ds in CFG:
        if (root / ds / "rows.json").exists():
            continue
        last_notice = 0
        while not out_ready(ds):
            save_json(root / "status.json", {"status": "waiting_out_shadows", "dataset": ds, "pid": os.getpid()})
            if time.monotonic() - last_notice > 300:
                print(f"[waiting] {ds} OUT shadow training", flush=True)
                last_notice = time.monotonic()
            time.sleep(30)
        print(f"[evaluate] {ds}", flush=True)
        evaluate_dataset(ds, root, args.num_workers)
        report(root)
    controls = {ds: json.loads((root / ds / "diagnostics.json").read_text())["control_pass"] for ds in CFG}
    save_json(root / "status.json", {"status": "completed" if all(controls.values()) else "completed_needs_control_review",
                                    "positive_controls": controls})
    print(f"[done] {root / 'results.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="u_lira_results_seed2027")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate existing shadows without training")
    parser.add_argument("--dataset", choices=tuple(CFG))
    parser.add_argument("--seed", type=int, choices=IN_SEEDS)
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    root = Path(args.output_dir).resolve()
    (root / "logs").mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args.dataset, args.seed, root, args.num_workers)
        return
    lock = (root / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if args.evaluate_only:
            for ds in CFG:
                evaluate_dataset(ds, root, args.num_workers)
            report(root)
            print(f"[evaluated existing models] {root / 'results.md'}", flush=True)
            return
        if args.preflight:
            preflight(root, args.num_workers)
            return
        # Keep the controller free of CUDA contexts before spawning workers.
        command = [sys.executable, "-u", str(ROOT / "u_lira.py"), "--preflight",
                   "--output-dir", str(root / "preflight_temp"), "--num-workers", str(args.num_workers)]
        # Preflight runs in a separate process, then its artifacts are moved into this run.
        subprocess.run(command, check=True)
        import shutil
        for ds in CFG:
            shutil.copytree(root / "preflight_temp" / ds, root / ds, dirs_exist_ok=True)
        shutil.copy2(root / "preflight_temp/preflight.json", root / "preflight.json")
        save_json(root / "manifest.json", {"protocol": str(ROOT / "u-lira.md"), "target_seed": 2027,
                  "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "in_seeds": IN_SEEDS, "out_seeds": OUT_SEEDS,
                  "concurrency": args.concurrency, "epochs": 150, "batch_size": 128,
                  "unlearning_epochs": 20, "spls_lr": 0.003, "delete_lr": 0.001,
                  "out_shadow_reuse": {ds: str(out_root(ds)) for ds in CFG}})
        supervise(args, root)
    except Exception:
        save_json(root / "status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
