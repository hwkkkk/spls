#!/usr/bin/env python3
"""Train four fixed-forget-class shadows, then evaluate the lira.md protocol."""
import argparse
import csv
import json
import os
import traceback
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100

from common import ROOT, PRETRAIN, REVISION, load_model, save_json
from shadow_model import SEEDS, train_shadow
from utils import get_transforms


@torch.no_grad()
def phi_scores(model, loader, forget=6, classes=10):
    model.eval()
    values = []
    for x, labels in loader:
        assert (labels == forget).all()
        logits = model(x.to("cuda", non_blocking=True)).float()
        assert logits.shape[1] == classes
        rest = torch.cat((logits[:, :forget], logits[:, forget + 1:]), dim=1)
        values.append((logits[:, forget] - torch.logsumexp(rest, dim=1)).cpu())
    phi = torch.cat(values).numpy().astype(np.float64)
    assert np.isfinite(phi).all()
    return phi


def shadow_stats(P):
    assert P.ndim == 2 and P.shape[0] >= 2 and np.isfinite(P).all()
    mu = P.mean(axis=0)
    sigma = float(np.sqrt(((P - mu) ** 2).sum() / (P.shape[1] * (P.shape[0] - 1))))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Invalid shadow global standard deviation")
    return mu, sigma


def attack_metrics(phi, labels, mu, sigma):
    z = (phi - mu) / sigma
    absolute = np.abs(z)
    fpr, tpr, thresholds = roc_curve(labels, absolute)
    idx = np.searchsorted(fpr, 0.01, side="right") - 1
    balanced = (tpr + 1 - fpr) / 2
    best = int(balanced.argmax())
    return {"auc_one_sided": float(roc_auc_score(labels, z) * 100),
            "auc_two_sided": float(roc_auc_score(labels, absolute) * 100),
            "tpr_at_1pct_fpr": float(tpr[idx] * 100),
            "achieved_fpr_pct": float(fpr[idx] * 100),
            "attack_balanced_acc": float(balanced[best] * 100),
            "mean_z_member": float(z[labels == 1].mean()),
            "mean_z_nonmember": float(z[labels == 0].mean())}, z


def write_results(out, rows, sanity, dataset="cifar10"):
    with (out / "lira_seed2027.csv.tmp").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "lira_seed2027.csv.tmp").replace(out / "lira_seed2027.csv")
    title, member_text = {
        "cifar10": ("CIFAR-10, ResNet-18, forget class 6 (frog)",
                    "Members: 5,000 train frogs. Nonmembers: 1,000 test frogs."),
        "cifar100": ("CIFAR-100, ResNet-18, forget class 73 (shark)",
                     "Members: 500 train sharks. Nonmembers: 100 test sharks. FPR 1% represents one nonmember.")}[dataset]
    lines = [f"# Offline LiRA: {title}", "",
             "Single target seed 2027, four retain-only shadow seeds 3001–3004.",
             member_text,
             "One query per image, deterministic test preprocessing, no horizontal flip.",
             "Per-image shadow mean and one global unbiased standard deviation.", "",
             "| Method | AUC one-sided | AUC two-sided | TPR@1%FPR | Best balanced accuracy |",
             "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| {method} | {auc_one_sided:.2f} | {auc_two_sided:.2f} | "
                     "{tpr_at_1pct_fpr:.2f} | {attack_balanced_acc:.2f} |".format(**r))
    lines += ["", "All values are percentages; attack accuracy is the optimal balanced accuracy on this evaluation set.",
              "", "Sanity diagnostics (heuristic flags, not significance tests):", "",
              "```json", json.dumps(sanity, indent=2), "```", "",
              "Chance-level results alone do not establish statistical indistinguishability.",
              "One-sided and absolute-z two-sided scores can miss other membership signals."]
    (out / "results.md").write_text("\n".join(lines) + "\n")


def run(args, out):
    dataset = args.dataset
    classes, forget = {"cifar10": (10, 6), "cifar100": (100, 73)}[dataset]
    checkpoint_root = args.checkpoint_dir
    targets = {"original": checkpoint_root / f"2027_{dataset}_resnet18_original_model.pth",
               "retrain": checkpoint_root / f"2027_{dataset}_resnet18_retrain_forget1_model.pth"}
    for method, token in (("delete", "delete"), ("spls", "ldmu")):
        paths = sorted(checkpoint_root.glob(f"2027_{dataset}_resnet18_forget1_{token}_*.pth"))
        if len(paths) != 1:
            raise ValueError(f"Expected one {method} checkpoint in {checkpoint_root}, got {paths}")
        targets[method] = paths[0]
    for path in targets.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = {"protocol": str(ROOT / "lira.md"), "shadow_seeds": list(SEEDS),
                "dataset": dataset, "target_seed": 2027, "forget_class": forget, "epochs_per_shadow": 150,
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__, "sklearn": sklearn.__version__, "flip": False,
                "checkpoint_selection": "best retain-test accuracy, same as trainer.py",
                "shadow_training": "shadow_model.py; extracted from main.py and trainer.py",
                "target_checkpoints": {name: str(path) for name, path in targets.items()}}
    save_json(out / "manifest.json", manifest)
    shadows = []
    for seed in SEEDS:
        save_json(out / "status.json", {"status": "training_shadows", "seed": seed,
                                        "completed_shadows": len(shadows), "pid": os.getpid()})
        shadows.append(train_shadow(seed, out / "shadows", args.num_workers, dataset))
    save_json(out / "status.json", {"status": "evaluating", "pid": os.getpid()})
    _, transform = get_transforms(dataset, "resnet18", wo_dataaug=True)
    factory = CIFAR10 if dataset == "cifar10" else CIFAR100
    trainset = factory(ROOT / "data", train=True, download=False, transform=transform)
    testset = factory(ROOT / "data", train=False, download=False, transform=transform)
    indices = [np.flatnonzero(np.asarray(ds.targets) == forget).tolist() for ds in (trainset, testset)]
    member_count, nonmember_count = 50000 // classes, 10000 // classes
    assert list(map(len, indices)) == [member_count, nonmember_count]
    loaders = [DataLoader(Subset(ds, ids), batch_size=128, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
               for ds, ids in zip((trainset, testset), indices)]
    labels = np.concatenate((np.ones(member_count, dtype=np.int64), np.zeros(nonmember_count, dtype=np.int64)))
    np.savez_compressed(out / "sample_indices.npz", member_indices=indices[0],
                        nonmember_indices=indices[1], membership=labels)

    def query(checkpoint):
        model = load_model(checkpoint, classes, "cuda")
        phi = np.concatenate([phi_scores(model, loader, forget, classes) for loader in loaders])
        assert phi.shape == (member_count + nonmember_count,)
        del model
        torch.cuda.empty_cache()
        return phi

    P = []
    for seed, path in zip(SEEDS, shadows):
        print(f"[shadow phi] seed={seed}", flush=True)
        P.append(query(path))
    P = np.stack(P)
    mu, sigma = shadow_stats(P)
    np.savez_compressed(out / "shadow_statistics.npz", phi=P, mu=mu, sigma=sigma, membership=labels)
    manifest["target_checkpoints"] = {name: str(path) for name, path in targets.items()}
    manifest["global_sigma"] = sigma
    save_json(out / "manifest.json", manifest)
    rows, sanity = [], {"global_sigma": sigma}
    for method, path in targets.items():
        print(f"[target] {method}", flush=True)
        phi = query(path)
        metrics, z = attack_metrics(phi, labels, mu, sigma)
        np.savez_compressed(out / f"{method}_scores.npz", phi=phi, z=z, membership=labels)
        rows.append(dict(method=method, **metrics))
        print(f"[result] {json.dumps(rows[-1])}", flush=True)
        if method == "original":
            sanity["original_one_sided_above_55pct"] = metrics["auc_one_sided"] > 55
        if method == "retrain":
            sanity["retrain_two_sided_within_5pp_of_chance"] = abs(metrics["auc_two_sided"] - 50) <= 5
        write_results(out, rows, sanity, dataset)
    shadow_accuracy = {str(seed): json.loads((out / "shadows" / f"shadow_seed{seed}.json").read_text())["best_acc"] * 100
                       for seed in SEEDS}
    sanity["shadow_best_retain_test_accuracy"] = shadow_accuracy
    write_results(out, rows, sanity, dataset)
    save_json(out / "sanity.json", sanity)
    save_json(out / "status.json", {"status": "completed", "rows": len(rows)})
    print(f"[done] {out / 'lira_seed2027.csv'}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    p.add_argument("--checkpoint-dir", type=Path, default=REVISION)
    p.add_argument("--output-dir")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    out = Path(args.output_dir or ("lira_results_seed2027" if args.dataset == "cifar10"
                                   else "lira_results_cifar100_seed2027")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        run(args, out)
    except Exception:
        save_json(out / "status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
