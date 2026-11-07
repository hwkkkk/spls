#!/usr/bin/env python3
"""CMIA evaluation using the exact test-only protocol specified in cmia.md."""
import argparse
import csv
import json
import os
import warnings
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import datasets

from linear_probe import ROOT, PRETRAIN, REVISION, SPLS_FILES, DELETE_FILES, load_model, save_json
from utils import get_transforms


def cmia_for_class(logits, labels, neighbor, forget):
    """Fit only on retain test samples; predict positive fraction on forget."""
    is_f = labels == forget
    pos = labels == neighbor
    neg = (~pos) & (~is_f)
    assert neighbor != forget and pos.any() and neg.any() and is_f.any()
    x = logits[:, [neighbor]]
    fit_x = np.concatenate([x[pos], x[neg]])
    fit_y = np.concatenate([np.ones(pos.sum()), np.zeros(neg.sum())])
    clf = make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", random_state=2027, max_iter=1000))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        clf.fit(fit_x, fit_y)
    score = float(clf.predict(x[is_f]).mean() * 100)
    return score


@torch.no_grad()
def get_logits(model, loader, classes):
    values, labels = [], []
    model.eval()
    for x, y in loader:
        values.append(model(x.to("cuda:0", non_blocking=True)).float().cpu())
        labels.append(y)
    logits, labels = torch.cat(values).numpy(), torch.cat(labels).numpy()
    assert logits.shape == (10000, classes) and np.isfinite(logits).all()
    assert np.array_equal(labels, np.asarray(loader.dataset.targets))
    return logits, labels


def write_results(out, rows, neighbors, missing):
    fields = ["dataset", "method", "forget_class", "forget_name", "r_n", "neighbor_name",
              "cmia", "delta", "forget_test_count", "checkpoint"]
    with (out / "cmia_seed2027.csv.tmp").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "cmia_seed2027.csv.tmp").replace(out / "cmia_seed2027.csv")
    lines = ["# CMIA: ResNet-18, one-class unlearning, seed 2027", "",
             "Single seed / single forget class. All fits and evaluation use test data only.",
             "Each binary attack uses only one class logit, StandardScaler and balanced logistic regression.",
             "Forget samples are excluded from fitting. Each model fits its own attack.",
             "Retrain selects the shared neighbor; ties select the smallest class index.", ""]
    for ds, info in neighbors.items():
        lines += [f"- {ds}: {info['forget_name']} ({info['forget_class']}) → "
                  f"{info['neighbor_name']} ({info['r_n']}); Retrain CMIA {info['cmia']:.2f}%."]
    lines += ["", "| Method | CIFAR-10 CMIA | Δ vs Retrain | CIFAR-100 CMIA | Δ vs Retrain |",
              "|---|---:|---:|---:|---:|"]
    for method in dict.fromkeys(row["method"] for row in rows):
        cells = [method]
        for ds in ("cifar10", "cifar100"):
            row = next((r for r in rows if r["dataset"] == ds and r["method"] == method), None)
            cells += [f"{row['cmia']:.2f}", f"{row['delta']:+.2f}"] if row else ["pending", "pending"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Optional baselines without saved checkpoints: " + ", ".join(missing) + ".",
              "", "CIFAR-100 has 100 forget test examples, so CMIA has 1 percentage point resolution.",
              "See neighbors.json for all retain-class scores and Retrain's forget argmax distribution."]
    (out / "results.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="cmia_results_seed2027")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    torch.manual_seed(2027)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    out = Path(args.output_dir).resolve()
    (out / "logits").mkdir(parents=True, exist_ok=True)
    missing = ["boundary_expand", "boundary_shrink", "gradient_ascent", "random_label",
               "bad_teacher_noretain", "salun_noretain"]
    manifest = {"protocol": str(ROOT / "cmia.md"), "seed": 2027, "single_forget_class": True,
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__, "sklearn": sklearn.__version__,
                "classifier": {"name": "LogisticRegression", "class_weight": "balanced",
                               "max_iter": 1000, "random_state": 2027, "C": 1.0,
                               "solver": "lbfgs", "preprocessing": "StandardScaler"},
                "missing_optional_baselines": missing, "checkpoints": {}}
    rows, neighbors = [], {}
    save_json(out / "status.json", {"status": "running", "pid": os.getpid()})
    for ds, classes, forget in [("cifar10", 10, 6), ("cifar100", 100, 73)]:
        _, transform = get_transforms(ds, "resnet18", wo_dataaug=True)
        factory = datasets.CIFAR10 if ds == "cifar10" else datasets.CIFAR100
        test = factory(ROOT / "data", train=False, download=False, transform=transform)
        loader = DataLoader(test, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)
        checkpoints = {"retrain": PRETRAIN / f"2027_{ds}_resnet18_retrain_forget1_model.pth",
                       "original": PRETRAIN / f"2027_{ds}_resnet18_original_model.pth",
                       "delete": REVISION / f"2027_{ds}_resnet18_forget1_delete_{DELETE_FILES[ds]}.pth",
                       "spls": REVISION / f"2027_{ds}_resnet18_forget1_ldmu_{SPLS_FILES[ds]}.pth"}
        manifest["checkpoints"][ds] = {k: str(v) for k, v in checkpoints.items()}
        save_json(out / "manifest.json", manifest)
        rn, reference = None, None
        for method, checkpoint in checkpoints.items():
            print(f"[logits] {ds} {method}: {checkpoint}", flush=True)
            model = load_model(checkpoint, classes, "cuda:0")
            Z, Y = get_logits(model, loader, classes)
            np.savez_compressed(out / "logits" / f"{ds}_{method}.npz", logits=Z, labels=Y)
            del model
            torch.cuda.empty_cache()
            if method == "retrain":
                scores = {r: cmia_for_class(Z, Y, r, forget) for r in range(classes) if r != forget}
                rn = max(scores, key=scores.get)
                reference = scores[rn]
                predictions = Z[Y == forget].argmax(1)
                values, counts = np.unique(predictions, return_counts=True)
                distribution = sorted([{"class": int(v), "name": test.classes[v], "count": int(c)}
                                       for v, c in zip(values, counts)], key=lambda item: -item["count"])
                neighbors[ds] = {"forget_class": forget, "forget_name": test.classes[forget],
                                 "r_n": rn, "neighbor_name": test.classes[rn], "cmia": reference,
                                 "all_scores": scores, "argmax_distribution": distribution,
                                 "argmax_mode_matches_neighbor": int(values[counts.argmax()]) == rn,
                                 "forget_prediction_count": int((predictions == forget).sum()),
                                 "top3": sorted(scores.items(), key=lambda pair: -pair[1])[:3]}
                save_json(out / "neighbors.json", neighbors)
                print(f"[neighbor] {json.dumps(neighbors[ds])}", flush=True)
            score = reference if method == "retrain" else cmia_for_class(Z, Y, rn, forget)
            row = {"dataset": ds, "method": method, "forget_class": forget,
                   "forget_name": test.classes[forget], "r_n": rn, "neighbor_name": test.classes[rn],
                   "cmia": score, "delta": score - reference,
                   "forget_test_count": int((Y == forget).sum()), "checkpoint": str(checkpoint)}
            rows.append(row)
            write_results(out, rows, neighbors, missing)
            print(f"[result] {json.dumps(row)}", flush=True)
        del loader
    save_json(out / "status.json", {"status": "completed", "rows": len(rows),
                                    "missing_optional_baselines": missing})
    print(f"[done] {out / 'cmia_seed2027.csv'}", flush=True)


if __name__ == "__main__":
    main()
