#!/usr/bin/env python3
"""Frozen ResNet18 linear probing following linear_probe.md (seed 2027)."""
import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models

from utils import get_transforms

ROOT = Path(__file__).resolve().parent
PRETRAIN = ROOT / "exps/pretrained_model"
REVISION = ROOT / "revision"
SPLS_FILES = {"cifar10": "0720-114556", "cifar100": "0720-145353"}
DELETE_FILES = {"cifar10": "0719-064453", "cifar100": "0719-090441"}


def save_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_model(path, classes, device):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        model = obj
    else:
        state = obj.get("state_dict", obj)
        model = models.resnet18(num_classes=classes)
        model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    model.requires_grad_(False)
    return model.eval().to(device)


def accuracy(pred, labels, forget):
    mask = labels == forget
    assert mask.any() and (~mask).any()
    return {"retain": (pred[~mask] == labels[~mask]).float().mean().item() * 100,
            "forget": (pred[mask] == labels[mask]).float().mean().item() * 100,
            "overall": (pred == labels).float().mean().item() * 100}


@torch.no_grad()
def extract(model, loader, device):
    head = getattr(model, "fc", getattr(model, "linear", None))
    if not isinstance(head, nn.Linear) or head.in_features != 512:
        raise ValueError("Expected a linear head with 512 input features")
    features, labels, predictions = [], [], []
    handle = head.register_forward_pre_hook(
        lambda module, inputs: features.append(inputs[0].detach().flatten(1).float().cpu()))
    try:
        for step, (x, y) in enumerate(loader):
            predictions.append(model(x.to(device, non_blocking=True)).argmax(1).cpu())
            labels.append(y)
            if step % 50 == 0:
                print(f"  extraction {step}/{len(loader)}", flush=True)
    finally:
        handle.remove()
    X, y, pred = torch.cat(features), torch.cat(labels), torch.cat(predictions)
    assert X.shape == (len(loader.dataset), 512) and torch.isfinite(X).all()
    return X, y, pred


def train_probe(cache, classes, epochs, device, output):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    Xtr, ytr = cache["Xtr"], cache["ytr"]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(TensorDataset((Xtr - mu) / sd, ytr), batch_size=1024,
                        shuffle=True, generator=generator, num_workers=0)
    clf = nn.Linear(512, classes).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(epochs):
        clf.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(clf(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f"  probe epoch {epoch + 1}/{epochs} loss={loss.item():.5f}", flush=True)
    clf.eval()
    with torch.no_grad():
        pred = clf(((cache["Xte"] - mu) / sd).to(device)).argmax(1).cpu()
    torch.save({"state_dict": {k: v.cpu() for k, v in clf.state_dict().items()},
                "mean": mu, "std": sd, "epochs": epochs, "seed": 0}, output)
    return accuracy(pred, cache["yte"], cache["forget_class"])


def write_results(out, rows, manifest):
    fields = ["dataset", "method", "forget_class", "output_retain", "output_forget",
              "output_overall", "lp_retain", "lp_forget", "lp_overall", "epochs", "checkpoint"]
    tmp = out / "lp_seed2027.csv.tmp"
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out / "lp_seed2027.csv")
    lines = ["# Linear probing: ResNet-18, seed 2027", "",
             "Single seed / single forget class. CIFAR-10: class 6; CIFAR-100: class 73.",
             "Output accuracy is re-evaluated on the same full test set.",
             "Training uses all 50,000 train images including the forget class, without augmentation.",
             "Test preprocessing preserves the checkpoint's 224x224 resize and normalization.", "",
             "| Dataset | Method | Output Retain | Output Forget | LP Retain | LP Forget | Epochs |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {dataset} | {method} | {output_retain:.2f} | {output_forget:.2f} | "
                     "{lp_retain:.2f} | {lp_forget:.2f} | {epochs} |".format(**row))
    lines += ["", "Freeze checkpoint status: " + json.dumps(manifest["freeze_checkpoints"]),
              "", "Original LP sanity checks (absolute difference <= 1 percentage point):"]
    for row in rows:
        if row["method"] == "original":
            delta = row["lp_overall"] - row["output_overall"]
            lines.append(f"- {row['dataset']}: LP {row['lp_overall']:.2f}%, "
                         f"output {row['output_overall']:.2f}%, delta {delta:+.2f} pp; "
                         f"{'PASS' if abs(delta) <= 1 else 'FAIL — inspect before reporting'}.")
    (out / "results.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="linear_probe_results_seed2027")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cifar10-freeze", type=Path)
    p.add_argument("--cifar100-freeze", type=Path)
    args = p.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES=1 selects physical GPU 1.
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "features").mkdir(exist_ok=True)
    (out / "probes").mkdir(exist_ok=True)
    manifest = {"seed": 2027, "probe_seed": 0, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "lr": 0.001, "optimizer": "Adam", "batch_size": 1024,
                "scheduler": "CosineAnnealingLR", "freeze_checkpoints": {}, "models": []}
    jobs = []
    for ds, classes in [("cifar10", 10), ("cifar100", 100)]:
        cfg = OmegaConf.load(ROOT / f"config/{ds}_resnet18.yaml")
        perm = list(cfg.permutation_map)
        random.Random(2027).shuffle(perm)
        forget = perm[0]
        assert forget == {"cifar10": 6, "cifar100": 73}[ds], "Mismatch with archived experiment logs"
        ckpts = {"original": PRETRAIN / f"2027_{ds}_resnet18_original_model.pth",
                 "retrain": PRETRAIN / f"2027_{ds}_resnet18_retrain_forget1_model.pth",
                 "delete": REVISION / f"2027_{ds}_resnet18_forget1_delete_{DELETE_FILES[ds]}.pth",
                 "spls": REVISION / f"2027_{ds}_resnet18_forget1_ldmu_{SPLS_FILES[ds]}.pth"}
        freeze_path = getattr(args, f"{ds}_freeze")
        manifest["freeze_checkpoints"][ds] = str(freeze_path) if freeze_path else "missing; not evaluated"
        if freeze_path:
            ckpts["spls_freeze"] = freeze_path.resolve()
        for method, path in ckpts.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            job = {"dataset": ds, "method": method, "checkpoint": str(path),
                   "forget_class": forget, "classes": classes}
            jobs.append(job)
            manifest["models"].append(job)
    save_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    # Extract once; feature caches permit resumed probing with a common epoch count.
    for ds, classes in [("cifar10", 10), ("cifar100", 100)]:
        _, transform = get_transforms(ds, "resnet18", wo_dataaug=True)
        factory = datasets.CIFAR10 if ds == "cifar10" else datasets.CIFAR100
        train = factory(ROOT / "data", train=True, download=False, transform=transform)
        test = factory(ROOT / "data", train=False, download=False, transform=transform)
        assert len(train) == 50000 and len(test) == 10000
        loaders = [DataLoader(data, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0) for data in (train, test)]
        for job in [j for j in jobs if j["dataset"] == ds]:
            path = Path(job["checkpoint"])
            stat = path.stat()
            signature = hashlib.sha256(f"{path}:{stat.st_size}:{stat.st_mtime_ns}:{transform}:v1".encode()).hexdigest()
            cache_path = out / "features" / f"{ds}_{job['method']}.pt"
            job["cache"] = str(cache_path)
            if cache_path.exists():
                existing = torch.load(cache_path, weights_only=True)
                if existing.get("signature") == signature:
                    print(f"[cached] {ds} {job['method']}", flush=True)
                    del existing
                    continue
            print(f"[extract] {ds} {job['method']}", flush=True)
            model = load_model(path, classes, device)
            Xtr, ytr, _ = extract(model, loaders[0], device)
            Xte, yte, pred = extract(model, loaders[1], device)
            assert torch.equal(ytr, torch.tensor(train.targets))
            assert torch.equal(yte, torch.tensor(test.targets))
            metrics = accuracy(pred, yte, job["forget_class"])
            print(f"[output] {ds} {job['method']} {metrics}", flush=True)
            torch.save({"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte,
                        "output": metrics, "forget_class": job["forget_class"],
                        "signature": signature}, cache_path.with_suffix(".tmp"))
            cache_path.with_suffix(".tmp").replace(cache_path)
            del model, Xtr, Xte, ytr, yte, pred
            torch.cuda.empty_cache()
        del loaders
    rows = []
    epochs = 100
    # Calibrate on CIFAR-100 Original before selecting the common epoch count.
    calibration = next(j for j in jobs if j["dataset"] == "cifar100" and j["method"] == "original")
    cache = torch.load(calibration["cache"], weights_only=True)
    print("[sanity] cifar100 original, 100 epochs", flush=True)
    initial = train_probe(cache, 100, 100, device, out / "probes/cifar100_original_e100.pt")
    if cache["output"]["overall"] - initial["overall"] > 1:
        epochs = 200
        print("[sanity] CIFAR-100 LP is >1 pp below output; use 200 epochs for EVERY model", flush=True)
    manifest["epochs"] = epochs
    manifest["cifar100_initial_sanity"] = {"lp": initial, "output": cache["output"]}
    save_json(out / "manifest.json", manifest)
    del cache
    for job in jobs:
        print(f"[probe] {job['dataset']} {job['method']} epochs={epochs}", flush=True)
        cache = torch.load(job["cache"], weights_only=True)
        lp = initial if job is calibration and epochs == 100 else train_probe(
            cache, job["classes"], epochs, device,
            out / "probes" / f"{job['dataset']}_{job['method']}_e{epochs}.pt")
        row = {k: job[k] for k in ("dataset", "method", "forget_class", "checkpoint")}
        row.update({f"output_{k}": v for k, v in cache["output"].items()})
        row.update({f"lp_{k}": v for k, v in lp.items()})
        row["epochs"] = epochs
        rows.append(row)
        write_results(out, rows, manifest)
        print(f"[result] {json.dumps(row)}", flush=True)
        del cache
    save_json(out / "status.json", {"status": "completed_available_checkpoints", "rows": len(rows),
                                    "freeze_missing": [ds for ds, v in manifest["freeze_checkpoints"].items()
                                                       if v.startswith("missing")]})
    print(f"[done] {out / 'lp_seed2027.csv'}", flush=True)


if __name__ == "__main__":
    main()
