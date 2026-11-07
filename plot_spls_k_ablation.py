#!/usr/bin/env python3

import re
import math
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EXPS_DIR = Path(__file__).resolve().parent / "exps"
OUT_DIR = Path("./spls_k_ablation_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = {
    ("cifar10", "resnet18"): 1,
    ("cifar10", "swin-t"): 1,
    ("cifar100", "resnet18"): 1,
    ("cifar100", "swin-t"): 1,
}

ACC_PATTERN = re.compile(
    r"train forget acc:([0-9.]+)%,\s*train_remain_acc:([0-9.]+)%,\s*test forget acc:([0-9.]+)%,\s*test remain acc:([0-9.]+)%",
    flags=re.IGNORECASE,
)


def parse_last_metrics(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = ACC_PATTERN.findall(text)
    if not matches:
        return None

    last = matches[-1]
    return {
        "train_forget_acc": float(last[0]),
        "train_remain_acc": float(last[1]),
        "test_forget_acc": float(last[2]),
        "test_remain_acc": float(last[3]),
    }


def collect_rows():
    rows = []

    for config_path in EXPS_DIR.rglob("config.yaml"):
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        method = cfg.get("method", None)
        if method not in ["ldmu", "spls"]:
            continue

        dataset = cfg.get("dataset_name")
        model = cfg.get("model_name")
        seed = cfg.get("seed")
        forget_class = cfg.get("forget_class")
        k = cfg.get("k", None)
        desc = cfg.get("description", "")

        if (dataset, model) not in TARGETS:
            continue

        if forget_class != TARGETS[(dataset, model)]:
            continue

        # keep only the SPLS k-ablation runs
        if "spls_k_" not in str(desc):
            continue

        log_path = config_path.parent / "train_log.log"
        if not log_path.exists():
            log_path = config_path.parent / "train.log"
        if not log_path.exists():
            continue

        metrics = parse_last_metrics(log_path)
        if metrics is None:
            continue

        rows.append({
            "dataset_name": dataset,
            "model_name": model,
            "forget_class": forget_class,
            "seed": int(seed),
            "k": float(k),
            "retain_test_acc": metrics["test_remain_acc"],
            "forget_test_acc": metrics["test_forget_acc"],
            "train_remain_acc": metrics["train_remain_acc"],
            "train_forget_acc": metrics["train_forget_acc"],
            "log_path": str(log_path),
            "config_path": str(config_path),
        })

    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame):
    group_cols = ["dataset_name", "model_name", "forget_class", "k"]

    agg_rows = []
    for keys, group in df.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        row["seed_count"] = group["seed"].nunique()

        for col in ["retain_test_acc", "forget_test_acc"]:
            vals = group[col].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                mean = math.nan
                std = math.nan
            else:
                mean = vals.mean()
                std = vals.std(ddof=0)

            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std

        agg_rows.append(row)

    return pd.DataFrame(agg_rows).sort_values(
        ["dataset_name", "model_name", "k"]
    ).reset_index(drop=True)


def plot_one(df_agg: pd.DataFrame, dataset: str, model: str):
    sub = df_agg[
        (df_agg["dataset_name"] == dataset) &
        (df_agg["model_name"] == model)
    ].sort_values("k")

    if len(sub) == 0:
        print(f"[WARN] No data for {dataset} / {model}")
        return

    x = sub["k"].to_numpy(dtype=float)

    retain_mean = sub["retain_test_acc_mean"].to_numpy(dtype=float)
    retain_std  = sub["retain_test_acc_std"].to_numpy(dtype=float)

    forget_mean = sub["forget_test_acc_mean"].to_numpy(dtype=float)
    forget_std  = sub["forget_test_acc_std"].to_numpy(dtype=float)

    plt.figure(figsize=(7, 5))

    plt.plot(x, retain_mean, marker="o", label="Retain Test Acc")
    plt.fill_between(x, retain_mean - retain_std, retain_mean + retain_std, alpha=0.2)

    plt.plot(x, forget_mean, marker="s", label="Forget Test Acc")
    plt.fill_between(x, forget_mean - forget_std, forget_mean + forget_std, alpha=0.2)

    plt.xlabel("k")
    plt.ylabel("Accuracy (%)")
    plt.title(f"SPLS lowest-std ablation: {dataset}, {model}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_png = OUT_DIR / f"spls_k_ablation_{dataset}_{model}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[SAVE] {out_png}")


def main():
    df = collect_rows()
    if len(df) == 0:
        print("[ERROR] No SPLS/ldmu ablation logs found.")
        return

    raw_csv = OUT_DIR / "spls_k_ablation_raw.csv"
    df.to_csv(raw_csv, index=False)
    print(f"[SAVE] {raw_csv}")

    agg = aggregate(df)
    agg_csv = OUT_DIR / "spls_k_ablation_agg.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"[SAVE] {agg_csv}")

    for dataset, model in TARGETS.keys():
        plot_one(agg, dataset, model)

    print("\n=== Aggregated Results ===")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()