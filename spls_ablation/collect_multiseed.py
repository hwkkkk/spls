"""Aggregate independent seeds, requiring all requested seeds for every plotted point."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from spls_ablation import collect as legacy

METRICS = ["train_forget_acc", "train_retain_acc", "test_forget_acc",
           "test_retain_acc", "mia_forget_member_ratio"]
KEYS = ["dataset", "model", "forget_class", "variant"]


def load_results(status_dir, seeds, models):
    rows, epochs = legacy.collect_ablation(status_dir)
    baseline, base_epochs = legacy.collect_baseline()
    df = pd.DataFrame(baseline + rows)
    df = df[df.seed.isin(seeds) & df.model.isin(models)].copy()
    df.loc[df.variant == "min_lam0", "variant"] = "spls_existing"
    if df.duplicated(KEYS + ["seed"]).any():
        raise ValueError("Multiple results for the same condition and seed")
    return df, pd.DataFrame(base_epochs + epochs)


def summarize(df, seeds):
    rows = []
    for keys, group in df.groupby(KEYS, sort=True):
        ok = group[group.status == "ok"]
        row = dict(zip(KEYS, keys))
        row.update(seed_count=ok.seed.nunique(),
                   seeds=",".join(map(str, sorted(ok.seed.unique()))),
                   complete=set(ok.seed) == set(seeds))
        for metric in METRICS:
            row[metric] = ok[metric].mean()
            row[metric + "_std"] = ok[metric].std(ddof=1)
            row[metric + "_count"] = ok[metric].count()
            row["complete"] = row["complete"] and ok[metric].count() == len(seeds)
        row["lambda"] = 0.0 if keys[-1] == "spls_existing" else legacy.VARIANT_INFO[keys[-1]][1]
        rows.append(row)
    return pd.DataFrame(rows)


def curve(ax, group, xcol, metric, label, color, marker="o", scale=1.):
    if group.empty:
        return
    group = group.sort_values(xcol)
    x = group[xcol].to_numpy(dtype=float)
    y = group[metric].to_numpy(dtype=float) * scale
    sd = group[metric + "_std"].to_numpy(dtype=float) * scale
    ax.errorbar(x, y, yerr=sd, label=label, color=color, marker=marker,
                linewidth=1.7, capsize=3, markersize=5)


def finish(fig, path, title):
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plots(raw, summary, out, seeds, models):
    complete = summary[summary.complete].copy()
    note = f"mean ± sample SD; seeds {', '.join(map(str, seeds))}"
    metrics = [("test_retain_acc", "Test retain accuracy (%)", 1.),
               ("test_forget_acc", "Test forget accuracy (%)", 1.),
               ("mia_forget_member_ratio", "Confidence MIA member ratio (%)", 100.)]
    # Existing k/lambda plots: one curve per forget-class count.
    c100 = complete[complete.dataset == "cifar100"]
    for metric, ylabel, scale in metrics:
        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5), squeeze=False)
        for ax, model in zip(axes.flat, models):
            legacy._style(ax)
            for fc in legacy.FORGET_COUNTS:
                g = c100[(c100.model == model) & (c100.forget_class == fc)
                         & c100["lambda"].notna()]
                curve(ax, g, "lambda", metric, f"forget {fc}", legacy.FORGET_COLORS[fc],
                      legacy.FORGET_MARKERS[fc], scale)
            legacy._lambda_axis(ax)
            ax.set(title=f"CIFAR100 / {model}", xlabel="suppression strength k (λ)", ylabel=ylabel)
            if ax.lines:
                ax.legend(fontsize=8, frameon=False)
        filename = {"test_retain_acc": "retain_acc", "test_forget_acc": "forget_acc",
                    "mia_forget_member_ratio": "mia"}[metric]
        finish(fig, out / f"{filename}_by_forget_count.png", note)

    # Compute the retain difference within each seed before averaging it.
    ok = raw[raw.status == "ok"].copy()
    baselines = ok[ok.variant == "spls_existing"][KEYS[:-1] + ["seed", "test_retain_acc"]]
    paired = ok.merge(baselines.rename(columns={"test_retain_acc": "base_retain"}),
                      on=KEYS[:-1] + ["seed"], validate="many_to_one")
    paired["delta"] = paired.test_retain_acc - paired.base_retain
    delta = paired.groupby(KEYS).delta.agg(["mean", "std", "count"]).reset_index()
    delta = delta[delta["count"] == len(seeds)].merge(complete[KEYS + ["lambda"]], on=KEYS)
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5), squeeze=False)
    for ax, model in zip(axes.flat, models):
        legacy._style(ax)
        ax.axhline(0, color=legacy.INK_2, linewidth=1)
        for fc in legacy.FORGET_COUNTS:
            g = delta[(delta.dataset == "cifar100") & (delta.model == model)
                      & (delta.forget_class == fc) & delta["lambda"].notna()]
            g = g.rename(columns={"mean": "delta", "std": "delta_std"})
            curve(ax, g, "lambda", "delta", f"forget {fc}", legacy.FORGET_COLORS[fc], legacy.FORGET_MARKERS[fc])
        legacy._lambda_axis(ax)
        ax.set(title=f"CIFAR100 / {model}", xlabel="suppression strength k (λ)",
               ylabel="Δ test retain accuracy vs. k=0 (pp)")
        if len(ax.lines) > 1:
            ax.legend(fontsize=8, frameon=False)
    finish(fig, out / "retain_drop_by_forget_count.png", note)

    for fc in legacy.FORGET_COUNTS:
        settings = [(ds, m) for ds, m, f in legacy.SETTINGS if f == fc and m in models]
        if not settings:
            continue
        for metric, ylabel, scale in metrics[:2]:
            fig, axes = plt.subplots(1, len(settings), figsize=(5 * len(settings), 4.5), squeeze=False)
            for ax, (ds, model) in zip(axes.flat, settings):
                legacy._style(ax)
                sub = complete[(complete.dataset == ds) & (complete.model == model)
                               & (complete.forget_class == fc)]
                curve(ax, sub[sub["lambda"].notna()], "lambda", metric, "z_min − kσ", legacy.SERIES["lambda"])
                for variant, label in [("second_min", "2nd-min"), ("mean_minus_std", "mean − σ")]:
                    g = sub[sub.variant == variant]
                    if len(g):
                        y, sd = g[metric].iloc[0], g[metric + "_std"].iloc[0]
                        ax.axhline(y, color=legacy.SERIES[variant], linestyle="--", label=label)
                        ax.axhspan(y - sd, y + sd, color=legacy.SERIES[variant], alpha=.1)
                legacy._lambda_axis(ax)
                ax.set(title=f"{ds.upper()} / {model}", xlabel="suppression strength k (λ)", ylabel=ylabel)
                if ax.lines:
                    ax.legend(fontsize=8, frameon=False)
            suffix = "retain" if metric == "test_retain_acc" else "forget"
            finish(fig, out / f"lambda_vs_test_{suffix}_forget{fc}.png", f"Forget {fc}; {note}")

        fig, axes = plt.subplots(1, len(settings), figsize=(5 * len(settings), 4.5), squeeze=False)
        for ax, (ds, model) in zip(axes.flat, settings):
            legacy._style(ax)
            sub = complete[(complete.dataset == ds) & (complete.model == model)
                           & (complete.forget_class == fc)]
            groups = [(sub[sub["lambda"].notna()], "z_min − kσ", legacy.SERIES["lambda"], "o"),
                      (sub[sub.variant == "second_min"], "2nd-min", legacy.SERIES["second_min"], "s"),
                      (sub[sub.variant == "mean_minus_std"], "mean − σ", legacy.SERIES["mean_minus_std"], "D")]
            for group, label, color, marker in groups:
                if group.empty:
                    continue
                ax.errorbar(group.test_forget_acc, group.test_retain_acc,
                            xerr=group.test_forget_acc_std, yerr=group.test_retain_acc_std,
                            fmt=marker, color=color, capsize=3, label=label)
            for _, r in sub[sub["lambda"].notna()].iterrows():
                ax.annotate(f"k={r['lambda']:g}", (r.test_forget_acc, r.test_retain_acc),
                            xytext=(5, 4), textcoords="offset points", fontsize=7)
            ax.set(title=f"{ds.upper()} / {model}", xlabel="Test forget accuracy (%)",
                   ylabel="Test retain accuracy (%)")
            if ax.lines:
                ax.legend(fontsize=8, frameon=False)
        finish(fig, out / f"forget_vs_retain_forget{fc}.png", f"Forget {fc}; {note}")

    fig, axes = plt.subplots(len(models), 3, figsize=(15, 4.5 * len(models)), squeeze=False)
    xpos = {fc: i for i, fc in enumerate(legacy.LOSS_FORGET_COUNTS)}
    for row, model in enumerate(models):
        for col, (metric, ylabel, scale) in enumerate(metrics):
            ax = axes[row, col]
            legacy._style(ax)
            for variant, label, color, marker in legacy.LOSS_SERIES:
                g = c100[(c100.model == model) & (c100.variant == variant)
                         & c100.forget_class.isin(legacy.LOSS_FORGET_COUNTS)].copy()
                g["x"] = g.forget_class.map(xpos)
                curve(ax, g, "x", metric, label, color, marker, scale)
            ax.set(title=f"CIFAR100 / {model}", xlabel="number of forget classes", ylabel=ylabel)
            ax.set_xticks(list(xpos.values()), list(map(str, xpos)))
            if col == 0 and ax.lines:
                ax.legend(fontsize=8, frameon=False)
    finish(fig, out / "loss_ablation.png", f"Loss space ablation; {note}")


def collect(status_dir, out, seeds, models):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    raw, epochs = load_results(status_dir, seeds, models)
    raw.to_csv(out / "ablation_results.csv", index=False)
    epochs = epochs[epochs.seed.isin(seeds) & epochs.model.isin(models)]
    epochs.to_csv(out / "ablation_epoch_metrics.csv", index=False)
    summary = summarize(raw, seeds)
    summary.to_csv(out / "ablation_summary.csv", index=False)
    summary[summary.variant.isin([v for v, *_ in legacy.LOSS_SERIES]) & (summary.dataset == "cifar100")].to_csv(
        out / "loss_ablation_summary.csv", index=False)
    # Partial summaries remain available, but plots are generated only after all cells complete.
    all_complete = bool(len(summary)) and bool(summary.complete.all())
    if all_complete:
        (out / "plots").mkdir(exist_ok=True)
        plots(raw, summary, out / "plots", seeds, models)
    info = {"seeds": seeds, "models": models, "std": "sample SD (ddof=1)",
            "complete_conditions": int(summary.complete.sum()), "observed_conditions": len(summary),
            "all_observed_conditions_complete": all_complete}
    (out / "aggregation.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--status-dir", type=Path, default=legacy.ABLATION_DIR / "ablation_logs")
    p.add_argument("--out-dir", type=Path, default=legacy.ABLATION_DIR / "multiseed_2027_2029")
    p.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029])
    p.add_argument("--models", nargs="+", default=["resnet18", "swin-t"])
    args = p.parse_args()
    collect(args.status_dir, args.out_dir, args.seeds, args.models)
