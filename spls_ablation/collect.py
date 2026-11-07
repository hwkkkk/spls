"""Collect SPLS target-ablation results (seed 2027) into CSVs and plots.

Outputs (in --out_dir, default spls_ablation/):
  ablation_results.csv        one row per run (final epoch) + existing SPLS (lambda=0) baseline rows
  ablation_epoch_metrics.csv  per-epoch train loss / train+test forget/retain acc
  ablation_summary.csv        compact config x variant table of final values (single seed: no mean/std)
  plots/{lambda_vs_test_retain,lambda_vs_test_forget,forget_vs_retain}_forget{1,5,10}.png
  plots/{retain_drop,forget_acc,mia}_by_forget_count.png   cifar100: vs. lambda, one line per forget-class count
  loss_ablation_summary.csv, plots/loss_ablation.png       cifar100 r18/swin-t: logit MSE vs. KL vs. masked-softmax MSE
"""
import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ABLATION_DIR = PROJECT_ROOT / "spls_ablation"
SEED = 2027
CONFIGS = [("cifar10", "resnet18"), ("cifar10", "swin-t"), ("cifar100", "resnet18"), ("cifar100", "swin-t")]
# (dataset, model, forget_class count); cifar10 only has forget 1 (main.py asserts forget_class < num_classes,
# and there is no existing forget-5 SPLS baseline for cifar10)
SETTINGS = [(ds, m, 1) for ds, m in CONFIGS] + [("cifar100", m, fc) for fc in (5, 10, 20) for m in ("resnet18", "swin-t")]
FORGET_COUNTS = (1, 5, 10, 20)
# loss ablation: cifar100, resnet18 and swin-t
LOSS_FORGET_COUNTS = (1, 2, 5, 10, 20)
LOSS_MODELS = ("resnet18", "swin-t")

# existing seed-2027 SPLS runs (lr 3e-3, 20 epochs) used as the lambda=0 baseline.
# cifar100 run folders no longer exist, so their final values come from the results CSV.
EXPS = PROJECT_ROOT / "exps"
BASELINE_LOGS = {
    ("cifar10", "resnet18", 1): EXPS / "2027_cifar10_resnet18_forget1/ldmu/lr_3e_3_0720-11:45:56/train_log.log",
    ("cifar10", "swin-t", 1): EXPS / "2027_cifar10_swin-t_forget1/ldmu/lr_3e_3_0720-13:37:29/train_log.log",
}
BASELINE_CSV = PROJECT_ROOT / "unlearning_results_retain_false.csv"
BASELINE_CSV_RUNS = {
    ("cifar100", "resnet18", 1): "lr_3e_3_0720-14:53:53",
    ("cifar100", "swin-t", 1): "lr_3e_3_0720-18:27:41",
    ("cifar100", "resnet18", 5): "lr_3e_3_0720-16:22:15",
    ("cifar100", "swin-t", 5): "lr_3e_3_0720-21:27:49",
    ("cifar100", "resnet18", 10): "lr_3e_3_0720-17:06:54",
    ("cifar100", "swin-t", 10): "lr_3e_3_0720-22:54:55",  # identical to the duplicate row lr_3e_3_0720-08:42:27
    ("cifar100", "resnet18", 20): "lr_3e_3_0720-17:42:17",
    ("cifar100", "swin-t", 20): "lr_3e_3_0721-00:27:47",
    ("cifar100", "resnet18", 2): "lr_3e_3_0720-15:42:43",  # loss ablation reference only
    ("cifar100", "swin-t", 2): "lr_3e_3_0720-19:57:30",  # loss ablation reference only
}

VARIANT_INFO = {  # variant -> (target label, lambda for the lambda-sweep axis)
    "min_lam0": ("z_min", 0.0),
    "min_lam1": ("z_min - 1σ", 1.0),
    "min_lam3": ("z_min - 3σ", 3.0),
    "min_lam5": ("z_min - 5σ", 5.0),
    "min_lam7": ("z_min - 7σ", 7.0),
    "min_lam10": ("z_min - 10σ", 10.0),
    "min_lam20": ("z_min - 20σ", 20.0),
    "min_lam30": ("z_min - 30σ", 30.0),
    "min_lam50": ("z_min - 50σ", 50.0),
    "min_lam100": ("z_min - 100σ", 100.0),
    "second_min": ("z_2nd-min", None),
    "mean_minus_std": ("mean - 1σ", None),
    "kl_softmax_zmin": ("KL(softmax(z_min target))", None),
    "mse_masked_softmax": ("MSE(prob, masked softmax)", None),
}

EPOCH_RE = re.compile(r"epoch (\d+) loss (\S+)")
ACC_RE = re.compile(r"train forget acc:(\S+)%, train_remain_acc:(\S+)%, test forget acc:(\S+)%, test remain acc:(\S+)%")
MIA_RE = re.compile(r"unlearn model\s*\n\s*(\{.*\})")


def parse_train_log(log_path):
    """Per-epoch rows and the final MIA dict from a main.py train_log.log."""
    text = Path(log_path).read_text()
    epochs, pending = [], None
    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if m:
            pending = {"epoch": int(m[1]), "train_loss": float(m[2])}
            continue
        m = ACC_RE.search(line)
        if m and pending is not None:
            pending.update(train_forget_acc=float(m[1]), train_retain_acc=float(m[2]),
                           test_forget_acc=float(m[3]), test_retain_acc=float(m[4]))
            epochs.append(pending)
            pending = None
    mia = None
    m = MIA_RE.search(text)
    if m:
        mia = ast.literal_eval(m[1].replace("nan", "None"))
    return epochs, mia


def mia_value(mia):
    # existing protocol: SVC on confidence, target_train = train forget set -> fraction predicted "member"
    if not mia:
        return None
    return mia["confidence"]["train_member_ratio"]


def final_row(epochs, mia):
    last = epochs[-1] if epochs else {}
    return {
        "epochs_completed": len(epochs),
        "final_train_loss": last.get("train_loss"),
        "train_forget_acc": last.get("train_forget_acc"),
        "train_retain_acc": last.get("train_retain_acc"),
        "test_forget_acc": last.get("test_forget_acc"),
        "test_retain_acc": last.get("test_retain_acc"),
        "mia_forget_member_ratio": mia_value(mia),
    }


def collect_ablation(status_dir):
    rows, epoch_rows = [], []
    for status_file in sorted(Path(status_dir).glob("*.status.json")):
        st = json.loads(status_file.read_text())
        label, lam = VARIANT_INFO[st["variant"]]
        fc = st.get("forget_class", 1)  # forget-1 runs predate the field
        row = {"seed": st["seed"], "dataset": st["dataset"], "model": st["model"], "forget_class": fc,
               "variant": st["variant"],
               "target": label, "lambda": lam, "status": st["status"], "source": "ablation"}
        log = Path(st["experiment_path"]) / "train_log.log" if st.get("experiment_path") else None
        epochs, mia = parse_train_log(log) if log and log.exists() else ([], None)
        row.update(final_row(epochs, mia))
        if st["status"] != "ok":  # keep what was measured, but never report a failed run as a result
            row["error"] = st.get("error")
        row["log_path"] = str(log) if log else None
        row["checkpoint"] = st.get("checkpoint")
        rows.append(row)
        for e in epochs:
            epoch_rows.append({"seed": st["seed"], "dataset": st["dataset"], "model": st["model"],
                               "forget_class": fc, "variant": st["variant"], "status": st["status"], **e})
    return rows, epoch_rows


def collect_baseline():
    rows, epoch_rows = [], []
    base = {"seed": SEED, "variant": "spls_existing", "target": "z_min (existing SPLS)", "lambda": 0.0,
            "status": "ok", "source": "existing"}
    for (ds, model, fc), log in BASELINE_LOGS.items():
        if not log.exists():
            continue
        epochs, mia = parse_train_log(log)
        rows.append({**base, "dataset": ds, "model": model, "forget_class": fc, **final_row(epochs, mia),
                     "log_path": str(log)})
        for e in epochs:
            epoch_rows.append({"seed": SEED, "dataset": ds, "model": model, "forget_class": fc,
                               "variant": "spls_existing", "status": "ok", **e})
    if BASELINE_CSV.exists():
        df = pd.read_csv(BASELINE_CSV)
        for (ds, model, fc), run_name in BASELINE_CSV_RUNS.items():
            r = df[(df.run_name == run_name) & (df.dataset_name == ds) & (df.model_name == model) & (df.seed == SEED)
                   & (df.forget_class == fc) & (df.method == "ldmu")]
            if len(r) != 1:
                print(f"WARNING: baseline {ds} {model} forget{fc} {run_name}: {len(r)} matching rows, skipped")
                continue
            r = r.iloc[0]
            rows.append({**base, "dataset": ds, "model": model, "forget_class": fc,
                         "epochs_completed": int(r.final_epoch),
                         "final_train_loss": r.final_loss, "train_forget_acc": r.forget_acc,
                         "train_retain_acc": r.train_remain_acc, "test_forget_acc": r.test_forget_acc,
                         "test_retain_acc": r.test_remain_acc,
                         "mia_forget_member_ratio": r.mia / 100 if pd.notna(r.mia) else None,
                         "log_path": f"{BASELINE_CSV.name}:{run_name} (final epoch only)"})
    return rows, epoch_rows


# ---------------------------------------------------------------- plots
SERIES = {"lambda": "#2a78d6", "second_min": "#eb6834", "mean_minus_std": "#1baf7a"}  # categorical slots 1-3
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)


LAMBDA_TICKS = [0, 1, 3, 5, 10, 20, 30, 50, 100]


def _lambda_axis(ax):
    """lambda spans 0-100: symlog x axis (linear below 1) so the small-lambda points stay readable."""
    ax.set_xscale("symlog", linthresh=1, linscale=0.5)
    ax.set_xticks(LAMBDA_TICKS)
    ax.set_xticklabels([str(t) for t in LAMBDA_TICKS])
    ax.minorticks_off()


def _clamp_pct(get_lim, set_lim, min_span=0.0):
    """Keep an accuracy axis inside [0, 100] (with a small margin) and at least min_span wide."""
    lo, hi = get_lim()
    lo, hi = max(lo, -0.5), min(hi, 100.5)
    if hi - lo < min_span:
        hi = min(lo + min_span, 100.5)
    set_lim(lo, hi)


def _grid(n):
    import matplotlib.pyplot as plt
    rows, cols = (2, 2) if n > 2 else (1, n)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.75 * rows + 0.5), squeeze=False)
    return fig, axes.flat


def _settings(fc):
    return [(ds, m) for ds, m, f in SETTINGS if f == fc]


def plot_lambda(df, metric, ylabel, fc, out_path):
    import matplotlib.pyplot as plt
    configs = _settings(fc)
    fig, axes = _grid(len(configs))
    for ax, (ds, model) in zip(axes, configs):
        _style(ax)
        sub = df[(df.dataset == ds) & (df.model == model) & (df.forget_class == fc) & (df.status == "ok")]
        sweep = sub[sub.variant.str.startswith("min_lam")].sort_values("lambda")
        base = sub[sub.variant == "spls_existing"]
        pts = pd.concat([base[["lambda", metric]], sweep[["lambda", metric]]]).sort_values("lambda")
        if len(pts):
            ax.plot(pts["lambda"], pts[metric], color=SERIES["lambda"], linewidth=2, marker="o", markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5, label="z_min − λσ", zorder=3)
        if len(base):
            ax.plot(base["lambda"], base[metric], linestyle="none", marker="o", markersize=11, markerfacecolor="white",
                    markeredgecolor=SERIES["lambda"], markeredgewidth=2, label="existing SPLS (λ=0)", zorder=4)
        for variant, name in (("second_min", "2nd-min"), ("mean_minus_std", "mean − σ")):
            r = sub[sub.variant == variant]
            if len(r):
                v = r[metric].iloc[0]
                ax.axhline(v, color=SERIES[variant], linewidth=2, linestyle="--", label=f"{name} ({v:.2f})")
        ax.set_title(f"{ds.upper()} / {model}", color=INK, fontsize=11)
        ax.set_xlabel("suppression strength λ", color=INK_2)
        ax.set_ylabel(ylabel, color=INK_2)
        _lambda_axis(ax)
        _clamp_pct(ax.get_ylim, ax.set_ylim)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{ylabel} vs. λ, forget {fc} class{'es' if fc > 1 else ''}  (seed = {SEED}, single run per point)",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_forget_retain(df, fc, out_path):
    import matplotlib.pyplot as plt
    configs = _settings(fc)
    fig, axes = _grid(len(configs))
    for ax, (ds, model) in zip(axes, configs):
        _style(ax)
        sub = df[(df.dataset == ds) & (df.model == model) & (df.forget_class == fc) & (df.status == "ok")]
        groups = [
            (sub[sub.variant.str.startswith("min_lam")], "z_min − λσ", SERIES["lambda"], "o", None),
            (sub[sub.variant == "spls_existing"], "existing SPLS (λ=0)", SERIES["lambda"], "o", "white"),
            (sub[sub.variant == "second_min"], "2nd-min", SERIES["second_min"], "s", None),
            (sub[sub.variant == "mean_minus_std"], "mean − σ", SERIES["mean_minus_std"], "D", None),
        ]
        for g, name, color, marker, face in groups:
            if not len(g):
                continue
            ax.scatter(g.test_forget_acc, g.test_retain_acc, s=70, marker=marker, label=name, zorder=3,
                       facecolor=face or color, edgecolor=color if face else "white", linewidth=2 if face else 1.5)
        for _, r in sub[sub.variant.str.startswith("min_lam")].iterrows():
            ax.annotate(f"λ={r['lambda']:g}", (r.test_forget_acc, r.test_retain_acc), xytext=(6, 4),
                        textcoords="offset points", fontsize=8, color=INK_2)
        ax.set_title(f"{ds.upper()} / {model}", color=INK, fontsize=11)
        ax.set_xlabel("test forget acc (%)", color=INK_2)
        ax.set_ylabel("test retain acc (%)", color=INK_2)
        _clamp_pct(ax.get_xlim, ax.set_xlim, min_span=5)
        _clamp_pct(ax.get_ylim, ax.set_ylim)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Forget vs. retain accuracy by target construction, forget {fc} class{'es' if fc > 1 else ''}  "
                 f"(seed = {SEED})", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


# Distinct hues and markers keep the counts distinguishable in color and grayscale.
FORGET_COLORS = {1: "#0072B2", 5: "#E69F00", 10: "#009E73", 20: "#CC79A7"}
FORGET_MARKERS = {1: "o", 5: "s", 10: "^", 20: "D"}


def plot_by_forget(df, metric, ylabel, title, out_path, delta=False, scale=1.0):
    """cifar100: metric vs. lambda (lambda=0 = existing SPLS run), one line per forget-class count.
    delta=True plots the change from the lambda=0 run."""
    import matplotlib.pyplot as plt
    models = [m for ds, m, fc in SETTINGS if ds == "cifar100" and fc == 1]
    fig, axes = _grid(len(models))
    for ax, model in zip(axes, models):
        _style(ax)
        if delta:
            ax.axhline(0, color=INK_2, linewidth=1)
        for fc in FORGET_COUNTS:
            sub = df[(df.dataset == "cifar100") & (df.model == model) & (df.forget_class == fc) & (df.status == "ok")]
            base = sub[sub.variant == "spls_existing"]
            sweep = sub[sub.variant.str.startswith("min_lam")]
            if not len(base) or not len(sweep):
                continue
            pts = pd.concat([base, sweep]).sort_values("lambda")
            y = pts[metric] * scale
            if delta:
                y = y - base[metric].iloc[0] * scale
            ax.plot(pts["lambda"], y, color=FORGET_COLORS[fc], linewidth=2, marker=FORGET_MARKERS[fc], markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5, label=f"forget {fc}", zorder=3)
        ax.set_title(f"CIFAR100 / {model}", color=INK, fontsize=11)
        ax.set_xlabel("suppression strength λ", color=INK_2)
        ax.set_ylabel(ylabel, color=INK_2)
        _lambda_axis(ax)
        if not delta:
            _clamp_pct(ax.get_ylim, ax.set_ylim, min_span=5)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{title}  (seed = {SEED}, λ=0 = existing SPLS)", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


LOSS_SERIES = [  # (variant, label, color, marker) - categorical slots 1-3
    ("spls_existing", "logit_mse(spls)", "#2a78d6", "o"),
    ("kl_softmax_zmin", "logit_space_kl", "#eb6834", "s"),
    ("mse_masked_softmax", "probability_space_mse", "#1baf7a", "D"),
]


def loss_ablation_table(df):
    sub = df[(df.dataset == "cifar100") & df.model.isin(LOSS_MODELS) & df.forget_class.isin(LOSS_FORGET_COUNTS)
             & df.variant.isin([v for v, *_ in LOSS_SERIES])]
    return sub[["seed", "dataset", "model", "forget_class", "variant", "target", "status", "train_forget_acc",
                "train_retain_acc", "test_forget_acc", "test_retain_acc", "mia_forget_member_ratio", "source"]]


def plot_loss_ablation(df, out_path):
    """cifar100: logit-MSE vs. probability-space losses, by number of forget classes; one row per model."""
    import matplotlib.pyplot as plt
    table = loss_ablation_table(df)
    table = table[table.status == "ok"]
    models = [m for m in LOSS_MODELS if len(table[(table.model == m) & (table.source == "ablation")])]
    if not models:
        return
    panels = [("test_retain_acc", "Test retain acc (%)", 1.0), ("test_forget_acc", "Test forget acc (%)", 1.0),
              ("mia_forget_member_ratio", "Confidence MIA on forget set (%)", 100.0)]
    fig, axes = plt.subplots(len(models), 3, figsize=(15, 4.5 * len(models)), squeeze=False)
    xpos = {fc: i for i, fc in enumerate(LOSS_FORGET_COUNTS)}  # forget counts as ordered categories
    for (row, model), (col, (metric, ylabel, scale)) in (
            (rm, cp) for rm in enumerate(models) for cp in enumerate(panels)):
        ax = axes[row][col]
        sub = table[table.model == model]
        _style(ax)
        ax.set_title(f"CIFAR100 / {model}", color=INK, fontsize=11)
        for variant, label, color, marker in LOSS_SERIES:
            g = sub[sub.variant == variant].sort_values("forget_class")
            if not len(g):
                continue
            ax.plot(g.forget_class.map(xpos), g[metric] * scale, color=color, linewidth=2, marker=marker,
                    markersize=7, markeredgecolor="white", markeredgewidth=1.5, label=label, zorder=3)
        ax.set_xticks(range(len(LOSS_FORGET_COUNTS)))
        ax.set_xticklabels([str(fc) for fc in LOSS_FORGET_COUNTS])
        ax.set_xlabel("number of forget classes", color=INK_2)
        ax.set_ylabel(ylabel, color=INK_2)
        _clamp_pct(ax.get_ylim, ax.set_ylim, min_span=5)
        if col == 0:
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Loss space ablation, CIFAR100  (seed = {SEED})", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status_dir", default=str(ABLATION_DIR / "ablation_logs"))
    parser.add_argument("--out_dir", default=str(ABLATION_DIR))
    parser.add_argument("--no_baseline", action="store_true")
    args = parser.parse_args()
    out = Path(args.out_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    rows, epoch_rows = collect_ablation(args.status_dir)
    if not args.no_baseline:
        b_rows, b_epochs = collect_baseline()
        rows, epoch_rows = b_rows + rows, b_epochs + epoch_rows
    if not rows:
        print("no runs found")
        return

    df = pd.DataFrame(rows)
    order = {v: i for i, v in enumerate(["spls_existing", *VARIANT_INFO])}
    df["_o"] = df.variant.map(order)
    df = df.sort_values(["forget_class", "dataset", "model", "_o"]).drop(columns="_o").reset_index(drop=True)
    df.to_csv(out / "ablation_results.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(out / "ablation_epoch_metrics.csv", index=False)

    summary = df[["seed", "dataset", "model", "forget_class", "variant", "target", "lambda", "status", "train_forget_acc",
                  "train_retain_acc", "test_forget_acc", "test_retain_acc", "mia_forget_member_ratio", "source"]]
    summary.to_csv(out / "ablation_summary.csv", index=False)

    for fc in FORGET_COUNTS:
        if not len(df[(df.forget_class == fc) & df.variant.str.startswith("min_lam")]):
            continue
        plot_lambda(df, "test_retain_acc", "Test retain accuracy (%)", fc,
                    out / f"plots/lambda_vs_test_retain_forget{fc}.png")
        plot_lambda(df, "test_forget_acc", "Test forget accuracy (%)", fc,
                    out / f"plots/lambda_vs_test_forget_forget{fc}.png")
        plot_forget_retain(df, fc, out / f"plots/forget_vs_retain_forget{fc}.png")
    plot_by_forget(df, "test_retain_acc", "Δ test retain acc vs. λ=0 (%p)", "Retain drop vs. λ by number of forget classes",
                   out / "plots/retain_drop_by_forget_count.png", delta=True)
    plot_by_forget(df, "test_forget_acc", "Test forget acc (%)", "Test forget accuracy vs. λ by number of forget classes",
                   out / "plots/forget_acc_by_forget_count.png")
    plot_by_forget(df, "mia_forget_member_ratio", "Confidence MIA on forget set (%)",
                   "Confidence MIA vs. λ by number of forget classes", out / "plots/mia_by_forget_count.png", scale=100.0)
    loss_ablation_table(df).to_csv(out / "loss_ablation_summary.csv", index=False)
    plot_loss_ablation(df, out / "plots/loss_ablation.png")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.to_string(index=False))
    failed = df[df.status != "ok"]
    if len(failed):
        print(f"\n{len(failed)} failed run(s):")
        print(failed[["dataset", "model", "forget_class", "variant", "status", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
