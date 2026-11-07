"""3-seed (2027-2029) SPLS target/loss ablation for the revision, plotted with collect_multiseed.plots (no suptitle).

λ ∈ {0,1,3,5,10,20,30,50,100} only (λ=7, 2nd-min, mean-σ dropped 11-04). λ=0 = EXISTING SPLS main runs (never re-run; min_lam0 ablation runs ignored).
Sources
  ResNet-18 : spls_ablation/ablation_logs (λ>0 + loss), λ=0 = unlearning_results_retain_false.csv lr_3e_3 ldmu rows (2027 via collect.collect_baseline)
  Swin-T    : NEW ckpts, lr 1e-3, seeds 2027-2029 (old seed-2027 Swin runs in ablation_logs are excluded: fc forward bug)
              new_swin/<seed>_<ds>_swin-t_forget<fc>/ldmu/lr_1e_3_*      λ=0 (main SPLS runs)
              new_swin/lambda_ablation/status                       λ=20,30,50,100 (+ cifar10 all λ>0)
              new_swin/analysis_<seed>/k_ablation (main.py lowest-std --k)   cifar100 λ=1,3,5,10 (same target, unbiased std)
              new_swin/analysis_<seed>/loss_ablation/status         loss variants
usage: python -m spls_ablation.collect_revision [--out-dir DIR]
"""
import argparse, json, re
from pathlib import Path
import pandas as pd
from spls_ablation import collect as legacy
from spls_ablation import collect_multiseed as cm
# forget 2 added to the λ ablation (user, 11-04)
legacy.FORGET_COUNTS = (1, 2, 5, 10, 20)
legacy.FORGET_COLORS = {**legacy.FORGET_COLORS, 2: "#56B4E9"}
legacy.FORGET_MARKERS = {**legacy.FORGET_MARKERS, 2: "v"}
legacy.SETTINGS = legacy.SETTINGS + [("cifar100", m, 2) for m in ("resnet18", "swin-t")]

import matplotlib.axes as _mpl_axes
_orig_errorbar = _mpl_axes.Axes.errorbar
def _errorbar_no_err(self, *a, **kw):  # mean only: drop the std error bars (figures too cluttered)
    for k in ("xerr", "yerr"):
        kw.pop(k, None)
    return _orig_errorbar(self, *a, **kw)
_mpl_axes.Axes.errorbar = _errorbar_no_err

ROOT = legacy.PROJECT_ROOT
SEEDS = [2027, 2028, 2029]
MODELS = ["resnet18", "swin-t"]
SWIN = ROOT / "new_swin"
K_SALVAGE = {1.0, 3.0, 5.0, 10.0}
LAMBDA_VARIANTS = {f'min_lam{l}' for l in (1, 3, 5, 10, 20, 30, 50, 100)}
KEEP = LAMBDA_VARIANTS | {v for v, *_ in legacy.LOSS_SERIES}
BASE_FC = [("cifar100", fc) for fc in (1, 2, 5, 10, 20)]  # cifar100 only (user, 11-04)


def k_ablation_rows(seed):
    rows = []
    for cfg in (SWIN / f"analysis_{seed}/k_ablation").glob("*/ldmu/*/config.yaml"):
        run = cfg.parent
        fc = int(re.search(r"forget(\d+)$", run.parent.parent.name)[1])
        k = float(re.search(r"_k([\d.]+)_", run.name)[1])
        if k not in K_SALVAGE or (("cifar100", "swin-t", fc) not in legacy.SETTINGS):
            continue
        epochs, mia = legacy.parse_train_log(run / "train_log.log")
        v = f"min_lam{int(k)}"
        rows.append({"seed": seed, "dataset": "cifar100", "model": "swin-t", "forget_class": fc, "variant": v,
                     "target": legacy.VARIANT_INFO[v][0], "lambda": k, "status": "ok" if epochs else "missing",
                     "source": "k_ablation(main.py lowest-std)", **legacy.final_row(epochs, mia),
                     "log_path": str(run / "train_log.log")})
    return rows


def existing_spls_rows():
    base = {"variant": "spls_existing", "target": "z_min (existing SPLS)", "lambda": 0.0, "status": "ok"}
    rows = []
    for s in SEEDS:  # Swin-T: new-ckpt main SPLS runs
        for ds, fc in BASE_FC:
            runs = sorted((SWIN / f"{s}_{ds}_swin-t_forget{fc}/ldmu").glob("lr_1e_3_*/train_log.log"))
            if len(runs) != 1:
                print(f"WARNING swin {s} {ds} f{fc}: {len(runs)} lr_1e_3 runs"); continue
            epochs, mia = legacy.parse_train_log(runs[0])
            rows.append({**base, "seed": s, "dataset": ds, "model": "swin-t", "forget_class": fc,
                         "source": "existing", **legacy.final_row(epochs, mia), "log_path": str(runs[0])})
    df = pd.read_csv(legacy.BASELINE_CSV)  # ResNet-18 2028/2029 (2027 from collect_baseline)
    for s in SEEDS[1:]:
        for ds, fc in BASE_FC:
            r = df[(df.method == "ldmu") & (df.model_name == "resnet18") & (df.seed == s) & (df.dataset_name == ds)
                   & (df.forget_class == fc) & df.run_name.str.startswith("lr_3e_3")]
            if len(r) != 1:
                print(f"WARNING resnet {s} {ds} f{fc}: {len(r)} rows"); continue
            r = r.iloc[0]
            rows.append({**base, "seed": s, "dataset": ds, "model": "resnet18", "forget_class": fc, "source": "existing",
                         "epochs_completed": int(r.final_epoch), "final_train_loss": r.final_loss,
                         "train_forget_acc": r.forget_acc, "train_retain_acc": r.train_remain_acc,
                         "test_forget_acc": r.test_forget_acc, "test_retain_acc": r.test_remain_acc,
                         "mia_forget_member_ratio": r.mia / 100 if pd.notna(r.mia) else None,
                         "log_path": f"{legacy.BASELINE_CSV.name}:{r.run_name}"})
    return rows


def load():
    r18, e18 = legacy.collect_ablation(legacy.ABLATION_DIR / "ablation_logs")
    base, base_e = legacy.collect_baseline()
    rows = [r for r in base + r18 if r["model"] == "resnet18"]
    epochs = [e for e in base_e + e18 if e["model"] == "resnet18"]
    for d in [SWIN / "lambda_ablation/status"] + [SWIN / f"analysis_{s}/loss_ablation/status" for s in SEEDS]:
        r, e = legacy.collect_ablation(d)
        rows += r; epochs += e
    for s in SEEDS:
        rows += k_ablation_rows(s)
    rows = [r for r in rows if r["variant"] in KEEP or (r["variant"] == "spls_existing" and r["model"] == "resnet18")]
    rows += existing_spls_rows()
    df = pd.DataFrame(rows)
    df = df[df.seed.isin(SEEDS) & df.model.isin(MODELS) & (df.dataset == "cifar100")].copy()
    dup = df[df.duplicated(cm.KEYS + ["seed"], keep=False)]
    if len(dup):
        raise ValueError(f"duplicate conditions:\n{dup[cm.KEYS + ['seed', 'source']]}")
    return df, pd.DataFrame(epochs)


def finish(fig, path, title, **pad):  # same as collect_multiseed.finish, without the suptitle (seed info goes in the caption)
    fig.tight_layout(**pad)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    cm.plt.close(fig)


def plot_loss(summary, path, models):
    """Same look as collect.plot_loss_ablation (one row per model), mean ± SD error bars, no suptitle."""
    plt = cm.plt
    t = summary[(summary.dataset == "cifar100") & summary.model.isin(models) & summary.complete]
    panels = [("test_retain_acc", "Test retain acc (%)", 1.0), ("test_forget_acc", "Test forget acc (%)", 1.0),
              ("mia_forget_member_ratio", "Confidence MIA on forget set (%)", 100.0)]
    series = [("spls_existing", *legacy.LOSS_SERIES[0][1:])] + list(legacy.LOSS_SERIES[1:]) \
        if legacy.LOSS_SERIES[0][0] == "min_lam0" else list(legacy.LOSS_SERIES)
    xpos = {fc: i for i, fc in enumerate(legacy.LOSS_FORGET_COUNTS)}
    fig, axes = plt.subplots(len(models), 3, figsize=(15, 4.5 * len(models)), squeeze=False)
    for row, model in enumerate(models):
        for col, (metric, ylabel, scale) in enumerate(panels):
            ax = axes[row][col]
            legacy._style(ax)
            ax.set_title(f"CIFAR100 / {model}", color=legacy.INK, fontsize=11)
            for variant, label, color, marker in series:
                g = t[(t.model == model) & (t.variant == variant) & t.forget_class.isin(xpos)].sort_values("forget_class")
                if not len(g):
                    continue
                ax.errorbar(g.forget_class.map(xpos), g[metric] * scale, yerr=g[metric + "_std"] * scale,
                            color=color, linewidth=2, marker=marker, markersize=7, markeredgecolor="white",
                            markeredgewidth=1.5, capsize=3, elinewidth=1.2, label=label, zorder=3)
            ax.set_xticks(range(len(xpos)))
            ax.set_xticklabels([str(fc) for fc in xpos])
            ax.set_xlabel("number of forget classes", color=legacy.INK_2)
            ax.set_ylabel(ylabel, color=legacy.INK_2)
            legacy._clamp_pct(ax.get_ylim, ax.set_ylim, min_span=5)
            if col == 0:
                ax.legend(fontsize=8, frameon=False)
    finish(fig, path, None)


def plot_lambda_combined(summary, path, models):
    """λ ablation in one figure: rows = models, cols = retain / forget / MIA, one line per forget count."""
    plt = cm.plt
    c = summary[summary.complete & (summary.dataset == "cifar100") & summary["lambda"].notna()]
    metrics = [("test_retain_acc", "Test retain accuracy (%)", 1.),
               ("test_forget_acc", "Test forget accuracy (%)", 1.),
               ("mia_forget_member_ratio", "Confidence MIA member ratio (%)", 100.)]
    fig, axes = plt.subplots(len(models), 3, figsize=(15, 4.5 * len(models)), squeeze=False)
    for row, model in zip(axes, models):
        for ax, (metric, ylabel, scale) in zip(row, metrics):
            legacy._style(ax)
            for fc in legacy.FORGET_COUNTS:
                g = c[(c.model == model) & (c.forget_class == fc)]
                if len(g):
                    cm.curve(ax, g, "lambda", metric, f"forget {fc}", legacy.FORGET_COLORS[fc],
                             legacy.FORGET_MARKERS[fc], scale)
            legacy._lambda_axis(ax)
            ax.set(title=f"CIFAR100 / {model}", xlabel="suppression strength k (λ)", ylabel=ylabel)
        row[0].legend(fontsize=9, frameon=False)
    finish(fig, path, "", h_pad=2.5, w_pad=2.5)
    print("lambda plot ->", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=legacy.ABLATION_DIR / "revision_3seed")
    p.add_argument("--partial", action="store_true", help="plot even if some conditions lack 3 seeds")
    out = p.parse_args().out_dir
    partial = p.parse_args().partial
    out.mkdir(parents=True, exist_ok=True)
    raw, epochs = load()
    raw.to_csv(out / "ablation_results.csv", index=False)
    summary = cm.summarize(raw, SEEDS)
    summary.to_csv(out / "ablation_summary.csv", index=False)
    summary[summary.variant.isin([v for v, *_ in legacy.LOSS_SERIES])].to_csv(out / "loss_ablation_summary.csv", index=False)
    missing = summary[~summary.complete][cm.KEYS + ["seeds"]]
    print(f"conditions {len(summary)}, complete {int(summary.complete.sum())}")
    if len(missing):
        print(missing.to_string(index=False))
    (out / "plots").mkdir(exist_ok=True)
    for ms, name in ((["swin-t"], "loss_ablation_swin-t.png"), (MODELS, "loss_ablation.png")):
        ok = summary[(summary.model.isin(ms)) & summary.variant.isin(["spls_existing", *[v for v, *_ in legacy.LOSS_SERIES]])
                     & summary.forget_class.isin(legacy.LOSS_FORGET_COUNTS)]
        if len(ok) and ok.complete.all():
            plot_loss(summary, out / "plots" / name, ms)
            print("loss plot ->", out / "plots" / name)
    if missing.empty or partial:
        (out / "plots").mkdir(exist_ok=True)
        cm.finish = finish
        legacy.SETTINGS = [x for x in legacy.SETTINGS if x[0] == "cifar100"]  # no empty cifar10 panels
        cm.plots(raw, summary, out / "plots", SEEDS, MODELS)
        plot_lambda_combined(summary, out / "plots" / "lambda_ablation.png", MODELS)
        print("plots ->", out / "plots")
