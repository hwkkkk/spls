#!/usr/bin/env python3
"""CIFAR-100 sequential SPLS (5 classes/round, 4 rounds): mean±std over seeds 2027-2029 for ResNet-18 and Swin-T."""
import json, csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
W = Path(__file__).resolve().parent
SRC = {  # (model, seed) -> results.json
    ('resnet18', 2027): W / 'sequential_results_c100_5step_20261024_082855/resnet18/results.json',
    ('resnet18', 2028): W / 'new_swin/analysis_2028/sequential_c100/resnet18/results.json',
    ('resnet18', 2029): W / 'new_swin/analysis_2029/sequential_c100/resnet18/results.json',
    ('swin-t', 2027): W / 'new_swin/analysis_2027/sequential_c100/swin-t/results.json',
    ('swin-t', 2028): W / 'new_swin/analysis_2028/sequential_c100/swin-t/results.json',
    ('swin-t', 2029): W / 'new_swin/analysis_2029/sequential_c100/swin-t/results.json',
}
MODELS = {'resnet18': 'ResNet-18', 'swin-t': 'Swin-T'}
METHODS = ('SPLS (Single-shot)', 'SPLS (Sequential)')
COLORS = {'SPLS (Single-shot)': '#de8c24', 'SPLS (Sequential)': '#3568bd'}
MARKERS = {'SPLS (Single-shot)': 'D', 'SPLS (Sequential)': 'o'}
METRICS = ('test_remain_acc', 'test_forget_acc', 'previous_forget_acc', 'mia')
COUNTS = (5, 10, 15, 20)

rows = []
for (model, seed), p in SRC.items():
    for r in json.loads(p.read_text()):
        if r['status'] == 'complete' and r['method'] in METHODS:
            assert r['seed'] == seed and r['model'] == model, p
            rows.append({k: r.get(k) for k in ('model', 'seed', 'round', 'count', 'method', *METRICS)})
            rows[-1].update({f'group_{g}': v for g, v in r.get('group_accuracy', {}).items()})

def stats(model, method, metric):
    m, s = [], []
    for c in COUNTS:
        v = [r[metric] for r in rows if r['model'] == model and r['method'] == method
             and r['count'] == c and r[metric] is not None]
        assert len(v) in (0, 3), (model, method, metric, c, len(v))
        m.append(np.mean(v) if v else np.nan); s.append(np.std(v, ddof=1) if v else np.nan)
    return np.array(m), np.array(s)

# per-seed rows + summary table
with open(HERE / 'results_all_seeds.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k.startswith('group_'), k))); w.writeheader(); w.writerows(rows)
with open(HERE / 'summary_mean_std.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['model', 'method', 'count'] + [f'{m}_{x}' for m in METRICS for x in ('mean', 'std')])
    for model in MODELS:
        for method in METHODS:
            st = {m: stats(model, method, m) for m in METRICS}
            for i, c in enumerate(COUNTS):
                w.writerow([model, method, c] + [f'{st[m][j][i]:.2f}' for m in METRICS for j in (0, 1)])


# Same layout/style as sequential.plot_summary (paper figure), mean ± std error bars, no suptitle.
from matplotlib.ticker import MaxNLocator
def errline(ax, x, m, sd, **kw):
    ax.plot(x, m, **kw)  # mean only (no std bars)

rc = {'font.size': 11, 'axes.titlesize': 13, 'axes.spines.top': False,
      'axes.spines.right': False, 'pdf.fonttype': 42}
for model in MODELS:
    out = HERE / model; out.mkdir(exist_ok=True)
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), layout='constrained')
        fig.get_layout_engine().set(hspace=0.12, wspace=0.08)  # extra row/column gaps
        for ax, metric, title, ylabel in (
            (axes[0, 0], 'mia', '(a) MIA', 'Confidence SVC member ratio (%)'),
            (axes[1, 0], 'test_forget_acc', '(c) Test forget accuracy', 'Cumulative forget accuracy (%)'),
            (axes[1, 1], 'test_remain_acc', '(d) Test remain accuracy', 'Retain accuracy (%)')):
            for method in METHODS:
                m, sd = stats(model, method, metric)
                errline(ax, COUNTS, m, sd, color=COLORS[method], marker=MARKERS[method], linewidth=2,
                        markersize=6, label=method, alpha=.9)
            ax.set(title=title, ylabel=ylabel)
            ax.legend(fontsize=9, frameon=False)
        ax = axes[0, 1]
        for i, g in enumerate(('B1', 'B2', 'B3', 'B4')):
            xs, ms, ss = [], [], []
            for c in COUNTS:
                v = [r[f'group_{g}'] for r in rows if r['model'] == model and r['method'] == 'SPLS (Sequential)'
                     and r['count'] == c and r.get(f'group_{g}') is not None]
                if v:
                    assert len(v) == 3; xs.append(c); ms.append(np.mean(v)); ss.append(np.std(v, ddof=1))
            errline(ax, xs, ms, ss, marker=('o', 's', '^', 'D', 'v')[i], linewidth=1.8,
                    markersize=6, markerfacecolor='none', label=g)
        ax.set(title='(b) Previously forgotten groups', ylabel='Group accuracy (%)')
        ax.legend(title='Sequential SPLS', ncol=3, fontsize=9, title_fontsize=9, frameon=False)
        for ax in axes.flat:
            ax.set_xlabel('Cumulative number of forgotten classes')
            ax.set_xticks(COUNTS)
            ax.grid(axis='y', alpha=.22)
            low, high = ax.get_ylim()
            ax.set_ylim(max(0, low), max(1, min(100, high)))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        for s in ('png', 'pdf'):
            fig.savefig(out / f'sequential_summary.{s}', dpi=220, bbox_inches='tight')
        plt.close(fig)
print('saved to', HERE)
