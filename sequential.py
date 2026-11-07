"""CIFAR-100/CIFAR-10 sequential SPLS, seed 2027.

Run both architectures on GPUs 1 and 4:
    python sequential.py
Inspect class order/checkpoints without training:
    python sequential.py --dry-run
Run one architecture (GPU IDs are physical IDs):
    python sequential.py --model-name resnet18 --gpu 1

SPLS uses main.py's ldmu/lowest-std settings: SGD, momentum=.9,
lr=.003, k=0, 20 epochs. Only NEW forget samples are used in a sequential
round. The teacher is the frozen previous round model. Both sequential and
single-shot use the original ldmu target exactly (only each sample's true
class is suppressed; every other logit matches the teacher, including
previously forgotten classes). No old samples are replayed. Retrain uses trainer.train_save_model and the same
100-way output head as main.py. Missing retrain checkpoints are trained
unless --skip-missing-retrain is supplied. All results are single-seed
measurements, never the mean/std reference table from revision.md.
"""

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DATASET = 'cifar100'
NUM_CLASSES = 100
COUNTS = (1, 2, 5, 10, 20)
DATASET_COUNTS = {'cifar100': (1, 2, 5, 10, 20), 'cifar10': (1, 2, 3, 4, 5)}
SEED = 2027


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=tuple(DATASET_COUNTS), default='cifar100')
    parser.add_argument('--counts', type=int, nargs='+', help='Cumulative forget counts (default depends on --dataset)')
    parser.add_argument('--model-name', '--model_name', choices=('both', 'resnet18', 'swin-t'), default='both')
    parser.add_argument('--gpu', help='Physical GPU ID for a single architecture')
    parser.add_argument('--gpus', nargs=2, default=['1', '4'], help='ResNet-18, Swin-T GPU IDs')
    parser.add_argument('--exps-dir', type=Path, default=ROOT / 'exps')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'sequential_results')
    parser.add_argument('--data-dir', type=Path, default=Path('~/data').expanduser())
    parser.add_argument('--original-checkpoint', type=Path, help='Override for a single architecture')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--unlearn-epoch', type=int, default=20)
    parser.add_argument('--unlearn-rate', type=float, default=3e-3)
    parser.add_argument('--k', type=float, default=0.0)
    parser.add_argument('--fixed-class-set', action='store_true', help='Only for references run with main.py --fixed_class_set')
    parser.add_argument('--skip-missing-retrain', action='store_true')
    parser.add_argument('--skip-retrain', action='store_true', help='Compare sequential with single-shot only')
    parser.add_argument('--sequential-only', action='store_true',
                        help='Train and evaluate only the sequential model')
    parser.add_argument('--reuse-single-shot', type=Path,
                        help='Previous output dir: copy its Single-shot/Retrain rows and train only sequential')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='default 2027')  # 2026-11-03: multi-seed runs
    args = parser.parse_args()
    if args.model_name == 'both' and (args.gpu is not None or args.original_checkpoint is not None):
        parser.error('--gpu and --original-checkpoint require a single --model-name')
    if args.unlearn_epoch < 1 or args.unlearn_rate <= 0 or args.k < 0:
        parser.error('epochs/rate must be positive and k must be nonnegative')
    if args.batch_size is not None and args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.num_workers < 0:
        parser.error('--num-workers must be nonnegative')
    global DATASET, NUM_CLASSES, COUNTS, SEED
    if args.seed is not None:
        SEED = args.seed
    DATASET, NUM_CLASSES, COUNTS = args.dataset, int(args.dataset.removeprefix('cifar')), DATASET_COUNTS[args.dataset]
    if args.counts:
        if any(b <= a for a, b in zip([0] + args.counts, args.counts)) or args.counts[-1] >= NUM_CLASSES:
            parser.error('--counts must be strictly increasing and below the number of classes')
        COUNTS = tuple(args.counts)
    args.exps_dir = args.exps_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    if args.reuse_single_shot is not None:
        args.reuse_single_shot = args.reuse_single_shot.resolve()
    return args


def class_order(config, fixed=False):
    """Match main.py: shuffle the YAML order, never sort/sample range(NUM_CLASSES)."""
    from omegaconf import OmegaConf
    order = OmegaConf.create({'order': list(config.permutation_map)}).order
    if not fixed:
        # Use the same ListConfig shuffle semantics as main.py, isolated from
        # random consumption by evaluation/training in subsequent rounds.
        random.Random(SEED).shuffle(order)
    order = list(order)
    if len(order) != NUM_CLASSES or set(order) != set(range(NUM_CLASSES)):
        raise ValueError(f'Expected a {DATASET} permutation of all {NUM_CLASSES} classes')
    return order


def safe_json(value):
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(safe_json(value), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_checkpoint(path, model_name):
    """Use main.py's model definitions without redundant pretrained downloads."""
    import torch
    from models import get_model
    from models.swin_transformer import swin_tiny_patch4_window7_224
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = (swin_tiny_patch4_window7_224(pretrained=False, num_classes=NUM_CLASSES)
             if model_name == 'swin-t' else get_model(model_name, NUM_CLASSES))
    model.load_state_dict({key.removeprefix('module.'): value for key, value in state.items()})
    return model.cuda()


def spls_loss(student_logits, teacher_logits, labels, k):
    """method.ldmu lowest-std: suppress only each sample's true class.

    Other logits preserve the teacher, including old forget classes.
    min/std are calculated over the teacher's complete output vector.
    """
    import torch
    import torch.nn.functional as F
    target = teacher_logits.detach().clone()
    rows = torch.arange(labels.numel(), device=labels.device)
    floor = target.min(1).values - k * target.std(1)
    target[rows, labels] = floor
    return F.mse_loss(student_logits, target)


def unlearn(model, loader, args):
    import torch
    teacher = copy.deepcopy(model).eval()
    teacher.requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.unlearn_rate, momentum=0.9)
    for epoch in range(args.unlearn_epoch):
        model.train()
        total_loss = total = 0
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                reference = teacher(x)
            loss = spls_loss(model(x), reference, y, args.k)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite SPLS loss')
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.numel()
            total += y.numel()
        print(f'epoch {epoch + 1}/{args.unlearn_epoch} loss={total_loss / total:.8f}', flush=True)
    del teacher, optimizer
    return model.eval()


def evaluate(model, loaders, groups, args):
    import torch
    from evaluation import SVC_MIA
    # Full test predictions yield retain, cumulative, per-group and per-class
    # accuracy in one pass. Train-side MIA keeps main.py's augmentation.
    correct = torch.zeros(NUM_CLASSES, dtype=torch.long)
    totals = torch.zeros(NUM_CLASSES, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for x, y in loaders['test']:
            prediction = model(x.cuda()).argmax(1).cpu()
            totals += torch.bincount(y, minlength=NUM_CLASSES)
            correct += torch.bincount(y[prediction == y], minlength=NUM_CLASSES)
    cumulative = [c for group in groups for c in group]
    retain = [c for c in range(NUM_CLASSES) if c not in cumulative]

    def accuracy(classes):
        return 100.0 * correct[classes].sum().item() / totals[classes].sum().item()

    # Same attack, features, balancing and target_test=None as main.py.
    # The balanced shadow subset is prepared once per round and shared by
    # all three models to make comparisons independent of training RNG use.
    mia = SVC_MIA(shadow_train=loaders['shadow_train'], shadow_test=loaders['test_remain'],
                  target_train=loaders['train_forget'], target_test=None, model=model)
    return {
        'test_remain_acc': accuracy(retain),
        'test_forget_acc': accuracy(cumulative),
        'previous_forget_acc': accuracy(cumulative[:-len(groups[-1])]) if len(groups) > 1 else None,
        'group_accuracy': {f'B{i + 1}': accuracy(group) for i, group in enumerate(groups)},
        'class_accuracy': {str(c): accuracy([c]) for c in cumulative},
        'mia': 100.0 * mia['confidence']['train_member_ratio'],
        'mia_metric': 'confidence SVC train_member_ratio (%)',
        'mia_raw': mia,
    }


def plot_summary(rows, output):
    """Render all four sequential diagnostics in one exportable figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    output = Path(output)
    complete = [r for r in rows if r.get('status') == 'complete']
    if not complete:
        return
    counts = sorted({r['count'] for r in complete})
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    dataset = manifest.get('dataset', DATASET).upper().replace('CIFAR', 'CIFAR-')
    model = manifest.get('model', complete[0].get('model', output.name))
    methods = ('Original', 'Retrain', 'SPLS (Single-shot)', 'SPLS (Sequential)')
    colors = dict(zip(methods, ('#8b929a', '#23936b', '#de8c24', '#3568bd')))
    markers = dict(zip(methods, ('s', '^', 'D', 'o')))
    with plt.rc_context({'font.size': 11, 'axes.titlesize': 13,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42}):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), layout='constrained')
        for ax, metric, title, ylabel in (
            (axes[0, 0], 'mia', '(a) MIA', 'Confidence SVC member ratio (%)'),
            (axes[1, 0], 'test_forget_acc', '(c) Test forget accuracy', 'Cumulative forget accuracy (%)'),
            (axes[1, 1], 'test_remain_acc', '(d) Test remain accuracy', 'Retain accuracy (%)')):
            for method in methods:
                data = sorted([r for r in complete if r['method'] == method and r.get(metric) is not None],
                              key=lambda r: r['count'])
                if data:
                    ax.plot([r['count'] for r in data], [r[metric] for r in data],
                            color=colors[method], marker=markers[method], linewidth=2,
                            markersize=6, label=method, alpha=.9)
            ax.set(title=title, ylabel=ylabel)
            ax.legend(fontsize=9, frameon=False)
        ax = axes[0, 1]
        seq = sorted([r for r in complete if r['method'] == 'SPLS (Sequential)'],
                     key=lambda r: r['count'])
        groups = sorted({g for r in seq for g in r.get('group_accuracy', {})},
                        key=lambda g: int(g[1:]))
        for i, group in enumerate(groups):
            data = [r for r in seq if group in r.get('group_accuracy', {})]
            ax.plot([r['count'] for r in data], [r['group_accuracy'][group] for r in data],
                    marker=('o', 's', '^', 'D', 'v')[i % 5], linewidth=1.8,
                    markersize=6, markerfacecolor='none', label=group)
        ax.set(title='(b) Previously forgotten groups', ylabel='Group accuracy (%)')
        ax.legend(title='Sequential SPLS', ncol=min(3, max(1, len(groups))),
                  fontsize=9, title_fontsize=9, frameon=False)
        for ax in axes.flat:
            ax.set_xlabel('Cumulative number of forgotten classes')
            ax.set_xticks(counts)
            ax.grid(axis='y', alpha=.22)
            low, high = ax.get_ylim()
            ax.set_ylim(max(0, low), max(1, min(100, high)))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        fig.suptitle(f'{dataset} / {model} — Sequential unlearning (seed {manifest.get("seed", SEED)})',
                     fontsize=16, fontweight='bold')
        for suffix in ('png', 'pdf'):
            fig.savefig(output / f'sequential_summary.{suffix}', dpi=220, bbox_inches='tight')
        plt.close(fig)


def plot_results(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric, ylabel in [('test_remain_acc', 'Retain Accuracy (%)'),
                           ('test_forget_acc', 'Cumulative Forget Accuracy (%)'),
                           ('mia', 'Confidence MIA member ratio (%)')]:
        fig, ax = plt.subplots()
        for method in ('Original', 'Retrain', 'SPLS (Single-shot)', 'SPLS (Sequential)'):
            data = [r for r in rows if r['method'] == method and r.get('status') == 'complete']
            if data:
                ax.plot([r['count'] for r in data], [r[metric] for r in data], 'o-', label=method)
        ax.set(xlabel='Cumulative Number of Forgotten Classes', ylabel=ylabel, xticks=COUNTS)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f'{metric}.png', dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots()
    seq = [r for r in rows if r['method'] == 'SPLS (Sequential)']
    for group in range(1, len(COUNTS) + 1):
        data = [r for r in seq if f'B{group}' in r['group_accuracy']]
        if data:
            ax.plot([r['count'] for r in data], [r['group_accuracy'][f'B{group}'] for r in data],
                    'o-', label=f'B{group}')
    ax.set(xlabel='Cumulative Number of Forgotten Classes',
           ylabel='Forgotten Group Accuracy (%)', xticks=COUNTS)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'previously_forgotten.png', dpi=180)
    plt.close(fig)
    plot_summary(rows, output)


def save_results(rows, output):
    write_json(output / 'results.json', rows)
    columns = ['seed', 'model', 'round', 'count', 'method', 'status', 'checkpoint',
               'input_checkpoint', 'new_classes', 'cumulative_classes', 'test_remain_acc',
               'test_forget_acc', 'previous_forget_acc', 'mia', 'mia_metric']
    with (output / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                             for key, value in row.items() if key in columns})
    plot_results(rows, output)
    report = [f'# Seed 2027 {DATASET} sequential SPLS', '',
              'All accuracies and MIA member ratios are percentages. Single-run results; no standard deviations.',
              '', '| Round | Forgotten | Method | Retain | Forget | Previous | MIA |',
              '|---|---|---|---|---|---|---|']
    for row in rows:
        values = [f'{row[key]:.4f}' if row.get(key) is not None else 'N/A'
                  for key in ('test_remain_acc', 'test_forget_acc', 'previous_forget_acc', 'mia')]
        report.append(f"| {row['round']} | {row['count']} | {row['method']} ({row['status']}) | " + ' | '.join(values) + ' |')
    report += ['', 'Group B1 tracks the first forgotten class throughout all rounds.',
               'An increase in group/class accuracy indicates classification recovery; zero accuracy alone does not establish privacy.',
               'Missing retrain results are explicitly omitted from plots. Existing checkpoints must use the recorded class order.']
    (output / 'report.md').write_text('\n'.join(report) + '\n')


def run_model(args):
    # Set GPU visibility before importing torch or any project modules.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, SubsetRandomSampler
    from trainer import train_save_model
    from utils import seed_torch, get_transforms, get_dataset, get_dataloader, get_unlearn_loader

    config = OmegaConf.load(ROOT / f'config/{DATASET}_{args.model_name}.yaml')
    order = class_order(config, args.fixed_class_set)
    checkpoint_dir = args.exps_dir / 'pretrained_model'
    original = args.original_checkpoint or checkpoint_dir / f'{SEED}_{DATASET}_{args.model_name}_original_model.pth'
    plan = {'seed': SEED, 'dataset': DATASET, 'model': args.model_name, 'gpu': args.gpu,
            'original_checkpoint': str(original), 'original_exists': original.is_file(),
            'class_order': order, 'fixed_class_set': args.fixed_class_set,
            'rounds': [{'round': i + 1, 'new': order[0 if i == 0 else COUNTS[i - 1]:count],
                        'cumulative': order[:count]} for i, count in enumerate(COUNTS)],
            'unlearn_epoch': args.unlearn_epoch, 'unlearn_rate': args.unlearn_rate, 'k': args.k,
            'suppression_rule': 'previous round teacher; new samples only; suppress only each sample true class with full-logit lowest-std targets and ordinary MSE (method.ldmu); preserve all other teacher logits',
            'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    if not original.is_file():
        raise FileNotFoundError(original)
    if not torch.cuda.is_available():
        raise RuntimeError(f'CUDA unavailable on requested GPU {args.gpu}')
    reused = None
    if args.reuse_single_shot is not None:
        # Single-shot rows are only comparable under the same class order and
        # unlearning hyperparameters.
        source = args.reuse_single_shot / args.model_name
        previous = json.loads((source / 'manifest.json').read_text())
        for key in ('class_order', 'unlearn_epoch', 'unlearn_rate', 'k', 'original_checkpoint'):
            if previous[key] != plan[key]:
                raise ValueError(f'{source} differs in {key}: {previous[key]} != {plan[key]}')
        reused = json.loads((source / 'results.json').read_text())
    output = args.output_dir / args.model_name
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise FileExistsError(f'{output} already contains a run; choose a new --output-dir')
    write_json(output / 'manifest.json', plan)
    seed_torch(SEED)
    train_transform, test_transform = get_transforms(DATASET, args.model_name, wo_dataaug=False)
    trainset, testset = get_dataset(DATASET, train_transform, test_transform, path=args.data_dir)
    batch_size = args.batch_size or config.batch_size
    _, test_loader = get_dataloader(trainset, testset, batch_size, args.num_workers)
    model = load_checkpoint(original, args.model_name)
    rows, groups = [], []
    previous_checkpoint = original

    for i, count in enumerate(COUNTS):
        new_classes = order[0 if i == 0 else COUNTS[i - 1]:count]
        cumulative = order[:count]
        groups.append(new_classes)
        seed_torch(SEED)
        splits = get_unlearn_loader(trainset, testset, cumulative, batch_size, float('inf'), args.num_workers)
        train_forget, _, test_forget, test_remain, _, _, train_remain_idx, _, test_remain_idx = splits
        shadow_indices = list(train_remain_idx)
        random.Random(SEED).shuffle(shadow_indices)
        shadow_indices = shadow_indices[:len(test_remain_idx)]
        loaders = {'test': test_loader, 'train_forget': train_forget, 'test_remain': test_remain,
                   'shadow_train': DataLoader(trainset, batch_size=batch_size,
                                              sampler=SubsetRandomSampler(shadow_indices), num_workers=args.num_workers)}
        write_json(output / f'round_{i + 1}_split.json', {
            'new_classes': new_classes, 'cumulative_classes': cumulative,
            'shadow_train_indices': shadow_indices,
            'train_forget_indices': splits[5], 'test_forget_indices': splits[7]})
        new_loader = get_unlearn_loader(trainset, testset, new_classes, batch_size,
                                        float('inf'), args.num_workers)[0]
        print(f'Round {i + 1}: new={new_classes}, cumulative={cumulative}', flush=True)
        # Explicitly reload the preceding round checkpoint, never M0 after round 1.
        if i:
            model = load_checkpoint(previous_checkpoint, args.model_name)
        seed_torch(SEED)
        model = unlearn(model, new_loader, args)
        seq_path = output / f'round_{i + 1}_forget{count}_sequential.pth'
        torch.save(model.state_dict(), seq_path)
        metadata = {'seed': SEED, 'model': args.model_name, 'round': i + 1, 'count': count,
                    'new_classes': new_classes, 'cumulative_classes': cumulative}

        def record(evaluated, method, checkpoint, input_checkpoint=None):
            seed_torch(SEED)  # identical MIA augmentation/sampler RNG across methods
            row = {**metadata, 'method': method, 'status': 'complete',
                   'checkpoint': str(checkpoint), 'input_checkpoint': str(input_checkpoint) if input_checkpoint else None,
                   **evaluate(evaluated, loaders, groups, args)}
            rows.append(row)
            write_json(checkpoint.with_suffix('.json') if checkpoint.parent == output else output / f'round_{i + 1}_{method.lower()}.json', row)
            save_results(rows, output)
            print(json.dumps(safe_json(row)), flush=True)

        record(model, 'SPLS (Sequential)', seq_path, previous_checkpoint)
        del model
        previous_checkpoint = seq_path
        torch.cuda.empty_cache()
        if args.sequential_only:
            continue
        if reused is not None:
            for method in ('SPLS (Single-shot)', 'Retrain'):
                matches = [r for r in reused if r['round'] == i + 1 and r['method'] == method]
                if len(matches) != 1:
                    raise ValueError(f'Expected one reused {method} row for round {i + 1}')
                rows.append({**matches[0], 'reused_from': str(args.reuse_single_shot)})
            save_results(rows, output)
            continue
        single_path = output / f'round_{i + 1}_forget{count}_single_shot.pth'
        single = load_checkpoint(original, args.model_name)
        seed_torch(SEED)
        single = unlearn(single, train_forget, args)
        torch.save(single.state_dict(), single_path)
        record(single, 'SPLS (Single-shot)', single_path, original)
        del single
        torch.cuda.empty_cache()
        if args.skip_retrain:
            continue

        retrain_path = checkpoint_dir / f'{SEED}_{DATASET}_{args.model_name}_retrain_forget{count}_model.pth'
        # Reuse existing individual seed-2027 reference measurements when the
        # corresponding checkpoint is absent. The reference evaluation uses
        # the same shuffled YAML permutation and confidence member ratio.
        references = []
        for reference_csv in sorted((ROOT / 'eval_original_retrain_mia/csv_shards').glob(f'{SEED}_{DATASET}_{args.model_name}_forget*.csv')):
            with reference_csv.open(newline='') as handle:
                references += [(reference_csv, row) for row in csv.DictReader(handle)
                               if row['seed'] == str(SEED) and row['method'] == 'retrain'
                               and row['model_name'] == args.model_name
                               and row['dataset_name'] == DATASET and int(row['forget_class']) == count]
        if not retrain_path.exists() and references and not args.fixed_class_set:
            if len({row['test_remain_acc'] for _, row in references}) == 1:
                reference_csv, reference = references[0]
                rows.append({**metadata, 'method': 'Retrain', 'status': 'complete',
                             'checkpoint': None, 'source': str(reference_csv),
                             'measurement': 'historical seed-2027 individual result',
                             'test_remain_acc': float(reference['test_remain_acc']),
                             'test_forget_acc': float(reference['test_forget_acc']),
                             'previous_forget_acc': None, 'mia': float(reference['mia']),
                             'mia_metric': 'confidence SVC train_member_ratio (%)'})
                save_results(rows, output)
                print(f'Retrain reference: {reference_csv} (forget={count})', flush=True)
                continue
        if not retrain_path.exists() and args.skip_missing_retrain:
            rows.append({**metadata, 'method': 'Retrain', 'status': 'missing_checkpoint', 'checkpoint': str(retrain_path)})
            save_results(rows, output)
            continue
        if not retrain_path.exists():
            seed_torch(SEED)
            retrain_path = output / f'round_{i + 1}_retrain.pth'
            # Keep dataset.targets available to trainer for its 100-class head.
            train_save_model(splits[1], test_remain, args.model_name, 'sgd', config.pretrain_lr,
                             config.pretrain_epoch, output, retrain_path.stem)
        retrain = load_checkpoint(retrain_path, args.model_name)
        record(retrain, 'Retrain', retrain_path)
        del retrain
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    if args.model_name != 'both':
        args.gpu = args.gpu if args.gpu is not None else args.gpus[0 if args.model_name == 'resnet18' else 1]
        run_model(args)
        return
    # Independent processes prevent accidental two-GPU DataParallel in trainer.
    children = []
    if not args.dry_run:
        for model in ('resnet18', 'swin-t'):
            if (args.output_dir / f'{model}.log').exists() or (args.output_dir / model / 'manifest.json').exists():
                raise FileExistsError(f'{args.output_dir} contains a previous run; choose a new --output-dir')
    for model, gpu in zip(('resnet18', 'swin-t'), args.gpus):
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                   '--model-name', model, '--gpu', str(gpu)]
        if args.dry_run:
            subprocess.run(command, check=True, cwd=ROOT)
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            log_path = args.output_dir / f'{model}.log'
            if log_path.exists():
                raise FileExistsError(f'{log_path} exists; choose a new --output-dir')
            with log_path.open('w') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            children.append((model, child))
            print(f'{model}: GPU {gpu}, PID {child.pid}, log {log_path}', flush=True)
    try:
        failed = [(model, child.wait()) for model, child in children]
    except KeyboardInterrupt:
        for _, child in children:
            child.terminate()
        for _, child in children:
            child.wait()
        raise
    if any(code for _, code in failed):
        raise SystemExit(f'Experiments failed: {failed}; inspect architecture logs')


if __name__ == '__main__':
    main()
