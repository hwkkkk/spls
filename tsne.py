#!/usr/bin/env python3
import argparse
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils import *
from trainer import *


def load_unlearn_checkpoint(path: Path, model_name: str, num_classes: int, device):
    """
    main.py saves unlearned models with torch.save(unlearn_model, path),
    so normally this returns a full nn.Module.
    Also supports a state_dict-style checkpoint as fallback.
    """
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=device)

    if isinstance(obj, nn.Module):
        return obj.to(device)

    return load_model(path, model_name, num_classes).to(device)


def resolve_forget_class_id(seed: int, model_name: str, explicit_id=None, fixed_class_set=False):
    if explicit_id is not None:
        return int(explicit_id)

    cfg_path = PROJECT_ROOT / f"config/cifar10_{model_name}.yaml"
    cfg = OmegaConf.load(cfg_path)
    perm = list(cfg.permutation_map)

    # Match main.py: seed_torch(seed) -> random.shuffle(permutation_map)
    seed_torch(seed)
    if not fixed_class_set:
        random.shuffle(perm)

    return int(perm[0])


@torch.no_grad()
def extract_penultimate_features(model, loader, device):
    if not hasattr(model, "fc"):
        raise AttributeError(
            "ResNet feature extraction expects model.fc. "
            "This script is intended for ResNet-18."
        )

    feats = []

    def pre_hook(module, inputs):
        feats.append(inputs[0].detach().flatten(1).cpu())

    handle = model.fc.register_forward_pre_hook(pre_hook)

    model.eval()
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)

    handle.remove()
    return torch.cat(feats, dim=0).numpy()


def balanced_indices(targets, per_class, seed):
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    selected = []

    for c in sorted(np.unique(targets)):
        idx = np.flatnonzero(targets == c)
        rng.shuffle(idx)
        selected.extend(idx[: min(per_class, len(idx))].tolist())

    rng.shuffle(selected)
    return selected


def run_tsne(features, seed):
    denom = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.clip(denom, 1e-12, None)

    if features.shape[1] > 50:
        features = PCA(n_components=50, random_state=seed).fit_transform(features)

    perplexity = min(30, max(5, (len(features) - 1) // 3))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(features)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--exps-dir",
        default="exps",
    )
    p.add_argument("--delete-ckpt", required=True)
    p.add_argument("--spls-ckpt", required=True, help="Exact NON-FREEZE ldmu checkpoint")
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--forget-class-id", type=int, default=None)
    p.add_argument("--fixed-class-set", action="store_true")
    p.add_argument("--samples-per-class", type=int, default=700)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default="visualizations")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(args.seed)

    dataset_name = "cifar10"
    model_name = "resnet18"
    forget_count = 1

    _, transform_test = get_transforms(
        dataset_name, model_name, wo_dataaug=True
    )
    _, testset = get_dataset(dataset_name, transform_test, transform_test)
    num_classes = max(testset.targets) + 1

    forget_id = resolve_forget_class_id(
        args.seed,
        model_name,
        explicit_id=args.forget_class_id,
        fixed_class_set=args.fixed_class_set,
    )

    pretrain_dir = Path(args.exps_dir) / "pretrained_model"
    original_path = pretrain_dir / f"{args.seed}_{dataset_name}_{model_name}_original_model.pth"
    retrain_path = pretrain_dir / (
        f"{args.seed}_{dataset_name}_{model_name}_retrain_forget{forget_count}_model.pth"
    )

    print(f"[forget class id] {forget_id}")
    print(f"[original] {original_path}")
    print(f"[retrain ] {retrain_path}")
    print(f"[delete  ] {args.delete_ckpt}")
    print(f"[SPLS    ] {args.spls_ckpt}")

    models = {
        "Original": load_model(original_path, model_name, num_classes).to(device),
        "Retrain": load_model(retrain_path, model_name, num_classes).to(device),
        "DELETE": load_unlearn_checkpoint(
            Path(args.delete_ckpt), model_name, num_classes, device
        ),
        "SPLS": load_unlearn_checkpoint(
            Path(args.spls_ckpt), model_name, num_classes, device
        ),
    }

    selected = balanced_indices(
        testset.targets, args.samples_per_class, args.seed
    )
    subset = Subset(testset, selected)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    labels = np.asarray(testset.targets)[selected]
    class_names = getattr(testset, "classes", [str(i) for i in range(num_classes)])

    embeddings = {}
    for name, model in models.items():
        print(f"[feature extraction] {name}")
        feats = extract_penultimate_features(model, loader, device)
        print(f"  features: {feats.shape}")
        print(f"[t-SNE] {name}")
        embeddings[name] = run_tsne(feats, args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
    cmap = plt.get_cmap("tab10")

    for ax, (name, z) in zip(axes, embeddings.items()):
        for c in range(num_classes):
            mask = labels == c
            if not np.any(mask):
                continue

            is_forget = c == forget_id
            ax.scatter(
                z[mask, 0],
                z[mask, 1],
                s=18 if is_forget else 8,
                alpha=0.9 if is_forget else 0.55,
                c=[cmap(c)],
                edgecolors="black" if is_forget else "none",
                linewidths=0.5 if is_forget else 0.0,
                label=f"{c}: {class_names[c]}",
            )

        ax.set_title(name, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=5,
        frameon=False,
        fontsize=8,
    )

    # fig.suptitle(
    #     f"CIFAR-10 / ResNet-18 / seed={args.seed} / "
    #     f"forget class={forget_id} ({class_names[forget_id]})",
    #     fontsize=13,
    # )

    stem = out_dir / f"tsne_cifar10_resnet18_seed{args.seed}_forget1"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    
    plt.close(fig)

    print(f"[saved] {stem.with_suffix('.png')}")
   


if __name__ == "__main__":
    main()