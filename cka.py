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
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf

import matplotlib.pyplot as plt

from utils import *
from trainer import *


# ============================================================
# Model loading
# ============================================================

def load_unlearn_checkpoint(
    path: Path,
    model_name: str,
    num_classes: int,
    device,
):
    try:
        obj = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        obj = torch.load(
            path,
            map_location=device,
        )

    if isinstance(obj, nn.Module):
        return obj.to(device)

    return load_model(
        path,
        model_name,
        num_classes,
    ).to(device)


# ============================================================
# Forget class order
# ============================================================

def resolve_class_order(
    seed,
    dataset_name,
    model_name,
    fixed_class_set=False,
):
    cfg_path = (
        PROJECT_ROOT
        / f"config/{dataset_name}_{model_name}.yaml"
    )

    cfg = OmegaConf.load(cfg_path)

    permutation_map = list(
        cfg.permutation_map
    )

    # Match main.py
    seed_torch(seed)

    if not fixed_class_set:
        random.shuffle(permutation_map)

    return [
        int(x)
        for x in permutation_map
    ]


# ============================================================
# Sample selection for CKA
# ============================================================

def sample_class_indices(
    targets,
    target_classes,
    num_samples,
    seed,
):
    targets = np.asarray(targets)

    mask = np.isin(
        targets,
        list(target_classes),
    )

    indices = np.flatnonzero(mask)

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    if num_samples is not None:
        indices = indices[
            :min(num_samples, len(indices))
        ]

    return indices.tolist()


def balanced_retain_indices(
    targets,
    retain_classes,
    num_samples,
    seed,
):
    """
    Approximately balanced retained-class sampling.
    """

    targets = np.asarray(targets)

    rng = np.random.default_rng(seed)

    selected = []

    base = num_samples // len(retain_classes)
    remainder = num_samples % len(retain_classes)

    for i, c in enumerate(retain_classes):

        idx = np.flatnonzero(
            targets == c
        ).copy()

        rng.shuffle(idx)

        take = base + (
            1 if i < remainder else 0
        )

        selected.extend(
            idx[:take].tolist()
        )

    rng.shuffle(selected)

    return selected[:num_samples]


# ============================================================
# ResNet layer1~4 embedding extractor
# ============================================================

class ResNetEmbeddingExtractor:

    def __init__(self, model):

        self.model = model

        for name in [
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]:
            if not hasattr(model, name):
                raise AttributeError(
                    f"Model does not contain {name}"
                )

        self.layers = {
            "layer1": model.layer1,
            "layer2": model.layer2,
            "layer3": model.layer3,
            "layer4": model.layer4,
        }

        self.current = {}
        self.handles = []

        for name, module in self.layers.items():

            handle = module.register_forward_hook(
                self._make_hook(name)
            )

            self.handles.append(handle)

    def _make_hook(self, name):

        def hook(module, inputs, output):
            self.current[name] = (
                output.detach()
            )

        return hook

    def close(self):

        for handle in self.handles:
            handle.remove()

    @torch.no_grad()
    def extract(
        self,
        loader,
        device,
        return_labels=False,
    ):
        """
        [B,C,H,W]
            ↓
        Global Average Pooling
            ↓
        [B,C]
        """

        self.model.eval()

        collected = {
            name: []
            for name in self.layers
        }

        labels = []

        for x, y in loader:

            x = x.to(
                device,
                non_blocking=True,
            )

            self.current = {}

            _ = self.model(x)

            for name in self.layers:

                feat = self.current[name]

                if feat.ndim == 4:

                    feat = (
                        F.adaptive_avg_pool2d(
                            feat,
                            1,
                        )
                        .flatten(1)
                    )

                else:
                    feat = feat.flatten(1)

                collected[name].append(
                    feat.cpu()
                )

            if return_labels:
                labels.append(
                    y.cpu()
                )

        embeddings = {
            name: torch.cat(
                values,
                dim=0,
            )
            for name, values
            in collected.items()
        }

        if return_labels:

            labels = torch.cat(
                labels,
                dim=0,
            )

            return embeddings, labels

        return embeddings


# ============================================================
# Linear CKA
# ============================================================

def linear_cka(X, Y):
    """
    Linear CKA.

    X: [N, Dx]
    Y: [N, Dy]

    Higher = more similar representation geometry.
    """

    X = X.double()
    Y = Y.double()

    X = X - X.mean(
        dim=0,
        keepdim=True,
    )

    Y = Y - Y.mean(
        dim=0,
        keepdim=True,
    )

    XY = X.T @ Y

    numerator = (
        XY.pow(2).sum()
    )

    XX = X.T @ X
    YY = Y.T @ Y

    denominator = torch.sqrt(
        XX.pow(2).sum()
        * YY.pow(2).sum()
    )

    return float(
        (
            numerator
            / (denominator + 1e-12)
        ).item()
    )


# ============================================================
# Extract CKA embeddings
# ============================================================

def extract_cka_embeddings(
    models,
    loader,
    device,
):
    outputs = {}

    for name, model in models.items():

        print(
            f"[CKA embedding extraction] {name}"
        )

        extractor = (
            ResNetEmbeddingExtractor(model)
        )

        outputs[name] = extractor.extract(
            loader,
            device,
        )

        extractor.close()

        for layer, feat in outputs[name].items():

            print(
                f"  {layer}: {tuple(feat.shape)}"
            )

    return outputs


# ============================================================
# CKA dataframe
# ============================================================

def calculate_cka_results(
    embeddings,
    split_name,
):
    """
    Retrain is ALWAYS the reference.
    """

    rows = []

    layers = [
        "layer1",
        "layer2",
        "layer3",
        "layer4",
    ]

    reference = "Retrain"

    methods = [
        "Original",
        "DELETE",
        "SPLS",
    ]

    for method in methods:

        for layer in layers:

            cka = linear_cka(
                embeddings[
                    reference
                ][layer],
                embeddings[
                    method
                ][layer],
            )

            rows.append({
                "split": split_name,
                "reference": reference,
                "method": method,
                "layer": layer,
                "cka": cka,
            })

    return rows


# ============================================================
# CKA plot
# ============================================================

def plot_cka(
    df,
    output_path,
):
    layers = [
        "layer1",
        "layer2",
        "layer3",
        "layer4",
    ]

    methods = [
        "Original",
        "DELETE",
        "SPLS",
    ]

    x = np.arange(len(layers))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.2),
    )

    for ax, split, title in [
        (
            axes[0],
            "retain",
            "(a) Retain samples",
        ),
        (
            axes[1],
            "forget",
            "(b) Forget samples",
        ),
    ]:

        subset = df[
            df["split"] == split
        ]

        for method in methods:

            values = []

            for layer in layers:

                value = subset[
                    (subset["method"] == method)
                    & (subset["layer"] == layer)
                ]["cka"].iloc[0]

                values.append(value)

            ax.plot(
                x,
                values,
                marker="o",
                linewidth=2,
                label=method,
            )

        ax.set_title(
            title,
            fontsize=12,
        )

        ax.set_ylabel(
            "CKA with Retrain ↑"
        )

        ax.set_xticks(x)

        ax.set_xticklabels([
            "Layer 1",
            "Layer 2",
            "Layer 3",
            "Layer 4",
        ])

        ax.set_ylim(
            0.0,
            1.02,
        )

        ax.grid(
            alpha=0.25
        )

        ax.legend(
            frameon=False
        )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[saved CKA figure] {output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    # ========================================================
    # Experiment setting
    # ========================================================

    parser.add_argument(
        "--dataset-name",
        choices=[
            "cifar10",
            "cifar100",
        ],
        default="cifar100",
    )

    parser.add_argument(
        "--model-name",
        default="resnet18",
    )

    parser.add_argument(
        "--forget-count",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--exps-dir",
        default=(
            "exps"
        ),
    )

    parser.add_argument(
        "--delete-ckpt",
        required=True,
    )

    parser.add_argument(
        "--spls-ckpt",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
    )

    parser.add_argument(
        "--forget-class-id",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--fixed-class-set",
        action="store_true",
    )

    # ========================================================
    # CKA setting
    # ========================================================

    parser.add_argument(
        "--cka-samples",
        type=int,
        default=500,
        help=(
            "Maximum samples per CKA split. "
            "Forget split may be smaller."
        ),
    )

    # ========================================================
    # DataLoader
    # ========================================================

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    # ========================================================
    # Output
    # ========================================================

    parser.add_argument(
        "--output-dir",
        default="representation_analysis",
    )

    args = parser.parse_args()

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[device] {device}"
    )

    seed_torch(args.seed)

    dataset_name = (
        args.dataset_name
    )

    model_name = (
        args.model_name
    )

    forget_count = (
        args.forget_count
    )

    # ========================================================
    # Dataset
    #
    # No augmentation for representation analysis.
    # ========================================================

    _, transform_test = get_transforms(
        dataset_name,
        model_name,
        wo_dataaug=True,
    )

    trainset, testset = get_dataset(
        dataset_name,
        transform_test,
        transform_test,
    )

    num_classes = (
        max(testset.targets)
        + 1
    )

    class_names = getattr(
        testset,
        "classes",
        [
            str(i)
            for i in range(num_classes)
        ],
    )

    # ========================================================
    # Forget class
    # ========================================================

    class_order = resolve_class_order(
        args.seed,
        dataset_name,
        model_name,
        fixed_class_set=(
            args.fixed_class_set
        ),
    )

    if args.forget_class_id is None:

        forget_id = (
            class_order[0]
        )

    else:

        forget_id = int(
            args.forget_class_id
        )

    retain_classes = [
        c
        for c in range(num_classes)
        if c != forget_id
    ]

    print(
        f"[dataset] {dataset_name}"
    )

    print(
        f"[model] {model_name}"
    )

    print(
        f"[forget class] "
        f"{forget_id}: "
        f"{class_names[forget_id]}"
    )

    # ========================================================
    # Checkpoints
    # ========================================================

    pretrain_dir = (
        Path(args.exps_dir)
        / "pretrained_model"
    )

    original_path = (
        pretrain_dir
        / (
            f"{args.seed}_"
            f"{dataset_name}_"
            f"{model_name}_"
            f"original_model.pth"
        )
    )

    retrain_path = (
        pretrain_dir
        / (
            f"{args.seed}_"
            f"{dataset_name}_"
            f"{model_name}_"
            f"retrain_forget"
            f"{forget_count}_"
            f"model.pth"
        )
    )

    print(
        f"[Original] {original_path}"
    )

    print(
        f"[Retrain ] {retrain_path}"
    )

    print(
        f"[DELETE  ] {args.delete_ckpt}"
    )

    print(
        f"[SPLS    ] {args.spls_ckpt}"
    )

    # ========================================================
    # Load models
    # ========================================================

    models = {

        "Original": load_model(
            original_path,
            model_name,
            num_classes,
        ).to(device),

        "Retrain": load_model(
            retrain_path,
            model_name,
            num_classes,
        ).to(device),

        "DELETE": load_unlearn_checkpoint(
            Path(args.delete_ckpt),
            model_name,
            num_classes,
            device,
        ),

        "SPLS": load_unlearn_checkpoint(
            Path(args.spls_ckpt),
            model_name,
            num_classes,
            device,
        ),
    }

    # ========================================================
    # Output directory
    # ========================================================

    out_dir = Path(
        args.output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"{dataset_name}_"
        f"{model_name}_"
        f"forget{forget_count}_"
        f"seed{args.seed}"
    )

    # ========================================================
    #
    # CKA
    #
    # ========================================================

    print(
        "\n"
        "========================================\n"
        "              CKA ANALYSIS\n"
        "========================================"
    )

    targets = np.asarray(
        testset.targets
    )

    # Use same N for retain / forget CKA where possible.
    forget_candidates = np.flatnonzero(
        targets == forget_id
    )

    cka_n = min(
        args.cka_samples,
        len(forget_candidates),
    )

    print(
        f"[CKA samples per split] {cka_n}"
    )

    forget_indices = (
        sample_class_indices(
            testset.targets,
            target_classes=[
                forget_id
            ],
            num_samples=cka_n,
            seed=args.seed,
        )
    )

    retain_indices = (
        balanced_retain_indices(
            testset.targets,
            retain_classes,
            num_samples=cka_n,
            seed=args.seed + 1,
        )
    )

    forget_loader = DataLoader(
        Subset(
            testset,
            forget_indices,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    retain_loader = DataLoader(
        Subset(
            testset,
            retain_indices,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(
        "\n===== RETAIN CKA EMBEDDINGS ====="
    )

    retain_embeddings = (
        extract_cka_embeddings(
            models,
            retain_loader,
            device,
        )
    )

    print(
        "\n===== FORGET CKA EMBEDDINGS ====="
    )

    forget_embeddings = (
        extract_cka_embeddings(
            models,
            forget_loader,
            device,
        )
    )

    cka_rows = []

    cka_rows += (
        calculate_cka_results(
            retain_embeddings,
            split_name="retain",
        )
    )

    cka_rows += (
        calculate_cka_results(
            forget_embeddings,
            split_name="forget",
        )
    )

    cka_df = pd.DataFrame(
        cka_rows
    )

    print(
        "\n===== CKA RESULTS ====="
    )

    print(
        cka_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    cka_csv = (
        out_dir
        / f"cka_{stem}.csv"
    )

    cka_df.to_csv(
        cka_csv,
        index=False,
    )

    cka_png = (
        out_dir
        / f"cka_{stem}.png"
    )

    plot_cka(
        cka_df,
        cka_png,
    )

    # Free CKA feature tensors
    del retain_embeddings
    del forget_embeddings

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # Done
    # ========================================================

    print(
        "\n===== DONE ====="
    )

    print(
        f"CKA CSV   : {cka_csv}"
    )

    print(
        f"CKA Figure: {cka_png}"
    )



if __name__ == "__main__":
    main()