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
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
from PIL import Image

from utils import *
from trainer import *


# ============================================================
# Checkpoint loader
# ============================================================

def load_unlearn_checkpoint(
    path: Path,
    model_name: str,
    num_classes: int,
    device,
):
    """
    main.py saves unlearned models with:

        torch.save(unlearn_model, path)

    Therefore, normally this loads a full nn.Module.

    state_dict-style checkpoints are also supported as fallback.
    """

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
# Recover forget-class order used in main.py
# ============================================================

def resolve_class_order(
    seed: int,
    model_name: str,
    fixed_class_set=False,
):
    cfg_path = (
        PROJECT_ROOT
        / f"config/cifar10_{model_name}.yaml"
    )

    cfg = OmegaConf.load(cfg_path)

    perm = list(
        cfg.permutation_map
    )

    # Match main.py behavior
    seed_torch(seed)

    if not fixed_class_set:
        random.shuffle(perm)

    return [int(x) for x in perm]


# ============================================================
# Prediction helper
# ============================================================

@torch.no_grad()
def predict_indices(
    model,
    dataset,
    indices,
    device,
    batch_size=64,
):
    subset = Subset(
        dataset,
        indices,
    )

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    preds = []
    probs = []

    model.eval()

    for x, _ in loader:

        x = x.to(device)

        logits = model(x)

        p = logits.softmax(dim=1)

        conf, pred = p.max(dim=1)

        preds.extend(
            pred.cpu().tolist()
        )

        probs.extend(
            conf.cpu().tolist()
        )

    return (
        np.asarray(preds),
        np.asarray(probs),
    )


# ============================================================
# CIFAR image for visualization
# ============================================================

def get_raw_cifar_image(
    dataset,
    idx,
):
    """
    CIFAR-10 raw image is 32x32.

    Only for visualization, resize it to 224x224.
    This does NOT change the input used by the model.
    """

    if hasattr(dataset, "data"):

        raw = Image.fromarray(
            dataset.data[idx]
        )

        raw = raw.resize(
            (224, 224),
            Image.Resampling.BICUBIC,
        )

        return np.asarray(raw)

    # Fallback
    x, _ = dataset[idx]

    x = (
        x.detach()
        .cpu()
        .float()
    )

    x = x - x.min()

    x = (
        x
        / x.max().clamp_min(1e-12)
    )

    x = (
        x.permute(1, 2, 0)
        .numpy()
    )

    return x


# ============================================================
# ResNet Grad-CAM
# ============================================================

class ResNetGradCAM:
    """
    Standard Grad-CAM for ResNet-18.

    Target layer:
        model.layer4[-1]

    Activation shape:
        [B, C, H, W]
    """

    def __init__(
        self,
        model,
    ):
        self.model = model
        self.activation = None

        if not hasattr(
            model,
            "layer4",
        ):
            raise AttributeError(
                "Expected a ResNet-style model "
                "with model.layer4."
            )

        # Last residual block
        self.target_layer = (
            model.layer4[-1]
        )

        self.handle = (
            self.target_layer
            .register_forward_hook(
                self._forward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):
        self.activation = output

        if output.requires_grad:
            output.retain_grad()

    def close(self):
        self.handle.remove()

    def __call__(
        self,
        x,
        target_class,
    ):
        self.model.eval()

        self.model.zero_grad(
            set_to_none=True
        )

        self.activation = None

        # --------------------------------------------------------
        # Forward
        # --------------------------------------------------------

        logits = self.model(x)

        # Grad-CAM target score
        score = (
            logits[:, target_class]
            .sum()
        )

        # --------------------------------------------------------
        # Backward
        # --------------------------------------------------------

        score.backward()

        if self.activation is None:
            raise RuntimeError(
                "Failed to capture layer4 activation."
            )

        if self.activation.grad is None:
            raise RuntimeError(
                "Failed to capture layer4 gradient."
            )

        activation = (
            self.activation
            .detach()
        )

        gradient = (
            self.activation.grad
            .detach()
        )

        if activation.ndim != 4:
            raise RuntimeError(
                "Unexpected ResNet activation shape: "
                f"{tuple(activation.shape)}"
            )

        # --------------------------------------------------------
        # Standard Grad-CAM
        #
        # alpha_k =
        # average gradient over spatial dimensions
        # --------------------------------------------------------

        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights
            * activation
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(cam)

        # --------------------------------------------------------
        # Per-image visualization normalization
        # --------------------------------------------------------

        cam_min = cam.amin(
            dim=(2, 3),
            keepdim=True,
        )

        cam_max = cam.amax(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            cam - cam_min
        ) / (
            cam_max
            - cam_min
            + 1e-12
        )

        return (
            logits.detach(),
            cam.detach(),
        )


# ============================================================
# Sample selection
# ============================================================

def select_samples(
    dataset,
    models,
    target_class,
    n,
    device,
    seed,
    require_all_correct,
):
    """
    Forget samples:
        Original must classify the forget class correctly.

    Retain samples:
        Original, Retrain, DELETE, and SPLS must all
        classify the retained class correctly.
    """

    targets = np.asarray(
        dataset.targets
    )

    candidates = np.flatnonzero(
        targets == target_class
    ).tolist()

    pred_by_model = {}

    for name, model in models.items():

        pred, _ = predict_indices(
            model,
            dataset,
            candidates,
            device,
        )

        pred_by_model[name] = pred

    keep = np.ones(
        len(candidates),
        dtype=bool,
    )

    # --------------------------------------------------------
    # Retain class:
    # all models must classify sample correctly
    # --------------------------------------------------------

    if require_all_correct:

        for pred in pred_by_model.values():

            keep &= (
                pred == target_class
            )

    # --------------------------------------------------------
    # Forget class:
    # Original must classify it correctly
    # --------------------------------------------------------

    else:

        keep &= (
            pred_by_model["Original"]
            == target_class
        )

    filtered = np.asarray(
        candidates
    )[keep]

    if len(filtered) < n:

        print(
            f"[warning] only {len(filtered)} "
            f"preferred samples for class "
            f"{target_class}; "
            f"falling back to arbitrary "
            f"class samples."
        )

        filtered = np.asarray(
            candidates
        )

    rng = np.random.default_rng(
        seed + target_class
    )

    rng.shuffle(filtered)

    return filtered[:n].tolist()


# ============================================================
# Plot
# ============================================================

def plot_attention_grid(
    dataset,
    indices,
    models,
    target_class,
    class_names,
    device,
    output_path,
    figure_title=None,
):
    """
    Figure layout:

        Input | Original | Retrain | DELETE | SPLS

    Number of rows = number of selected samples.
    Default = 4.
    """

    model_names = [
        "Original",
        "Retrain",
        "DELETE",
        "SPLS",
    ]

    cams = {
        name: ResNetGradCAM(
            models[name]
        )
        for name in model_names
    }

    n = len(indices)

    # ========================================================
    # 4 rows x 5 columns
    # ========================================================

    fig, axes = plt.subplots(
        n,
        5,
        figsize=(
            13.0,
            2.55 * n,
        ),
    )

    if n == 1:
        axes = np.expand_dims(
            axes,
            axis=0,
        )

    column_names = [
        "Input",
        "Original",
        "Retrain",
        "DELETE",
        "SPLS",
    ]

    for j, title in enumerate(
        column_names
    ):
        axes[0, j].set_title(
            title,
            fontsize=20,
        )

    # ========================================================
    # Each row = one sample
    # ========================================================

    for row, idx in enumerate(
        indices
    ):

        raw = get_raw_cifar_image(
            dataset,
            idx,
        )

        x, y = dataset[idx]

        y = int(y)

        x = (
            x.unsqueeze(0)
            .to(device)
        )

        # ====================================================
        # Column 0: Input image
        # ====================================================

        axes[row, 0].imshow(raw)

        axes[row, 0].set_ylabel(
            f"sample {row + 1}\n"
            f"true: {class_names[y]}",
            fontsize=9,
        )

        axes[row, 0].axis("off")

        # ====================================================
        # Original / Retrain / DELETE / SPLS
        # ====================================================

        for col, name in enumerate(
            model_names,
            start=1,
        ):

            logits, cam = cams[name](
                x,
                target_class=target_class,
            )

            probability = (
                logits.softmax(dim=1)
            )

            pred = int(
                probability
                .argmax(dim=1)
                .item()
            )

            pred_conf = float(
                probability[
                    0,
                    pred
                ].item()
            )

            target_prob = float(
                probability[
                    0,
                    target_class
                ].item()
            )

            # ------------------------------------------------
            # Upsample CAM to visualization size
            # ------------------------------------------------

            cam_up = F.interpolate(
                cam,
                size=raw.shape[:2],
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            cam_up = (
                cam_up
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            # ------------------------------------------------
            # Base image
            # ------------------------------------------------

            axes[row, col].imshow(
                raw
            )

            # ------------------------------------------------
            # Grad-CAM overlay
            # ------------------------------------------------

            axes[row, col].imshow(
                cam_up,
                cmap="jet",
                alpha=0.48,
                vmin=0.0,
                vmax=1.0,
                interpolation="bilinear",
            )

            # ------------------------------------------------
            # Prediction information
            # ------------------------------------------------

            axes[row, col].set_xlabel(
                f"pred: "
                f"{class_names[pred]} "
                f"({pred_conf:.2f})\n"
                f"p(target)="
                f"{target_prob:.4f}",
                fontsize=8,
            )

            axes[row, col].axis(
                "off"
            )

    # ========================================================
    # Remove hooks
    # ========================================================

    for cam in cams.values():
        cam.close()

    # ========================================================
    # Save
    # ========================================================

    fig.suptitle(
        figure_title,
        fontsize=13,
        y=1.002,
    )

    fig.tight_layout()

    png_path = (
        output_path
        .with_suffix(".png")
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[saved] {png_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Experiment paths
    # --------------------------------------------------------

    parser.add_argument(
        "--exps-dir",
        default=(
            "exps"
        ),
    )

    parser.add_argument(
        "--delete-ckpt",
        required=True,
        help=(
            "Exact CIFAR-10 ResNet-18 "
            "DELETE checkpoint"
        ),
    )

    parser.add_argument(
        "--spls-ckpt",
        required=True,
        help=(
            "Exact NON-FREEZE "
            "CIFAR-10 ResNet-18 "
            "ldmu checkpoint"
        ),
    )

    # --------------------------------------------------------
    # Experiment setting
    # --------------------------------------------------------

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
        "--retain-class-id",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--fixed-class-set",
        action="store_true",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--output-dir",
        default="visualizations",
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

    seed_torch(
        args.seed
    )

    # ========================================================
    # Dataset / model
    # ========================================================

    dataset_name = "cifar10"
    model_name = "resnet18"
    forget_count = 1

    # ========================================================
    # Dataset
    # ========================================================

    _, transform_test = get_transforms(
        dataset_name,
        model_name,
        wo_dataaug=True,
    )

    _, testset = get_dataset(
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
            for i
            in range(num_classes)
        ],
    )

    # ========================================================
    # Resolve forgotten class
    # ========================================================

    class_order = resolve_class_order(
        args.seed,
        model_name,
        fixed_class_set=(
            args.fixed_class_set
        ),
    )

    if args.forget_class_id is not None:

        forget_id = int(
            args.forget_class_id
        )

    else:

        forget_id = (
            class_order[0]
        )

    # ========================================================
    # Resolve retained class
    # ========================================================

    if args.retain_class_id is not None:

        retain_id = int(
            args.retain_class_id
        )

    else:

        retain_id = next(
            c
            for c in class_order
            if c != forget_id
        )

    # ========================================================
    # Original / Retrain paths
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

    # ========================================================
    # Print experiment
    # ========================================================

    print(
        f"[forget class] "
        f"{forget_id}: "
        f"{class_names[forget_id]}"
    )

    print(
        f"[retain class] "
        f"{retain_id}: "
        f"{class_names[retain_id]}"
    )

    print(
        f"[original] "
        f"{original_path}"
    )

    print(
        f"[retrain ] "
        f"{retrain_path}"
    )

    print(
        f"[DELETE  ] "
        f"{args.delete_ckpt}"
    )

    print(
        f"[SPLS    ] "
        f"{args.spls_ckpt}"
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
            Path(
                args.delete_ckpt
            ),
            model_name,
            num_classes,
            device,
        ),

        "SPLS": load_unlearn_checkpoint(
            Path(
                args.spls_ckpt
            ),
            model_name,
            num_classes,
            device,
        ),
    }

    # ========================================================
    # Select forget samples
    #
    # Original only needs to classify these correctly.
    # ========================================================

    forget_indices = select_samples(
        testset,
        models,
        target_class=forget_id,
        n=args.num_samples,
        device=device,
        seed=args.seed,
        require_all_correct=False,
    )

    # ========================================================
    # Select retain samples
    #
    # Original / Retrain / DELETE / SPLS
    # must all classify these correctly.
    # ========================================================

    retain_indices = select_samples(
        testset,
        models,
        target_class=retain_id,
        n=args.num_samples,
        device=device,
        seed=args.seed,
        require_all_correct=True,
    )

    print(
        f"[forget sample indices] "
        f"{forget_indices}"
    )

    print(
        f"[retain sample indices] "
        f"{retain_indices}"
    )

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

    # ========================================================
    # Forget-class visualization
    # ========================================================

    plot_attention_grid(
        testset,
        forget_indices,
        models,
        target_class=forget_id,
        class_names=class_names,
        device=device,
        output_path=(
            out_dir
            / (
                "gradcam_forget_"
                "cifar10_resnet18_"
                f"seed{args.seed}"
            )
        ),
        # figure_title=(
        #     f"Forget class: "
        #     f"{forget_id} "
        #     f"({class_names[forget_id]}) "
        #     f"— Grad-CAM target = "
        #     f"forgotten class"
        # ),
    )

    # ========================================================
    # Retain-class visualization
    # ========================================================

    plot_attention_grid(
        testset,
        retain_indices,
        models,
        target_class=retain_id,
        class_names=class_names,
        device=device,
        output_path=(
            out_dir
            / (
                "gradcam_retain_"
                "cifar10_resnet18_"
                f"seed{args.seed}"
            )
        ),
        # figure_title=(
        #     f"Retain class: "
        #     f"{retain_id} "
        #     f"({class_names[retain_id]}) "
        #     f"— Grad-CAM target = "
        #     f"retained class"
        # ),
    )


if __name__ == "__main__":
    main()