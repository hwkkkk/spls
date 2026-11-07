"""Suppression-target construction for the SPLS (= ldmu) target ablation.

All variants take the frozen original model's logits z^o (shape [B, K]) and only
replace the forget-class (target) logit; the K-1 non-target logits are kept as is.
Standard deviations use correction=0 (unbiased=False) over all K logits.
"""
import torch
import torch.nn.functional as F

# name -> (kind, lambda). "min_std" with lambda=0 is the original SPLS target (z_min).
VARIANTS = {
    "min_lam0": ("min_std", 0.0),
    "min_lam1": ("min_std", 1.0),
    "min_lam3": ("min_std", 3.0),
    "min_lam5": ("min_std", 5.0),
    "min_lam7": ("min_std", 7.0),
    "min_lam10": ("min_std", 10.0),
    "min_lam20": ("min_std", 20.0),
    "min_lam30": ("min_std", 30.0),
    "min_lam50": ("min_std", 50.0),
    "min_lam100": ("min_std", 100.0),
    "second_min": ("second_min", None),
    "mean_minus_std": ("mean_std", 1.0),
}

# the 11 new ablation variants (min_lam0 is the existing SPLS baseline)
ABLATION_VARIANTS = ["min_lam1", "min_lam3", "min_lam5", "min_lam7", "min_lam10", "min_lam20", "min_lam30", "min_lam50",
                     "min_lam100", "second_min", "mean_minus_std"]


# loss ablation (target z_min, i.e. the existing SPLS logit target, compared in probability space)
#   kl_softmax_zmin   : KLDiv(batchmean)(log_softmax(student), softmax(SPLS target logits))
#   mse_masked_softmax: DELETE-style target, forget logit masked to -1e10 then softmax;
#                       MSE(mean)(softmax(student), masked softmax)
LOSS_VARIANTS = ["kl_softmax_zmin", "mse_masked_softmax"]
MASK_VALUE = -1e10  # method/delete.py


def target_value(orig_logits, kind, lam=None):
    """Return the replacement value for the target logit, shape [B]."""
    z = orig_logits
    if kind == "min_std":
        return z.min(dim=1).values - lam * z.std(dim=1, unbiased=False)
    if kind == "second_min":
        return z.sort(dim=1).values[:, 1]
    if kind == "mean_std":
        return z.mean(dim=1) - lam * z.std(dim=1, unbiased=False)
    raise ValueError(f"unknown target kind: {kind}")


def build_target(orig_logits, y, variant):
    """Copy of orig_logits with the label-y logit replaced according to `variant`."""
    kind, lam = VARIANTS[variant]
    target = orig_logits.clone()
    batch_idx = torch.arange(target.size(0), device=target.device)
    target[batch_idx, y] = target_value(orig_logits, kind, lam)
    return target


def spls_target_and_loss(student_logits, orig_logits, y, variant):
    """(target, loss) for one batch. Target-variants use the existing SPLS logit MSE."""
    if variant in VARIANTS:
        target = build_target(orig_logits, y, variant)
        return target, F.mse_loss(student_logits, target, reduction="mean")
    if variant == "kl_softmax_zmin":
        target = F.softmax(build_target(orig_logits, y, "min_lam0"), dim=1)
        return target, F.kl_div(F.log_softmax(student_logits, dim=1), target, reduction="batchmean")
    if variant == "mse_masked_softmax":
        masked = orig_logits.clone()
        masked[torch.arange(masked.size(0), device=masked.device), y] = MASK_VALUE
        target = F.softmax(masked, dim=1)
        return target, F.mse_loss(F.softmax(student_logits, dim=1), target, reduction="mean")
    raise ValueError(f"unknown variant: {variant}")
