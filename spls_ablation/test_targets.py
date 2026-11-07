"""Unit tests for the SPLS target ablation.  Run: CUDA_VISIBLE_DEVICES=1 python spls_ablation/test_targets.py"""
import logging
import math
import sys
from pathlib import Path

import tempfile
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spls_ablation.targets import VARIANTS, build_target, target_value  # noqa: E402

Z = torch.tensor([[3.0, -1.0, 0.5, -2.0],
                  [0.2, 4.0, -3.0, 1.0]])
Y = torch.tensor([0, 1])


def pop_std(row):
    m = sum(row) / len(row)
    return math.sqrt(sum((v - m) ** 2 for v in row) / len(row))


def test_target_value_by_hand():
    for variant in VARIANTS:
        kind, lam = VARIANTS[variant]
        target = build_target(Z, Y, variant)
        for i, row in enumerate(Z.tolist()):
            s = pop_std(row)
            if kind == "min_std":
                expected = min(row) - lam * s
            elif kind == "second_min":
                expected = sorted(row)[1]
            else:
                expected = sum(row) / len(row) - lam * s
            assert abs(target[i, Y[i]].item() - expected) < 1e-5, (variant, i)


def test_non_target_logits_unchanged_and_input_untouched():
    for variant in VARIANTS:
        z = Z.clone()
        target = build_target(z, Y, variant)
        assert torch.equal(z, Z)
        mask = torch.ones_like(Z, dtype=torch.bool)
        mask[torch.arange(2), Y] = False
        assert torch.equal(target[mask], Z[mask])


def test_std_uses_correction_zero():
    got = target_value(Z, "min_std", 1.0)
    expected = Z.min(1).values - Z.std(1, correction=0)
    assert torch.allclose(got, expected)
    assert not torch.allclose(got, Z.min(1).values - Z.std(1))  # unbiased would differ


def test_mean_minus_std_is_one_sigma():
    assert VARIANTS["mean_minus_std"] == ("mean_std", 1.0)
    got = build_target(Z, Y, "mean_minus_std")[torch.arange(2), Y]
    assert torch.allclose(got, Z.mean(1) - Z.std(1, correction=0))


def test_lambda0_equals_existing_spls_target():
    """min_lam0 must equal the existing ldmu targets ('lowest' and 'lowest-std' with k=0)."""
    batch_idx = torch.arange(Z.size(0))
    # copied from method/ldmu.py, soft_label == "lowest"
    lowest = Z.clone()
    tmp = lowest.clone()
    lowest[batch_idx, Y] = tmp.min(dim=1).values
    # copied from method/ldmu.py, soft_label == "lowest-std", k = 0
    lowest_std = Z.clone()
    lowest_std[batch_idx, Y] = lowest_std.min(dim=1).values - lowest_std.std(dim=1) * 0
    ours = build_target(Z, Y, "min_lam0")
    assert torch.equal(ours, lowest)
    assert torch.equal(ours, lowest_std)


def test_loss_variants_match_reference_code():
    from torch import nn
    import torch.nn.functional as F
    from spls_ablation.targets import spls_target_and_loss
    torch.manual_seed(0)
    student = torch.randn(2, 4)
    # KL on softmaxed SPLS (z_min) target, written like method/delete.py
    t = Z.clone()
    t[torch.arange(2), Y] = Z.min(1).values
    ref = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(student, dim=1), F.softmax(t, dim=1))
    _, got = spls_target_and_loss(student, Z, Y, "kl_softmax_zmin")
    assert torch.allclose(got, ref)
    # DELETE masked softmax target (-1e10), MSE in probability space
    m = Z.clone()
    m[torch.arange(2), Y] = -1e10
    target_prob = F.softmax(m, dim=1)
    assert torch.all(target_prob[torch.arange(2), Y] == 0)
    ref = nn.MSELoss(reduction="mean")(F.softmax(student, dim=1), target_prob)
    _, got = spls_target_and_loss(student, Z, Y, "mse_masked_softmax")
    assert torch.allclose(got, ref)
    assert torch.equal(Z, torch.tensor([[3.0, -1.0, 0.5, -2.0], [0.2, 4.0, -3.0, 1.0]]))  # input untouched


def test_lambda0_training_matches_existing_ldmu():
    for soft_label in ["lowest", "lowest-std"]:
        _check_training_equivalence(Path(tempfile.mkdtemp()), soft_label)


def _check_training_equivalence(tmp_path, soft_label):
    """Same model + data: existing method.ldmu and the ablation loop (min_lam0) give identical weights."""
    import method
    from spls_ablation.ablation_ldmu import make_ablation_ldmu

    torch.manual_seed(0)
    x = torch.randn(64, 12)
    y = torch.randint(0, 5, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)
    loader_dict = {k: loader for k in ["train_forget", "train_remain", "test_forget", "test_remain"]}
    model = torch.nn.Sequential(torch.nn.Linear(12, 32), torch.nn.ReLU(), torch.nn.Linear(32, 5))

    logger = logging.getLogger(f"test_{soft_label}")
    handler = logging.StreamHandler()

    def run(fn, sub):
        (tmp_path / sub).mkdir()
        torch.manual_seed(1)
        return fn(0.0, model, loader, 3, 1e-2, logger, handler, loader_dict, tmp_path / sub, soft_label)

    ref = run(method.ldmu, "ref")
    ours = run(make_ablation_ldmu("min_lam0"), "ours")
    for p_ref, p_ours in zip(ref.parameters(), ours.parameters()):
        assert torch.equal(p_ref, p_ours)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
