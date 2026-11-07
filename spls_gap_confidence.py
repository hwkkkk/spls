"""Reviewer analysis (inference only): (1) gap z_u^o - z_min^o distribution on the forget set,
(2) post-SPLS forget acc / SVC-MIA member ratio per original-confidence bin.
Deterministic test transforms everywhere. SVC-MIA follows evaluation/SVC_MIA.py (confidence
feature, shadow = train-remain subset (|test-remain|) vs test-remain, evaluated on the unlearned model)."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from models import load_model
from utils import get_transforms

ROOT = Path(__file__).resolve().parent
BINS = [0, .5, .9, .99, 1.0]
BIN_LABELS = ["<0.5", "0.5-0.9", "0.9-0.99", ">0.99"]


def load(path, arch, device):
    obj = torch.load(path, map_location="cpu", weights_only=False)  # own checkpoints
    model = obj if isinstance(obj, torch.nn.Module) else load_model(path, arch, 10)
    return model.eval().requires_grad_(False).to(device)


@torch.inference_mode()
def logits(model, ds, idx, device, bs=256):
    out = []
    for x, _ in DataLoader(Subset(ds, list(map(int, idx))), batch_size=bs, num_workers=4):
        out.append(model(x.to(device)).float().cpu())
    z = torch.cat(out)
    assert z.shape[1] == 10, z.shape
    return z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True)
    p.add_argument("--original", required=True)
    p.add_argument("--unlearned", required=True)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--out", type=Path, default=ROOT / "spls_gap_confidence_results")
    a = p.parse_args()
    a.out.mkdir(exist_ok=True)
    dev = torch.device("cuda:0")

    perm = list(OmegaConf.load(ROOT / f"config/cifar10_{a.arch}.yaml").permutation_map)
    random.Random(a.seed).shuffle(perm)
    u = int(perm[0])
    _, tf = get_transforms("cifar10", a.arch, wo_dataaug=True)
    train = CIFAR10(ROOT / "data", train=True, download=False, transform=tf)
    test = CIFAR10(ROOT / "data", train=False, download=False, transform=tf)
    ytr, yte = np.asarray(train.targets), np.asarray(test.targets)
    idx = {"train_forget": np.flatnonzero(ytr == u), "test_forget": np.flatnonzero(yte == u)}
    test_remain = np.flatnonzero(yte != u)
    rng = np.random.default_rng(a.seed)
    shadow_in = rng.choice(np.flatnonzero(ytr != u), size=len(test_remain), replace=False)

    ori, unl = load(a.original, a.arch, dev), load(a.unlearned, a.arch, dev)
    frames = []
    for split, ds in [("train_forget", train), ("test_forget", test)]:
        zo, zn = logits(ori, ds, idx[split], dev), logits(unl, ds, idx[split], dev)
        po = zo.softmax(1)[:, u]
        frames.append(pd.DataFrame({
            "split": split, "index": idx[split],
            "gap": (zo[:, u] - zo.min(1).values).numpy(),
            "p_u_original": po.numpy(), "correct_original": (zo.argmax(1) == u).numpy(),
            "p_u_unlearned": zn.softmax(1)[:, u].numpy(), "correct_unlearned": (zn.argmax(1) == u).numpy()}))
    df = pd.concat(frames, ignore_index=True)

    # SVC-MIA on the unlearned model (same recipe as evaluation/SVC_MIA.py)
    def conf(ds, ix, y):
        return logits(unl, ds, ix, dev).softmax(1)[torch.arange(len(ix)), torch.as_tensor(y[ix])].numpy()
    s_in, s_out = conf(train, shadow_in, ytr), conf(test, test_remain, yte)
    clf = SVC(C=3, gamma="auto", kernel="rbf").fit(
        np.concatenate([s_in, s_out])[:, None], np.r_[np.ones(len(s_in)), np.zeros(len(s_out))])
    df["mia_member"] = clf.predict(df.p_u_unlearned.to_numpy()[:, None]).astype(bool)
    df.insert(0, "arch", a.arch)
    df.to_csv(a.out / f"cifar10_{a.arch}_samples.csv", index=False)

    gap_rows, bin_rows = [], []
    for split, g in df.groupby("split", sort=False):
        q = g.gap.quantile([0, .01, .05, .5])
        gap_rows.append(dict(arch=a.arch, split=split, n=len(g), gap_min=q[0], gap_p01=q[.01], gap_p05=q[.05],
                             gap_median=q[.5], zero_gap_pct=100 * (g.gap <= 0).mean(),
                             gap_lt_1_pct=100 * (g.gap < 1).mean(),
                             orig_acc=100 * g.correct_original.mean()))
        b = pd.cut(g.p_u_original, BINS, labels=BIN_LABELS, include_lowest=True)
        for lab, s in [("all", g)] + list(g.groupby(b, observed=False)):
            bin_rows.append(dict(arch=a.arch, split=split, bin=lab, n=len(s),
                                 orig_acc=100 * s.correct_original.mean() if len(s) else np.nan,
                                 forget_acc_after=100 * s.correct_unlearned.mean() if len(s) else np.nan,
                                 mia_after=100 * s.mia_member.mean() if len(s) else np.nan,
                                 mean_gap=s.gap.mean() if len(s) else np.nan))
    pd.DataFrame(gap_rows).to_csv(a.out / f"cifar10_{a.arch}_gap.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(a.out / f"cifar10_{a.arch}_bins.csv", index=False)
    (a.out / f"cifar10_{a.arch}_manifest.json").write_text(json.dumps(dict(vars(a), out=str(a.out), forget_class=u), indent=2))
    print(pd.DataFrame(gap_rows).round(3).to_string(index=False))
    print(pd.DataFrame(bin_rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
