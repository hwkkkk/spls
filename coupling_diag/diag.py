"""Gradient-coupling diagnostic (Swin-T, CIFAR-100, new ckpts): SPLS (logit MSE, λ=0) vs logit-space KL, forget 1/5/20, seeds 2027-2029.
 (1) Is the per-class retain accuracy drop concentrated on classes the original model confuses with the forget classes?
     coupling weight w_c = mean over train-forget samples of original softmax prob of retain class c, renormalised over retain
     classes (= the non-target mass the KL target pushes up).
 (2) Retain-data output drift vs original: mean KL(p_orig || p_unl) on test-retain samples.
"""
import sys, random, glob, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.stats import spearmanr
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from utils import get_transforms, get_dataset
from models import load_model

W = Path(__file__).resolve().parents[1]; S = W / 'new_swin'; OUT = S / 'coupling_diag'
SEEDS, FCS, TOPK = (2027, 2028, 2029), (1, 5, 20), 5
cfg = OmegaConf.load(W / 'config/cifar100_swin-t.yaml')

def order(seed):  # = main.py: random.seed via seed_torch, ListConfig shuffle
    o = OmegaConf.create({'order': list(cfg.permutation_map)}).order
    random.Random(seed).shuffle(o); return [int(x) for x in o]

_, tt = get_transforms('cifar100', 'swin-t', wo_dataaug=False)
train, test = get_dataset('cifar100', tt, tt)
ytr, yte = np.array(train.targets), np.array(test.targets)

@torch.no_grad()
def logits(model, ds, idx):
    model.eval(); out = []
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=256, num_workers=8)
    for x, _ in dl:
        out.append(model(x.cuda()).float().cpu())
    return torch.cat(out)

def one(pattern):
    f = sorted(glob.glob(str(pattern))); assert len(f) == 1, (pattern, f); return f[0]

rows = []
for seed in SEEDS:
    orig = load_model(S / f'pretrained_model/{seed}_cifar100_swin-t_original_model.pth', 'swin-t', 100)
    te_idx = np.arange(len(yte)); zo_te = logits(orig, test, te_idx)
    for fc in FCS:
        fcls = order(seed)[:fc]; rcls = [c for c in range(100) if c not in fcls]
        tr_f = np.where(np.isin(ytr, fcls))[0]
        po_f = F.softmax(logits(orig, train, tr_f), 1)[:, rcls]
        w = (po_f / po_f.sum(1, keepdim=True)).mean(0).numpy()             # coupling weight per retain class
        top = np.argsort(-w)[:TOPK]
        ret_mask = ~np.isin(yte, fcls)
        acc_o = np.array([(zo_te[yte == c].argmax(1).numpy() == c).mean() * 100 for c in rcls])
        for meth, pat in (('SPLS', OUT / f'checkpoints/{seed}_cifar100_swin-t_forget{fc}_min_lam0_*.pth'),
                          ('KL', S / f'analysis_{seed}/loss_ablation/checkpoints/{seed}_cifar100_swin-t_forget{fc}_kl_softmax_zmin_*.pth')):
            m = load_model(one(pat), 'swin-t', 100); zu = logits(m, test, te_idx); del m
            pred = zu.argmax(1).numpy()
            forget_acc = (pred[~ret_mask] == yte[~ret_mask]).mean() * 100
            acc_u = np.array([(pred[yte == c] == c).mean() * 100 for c in rcls])
            drop = acc_o - acc_u
            pos = np.clip(drop, 0, None)
            kl = F.kl_div(F.log_softmax(zu[ret_mask], 1), F.log_softmax(zo_te[ret_mask], 1), log_target=True,
                          reduction='none').sum(1).mean().item()
            # where do wrongly-flipped retain predictions go? fraction that land on top-K confusers
            flip = ret_mask & (zo_te.argmax(1).numpy() == yte) & (pred != yte)
            rows.append(dict(seed=seed, forget=fc, method=meth, forget_acc=forget_acc,
                             retain_acc=(pred[ret_mask] == yte[ret_mask]).mean() * 100,
                             drop_topK=drop[top].mean(), drop_rest=np.delete(drop, top).mean(),
                             share_drop_topK=pos[top].sum() / max(pos.sum(), 1e-9) * 100,
                             spearman_w_drop=spearmanr(w, drop).correlation,
                             retain_KL=kl, n_flip=int(flip.sum()),
                             flip_to_forget_neighbors=float(np.isin(pred[flip], np.array(rcls)[top]).mean() * 100) if flip.any() else np.nan))
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
    del orig; torch.cuda.empty_cache()

import pandas as pd
df = pd.DataFrame(rows); df.to_csv(OUT / 'coupling_diag_raw.csv', index=False)
agg = df.groupby(['forget', 'method']).agg(['mean', 'std']).drop(columns='seed')
agg.to_csv(OUT / 'coupling_diag_summary.csv')
pd.set_option('display.width', 250)
print(df.groupby(['forget', 'method']).mean(numeric_only=True).drop(columns='seed').round(3).to_string())
