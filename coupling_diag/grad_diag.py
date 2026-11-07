"""Parameter-gradient coupling at the start of unlearning (original model, eval mode), Swin-T CIFAR-100, seeds 2027-2029.
For loss L in {SPLS logit-MSE (min_lam0), logit-space KL (kl_softmax_zmin)} and forget count fc in {1,5,20}:
  g_c  = grad of mean forget loss over the 500 train samples of forget class c   (per-class)
  g_f  = mean_c g_c  (= full forget-set gradient; equal class sizes)
  g_r  = grad of CE on 1024 random retain train samples (fixed per seed/fc)
Reported:
  cos_f_r        cos(g_f, g_r)            < 0  => a step along -g_f increases retain loss (first order)
  cos_c_r        mean_c cos(g_c, g_r)     per-class harm before averaging
  cos_c_c        mean pairwise cos(g_c, g_c')  coherence between forget classes
  cancel         ||g_f|| / mean_c ||g_c||  (1 = no cancellation)
  nontarget_frac fraction of the output-logit gradient norm on non-target logits (coupling in output space)
"""
import sys, random, json, itertools
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from omegaconf import OmegaConf
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from utils import get_transforms, get_dataset
from models import load_model
from spls_ablation.targets import spls_target_and_loss

W = Path(__file__).resolve().parents[1]; S = W / 'new_swin'; OUT = S / 'coupling_diag'
SEEDS, FCS, BS = (2027, 2028, 2029), (1, 5, 20), 50
LOSSES = {'SPLS': 'min_lam0', 'KL': 'kl_softmax_zmin'}
cfg = OmegaConf.load(W / 'config/cifar100_swin-t.yaml')
def order(seed):
    o = OmegaConf.create({'order': list(cfg.permutation_map)}).order
    random.Random(seed).shuffle(o); return [int(x) for x in o]
_, tt = get_transforms('cifar100', 'swin-t', wo_dataaug=False)
train, _ = get_dataset('cifar100', tt, tt)
ytr = np.array(train.targets)

def flat_grad(model):
    return torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])

def grad_over(model, idx, loss_fn):
    model.zero_grad(set_to_none=True)
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(train, idx), batch_size=BS, num_workers=8)
    nt, tot = 0.0, 0.0
    for x, y in dl:
        x, y = x.cuda(), y.cuda()
        z = model(x)
        loss, zg = loss_fn(z, y)
        (loss * len(y) / len(idx)).backward()
        if zg is not None:
            nt += zg[0]; tot += zg[1]
    return flat_grad(model).clone(), (nt / tot if tot else float('nan'))

cos = lambda a, b: F.cosine_similarity(a, b, dim=0).item()
rows = []
for seed in SEEDS:
    model = load_model(S / f'pretrained_model/{seed}_cifar100_swin-t_original_model.pth', 'swin-t', 100); model.eval()
    rng = np.random.RandomState(seed)
    for fc in FCS:
        fcls = order(seed)[:fc]
        ridx = rng.choice(np.where(~np.isin(ytr, fcls))[0], 1024, replace=False)
        g_r, _ = grad_over(model, ridx, lambda z, y: (F.cross_entropy(z, y), None))
        for name, variant in LOSSES.items():
            def lf(z, y):
                with torch.no_grad():
                    zo = z.detach()            # model == original at step 0, so student logits = original logits
                _, loss = spls_target_and_loss(z, zo, y, variant)
                gz = torch.autograd.grad(loss, z, retain_graph=True)[0]
                mask = torch.ones_like(gz, dtype=torch.bool); mask[torch.arange(len(y)), y] = False
                return loss, ((gz[mask] ** 2).sum().sqrt().item(), (gz ** 2).sum().sqrt().item())
            gcs, ntf = [], []
            for c in fcls:
                g, f = grad_over(model, np.where(ytr == c)[0], lf); gcs.append(g); ntf.append(f)
            G = torch.stack(gcs); g_f = G.mean(0)
            pair = [cos(G[i], G[j]) for i, j in itertools.combinations(range(len(gcs)), 2)]
            rows.append(dict(seed=seed, forget=fc, loss=name, cos_f_r=cos(g_f, g_r),
                             cos_c_r=float(np.mean([cos(g, g_r) for g in gcs])),
                             cos_c_c=float(np.mean(pair)) if pair else np.nan,
                             cancel=(g_f.norm() / G.norm(dim=1).mean()).item(),
                             nontarget_frac=float(np.mean(ntf))))
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
            del G, gcs, g_f; torch.cuda.empty_cache()
    del model; torch.cuda.empty_cache()

import pandas as pd
df = pd.DataFrame(rows); df.to_csv(OUT / 'grad_diag_raw.csv', index=False)
s = df.drop(columns='seed').groupby(['forget', 'loss']).agg(['mean', 'std']); s.to_csv(OUT / 'grad_diag_summary.csv')
print(df.drop(columns='seed').groupby(['forget', 'loss']).mean().round(4).to_string())
