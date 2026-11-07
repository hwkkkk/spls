"""Function-space effect of ONE SPLS gradient step (full forget-set gradient, plain SGD, eval mode) at lr in {1e-3, 3e-3}.
ResNet-18 vs Swin-T, CIFAR-100, forget 1/5/20, seeds 2027-2029.
  forget_gap_drop : decrease of (z_y - z_min) on train-forget samples   (forgetting progress)
  retain_kl       : mean KL(p_orig || p_new) on 2000 random test-retain samples   (collateral change)
  retain_acc_drop : test-retain accuracy change on those samples (pp)
"""
import sys, random, json, copy
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from omegaconf import OmegaConf
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from utils import get_transforms, get_dataset
from models import load_model
from spls_ablation.targets import spls_target_and_loss
W = Path(__file__).resolve().parents[1]
import os
DS = os.environ.get('DS', 'cifar100')
CKPT = {'swin-t': W / ('new_swin/pretrained_model/{s}_' + DS + '_swin-t_original_model.pth'),
        'resnet18': W / ('revision/{s}_' + DS + '_resnet18_original_model.pth')}
FCS = (1,) if DS == 'cifar10' else (1, 5, 20)
NC = 10 if DS == 'cifar10' else 100
LRS = (1e-3, 3e-3)

@torch.no_grad()
def out(m, ds, idx):
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=250, num_workers=8)
    return torch.cat([m(x.cuda()).float() for x, _ in dl])

rows = []
for model_name, ck in CKPT.items():
    cfg = OmegaConf.load(W / f'config/{DS}_{model_name}.yaml')
    _, tt = get_transforms(DS, model_name, wo_dataaug=False)
    train, test = get_dataset(DS, tt, tt); ytr, yte = np.array(train.targets), np.array(test.targets)
    for s in (2027, 2028, 2029):
        o = OmegaConf.create({'order': list(cfg.permutation_map)}).order; random.Random(s).shuffle(o)
        m0 = load_model(str(ck).format(s=s), model_name, NC); m0.eval()
        for fc in FCS:
            f = [int(c) for c in o[:fc]]
            fidx = np.where(np.isin(ytr, f))[0]
            ridx = np.random.RandomState(s).choice(np.where(~np.isin(yte, f))[0], 2000, replace=False)
            yf, yr = torch.tensor(ytr[fidx]).cuda(), torch.tensor(yte[ridx]).cuda()
            zf0, zr0 = out(m0, train, fidx), out(m0, test, ridx)
            gap0 = (zf0.gather(1, yf[:, None])[:, 0] - zf0.min(1).values).mean().item()
            acc0 = (zr0.argmax(1) == yr).float().mean().item() * 100
            m0.zero_grad(set_to_none=True)
            dl = torch.utils.data.DataLoader(torch.utils.data.Subset(train, fidx), batch_size=50, num_workers=8)
            for x, y in dl:
                x, y = x.cuda(), y.cuda(); z = m0(x)
                _, loss = spls_target_and_loss(z, z.detach(), y, 'min_lam0'); (loss * len(y) / len(fidx)).backward()
            grads = [p.grad.detach().clone() if p.grad is not None else None for p in m0.parameters()]
            m0.zero_grad(set_to_none=True)
            for lr in LRS:
                m = copy.deepcopy(m0)
                with torch.no_grad():
                    for p, g in zip(m.parameters(), grads):
                        if g is not None: p -= lr * g
                zf, zr = out(m, train, fidx), out(m, test, ridx)
                gap = (zf.gather(1, yf[:, None])[:, 0] - zf.min(1).values).mean().item()
                kl = F.kl_div(F.log_softmax(zr, 1), F.log_softmax(zr0, 1), log_target=True, reduction='batchmean').item()
                r = dict(model=model_name, seed=s, forget=fc, lr=lr, forget_gap0=gap0, forget_gap_drop=gap0 - gap,
                         retain_kl=kl, retain_acc_drop=acc0 - (zr.argmax(1) == yr).float().mean().item() * 100,
                         kl_per_gap=kl / max(gap0 - gap, 1e-9))
                rows.append(r); print(json.dumps({k: (float(f'{v:.4g}') if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
                del m
        del m0; torch.cuda.empty_cache()
import pandas as pd
df = pd.DataFrame(rows); df.to_csv(W / f'new_swin/coupling_diag/step_func_raw{"" if DS == "cifar100" else "_" + DS}.csv', index=False)
pd.set_option('display.width', 250)
print(df.drop(columns='seed').groupby(['model', 'forget', 'lr']).mean().to_string(float_format=lambda v: f'{v:.4g}'))
