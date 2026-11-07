"""Why Swin-T needs lr 1e-3 while ResNet-18 uses 3e-3 (CIFAR-100, forget 1/5/20, seeds 2027-2029), measured at the start of unlearning.
SPLS loss = MSELoss(mean over B x C) => d loss / d z_y = 2 (z_y - z_min) / (B C): the step is set by the logit gap z_y - z_min.
Reported per (model, loss): logit gap, logit scale, ||g|| (param grad of the mean forget loss, eval mode), ||theta||,
and relative first-step size lr * ||g|| / ||theta|| for lr in {1e-3, 3e-3}.
"""
import sys, random, json
from pathlib import Path
import numpy as np, torch
from omegaconf import OmegaConf
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from utils import get_transforms, get_dataset
from models import load_model
from spls_ablation.targets import spls_target_and_loss
W = Path(__file__).resolve().parents[1]
CKPT = {'swin-t': W / 'new_swin/pretrained_model/{s}_cifar100_swin-t_original_model.pth',
        'resnet18': W / 'revision/{s}_cifar100_resnet18_original_model.pth'}
rows = []
for model_name, ck in CKPT.items():
    cfg = OmegaConf.load(W / f'config/cifar100_{model_name}.yaml')
    _, tt = get_transforms('cifar100', model_name, wo_dataaug=False)
    train, _ = get_dataset('cifar100', tt, tt); ytr = np.array(train.targets)
    for s in (2027, 2028, 2029):
        o = OmegaConf.create({'order': list(cfg.permutation_map)}).order; random.Random(s).shuffle(o)
        m = load_model(str(ck).format(s=s), model_name, 100); m.eval()
        theta = torch.sqrt(sum((p.detach().float() ** 2).sum() for p in m.parameters())).item()
        for fc in (1, 5, 20):
            idx = np.where(np.isin(ytr, [int(c) for c in o[:fc]]))[0]
            dl = torch.utils.data.DataLoader(torch.utils.data.Subset(train, idx), batch_size=50, num_workers=8)
            for loss_name, variant in (('SPLS', 'min_lam0'), ('KL', 'kl_softmax_zmin')):
                m.zero_grad(set_to_none=True); gaps, scales = [], []
                for x, y in dl:
                    x, y = x.cuda(), y.cuda(); z = m(x)
                    zd = z.detach(); gaps.append((zd.gather(1, y[:, None])[:, 0] - zd.min(1).values).cpu())
                    scales.append(zd.std(1).cpu())
                    _, loss = spls_target_and_loss(z, zd, y, variant)
                    (loss * len(y) / len(idx)).backward()
                g = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in m.parameters() if p.grad is not None)).item()
                r = dict(model=model_name, seed=s, forget=fc, loss=loss_name, logit_gap=torch.cat(gaps).mean().item(),
                         logit_std=torch.cat(scales).mean().item(), grad_norm=g, theta_norm=theta,
                         rel_step_1e3=1e-3 * g / theta, rel_step_3e3=3e-3 * g / theta)
                rows.append(r); print(json.dumps({k: (float(f'{v:.4g}') if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        del m; torch.cuda.empty_cache()
import pandas as pd
df = pd.DataFrame(rows); df.to_csv(W / 'new_swin/coupling_diag/step_scale_raw.csv', index=False)
pd.set_option('display.width', 250)
print(df.drop(columns='seed').groupby(['model', 'loss', 'forget']).mean().to_string(float_format=lambda v: f'{v:.4g}'))
