"""||θ_unlearned − θ_original|| (L2 over all params, and relative) for SPLS vs KL checkpoints, Swin-T CIFAR-100 forget 1/5/20."""
import glob, sys, torch, pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from pathlib import Path
S = Path(__file__).resolve().parents[1] / 'new_swin'
def sd(p):
    m = torch.load(p, map_location='cpu', weights_only=False)
    m = m if isinstance(m, dict) else m.state_dict()
    return {k.replace('module.', ''): v.float() for k, v in m.items() if v.dtype.is_floating_point}
rows = []
for seed in (2027, 2028, 2029):
    o = sd(S / f'pretrained_model/{seed}_cifar100_swin-t_original_model.pth')
    on = torch.sqrt(sum((v ** 2).sum() for v in o.values())).item()
    for fc in (1, 5, 20):
        for meth, pat in (('SPLS', S / f'coupling_diag/checkpoints/{seed}_cifar100_swin-t_forget{fc}_min_lam0_*.pth'),
                          ('KL', S / f'analysis_{seed}/loss_ablation/checkpoints/{seed}_cifar100_swin-t_forget{fc}_kl_softmax_zmin_*.pth')):
            u = sd(glob.glob(str(pat))[0])
            d = torch.sqrt(sum(((u[k] - o[k]) ** 2).sum() for k in o)).item()
            dh = torch.sqrt(sum(((u[k] - o[k]) ** 2).sum() for k in o if k.startswith('head') or k.startswith('fc'))).item()
            rows.append(dict(seed=seed, forget=fc, method=meth, dist=d, rel_dist=d / on, head_dist=dh))
df = pd.DataFrame(rows); df.to_csv(S / 'coupling_diag/param_dist_raw.csv', index=False)
print(df.drop(columns='seed').groupby(['forget', 'method']).agg(['mean', 'std']).round(4).to_string())

# --- head row decomposition: share of classifier-weight change in forget-class rows vs retain rows; backbone distance
import random
from omegaconf import OmegaConf
cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / 'config/cifar100_swin-t.yaml')
def order(seed):
    o = OmegaConf.create({'order': list(cfg.permutation_map)}).order
    random.Random(seed).shuffle(o); return [int(x) for x in o]
rows2 = []
for seed in (2027, 2028, 2029):
    o = sd(S / f'pretrained_model/{seed}_cifar100_swin-t_original_model.pth')
    hk = [k for k in o if (k.startswith('head') or k.startswith('fc')) and o[k].shape[0] == 100]
    for fc in (1, 5, 20):
        f = order(seed)[:fc]; r = [c for c in range(100) if c not in f]
        for meth, pat in (('SPLS', S / f'coupling_diag/checkpoints/{seed}_cifar100_swin-t_forget{fc}_min_lam0_*.pth'),
                          ('KL', S / f'analysis_{seed}/loss_ablation/checkpoints/{seed}_cifar100_swin-t_forget{fc}_kl_softmax_zmin_*.pth')):
            u = sd(glob.glob(str(pat))[0])
            df_ = sum(((u[k] - o[k])[f] ** 2).sum() for k in hk).item()
            dr = sum(((u[k] - o[k])[r] ** 2).sum() for k in hk).item()
            bb = sum(((u[k] - o[k]) ** 2).sum() for k in o if k not in hk).item()
            rows2.append(dict(seed=seed, forget=fc, method=meth, backbone_dist=bb ** .5,
                              head_retain_rows_dist=dr ** .5, head_forget_rows_dist=df_ ** .5,
                              head_retain_rows_per_row=(dr / len(r)) ** .5))
d2 = pd.DataFrame(rows2); d2.to_csv(S / 'coupling_diag/param_decomp_raw.csv', index=False)
print(d2.drop(columns='seed').groupby(['forget', 'method']).agg(['mean', 'std']).round(4).to_string())
