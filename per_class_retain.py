import sys, glob, random, torch, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from utils import get_transforms, get_dataset
from models import load_model
R = Path(__file__).resolve().parent
CASES = [("cifar100", 2031), ("cifar100", 2030), ("cifar100", 2027), ("tiny_imagenet", 2031), ("tiny_imagenet", 2027), ("tiny_imagenet", 2028)]
cache = {}
def load(p):
    o = torch.load(p, map_location="cpu", weights_only=False)
    m = o if isinstance(o, torch.nn.Module) else load_model(p, "swin-t", 100 if "cifar100" in str(p) else 200)
    return m.eval().cuda()
@torch.inference_mode()
def preds(m, ds):
    return torch.cat([m(x.cuda()).argmax(1).cpu() for x, _ in DataLoader(ds, batch_size=256, num_workers=8)]).numpy()
for dsn, s in CASES:
    cfg = "config/cifar100_swin-t.yaml" if dsn == "cifar100" else "config/tiny_imagenet_siwn-t.yaml"
    perm = list(OmegaConf.load(R / cfg).permutation_map); random.Random(s).shuffle(perm); u = perm[0]
    if dsn not in cache:
        _, tf = get_transforms(dsn, "swin-t", wo_dataaug=True)
        _, test = get_dataset(dsn, tf, tf, path=R / "data")
        cache[dsn] = test
    test = cache[dsn]; y = np.asarray(test.targets); C = y.max() + 1
    ori = R / f"new_swin/pretrained_model/{s}_{dsn}_swin-t_original_model.pth"
    dl = sorted(glob.glob(str(R / f"model_checkpoint/{s}_{dsn}_swin-t_forget1_delete_11*.pth")))[-1]
    sp = sorted(glob.glob(str(R / f"model_checkpoint/{s}_{dsn}_swin-t_forget1_ldmu_11*.pth")))[-1]
    P = {k: preds(load(p), test) for k, p in [("ori", ori), ("del", dl), ("spls", sp)]}
    acc = {k: np.array([(v[y == c] == c).mean() for c in range(C)]) * 100 for k, v in P.items()}
    keep = np.arange(C) != u
    # ori confusion: which classes absorb forget-class test samples / where DELETE sends them
    f = y == u
    top = lambda a: [(int(c), int(n)) for c, n in zip(*np.unique(a, return_counts=True))]
    print(f"\n== {dsn} {s} u={u}  retain mean ori {acc['ori'][keep].mean():.2f} del {acc['del'][keep].mean():.2f} spls {acc['spls'][keep].mean():.2f}")
    print("forget test preds DELETE (cls,count) top5:", sorted(top(P['del'][f]), key=lambda t: -t[1])[:5])
    print("forget test preds SPLS   top5:", sorted(top(P['spls'][f]), key=lambda t: -t[1])[:5])
    # retain samples that DELETE newly predicts as which class
    newerr = keep[y] & (P['ori'] == y) & (P['del'] != y)
    print(f"newly wrong retain samples: DELETE {newerr.sum()}  SPLS {(keep[y] & (P['ori']==y) & (P['spls']!=y)).sum()}; DELETE new-wrong pred class top5:", sorted(top(P['del'][newerr]), key=lambda t: -t[1])[:5])
    d = acc['del'] - acc['ori']; idx = np.argsort(d)[:6]
    print("largest per-class drops DELETE (cls, ori->del, spls):", [(int(c), round(acc['ori'][c]), round(acc['del'][c]), round(acc['spls'][c])) for c in idx if c != u])
