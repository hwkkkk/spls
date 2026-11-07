"""Evaluate Retrain (or Original, --kind original) checkpoints with the SAME metrics/protocol as main.py unlearning runs.

Metrics (%): train_forget, train_retain, test_forget, test_retain, mia
  - forget classes: identical RNG sequence to main.py (seed_torch -> dataset/loader -> shuffle(permutation_map))
  - MIA: SVC_MIA confidence, shadow_train = train_remain subsampled to len(test_remain) (as main.py)

usage: python eval_retrain.py --dataset_name cifar10 --model_name swin-t --seed 2027 \
           --ckpt_dir new_swin/pretrained_model --forget_classes 1 --output_csv out.csv
"""
import argparse
import csv
import random
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SubsetRandomSampler

import evaluation
from utils import *
from trainer import *


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", required=True, choices=["cifar10", "cifar100", "tiny_imagenet"])
    p.add_argument("--model_name", default="swin-t")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--ckpt_dir", required=True, help="dir with {seed}_{ds}_{model}_retrain_forget{k}_model.pth")
    p.add_argument("--forget_classes", type=int, nargs="+", default=[1])
    p.add_argument("--kind", default="retrain", choices=["retrain", "original"])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_csv", required=True)
    args = p.parse_args()

    cfgs = [Path(f"config/{args.dataset_name}_{args.model_name}.yaml")]
    if args.dataset_name == "tiny_imagenet" and args.model_name == "swin-t":
        cfgs.append(Path("config/tiny_imagenet_siwn-t.yaml"))
    cfg_path = next((c for c in cfgs if c.exists()), None)
    if cfg_path is None:
        raise FileNotFoundError(f"config not found: {cfgs}")
    config = OmegaConf.load(cfg_path)
    bs = config.batch_size

    # --- same RNG order as main.py ---
    seed_torch(args.seed)
    transform_train, transform_test = get_transforms(args.dataset_name, args.model_name, wo_dataaug=False)
    trainset, testset = get_dataset(args.dataset_name, transform_train, transform_test)
    train_loader, test_loader = get_dataloader(trainset, testset, bs, args.num_workers)
    num_classes = int(max(train_loader.dataset.targets)) + 1
    permutation_map = list(config.permutation_map)
    random.shuffle(permutation_map)

    rows = []
    for k in sorted(set(args.forget_classes)):
        forget_idx = permutation_map[:k]
        print(f"\n### seed={args.seed} {args.dataset_name} forget={k} classes={forget_idx}", flush=True)
        (tr_f, tr_r, te_f, te_r, _, _, tr_r_index, _, te_r_index) = get_unlearn_loader(
            trainset, testset, forget_idx, bs, float("inf"), args.num_workers)

        tag = f"retrain_forget{k}" if args.kind == "retrain" else "original"
        ckpt = Path(args.ckpt_dir) / f"{args.seed}_{args.dataset_name}_{args.model_name}_{tag}_model.pth"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        model = load_model(str(ckpt), args.model_name, num_classes)
        model.eval()

        res = {}
        for name, loader in [("train_forget", tr_f), ("train_retain", tr_r), ("test_forget", te_f), ("test_retain", te_r)]:
            _, acc = test(model, loader)
            res[name] = float(acc) * 100
            print(f"{name:<13}: {res[name]:.2f}", flush=True)

        # MIA exactly as main.py: shadow_train = random subset of train_remain with len(test_remain)
        idx = list(tr_r_index)
        random.shuffle(idx)
        idx = idx[:len(te_r_index)]
        shadow_train = DataLoader(tr_r.dataset, batch_size=bs, sampler=SubsetRandomSampler(idx))
        mia = evaluation.SVC_MIA(shadow_train=shadow_train, shadow_test=te_r,
                                 target_train=tr_f, target_test=None, model=model)
        res["mia"] = float(mia["confidence"]["avg"]) * 100
        print(f"mia          : {res['mia']:.2f}  {mia}", flush=True)

        rows.append({"kind": args.kind, "seed": args.seed, "dataset": args.dataset_name, "model": args.model_name,
                     "forget_class": k, "forget_idx": " ".join(map(str, forget_idx)), **res})
        del model
        torch.cuda.empty_cache()

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
