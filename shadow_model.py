#!/usr/bin/env python3
"""Retrain-only extraction from main.py/trainer.py for fixed-class LiRA shadows."""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.datasets import CIFAR10, CIFAR100

from models import get_model
from trainer import optimizer_picker, train, test
from utils import get_transforms, get_unlearn_loader, get_dataloader, seed_torch, note_print
from linear_probe import ROOT, save_json

SEEDS = (3001, 3002, 3003, 3004)


def atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def train_shadow(seed, output_dir, num_workers=4, dataset="cifar10", mode="retrain"):
    assert seed != 2027
    assert mode in ("retrain", "original")
    classes, forget = {"cifar10": (10, 6), "cifar100": (100, 73)}[dataset]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / f"shadow_seed{seed}_best.pth"
    last_path = output_dir / f"shadow_seed{seed}_last.pt"
    status_path = output_dir / f"shadow_seed{seed}.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        assert status.get("dataset", "cifar10") == dataset
        assert status.get("mode", "retrain") == mode
        if status.get("status") == "completed":
            assert best_path.is_file()
            print(f"[shadow cached] seed={seed}", flush=True)
            return best_path
    seed_torch(seed)
    config = OmegaConf.load(ROOT / f"config/{dataset}_resnet18.yaml")
    permutation_map = list(config.permutation_map)
    random.shuffle(permutation_map)
    permutation_map.insert(0, permutation_map.pop(permutation_map.index(forget)))
    forget_class_index = permutation_map[:1]
    assert forget_class_index == [forget]
    note_print(f"forget class index: {forget_class_index}")
    transform_train, transform_test = get_transforms(dataset, "resnet18", wo_dataaug=False)
    factory = CIFAR10 if dataset == "cifar10" else CIFAR100
    trainset = factory(ROOT / "data", train=True, download=False, transform=transform_train)
    testset = factory(ROOT / "data", train=False, download=False, transform=transform_test)
    loaders = get_unlearn_loader(trainset, testset, forget_class_index, 128, float("inf"), num_workers)
    train_loader, test_loader = loaders[1], loaders[3]
    train_count, test_count = 50000 - 50000 // classes, 10000 - 10000 // classes
    if mode == "original":
        train_loader, test_loader = get_dataloader(trainset, testset, 128, num_workers)
        train_count, test_count = 50000, 10000
    assert len(train_loader.sampler) == train_count and len(test_loader.sampler) == test_count
    if mode == "retrain":
        assert all(trainset.targets[i] != forget for i in train_loader.sampler.indices)
    model = get_model("resnet18", classes).to("cuda")
    optimizer = optimizer_picker("sgd", model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    first_epoch, best_acc, best_epoch = 0, 0.0, 0
    history = []
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        assert checkpoint["seed"] == seed and checkpoint["forget_class"] == forget
        assert checkpoint.get("dataset", "cifar10") == dataset
        assert checkpoint.get("mode", "retrain") == mode
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        first_epoch, best_acc = checkpoint["epoch"], checkpoint["best_acc"]
        best_epoch, history = checkpoint["best_epoch"], checkpoint["history"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        print(f"[shadow resume] seed={seed} next_epoch={first_epoch + 1}", flush=True)
    config_record = {"seed": seed, "dataset": dataset, "mode": mode, "forget_class": forget, "permutation_map": permutation_map,
                     "epochs": 150, "batch_size": 128, "lr": 0.1, "momentum": 0.9,
                     "weight_decay": 0.0005, "scheduler": "StepLR(step_size=40,gamma=0.1)",
                     "train_count": train_count, "retain_test_count": test_count,
                     "train_transform": str(transform_train), "test_transform": str(transform_test),
                     "selection": "best retain test accuracy; last tie, matching trainer.py",
                     "train_evaluation": "augmented retain train set, matching trainer.py",
                     "best_checkpoint": str(best_path.resolve())}
    start = time.monotonic()
    for epoch in range(first_epoch, 150):
        begin = time.monotonic()
        print(f"[shadow train] seed={seed} epoch={epoch + 1}/150 lr={optimizer.param_groups[0]['lr']}", flush=True)
        train(model, train_loader, optimizer, epoch, tqdm_on=False)
        train_loss, train_acc = test(model, train_loader)
        test_loss, test_acc = test(model, test_loader)
        if test_acc >= best_acc:
            best_acc, best_epoch = test_acc, epoch + 1
            atomic_save(model.state_dict(), best_path)
        scheduler.step()
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
                        "retain_test_loss": test_loss, "retain_test_acc": test_acc,
                        "seconds": time.monotonic() - begin})
        atomic_save({"seed": seed, "dataset": dataset, "mode": mode, "forget_class": forget, "epoch": epoch + 1,
                     "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "best_acc": best_acc,
                     "best_epoch": best_epoch, "history": history,
                     "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                     "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, last_path)
        status = dict(config_record, status="training", epoch=epoch + 1, best_acc=best_acc,
                      best_epoch=best_epoch, history=history)
        save_json(status_path, status)
        print(f"[shadow epoch] seed={seed} epoch={epoch + 1} train={train_acc:.4%} "
              f"retain_test={test_acc:.4%} best={best_acc:.4%} seconds={history[-1]['seconds']:.1f}", flush=True)
    status = dict(config_record, status="completed", epoch=150, best_acc=best_acc,
                  best_epoch=best_epoch, history=history, session_seconds=time.monotonic() - start)
    save_json(status_path, status)
    del model, optimizer, scheduler, loaders
    torch.cuda.empty_cache()
    return best_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--output-dir")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    output_dir = args.output_dir or ("lira_results_seed2027/shadows" if args.dataset == "cifar10"
                                      else "lira_results_cifar100_seed2027/shadows")
    train_shadow(args.seed, output_dir, args.num_workers, args.dataset)


if __name__ == "__main__":
    main()
