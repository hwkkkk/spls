"""Shared paths and helpers for the RMIA scripts."""
import json
from pathlib import Path

import torch
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parent
PRETRAIN = ROOT / "exps/pretrained_model"
REVISION = ROOT / "revision"


def save_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_model(path, classes, device):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        model = obj
    else:
        state = obj.get("state_dict", obj)
        model = models.resnet18(num_classes=classes)
        model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    model.requires_grad_(False)
    return model.eval().to(device)


def atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)
