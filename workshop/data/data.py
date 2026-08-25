import os
import urllib.request
import zipfile
from pathlib import Path

from torchvision.datasets import CIFAR10, CIFAR100

DATA_ROOT = Path("DELETE/data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

print(f"[INFO] DATA_ROOT = {DATA_ROOT}")

# =========================
# CIFAR-10
# =========================
print("[INFO] Downloading CIFAR-10...")
CIFAR10(
    root=str(DATA_ROOT),
    train=True,
    download=True,
)

CIFAR10(
    root=str(DATA_ROOT),
    train=False,
    download=True,
)

# =========================
# CIFAR-100
# =========================
print("[INFO] Downloading CIFAR-100...")
CIFAR100(
    root=str(DATA_ROOT),
    train=True,
    download=True,
)

CIFAR100(
    root=str(DATA_ROOT),
    train=False,
    download=True,
)

# =========================
# Tiny ImageNet
# =========================
tiny_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
tiny_zip = DATA_ROOT / "tiny-imagenet-200.zip"
tiny_dir = DATA_ROOT / "tiny-imagenet-200"

if tiny_dir.exists():
    print(f"[INFO] Tiny ImageNet already exists: {tiny_dir}")
else:
    if not tiny_zip.exists():
        print("[INFO] Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(tiny_url, tiny_zip)
        print(f"[INFO] Downloaded: {tiny_zip}")
    else:
        print(f"[INFO] Tiny ImageNet zip already exists: {tiny_zip}")

    print("[INFO] Extracting Tiny ImageNet...")
    with zipfile.ZipFile(tiny_zip, "r") as zf:
        zf.extractall(DATA_ROOT)

# =========================
# Rearrange Tiny ImageNet val set for ImageFolder
# =========================
val_dir = tiny_dir / "val"
val_img_dir = val_dir / "images"
val_anno = val_dir / "val_annotations.txt"
val_by_class = val_dir / "by_class"

if val_img_dir.exists() and val_anno.exists():
    val_by_class.mkdir(exist_ok=True)

    print("[INFO] Rearranging Tiny ImageNet val images by class...")

    with open(val_anno, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            img_name = parts[0]
            cls = parts[1]

            src = val_img_dir / img_name
            dst_dir = val_by_class / cls
            dst_dir.mkdir(exist_ok=True)
            dst = dst_dir / img_name

            if src.exists() and not dst.exists():
                # copy 대신 hardlink 시도. 실패하면 일반 copy.
                try:
                    os.link(src, dst)
                except OSError:
                    import shutil
                    shutil.copy2(src, dst)

    print(f"[INFO] Tiny ImageNet val ImageFolder path: {val_by_class}")

print("[DONE] Dataset download complete.")
print()
print("[PATHS]")
print(f"CIFAR-10:      {DATA_ROOT / 'cifar-10-batches-py'}")
print(f"CIFAR-100:     {DATA_ROOT / 'cifar-100-python'}")
print(f"TinyImageNet:  {DATA_ROOT / 'tiny-imagenet-200'}")
print(f"Tiny val IF:   {DATA_ROOT / 'tiny-imagenet-200/val/by_class'}")