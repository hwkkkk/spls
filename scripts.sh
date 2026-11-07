#!/bin/bash
# Example commands (seed 2027, forget 1 class). Run from the repository root.
# Datasets are expected under ./data (CIFAR-10/100 auto-format, Tiny-ImageNet in ./data/tiny-imagenet-200).
# SPLS = --method ldmu --soft_label lowest (lambda = 0, target logit = z_min).

SEED=2027

# 1) Original and retrain models  (saved to ./exps/pretrained_model)
python main.py --method pass --model_name resnet18 --dataset_name cifar10 --train_from_scratch --retrain_from_scratch --seed $SEED

# 2) SPLS unlearning
python main.py --method ldmu --soft_label lowest --model_name resnet18 --dataset_name cifar10 --unlearn_rate 3e-3 --forget_class 1 --seed $SEED
python main.py --method ldmu --soft_label lowest --model_name resnet18 --dataset_name cifar100 --unlearn_rate 3e-3 --forget_class 1 --seed $SEED
python main.py --method ldmu --soft_label lowest --model_name swin-t --dataset_name cifar10 --unlearn_rate 1e-3 --forget_class 1 --seed $SEED
python main.py --method ldmu --soft_label lowest --model_name swin-t --dataset_name cifar100 --unlearn_rate 1e-3 --forget_class 1 --seed $SEED

# 3) Baselines (example: DELETE)
python main.py --method delete --model_name resnet18 --dataset_name cifar10 --forget_class 1 --seed $SEED

# 4) Gap / confidence-bin analysis (inference only)
python spls_gap_confidence.py --arch resnet18 \
  --original exps/pretrained_model/${SEED}_cifar10_resnet18_original_model.pth \
  --unlearned model_checkpoint/<SPLS checkpoint>.pth
