# SPLS: Structure-Preserving Logit Suppression for Machine Unlearning

This repository contains the implementation of **SPLS (Structure-Preserving Logit Suppression)**, a retain-data-free class unlearning method.

> **Note:** In the source code, some files, classes, functions, or experiment outputs may still use the name **LDMU**.  
> **LDMU is the previous implementation name of SPLS**, and both refer to the same proposed method in this repository.

## Overview

SPLS performs class unlearning without using retained training data.

The main idea is to construct a logit-level target for each forget sample by:

- suppressing the logit corresponding to the target class, and
- preserving the original logits of the non-target classes.

The unlearned model is then optimized to match this constructed target.

## Repository Structure

The exact directory structure may vary depending on the experiment configuration, but the repository generally contains:

```text
.
├── models/             # Model architectures
├── ...                 # Training / unlearning / evaluation code
└── README.md
```

Some implementation components were developed on top of the model code from the DELETE repository described below.

## Code Base / Acknowledgement

This project was developed by cloning and modifying parts of the following repository:

**DELETE: Machine Unlearning without Retain Data**

https://github.com/shaaaaron/DELETE/tree/main/models

In particular, the model implementations under the `models` directory were used as a starting point and modified as needed for our experiments.

We thank the authors of DELETE for making their implementation publicly available.

## Method Name

For clarity:

```text
LDMU  ->  SPLS
```

`LDMU` was the earlier name used during implementation.  
The final method name used in the paper is **SPLS (Structure-Preserving Logit Suppression)**.

Therefore, commands, filenames, checkpoints, logs, or source-code identifiers containing `ldmu` refer to SPLS unless otherwise stated.

## Datasets

Experiments in the SPLS paper include:

- CIFAR-10
- CIFAR-100
- Tiny-ImageNet

## Models

The evaluated architectures include:

- ResNet-18
- VGG-16
- ViT-S/16
- Swin-T

## Evaluation

The experiments evaluate both forgetting effectiveness and retained utility using metrics such as:

- Forget-set accuracy
- Retain-set accuracy
- Test forget accuracy
- Test retain accuracy
- Membership Inference Attack (MIA)

## Installation

Create a Python environment and install the dependencies required by the repository.

For example:

```bash
conda create -n spls python=3.10
conda activate spls
```

Then install the packages required by the experiment code.

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not provided, install the corresponding PyTorch, torchvision, and other dependencies according to your environment and CUDA version.

## Usage

Experiment entry points and arguments depend on the scripts included in the repository.

A typical workflow is:

```text
1. Prepare the dataset.
2. Train or load the original model.
3. Select the class or classes to forget.
4. Run the SPLS/LDMU unlearning procedure.
5. Evaluate forget and retain performance.
```

When using the current codebase, remember that an experiment option or implementation named `LDMU` corresponds to **SPLS**.

## Reproducibility

The experiments reported in the paper were conducted over multiple random seeds.  
For exact reproduction, use the hyperparameters, optimizer settings, datasets, architectures, and random seeds described in the paper and experiment configuration files.

## Citation

If you use this repository in your research, please cite the SPLS paper.

```bibtex
@article{spls,
  title   = {Structure-Preserving Logit Suppression for Machine Unlearning},
  author  = {Hyeon-Uk Kang and Jong-Ryul Lee},
  journal = {Knowledge-Based Systems},
  year    = {2026}
}
```

> The citation information above can be updated after the final bibliographic information (volume, pages, DOI, etc.) becomes available.

## License

Please check the licenses of this repository and all upstream repositories before redistribution or commercial use.

Parts of the codebase are derived from or based on the DELETE repository, so the corresponding upstream license and attribution requirements should also be followed.
