"""Run one SPLS target-ablation variant through the unmodified main.py pipeline.

main.py is executed as-is (same seeding, forget-class selection, loaders, config yaml values,
evaluation and SVC confidence MIA); only method.ldmu is swapped for the ablation loop.
Outputs go under spls_ablation/ so no existing log, result or checkpoint is touched:
  runs/<seed>_<dataset>_<model>_forget1/ldmu/<variant>_<time>/   train_log.log, config.yaml, plot
  checkpoints/                                                   unlearned models
  ablation_logs/<run>.status.json                                ok / failed_nonfinite / failed

usage: CUDA_VISIBLE_DEVICES=1 python spls_ablation/run_variant.py --dataset cifar10 --model resnet18 --variant min_lam3 [--forget_class 5]
"""
import argparse
import json
import runpy
import sys
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ABLATION_DIR = PROJECT_ROOT / "spls_ablation"
PRETRAINED_DIR = PROJECT_ROOT / "exps/pretrained_model"

# identical to the existing seed-2027 SPLS runs (logs/ldmu_all_lr_3e_3, revision/*.pth)
SEED = 2027
FORGET_CLASS = 1
UNLEARN_RATE = "3e-3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    parser.add_argument("--model", required=True, choices=["resnet18", "swin-t"])
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--original_checkpoint", type=Path)
    parser.add_argument("--forget_class", type=int, default=FORGET_CLASS, help="number of forget classes")
    parser.add_argument("--runs_dir", default=str(ABLATION_DIR / "runs"))
    parser.add_argument("--ckpt_dir", default=str(ABLATION_DIR / "checkpoints"))
    parser.add_argument("--status_dir", default=str(ABLATION_DIR / "ablation_logs"))
    parser.add_argument("--unlearn_rate", default=UNLEARN_RATE)  # 2026-11-03: new Swin-T uses 1e-3
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))
    import os
    os.chdir(PROJECT_ROOT)  # main.py resolves config/ and data paths relative to the project root

    import torch
    import method
    from spls_ablation.targets import LOSS_VARIANTS, VARIANTS
    from spls_ablation import ablation_ldmu

    if args.variant not in VARIANTS and args.variant not in LOSS_VARIANTS:
        raise SystemExit(f"unknown variant {args.variant}; choose from {list(VARIANTS) + LOSS_VARIANTS}")

    run_name = f"{args.seed}_{args.dataset}_{args.model}_forget{args.forget_class}_{args.variant}"
    status_path = Path(args.status_dir) / f"{run_name}.status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # swap only the ldmu target construction
    method.ldmu = ablation_ldmu.make_ablation_ldmu(args.variant)

    # main.py saves to ./model_checkpoint/<name without variant>; redirect into the ablation folder
    original_save = torch.save
    saved = {}

    def redirected_save(obj, f, *a, **kw):
        p = Path(f)
        if not p.is_absolute() and p.parent == Path("model_checkpoint"):
            f = ckpt_dir / f"{run_name}_{datetime.now().strftime('%m%d-%H%M%S')}.pth"
            saved["checkpoint"] = str(f)
        return original_save(obj, f, *a, **kw)

    torch.save = redirected_save

    original_ckpt = args.original_checkpoint or PRETRAINED_DIR / f"{args.seed}_{args.dataset}_{args.model}_original_model.pth"
    if not original_ckpt.is_file():
        raise FileNotFoundError(original_ckpt)
    sys.argv = [
        "main.py",
        "--method", "ldmu",
        "--dataset_name", args.dataset,
        "--model_name", args.model,
        "--forget_class", str(args.forget_class),
        "--seed", str(args.seed),
        "--unlearn_rate", str(args.unlearn_rate),
        "--description", args.variant,
        "--exps_dir", args.runs_dir,
        "--load_original_model_path", str(original_ckpt),
    ]

    status = {"run": run_name, "dataset": args.dataset, "model": args.model, "seed": args.seed,
              "forget_class": args.forget_class, "variant": args.variant, "original_checkpoint": str(original_ckpt),
              "started": datetime.now().isoformat(timespec="seconds")}
    exit_code = 0
    try:
        runpy.run_path(str(PROJECT_ROOT / "main.py"), run_name="__main__")
        status["status"] = "ok"
    except SystemExit as e:  # main.py ends with exit()
        if e.code in (None, 0):
            status["status"] = "ok"
        else:
            status["status"] = "failed"
            status["error"] = f"SystemExit({e.code})"
            exit_code = 1
    except ablation_ldmu.NonFiniteError as e:
        status["status"] = "failed_nonfinite"
        status["error"] = str(e)
        exit_code = 2
    except Exception as e:
        status["status"] = "failed"
        status["error"] = f"{type(e).__name__}: {e}"
        status["traceback"] = traceback.format_exc()
        exit_code = 1
    finally:
        torch.save = original_save

    status["finished"] = datetime.now().isoformat(timespec="seconds")
    status["experiment_path"] = str(ablation_ldmu.LAST_EXPERIMENT_PATH.get("path", ""))
    status["checkpoint"] = saved.get("checkpoint")
    status_path.write_text(json.dumps(status, indent=2))
    print(f"[ablation] {run_name}: {status['status']}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
