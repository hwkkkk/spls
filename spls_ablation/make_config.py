"""Write ablation_config.json: every setting the ablation runs use, read from the actual config files."""
import hashlib
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spls_ablation.collect import BASELINE_CSV, BASELINE_CSV_RUNS, BASELINE_LOGS, LOSS_FORGET_COUNTS, LOSS_MODELS, SEED, SETTINGS  # noqa: E402
from spls_ablation.run_variant import PRETRAINED_DIR, UNLEARN_RATE  # noqa: E402
from spls_ablation.targets import ABLATION_VARIANTS, LOSS_VARIANTS, VARIANTS  # noqa: E402


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(out_path=PROJECT_ROOT / "spls_ablation/ablation_config.json", gpus=(1, 4)):
    configs = []
    for ds, model, fc in SETTINGS:
        yaml = OmegaConf.load(PROJECT_ROOT / f"config/{ds}_{model}.yaml")
        ckpt = PRETRAINED_DIR / f"{SEED}_{ds}_{model}_original_model.pth"
        configs.append({
            "dataset": ds, "model": model, "forget_class_count": fc,
            "batch_size": yaml.batch_size, "unlearn_epoch": yaml.unlearn_epoch,
            "config_file": f"config/{ds}_{model}.yaml",
            "original_checkpoint": str(ckpt), "original_checkpoint_md5": md5(ckpt),
            "baseline_source": str(BASELINE_LOGS.get((ds, model, fc)) or f"{BASELINE_CSV}:{BASELINE_CSV_RUNS[(ds, model, fc)]}"),
        })
    config = {
        "seed": SEED,
        "variants": {v: {"kind": VARIANTS[v][0], "lambda": VARIANTS[v][1]} for v in ABLATION_VARIANTS},
        "baseline": "existing SPLS (ldmu) seed-2027 runs, target = z_min (lambda=0); not re-run",
        "std": "torch.std(dim=1, unbiased=False) over all K original logits",
        "non_target_logits": "original model logits, unchanged",
        "unlearning": {
            "method": "ldmu (SPLS) loop, target construction swapped only",
            "loss": "nn.MSELoss(reduction='mean') between unlearn-model logits and target logits",
            "optimizer": "SGD(momentum=0.9)", "unlearn_rate": float(UNLEARN_RATE), "lr_scheduler": None,
            "data": "train forget set only", "original_model": "frozen copy (eval mode) generates targets",
        },
        "forget_class_selection": "main.py: --forget_class is the number of forget classes; random.shuffle(config.permutation_map) "
                              "after seed_torch(seed), first forget_class entries",
        "evaluation": {
            "per_epoch": "train/test forget/retain accuracy (method.utils.evaluate_model_on_all_loaders)",
            "mia": "evaluation.SVC_MIA confidence attack (main.py protocol): shadow = retain-train subsample vs retain-test, "
                   "target = train forget set; reported value = fraction of forget samples predicted 'member'",
        },
        "loss_ablation": {
            "settings": [{"dataset": "cifar100", "model": m, "forget_class_count": fc}
                         for m in LOSS_MODELS for fc in LOSS_FORGET_COUNTS],
            "variants": {
                "kl_softmax_zmin": "KLDiv(batchmean)(log_softmax(student), softmax(SPLS z_min target logits))",
                "mse_masked_softmax": "MSE(mean)(softmax(student), softmax(original logits with forget logit = -1e10)), "
                                      "masking as in method/delete.py",
            },
            "reference": "existing SPLS logit MSE runs (same baseline rows as the target ablation)",
            "baseline_sources": {f"{m}_forget{fc}": f"{BASELINE_CSV}:{BASELINE_CSV_RUNS[('cifar100', m, fc)]}"
                                 for m in LOSS_MODELS for fc in LOSS_FORGET_COUNTS},
        },
        "nonfinite_policy": "non-finite target/loss/accuracy aborts the run -> status failed_nonfinite; no values patched",
        "wall_clock_time": "not recorded",
        "configs": configs,
        "gpus": list(gpus),
        "total_runs": len(ABLATION_VARIANTS) * len(SETTINGS) + len(LOSS_VARIANTS) * len(LOSS_FORGET_COUNTS) * len(LOSS_MODELS),
    }
    Path(out_path).write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
