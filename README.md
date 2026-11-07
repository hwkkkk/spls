# SPLS: class unlearning by suppressing the forget-class logit

Anonymous code release for review.

## Environment
Python 3.12, PyTorch 2.x, torchvision, omegaconf, scikit-learn, matplotlib, pandas, tqdm.

## Layout
- `main.py`: entry point (original / retrain training, unlearning, evaluation incl. SVC-MIA)
- `method/ldmu.py`: SPLS (`--method ldmu --soft_label lowest`, i.e. lambda = 0)
- `method/`: baselines (DELETE, Boundary Shrink/Expand, Gradient Ascent, Random Label, Bad Teacher, SalUn, Fine-tune)
- `evaluation/`: membership inference evaluation
- `config/`: per dataset/architecture defaults (epochs, lr, batch size, class permutation)

## Additional experiments
| Experiment | Scripts |
|---|---|
| Sequential unlearning | `sequential.py`, `plot_sequential_summary.py`, `plot_sequential_3seed.py` |
| Target (λ) / loss ablation | `spls_ablation/`, `run_lambda_ablation.sh` |
| k ablation | `run_spls_k_ablation.sh`, `plot_spls_k_ablation.py` |
| LiRA / U-LiRA | `lira.py`, `u_lira.py`, `shadow_model.py` |
| RMIA | `run_rmia.py`, `rmia_common.py`, `train_reference.py`, `score_offline_rmia.py`, `evaluate_rmia.py` |
| Representation analysis (CKA, attention) | `cka.py`, `attention.py` |
| Gradient coupling / step-size diagnostics | `coupling_diag/` |
| Gap distribution, confidence bins | `spls_gap_confidence.py` |
| Per-class retain accuracy | `per_class_retain.py` |
| Retrain / original model evaluation | `eval_retrain.py` |

Analysis scripts read checkpoints from `./exps/pretrained_model`, `./model_checkpoint` and the
output folders of the runs above; run them from the repository root.

## Usage
See `scripts.sh`. Forget classes are drawn from `permutation_map` in the config, shuffled with `--seed`.
Per-method learning rates follow the paper's hyperparameter table (pass `--unlearn_rate`).
Unlearned models are saved to `./model_checkpoint`, logs to `./exps`.

## Acknowledgement
The training/evaluation framework and baseline implementations build on the public DELETE codebase
(Zhou et al., CVPR 2025, https://github.com/shaaaaron/DELETE).
