#!/bin/bash
# SPLS target (λ) ablation, 3 seeds, z_min - λσ with λ ∈ {0,1,3,5,7,10,20,30,50,100} + second_min, mean_minus_std (spls_ablation/targets.py)
#  - ResNet-18 seeds 2028/2029 (resume of spls_ablation/run_multiseed.py --models resnet18, killed 10-31 at 22/142):
#      default spls_ablation/{ablation_logs,runs,checkpoints}, lr 3e-3; also loss variants. Finished (status ok) runs skipped.
#  - Swin-T NEW ckpts (new_swin/pretrained_model) seeds 2027-2029, lr 1e-3, outputs new_swin/lambda_ablation/.
#      cifar100 λ=1,3,5,7,10 are NOT re-run: taken from new_swin/analysis_<seed>/k_ablation (main.py lowest-std --k, same target, unbiased std).
#      loss variants already in new_swin/analysis_<seed>/loss_ablation.
# GPUs 1 and 4 share one queue (mkdir locks). usage: nohup bash run_lambda_ablation.sh > new_swin/lambda_ablation_launcher.log 2>&1 &
cd "$(dirname "$0")" || exit 1
PY=${PYTHON_BIN:-python}
GPUS="1 4"
OUT=new_swin/lambda_ablation
LOCK=$OUT/locks; mkdir -p $LOCK $OUT/status $OUT/logs
LAMS="min_lam1 min_lam3 min_lam5 min_lam10 min_lam20 min_lam30 min_lam50 min_lam100"   # λ=0 = main SPLS results
SETTINGS="cifar100:1 cifar100:5 cifar100:10 cifar100:20"   # cifar10 dropped (user, 11-04)

JOBS=()   # name|status_file|command
# Swin first (longer)
for SEED in 2027 2028 2029; do
  for st in $SETTINGS; do ds=${st%:*}; fc=${st#*:}
    for v in $LAMS; do
      [[ $ds == cifar100 && $v =~ ^min_lam(1|3|5|7|10)$ ]] && continue   # salvaged from k_ablation
      n=${SEED}_${ds}_swin-t_forget${fc}_${v}
      JOBS+=("$n|$OUT/status/$n.status.json|$PY spls_ablation/run_variant.py --dataset $ds --model swin-t --variant $v --forget_class $fc --seed $SEED \
--unlearn_rate 1e-3 --original_checkpoint new_swin/pretrained_model/${SEED}_${ds}_swin-t_original_model.pth \
--runs_dir $OUT/runs --ckpt_dir $OUT/checkpoints --status_dir $OUT/status")
    done
  done
done
# ResNet-18 seeds 2028/2029: same queue as run_multiseed.py
for SEED in 2028 2029; do
  for st in $SETTINGS; do ds=${st%:*}; fc=${st#*:}
    for v in $LAMS; do
      n=${SEED}_${ds}_resnet18_forget${fc}_${v}
      JOBS+=("$n|spls_ablation/ablation_logs/$n.status.json|$PY spls_ablation/run_variant.py --dataset $ds --model resnet18 --variant $v --forget_class $fc --seed $SEED \
--original_checkpoint revision/${SEED}_${ds}_resnet18_original_model.pth")
    done
  done
  for fc in 1 2 5 10 20; do
    vs="kl_softmax_zmin mse_masked_softmax"; [ $fc = 2 ] && vs="$vs"
    for v in $vs; do
      n=${SEED}_cifar100_resnet18_forget${fc}_${v}
      JOBS+=("$n|spls_ablation/ablation_logs/$n.status.json|$PY spls_ablation/run_variant.py --dataset cifar100 --model resnet18 --variant $v --forget_class $fc --seed $SEED \
--original_checkpoint revision/${SEED}_cifar100_resnet18_original_model.pth")
    done
  done
done

ok() { [ -f "$1" ] && grep -q '"status": "ok"' "$1"; }
worker() {
  gpu=$1
  for job in "${JOBS[@]}"; do
    IFS='|' read -r name st cmd <<< "$job"
    ok "$st" && continue
    mkdir "$LOCK/$name" 2>/dev/null || continue
    echo "[$(date '+%F %T')] GPU$gpu start $name"
    CUDA_VISIBLE_DEVICES=$gpu MPLBACKEND=Agg TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=4 $cmd > "$OUT/logs/$name.out" 2>&1
    rc=$?; ok "$st" || rc="${rc}(status not ok)"
    echo "[$(date '+%F %T')] GPU$gpu done  $name (exit $rc)"
  done
}
n_todo=0; for job in "${JOBS[@]}"; do IFS='|' read -r name st cmd <<< "$job"; ok "$st" || n_todo=$((n_todo+1)); done
echo "[$(date '+%F %T')] ${#JOBS[@]} jobs, $n_todo to run, GPUs $GPUS"
for g in $GPUS; do worker $g & sleep 5; done
wait
echo "[$(date '+%F %T')] all done"
