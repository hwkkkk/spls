#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
EXPS_DIR="${EXPS_DIR:-${PROJECT_ROOT}/exps}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_PY="${MAIN_PY:-${PROJECT_ROOT}/main.py}"

LOG_DIR="${PROJECT_ROOT}/spls_k_ablation_real"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}" || exit 1

# ============================================================
# Prevent duplicate execution of this script
# ============================================================

LOCK_FILE="/tmp/spls_k_ablation.lock"

exec 9>"${LOCK_FILE}"

if ! flock -n 9; then
    echo "[ERROR] SPLS k-ablation script is already running."
    exit 1
fi

# ============================================================
# Settings
# ============================================================

SEED=2027

K_VALUES=(
    1.0
    2.0
    3.0
    4.0
    5.0
    6.0
    7.0
    8.0
    9.0
    10.0
)

# ============================================================
# One experiment at a time
# ============================================================

run_one() {
    GPU="$1"
    DATASET="$2"
    MODEL="$3"
    FORGET="$4"
    K="$5"

    K_TAG="${K_real}"

    echo "============================================================"
    echo "[START]"
    echo "GPU     = ${GPU}"
    echo "Dataset = ${DATASET}"
    echo "Model   = ${MODEL}"
    echo "Forget  = ${FORGET}"
    echo "k       = ${K}"
    echo "Seed    = ${SEED}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PYTHON_BIN}" "${MAIN_PY}" \
        --method ldmu \
        --dataset_name "${DATASET}" \
        --model_name "${MODEL}" \
        --forget_class "${FORGET}" \
        --soft_label lowest-std \
        --k "${K}" \
        --seed "${SEED}" \
        --unlearn_rate 3e-3 \
        --description "spls_k_ablation_k${K_TAG}" \
        --exps_dir "${EXPS_DIR}"

    STATUS=$?

    if [ "${STATUS}" -ne 0 ]; then
        echo "[ERROR] ${DATASET} ${MODEL} forget=${FORGET} k=${K}"
        echo "exit status = ${STATUS}"
    else
        echo "[DONE] ${DATASET} ${MODEL} forget=${FORGET} k=${K}"
    fi

    echo

    return "${STATUS}"
}

# ============================================================
# GPU 3
# ResNet18
#
# IMPORTANT:
# No "&" anywhere inside this function.
# Therefore only one main.py process exists on GPU 3.
# ============================================================

run_resnet18() {
    GPU=3
    DATASET="cifar100"
    MODEL="resnet18"

    for FORGET in 1 2 5 10 20; do
        for K in "${K_VALUES[@]}"; do
            run_one \
                "${GPU}" \
                "${DATASET}" \
                "${MODEL}" \
                "${FORGET}" \
                "${K}"
        done
    done
}

# ============================================================
# GPU 5
# Swin-T
#
# Again: sequential inside GPU 5.
# ============================================================

run_swint() {
    GPU=5
    DATASET="cifar100"
    MODEL="swin-t"

    for FORGET in 1 2 5 10 20; do
        for K in "${K_VALUES[@]}"; do
            run_one \
                "${GPU}" \
                "${DATASET}" \
                "${MODEL}" \
                "${FORGET}" \
                "${K}"
        done
    done
}

# ============================================================
# Launch
#
# Exactly TWO background jobs:
#   job 1 -> GPU 3
#   job 2 -> GPU 5
#
# main.py itself is NEVER backgrounded.
# ============================================================

run_resnet18 \
    > "${LOG_DIR}/gpu3_cifar100_resnet18_k_sweep.log" \
    2>&1 &
PID3=$!

run_swint \
    > "${LOG_DIR}/gpu5_cifar100_swint_k_sweep.log" \
    2>&1 &
PID5=$!

echo "============================================================"
echo "SPLS k-ablation launched"
echo
echo "GPU 3 : ResNet18 sequential worker"
echo "PID   : ${PID3}"
echo
echo "GPU 5 : Swin-T sequential worker"
echo "PID   : ${PID5}"
echo "============================================================"

wait "${PID3}"
STATUS3=$?

wait "${PID5}"
STATUS5=$?

echo
echo "============================================================"
echo "[ALL DONE]"
echo "GPU3 / ResNet18 status = ${STATUS3}"
echo "GPU5 / Swin-T   status = ${STATUS5}"
echo "============================================================"