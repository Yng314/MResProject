#!/bin/bash
#SBATCH --job-name=mbv3-noes-clean
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mbv3_noes_clean_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mbv3_noes_clean_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00

set -euo pipefail

SEED="${1:?seed must be provided}"
OUTPUT_ROOT="${2:?output_root must be provided}"
LOOP_COUNT="${3:-5}"
TOP_FRACTION="${4:-0.20}"

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_DIR}/NoiseRate/cxr_real_experiment"
SEED_ROOT="${OUTPUT_ROOT}/seed_${SEED}"
BASELINE_DIR="${SEED_ROOT}/baseline_no_clean"
REMOVE_ROOT="${SEED_ROOT}/sample20_remove_loop"

mkdir -p "${SEED_ROOT}" "${PROJECT_DIR}/slurm_logs"

echo "=========================================================="
echo "MobileNetV3 noES clean seed experiment"
echo "=========================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Seed: ${SEED}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Seed root: ${SEED_ROOT}"
echo "Baseline dir: ${BASELINE_DIR}"
echo "Remove root: ${REMOVE_ROOT}"
echo "Loop count: ${LOOP_COUNT}"
echo "Top fraction: ${TOP_FRACTION}"
echo "Started: $(date)"
echo "=========================================================="

cd "${PROJECT_DIR}" || exit 1

if [[ -s "${BASELINE_DIR}/baseline_run_summary.csv" ]]; then
  echo "[baseline] Existing summary found, skipping baseline: ${BASELINE_DIR}/baseline_run_summary.csv"
else
  echo "[baseline] Running no-clean fixed-epoch MobileNet baseline"
  SEED_OVERRIDE="${SEED}" \
  OUTPUT_DIR_OVERRIDE="${BASELINE_DIR}" \
  EPOCHS_OVERRIDE=50 \
  BATCH_SIZE_OVERRIDE=32 \
  bash "${SCRIPT_DIR}/run_mobilenetv3_baseline_no_early_stop.sh"
fi

echo "[remove] Running sample20 simple-remove loop"
MODEL_BACKBONE_OVERRIDE=mobilenet_v3_small_scratch \
MODEL_TAG_OVERRIDE=mobilenet_v3_small_scratch_noes \
OOF_BATCH_SIZE_OVERRIDE=32 \
TRAIN_BATCH_SIZE_OVERRIDE=32 \
TRAIN_EPOCHS_OVERRIDE=50 \
TRAIN_EARLY_STOPPING_PATIENCE_OVERRIDE=100000 \
TRAIN_RECOVER_BEST_WEIGHTS_OVERRIDE=0 \
bash "${SCRIPT_DIR}/run_xrv_iterative_sample20_branch.sh" \
  "remove_only" "${REMOVE_ROOT}" "${LOOP_COUNT}" "${TOP_FRACTION}" "${SEED}"

cat > "${SEED_ROOT}/run_summary.txt" <<TXT
JOB_ID=${SLURM_JOB_ID}
NODE=${SLURMD_NODENAME}
SEED=${SEED}
OUTPUT_ROOT=${OUTPUT_ROOT}
BASELINE_DIR=${BASELINE_DIR}
REMOVE_ROOT=${REMOVE_ROOT}
LOOP_COUNT=${LOOP_COUNT}
TOP_FRACTION=${TOP_FRACTION}
MODEL_BACKBONE=mobilenet_v3_small_scratch
TRAIN_EPOCHS=50
TRAIN_EARLY_STOPPING_PATIENCE=100000
TRAIN_RECOVER_BEST_WEIGHTS=0
COMPLETED_AT=$(date)
TXT

echo "Completed at $(date)"
echo "Seed root: ${SEED_ROOT}"
