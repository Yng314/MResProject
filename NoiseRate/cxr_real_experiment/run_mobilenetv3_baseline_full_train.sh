#!/bin/bash
#SBATCH --job-name=mbv3-baseline
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mbv3_baseline_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mbv3_baseline_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00

set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_DIR}/NoiseRate/cxr_real_experiment"
DATA_ROOT="${PROJECT_DIR}/MedSoul/datasets"
IMAGE_ROOT="${DATA_ROOT}/mimic-cxr-jpg-224"
CHEXPERT_CSV="${IMAGE_ROOT}/mimic-cxr-2.0.0-chexpert.csv"
SPLIT_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-split.csv.gz"
TEST_CSV="${DATA_ROOT}/mimic-cxr-2.1.0-test-set-labeled.csv"
METADATA_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz"
STAMP="${STAMP_OVERRIDE:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_baseline_full_train/${STAMP}}"

mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/slurm_logs"
cd "${PROJECT_DIR}" || exit 1

source /vol/cuda/12.5.0/setup.sh
source /vol/gpudata/yz3522-llmtest/venv/bin/activate

export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}"

echo "=========================================================="
echo "MobileNetV3 small scratch full-data baseline"
echo "=========================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Started: $(date)"
echo "=========================================================="

python -u "${SCRIPT_DIR}/cxr_real_full_train_eval_cleanlab_xrv12.py" \
  --image-root "${IMAGE_ROOT}" \
  --chexpert-csv "${CHEXPERT_CSV}" \
  --split-csv "${SPLIT_CSV}" \
  --test-csv "${TEST_CSV}" \
  --metadata-csv "${METADATA_CSV}" \
  --allowed-views AP PA \
  --train-split train \
  --train-limit -1 \
  --test-limit -1 \
  --epochs "${EPOCHS_OVERRIDE:-100}" \
  --val-fraction 0.1 \
  --early-stopping-patience 10 \
  --recover-best-weights \
  --batch-size "${BATCH_SIZE_OVERRIDE:-32}" \
  --image-size 224 \
  --learning-rate "${LR_OVERRIDE:-0.001}" \
  --num-workers 4 \
  --model-backbone mobilenet_v3_small_scratch \
  --device cuda \
  --seed "${SEED_OVERRIDE:-13}" \
  --study-aggregation max \
  --output-dir "${OUTPUT_DIR}"

echo "Completed: $(date)"
