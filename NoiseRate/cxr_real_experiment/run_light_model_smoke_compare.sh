#!/bin/bash
#SBATCH --job-name=light-model-smoke
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/light_model_smoke_%A_%a.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/light_model_smoke_%A_%a.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --array=0-1

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
OUTPUT_ROOT="${OUTPUT_ROOT_OVERRIDE:-/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/light_model_smoke_compare/${STAMP}}"

MODELS=("resnet18_scratch" "mobilenet_v3_small_scratch")
MODEL_BACKBONE="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_BACKBONE}"

mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/slurm_logs"
cd "${PROJECT_DIR}" || exit 1

source /vol/cuda/12.5.0/setup.sh
source /vol/gpudata/yz3522-llmtest/venv/bin/activate

export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}"

echo "=========================================================="
echo "Light model smoke compare"
echo "=========================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Model: ${MODEL_BACKBONE}"
echo "Output root: ${OUTPUT_ROOT}"
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
  --train-limit "${TRAIN_LIMIT_OVERRIDE:-5000}" \
  --test-limit "${TEST_LIMIT_OVERRIDE:-1000}" \
  --epochs "${EPOCHS_OVERRIDE:-3}" \
  --val-fraction 0.1 \
  --early-stopping-patience 2 \
  --recover-best-weights \
  --batch-size "${BATCH_SIZE_OVERRIDE:-32}" \
  --image-size 224 \
  --learning-rate "${LR_OVERRIDE:-0.001}" \
  --num-workers 4 \
  --model-backbone "${MODEL_BACKBONE}" \
  --device cuda \
  --seed "${SEED_OVERRIDE:-13}" \
  --study-aggregation max \
  --output-dir "${OUTPUT_DIR}"

echo "Completed: $(date)"
