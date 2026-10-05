#!/bin/bash
#SBATCH --job-name=xrv_topk_entry
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/xrv_topk_entry_%A_%a.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/xrv_topk_entry_%A_%a.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2%3

set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
MAIN_SCRIPT="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/cxr_real_full_train_eval_cleanlab_xrv12.py"
SELECT_SCRIPT="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/select_topk_issue_entries.py"
DATA_ROOT="${PROJECT_DIR}/MedSoul/datasets"
IMAGE_ROOT="${DATA_ROOT}/mimic-cxr-jpg-224"
CHEXPERT_CSV="${IMAGE_ROOT}/mimic-cxr-2.0.0-chexpert.csv"
SPLIT_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-split.csv.gz"
TEST_CSV="${DATA_ROOT}/mimic-cxr-2.1.0-test-set-labeled.csv"
METADATA_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz"
PIPELINE_ROOT="${1:?pipeline_root must be provided as arg1}"
OOF_RUN_ROOT="${2:?oof_run_root must be provided as arg2}"

FRACTIONS=("0.05" "0.10" "0.20")
TAGS=("top05_entry_fraction" "top10_entry_fraction" "top20_entry_fraction")
IDX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be provided}"
FRACTION="${FRACTIONS[$IDX]}"
TAG="${TAGS[$IDX]}"

RUN_ROOT="${PIPELINE_ROOT}/03_entry_array_job_${SLURM_ARRAY_JOB_ID}"
SUBSET_CSV="${RUN_ROOT}/${TAG}_issue_subset.csv"
OUT_DIR="${RUN_ROOT}/${TAG}"
ISSUE_CSV="${OOF_RUN_ROOT}/train_cleanlab_entry_issues_only.csv"

mkdir -p "${PROJECT_DIR}/slurm_logs" "${RUN_ROOT}" "${OUT_DIR}"
cd "${PROJECT_DIR}" || exit 1

source /vol/cuda/12.5.0/setup.sh
source /vol/gpudata/yz3522-llmtest/venv/bin/activate

export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}" "/vol/gpudata/yz3522-llmtest/.cache/torchxrayvision/models_data"

echo "=========================================================="
echo "XRV Entry Top-k Array Formal"
echo "=========================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Time: $(date)"
echo "Fraction: ${FRACTION}"
echo "Tag: ${TAG}"
echo "OOF run root: ${OOF_RUN_ROOT}"
echo "Run root: ${RUN_ROOT}"
echo "=========================================================="

python -u "${SELECT_SCRIPT}" \
  --input-csv "${ISSUE_CSV}" \
  --output-csv "${SUBSET_CSV}" \
  --top-fraction "${FRACTION}"

python -u "${MAIN_SCRIPT}" \
  --image-root "${IMAGE_ROOT}" \
  --chexpert-csv "${CHEXPERT_CSV}" \
  --split-csv "${SPLIT_CSV}" \
  --test-csv "${TEST_CSV}" \
  --metadata-csv "${METADATA_CSV}" \
  --allowed-views AP PA \
  --train-split train \
  --train-limit -1 \
  --test-limit -1 \
  --epochs 100 \
  --val-fraction 0.1 \
  --early-stopping-patience 10 \
  --recover-best-weights \
  --batch-size 16 \
  --image-size 224 \
  --learning-rate 0.001 \
  --num-workers 4 \
  --model-backbone xrv_densenet121_linearhead \
  --xrv-weights densenet121-res224-all \
  --xrv-cache-dir /vol/gpudata/yz3522-llmtest/.cache/torchxrayvision/models_data \
  --device cuda \
  --study-aggregation max \
  --exclude-entry-csv "${SUBSET_CSV}" \
  --exclude-entry-issue-col est_issue_entry \
  --output-dir "${OUT_DIR}"

echo ""
echo "Completed at $(date)"
