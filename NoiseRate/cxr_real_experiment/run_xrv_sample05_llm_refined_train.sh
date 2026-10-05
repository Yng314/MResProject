#!/bin/bash
#SBATCH --job-name=xrv-s05-llm-refine
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/xrv_s05_llm_refine_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/xrv_s05_llm_refine_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00

set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_DIR}/NoiseRate/cxr_real_experiment"
MAIN_SCRIPT="${SCRIPT_DIR}/cxr_real_full_train_eval_cleanlab_xrv12.py"
BUILD_SCRIPT="${SCRIPT_DIR}/build_llm_refinement_tables.py"
DATA_ROOT="${PROJECT_DIR}/MedSoul/datasets"
IMAGE_ROOT="${DATA_ROOT}/mimic-cxr-jpg-224"
CHEXPERT_CSV="${IMAGE_ROOT}/mimic-cxr-2.0.0-chexpert.csv"
SPLIT_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-split.csv.gz"
TEST_CSV="${DATA_ROOT}/mimic-cxr-2.1.0-test-set-labeled.csv"
METADATA_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz"
LLM_RESULTS_CSV="${LLM_RESULTS_CSV_OVERRIDE:-/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/sample_top05pct_entry_llm_real/slurm_250525/llm_review/results.csv}"
RUN_TAG="${RUN_TAG_OVERRIDE:-xrv_sample05_llm_refined}"
OUTPUT_PARENT="${OUTPUT_PARENT_OVERRIDE:-/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_${RUN_TAG}}"
OUTPUT_DIR="${OUTPUT_PARENT}/slurm_${SLURM_JOB_ID}"
RELABEL_CSV="${OUTPUT_DIR}/llm_relabel_entries.csv"
MASK_CSV="${OUTPUT_DIR}/llm_mask_entries.csv"
SUMMARY_CSV="${OUTPUT_DIR}/llm_refinement_summary.csv"
SEED="${SEED_OVERRIDE:-42}"

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUTPUT_DIR}"
cd "${PROJECT_DIR}" || exit 1

source /vol/cuda/12.5.0/setup.sh
source /vol/gpudata/yz3522-llmtest/venv/bin/activate

export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}" "/vol/gpudata/yz3522-llmtest/.cache/torchxrayvision/models_data"

echo "=============================================="
echo "${RUN_TAG} Train"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Time: $(date)"
echo "Output Dir: ${OUTPUT_DIR}"
echo "LLM Results: ${LLM_RESULTS_CSV}"
echo "Seed: ${SEED}"
echo "=============================================="

python -u "${BUILD_SCRIPT}" \
  --llm-results-csv "${LLM_RESULTS_CSV}" \
  --relabel-csv "${RELABEL_CSV}" \
  --mask-csv "${MASK_CSV}" \
  --summary-csv "${SUMMARY_CSV}"

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
  --seed "${SEED}" \
  --study-aggregation max \
  --override-entry-csv "${RELABEL_CSV}" \
  --exclude-entry-csv "${MASK_CSV}" \
  --exclude-entry-issue-col est_issue_entry \
  --output-dir "${OUTPUT_DIR}/refined_train"

echo ""
echo "Completed at $(date)"
