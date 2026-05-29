#!/bin/bash
#SBATCH --job-name=cxr_real_entry
#SBATCH --output=slurm_logs/cxr_real_entry_%j.log
#SBATCH --error=slurm_logs/cxr_real_entry_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00

set -e

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MRes"
cd "${PROJECT_DIR}" || exit 1

mkdir -p slurm_logs

BASELINE_DIR="/vol/gpudata/yz3522-llmtest/MRes/NoiseRate/cxr_real_experiment/results_baseline_es/slurm_234377"
ENTRY_ISSUE_CSV="${BASELINE_DIR}/train_cleanlab_entry_issues_only.csv"

echo "=============================================="
echo "Real CXR Entry-Cleaned Train + Test + Cleanlab"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Entry Issue CSV: ${ENTRY_ISSUE_CSV}"
echo "=============================================="

source /vol/cuda/12.5.0/setup.sh
source /vol/gpudata/yz3522-llmtest/venv/bin/activate

export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}"

echo "[GPU]"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

python -u NoiseRate/cxr_real_experiment/cxr_real_full_train_eval_cleanlab.py \
  --image-root /vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-jpg-224 \
  --chexpert-csv /vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-jpg-224/mimic-cxr-2.0.0-chexpert.csv \
  --split-csv /vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-2.0.0-split.csv.gz \
  --test-csv /vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-2.1.0-test-set-labeled.csv \
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
  --model-backbone densenet121_pretrained \
  --device cuda \
  --study-aggregation max \
  --exclude-entry-csv "${ENTRY_ISSUE_CSV}" \
  --exclude-entry-issue-col est_issue_entry \
  --output-dir /vol/gpudata/yz3522-llmtest/MRes/NoiseRate/cxr_real_experiment/results_entry_cleaned_es/slurm_${SLURM_JOB_ID}

echo ""
echo "Completed at $(date)"
