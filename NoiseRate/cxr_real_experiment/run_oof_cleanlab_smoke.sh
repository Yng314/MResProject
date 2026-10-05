#!/bin/bash
#SBATCH --job-name=cxr_oof_smoke
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/cxr_oof_smoke_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/cxr_oof_smoke_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_PATH="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/cxr_real_oof_cleanlab_smoke.py"
SELECT_SAMPLE_SCRIPT="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/select_topk_issue_samples.py"
SELECT_ENTRY_SCRIPT="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/select_topk_issue_entries.py"
DATA_ROOT="${PROJECT_DIR}/MedSoul/datasets"
IMAGE_ROOT="${DATA_ROOT}/mimic-cxr-jpg-224"
CHEXPERT_CSV="${IMAGE_ROOT}/mimic-cxr-2.0.0-chexpert.csv"
SPLIT_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-split.csv.gz"
METADATA_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz"
RUN_ROOT="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/results_oof_smoke/slurm_${SLURM_JOB_ID}"

mkdir -p "${PROJECT_DIR}/slurm_logs" "${RUN_ROOT}"
cd "${PROJECT_DIR}" || exit 1

source /vol/cuda/12.5.0/setup.sh
source /vol/gpudata/yz3522-llmtest/venv/bin/activate

export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}" "/vol/gpudata/yz3522-llmtest/.cache/torchxrayvision/models_data"

echo "=============================================="
echo "Real CXR OOF Cleanlab Smoke"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Time: $(date)"
echo "Run root: ${RUN_ROOT}"
echo "=============================================="

python -u "${SCRIPT_PATH}" \
  --image-root "${IMAGE_ROOT}" \
  --chexpert-csv "${CHEXPERT_CSV}" \
  --split-csv "${SPLIT_CSV}" \
  --metadata-csv "${METADATA_CSV}" \
  --allowed-views AP PA \
  --train-split train \
  --num-samples 1000 \
  --n-splits 3 \
  --epochs 2 \
  --batch-size 16 \
  --image-size 224 \
  --learning-rate 0.001 \
  --early-stopping-patience 2 \
  --model-backbone xrv_densenet121_linearhead \
  --xrv-weights densenet121-res224-all \
  --xrv-cache-dir /vol/gpudata/yz3522-llmtest/.cache/torchxrayvision/models_data \
  --device cuda \
  --seed 42 \
  --num-workers 0 \
  --output-dir "${RUN_ROOT}"

python -u "${SELECT_SAMPLE_SCRIPT}" \
  --input-csv "${RUN_ROOT}/train_cleanlab_sample_issues_only.csv" \
  --output-csv "${RUN_ROOT}/top10_sample_issue_subset.csv" \
  --top-fraction 0.10

python -u "${SELECT_ENTRY_SCRIPT}" \
  --input-csv "${RUN_ROOT}/train_cleanlab_entry_issues_only.csv" \
  --output-csv "${RUN_ROOT}/top10_entry_issue_subset.csv" \
  --top-fraction 0.10

export RUN_ROOT
python - <<'PY'
import os
from pathlib import Path
import pandas as pd

run_root = Path(os.environ["RUN_ROOT"])
sample_df = pd.read_csv(run_root / "train_cleanlab_sample_details.csv")
entry_df = pd.read_csv(run_root / "train_cleanlab_entry_details.csv")
sample_topk = pd.read_csv(run_root / "top10_sample_issue_subset.csv")
entry_topk = pd.read_csv(run_root / "top10_entry_issue_subset.csv")

print("sample rows:", len(sample_df))
print("sample issues:", int(sample_df["est_issue_sample"].sum()))
print("entry rows:", len(entry_df))
print("entry issues:", int(entry_df["est_issue_entry"].sum()))
print("top10 sample subset rows:", len(sample_topk))
print("top10 entry subset rows:", len(entry_topk))
print("")
print("entry label distribution:")
print(entry_df[entry_df["est_issue_entry"] == 1]["label_name"].value_counts().head(10).to_string())
PY

echo ""
echo "Completed at $(date)"
