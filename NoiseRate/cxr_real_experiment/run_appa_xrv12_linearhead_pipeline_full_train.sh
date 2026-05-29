#!/bin/bash
#SBATCH --job-name=cxr_appa_xrv12h
#SBATCH --output=slurm_logs/cxr_appa_xrv12h_%j.log
#SBATCH --error=slurm_logs/cxr_appa_xrv12h_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00

set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MRes"
SCRIPT_PATH="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/cxr_real_full_train_eval_cleanlab_xrv12.py"
DATA_ROOT="/vol/gpudata/yz3522-llmtest/MedSoul/datasets"
IMAGE_ROOT="${DATA_ROOT}/mimic-cxr-jpg-224"
CHEXPERT_CSV="${IMAGE_ROOT}/mimic-cxr-2.0.0-chexpert.csv"
SPLIT_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-split.csv.gz"
TEST_CSV="${DATA_ROOT}/mimic-cxr-2.1.0-test-set-labeled.csv"
METADATA_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz"
VIEW_ARGS=(AP PA)
RUN_ROOT="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/results_appa_xrv12_linearhead_pipeline/slurm_${SLURM_JOB_ID}"
BASELINE_DIR="${RUN_ROOT}/01_baseline"
SAMPLE_CLEAN_DIR="${RUN_ROOT}/02_sample_cleaned"
ENTRY_CLEAN_DIR="${RUN_ROOT}/03_entry_cleaned"

cd "${PROJECT_DIR}" || exit 1
mkdir -p slurm_logs "${RUN_ROOT}"

echo "=============================================="
echo "Real CXR AP/PA XRV12 Linear-Head Pipeline"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Time: $(date)"
echo "Views: ${VIEW_ARGS[*]}"
echo "Metadata CSV: ${METADATA_CSV}"
echo "Run root: ${RUN_ROOT}"
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

common_args=(
  --image-root "${IMAGE_ROOT}"
  --chexpert-csv "${CHEXPERT_CSV}"
  --split-csv "${SPLIT_CSV}"
  --test-csv "${TEST_CSV}"
  --metadata-csv "${METADATA_CSV}"
  --allowed-views "${VIEW_ARGS[@]}"
  --train-split train
  --train-limit -1
  --test-limit -1
  --epochs 100
  --val-fraction 0.1
  --early-stopping-patience 10
  --recover-best-weights
  --batch-size 16
  --image-size 224
  --learning-rate 0.001
  --num-workers 4
  --model-backbone xrv_densenet121_linearhead
  --xrv-weights densenet121-res224-all
  --xrv-cache-dir /vol/gpudata/yz3522-llmtest/.cache/torchxrayvision/models_data
  --device cuda
  --study-aggregation max
)

echo "[Stage 1/3] baseline"
python -u "${SCRIPT_PATH}" \
  "${common_args[@]}" \
  --output-dir "${BASELINE_DIR}"

SAMPLE_ISSUE_CSV="${BASELINE_DIR}/train_cleanlab_sample_issues_only.csv"
ENTRY_ISSUE_CSV="${BASELINE_DIR}/train_cleanlab_entry_issues_only.csv"

echo "[Stage 2/3] sample-cleaned"
python -u "${SCRIPT_PATH}" \
  "${common_args[@]}" \
  --exclude-sample-csv "${SAMPLE_ISSUE_CSV}" \
  --exclude-issue-col est_issue_sample \
  --output-dir "${SAMPLE_CLEAN_DIR}"

echo "[Stage 3/3] entry-cleaned"
python -u "${SCRIPT_PATH}" \
  "${common_args[@]}" \
  --exclude-entry-csv "${ENTRY_ISSUE_CSV}" \
  --exclude-entry-issue-col est_issue_entry \
  --output-dir "${ENTRY_CLEAN_DIR}"

python - <<PY
from pathlib import Path
import pandas as pd

run_root = Path("${RUN_ROOT}")
rows = []
for stage_name, subdir in [
    ("baseline", "01_baseline"),
    ("sample_cleaned", "02_sample_cleaned"),
    ("entry_cleaned", "03_entry_cleaned"),
]:
    summary_path = run_root / subdir / "baseline_run_summary.csv"
    df = pd.read_csv(summary_path)
    row = df.iloc[0].to_dict()
    row["stage"] = stage_name
    rows.append(row)
compare_df = pd.DataFrame(rows)
cols = ["stage"] + [c for c in compare_df.columns if c != "stage"]
compare_df = compare_df[cols]
out_path = run_root / "appa_xrv12_linearhead_pipeline_compare.csv"
compare_df.to_csv(out_path, index=False)
print(f"Wrote comparison CSV: {out_path}")
PY

echo ""
echo "Completed at $(date)"
