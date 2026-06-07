#!/bin/bash
#SBATCH --job-name=cxr_xrv12_ms1e
#SBATCH --output=slurm_logs/cxr_xrv12_ms1e_%A_%a.log
#SBATCH --error=slurm_logs/cxr_xrv12_ms1e_%A_%a.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --array=0-2

set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MRes"
MAIN_SCRIPT="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/cxr_real_full_train_eval_cleanlab_xrv12.py"
SELECT_ENTRY_SCRIPT="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/select_topk_issue_entries.py"
SELECT_ENTRY_PERLABEL_SCRIPT="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/select_topk_issue_entries_per_label.py"
DATA_ROOT="/vol/gpudata/yz3522-llmtest/MedSoul/datasets"
IMAGE_ROOT="${DATA_ROOT}/mimic-cxr-jpg-224"
CHEXPERT_CSV="${IMAGE_ROOT}/mimic-cxr-2.0.0-chexpert.csv"
SPLIT_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-split.csv.gz"
TEST_CSV="${DATA_ROOT}/mimic-cxr-2.1.0-test-set-labeled.csv"
METADATA_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz"
VIEW_ARGS=(AP PA)
SEEDS=(13 42 97)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
SAMPLE_ROOT="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/results_appa_xrv12_linearhead_multiseed_phase1_sample/seed_${SEED}"
BASELINE_DIR="${SAMPLE_ROOT}/01_baseline"
ISSUE_CSV="${BASELINE_DIR}/train_cleanlab_entry_issues_only.csv"
RUN_ROOT="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/results_appa_xrv12_linearhead_multiseed_phase1_entry/seed_${SEED}"

cd "${PROJECT_DIR}" || exit 1
mkdir -p slurm_logs "${RUN_ROOT}"

if [[ ! -f "${ISSUE_CSV}" ]]; then
  echo "Missing baseline issue CSV: ${ISSUE_CSV}" >&2
  exit 1
fi

echo "======================================================================"
echo "XRV Linear-Head Multi-seed Phase 1 Entry"
echo "======================================================================"
echo "Array job: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Time: $(date)"
echo "Seed: ${SEED}"
echo "Baseline dir: ${BASELINE_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "======================================================================"

source /vol/cuda/12.5.0/setup.sh
source /vol/gpudata/yz3522-llmtest/venv/bin/activate

export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}" "/vol/gpudata/yz3522-llmtest/.cache/torchxrayvision/models_data"

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
  --seed "${SEED}"
)

TOP10_ENTRY_SUBSET="${RUN_ROOT}/top10_entry_issue_subset.csv"
TOP10_ENTRY_PERLABEL_SUBSET="${RUN_ROOT}/top10_entry_perlabel_issue_subset.csv"

python -u "${SELECT_ENTRY_SCRIPT}" \
  --input-csv "${ISSUE_CSV}" \
  --output-csv "${TOP10_ENTRY_SUBSET}" \
  --top-fraction 0.10

python -u "${SELECT_ENTRY_PERLABEL_SCRIPT}" \
  --input-csv "${ISSUE_CSV}" \
  --output-csv "${TOP10_ENTRY_PERLABEL_SUBSET}" \
  --top-fraction 0.10

echo "[Stage 1/2] top10 entry (global)"
python -u "${MAIN_SCRIPT}" \
  "${common_args[@]}" \
  --exclude-entry-csv "${TOP10_ENTRY_SUBSET}" \
  --exclude-entry-issue-col est_issue_entry \
  --output-dir "${RUN_ROOT}/01_top10_entry_global"

echo "[Stage 2/2] top10 entry (per-label)"
python -u "${MAIN_SCRIPT}" \
  "${common_args[@]}" \
  --exclude-entry-csv "${TOP10_ENTRY_PERLABEL_SUBSET}" \
  --exclude-entry-issue-col est_issue_entry \
  --output-dir "${RUN_ROOT}/02_top10_entry_perlabel"

python - <<PY
from pathlib import Path
import pandas as pd

run_root = Path("${RUN_ROOT}")
rows = []
stage_map = [
    ("top10_entry_global", "01_top10_entry_global"),
    ("top10_entry_perlabel", "02_top10_entry_perlabel"),
]
for stage_name, stage_dir in stage_map:
    summary_path = run_root / stage_dir / "baseline_run_summary.csv"
    df = pd.read_csv(summary_path)
    row = df.iloc[0].to_dict()
    row["stage"] = stage_name
    rows.append(row)
compare_df = pd.DataFrame(rows)
cols = ["stage"] + [c for c in compare_df.columns if c != "stage"]
compare_df = compare_df[cols]
out_path = run_root / "phase1_entry_compare.csv"
compare_df.to_csv(out_path, index=False)
print(f"Wrote comparison CSV: {out_path}")
PY

echo "Completed at $(date)"
