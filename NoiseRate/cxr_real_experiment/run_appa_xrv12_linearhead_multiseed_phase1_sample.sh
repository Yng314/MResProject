#!/bin/bash
#SBATCH --job-name=cxr_xrv12_ms1s
#SBATCH --output=slurm_logs/cxr_xrv12_ms1s_%A_%a.log
#SBATCH --error=slurm_logs/cxr_xrv12_ms1s_%A_%a.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --array=0-2

set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MRes"
MAIN_SCRIPT="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/cxr_real_full_train_eval_cleanlab_xrv12.py"
SELECT_SAMPLE_SCRIPT="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/select_topk_issue_samples.py"
DATA_ROOT="/vol/gpudata/yz3522-llmtest/MedSoul/datasets"
IMAGE_ROOT="${DATA_ROOT}/mimic-cxr-jpg-224"
CHEXPERT_CSV="${IMAGE_ROOT}/mimic-cxr-2.0.0-chexpert.csv"
SPLIT_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-split.csv.gz"
TEST_CSV="${DATA_ROOT}/mimic-cxr-2.1.0-test-set-labeled.csv"
METADATA_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz"
VIEW_ARGS=(AP PA)
SEEDS=(13 42 97)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
RUN_ROOT="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/results_appa_xrv12_linearhead_multiseed_phase1_sample/seed_${SEED}"
BASELINE_DIR="${RUN_ROOT}/01_baseline"

cd "${PROJECT_DIR}" || exit 1
mkdir -p slurm_logs "${RUN_ROOT}"

echo "======================================================================"
echo "XRV Linear-Head Multi-seed Phase 1 Sample"
echo "======================================================================"
echo "Array job: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Time: $(date)"
echo "Seed: ${SEED}"
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

echo "[Stage 1/3] baseline"
python -u "${MAIN_SCRIPT}" \
  "${common_args[@]}" \
  --output-dir "${BASELINE_DIR}"

SAMPLE_ISSUE_CSV="${BASELINE_DIR}/train_cleanlab_sample_issues_only.csv"
TOP10_SAMPLE_SUBSET="${RUN_ROOT}/top10_sample_issue_subset.csv"
TOP20_SAMPLE_SUBSET="${RUN_ROOT}/top20_sample_issue_subset.csv"

python -u "${SELECT_SAMPLE_SCRIPT}" \
  --input-csv "${SAMPLE_ISSUE_CSV}" \
  --output-csv "${TOP10_SAMPLE_SUBSET}" \
  --top-fraction 0.10

python -u "${SELECT_SAMPLE_SCRIPT}" \
  --input-csv "${SAMPLE_ISSUE_CSV}" \
  --output-csv "${TOP20_SAMPLE_SUBSET}" \
  --top-fraction 0.20

echo "[Stage 2/3] top10 sample"
python -u "${MAIN_SCRIPT}" \
  "${common_args[@]}" \
  --exclude-sample-csv "${TOP10_SAMPLE_SUBSET}" \
  --exclude-issue-col est_issue_sample \
  --output-dir "${RUN_ROOT}/02_top10_sample"

echo "[Stage 3/3] top20 sample"
python -u "${MAIN_SCRIPT}" \
  "${common_args[@]}" \
  --exclude-sample-csv "${TOP20_SAMPLE_SUBSET}" \
  --exclude-issue-col est_issue_sample \
  --output-dir "${RUN_ROOT}/03_top20_sample"

python - <<PY
from pathlib import Path
import pandas as pd

run_root = Path("${RUN_ROOT}")
rows = []
stage_map = [
    ("baseline", "01_baseline"),
    ("top10_sample", "02_top10_sample"),
    ("top20_sample", "03_top20_sample"),
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
out_path = run_root / "phase1_sample_compare.csv"
compare_df.to_csv(out_path, index=False)
print(f"Wrote comparison CSV: {out_path}")
PY

echo "Completed at $(date)"
