#!/bin/bash
#SBATCH --job-name=cxr_appa_etopk
#SBATCH --output=slurm_logs/cxr_appa_etopk_%j.log
#SBATCH --error=slurm_logs/cxr_appa_etopk_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00

set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MRes"
MAIN_SCRIPT="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/cxr_real_full_train_eval_cleanlab.py"
SELECT_SCRIPT="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/select_topk_issue_entries.py"
DATA_ROOT="/vol/gpudata/yz3522-llmtest/MedSoul/datasets"
IMAGE_ROOT="${DATA_ROOT}/mimic-cxr-jpg-224"
CHEXPERT_CSV="${IMAGE_ROOT}/mimic-cxr-2.0.0-chexpert.csv"
SPLIT_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-split.csv.gz"
TEST_CSV="${DATA_ROOT}/mimic-cxr-2.1.0-test-set-labeled.csv"
METADATA_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz"
VIEW_ARGS=(AP PA)
BASELINE_DIR="${BASELINE_DIR:-/vol/gpudata/yz3522-llmtest/MRes/NoiseRate/cxr_real_experiment/results_appa_pipeline/slurm_236304/01_baseline}"
ISSUE_CSV="${BASELINE_DIR}/train_cleanlab_entry_issues_only.csv"
RUN_ROOT="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/results_appa_densenet_entry_topk/slurm_${SLURM_JOB_ID}"

cd "${PROJECT_DIR}" || exit 1
mkdir -p slurm_logs "${RUN_ROOT}"

echo "==================================================="
echo "Real CXR AP/PA DenseNet Entry Top-k Sweep"
echo "==================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Time: $(date)"
echo "Views: ${VIEW_ARGS[*]}"
echo "Baseline dir: ${BASELINE_DIR}"
echo "Issue CSV: ${ISSUE_CSV}"
echo "Run root: ${RUN_ROOT}"
echo "==================================================="

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
  --model-backbone densenet121_pretrained
  --device cuda
  --study-aggregation max
)

run_one () {
  local tag="$1"
  local frac="$2"
  local subset_csv="${RUN_ROOT}/${tag}_issue_subset.csv"
  local out_dir="${RUN_ROOT}/${tag}"

  echo "[Top-k Entry] building subset ${tag} from fraction=${frac}"
  python -u "${SELECT_SCRIPT}" \
    --input-csv "${ISSUE_CSV}" \
    --output-csv "${subset_csv}" \
    --top-fraction "${frac}"

  echo "[Top-k Entry] running ${tag}"
  python -u "${MAIN_SCRIPT}" \
    "${common_args[@]}" \
    --exclude-entry-csv "${subset_csv}" \
    --exclude-entry-issue-col est_issue_entry \
    --output-dir "${out_dir}"
}

run_one "top05_entry_fraction" "0.05"
run_one "top10_entry_fraction" "0.10"
run_one "top20_entry_fraction" "0.20"

python - <<PY
from pathlib import Path
import pandas as pd

run_root = Path("${RUN_ROOT}")
rows = []
for stage_name in ["top05_entry_fraction", "top10_entry_fraction", "top20_entry_fraction"]:
    summary_path = run_root / stage_name / "baseline_run_summary.csv"
    df = pd.read_csv(summary_path)
    row = df.iloc[0].to_dict()
    row["stage"] = stage_name
    rows.append(row)
compare_df = pd.DataFrame(rows)
cols = ["stage"] + [c for c in compare_df.columns if c != "stage"]
compare_df = compare_df[cols]
out_path = run_root / "topk_entry_sweep_compare.csv"
compare_df.to_csv(out_path, index=False)
print(f"Wrote comparison CSV: {out_path}")
PY

echo ""
echo "Completed at $(date)"
