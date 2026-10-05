#!/bin/bash
#SBATCH --job-name=mb-ds100
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mb_ds100_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mb_ds100_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

EXPERIMENT_ROOT="${1:?experiment root is required}"
SEED="${2:?seed is required}"
TOP_FRACTION="${3:?top fraction is required}"
START_LOOP="${4:-1}"
END_LOOP="${5:-8}"
INCLUDE_BASELINE="${6:-1}"

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_DIR}/NoiseRate/cxr_real_experiment"
TRAIN_SCRIPT="${SCRIPT_DIR}/cxr_real_full_train_eval_cleanlab_xrv12.py"
SUMMARY_SCRIPT="${SCRIPT_DIR}/summarize_xrv_loop_metrics.py"
DATA_ROOT="${PROJECT_DIR}/MedSoul/datasets"
IMAGE_ROOT="${DATA_ROOT}/mimic-cxr-jpg-224"
CHEXPERT_CSV="${IMAGE_ROOT}/mimic-cxr-2.0.0-chexpert.csv"
SPLIT_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-split.csv.gz"
TEST_CSV="${DATA_ROOT}/mimic-cxr-2.1.0-test-set-labeled.csv"
METADATA_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz"
SEED_ROOT="${EXPERIMENT_ROOT}/seed_${SEED}"
RUN_ROOT="${SEED_ROOT}/llm_refine"
OUTPUT_BASENAME="downstream_mobilenet100_es10_best"
BASELINE_DIR="${SEED_ROOT}/${OUTPUT_BASENAME}/baseline"
METRICS_CSV="${RUN_ROOT}/loop_metrics.csv"
LEGACY_METRICS_CSV="${RUN_ROOT}/loop_metrics_fixed50_legacy.csv"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"

if (( START_LOOP < 1 || (END_LOOP != 0 && END_LOOP < START_LOOP) )); then
  echo "Invalid loop range: ${START_LOOP} -> ${END_LOOP}" >&2
  exit 2
fi
if [[ ! -d "${RUN_ROOT}" ]]; then
  if (( END_LOOP == 0 )); then
    mkdir -p "${RUN_ROOT}"
  else
    echo "Missing refinement run root: ${RUN_ROOT}" >&2
    exit 2
  fi
fi

source /vol/cuda/12.5.0/setup.sh
source /vol/gpudata/yz3522-llmtest/venv/bin/activate
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${PROJECT_DIR}/slurm_logs" "${TORCH_HOME}" "${MPLCONFIGDIR}"
if [[ -s "${METRICS_CSV}" && ! -e "${LEGACY_METRICS_CSV}" ]]; then
  cp -p "${METRICS_CSV}" "${LEGACY_METRICS_CSV}"
  echo "Preserved legacy fixed-50 loop metrics: ${LEGACY_METRICS_CSV}"
fi

validate_complete_output() {
  local output_dir="$1"
  "${PYTHON}" - "${output_dir}" "${SEED}" <<'PY'
from pathlib import Path
import sys
import pandas as pd

output_dir = Path(sys.argv[1])
seed = int(sys.argv[2])
required = [
    output_dir / "baseline_run_summary.csv",
    output_dir / "test_study_auroc_summary.csv",
    output_dir / "test_study_predictions.csv",
]
if not all(path.is_file() and path.stat().st_size > 0 for path in required):
    raise SystemExit(1)
row = pd.read_csv(required[0]).iloc[0]
expected = {
    "seed": seed,
    "epochs": 100,
    "early_stopping_patience": 10,
    "recover_best_weights": 1,
    "batch_size": 32,
    "model_backbone": "mobilenet_v3_small_scratch",
}
for key, value in expected.items():
    if str(row[key]) != str(value):
        raise SystemExit(f"Unexpected {key} in {output_dir}: {row[key]!r} != {value!r}")
PY
}

train_state() {
  local final_dir="$1"
  local relabel_csv="${2:-}"
  local mask_csv="${3:-}"
  local partial_dir

  if validate_complete_output "${final_dir}" 2>/dev/null; then
    echo "Unified downstream output already complete: ${final_dir}"
    return 0
  fi
  if [[ -e "${final_dir}" ]]; then
    echo "Refusing to overwrite an incomplete unified output: ${final_dir}" >&2
    exit 3
  fi

  partial_dir="${final_dir}.partial_${SLURM_JOB_ID:-manual}_$(date +%s)"
  mkdir -p "${partial_dir}"
  args=(
    --image-root "${IMAGE_ROOT}"
    --chexpert-csv "${CHEXPERT_CSV}"
    --split-csv "${SPLIT_CSV}"
    --test-csv "${TEST_CSV}"
    --metadata-csv "${METADATA_CSV}"
    --allowed-views AP PA
    --train-split train
    --train-limit -1
    --test-limit -1
    --epochs 100
    --val-fraction 0.1
    --early-stopping-patience 10
    --recover-best-weights
    --batch-size 32
    --image-size 224
    --learning-rate 0.001
    --num-workers 4
    --model-backbone mobilenet_v3_small_scratch
    --xrv-weights densenet121-res224-all
    --xrv-cache-dir /vol/gpudata/yz3522-llmtest/.cache/torchxrayvision/models_data
    --device cuda
    --seed "${SEED}"
    --study-aggregation max
    --output-dir "${partial_dir}"
  )
  if [[ -n "${relabel_csv}" || -n "${mask_csv}" ]]; then
    if [[ ! -s "${relabel_csv}" || ! -s "${mask_csv}" ]]; then
      echo "Missing cumulative relabel/mask table for ${final_dir}" >&2
      exit 3
    fi
    args+=(
      --override-entry-csv "${relabel_csv}"
      --exclude-entry-csv "${mask_csv}"
      --exclude-entry-issue-col est_issue_entry
    )
  fi

  "${PYTHON}" -u "${TRAIN_SCRIPT}" "${args[@]}"
  validate_complete_output "${partial_dir}"
  mv "${partial_dir}" "${final_dir}"
  echo "Completed unified downstream output: ${final_dir}"
}

echo "=========================================================="
echo "Unified independent MobileNet downstream evaluation"
echo "Experiment root: ${EXPERIMENT_ROOT}"
echo "Seed: ${SEED}"
echo "Loop range: ${START_LOOP} -> ${END_LOOP}"
echo "Maximum epochs: 100"
echo "Early-stopping patience: 10"
echo "Restore best validation checkpoint: yes"
echo "Output basename: ${OUTPUT_BASENAME}"
echo "Started: $(date --iso-8601=seconds)"
echo "=========================================================="

if [[ "${INCLUDE_BASELINE}" == "1" ]]; then
  train_state "${BASELINE_DIR}"
fi

if (( END_LOOP == 0 )); then
  echo "Baseline-only unified downstream evaluation finished at $(date --iso-8601=seconds)."
  exit 0
fi

for loop_id in $(seq "${START_LOOP}" "${END_LOOP}"); do
  loop_pad="$(printf '%02d' "${loop_id}")"
  loop_dir="${RUN_ROOT}/loop_${loop_pad}"
  relabel_csv="${loop_dir}/applied_relabel_entries.csv"
  mask_csv="${loop_dir}/applied_mask_entries.csv"
  output_dir="${loop_dir}/${OUTPUT_BASENAME}"
  if [[ ! -f "${loop_dir}/.action_tables_complete" ]]; then
    echo "Loop ${loop_id} has no completed refinement state; stopping downstream backfill here."
    break
  fi

  train_state "${output_dir}" "${relabel_csv}" "${mask_csv}"

  summary_args=(
    --branch llm_refine
    --loop-id "${loop_id}"
    --top-fraction "${TOP_FRACTION}"
    --oof-dir "${loop_dir}/oof"
    --selected-sample-csv "${loop_dir}/sample_top_fraction_issue_subset.csv"
    --train-dir "${output_dir}"
    --metrics-csv "${METRICS_CSV}"
    --cumulative-relabel-csv "${relabel_csv}"
    --cumulative-mask-csv "${mask_csv}"
  )
  if [[ -s "${loop_dir}/llm_review/results.csv" ]]; then
    summary_args+=(--llm-results-csv "${loop_dir}/llm_review/results.csv")
  fi
  if [[ -s "${loop_dir}/llm_review/errors.csv" ]]; then
    summary_args+=(--llm-errors-csv "${loop_dir}/llm_review/errors.csv")
  fi
  "${PYTHON}" -u "${SUMMARY_SCRIPT}" "${summary_args[@]}"
  touch "${loop_dir}/.${OUTPUT_BASENAME}_complete"
done

echo "Unified downstream evaluation finished at $(date --iso-8601=seconds)."
