#!/bin/bash
#SBATCH --job-name=mbv3-llm-fixed
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mbv3_llm_fixed_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mbv3_llm_fixed_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00

set -euo pipefail

SEED="${1:?seed must be provided}"
OUTPUT_ROOT="${2:?output_root must be provided}"
REFINE_SOURCE_ROOT="${3:?refine_source_root must be provided}"
LOOP_COUNT="${4:-5}"

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_DIR}/NoiseRate/cxr_real_experiment"
TRAIN_SCRIPT="${SCRIPT_DIR}/cxr_real_full_train_eval_cleanlab_xrv12.py"
DATA_ROOT="${PROJECT_DIR}/MedSoul/datasets"
IMAGE_ROOT="${DATA_ROOT}/mimic-cxr-jpg-224"
CHEXPERT_CSV="${IMAGE_ROOT}/mimic-cxr-2.0.0-chexpert.csv"
SPLIT_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-split.csv.gz"
TEST_CSV="${DATA_ROOT}/mimic-cxr-2.1.0-test-set-labeled.csv"
METADATA_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz"
SEED_ROOT="${OUTPUT_ROOT}/seed_${SEED}"
BASELINE_SUMMARY="${SEED_ROOT}/baseline_no_clean/baseline_run_summary.csv"
RUN_ROOT="${SEED_ROOT}/sample20_llm_refine_fixed_xrv_s13"

if [[ ! -s "${BASELINE_SUMMARY}" ]]; then
  echo "Missing completed baseline for seed ${SEED}: ${BASELINE_SUMMARY}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${PROJECT_DIR}/slurm_logs"

source /vol/cuda/12.5.0/setup.sh
source /vol/gpudata/yz3522-llmtest/venv/bin/activate

export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}" \
  "/vol/gpudata/yz3522-llmtest/.cache/torchxrayvision/models_data"

echo "=========================================================="
echo "MobileNetV3 noES fixed LLM-refined dataset evaluation"
echo "=========================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Seed: ${SEED}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Run root: ${RUN_ROOT}"
echo "Refinement source: ${REFINE_SOURCE_ROOT}"
echo "Loop count: ${LOOP_COUNT}"
echo "Started: $(date)"
echo "=========================================================="

cd "${PROJECT_DIR}" || exit 1

for LOOP_ID in $(seq 1 "${LOOP_COUNT}"); do
  LOOP_PAD="$(printf "%02d" "${LOOP_ID}")"
  SOURCE_LOOP="${REFINE_SOURCE_ROOT}/loop_${LOOP_PAD}"
  RELABEL_CSV="${SOURCE_LOOP}/applied_relabel_entries.csv"
  MASK_CSV="${SOURCE_LOOP}/applied_mask_entries.csv"
  TRAIN_DIR="${RUN_ROOT}/loop_${LOOP_PAD}/train_eval"
  SUMMARY_CSV="${TRAIN_DIR}/baseline_run_summary.csv"

  if [[ ! -s "${RELABEL_CSV}" || ! -s "${MASK_CSV}" ]]; then
    echo "Missing fixed refinement tables for loop ${LOOP_ID}: ${SOURCE_LOOP}" >&2
    exit 3
  fi

  if [[ -s "${SUMMARY_CSV}" ]]; then
    echo "[loop ${LOOP_ID}] Existing summary found, skipping: ${SUMMARY_CSV}"
    continue
  fi

  mkdir -p "${TRAIN_DIR}"
  echo ""
  echo "================ Loop ${LOOP_ID}/${LOOP_COUNT}: train/eval ================="
  echo "Relabel table: ${RELABEL_CSV}"
  echo "Mask table: ${MASK_CSV}"

  python -u "${TRAIN_SCRIPT}" \
    --image-root "${IMAGE_ROOT}" \
    --chexpert-csv "${CHEXPERT_CSV}" \
    --split-csv "${SPLIT_CSV}" \
    --test-csv "${TEST_CSV}" \
    --metadata-csv "${METADATA_CSV}" \
    --allowed-views AP PA \
    --train-split train \
    --train-limit -1 \
    --test-limit -1 \
    --epochs 50 \
    --val-fraction 0.1 \
    --early-stopping-patience 100000 \
    --batch-size 32 \
    --image-size 224 \
    --learning-rate 0.001 \
    --num-workers 4 \
    --model-backbone mobilenet_v3_small_scratch \
    --xrv-weights densenet121-res224-all \
    --xrv-cache-dir /vol/gpudata/yz3522-llmtest/.cache/torchxrayvision/models_data \
    --device cuda \
    --seed "${SEED}" \
    --study-aggregation max \
    --override-entry-csv "${RELABEL_CSV}" \
    --exclude-entry-csv "${MASK_CSV}" \
    --exclude-entry-issue-col est_issue_entry \
    --output-dir "${TRAIN_DIR}"
done

python - "${RUN_ROOT}" "${REFINE_SOURCE_ROOT}" "${SEED}" "${LOOP_COUNT}" <<'PY'
from pathlib import Path
import sys

import numpy as np
import pandas as pd

run_root = Path(sys.argv[1])
source_root = Path(sys.argv[2])
seed = int(sys.argv[3])
loop_count = int(sys.argv[4])
rows = []

for loop_id in range(1, loop_count + 1):
    loop_name = f"loop_{loop_id:02d}"
    train_dir = run_root / loop_name / "train_eval"
    summary = pd.read_csv(train_dir / "baseline_run_summary.csv").iloc[0]
    study_auc = pd.read_csv(train_dir / "test_study_auroc_summary.csv")
    valid = study_auc["study_auroc_binary"].notna() & (study_auc["study_valid_count"] > 0)
    weighted_auc = float(
        np.average(
            study_auc.loc[valid, "study_auroc_binary"],
            weights=study_auc.loc[valid, "study_valid_count"],
        )
    )
    source_loop = source_root / loop_name
    relabel_count = len(pd.read_csv(source_loop / "applied_relabel_entries.csv"))
    mask_count = len(pd.read_csv(source_loop / "applied_mask_entries.csv"))
    rows.append(
        {
            "branch": "llm_refine_fixed_xrv_s13",
            "loop_id": loop_id,
            "source_relabel_entries": relabel_count,
            "source_mask_entries": mask_count,
            "applied_relabel_entries": int(summary["relabeled_entry_count"]),
            "applied_mask_entries": int(summary["excluded_entry_count"]),
            "train_samples": int(summary["train_samples"]),
            "train_cleanlab_sample_issue_rate": float(summary["train_cleanlab_sample_issue_rate"]),
            "train_cleanlab_entry_issue_rate": float(summary["train_cleanlab_entry_issue_rate"]),
            "test_image_macro_auroc": float(summary["test_image_macro_auroc_binary"]),
            "test_study_macro_auroc": float(summary["test_study_macro_auroc_binary"]),
            "test_study_weighted_auroc": weighted_auc,
            "best_epoch": int(summary["best_epoch"]),
            "best_val_loss": float(summary["best_val_loss"]),
            "seed": seed,
        }
    )

pd.DataFrame(rows).to_csv(run_root / "loop_metrics.csv", index=False)
PY

cat > "${RUN_ROOT}/run_summary.txt" <<TXT
JOB_ID=${SLURM_JOB_ID}
NODE=${SLURMD_NODENAME}
SEED=${SEED}
OUTPUT_ROOT=${OUTPUT_ROOT}
RUN_ROOT=${RUN_ROOT}
REFINE_SOURCE_ROOT=${REFINE_SOURCE_ROOT}
LOOP_COUNT=${LOOP_COUNT}
MODEL_BACKBONE=mobilenet_v3_small_scratch
TRAIN_EPOCHS=50
TRAIN_EARLY_STOPPING_PATIENCE=100000
TRAIN_RECOVER_BEST_WEIGHTS=0
REFINEMENT_MODE=fixed_existing_xrv_seed13_tables_no_new_llm_calls
COMPLETED_AT=$(date)
TXT

echo "Completed at $(date)"
echo "Metrics: ${RUN_ROOT}/loop_metrics.csv"
