#!/bin/bash
#SBATCH --job-name=mb-fullpool
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mb_fullpool_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mb_fullpool_%j.err
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
SOURCE_LOOP="${3:?source Loop 1 path is required}"
TARGET_LOOPS="${4:-8}"
PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_DIR}/NoiseRate/cxr_real_experiment"
PIPELINE="${SCRIPT_DIR}/run_xrv_iterative_sample20_unseen_branch.sh"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
SEED_ROOT="${EXPERIMENT_ROOT}/seed_${SEED}"
TARGET_LOOP="${SEED_ROOT}/llm_refine/loop_01"
PROTOCOL="${SCRIPT_DIR}/mimic_full_issue_pool_no_repeat_protocol_20260820.md"

if ! [[ "${TARGET_LOOPS}" =~ ^[1-8]$ ]]; then
  echo "target loops must be an integer from 1 to 8: ${TARGET_LOOPS}" >&2
  exit 2
fi

verify_hash() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(sha256sum "${path}" | cut -d' ' -f1)"
  if [[ "${observed}" != "${expected}" ]]; then
    echo "Protocol hash mismatch for ${path}: ${observed} != ${expected}" >&2
    exit 3
  fi
}

verify_hash "${PIPELINE}" "792d34003944c4e9adbfe833b03eaf509a0ff05d3772cc9427f9ff44943212b9"
verify_hash "${SCRIPT_DIR}/select_topk_unreviewed_issue_samples.py" "5f7a0248e51a0ff9c255eb6bebf5b1787abe99fe6707e5508cf331997bb84001"
verify_hash "${SCRIPT_DIR}/cxr_real_oof_cleanlab_smoke.py" "dc6b2d577985bbeec4d1c9e71911fca78419f63126c7c4eaa1d2397943c29a7b"
verify_hash "${SCRIPT_DIR}/cxr_real_full_train_eval_cleanlab_xrv12.py" "f9b1eb32aaa7d6df9e1d1ebd3ba66fd2ef637bde47676fda4c7b629a7c7f452d"
verify_hash "${SCRIPT_DIR}/export_sample_topk_entries_for_llm.py" "3d34829e686ee9680f702f6ad6e91c0f76519cf89e194d0f46ddf1b968765bfa"
verify_hash "${SCRIPT_DIR}/run_entry_level_llm_review.py" "4c42d95eb13d756ab76da721c67ce3228eb96cea394f6157e311764ccd7205d1"
verify_hash "${SCRIPT_DIR}/run_entry_level_llm_review_sharded.py" "941dd99cc396cca16a54ad07e0a83263f5d5b2b5c83053af376e9fa2fc786eed"
verify_hash "${SCRIPT_DIR}/build_llm_refinement_tables.py" "541d6591de6a47ea320e14c6862397d0c37339b71ec4d374afc057b7080c4c67"
verify_hash "${SCRIPT_DIR}/summarize_xrv_loop_metrics.py" "c3e47d50566fe65d113f7ae807890d4690ccdce0e997dd49dab80ecc295e4516"
verify_hash "${PROTOCOL}" "18c85bcf48ae3a16d190955e20b749ae54285abfca9d2c8e2820f35210f17025"

"${PYTHON}" - "${SOURCE_LOOP}" "${SEED}" <<'PY'
from pathlib import Path
import sys

import pandas as pd

loop_dir = Path(sys.argv[1])
seed = int(sys.argv[2])
summary_path = loop_dir / "oof" / "oof_cleanlab_smoke_summary.csv"
issue_path = loop_dir / "oof" / "train_cleanlab_sample_issues_only.csv"
if not summary_path.is_file() or not issue_path.is_file():
    raise SystemExit(f"Missing reusable baseline OOF evidence under {loop_dir}")
summary = pd.read_csv(summary_path).iloc[0]
expected = {
    "seed": seed,
    "n_splits": 4,
    "epochs": 100,
    "batch_size": 32,
    "model_backbone": "mobilenet_v3_small_scratch",
}
for key, value in expected.items():
    observed = summary[key]
    if str(observed) != str(value):
        raise SystemExit(f"Reusable OOF mismatch for {key}: {observed!r} != {value!r}")
issues = pd.read_csv(issue_path, usecols=["pool_row_id", "est_issue_entry_count"])
if issues.empty or issues["pool_row_id"].duplicated().any():
    raise SystemExit("Reusable baseline issue table is empty or has duplicate samples")
print(
    f"Validated seed {seed} baseline OOF: {len(issues)} issue samples, "
    f"{int(issues['est_issue_entry_count'].sum())} issue entries."
)
PY

mkdir -p "${TARGET_LOOP}" "${PROJECT_DIR}/slurm_logs" "${EXPERIMENT_ROOT}"
if [[ -L "${TARGET_LOOP}/oof" ]]; then
  if [[ "$(readlink -f "${TARGET_LOOP}/oof")" != "$(readlink -f "${SOURCE_LOOP}/oof")" ]]; then
    echo "Existing Loop 1 OOF symlink points to the wrong source." >&2
    exit 3
  fi
elif [[ -e "${TARGET_LOOP}/oof" ]]; then
  echo "Refusing to replace existing non-symlink Loop 1 OOF path." >&2
  exit 3
else
  ln -s "${SOURCE_LOOP}/oof" "${TARGET_LOOP}/oof"
fi

source ~/.llm_review_env
export OPENAI_MODEL="gpt-5.4"
ulimit -n 4096

cat > "${SEED_ROOT}/launch_manifest.txt" <<EOF
protocol=mimic_full_issue_pool_no_repeat_20260820
seed=${SEED}
planned_loops=8
current_stop_loop=${TARGET_LOOPS}
top_fraction=1.0
history=successful_llm_results_only
model=${OPENAI_MODEL}
llm_total_concurrency=10000
llm_review_shards=3
baseline_oof=${SOURCE_LOOP}/oof
downstream_model=mobilenet_v3_small_scratch
downstream_max_epochs=100
downstream_early_stopping_patience=10
downstream_restore_best_weights=1
downstream_output_basename=downstream_mobilenet100_es10_best
slurm_job_id=${SLURM_JOB_ID}
EOF

echo "=========================================================="
echo "MIMIC full issue-pool no-repeat refinement"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Seed: ${SEED}"
echo "Current stop loop: ${TARGET_LOOPS} (planned maximum: 8)"
echo "Selection: all current CL issue entries not successfully reviewed earlier"
echo "LLM model: ${OPENAI_MODEL}"
echo "Run root: ${SEED_ROOT}/llm_refine"
echo "Open-file limit: $(ulimit -Sn)"
echo "Started: $(date --iso-8601=seconds)"
echo "=========================================================="

export MODEL_BACKBONE_OVERRIDE="mobilenet_v3_small_scratch"
export MODEL_TAG_OVERRIDE="mobilenet_v3_small_scratch_noes"
export OOF_BATCH_SIZE_OVERRIDE="32"
export TRAIN_BATCH_SIZE_OVERRIDE="32"
export OOF_EPOCHS_OVERRIDE="100"
export TRAIN_EPOCHS_OVERRIDE="100"
export TRAIN_EARLY_STOPPING_PATIENCE_OVERRIDE="10"
export TRAIN_RECOVER_BEST_WEIGHTS_OVERRIDE="1"
export TRAIN_DIR_BASENAME_OVERRIDE="downstream_mobilenet100_es10_best"
export BATCH_SIZE_OVERRIDE="10000"
export CONCURRENCY_OVERRIDE="10000"
export LLM_REVIEW_SHARDS_OVERRIDE="3"
export RETRY_LIMIT_OVERRIDE="2"
export LLM_RESUME_PASSES_OVERRIDE="4"
export REQUEST_TIMEOUT_OVERRIDE="180"
export MAX_COMPLETION_TOKENS_OVERRIDE="400"
export UNSEEN_REVIEW_ONLY_OVERRIDE="1"
export FULL_ISSUE_POOL_OVERRIDE="1"
export SUCCESSFUL_HISTORY_ONLY_OVERRIDE="1"
export REQUIRE_COMPLETE_LLM_REVIEW_OVERRIDE="1"

exec bash "${PIPELINE}" llm_refine "${SEED_ROOT}" "${TARGET_LOOPS}" 1.0 "${SEED}" 1 "${TARGET_LOOPS}"
