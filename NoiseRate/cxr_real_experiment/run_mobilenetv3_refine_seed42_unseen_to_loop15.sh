#!/bin/bash
#SBATCH --job-name=mb-s42-new15
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mb_s42_new15_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mb_s42_new15_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail

SEED=42
LOOP_COUNT=15
TOP_FRACTION=0.20
PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
NOISERATE_DIR="${PROJECT_DIR}/NoiseRate"
SCRIPT_DIR="${NOISERATE_DIR}/cxr_real_experiment"
PIPELINE="${SCRIPT_DIR}/run_xrv_iterative_sample20_unseen_branch.sh"
SELECTOR="${SCRIPT_DIR}/select_topk_unreviewed_issue_samples.py"
SELECTOR_TEST="${SCRIPT_DIR}/test_select_topk_unreviewed_issue_samples.py"
PROTOCOL="${SCRIPT_DIR}/seed42_unseen_review_extension_protocol_20260726.md"
SOURCE_EXPERIMENT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_own_top20_binary_v5_llm_refine_5seed/20260714_065539"
SOURCE_RUN_ROOT="${SOURCE_EXPERIMENT_ROOT}/seed_${SEED}/llm_refine"
EXPERIMENT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_own_top20_binary_v5_refine_unseen_extension/20260726_seed42_to_loop15"
SEED_ROOT="${EXPERIMENT_ROOT}/seed_${SEED}"
RUN_ROOT="${SEED_ROOT}/llm_refine"
FROZEN_DIR="${SOURCE_EXPERIMENT_ROOT}/protocol_snapshot_v5_binary_label_locked"
EXPECTED_OPENAI_MODEL="gpt-5.4"
EXPECTED_PIPELINE_SHA256="88613a0f7a594f1e08ddcffbf734d845e55d52b2762d6d1435269509366fe7e9"
EXPECTED_SELECTOR_SHA256="8b573edfafa693100ac642b9c9e21503e4b3bd1dd53f71ab7f106a434d142901"
EXPECTED_SELECTOR_TEST_SHA256="86758e65b75543a4f8d26f87bd0301ed7bee0276a3962a182f50595824994199"
EXPECTED_PROTOCOL_SHA256="10dba489a4bd70a23b214a96ea4d7a330ba7458fb6d429e00728a0bea15e73a6"

verify_sha256() {
  local expected="$1"
  local path="$2"
  local observed
  observed="$(sha256sum "${path}" | cut -d' ' -f1)"
  if [[ "${observed}" != "${expected}" ]]; then
    echo "Checksum mismatch for ${path}: ${observed} != ${expected}" >&2
    exit 3
  fi
}

if ! (cd "${FROZEN_DIR}" && sha256sum --check checksums.sha256); then
  echo "Frozen v5 snapshot checksum verification failed." >&2
  exit 3
fi
for frozen_name in \
  build_llm_refinement_tables.py \
  cxr_real_full_train_eval_cleanlab_xrv12.py \
  cxr_real_oof_cleanlab_smoke.py \
  export_sample_topk_entries_for_llm.py \
  run_entry_level_llm_review.py \
  select_topk_issue_samples.py \
  summarize_xrv_loop_metrics.py; do
  if ! cmp -s "${SCRIPT_DIR}/${frozen_name}" "${FROZEN_DIR}/${frozen_name}"; then
    echo "Live execution file differs from frozen v5: ${frozen_name}" >&2
    exit 3
  fi
done
verify_sha256 "${EXPECTED_PIPELINE_SHA256}" "${PIPELINE}"
verify_sha256 "${EXPECTED_SELECTOR_SHA256}" "${SELECTOR}"
verify_sha256 "${EXPECTED_SELECTOR_TEST_SHA256}" "${SELECTOR_TEST}"
verify_sha256 "${EXPECTED_PROTOCOL_SHA256}" "${PROTOCOL}"

source /vol/gpudata/yz3522-llmtest/venv/bin/activate
python "${SELECTOR_TEST}"

source ~/.llm_review_env
if [[ "${OPENAI_MODEL:-}" != "${EXPECTED_OPENAI_MODEL}" ]]; then
  echo "OPENAI_MODEL must be ${EXPECTED_OPENAI_MODEL}; found ${OPENAI_MODEL:-unset}." >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/state"
for loop_id in $(seq 1 8); do
  loop_pad="$(printf '%02d' "${loop_id}")"
  source_loop="${SOURCE_RUN_ROOT}/loop_${loop_pad}"
  target_loop="${RUN_ROOT}/loop_${loop_pad}"
  if [[ ! -d "${source_loop}" ]]; then
    echo "Missing completed source loop: ${source_loop}" >&2
    exit 4
  fi
  if [[ -L "${target_loop}" ]]; then
    if [[ "$(readlink -f "${target_loop}")" != "$(readlink -f "${source_loop}")" ]]; then
      echo "Existing loop link points to the wrong source: ${target_loop}" >&2
      exit 4
    fi
  elif [[ -e "${target_loop}" ]]; then
    echo "Refusing to replace existing non-symlink loop: ${target_loop}" >&2
    exit 4
  else
    ln -s "${source_loop}" "${target_loop}"
  fi
done

if [[ ! -s "${RUN_ROOT}/loop_metrics.csv" ]]; then
  cp "${SOURCE_RUN_ROOT}/loop_metrics.csv" "${RUN_ROOT}/loop_metrics.csv"
fi
for state_name in \
  cumulative_removed_samples.csv \
  cumulative_relabel_entries.csv \
  cumulative_mask_entries.csv; do
  if [[ ! -s "${RUN_ROOT}/state/${state_name}" ]]; then
    cp "${SOURCE_RUN_ROOT}/state/${state_name}" "${RUN_ROOT}/state/${state_name}"
  fi
done

/vol/gpudata/yz3522-llmtest/venv/bin/python - "${RUN_ROOT}" <<'PY'
from pathlib import Path
import json
import sys

import pandas as pd

run_root = Path(sys.argv[1])
metrics = pd.read_csv(run_root / "loop_metrics.csv")
observed = pd.to_numeric(metrics["loop_id"], errors="raise").astype(int).tolist()
if not observed or observed != list(range(1, max(observed) + 1)):
    raise SystemExit(f"Completed loops are not contiguous from Loop 1: {observed}")
if max(observed) < 8 or max(observed) > 15:
    raise SystemExit(f"Expected a resumable endpoint between Loop 8 and 15: {observed}")

last_loop = run_root / f"loop_{max(observed):02d}"
required = [
    last_loop / "sample_top_fraction_expanded_entries.csv",
    last_loop / "applied_relabel_entries.csv",
    last_loop / "applied_mask_entries.csv",
    last_loop / "train_eval" / "baseline_run_summary.csv",
    last_loop / "train_eval" / "test_study_predictions.csv",
]
missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise SystemExit(f"Last completed transaction is incomplete: {missing}")

for loop_id in range(9, max(observed) + 1):
    audit_path = run_root / f"loop_{loop_id:02d}" / "unseen_selection_audit.json"
    if not audit_path.is_file():
        raise SystemExit(f"Missing unseen-selection audit for completed Loop {loop_id}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit["full_budget_met"] or int(audit["history_overlap_entries"]) != 0:
        raise SystemExit(f"Invalid unseen-selection audit for completed Loop {loop_id}")

print(
    f"Validated seed 42 state through Loop {max(observed)}; "
    f"next loop is {max(observed) + 1}."
)
PY

if [[ "${1:-}" == "--preflight-only" ]]; then
  echo "Seed42 unseen-review extension preflight passed."
  exit 0
fi

echo "=========================================================="
echo "Pre-registered unseen-review refinement extension"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Seed: ${SEED} (smallest current analysis seed other than seed13)"
echo "Source completed state: Loop 8 corrected-binary v5 refinement"
echo "Selection: full sample top20 budget, globally unseen entries only"
echo "History unit: pool_row_id::label_index across every prior loop"
echo "Maximum loop: ${LOOP_COUNT}"
echo "LLM model: ${OPENAI_MODEL}"
echo "Protocol: ${PROTOCOL}"
echo "Run root: ${RUN_ROOT}"
echo "Started: $(date --iso-8601=seconds)"
echo "=========================================================="

export MODEL_BACKBONE_OVERRIDE="mobilenet_v3_small_scratch"
export MODEL_TAG_OVERRIDE="mobilenet_v3_small_scratch_noes"
export OOF_BATCH_SIZE_OVERRIDE="32"
export TRAIN_BATCH_SIZE_OVERRIDE="32"
export OOF_EPOCHS_OVERRIDE="100"
export TRAIN_EPOCHS_OVERRIDE="50"
export TRAIN_EARLY_STOPPING_PATIENCE_OVERRIDE="100000"
export TRAIN_RECOVER_BEST_WEIGHTS_OVERRIDE="0"
export BATCH_SIZE_OVERRIDE="100"
export CONCURRENCY_OVERRIDE="10"
export RETRY_LIMIT_OVERRIDE="2"
export LLM_RESUME_PASSES_OVERRIDE="2"
export REQUEST_TIMEOUT_OVERRIDE="180"
export MAX_COMPLETION_TOKENS_OVERRIDE="400"
export UNSEEN_REVIEW_ONLY_OVERRIDE="1"

exec bash "${PIPELINE}" \
  llm_refine "${SEED_ROOT}" "${LOOP_COUNT}" "${TOP_FRACTION}" "${SEED}" 1 "${LOOP_COUNT}"
