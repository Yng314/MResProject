#!/bin/bash
#SBATCH --job-name=xrv-unseen-s20
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/xrv_unseen_s20_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/xrv_unseen_s20_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00

set -euo pipefail

BRANCH="${1:?branch must be remove_only or llm_refine}"
OUTPUT_ROOT="${2:?output_root must be provided}"
LOOP_COUNT="${3:-5}"
TOP_FRACTION="${4:-0.20}"
SEED="${5:-13}"
START_LOOP="${6:-1}"
END_LOOP="${7:-${LOOP_COUNT}}"
MODEL_BACKBONE="${MODEL_BACKBONE_OVERRIDE:-xrv_densenet121_linearhead}"
MODEL_TAG="${MODEL_TAG_OVERRIDE:-${MODEL_BACKBONE}}"
OOF_BATCH_SIZE="${OOF_BATCH_SIZE_OVERRIDE:-16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE_OVERRIDE:-16}"
OOF_EPOCHS="${OOF_EPOCHS_OVERRIDE:-100}"
TRAIN_EPOCHS="${TRAIN_EPOCHS_OVERRIDE:-100}"
LEARNING_RATE="${LR_OVERRIDE:-0.001}"
TRAIN_EARLY_STOPPING_PATIENCE="${TRAIN_EARLY_STOPPING_PATIENCE_OVERRIDE:-10}"
TRAIN_RECOVER_BEST_WEIGHTS="${TRAIN_RECOVER_BEST_WEIGHTS_OVERRIDE:-1}"
TRAIN_DIR_BASENAME="${TRAIN_DIR_BASENAME_OVERRIDE:-train_eval}"

if [[ "${BRANCH}" != "remove_only" && "${BRANCH}" != "llm_refine" ]]; then
  echo "Invalid branch: ${BRANCH}" >&2
  exit 2
fi

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_DIR}/NoiseRate/cxr_real_experiment"
OOF_SCRIPT="${SCRIPT_DIR}/cxr_real_oof_cleanlab_smoke.py"
TRAIN_SCRIPT="${SCRIPT_DIR}/cxr_real_full_train_eval_cleanlab_xrv12.py"
SELECT_SCRIPT="${SCRIPT_DIR}/select_topk_issue_samples.py"
UNSEEN_SELECT_SCRIPT="${SCRIPT_DIR}/select_topk_unreviewed_issue_samples.py"
EXPAND_SCRIPT="${SCRIPT_DIR}/export_sample_topk_entries_for_llm.py"
LLM_SCRIPT="${SCRIPT_DIR}/run_entry_level_llm_review.py"
SHARDED_LLM_SCRIPT="${SCRIPT_DIR}/run_entry_level_llm_review_sharded.py"
BUILD_REFINE_SCRIPT="${SCRIPT_DIR}/build_llm_refinement_tables.py"
SUMMARY_SCRIPT="${SCRIPT_DIR}/summarize_xrv_loop_metrics.py"
DATA_ROOT="${PROJECT_DIR}/MedSoul/datasets"
IMAGE_ROOT="${DATA_ROOT}/mimic-cxr-jpg-224"
CHEXPERT_CSV="${IMAGE_ROOT}/mimic-cxr-2.0.0-chexpert.csv"
SPLIT_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-split.csv.gz"
TEST_CSV="${DATA_ROOT}/mimic-cxr-2.1.0-test-set-labeled.csv"
METADATA_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz"
REPORT_ARCHIVE="${DATA_ROOT}/mimic-cxr-reports.tar.gz"
RUN_ROOT="${OUTPUT_ROOT}/${BRANCH}"
STATE_DIR="${RUN_ROOT}/state"
METRICS_CSV="${RUN_ROOT}/loop_metrics.csv"
CUM_SAMPLE_CSV="${STATE_DIR}/cumulative_removed_samples.csv"
CUM_RELABEL_CSV="${STATE_DIR}/cumulative_relabel_entries.csv"
CUM_MASK_CSV="${STATE_DIR}/cumulative_mask_entries.csv"
UNSEEN_REVIEW_ONLY="${UNSEEN_REVIEW_ONLY_OVERRIDE:-1}"
FULL_ISSUE_POOL="${FULL_ISSUE_POOL_OVERRIDE:-0}"
SUCCESSFUL_HISTORY_ONLY="${SUCCESSFUL_HISTORY_ONLY_OVERRIDE:-0}"
REQUIRE_COMPLETE_LLM_REVIEW="${REQUIRE_COMPLETE_LLM_REVIEW_OVERRIDE:-0}"

mkdir -p "${PROJECT_DIR}/slurm_logs" "${RUN_ROOT}" "${STATE_DIR}"
cd "${PROJECT_DIR}" || exit 1

source /vol/cuda/12.5.0/setup.sh
source /vol/gpudata/yz3522-llmtest/venv/bin/activate
if [[ "${BRANCH}" == "llm_refine" ]]; then
  source ~/.llm_review_env
fi

export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}" "/vol/gpudata/yz3522-llmtest/.cache/torchxrayvision/models_data"

next_loop_to_run() {
  python - "$METRICS_CSV" "$LOOP_COUNT" <<'PY'
from pathlib import Path
import sys
import pandas as pd

metrics_path = Path(sys.argv[1])
loop_count = int(sys.argv[2])
if not metrics_path.exists() or metrics_path.stat().st_size == 0:
    print(1)
    raise SystemExit

df = pd.read_csv(metrics_path)
if df.empty or "loop_id" not in df.columns:
    print(1)
    raise SystemExit

completed = set(pd.to_numeric(df["loop_id"], errors="coerce").dropna().astype(int))
for loop_id in range(1, loop_count + 1):
    if loop_id not in completed:
        print(loop_id)
        break
else:
    print(loop_count + 1)
PY
}

all_files_nonempty() {
  local path
  for path in "$@"; do
    if [[ ! -s "${path}" ]]; then
      return 1
    fi
  done
}

sync_state_from_completed_loop() {
  local loop_id="$1"
  local loop_pad
  local loop_dir
  if (( loop_id < 1 )); then
    return 0
  fi
  loop_pad="$(printf "%02d" "${loop_id}")"
  loop_dir="${RUN_ROOT}/loop_${loop_pad}"
  if [[ "${BRANCH}" == "remove_only" ]]; then
    if [[ ! -s "${loop_dir}/applied_removed_samples.csv" ]]; then
      echo "Completed loop ${loop_id} is missing applied_removed_samples.csv" >&2
      exit 3
    fi
    cp "${loop_dir}/applied_removed_samples.csv" "${CUM_SAMPLE_CSV}"
  else
    if ! all_files_nonempty \
      "${loop_dir}/applied_relabel_entries.csv" \
      "${loop_dir}/applied_mask_entries.csv"; then
      echo "Completed loop ${loop_id} is missing cumulative refinement tables" >&2
      exit 3
    fi
    cp "${loop_dir}/applied_relabel_entries.csv" "${CUM_RELABEL_CSV}"
    cp "${loop_dir}/applied_mask_entries.csv" "${CUM_MASK_CSV}"
  fi
  echo "Synchronized cumulative state from completed loop ${loop_id}."
}

if [[ ! -f "${CUM_SAMPLE_CSV}" || ! -f "${CUM_RELABEL_CSV}" || ! -f "${CUM_MASK_CSV}" ]]; then
python - <<PY
from pathlib import Path
import pandas as pd
pd.DataFrame(columns=["pool_row_id", "est_issue_sample"]).to_csv(Path(r'''${CUM_SAMPLE_CSV}'''), index=False)
pd.DataFrame(columns=["pool_row_id", "label_index", "label_name", "old_raw_label", "old_binary_label", "new_binary_label", "new_raw_label", "valid_label", "mismatch_type", "recommended_action", "response_consistent", "action_resolution", "action_label_basis", "sample_selected_rank", "study_id"]).to_csv(Path(r'''${CUM_RELABEL_CSV}'''), index=False)
pd.DataFrame(columns=["pool_row_id", "label_index", "label_name", "raw_label", "binary_label", "valid_label", "mismatch_type", "recommended_action", "response_consistent", "action_resolution", "action_label_basis", "sample_selected_rank", "study_id", "est_issue_entry"]).to_csv(Path(r'''${CUM_MASK_CSV}'''), index=False)
PY
fi

echo "=========================================================="
echo "Iterative sample20 branch"
echo "=========================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Branch: ${BRANCH}"
echo "Loop count: ${LOOP_COUNT}"
echo "Start loop: ${START_LOOP}"
echo "End loop: ${END_LOOP}"
echo "Top fraction: ${TOP_FRACTION}"
echo "Seed: ${SEED}"
echo "Run root: ${RUN_ROOT}"
echo "Current Slurm job: ${SLURM_JOB_ID}"
echo "Model backbone: ${MODEL_BACKBONE}"
echo "Model tag: ${MODEL_TAG}"
echo "Train epochs: ${TRAIN_EPOCHS}"
echo "Train early stopping patience: ${TRAIN_EARLY_STOPPING_PATIENCE}"
echo "Train recover best weights: ${TRAIN_RECOVER_BEST_WEIGHTS}"
echo "Unseen-review-only selection: ${UNSEEN_REVIEW_ONLY}"
echo "Full issue-pool selection: ${FULL_ISSUE_POOL}"
echo "Successful-result history only: ${SUCCESSFUL_HISTORY_ONLY}"
echo "Require zero residual LLM errors: ${REQUIRE_COMPLETE_LLM_REVIEW}"
echo "LLM review shards: ${LLM_REVIEW_SHARDS_OVERRIDE:-1}"
echo "LLM total concurrency: ${CONCURRENCY_OVERRIDE:-10}"
echo "Model: ${OPENAI_MODEL:-not_used}"
echo "Started: $(date)"
echo "=========================================================="

AUTO_START_LOOP="$(next_loop_to_run)"
if (( AUTO_START_LOOP > LOOP_COUNT )); then
  echo "All ${LOOP_COUNT} loops already completed for ${BRANCH}."
  exit 0
fi

if [[ "${START_LOOP}" == "1" && "${END_LOOP}" == "${LOOP_COUNT}" ]]; then
  START_LOOP="${AUTO_START_LOOP}"
  END_LOOP="${LOOP_COUNT}"
fi

echo "Resolved loop range: ${START_LOOP} -> ${END_LOOP}"
sync_state_from_completed_loop "$((AUTO_START_LOOP - 1))"

for LOOP_ID in $(seq "${START_LOOP}" "${END_LOOP}"); do
  LOOP_PAD="$(printf "%02d" "${LOOP_ID}")"
  LOOP_DIR="${RUN_ROOT}/loop_${LOOP_PAD}"
  OOF_DIR="${LOOP_DIR}/oof"
  SELECTED_SAMPLE_CSV="${LOOP_DIR}/sample_top_fraction_issue_subset.csv"
  TRAIN_DIR="${LOOP_DIR}/${TRAIN_DIR_BASENAME}"
  mkdir -p "${OOF_DIR}" "${TRAIN_DIR}"
  APPLIED_SAMPLE_CSV="${LOOP_DIR}/applied_removed_samples.csv"
  APPLIED_RELABEL_CSV="${LOOP_DIR}/applied_relabel_entries.csv"
  APPLIED_MASK_CSV="${LOOP_DIR}/applied_mask_entries.csv"
  OOF_COMPLETE_MARKER="${LOOP_DIR}/.oof_complete"
  SELECTION_COMPLETE_MARKER="${LOOP_DIR}/.selection_complete"
  ACTION_COMPLETE_MARKER="${LOOP_DIR}/.action_tables_complete"
  TRAIN_COMPLETE_MARKER="${LOOP_DIR}/.${TRAIN_DIR_BASENAME}_complete"

  echo ""
  echo "================ Loop ${LOOP_ID}/${LOOP_COUNT}: OOF ================="
  OOF_ARGS=(
    --image-root "${IMAGE_ROOT}"
    --chexpert-csv "${CHEXPERT_CSV}"
    --split-csv "${SPLIT_CSV}"
    --metadata-csv "${METADATA_CSV}"
    --allowed-views AP PA
    --train-split train
    --num-samples -1
    --n-splits 4
    --epochs "${OOF_EPOCHS}"
    --batch-size "${OOF_BATCH_SIZE}"
    --image-size 224
    --learning-rate "${LEARNING_RATE}"
    --early-stopping-patience 10
    --model-backbone "${MODEL_BACKBONE}"
    --xrv-weights densenet121-res224-all
    --xrv-cache-dir /vol/gpudata/yz3522-llmtest/.cache/torchxrayvision/models_data
    --device cuda
    --seed "${SEED}"
    --num-workers 4
    --output-dir "${OOF_DIR}"
  )
  if [[ "${BRANCH}" == "remove_only" ]]; then
    OOF_ARGS+=(--exclude-sample-csv "${CUM_SAMPLE_CSV}" --exclude-issue-col est_issue_sample)
  else
    OOF_ARGS+=(--override-entry-csv "${CUM_RELABEL_CSV}" --exclude-entry-csv "${CUM_MASK_CSV}" --exclude-entry-issue-col est_issue_entry)
  fi
  if all_files_nonempty \
    "${OOF_DIR}/oof_cleanlab_smoke_summary.csv" \
    "${OOF_DIR}/train_cleanlab_sample_issues_only.csv"; then
    echo "OOF stage already complete; reusing ${OOF_DIR}."
    touch "${OOF_COMPLETE_MARKER}"
  else
    if [[ -L "${OOF_DIR}" ]]; then
      echo "Reused OOF symlink is incomplete; refusing to overwrite its source: ${OOF_DIR}" >&2
      exit 4
    fi
    python -u "${OOF_SCRIPT}" "${OOF_ARGS[@]}"
    if ! all_files_nonempty \
      "${OOF_DIR}/oof_cleanlab_smoke_summary.csv" \
      "${OOF_DIR}/train_cleanlab_sample_issues_only.csv"; then
      echo "OOF command returned without complete required outputs." >&2
      exit 4
    fi
    touch "${OOF_COMPLETE_MARKER}"
  fi

  echo ""
  echo "================ Loop ${LOOP_ID}/${LOOP_COUNT}: select sample top fraction ================="
  UNSEEN_SELECTION_AUDIT="${LOOP_DIR}/unseen_selection_audit.json"
  if [[ -s "${SELECTED_SAMPLE_CSV}" ]]; then
    echo "Selection stage already complete; reusing ${SELECTED_SAMPLE_CSV}."
    touch "${SELECTION_COMPLETE_MARKER}"
  else
    if [[ -L "${SELECTED_SAMPLE_CSV}" ]]; then
      echo "Reused selection symlink is incomplete; refusing to overwrite its source: ${SELECTED_SAMPLE_CSV}" >&2
      exit 4
    fi
    if [[ "${BRANCH}" == "llm_refine" && "${UNSEEN_REVIEW_ONLY}" == "1" ]]; then
      UNSEEN_SELECT_ARGS=(
        --input-csv "${OOF_DIR}/train_cleanlab_sample_issues_only.csv"
        --output-csv "${SELECTED_SAMPLE_CSV}"
        --audit-json "${UNSEEN_SELECTION_AUDIT}"
        --history-run-root "${RUN_ROOT}"
        --current-loop "${LOOP_ID}"
        --top-fraction "${TOP_FRACTION}"
      )
      if [[ "${SUCCESSFUL_HISTORY_ONLY}" == "1" ]]; then
        UNSEEN_SELECT_ARGS+=(--history-source successful-results)
      fi
      if [[ "${FULL_ISSUE_POOL}" == "1" ]]; then
        UNSEEN_SELECT_ARGS+=(--allow-budget-shortfall)
      fi
      python -u "${UNSEEN_SELECT_SCRIPT}" "${UNSEEN_SELECT_ARGS[@]}"
    else
      python -u "${SELECT_SCRIPT}" \
        --input-csv "${OOF_DIR}/train_cleanlab_sample_issues_only.csv" \
        --output-csv "${SELECTED_SAMPLE_CSV}" \
        --top-fraction "${TOP_FRACTION}"
    fi
    if [[ ! -s "${SELECTED_SAMPLE_CSV}" ]]; then
      echo "Selection command returned without a non-empty output." >&2
      exit 4
    fi
    touch "${SELECTION_COMPLETE_MARKER}"
  fi

  LLM_RESULTS_CSV=""
  LLM_ERRORS_CSV=""
  SUMMARY_SAMPLE_CSV="${CUM_SAMPLE_CSV}"
  SUMMARY_RELABEL_CSV="${CUM_RELABEL_CSV}"
  SUMMARY_MASK_CSV="${CUM_MASK_CSV}"
  if [[ "${BRANCH}" == "remove_only" ]]; then
    if [[ -f "${ACTION_COMPLETE_MARKER}" ]] && [[ -s "${APPLIED_SAMPLE_CSV}" ]]; then
      echo "Removal action table already complete; reusing ${APPLIED_SAMPLE_CSV}."
    else
    python - <<PY
from pathlib import Path
import pandas as pd
cum_path = Path(r'''${CUM_SAMPLE_CSV}''')
new_path = Path(r'''${SELECTED_SAMPLE_CSV}''')
applied_path = Path(r'''${APPLIED_SAMPLE_CSV}''')
cum = pd.read_csv(cum_path)
new = pd.read_csv(new_path)
out = pd.concat([cum, new], ignore_index=True)
out = out.drop_duplicates(subset=["pool_row_id"], keep="first")
out["est_issue_sample"] = 1
out.to_csv(applied_path, index=False)
print(f"applied removed samples for this loop: {out['pool_row_id'].nunique()}")
PY
      if [[ ! -s "${APPLIED_SAMPLE_CSV}" ]]; then
        echo "Removal action-table stage returned without a non-empty output." >&2
        exit 4
      fi
      touch "${ACTION_COMPLETE_MARKER}"
    fi
    SUMMARY_SAMPLE_CSV="${APPLIED_SAMPLE_CSV}"
  else
    EXPANDED_CSV="${LOOP_DIR}/sample_top_fraction_expanded_entries.csv"
    LLM_DIR="${LOOP_DIR}/llm_review"
    EXPANSION_COMPLETE_MARKER="${LOOP_DIR}/.entry_expansion_complete"
    LLM_COMPLETE_MARKER="${LOOP_DIR}/.llm_review_complete"
    RELABEL_THIS_LOOP="${LOOP_DIR}/llm_relabel_entries.csv"
    MASK_THIS_LOOP="${LOOP_DIR}/llm_mask_entries.csv"
    REFINE_SUMMARY_THIS_LOOP="${LOOP_DIR}/llm_refinement_summary.csv"
    mkdir -p "${LLM_DIR}"

    echo ""
    echo "================ Loop ${LOOP_ID}/${LOOP_COUNT}: LLM review ================="
    LLM_RESULTS_CSV="${LLM_DIR}/results.csv"
    LLM_ERRORS_CSV="${LLM_DIR}/errors.csv"
    if [[ -f "${EXPANSION_COMPLETE_MARKER}" ]] && [[ -s "${EXPANDED_CSV}" ]]; then
      echo "Entry expansion already complete; reusing ${EXPANDED_CSV}."
    else
      python -u "${EXPAND_SCRIPT}" \
        --sample-issues-csv "${SELECTED_SAMPLE_CSV}" \
        --output-csv "${EXPANDED_CSV}" \
        --top-n-samples 100000000
      if [[ ! -s "${EXPANDED_CSV}" ]]; then
        echo "Entry expansion returned without a non-empty output." >&2
        exit 4
      fi
      touch "${EXPANSION_COMPLETE_MARKER}"
    fi

    python - "${SELECTED_SAMPLE_CSV}" "${EXPANDED_CSV}" <<'PY'
from pathlib import Path
import sys
import pandas as pd

selected_path = Path(sys.argv[1])
expanded_path = Path(sys.argv[2])
selected = pd.read_csv(selected_path, usecols=["pool_row_id"])
expanded = pd.read_csv(
    expanded_path,
    usecols=[
        "pool_row_id",
        "label_index",
        "raw_label",
        "binary_label",
        "valid_label",
        "est_issue_entry",
        "entry_key",
    ],
)
if selected.empty or selected["pool_row_id"].duplicated().any():
    raise SystemExit("Selected sample table is empty or has duplicate pool_row_id values")
if expanded.empty or expanded["entry_key"].astype(str).duplicated().any():
    raise SystemExit("Binary expansion is empty or has duplicate entry keys")
selected_ids = set(selected["pool_row_id"].astype(int))
expanded_ids = set(expanded["pool_row_id"].astype(int))
if expanded_ids != selected_ids:
    raise SystemExit(
        f"Binary expansion sample coverage mismatch: {len(expanded_ids)}/{len(selected_ids)}"
    )
if not expanded["valid_label"].eq(1).all() or not expanded["est_issue_entry"].eq(1).all():
    raise SystemExit("Binary expansion contains invalid or unflagged review entries")
if not expanded["raw_label"].isin([-1.0, 0.0, 1.0]).all():
    raise SystemExit("Binary expansion contains unsupported raw labels")
expected_binary = expanded["raw_label"].map({-1.0: 1.0, 0.0: 0.0, 1.0: 1.0})
if not expanded["binary_label"].eq(expected_binary).all():
    raise SystemExit("Binary expansion violates the fixed U-Ones projection")
print(
    f"Validated binary expansion: {len(expanded)} entries, "
    f"{len(expanded_ids)}/{len(selected_ids)} selected samples, "
    f"raw_-1={int(expanded['raw_label'].eq(-1.0).sum())}."
)
PY

    if [[ "${UNSEEN_REVIEW_ONLY}" == "1" ]]; then
      python - "${RUN_ROOT}" "${LOOP_ID}" "${EXPANDED_CSV}" "${UNSEEN_SELECTION_AUDIT}" <<'PY'
from pathlib import Path
import json
import sys
import pandas as pd

run_root = Path(sys.argv[1])
current_loop = int(sys.argv[2])
expanded_path = Path(sys.argv[3])
audit_path = Path(sys.argv[4])
successful_history_only = r'''${SUCCESSFUL_HISTORY_ONLY}''' == "1"

history = set()
for loop_id in range(1, current_loop):
    loop_dir = run_root / f"loop_{loop_id:02d}"
    path = (
        loop_dir / "llm_review" / "results.csv"
        if successful_history_only
        else loop_dir / "sample_top_fraction_expanded_entries.csv"
    )
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Missing prior review history: {path}")
    history.update(pd.read_csv(path, usecols=["entry_key"])["entry_key"].astype(str))

current = pd.read_csv(expanded_path, usecols=["entry_key"])
keys = set(current["entry_key"].astype(str))
overlap = sorted(keys & history)
if overlap:
    raise SystemExit(
        f"Unseen-review invariant failed: {len(overlap)} prior keys were selected, "
        f"examples={overlap[:10]}"
    )
if not audit_path.is_file():
    raise SystemExit(f"Missing unseen-selection audit: {audit_path}")
audit = json.loads(audit_path.read_text(encoding="utf-8"))
if int(audit["history_overlap_entries"]) != 0:
    raise SystemExit("Unseen-selection audit reports nonzero history overlap")
if int(audit["unseen_entries_selected_for_review"]) != len(current):
    raise SystemExit(
        "Expanded review count differs from unseen-selection audit: "
        f"{len(current)} != {audit['unseen_entries_selected_for_review']}"
    )
print(
    f"Validated unseen-review invariant: current={len(current)}, "
    f"prior_unique={len(history)}, overlap=0."
)
PY
    fi

    if [[ -f "${LLM_COMPLETE_MARKER}" ]] && [[ -s "${LLM_RESULTS_CSV}" ]]; then
      echo "LLM review already complete; reusing ${LLM_RESULTS_CSV}."
    else
      for REVIEW_PASS in $(seq 1 "${LLM_RESUME_PASSES_OVERRIDE:-2}"); do
        echo "LLM review pass ${REVIEW_PASS}/${LLM_RESUME_PASSES_OVERRIDE:-2}."
        if (( ${LLM_REVIEW_SHARDS_OVERRIDE:-1} > 1 )); then
          python -u "${SHARDED_LLM_SCRIPT}" \
            --input-csv "${EXPANDED_CSV}" \
            --report-archive "${REPORT_ARCHIVE}" \
            --output-dir "${LLM_DIR}" \
            --review-script "${LLM_SCRIPT}" \
            --top-k 100000000 \
            --total-concurrency "${CONCURRENCY_OVERRIDE:-10}" \
            --shards "${LLM_REVIEW_SHARDS_OVERRIDE}" \
            --retry-limit "${RETRY_LIMIT_OVERRIDE:-2}" \
            --llm-mode real \
            --request-timeout-seconds "${REQUEST_TIMEOUT_OVERRIDE:-180}" \
            --temperature 0 \
            --max-completion-tokens "${MAX_COMPLETION_TOKENS_OVERRIDE:-400}"
        else
          python -u "${LLM_SCRIPT}" \
            --input-csv "${EXPANDED_CSV}" \
            --report-archive "${REPORT_ARCHIVE}" \
            --output-dir "${LLM_DIR}" \
            --top-k 100000000 \
            --batch-size "${BATCH_SIZE_OVERRIDE:-100}" \
            --concurrency "${CONCURRENCY_OVERRIDE:-10}" \
            --retry-limit "${RETRY_LIMIT_OVERRIDE:-2}" \
            --resume \
            --llm-mode real \
            --request-timeout-seconds "${REQUEST_TIMEOUT_OVERRIDE:-180}" \
            --temperature 0 \
            --max-completion-tokens "${MAX_COMPLETION_TOKENS_OVERRIDE:-400}"
        fi
        REMAINING_ERRORS="$(python - "${LLM_ERRORS_CSV}" <<'PY'
from pathlib import Path
import sys
import pandas as pd
path = Path(sys.argv[1])
if not path.exists() or path.stat().st_size == 0:
    print(0)
else:
    print(len(pd.read_csv(path)))
PY
)"
        echo "Residual LLM errors after pass ${REVIEW_PASS}: ${REMAINING_ERRORS}"
        if (( REMAINING_ERRORS == 0 )); then
          break
        fi
        if (( REVIEW_PASS < ${LLM_RESUME_PASSES_OVERRIDE:-2} )); then
          sleep 30
        fi
      done
      if [[ ! -s "${LLM_RESULTS_CSV}" ]]; then
        echo "LLM review returned without any successful result rows." >&2
        exit 4
      fi
      if [[ "${REQUIRE_COMPLETE_LLM_REVIEW}" == "1" ]] && (( REMAINING_ERRORS > 0 )); then
        echo "LLM review still has ${REMAINING_ERRORS} residual errors; resubmit to resume before training." >&2
        exit 5
      fi
      touch "${LLM_COMPLETE_MARKER}"
    fi

    if [[ -f "${ACTION_COMPLETE_MARKER}" ]] && all_files_nonempty \
      "${APPLIED_RELABEL_CSV}" \
      "${APPLIED_MASK_CSV}"; then
      echo "Refinement action tables already complete; reusing cumulative tables."
    else
      python -u "${BUILD_REFINE_SCRIPT}" \
        --llm-results-csv "${LLM_RESULTS_CSV}" \
        --relabel-csv "${RELABEL_THIS_LOOP}" \
        --mask-csv "${MASK_THIS_LOOP}" \
        --summary-csv "${REFINE_SUMMARY_THIS_LOOP}"

    python - <<PY
from pathlib import Path
import pandas as pd

def merge_unique(base_path, out_path, new_path, subset, defaults):
    base_path = Path(base_path)
    out_path = Path(out_path)
    new_path = Path(new_path)
    cum = pd.read_csv(base_path)
    new = pd.read_csv(new_path)
    out = pd.concat([cum, new], ignore_index=True)
    for col, value in defaults.items():
        if col not in out.columns:
            out[col] = value
        out[col] = out[col].fillna(value)
    if len(out):
        out = out.drop_duplicates(subset=subset, keep="last")
    out.to_csv(out_path, index=False)
    return len(out)

relabel_n = merge_unique(
    r'''${CUM_RELABEL_CSV}''',
    r'''${APPLIED_RELABEL_CSV}''',
    r'''${RELABEL_THIS_LOOP}''',
    ["pool_row_id", "label_index"],
    {},
)
mask_n = merge_unique(
    r'''${CUM_MASK_CSV}''',
    r'''${APPLIED_MASK_CSV}''',
    r'''${MASK_THIS_LOOP}''',
    ["pool_row_id", "label_index"],
    {"est_issue_entry": 1},
)
print(f"cumulative relabel entries: {relabel_n}")
print(f"cumulative mask entries: {mask_n}")
PY
      if ! all_files_nonempty \
        "${APPLIED_RELABEL_CSV}" \
        "${APPLIED_MASK_CSV}"; then
        echo "Refinement action-table stage returned without complete outputs." >&2
        exit 4
      fi
      touch "${ACTION_COMPLETE_MARKER}"
    fi
    SUMMARY_RELABEL_CSV="${APPLIED_RELABEL_CSV}"
    SUMMARY_MASK_CSV="${APPLIED_MASK_CSV}"
  fi

  echo ""
  echo "================ Loop ${LOOP_ID}/${LOOP_COUNT}: full train/eval ================="
  TRAIN_ARGS=(
    --image-root "${IMAGE_ROOT}"
    --chexpert-csv "${CHEXPERT_CSV}"
    --split-csv "${SPLIT_CSV}"
    --test-csv "${TEST_CSV}"
    --metadata-csv "${METADATA_CSV}"
    --allowed-views AP PA
    --train-split train
    --train-limit -1
    --test-limit -1
    --epochs "${TRAIN_EPOCHS}"
    --val-fraction 0.1
    --early-stopping-patience "${TRAIN_EARLY_STOPPING_PATIENCE}"
    --batch-size "${TRAIN_BATCH_SIZE}"
    --image-size 224
    --learning-rate "${LEARNING_RATE}"
    --num-workers 4
    --model-backbone "${MODEL_BACKBONE}"
    --xrv-weights densenet121-res224-all
    --xrv-cache-dir /vol/gpudata/yz3522-llmtest/.cache/torchxrayvision/models_data
    --device cuda
    --seed "${SEED}"
    --study-aggregation max
    --output-dir "${TRAIN_DIR}"
  )
  if [[ "${TRAIN_RECOVER_BEST_WEIGHTS}" == "1" ]]; then
    TRAIN_ARGS+=(--recover-best-weights)
  fi
  if [[ "${BRANCH}" == "remove_only" ]]; then
    TRAIN_ARGS+=(--exclude-sample-csv "${APPLIED_SAMPLE_CSV}" --exclude-issue-col est_issue_sample)
  else
    TRAIN_ARGS+=(--override-entry-csv "${APPLIED_RELABEL_CSV}" --exclude-entry-csv "${APPLIED_MASK_CSV}" --exclude-entry-issue-col est_issue_entry)
  fi
  if all_files_nonempty \
    "${TRAIN_DIR}/baseline_run_summary.csv" \
    "${TRAIN_DIR}/test_study_auroc_summary.csv"; then
    echo "Full train/eval already complete; reusing ${TRAIN_DIR}."
    touch "${TRAIN_COMPLETE_MARKER}"
  else
    python -u "${TRAIN_SCRIPT}" "${TRAIN_ARGS[@]}"
    if ! all_files_nonempty \
      "${TRAIN_DIR}/baseline_run_summary.csv" \
      "${TRAIN_DIR}/test_study_auroc_summary.csv"; then
      echo "Full train/eval returned without complete required outputs." >&2
      exit 4
    fi
    touch "${TRAIN_COMPLETE_MARKER}"
  fi

  SUMMARY_ARGS=(
    --branch "${BRANCH}"
    --loop-id "${LOOP_ID}"
    --top-fraction "${TOP_FRACTION}"
    --oof-dir "${OOF_DIR}"
    --selected-sample-csv "${SELECTED_SAMPLE_CSV}"
    --train-dir "${TRAIN_DIR}"
    --metrics-csv "${METRICS_CSV}"
    --cumulative-sample-csv "${SUMMARY_SAMPLE_CSV}"
    --cumulative-relabel-csv "${SUMMARY_RELABEL_CSV}"
    --cumulative-mask-csv "${SUMMARY_MASK_CSV}"
  )
  if [[ -n "${LLM_RESULTS_CSV}" ]]; then
    SUMMARY_ARGS+=(--llm-results-csv "${LLM_RESULTS_CSV}")
  fi
  if [[ -n "${LLM_ERRORS_CSV}" ]]; then
    SUMMARY_ARGS+=(--llm-errors-csv "${LLM_ERRORS_CSV}")
  fi
  python -u "${SUMMARY_SCRIPT}" "${SUMMARY_ARGS[@]}"

  if [[ "${BRANCH}" == "remove_only" ]]; then
    cp "${APPLIED_SAMPLE_CSV}" "${CUM_SAMPLE_CSV}"
  else
    cp "${APPLIED_RELABEL_CSV}" "${CUM_RELABEL_CSV}"
    cp "${APPLIED_MASK_CSV}" "${CUM_MASK_CSV}"
  fi
done

echo ""
echo "Completed at $(date)"
echo "Run root: ${RUN_ROOT}"
echo "Metrics: ${METRICS_CSV}"
