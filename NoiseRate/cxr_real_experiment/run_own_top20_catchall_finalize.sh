#!/bin/bash
#SBATCH --job-name=own-ref-final
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/own_ref_finalize_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/own_ref_finalize_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
NOISERATE_DIR="${PROJECT_DIR}/NoiseRate"
SCRIPT_DIR="${NOISERATE_DIR}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
REFINE_ROOT="${1:-/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_own_top20_binary_v5_llm_refine_5seed/20260714_065539}"
OUT_DIR="${REFINE_ROOT}/evaluation_prelocked_loop5_loop8"
FROZEN_DIR="${REFINE_ROOT}/protocol_snapshot_v5_binary_label_locked"

mkdir -p "${PROJECT_DIR}/slurm_logs"

echo "=========================================================="
echo "Own-top20 catch-all continuation and final evaluation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Refine root: ${REFINE_ROOT}"
echo "Started: $(date --iso-8601=seconds)"
echo "=========================================================="

if ! (cd "${FROZEN_DIR}" && sha256sum --check checksums.sha256); then
  echo "Frozen v5 snapshot checksum verification failed." >&2
  exit 3
fi
for frozen_name in \
  build_llm_refinement_tables.py \
  build_evaluation_followup_deck_20260713.py \
  cxr_real_full_train_eval_cleanlab_xrv12.py \
  cxr_real_noise_validation_smoke.py \
  cxr_real_oof_cleanlab_smoke.py \
  evaluate_mobilenet_data_quality_5seed.py \
  evaluate_own_top20_refinement_endpoints.py \
  evaluate_own_top20_refinement_quality.py \
  export_sample_topk_entries_for_llm.py \
  prepare_own_top20_binary_v5.py \
  run_entry_level_llm_review.py \
  run_mobilenetv3_own_top20_llm_refine_seed.sh \
  run_own_top20_catchall_finalize.sh \
  run_own_top20_followup_evaluation.sh \
  run_xrv_iterative_sample20_branch.sh \
  select_topk_issue_samples.py \
  summarize_xrv_loop_metrics.py \
  test_binary_label_refinement_pipeline.py; do
  if ! cmp -s "${SCRIPT_DIR}/${frozen_name}" "${FROZEN_DIR}/${frozen_name}"; then
    echo "Live protocol file differs from the frozen snapshot: ${frozen_name}" >&2
    exit 3
  fi
done
if ! cmp -s \
  "${SCRIPT_DIR}/evaluation_followup_20260713/own_top20_refinement_protocol.md" \
  "${FROZEN_DIR}/own_top20_refinement_protocol.md"; then
  echo "Live own-top20 protocol differs from the frozen v5 snapshot." >&2
  exit 3
fi
echo "Live protocol files match the frozen snapshot."

seed_loop8_complete() {
  local seed="$1"
  local branch="${REFINE_ROOT}/seed_${seed}/llm_refine"
  local loop="${branch}/loop_08"
  local required
  for required in \
    "${loop}/.oof_complete" \
    "${loop}/.selection_complete" \
    "${loop}/.entry_expansion_complete" \
    "${loop}/.llm_review_complete" \
    "${loop}/.action_tables_complete" \
    "${loop}/.train_eval_complete"; do
    [[ -f "${required}" ]] || return 1
  done
  for required in \
    "${loop}/llm_review/results.csv" \
    "${loop}/applied_relabel_entries.csv" \
    "${loop}/applied_mask_entries.csv" \
    "${loop}/train_eval/baseline_run_summary.csv" \
    "${loop}/train_eval/test_study_auroc_summary.csv" \
    "${loop}/train_eval/test_study_predictions.csv"; do
    [[ -s "${required}" ]] || return 1
  done
  "${PYTHON}" - "${branch}/loop_metrics.csv" <<'PY'
from pathlib import Path
import sys
import pandas as pd

path = Path(sys.argv[1])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(1)
frame = pd.read_csv(path)
if "loop_id" not in frame or not frame["loop_id"].astype(int).eq(8).any():
    raise SystemExit(1)
PY
}

for seed in 7 13 42 97 123; do
  if seed_loop8_complete "${seed}"; then
    echo "Seed ${seed}: complete Loop 8 transaction already present; skipping continuation."
  else
    echo "Seed ${seed}: Loop 8 transaction incomplete; resuming the frozen own-top20 pipeline."
    bash "${SCRIPT_DIR}/run_mobilenetv3_own_top20_llm_refine_seed.sh" \
      "${seed}" \
      "${REFINE_ROOT}"
  fi
done

echo "All five Loop 8 prediction files are present; running pre-locked evaluation."
bash "${SCRIPT_DIR}/run_own_top20_followup_evaluation.sh" \
  "${REFINE_ROOT}" \
  "${OUT_DIR}"

echo "Rebuilding the follow-up deck from the pre-locked result CSVs."
cd "${NOISERATE_DIR}"
export OWN_TOP20_ROOT="${REFINE_ROOT}"
"${PYTHON}" "${SCRIPT_DIR}/build_evaluation_followup_deck_20260713.py"

"${PYTHON}" - "${OUT_DIR}" "${SCRIPT_DIR}/evaluation_followup_deck_20260713/postflight_checks.json" <<'PY'
from pathlib import Path
import json
import sys
import pandas as pd

evaluation_dir = Path(sys.argv[1])
postflight_path = Path(sys.argv[2])
required = [
    evaluation_dir / "performance" / "prelocked_seed_paired_statistics.csv",
    evaluation_dir / "performance" / "prelocked_hierarchical_bootstrap.csv",
    evaluation_dir / "quality" / "own_top20_quality_summary.csv",
    evaluation_dir / "quality" / "own_top20_quality_per_seed.csv",
    evaluation_dir / "quality" / "own_top20_quality_metadata.json",
    postflight_path,
]
missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise SystemExit(f"Finalizer missing required outputs: {missing}")
postflight = json.loads(postflight_path.read_text(encoding="utf-8"))
if postflight.get("status") != "passed":
    raise SystemExit(f"Deck postflight did not pass: {postflight}")
quality = pd.read_csv(evaluation_dir / "quality" / "own_top20_quality_per_seed.csv")
if "n_evaluable_samples" not in quality:
    raise SystemExit("Final quality output lacks the corrected evaluable-sample denominator")
quality_metadata = json.loads(
    (evaluation_dir / "quality" / "own_top20_quality_metadata.json").read_text(
        encoding="utf-8"
    )
)
if quality_metadata.get("sample_zero_valid_policy") != (
    "samples with zero valid target labels are unevaluable, not issue-free"
):
    raise SystemExit("Final quality metadata lacks the zero-valid sample policy")
print("Final evaluation and deck verification passed.")
PY

echo "Completed: $(date --iso-8601=seconds)"
