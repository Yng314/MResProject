#!/bin/bash
#SBATCH --job-name=mb-own-refine
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mb_own_refine_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mb_own_refine_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail

SEED="${1:?seed is required}"
EXPERIMENT_ROOT="${2:?experiment root is required}"
PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
NOISERATE_DIR="${PROJECT_DIR}/NoiseRate"
PIPELINE="${NOISERATE_DIR}/cxr_real_experiment/run_xrv_iterative_sample20_branch.sh"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
SOURCE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_noes50_clean_3seed/20260707_123320"
SOURCE_LOOP="${SOURCE_ROOT}/seed_${SEED}/sample20_remove_loop/remove_only/loop_01"
SEED_ROOT="${EXPERIMENT_ROOT}/seed_${SEED}"
TARGET_LOOP="${SEED_ROOT}/llm_refine/loop_01"
EXPECTED_OPENAI_MODEL="gpt-5.4"
FROZEN_DIR="${EXPERIMENT_ROOT}/protocol_snapshot_v5_binary_label_locked"

case "${SEED}" in
  7|13|42|97|123) ;;
  *)
    echo "Seed must be one of 7, 13, 42, 97, 123; received ${SEED}." >&2
    exit 2
    ;;
esac

if ! (cd "${FROZEN_DIR}" && sha256sum --check checksums.sha256); then
  echo "Frozen v5 snapshot checksum verification failed." >&2
  exit 3
fi
for frozen_name in \
  build_llm_refinement_tables.py \
  cxr_real_full_train_eval_cleanlab_xrv12.py \
  cxr_real_noise_validation_smoke.py \
  cxr_real_oof_cleanlab_smoke.py \
  export_sample_topk_entries_for_llm.py \
  run_entry_level_llm_review.py \
  run_mobilenetv3_own_top20_llm_refine_seed.sh \
  run_xrv_iterative_sample20_branch.sh \
  select_topk_issue_samples.py \
  summarize_xrv_loop_metrics.py; do
  if ! cmp -s "${NOISERATE_DIR}/cxr_real_experiment/${frozen_name}" "${FROZEN_DIR}/${frozen_name}"; then
    echo "Live execution file differs from the frozen v5 snapshot: ${frozen_name}" >&2
    exit 3
  fi
done
if ! cmp -s \
  "${NOISERATE_DIR}/cxr_real_experiment/evaluation_followup_20260713/own_top20_refinement_protocol.md" \
  "${FROZEN_DIR}/own_top20_refinement_protocol.md"; then
  echo "Live own-top20 protocol differs from the frozen v5 snapshot." >&2
  exit 3
fi

source ~/.llm_review_env
if [[ "${OPENAI_MODEL:-}" != "${EXPECTED_OPENAI_MODEL}" ]]; then
  echo "OPENAI_MODEL must remain ${EXPECTED_OPENAI_MODEL}; found ${OPENAI_MODEL:-unset}." >&2
  exit 2
fi

"${PYTHON}" - "${SOURCE_LOOP}" "${SEED}" <<'PY'
from pathlib import Path
import sys
import pandas as pd

loop_dir = Path(sys.argv[1])
seed = int(sys.argv[2])
summary_path = loop_dir / "oof" / "oof_cleanlab_smoke_summary.csv"
selected_path = loop_dir / "sample_top_fraction_issue_subset.csv"
if not summary_path.is_file() or not selected_path.is_file():
    raise SystemExit(f"Missing reusable Loop 1 source under {loop_dir}")
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
selected = pd.read_csv(selected_path, usecols=["pool_row_id"])
if selected.empty or selected["pool_row_id"].nunique() != len(selected):
    raise SystemExit("Reusable top20 selection is empty or contains duplicate samples")
print(f"Validated seed {seed} reusable Loop 1 OOF and {len(selected)} selected samples.")
PY

mkdir -p "${TARGET_LOOP}" "${PROJECT_DIR}/slurm_logs"

link_reused_stage() {
  local source="$1"
  local target="$2"
  if [[ -L "${target}" ]]; then
    if [[ "$(readlink -f "${target}")" != "$(readlink -f "${source}")" ]]; then
      echo "Existing symlink points to the wrong source: ${target}" >&2
      exit 3
    fi
  elif [[ -e "${target}" ]]; then
    echo "Refusing to replace existing non-symlink path: ${target}" >&2
    exit 3
  else
    ln -s "${source}" "${target}"
  fi
}

link_reused_stage "${SOURCE_LOOP}/oof" "${TARGET_LOOP}/oof"
link_reused_stage \
  "${SOURCE_LOOP}/sample_top_fraction_issue_subset.csv" \
  "${TARGET_LOOP}/sample_top_fraction_issue_subset.csv"

"${PYTHON}" - "${EXPERIMENT_ROOT}" "${SEED}" <<'PY'
from pathlib import Path
import json
import sys
import pandas as pd

root = Path(sys.argv[1])
seed = int(sys.argv[2])
manifest_path = root / "binary_v5_reuse_manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"Missing v5 preparation manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("protocol_version") != "own_top20_binary_v5":
    raise SystemExit(f"Unexpected preparation protocol: {manifest.get('protocol_version')}")
records = [row for row in manifest.get("seeds", []) if int(row.get("seed", -1)) == seed]
if len(records) != 1:
    raise SystemExit(f"Expected one preparation record for seed {seed}; found {len(records)}")
record = records[0]
if float(record["selected_sample_review_coverage"]) != 1.0:
    raise SystemExit(f"Seed {seed} does not have complete selected-sample review coverage")

loop = root / f"seed_{seed}" / "llm_refine" / "loop_01"
expanded = pd.read_csv(loop / "sample_top_fraction_expanded_entries.csv")
if len(expanded) != int(record["expanded_binary_entries"]):
    raise SystemExit(f"Seed {seed} expansion row count differs from its preparation manifest")
if expanded["entry_key"].astype(str).duplicated().any():
    raise SystemExit(f"Seed {seed} expansion contains duplicate entry keys")
expected_binary = expanded["raw_label"].map({-1.0: 1.0, 0.0: 0.0, 1.0: 1.0})
if not expanded["binary_label"].eq(expected_binary).all():
    raise SystemExit(f"Seed {seed} expansion violates the fixed U-Ones projection")

results_path = loop / "llm_review" / "results.csv"
if results_path.is_file() and results_path.stat().st_size:
    results = pd.read_csv(results_path)
    if results["entry_key"].astype(str).duplicated().any():
        raise SystemExit(f"Seed {seed} Loop1 results contain duplicate entry keys")
    expanded_by_key = expanded.assign(entry_key=expanded["entry_key"].astype(str)).set_index("entry_key")
    results = results.assign(entry_key=results["entry_key"].astype(str))
    if not set(results["entry_key"]).issubset(expanded_by_key.index):
        raise SystemExit(f"Seed {seed} Loop1 result keys are outside the corrected expansion")
    matched = expanded_by_key.loc[results["entry_key"]].reset_index()
    if not pd.to_numeric(results["raw_label"]).reset_index(drop=True).eq(
        pd.to_numeric(matched["raw_label"]).reset_index(drop=True)
    ).all():
        raise SystemExit(f"Seed {seed} Loop1 result raw labels differ from corrected expansion")
    if not pd.to_numeric(results["binary_label"]).reset_index(drop=True).eq(
        pd.to_numeric(matched["binary_label"]).reset_index(drop=True)
    ).all():
        raise SystemExit(f"Seed {seed} Loop1 result binary targets differ from corrected expansion")
print(
    f"Validated seed {seed} v5 preparation: expanded={len(expanded)}, "
    f"existing_results={len(results) if results_path.is_file() and results_path.stat().st_size else 0}."
)
PY

echo "=========================================================="
echo "Five-seed own-top20 MobileNet LLM refinement"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Seed: ${SEED}"
echo "Experiment root: ${EXPERIMENT_ROOT}"
echo "Loop endpoints: 5 and 8 (pre-locked)"
echo "LLM model: ${OPENAI_MODEL}"
echo "Loop 1 reuse source: ${SOURCE_LOOP}"
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

exec bash "${PIPELINE}" llm_refine "${SEED_ROOT}" 8 0.20 "${SEED}" 1 8
