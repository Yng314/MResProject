#!/bin/bash

set -euo pipefail

EXPERIMENT_ROOT="${1:?corrected v5 experiment root is required}"
PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_DIR}/NoiseRate/cxr_real_experiment"
SNAPSHOT_DIR="${EXPERIMENT_ROOT}/protocol_snapshot_v5_binary_label_locked"

if [[ ! -f "${EXPERIMENT_ROOT}/binary_v5_reuse_manifest.json" ]]; then
  echo "Missing binary v5 preparation manifest under ${EXPERIMENT_ROOT}." >&2
  exit 2
fi
if [[ -e "${SNAPSHOT_DIR}" ]]; then
  echo "Refusing to overwrite existing snapshot: ${SNAPSHOT_DIR}" >&2
  exit 2
fi

files=(
  build_evaluation_followup_deck_20260713.py
  build_llm_refinement_tables.py
  cxr_real_full_train_eval_cleanlab_xrv12.py
  cxr_real_noise_validation_smoke.py
  cxr_real_oof_cleanlab_smoke.py
  evaluate_mobilenet_data_quality_5seed.py
  evaluate_own_top20_refinement_endpoints.py
  evaluate_own_top20_refinement_quality.py
  export_sample_topk_entries_for_llm.py
  freeze_own_top20_binary_v5_protocol.sh
  prepare_own_top20_binary_v5.py
  run_entry_level_llm_review.py
  run_mobilenetv3_own_top20_llm_refine_seed.sh
  run_own_top20_catchall_finalize.sh
  run_own_top20_followup_evaluation.sh
  run_prepare_own_top20_binary_v5.sh
  run_xrv_iterative_sample20_branch.sh
  select_topk_issue_samples.py
  summarize_xrv_loop_metrics.py
  test_binary_label_refinement_pipeline.py
)

mkdir -p "${SNAPSHOT_DIR}"
for name in "${files[@]}"; do
  cp "${SCRIPT_DIR}/${name}" "${SNAPSHOT_DIR}/${name}"
done
cp \
  "${SCRIPT_DIR}/evaluation_followup_20260713/own_top20_refinement_protocol.md" \
  "${SNAPSHOT_DIR}/own_top20_refinement_protocol.md"

(
  cd "${SNAPSHOT_DIR}"
  sha256sum "${files[@]}" own_top20_refinement_protocol.md > checksums.sha256
  sha256sum --check checksums.sha256
)

echo "Frozen v5 snapshot: ${SNAPSHOT_DIR}"
sha256sum "${SNAPSHOT_DIR}/checksums.sha256"
