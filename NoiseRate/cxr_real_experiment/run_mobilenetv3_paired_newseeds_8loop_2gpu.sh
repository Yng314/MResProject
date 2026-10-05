#!/bin/bash
#SBATCH --job-name=mb-new2-paired8
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mb_new2_paired8_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mb_new2_paired8_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
NOISERATE_DIR="${PROJECT_DIR}/NoiseRate"
SCRIPT_DIR="${NOISERATE_DIR}/cxr_real_experiment"
PIPELINE="${SCRIPT_DIR}/run_xrv_iterative_sample20_branch.sh"
BASELINE_RUNNER="${SCRIPT_DIR}/run_mobilenetv3_baseline_no_early_stop.sh"
RESULT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_own_top20_binary_v5_newseed_followup/20260724_122307"
REFERENCE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_own_top20_binary_v5_llm_refine_5seed/20260714_065539"
FROZEN_DIR="${REFERENCE_ROOT}/protocol_snapshot_v5_binary_label_locked"
EXPECTED_BASELINE_SHA256="77a24100f8ac37a121307b38ff13c082466b26a3e51bb013dced12cf675d4159"
EXPECTED_OPENAI_MODEL="gpt-5.4"
LOOP_COUNT=8
TOP_FRACTION=0.20

verify_protocol() {
  if ! (cd "${FROZEN_DIR}" && sha256sum --check checksums.sha256); then
    echo "Frozen v5 snapshot checksum verification failed." >&2
    exit 3
  fi

  local frozen_name
  for frozen_name in \
    build_llm_refinement_tables.py \
    cxr_real_full_train_eval_cleanlab_xrv12.py \
    cxr_real_oof_cleanlab_smoke.py \
    export_sample_topk_entries_for_llm.py \
    run_entry_level_llm_review.py \
    run_xrv_iterative_sample20_branch.sh \
    select_topk_issue_samples.py \
    summarize_xrv_loop_metrics.py; do
    if ! cmp -s "${SCRIPT_DIR}/${frozen_name}" "${FROZEN_DIR}/${frozen_name}"; then
      echo "Live execution file differs from frozen v5: ${frozen_name}" >&2
      exit 3
    fi
  done

  local observed_baseline_sha256
  observed_baseline_sha256="$(sha256sum "${BASELINE_RUNNER}" | cut -d' ' -f1)"
  if [[ "${observed_baseline_sha256}" != "${EXPECTED_BASELINE_SHA256}" ]]; then
    echo "Baseline runner checksum changed: ${observed_baseline_sha256}" >&2
    exit 3
  fi
}

link_initial_evidence_for_removal() {
  local seed_root="$1"
  local source_loop="${seed_root}/llm_refine/loop_01"
  local target_loop="${seed_root}/remove_only/loop_01"
  local source
  local target

  if [[ ! -s "${source_loop}/oof/oof_cleanlab_smoke_summary.csv" ]] ||
     [[ ! -s "${source_loop}/sample_top_fraction_issue_subset.csv" ]]; then
    echo "Refinement Loop 1 evidence is incomplete: ${source_loop}" >&2
    exit 4
  fi

  mkdir -p "${target_loop}"
  for source in \
    "${source_loop}/oof" \
    "${source_loop}/sample_top_fraction_issue_subset.csv"; do
    target="${target_loop}/$(basename "${source}")"
    if [[ -L "${target}" ]]; then
      if [[ "$(readlink -f "${target}")" != "$(readlink -f "${source}")" ]]; then
        echo "Existing reuse link points to the wrong source: ${target}" >&2
        exit 4
      fi
    elif [[ -e "${target}" ]]; then
      echo "Refusing to replace existing non-symlink evidence: ${target}" >&2
      exit 4
    else
      ln -s "${source}" "${target}"
    fi
  done
}

run_seed() {
  local seed="$1"
  local seed_root="${RESULT_ROOT}/seed_${seed}"
  local baseline_dir="${seed_root}/baseline_no_clean"

  verify_protocol
  source ~/.llm_review_env
  if [[ "${OPENAI_MODEL:-}" != "${EXPECTED_OPENAI_MODEL}" ]]; then
    echo "OPENAI_MODEL must be ${EXPECTED_OPENAI_MODEL}; found ${OPENAI_MODEL:-unset}." >&2
    exit 2
  fi

  mkdir -p "${seed_root}" "${PROJECT_DIR}/slurm_logs"
  echo "=========================================================="
  echo "Pre-registered paired new-seed experiment"
  echo "Job ID: ${SLURM_JOB_ID}"
  echo "Seed: ${seed}"
  echo "Seed selection: fixed before outcomes; prior seed 7 excluded"
  echo "Methods: no-clean baseline, LLM refinement, simple removal"
  echo "Loop range: 1-${LOOP_COUNT}"
  echo "Result root: ${seed_root}"
  echo "Started: $(date --iso-8601=seconds)"
  echo "=========================================================="

  if [[ -s "${baseline_dir}/baseline_run_summary.csv" ]] &&
     [[ -s "${baseline_dir}/test_study_predictions.csv" ]]; then
    echo "[baseline] Complete outputs found; reusing ${baseline_dir}."
  else
    echo "[baseline] Running matched no-clean baseline."
    SEED_OVERRIDE="${seed}" \
    OUTPUT_DIR_OVERRIDE="${baseline_dir}" \
    EPOCHS_OVERRIDE=50 \
    BATCH_SIZE_OVERRIDE=32 \
      bash "${BASELINE_RUNNER}"
  fi

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

  echo "[refinement] Running or resuming Loop 1-${LOOP_COUNT}."
  bash "${PIPELINE}" \
    llm_refine "${seed_root}" "${LOOP_COUNT}" "${TOP_FRACTION}" "${seed}" 1 "${LOOP_COUNT}"

  # The paired removal branch must start from exactly the same Loop 1 OOF
  # evidence and top-20% selection as refinement.
  link_initial_evidence_for_removal "${seed_root}"
  echo "[removal] Running or resuming Loop 1-${LOOP_COUNT}."
  bash "${PIPELINE}" \
    remove_only "${seed_root}" "${LOOP_COUNT}" "${TOP_FRACTION}" "${seed}" 1 "${LOOP_COUNT}"

  echo "Seed ${seed} paired experiment completed at $(date --iso-8601=seconds)."
}

if [[ "${1:-}" == "--worker" ]]; then
  run_seed "${2:?worker seed is required}"
  exit 0
fi

verify_protocol
mkdir -p "${RESULT_ROOT}" "${PROJECT_DIR}/slurm_logs"

echo "Launching two pre-registered seed workers: 314 and 2718."
srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=6 --mem=48G --gres=gpu:1 \
  bash "$0" --worker 314 \
  >"${PROJECT_DIR}/slurm_logs/mb_new2_paired8_${SLURM_JOB_ID}_seed314.out" \
  2>"${PROJECT_DIR}/slurm_logs/mb_new2_paired8_${SLURM_JOB_ID}_seed314.err" &
pid_314=$!
srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=6 --mem=48G --gres=gpu:1 \
  bash "$0" --worker 2718 \
  >"${PROJECT_DIR}/slurm_logs/mb_new2_paired8_${SLURM_JOB_ID}_seed2718.out" \
  2>"${PROJECT_DIR}/slurm_logs/mb_new2_paired8_${SLURM_JOB_ID}_seed2718.err" &
pid_2718=$!

set +e
wait "${pid_314}"
status_314=$?
wait "${pid_2718}"
status_2718=$?
set -e

if (( status_314 != 0 || status_2718 != 0 )); then
  echo "One or more seed workers failed: seed314=${status_314}, seed2718=${status_2718}." >&2
  exit 1
fi

echo "Both paired new-seed workers completed at $(date --iso-8601=seconds)."
