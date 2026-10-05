#!/bin/bash
set -euo pipefail

SCRIPT_DIR="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/cxr_real_experiment"
RUNNER="${SCRIPT_DIR}/run_mimic_fullpool_resume_unified_seed.sh"
EXPERIMENT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_full_issue_pool_no_repeat_gpt54/20260823_additional5_v1"
SOURCE_LOOP="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_noes50_clean_3seed/20260707_123320/seed_123/sample20_remove_loop/remove_only/loop_01"
RUN_ROOT="${EXPERIMENT_ROOT}/seed_123/llm_refine"
FAILED_LOOP="${RUN_ROOT}/loop_05"
DEFER_RUNNING_JOB="278312"
DEFER_PENDING_JOB="278313"

for command_name in sbatch scontrol squeue sacct; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "Missing Slurm command ${command_name}; run this script on gpucluster2 or gpucluster3." >&2
    exit 2
  }
done

bash -n "${RUNNER}" \
  "${SCRIPT_DIR}/run_mobilenetv3_unified_downstream_seed.sh" \
  "${SCRIPT_DIR}/run_mobilenetv3_full_issue_pool_no_repeat_seed.sh"

for loop_id in 1 2 3 4; do
  loop_pad="$(printf '%02d' "${loop_id}")"
  [[ -e "${RUN_ROOT}/loop_${loop_pad}/.action_tables_complete" ]] || {
    echo "Seed123 Loop ${loop_id} is not transaction-complete; refusing automated resume." >&2
    exit 3
  }
done

for required in \
  "${FAILED_LOOP}/oof/oof_cleanlab_smoke_summary.csv" \
  "${FAILED_LOOP}/sample_top_fraction_expanded_entries.csv" \
  "${FAILED_LOOP}/llm_review/.sharded_review/pass_001/metadata.json" \
  "${FAILED_LOOP}/llm_review/.sharded_review/pass_001/input_shard_0.csv" \
  "${FAILED_LOOP}/llm_review/.sharded_review/pass_001/input_shard_1.csv" \
  "${FAILED_LOOP}/llm_review/.sharded_review/pass_001/input_shard_2.csv"; do
  [[ -s "${required}" ]] || {
    echo "Missing locked Loop5 resume input: ${required}" >&2
    exit 3
  }
done
[[ ! -e "${FAILED_LOOP}/.action_tables_complete" ]] || {
  echo "Seed123 Loop5 is already complete; no quota resume is required." >&2
  exit 3
}

old_state=""
if old_state="$(sacct -X -n -j 278306 --format=State 2>/dev/null | awk 'NF {print $1; exit}')" \
  && [[ -n "${old_state}" ]]; then
  case "${old_state}" in
    FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*) ;;
    *)
      echo "Old seed123 job 278306 is not in a resumable terminal state: ${old_state}" >&2
      exit 3
      ;;
  esac
else
  # The accounting daemon may be temporarily unavailable after a job has
  # already left the controller. Refuse the resume if the old allocation is
  # still active; otherwise the locked incomplete Loop5 checkpoint above is
  # sufficient to identify the exact continuation point.
  old_live_state="$(squeue -h -j 278306 -o '%T' 2>/dev/null | head -n 1 || true)"
  [[ -z "${old_live_state}" ]] || {
    echo "Old seed123 job 278306 is still active: ${old_live_state}." >&2
    exit 3
  }
  echo "sacct is unavailable; verified that old job 278306 is absent from the live queue."
fi

newer_running_state="$(squeue -h -j "${DEFER_RUNNING_JOB}" -o '%T' | head -n 1)"
newer_pending_state="$(squeue -h -j "${DEFER_PENDING_JOB}" -o '%T' | head -n 1)"
[[ "${newer_running_state}" == "RUNNING" ]] || {
  echo "Expected job ${DEFER_RUNNING_JOB} to be RUNNING, observed ${newer_running_state:-absent}." >&2
  exit 3
}
[[ "${newer_pending_state}" == "PENDING" ]] || {
  echo "Expected job ${DEFER_PENDING_JOB} to be PENDING, observed ${newer_pending_state:-absent}." >&2
  exit 3
}

sbatch --test-only "${RUNNER}" "${EXPERIMENT_ROOT}" 123 "${SOURCE_LOOP}" 0

# Prevent the second top-20% group from starting while the recovery job is
# being submitted and the first top-20% group is being safely requeued.
scontrol hold "${DEFER_PENDING_JOB}"
submission="$(sbatch --parsable "${RUNNER}" "${EXPERIMENT_ROOT}" 123 "${SOURCE_LOOP}" 0)"
resume_job="${submission%%;*}"

# Requeue the newly started top-20% job and gate both top-20% groups behind
# the recovered full-pool seed123 job. Their original job IDs and output roots
# are retained, and their runners will skip any completed final model outputs.
scontrol requeuehold "${DEFER_RUNNING_JOB}"
scontrol update JobId="${DEFER_RUNNING_JOB}" Dependency="afterok:${resume_job}"
scontrol update JobId="${DEFER_PENDING_JOB}" Dependency="afterok:${resume_job}"
scontrol release "${DEFER_RUNNING_JOB}"
scontrol release "${DEFER_PENDING_JOB}"

echo "seed123_resume_job=${resume_job}"
echo "deferred_jobs=${DEFER_RUNNING_JOB},${DEFER_PENDING_JOB}"
squeue -j "${resume_job},${DEFER_RUNNING_JOB},${DEFER_PENDING_JOB}" \
  -o '%.18i %.12P %.24j %.8T %.10M %.10L %R'
