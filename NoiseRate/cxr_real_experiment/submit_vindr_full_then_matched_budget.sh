#!/bin/bash
set -euo pipefail

SCRIPT_DIR="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/cxr_real_experiment"
JOB="${SCRIPT_DIR}/run_vindr_full_then_matched_budget_serial.sh"
FULL_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_full_issue_pool_iteration_mobilenet/20260824_multiseed_v1"
MATCHED_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_matched_budget_refinement_mobilenet/20260824_v1"

command -v sbatch >/dev/null 2>&1 || {
  echo "Run this submission script on gpucluster2 or gpucluster3" >&2
  exit 2
}
bash -n "${JOB}" \
  "${SCRIPT_DIR}/run_vindr_full_issue_pool_mobilenet_seed.sh" \
  "${SCRIPT_DIR}/run_vindr_full_issue_pool_mobilenet_remaining_serial.sh" \
  "${SCRIPT_DIR}/run_vindr_full_issue_pool_mobilenet_aggregate.sh" \
  "${SCRIPT_DIR}/run_vindr_matched_budget_refinement_seed.sh" \
  "${SCRIPT_DIR}/run_vindr_matched_budget_refinement_aggregate.sh"
[[ -e "${FULL_ROOT}/seed_11003/.worker_complete" ]] || {
  echo "Completed full-pool pilot seed11003 is missing" >&2
  exit 2
}
[[ ! -e "${MATCHED_ROOT}" ]] || {
  echo "Refusing to submit over an existing matched-budget root: ${MATCHED_ROOT}" >&2
  exit 2
}

sbatch --test-only "${JOB}"
submission="$(sbatch --parsable "${JOB}")"
job_id="${submission%%;*}"
printf 'job_id=%s\njob_script=%s\n' "${job_id}" "${JOB}"
squeue -j "${job_id}" -o "%.18i %.12P %.24j %.8T %.10M %.10l %.6D %R"
