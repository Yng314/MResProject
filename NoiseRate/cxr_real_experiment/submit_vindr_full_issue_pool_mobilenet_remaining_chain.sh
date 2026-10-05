#!/bin/bash
set -euo pipefail

SCRIPT_DIR="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/cxr_real_experiment"
REMAINING="${SCRIPT_DIR}/run_vindr_full_issue_pool_mobilenet_remaining_array.sh"
AGGREGATE="${SCRIPT_DIR}/run_vindr_full_issue_pool_mobilenet_aggregate.sh"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_full_issue_pool_iteration_mobilenet/20260824_multiseed_v1"

command -v sbatch >/dev/null 2>&1 || { echo "Run this script on gpucluster2 or gpucluster3" >&2; exit 2; }
[[ -e "${OUTPUT_ROOT}/seed_11003/.worker_complete" ]] || {
  echo "Seed-11003 pilot has not passed postflight" >&2
  exit 2
}
for seed in 13007 17011 19001 23003 27011 31013 37003; do
  [[ ! -e "${OUTPUT_ROOT}/seed_${seed}" && ! -e "${OUTPUT_ROOT}/loop0_evidence/seed_${seed}" ]] || {
    echo "Refusing to submit over existing seed ${seed}" >&2
    exit 2
  }
done
bash -n "${REMAINING}" "${AGGREGATE}"
current_jobs="$(squeue -h -u "${USER}" -r | wc -l)"
if [[ "${current_jobs}" -ne 0 ]]; then
  echo "This chain submits eight Slurm elements; wait until the current queue is empty (currently ${current_jobs})" >&2
  exit 2
fi
remaining_submission="$(sbatch --parsable "${REMAINING}")"
remaining_id="${remaining_submission%%;*}"
aggregate_submission="$(sbatch --parsable --dependency="afterok:${remaining_id}" "${AGGREGATE}")"
aggregate_id="${aggregate_submission%%;*}"
printf 'remaining_array_job_id=%s\naggregate_job_id=%s\n' "${remaining_id}" "${aggregate_id}"
squeue -j "${remaining_id},${aggregate_id}" -o "%.18i %.12P %.24j %.8T %.10M %.10l %.6D %R"
