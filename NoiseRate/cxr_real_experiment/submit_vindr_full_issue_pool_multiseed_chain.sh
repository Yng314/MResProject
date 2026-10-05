#!/bin/bash
set -euo pipefail

SCRIPT_DIR="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/cxr_real_experiment"
ARRAY="${SCRIPT_DIR}/run_vindr_full_issue_pool_multiseed_array.sh"
AGGREGATE="${SCRIPT_DIR}/run_vindr_full_issue_pool_multiseed_aggregate.sh"

command -v sbatch >/dev/null 2>&1 || { echo "Run this script on gpucluster2 or gpucluster3" >&2; exit 2; }
array_submission="$(sbatch --parsable "${ARRAY}")"
array_id="${array_submission%%;*}"
aggregate_submission="$(sbatch --parsable --dependency="afterok:${array_id}" "${AGGREGATE}")"
aggregate_id="${aggregate_submission%%;*}"
printf 'array_job_id=%s\naggregate_job_id=%s\n' "${array_id}" "${aggregate_id}"
squeue -j "${array_id},${aggregate_id}" -o "%.18i %.12P %.24j %.8T %.10M %.10l %.6D %R"
