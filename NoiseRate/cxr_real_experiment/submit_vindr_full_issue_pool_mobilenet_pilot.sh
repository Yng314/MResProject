#!/bin/bash
set -euo pipefail

SCRIPT_DIR="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/cxr_real_experiment"
PILOT="${SCRIPT_DIR}/run_vindr_full_issue_pool_mobilenet_seed.sh"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_full_issue_pool_iteration_mobilenet/20260824_multiseed_v1"

command -v sbatch >/dev/null 2>&1 || { echo "Run this script on gpucluster2 or gpucluster3" >&2; exit 2; }
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Refusing to submit over ${OUTPUT_ROOT}" >&2; exit 2; }
bash -n "${PILOT}"
pilot_submission="$(sbatch --parsable "${PILOT}" 11003)"
pilot_id="${pilot_submission%%;*}"
printf 'pilot_job_id=%s\n' "${pilot_id}"
squeue -j "${pilot_id}" -o "%.18i %.12P %.24j %.8T %.10M %.10l %.6D %R"
