#!/bin/bash
# Two-stage submission helper for the three-job QOS limit.

set -euo pipefail

SCRIPT_DIR="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/cxr_real_experiment"
SMOKE="${SCRIPT_DIR}/run_vindr_iterative_evidence_improvement_smoke.sh"
FORMAL="${SCRIPT_DIR}/run_vindr_iterative_evidence_improvement_formal_array.sh"
SMOKE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_iterative_evidence_improvement_smoke/20260813_v2"

command -v sbatch >/dev/null 2>&1 || { echo "Run this script on gpucluster2" >&2; exit 2; }
bash -n "${SMOKE}"
bash -n "${FORMAL}"

if [[ -e "${SMOKE_ROOT}/scratch/seed_13/.smoke_branch_complete" \
   && -e "${SMOKE_ROOT}/xrv_pretrained/seed_13/.smoke_branch_complete" ]]; then
  formal_submission="$(sbatch --parsable "${FORMAL}")"
  formal_id="${formal_submission%%;*}"
  printf 'FORMAL_WORKER_ARRAY_JOB_ID=%s\n' "${formal_id}"
else
  smoke_submission="$(sbatch --parsable "${SMOKE}")"
  smoke_id="${smoke_submission%%;*}"
  printf 'SMOKE_JOB_ID=%s\n' "${smoke_id}"
  printf 'Rerun this helper after both smoke branches complete to submit the three formal workers.\n'
fi
squeue -u "${USER}" -o "%.18i %.12P %.28j %.8T %.10M %.10l %.6D %R"
