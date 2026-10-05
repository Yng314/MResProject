#!/bin/bash
# Submit the smoke-gated XRV OOF comparison chain from gpucluster2.

set -euo pipefail

SCRIPT_DIR="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/cxr_real_experiment"
SMOKE="${SCRIPT_DIR}/run_vindr_xrv_oof_comparison_smoke.sh"
ARRAY="${SCRIPT_DIR}/run_vindr_xrv_oof_comparison_array.sh"
EVALUATE="${SCRIPT_DIR}/run_vindr_xrv_oof_comparison_evaluate.sh"

command -v sbatch >/dev/null 2>&1 || { echo "Run this script on gpucluster2" >&2; exit 2; }
for script in "${SMOKE}" "${ARRAY}" "${EVALUATE}"; do
  bash -n "${script}"
done

squeue -u "${USER}" -o "%.18i %.12P %.24j %.8T %.10M %.10l %.6D %R"
smoke_submission="$(sbatch --parsable "${SMOKE}")"
smoke_id="${smoke_submission%%;*}"
array_submission="$(sbatch --parsable --dependency="afterok:${smoke_id}" "${ARRAY}")"
array_id="${array_submission%%;*}"
evaluate_submission="$(sbatch --parsable --dependency="afterok:${array_id}" "${EVALUATE}")"
evaluate_id="${evaluate_submission%%;*}"

printf 'SMOKE_JOB_ID=%s\n' "${smoke_id}"
printf 'OOF_ARRAY_JOB_ID=%s\n' "${array_id}"
printf 'EVALUATE_JOB_ID=%s\n' "${evaluate_id}"
squeue -u "${USER}" -o "%.18i %.12P %.24j %.8T %.10M %.10l %.6D %R"
