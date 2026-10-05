#!/bin/bash
# Submit the smoke-gated detector v2 confirmation chain from gpucluster2.

set -euo pipefail

SCRIPT_DIR="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/cxr_real_experiment"
SMOKE="${SCRIPT_DIR}/run_vindr_detector_v2_smoke.sh"
PREPARE="${SCRIPT_DIR}/run_vindr_detector_v2_prepare.sh"
OOF="${SCRIPT_DIR}/run_vindr_detector_v2_oof_array.sh"
EVALUATE="${SCRIPT_DIR}/run_vindr_detector_v2_evaluate.sh"

command -v sbatch >/dev/null 2>&1 || { echo "Run this script on gpucluster2" >&2; exit 2; }
for script in "${SMOKE}" "${PREPARE}" "${OOF}" "${EVALUATE}"; do
  bash -n "${script}"
done

echo "Current jobs before submission:"
squeue -u "${USER}" -o "%.18i %.12P %.24j %.8T %.10M %.10l %.6D %R"

smoke_submission="$(sbatch --parsable "${SMOKE}")"
smoke_id="${smoke_submission%%;*}"
prepare_submission="$(sbatch --parsable --dependency="afterok:${smoke_id}" "${PREPARE}")"
prepare_id="${prepare_submission%%;*}"
oof_submission="$(sbatch --parsable --dependency="afterok:${prepare_id}" "${OOF}")"
oof_id="${oof_submission%%;*}"
evaluate_submission="$(sbatch --parsable --dependency="afterok:${oof_id}" "${EVALUATE}")"
evaluate_id="${evaluate_submission%%;*}"

printf 'SMOKE_JOB_ID=%s\n' "${smoke_id}"
printf 'PREPARE_JOB_ID=%s\n' "${prepare_id}"
printf 'OOF_ARRAY_JOB_ID=%s\n' "${oof_id}"
printf 'EVALUATE_JOB_ID=%s\n' "${evaluate_id}"
squeue -u "${USER}" -o "%.18i %.12P %.24j %.8T %.10M %.10l %.6D %R"
