#!/bin/bash
# Submit the detector comparison and serial AUM screening chain.

set -euo pipefail

SCRIPT_DIR="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/cxr_real_experiment"
DETECTOR_RUNNER="${SCRIPT_DIR}/run_vindr_detector_alternatives_benchmark.sh"
AUM_RUNNER="${SCRIPT_DIR}/run_vindr_oof_aum_screen_array.sh"
AUM_AGGREGATE_RUNNER="${SCRIPT_DIR}/run_vindr_oof_aum_screen_aggregate.sh"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this script on gpucluster2" >&2
  exit 2
fi
for runner in "${DETECTOR_RUNNER}" "${AUM_RUNNER}" "${AUM_AGGREGATE_RUNNER}"; do
  bash -n "${runner}"
done

echo "Current jobs before submission:"
squeue -u "${USER}" -o "%.18i %.10P %.24j %.2t %.10M %.24R"

detector_submission="$(sbatch --parsable "${DETECTOR_RUNNER}")"
aum_submission="$(sbatch --parsable "${AUM_RUNNER}")"
detector_id="${detector_submission%%;*}"
aum_id="${aum_submission%%;*}"
aggregate_submission="$(sbatch --parsable --dependency="afterok:${aum_id}" "${AUM_AGGREGATE_RUNNER}")"
aggregate_id="${aggregate_submission%%;*}"

printf 'DETECTOR_JOB_ID=%s\n' "${detector_id}"
printf 'AUM_ARRAY_JOB_ID=%s\n' "${aum_id}"
printf 'AUM_AGGREGATE_JOB_ID=%s\n' "${aggregate_id}"
echo "AUM array concurrency is locked to one GPU task; aggregate waits for both tasks."
squeue -u "${USER}" -o "%.18i %.10P %.24j %.2t %.10M %.24R"
