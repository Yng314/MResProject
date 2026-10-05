#!/bin/bash
set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_DIR}/NoiseRate/cxr_real_experiment"
STAMP="${STAMP_OVERRIDE:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT_OVERRIDE:-/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_noes50_clean_3seed/${STAMP}}"
LOOP_COUNT="${LOOP_COUNT_OVERRIDE:-5}"
TOP_FRACTION="${TOP_FRACTION_OVERRIDE:-0.20}"
SEEDS=(13 97 123)

mkdir -p "${OUTPUT_ROOT}" "${PROJECT_DIR}/slurm_logs"

declare -a JOB_LINES=()
for SEED in "${SEEDS[@]}"; do
  if [[ "${SEED}" == "13" ]]; then
    PARTITION="a40,a30"
  else
    PARTITION="a30"
  fi

  JOB_ID="$(
    sbatch --parsable \
      --partition="${PARTITION}" \
      --job-name="mbv3-noes-s${SEED}" \
      "${SCRIPT_DIR}/run_mobilenetv3_noes50_clean_seed.sh" \
      "${SEED}" "${OUTPUT_ROOT}" "${LOOP_COUNT}" "${TOP_FRACTION}"
  )"
  JOB_LINES+=("seed=${SEED} partition=${PARTITION} job=${JOB_ID}")
done

{
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "LOOP_COUNT=${LOOP_COUNT}"
  echo "TOP_FRACTION=${TOP_FRACTION}"
  echo "SEEDS=${SEEDS[*]}"
  echo "SUBMITTED_AT=$(date)"
  for LINE in "${JOB_LINES[@]}"; do
    echo "${LINE}"
  done
} | tee "${OUTPUT_ROOT}/submission_summary.txt"
