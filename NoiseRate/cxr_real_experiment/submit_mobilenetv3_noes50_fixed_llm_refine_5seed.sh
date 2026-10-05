#!/bin/bash
set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_DIR}/NoiseRate/cxr_real_experiment"
OUTPUT_ROOT="${OUTPUT_ROOT_OVERRIDE:-/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_noes50_clean_3seed/20260707_123320}"
REFINE_SOURCE_ROOT="${REFINE_SOURCE_ROOT_OVERRIDE:-/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_xrv_iterative_sample20_dual_loop/20260624_150533/llm_refine}"
LOOP_COUNT="${LOOP_COUNT_OVERRIDE:-5}"
SEEDS_TEXT="${SEEDS_OVERRIDE:-7 13 42 97 123}"
MAIL_USER="${MAIL_USER_OVERRIDE:-yz3522@ic.ac.uk}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"

mkdir -p "${OUTPUT_ROOT}" "${PROJECT_DIR}/slurm_logs"

declare -a JOB_LINES=()
for SEED in "${SEEDS[@]}"; do
  JOB_ID="$(
    sbatch --parsable \
      --partition=a30 \
      --job-name="mbv3-llm-s${SEED}" \
      --mail-type=END,FAIL \
      --mail-user="${MAIL_USER}" \
      "${SCRIPT_DIR}/run_mobilenetv3_noes50_fixed_llm_refine_seed.sh" \
      "${SEED}" "${OUTPUT_ROOT}" "${REFINE_SOURCE_ROOT}" "${LOOP_COUNT}"
  )"
  JOB_LINES+=("seed=${SEED} partition=a30 job=${JOB_ID}")
done

SUMMARY_PATH="${OUTPUT_ROOT}/llm_refine_fixed_submission_$(date +%Y%m%d_%H%M%S).txt"
{
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "REFINE_SOURCE_ROOT=${REFINE_SOURCE_ROOT}"
  echo "LOOP_COUNT=${LOOP_COUNT}"
  echo "SEEDS=${SEEDS[*]}"
  echo "REFINEMENT_MODE=fixed_existing_xrv_seed13_tables_no_new_llm_calls"
  echo "SUBMITTED_AT=$(date)"
  for LINE in "${JOB_LINES[@]}"; do
    echo "${LINE}"
  done
} | tee "${SUMMARY_PATH}"
