#!/bin/bash
set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_DIR}/NoiseRate/cxr_real_experiment"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT_OVERRIDE:-/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_xrv_iterative_sample20_dual_loop/${STAMP}}"
LOOP_COUNT="${LOOP_COUNT_OVERRIDE:-5}"
TOP_FRACTION="${TOP_FRACTION_OVERRIDE:-0.20}"
SEED="${SEED_OVERRIDE:-13}"

mkdir -p "${OUTPUT_ROOT}" "${PROJECT_DIR}/slurm_logs"

JOB_REMOVE="$(sbatch --parsable --job-name=xrv-s20-remove "${SCRIPT_DIR}/run_xrv_iterative_sample20_branch.sh" "remove_only" "${OUTPUT_ROOT}" "${LOOP_COUNT}" "${TOP_FRACTION}" "${SEED}")"
JOB_LLM="$(sbatch --parsable --job-name=xrv-s20-llm "${SCRIPT_DIR}/run_xrv_iterative_sample20_branch.sh" "llm_refine" "${OUTPUT_ROOT}" "${LOOP_COUNT}" "${TOP_FRACTION}" "${SEED}")"

cat > "${OUTPUT_ROOT}/submission_summary.txt" <<TXT
OUTPUT_ROOT=${OUTPUT_ROOT}
LOOP_COUNT=${LOOP_COUNT}
TOP_FRACTION=${TOP_FRACTION}
SEED=${SEED}
REMOVE_ONLY_JOB=${JOB_REMOVE}
LLM_REFINE_JOB=${JOB_LLM}
TXT

echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "LOOP_COUNT=${LOOP_COUNT}"
echo "TOP_FRACTION=${TOP_FRACTION}"
echo "SEED=${SEED}"
echo "REMOVE_ONLY_JOB=${JOB_REMOVE}"
echo "LLM_REFINE_JOB=${JOB_LLM}"
