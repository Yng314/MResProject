#!/bin/bash
set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_DIR}/NoiseRate/cxr_real_experiment"
STAMP="$(date +%Y%m%d_%H%M%S)"
PIPELINE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_xrv_k4_pipeline_formal/${STAMP}"

mkdir -p "${PIPELINE_ROOT}" "${PROJECT_DIR}/slurm_logs"

JOB1=$(sbatch --parsable "${SCRIPT_DIR}/run_xrv_k4_oof_pipeline_formal.sh" "${PIPELINE_ROOT}")
OOF_RUN_ROOT="${PIPELINE_ROOT}/01_oof_job_${JOB1}"

JOB2=$(sbatch --parsable \
  --dependency=afterok:${JOB1} \
  "${SCRIPT_DIR}/run_xrv_sample_topk_array_formal.sh" "${PIPELINE_ROOT}" "${OOF_RUN_ROOT}")

JOB3=$(sbatch --parsable \
  --dependency=afterok:${JOB2} \
  "${SCRIPT_DIR}/run_xrv_entry_topk_array_formal.sh" "${PIPELINE_ROOT}" "${OOF_RUN_ROOT}")

echo "PIPELINE_ROOT=${PIPELINE_ROOT}"
echo "JOB1_OOF=${JOB1}"
echo "JOB2_SAMPLE_ARRAY=${JOB2}"
echo "JOB3_ENTRY_ARRAY=${JOB3}"
