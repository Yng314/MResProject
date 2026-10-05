#!/bin/bash
#SBATCH --job-name=s20-l1-audit
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/s20_l1_audit_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/s20_l1_audit_%j.err
#SBATCH --partition=a16
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_DIR}/NoiseRate/cxr_real_experiment"
ARCHIVE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment"
RUN_ROOT="${ARCHIVE_ROOT}/results_xrv_iterative_sample20_dual_loop/20260624_150533"
OUTPUT_DIR="${ARCHIVE_ROOT}/llm_refinement_audit/iterative_sample20_loop01_slurm_${SLURM_JOB_ID}"

LLM_RESULTS_CSV="${RUN_ROOT}/llm_refine/loop_01/llm_review/results.csv"
TEST_LABELS_CSV="${PROJECT_DIR}/MedSoul/datasets/mimic-cxr-2.1.0-test-set-labeled.csv"
BASELINE_STUDY_AUC="${ARCHIVE_ROOT}/results_appa_xrv12_linearhead_pipeline/slurm_236337/01_baseline/test_study_auroc_summary.csv"
REMOVE_STUDY_AUC="${RUN_ROOT}/remove_only/loop_01/train_eval/test_study_auroc_summary.csv"
REFINE_STUDY_AUC="${RUN_ROOT}/llm_refine/loop_01/train_eval/test_study_auroc_summary.csv"

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUTPUT_DIR}"

source /vol/gpudata/yz3522-llmtest/venv/bin/activate

echo "=========================================================="
echo "Iterative sample20 loop1 LLM refinement audit"
echo "=========================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Output dir: ${OUTPUT_DIR}"
echo "LLM results: ${LLM_RESULTS_CSV}"
echo "Remove metric source: ${REMOVE_STUDY_AUC}"
echo "Refine metric source: ${REFINE_STUDY_AUC}"
echo "=========================================================="

python -u "${SCRIPT_DIR}/build_llm_refinement_audit.py" \
  --llm-results-csv "${LLM_RESULTS_CSV}" \
  --test-labels-csv "${TEST_LABELS_CSV}" \
  --baseline-study-auroc-csv "${BASELINE_STUDY_AUC}" \
  --remove-study-auroc-csv "${REMOVE_STUDY_AUC}" \
  --refine-study-auroc-csv "${REFINE_STUDY_AUC}" \
  --output-dir "${OUTPUT_DIR}" \
  --manual-review-cases-per-group 5

echo "Completed at $(date)"
