#!/bin/bash
#SBATCH --job-name=mbv3-dq-eval
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mbv3_dq_eval_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mbv3_dq_eval_%j.err
#SBATCH --partition=a30
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_DIR}/NoiseRate/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"

ROOT="${ROOT_OVERRIDE:-/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_noes50_clean_3seed/20260707_123320}"
REFINE_SOURCE="${REFINE_SOURCE_OVERRIDE:-/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_xrv_iterative_sample20_dual_loop/20260624_150533/llm_refine}"
OUT_DIR="${OUT_DIR_OVERRIDE:-${ROOT}/evaluation_20260711_data_quality_5seed_ars}"

mkdir -p "${OUT_DIR}" "${PROJECT_DIR}/slurm_logs"

echo "MobileNet five-seed data-quality evaluation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Root: ${ROOT}"
echo "Refinement source: ${REFINE_SOURCE}"
echo "Output: ${OUT_DIR}"
echo "Started: $(date)"

cd "${PROJECT_DIR}/NoiseRate" || exit 1

"${PYTHON}" "${SCRIPT_DIR}/evaluate_mobilenet_data_quality_5seed.py" \
  --root "${ROOT}" \
  --refine-source "${REFINE_SOURCE}" \
  --out-dir "${OUT_DIR}" \
  --seeds 7 13 42 97 123 \
  --loops 1 2 3 4 5 \
  --workers 5

echo "Completed: $(date)"
