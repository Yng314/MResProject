#!/bin/bash
#SBATCH --job-name=own-ref-eval
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/own_ref_eval_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/own_ref_eval_%j.err
#SBATCH --partition=a16
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
NOISERATE_DIR="${PROJECT_DIR}/NoiseRate"
SCRIPT_DIR="${NOISERATE_DIR}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
BASE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_noes50_clean_3seed/20260707_123320"
REFINE_ROOT="${1:-/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_own_top20_llm_refine_5seed/20260713_184706}"
OUT_DIR="${2:-${REFINE_ROOT}/evaluation_prelocked_loop5_loop8}"

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUT_DIR}/performance" "${OUT_DIR}/quality"

for seed in 7 13 42 97 123; do
  required="${REFINE_ROOT}/seed_${seed}/llm_refine/loop_08/train_eval/test_study_predictions.csv"
  if [[ ! -s "${required}" ]]; then
    echo "Final evaluation requires a complete Loop 8 prediction file: ${required}" >&2
    exit 2
  fi
done

echo "Pre-locked own-top20 evaluation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Refine root: ${REFINE_ROOT}"
echo "Output: ${OUT_DIR}"
echo "Started: $(date --iso-8601=seconds)"

cd "${NOISERATE_DIR}"

"${PYTHON}" "${SCRIPT_DIR}/evaluate_own_top20_refinement_endpoints.py" \
  --base-root "${BASE_ROOT}" \
  --refine-root "${REFINE_ROOT}" \
  --out-dir "${OUT_DIR}/performance" \
  --seeds 7 13 42 97 123 \
  --remove-loops 1 2 3 4 5 \
  --refine-loops 1 2 3 4 5 6 7 8 \
  --checkpoint-loop 5 \
  --extension-loop 8 \
  --remove-comparator-loop 5 \
  --bootstrap-iters 1000 \
  --bootstrap-seed 20260713 \
  --workers 5 \
  --refine-layout own

"${PYTHON}" "${SCRIPT_DIR}/evaluate_own_top20_refinement_quality.py" \
  --base-root "${BASE_ROOT}" \
  --refine-root "${REFINE_ROOT}" \
  --out-dir "${OUT_DIR}/quality" \
  --seeds 7 13 42 97 123 \
  --loops 1 2 3 4 5 6 7 8 \
  --checkpoint-loop 5 \
  --extension-loop 8 \
  --workers 5 \
  --refine-layout own

echo "Completed: $(date --iso-8601=seconds)"
