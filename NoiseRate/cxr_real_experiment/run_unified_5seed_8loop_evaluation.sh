#!/bin/bash
#SBATCH --job-name=unified-5s8l
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/unified_5s8l_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/unified_5s8l_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
NOISERATE_DIR="${PROJECT_DIR}/NoiseRate"
SCRIPT_DIR="${NOISERATE_DIR}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
BASE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_noes50_clean_3seed/20260707_123320"
REFINE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_own_top20_binary_v5_llm_refine_5seed/20260714_065539"
OUT_DIR="${1:-${REFINE_ROOT}/evaluation_unified_5seed_8loop_20260723}"

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUT_DIR}"

if [[ -f "${OUT_DIR}/.evaluation_complete" ]]; then
  echo "Refusing to overwrite completed evaluation: ${OUT_DIR}" >&2
  exit 2
fi

for seed in 7 13 42 97 123; do
  for loop in 01 02 03 04 05 06 07 08; do
    remove_prediction="${BASE_ROOT}/seed_${seed}/sample20_remove_loop/remove_only/loop_${loop}/train_eval/test_study_predictions.csv"
    refine_prediction="${REFINE_ROOT}/seed_${seed}/llm_refine/loop_${loop}/train_eval/test_study_predictions.csv"
    for required in \
      "${remove_prediction}" \
      "${refine_prediction}"; do
      if [[ ! -s "${required}" ]]; then
        echo "Missing or empty frozen prediction input: ${required}" >&2
        exit 2
      fi
    done
  done
  for marker in \
    "${BASE_ROOT}/seed_${seed}/sample20_remove_loop/remove_only/loop_08/.train_eval_complete" \
    "${REFINE_ROOT}/seed_${seed}/llm_refine/loop_08/.train_eval_complete"; do
    if [[ ! -e "${marker}" ]]; then
      echo "Missing required Loop8 transaction marker: ${marker}" >&2
      exit 2
    fi
  done
done

"${PYTHON}" -m py_compile \
  "${SCRIPT_DIR}/evaluate_unified_5seed_8loop.py"

echo "Five-seed, eight-loop unified evaluation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Base root: ${BASE_ROOT}"
echo "Refine root: ${REFINE_ROOT}"
echo "Output: ${OUT_DIR}"
echo "Started: $(date --iso-8601=seconds)"

cd "${NOISERATE_DIR}"

"${PYTHON}" "${SCRIPT_DIR}/evaluate_unified_5seed_8loop.py" \
  --base-root "${BASE_ROOT}" \
  --refine-root "${REFINE_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --seeds 7 13 42 97 123 \
  --loops 1 2 3 4 5 6 7 8 \
  --bootstrap-iters 2000 \
  --bootstrap-seed 20260723 \
  --workers 5

echo "Completed: $(date --iso-8601=seconds)"
