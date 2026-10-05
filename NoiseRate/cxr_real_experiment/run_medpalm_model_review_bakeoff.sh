#!/bin/bash
#SBATCH --job-name=medpalm-luna
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/medpalm_luna_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/medpalm_luna_%j.err
#SBATCH --partition=training
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail

PROJECT_ROOT="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT="${PROJECT_ROOT}/NoiseRate/cxr_real_experiment/medpalm_model_review_bakeoff.py"
SOURCE="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/medpalm_external_entry_benchmark/20260730_binary_external_no_oof"
OUTPUT="${MEDPALM_BAKEOFF_OUTPUT_OVERRIDE:-/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/medpalm_model_review_bakeoff/20260820_gpt54_vs_gpt56luna_n100}"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${OUTPUT}"
source ~/.llm_review_env
export OPENAI_MODEL="gpt-5.6-luna"

echo "Model: ${OPENAI_MODEL}"
echo "Output: ${OUTPUT}"
echo "Started: $(date --iso-8601=seconds)"

"${PYTHON}" -u "${SCRIPT}" prepare --source-dir "${SOURCE}" --output-dir "${OUTPUT}"
for pass in 1 2; do
  echo "Review pass ${pass}/2"
  "${PYTHON}" -u "${SCRIPT}" review \
    --source-dir "${SOURCE}" \
    --output-dir "${OUTPUT}" \
    --resume \
    --concurrency 10 \
    --retry-limit 2 \
    --timeout 180
done
"${PYTHON}" -u "${SCRIPT}" evaluate --source-dir "${SOURCE}" --output-dir "${OUTPUT}"
"${PYTHON}" -u "${SCRIPT}" verify --source-dir "${SOURCE}" --output-dir "${OUTPUT}"

echo "Completed: $(date --iso-8601=seconds)"
