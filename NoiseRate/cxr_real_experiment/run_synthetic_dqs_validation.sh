#!/bin/bash
#SBATCH --job-name=synthetic-dqs
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/synthetic_dqs_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/synthetic_dqs_%j.err
#SBATCH --partition=a16
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
SCRIPT="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/validate_synthetic_dqs_adjustment.py"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-${PROJECT_DIR}/NoiseRate/cxr_real_experiment/evaluation_followup_20260713/synthetic_dqs_validation}"

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUTPUT_DIR}"

echo "Synthetic DQS validation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Output: ${OUTPUT_DIR}"
echo "Started: $(date --iso-8601=seconds)"

cd "${PROJECT_DIR}/NoiseRate"
"${PYTHON}" "${SCRIPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --n-samples 5000 \
  --n-seeds 10 \
  --noise-rate 0.12 \
  --loops 5 \
  --cv-folds 4

echo "Completed: $(date --iso-8601=seconds)"
