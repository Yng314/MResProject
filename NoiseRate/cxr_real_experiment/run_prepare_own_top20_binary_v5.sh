#!/bin/bash
#SBATCH --job-name=own-v5-prep
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/own_v5_prep_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/own_v5_prep_%j.err
#SBATCH --partition=a16
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail

NEW_ROOT="${1:?new v5 experiment root is required}"
PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
NOISERATE_DIR="${PROJECT_DIR}/NoiseRate"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
PILOT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_own_top20_llm_refine_5seed/20260713_184706"
SOURCE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_noes50_clean_3seed/20260707_123320"

mkdir -p "${PROJECT_DIR}/slurm_logs"

echo "Own-top20 corrected binary-label v5 preparation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "New root: ${NEW_ROOT}"
echo "Started: $(date --iso-8601=seconds)"

cd "${NOISERATE_DIR}"
"${PYTHON}" "${NOISERATE_DIR}/cxr_real_experiment/prepare_own_top20_binary_v5.py" \
  --pilot-root "${PILOT_ROOT}" \
  --source-root "${SOURCE_ROOT}" \
  --new-root "${NEW_ROOT}"

echo "Completed: $(date --iso-8601=seconds)"
