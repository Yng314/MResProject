#!/bin/bash
#SBATCH --job-name=vindr-verify
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_verify_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_verify_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 077

NOISE_ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
SCRIPT="${NOISE_ROOT}/cxr_real_experiment/download_vindr_test_parallel.py"
OUTPUT_DIR="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0"
EXPECTED_SCRIPT_SHA256="f90ca5ff017f5e36eda8f93ca3dac12c14ef10ae5befea1c1129ee6b866d7d9d"

actual_hash="$(sha256sum "${SCRIPT}" | cut -d' ' -f1)"
if [[ "${actual_hash}" != "${EXPECTED_SCRIPT_SHA256}" ]]; then
  echo "Verifier hash mismatch: expected ${EXPECTED_SCRIPT_SHA256}, found ${actual_hash}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}/.download_verified" ]]; then
  echo "Verified marker already exists; refusing to overwrite immutable result." >&2
  exit 2
fi

echo "VinDr-CXR test-cohort verification"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Output: ${OUTPUT_DIR}"
echo "Started: $(date --iso-8601=seconds)"

"${PYTHON}" -u "${SCRIPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --workers 4 \
  --verify-only

echo "Completed: $(date --iso-8601=seconds)"
