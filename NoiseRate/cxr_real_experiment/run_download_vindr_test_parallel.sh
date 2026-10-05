#!/bin/bash
#SBATCH --job-name=vindr-download
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_download_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_download_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=16:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 077

PROJECT_ROOT="/vol/gpudata/yz3522-llmtest/MResProject"
NOISE_ROOT="${PROJECT_ROOT}/NoiseRate"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
SCRIPT="${NOISE_ROOT}/cxr_real_experiment/download_vindr_test_parallel.py"
OUTPUT_DIR="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0"
HOME_CREDENTIAL="/homes/yz3522/.physionet_vindr_netrc"
LOCAL_HOME="${SLURM_TMPDIR:-/tmp}/physionet_vindr_${SLURM_JOB_ID:-manual}"
LOCAL_CREDENTIAL="${LOCAL_HOME}/.netrc"
EXPECTED_SCRIPT_SHA256="f90ca5ff017f5e36eda8f93ca3dac12c14ef10ae5befea1c1129ee6b866d7d9d"

cleanup() {
  rm -rf "${LOCAL_HOME}"
}
trap cleanup EXIT INT TERM

actual_hash="$(sha256sum "${SCRIPT}" | cut -d' ' -f1)"
if [[ "${actual_hash}" != "${EXPECTED_SCRIPT_SHA256}" ]]; then
  echo "Downloader hash mismatch: expected ${EXPECTED_SCRIPT_SHA256}, found ${actual_hash}" >&2
  exit 2
fi
if [[ ! -f "${HOME_CREDENTIAL}" ]]; then
  echo "Missing protected credential file: ${HOME_CREDENTIAL}" >&2
  exit 2
fi
credential_mode="$(stat -c '%a' "${HOME_CREDENTIAL}")"
if [[ "${credential_mode}" != "600" ]]; then
  echo "Credential file mode must be 600, found ${credential_mode}" >&2
  exit 2
fi

install -d -m 700 "${LOCAL_HOME}"
install -m 600 "${HOME_CREDENTIAL}" "${LOCAL_CREDENTIAL}"
rm -f "${HOME_CREDENTIAL}"

echo "VinDr-CXR test-cohort download"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Output: ${OUTPUT_DIR}"
echo "Workers: 4"
echo "Started: $(date --iso-8601=seconds)"

"${PYTHON}" -u "${SCRIPT}" \
  --credential-file "${LOCAL_CREDENTIAL}" \
  --output-dir "${OUTPUT_DIR}" \
  --workers 4 \
  --retries 8

echo "Completed: $(date --iso-8601=seconds)"
