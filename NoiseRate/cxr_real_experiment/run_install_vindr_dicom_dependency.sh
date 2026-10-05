#!/bin/bash
#SBATCH --job-name=vindr-pydicom
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_pydicom_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_pydicom_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
TARGET="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/.vendor/vindr_dicom"
EXPECTED_VERSION="3.0.1"

if [[ -d "${TARGET}" ]]; then
  installed_version="$(PYTHONPATH="${TARGET}" "${PYTHON}" -c 'import pydicom; print(pydicom.__version__)')"
  if [[ "${installed_version}" == "${EXPECTED_VERSION}" ]]; then
    echo "pydicom ${EXPECTED_VERSION} already installed at ${TARGET}"
    exit 0
  fi
  echo "Refusing to overwrite existing dependency directory with version ${installed_version}" >&2
  exit 2
fi

mkdir -p "$(dirname "${TARGET}")"
temporary="${TARGET}.job-${SLURM_JOB_ID:-manual}.tmp"
trap 'rm -rf "${temporary}"' EXIT

"${PYTHON}" -m pip install \
  --disable-pip-version-check \
  --no-deps \
  --target "${temporary}" \
  "pydicom==${EXPECTED_VERSION}"

installed_version="$(PYTHONPATH="${temporary}" "${PYTHON}" -c 'import pydicom; print(pydicom.__version__)')"
if [[ "${installed_version}" != "${EXPECTED_VERSION}" ]]; then
  echo "Installed pydicom version mismatch: ${installed_version}" >&2
  exit 2
fi

mv "${temporary}" "${TARGET}"
echo "Installed pydicom ${installed_version} at ${TARGET}"
