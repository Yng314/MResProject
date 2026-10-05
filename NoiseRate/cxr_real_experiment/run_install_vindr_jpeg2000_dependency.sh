#!/bin/bash
#SBATCH --job-name=vindr-j2k
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_j2k_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_j2k_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:20:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
PYDICOM_DIR="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/.vendor/vindr_dicom"
TARGET="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/.vendor/vindr_pixel_handlers"
TEST_DICOM="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/test/18f0e2093719a4b98c7e79d4f202b33f.dicom"

if [[ -d "${TARGET}" ]]; then
  versions="$(PYTHONPATH="${PYDICOM_DIR}:${TARGET}" "${PYTHON}" -c 'import pylibjpeg, _openjpeg; print(pylibjpeg.__version__)')"
  if [[ "${versions}" == "2.1.0" ]]; then
    echo "JPEG 2000 dependencies already installed at ${TARGET}"
    exit 0
  fi
  echo "Refusing to overwrite existing JPEG 2000 dependency directory" >&2
  exit 2
fi

mkdir -p "$(dirname "${TARGET}")"
temporary="${TARGET}.job-${SLURM_JOB_ID:-manual}.tmp"
trap 'rm -rf "${temporary}"' EXIT

"${PYTHON}" -m pip install \
  --disable-pip-version-check \
  --no-deps \
  --target "${temporary}" \
  "pylibjpeg==2.1.0" \
  "pylibjpeg-openjpeg==2.5.0"

PYTHONPATH="${PYDICOM_DIR}:${temporary}" "${PYTHON}" - "${TEST_DICOM}" <<'PY'
import sys

import numpy as np
import pydicom
import pylibjpeg

if pydicom.__version__ != "3.0.1":
    raise RuntimeError(f"Unexpected pydicom version: {pydicom.__version__}")
if pylibjpeg.__version__ != "2.1.0":
    raise RuntimeError(f"Unexpected pylibjpeg version: {pylibjpeg.__version__}")

dataset = pydicom.dcmread(sys.argv[1])
array = dataset.pixel_array
if array.ndim != 2 or not np.isfinite(array).all() or np.ptp(array) <= 0:
    raise RuntimeError("JPEG 2000 smoke decode did not produce a valid grayscale image")
print(
    f"Decoded {array.shape[0]}x{array.shape[1]} "
    f"{array.dtype} range=[{array.min()}, {array.max()}]"
)
PY

mv "${temporary}" "${TARGET}"
echo "Installed and decoded with pylibjpeg 2.1.0 + pylibjpeg-openjpeg 2.5.0"
