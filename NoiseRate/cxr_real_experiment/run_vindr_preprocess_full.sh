#!/bin/bash
#SBATCH --job-name=vindr-png-full
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_png_full_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_png_full_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
PROGRAM="${ROOT}/cxr_real_experiment/preprocess_vindr_dicoms.py"
PROTOCOL="${ROOT}/cxr_real_experiment/vindr_known_gt_cl_protocol_20260804.md"
PYTHONPATH="${ROOT}/.vendor/vindr_dicom:${ROOT}/.vendor/vindr_pixel_handlers"
DATA_ROOT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0"
AUDIT_ROOT="${DATA_ROOT}/derived/header_audit_20260804_v2"
SMOKE_ROOT="${DATA_ROOT}/derived/png224_smoke_20260804_v2"
OUTPUT_ROOT="${DATA_ROOT}/derived/png224_20260804_v1"
EXPECTED_PROGRAM_SHA="a592e74960357f6b6f6e953dfab51e7b25a74f46cff6592425d103807aade53c"
EXPECTED_PROTOCOL_SHA="b84ec993f14565d9ffb63302986fd20b81fb7c35e7fd4601261becca6c5f81b4"

export PYTHONPATH

if [[ "$(sha256sum "${PROGRAM}" | cut -d' ' -f1)" != "${EXPECTED_PROGRAM_SHA}" ]]; then
  echo "Preprocessor hash mismatch" >&2
  exit 2
fi
if [[ "$(sha256sum "${PROTOCOL}" | cut -d' ' -f1)" != "${EXPECTED_PROTOCOL_SHA}" ]]; then
  echo "Protocol hash mismatch" >&2
  exit 2
fi
for marker in \
  "${DATA_ROOT}/.download_verified" \
  "${AUDIT_ROOT}/.header_audit_complete" \
  "${SMOKE_ROOT}/.smoke_complete"; do
  if [[ ! -e "${marker}" ]]; then
    echo "Required upstream marker is missing: ${marker}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Full conversion output already exists; refusing to overwrite" >&2
  exit 2
fi

echo "VinDr full PNG-224 conversion"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Started: $(date --iso-8601=seconds)"

"${PYTHON}" -u "${PROGRAM}" convert \
  --dicom-dir "${DATA_ROOT}/test" \
  --labels-csv "${DATA_ROOT}/annotations/image_labels_test.csv" \
  --checksums "${DATA_ROOT}/SHA256SUMS.txt" \
  --output-dir "${OUTPUT_ROOT}" \
  --workers 4 \
  --image-size 224 \
  --sample-seed 13

"${PYTHON}" - "${DATA_ROOT}" "${AUDIT_ROOT}" "${OUTPUT_ROOT}" <<'PY'
import hashlib
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

data_root = Path(sys.argv[1])
audit_root = Path(sys.argv[2])
output_root = Path(sys.argv[3])
labels = pd.read_csv(data_root / "annotations/image_labels_test.csv", usecols=["image_id"])
audit = pd.read_csv(audit_root / "dicom_header_audit.csv")
manifest = pd.read_csv(output_root / "conversion_manifest.csv")

expected_ids = set(labels["image_id"].astype(str))
if len(manifest) != 3000 or manifest["image_id"].nunique() != 3000:
    raise RuntimeError("Full conversion manifest does not contain exactly 3,000 images")
if set(manifest["image_id"].astype(str)) != expected_ids:
    raise RuntimeError("Full conversion image ids do not match consensus labels")
if set(audit["image_id"].astype(str)) != expected_ids:
    raise RuntimeError("Header audit image ids do not match consensus labels")

audit_by_id = audit.set_index("image_id")
manifest_by_id = manifest.set_index("image_id")
for column in ["transfer_syntax_uid", "photometric_interpretation", "bits_stored"]:
    left = audit_by_id.loc[manifest_by_id.index, column].astype(str)
    right = manifest_by_id[column].astype(str)
    if not left.equals(right):
        raise RuntimeError(f"Conversion provenance differs from header audit: {column}")
expected_voi = int((audit["window_center"].notna() & audit["window_width"].notna()).sum())
if int(manifest["voi_applied"].sum()) != expected_voi:
    raise RuntimeError("VOI application count differs from the header audit")
if int(manifest["inverted"].sum()) != int(audit["photometric_interpretation"].eq("MONOCHROME1").sum()):
    raise RuntimeError("Monochrome inversion count differs from the header audit")
if manifest["output_std"].le(0).any():
    raise RuntimeError("At least one full-conversion PNG is constant")

for row in manifest.itertuples(index=False):
    path = Path(row.output_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row.output_sha256:
        raise RuntimeError(f"PNG checksum mismatch: {path}")
    with Image.open(path) as image:
        image.load()
        if image.mode != "L" or image.size != (224, 224):
            raise RuntimeError(f"Invalid PNG: {path}")

if list(output_root.rglob("*.part")):
    raise RuntimeError("Full conversion left partial files")
if not (output_root / ".conversion_complete").is_file():
    raise RuntimeError("Full conversion completion marker is missing")
print("Full VinDr PNG-224 postflight passed: 3,000/3,000")
PY

echo "Completed: $(date --iso-8601=seconds)"
