#!/bin/bash
#SBATCH --job-name=vindr-png-smoke
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_png_smoke_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_png_smoke_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
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
if [[ ! -e "${DATA_ROOT}/.download_verified" ]]; then
  echo "VinDr source verification marker is missing" >&2
  exit 2
fi
if [[ -e "${AUDIT_ROOT}" || -e "${SMOKE_ROOT}" ]]; then
  echo "Smoke output already exists; refusing to overwrite" >&2
  exit 2
fi

echo "VinDr header audit + PNG-224 smoke"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Started: $(date --iso-8601=seconds)"

"${PYTHON}" -u "${PROGRAM}" audit \
  --dicom-dir "${DATA_ROOT}/test" \
  --labels-csv "${DATA_ROOT}/annotations/image_labels_test.csv" \
  --checksums "${DATA_ROOT}/SHA256SUMS.txt" \
  --output-dir "${AUDIT_ROOT}" \
  --workers 4

"${PYTHON}" -u "${PROGRAM}" convert \
  --dicom-dir "${DATA_ROOT}/test" \
  --labels-csv "${DATA_ROOT}/annotations/image_labels_test.csv" \
  --checksums "${DATA_ROOT}/SHA256SUMS.txt" \
  --output-dir "${SMOKE_ROOT}" \
  --workers 4 \
  --image-size 224 \
  --limit 20 \
  --sample-seed 13 \
  --audit-csv "${AUDIT_ROOT}/dicom_header_audit.csv"

"${PYTHON}" - "${AUDIT_ROOT}" "${SMOKE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

audit_root = Path(sys.argv[1])
smoke_root = Path(sys.argv[2])
audit = pd.read_csv(audit_root / "dicom_header_audit.csv")
manifest = pd.read_csv(smoke_root / "conversion_manifest.csv")
if len(audit) != 3000 or audit["image_id"].nunique() != 3000:
    raise RuntimeError("Header audit did not cover exactly 3,000 images")
if len(manifest) != 20 or manifest["image_id"].nunique() != 20:
    raise RuntimeError("Smoke conversion did not cover exactly 20 images")
expected_syntaxes = {
    "1.2.840.10008.1.2",
    "1.2.840.10008.1.2.1",
    "1.2.840.10008.1.2.4.90",
}
if set(audit["transfer_syntax_uid"].astype(str)) != expected_syntaxes:
    raise RuntimeError("Header audit transfer syntaxes differ from the audited lossless set")
if manifest["output_std"].le(0).any():
    raise RuntimeError("Smoke conversion contains a constant image")
covered = audit[audit["image_id"].isin(manifest["image_id"])].copy()
audit["window_status"] = audit["window_center"].notna() & audit["window_width"].notna()
covered["window_status"] = covered["window_center"].notna() & covered["window_width"].notna()
for column in ["transfer_syntax_uid", "photometric_interpretation", "bits_stored", "window_status"]:
    if set(covered[column].astype(str)) != set(audit[column].astype(str)):
        raise RuntimeError(f"Smoke does not cover all marginal levels of {column}")
for output_path in manifest["output_path"]:
    with Image.open(output_path) as image:
        image.load()
        if image.mode != "L" or image.size != (224, 224):
            raise RuntimeError(f"Invalid PNG: {output_path}")
for marker in [audit_root / ".header_audit_complete", smoke_root / ".smoke_complete"]:
    if not marker.is_file():
        raise RuntimeError(f"Missing marker: {marker}")
print(json.dumps({"audit_rows": len(audit), "smoke_images": len(manifest), "postflight": "passed"}))
PY

echo "Completed: $(date --iso-8601=seconds)"
