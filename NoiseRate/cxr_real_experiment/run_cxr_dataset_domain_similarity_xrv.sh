#!/bin/bash
#SBATCH --job-name=xrv-domain-gap
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/xrv_domain_gap_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/xrv_domain_gap_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
PROGRAM="${ROOT}/cxr_real_experiment/cxr_dataset_domain_similarity.py"
PROTOCOL="${ROOT}/cxr_real_experiment/cxr_dataset_domain_similarity_protocol_20260807.md"
TEST_FILE="${ROOT}/cxr_real_experiment/test_cxr_dataset_domain_similarity.py"
DATA_ROOT="/vol/gpudata/yz3522-llmtest/MResProject/MedSoul/datasets"
VINDR_ROOT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/xrv_vindr_mimic_domain_similarity/20260807_v1"

EXPECTED_PROGRAM_SHA="a3552bad27c59019c296622ad354f89b4d6f808b95d3606f36ba926c57510999"
EXPECTED_PROTOCOL_SHA="aef87b4293f42266a327537d6ff2d86241bb54400dd26121024454baed27a736"
EXPECTED_TEST_SHA="212d7231c28fbd73dcd0bd158b180f70a2ae2f98f0bb656a64ea8cfb80c543da"

for item in \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${TEST_FILE}:${EXPECTED_TEST_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  if [[ "$(sha256sum "${path}" | cut -d' ' -f1)" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done
for marker in \
  "${VINDR_ROOT}/.download_verified" \
  "${VINDR_ROOT}/derived/png224_20260804_v1/.conversion_complete"; do
  if [[ ! -e "${marker}" ]]; then
    echo "Required source marker is missing: ${marker}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_ROOT}/.complete" ]]; then
  echo "Output root is already complete; refusing to rerun: ${OUTPUT_ROOT}" >&2
  exit 2
fi

source /vol/cuda/12.5.0/setup.sh
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
export PYTHONPATH="${ROOT}/cxr_real_experiment:${PYTHONPATH:-}"
mkdir -p "${MPLCONFIGDIR}" "${ROOT}/../slurm_logs"

echo "XRV VinDr-CXR vs MIMIC-CXR domain comparison"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Output: ${OUTPUT_ROOT}"
echo "Started: $(date --iso-8601=seconds)"

"${PYTHON}" "${TEST_FILE}"
"${PYTHON}" - <<'PY'
import torchxrayvision as xrv

cache = "/vol/gpudata/yz3522-llmtest/.cache/torchxrayvision/models_data"
for weights in [
    "densenet121-res224-all",
    "densenet121-res224-nih",
    "densenet121-res224-pc",
]:
    model = xrv.models.DenseNet(weights=weights, cache_dir=cache)
    del model
    print(f"Cached XRV weights: {weights}")
PY
"${PYTHON}" -u "${PROGRAM}" \
  --vindr-index "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_known_gt_cl/20260804_v1/image_index.csv" \
  --vindr-image-root "${VINDR_ROOT}/derived/png224_20260804_v1/images" \
  --mimic-split-csv "${DATA_ROOT}/mimic-cxr-2.0.0-split.csv.gz" \
  --mimic-metadata-csv "${DATA_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz" \
  --mimic-image-root "${DATA_ROOT}/mimic-cxr-jpg-224" \
  --mimic-split train \
  --output-dir "${OUTPUT_ROOT}" \
  --xrv-cache-dir "/vol/gpudata/yz3522-llmtest/.cache/torchxrayvision/models_data" \
  --weights densenet121-res224-all densenet121-res224-nih densenet121-res224-pc \
  --samples 3000 \
  --seed 20260807 \
  --batch-size 64 \
  --workers 4 \
  --pca-dimensions 128 \
  --bootstrap-draws 1000 \
  --mmd-permutations 1000 \
  --internal-repeats 50 \
  --device cuda \
  --resume

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
index = pd.read_csv(root / "locked_image_index.csv")
if index.groupby("source").size().to_dict() != {"MIMIC-CXR": 3000, "VinDr-CXR": 3000}:
    raise RuntimeError("Locked index counts failed")
if index.loc[index["source"].eq("MIMIC-CXR"), "subject_id"].nunique() != 3000:
    raise RuntimeError("MIMIC subject uniqueness failed")
summary = pd.read_csv(root / "domain_similarity_summary.csv")
if len(summary) != 3 or set(summary["weights"]) != {
    "densenet121-res224-all", "densenet121-res224-nih", "densenet121-res224-pc"
}:
    raise RuntimeError("Summary encoder coverage failed")
if not np.isfinite(summary.select_dtypes(include=["number"]).to_numpy()).all():
    raise RuntimeError("Summary contains non-finite metrics")
for weights in summary["weights"]:
    payload = np.load(root / f"{weights}_features.npz")
    if payload["features"].shape != (6000, 1024) or not np.isfinite(payload["features"]).all():
        raise RuntimeError(f"Feature postflight failed: {weights}")
    metrics = json.loads((root / f"{weights}_metrics.json").read_text())
    if not 0.0 <= metrics["c2st"]["auc"] <= 1.0:
        raise RuntimeError(f"C2ST range failed: {weights}")
for required in ["all_metrics.json", "xrv_domain_similarity_summary.png", ".complete"]:
    if not (root / required).is_file():
        raise RuntimeError(f"Missing final artifact: {required}")
print("Domain-similarity postflight passed")
PY

echo "Completed: $(date --iso-8601=seconds)"
