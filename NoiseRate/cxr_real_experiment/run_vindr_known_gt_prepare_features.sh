#!/bin/bash
#SBATCH --job-name=vindr-gt-features
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_gt_features_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_gt_features_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
PROGRAM="${ROOT}/cxr_real_experiment/vindr_known_gt_cl_benchmark.py"
PROTOCOL="${ROOT}/cxr_real_experiment/vindr_known_gt_cl_protocol_20260804.md"
HELPER="${ROOT}/cxr_real_experiment/cxr_real_noise_validation_smoke.py"
DATA_ROOT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0"
PNG_ROOT="${DATA_ROOT}/derived/png224_20260804_v1"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_known_gt_cl/20260804_v1"
EXPECTED_PROGRAM_SHA="f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83"
EXPECTED_PROTOCOL_SHA="733af6031ec02bb5f3559085d97199d87cd34a0c9e0e025efac1765811a51622"
EXPECTED_HELPER_SHA="62da5b019f9809b4641b58f3c9bda56c1f987057fc765f9a76f32feade7046e0"

for item in \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${HELPER}:${EXPECTED_HELPER_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  if [[ "$(sha256sum "${path}" | cut -d' ' -f1)" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done
for marker in "${DATA_ROOT}/.download_verified" "${PNG_ROOT}/.conversion_complete"; do
  if [[ ! -e "${marker}" ]]; then
    echo "Required source marker is missing: ${marker}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Benchmark output root already exists; refusing to overwrite" >&2
  exit 2
fi

source /vol/cuda/12.5.0/setup.sh
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
export PYTHONPATH="${ROOT}/cxr_real_experiment:${PYTHONPATH:-}"

echo "VinDr known-GT prepare + outcome-blind feature extraction"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Output: ${OUTPUT_ROOT}"
echo "Started: $(date --iso-8601=seconds)"

"${PYTHON}" -u "${PROGRAM}" prepare \
  --labels-csv "${DATA_ROOT}/annotations/image_labels_test.csv" \
  --conversion-manifest "${PNG_ROOT}/conversion_manifest.csv" \
  --output-root "${OUTPUT_ROOT}" \
  --seeds "13,42,97,123" \
  --n-splits 4 \
  --noise-fraction 0.20

"${PYTHON}" -u "${PROGRAM}" extract-features \
  --image-index "${OUTPUT_ROOT}/image_index.csv" \
  --image-root "${PNG_ROOT}/images" \
  --output-dir "${OUTPUT_ROOT}/features" \
  --xrv-cache-dir "/vol/gpudata/yz3522-llmtest/.cache/torchxrayvision/models_data" \
  --device cuda \
  --batch-size 32 \
  --num-workers 4 \
  --seed 13

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
if not (root / ".prepare_complete").is_file():
    raise RuntimeError("Root prepare marker is missing")
index = pd.read_csv(root / "image_index.csv")
if len(index) != 3000 or index["image_id"].nunique() != 3000:
    raise RuntimeError("Image index postflight failed")
for seed in [13, 42, 97, 123]:
    prepared = root / f"seed_{seed}" / "prepared"
    blind = pd.read_csv(prepared / "blind_noisy_cohort.csv")
    private = pd.read_csv(prepared / "private_reference.csv")
    if len(blind) != 3000 or len(private) != 18000:
        raise RuntimeError(f"Seed {seed} prepared row counts are wrong")
    forbidden = [column for column in blind.columns if any(token in column.lower() for token in ["clean", "reference", "injected", "true_", "error"])]
    if forbidden:
        raise RuntimeError(f"Seed {seed} blind columns leak outcomes: {forbidden}")
    if not (prepared / ".prepare_complete").is_file():
        raise RuntimeError(f"Seed {seed} prepare marker is missing")
payload = np.load(root / "features" / "xrv_features.npz")
if payload["features"].shape != (3000, 1024) or not np.isfinite(payload["features"]).all():
    raise RuntimeError("Feature payload postflight failed")
if len(set(payload["image_id"].astype(str))) != 3000:
    raise RuntimeError("Feature image ids are not unique")
summary = json.loads((root / "features" / "feature_summary.json").read_text())
if not summary["outcome_blind"] or summary["samples"] != 3000:
    raise RuntimeError("Feature summary does not preserve outcome isolation")
if not (root / "features" / ".features_complete").is_file():
    raise RuntimeError("Feature completion marker is missing")
print("Prepare + feature extraction postflight passed")
PY

echo "Completed: $(date --iso-8601=seconds)"
