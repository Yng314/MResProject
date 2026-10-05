#!/bin/bash
#SBATCH --job-name=vindr-det-v2-prepare
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_det_v2_prepare_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_det_v2_prepare_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
SCRIPT_DIR="${ROOT}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
PROGRAM="${SCRIPT_DIR}/vindr_noise_direction_sensitivity.py"
TEST_PROGRAM="${SCRIPT_DIR}/test_vindr_noise_direction_sensitivity.py"
PROTOCOL="${SCRIPT_DIR}/vindr_detector_benchmark_v2_protocol_20260812.md"
DATA_ROOT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0"
PARENT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_known_gt_cl/20260804_v1"
OUTPUT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_detector_benchmark_v2/20260812_v1"
SEEDS="887,1597,3253,5003,7867,10007"

EXPECTED_PROGRAM_SHA="eabe258dd8ce0b860d458c451cd4064514bc1b7a8fa76679347b6c532602390e"
EXPECTED_TEST_SHA="15b7557edd445a29031a69698e514419bb736e114f9c9144cd1e76b7100f1bd6"
EXPECTED_PROTOCOL_SHA="c4524cb159d5f8ff7a6f18a5bdf444aa15f77407a0b8731077dc21cd661c9de7"
EXPECTED_LABELS_SHA="9874c665991c5098db571082f9d9d096fe80c40ac3c2b2007a76d8d226c87a02"
EXPECTED_INDEX_SHA="645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2"

for item in \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${TEST_PROGRAM}:${EXPECTED_TEST_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${DATA_ROOT}/annotations/image_labels_test.csv:${EXPECTED_LABELS_SHA}" \
  "${PARENT}/image_index.csv:${EXPECTED_INDEX_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  [[ "$(sha256sum "${path}" | cut -d' ' -f1)" == "${expected}" ]] || {
    echo "Hash mismatch: ${path}" >&2
    exit 2
  }
done
for marker in "${DATA_ROOT}/.download_verified" "${PARENT}/.benchmark_verified"; do
  [[ -e "${marker}" ]] || { echo "Required marker is missing: ${marker}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "Formal output already exists: ${OUTPUT}" >&2; exit 2; }

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON}" -m unittest -v test_vindr_noise_direction_sensitivity
"${PYTHON}" -u "${PROGRAM}" prepare \
  --labels-csv "${DATA_ROOT}/annotations/image_labels_test.csv" \
  --parent-image-index "${PARENT}/image_index.csv" \
  --output-root "${OUTPUT}" \
  --seeds "${SEEDS}" \
  --rates "0.10,0.20,0.30" \
  --n-splits 4

"${PYTHON}" - "${OUTPUT}" "${SEEDS}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
expected_seeds = [int(value) for value in sys.argv[2].split(",")]
manifest = pd.read_csv(root / "blind_run_manifest.csv")
scenarios = pd.read_csv(root / "scenarios.csv")
seeds = pd.read_csv(root / "seed_manifest.csv")
if len(manifest) != 60 or manifest["array_index"].tolist() != list(range(60)):
    raise RuntimeError("Formal blind manifest must contain ordered indices 0-59")
if manifest[["scenario_id", "seed"]].duplicated().any():
    raise RuntimeError("Formal blind manifest contains duplicate runs")
if scenarios["scenario_id"].tolist() != [
    "clean", "balanced_r10", "fp_only_r10", "fn_only_r10",
    "balanced_r20", "fp_only_r20", "fn_only_r20",
    "balanced_r30", "fp_only_r30", "fn_only_r30",
]:
    raise RuntimeError("Formal scenario order differs from the locked grid")
if seeds["seed"].tolist() != expected_seeds:
    raise RuntimeError("Formal seeds differ from the locked new-seed list")
for row in manifest.itertuples(index=False):
    prepared = root / row.prepared_relpath
    if not (prepared / ".prepare_complete").is_file():
        raise RuntimeError(f"Prepared marker missing: {prepared}")
    if (root / row.blind_run_relpath).exists():
        raise RuntimeError("Preparation created a blind-run result directory")
summary = json.loads((root / "prepare_summary.json").read_text())
if summary["seeds"] != expected_seeds or summary["blind_runs"] != 60:
    raise RuntimeError("Formal prepare summary differs from the locked design")
print("Detector v2 formal preparation postflight passed")
PY
