#!/bin/bash
#SBATCH --job-name=vindr-global-prepare
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_global_prepare_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_global_prepare_%j.err
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
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
PROGRAM="${ROOT}/cxr_real_experiment/vindr_global_symmetric_noise.py"
PROTOCOL="${ROOT}/cxr_real_experiment/vindr_global_symmetric_noise_protocol_20260805.md"
TEST_FILE="${ROOT}/cxr_real_experiment/test_vindr_global_symmetric_noise.py"
DATA_ROOT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0"
PARENT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_known_gt_cl/20260804_v1"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_global_symmetric_noise_mobilenet/20260805_v1"

EXPECTED_PROGRAM_SHA="80316491dd61e7f9aa13d52a44bf76448968889bf84a2a33e7aa2a1b70431404"
EXPECTED_PROTOCOL_SHA="033c08d7a93eb06076f35245c7bb2fd36753abf206970cf6e6e78ab87fd4d3da"
EXPECTED_TEST_SHA="4e26804183020f650dd9e74cb35ebaf4bb06512dff4bf93a6d7f33c900ca79fb"
EXPECTED_PARENT_INDEX_SHA="645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2"

for item in \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${TEST_FILE}:${EXPECTED_TEST_SHA}" \
  "${PARENT_ROOT}/image_index.csv:${EXPECTED_PARENT_INDEX_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  if [[ "$(sha256sum "${path}" | cut -d' ' -f1)" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done
for marker in \
  "${DATA_ROOT}/.download_verified" \
  "${PARENT_ROOT}/.prepare_complete"; do
  if [[ ! -e "${marker}" ]]; then
    echo "Required marker is missing: ${marker}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Output root already exists; refusing to overwrite: ${OUTPUT_ROOT}" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}/cxr_real_experiment:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

echo "VinDr exact global symmetric-noise prepare"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Started: $(date --iso-8601=seconds)"

"${PYTHON}" -u "${PROGRAM}" prepare \
  --labels-csv "${DATA_ROOT}/annotations/image_labels_test.csv" \
  --parent-image-index "${PARENT_ROOT}/image_index.csv" \
  --output-root "${OUTPUT_ROOT}" \
  --seeds "13,42,97,123,211,307" \
  --rates "0.10,0.20,0.30" \
  --n-splits 4

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
manifest = pd.read_csv(root / "scenario_manifest.csv")
blind_manifest = pd.read_csv(root / "blind_run_manifest.csv")
scenarios = pd.read_csv(root / "scenarios.csv")
seeds = pd.read_csv(root / "seed_manifest.csv")
if len(manifest) != 24 or manifest[["scenario_id", "seed"]].duplicated().any():
    raise RuntimeError("Prepare manifest does not contain 24 unique runs")
if manifest["array_index"].tolist() != list(range(24)):
    raise RuntimeError("Prepare manifest array indices are not ordered 0-23")
expected_blind_columns = {
    "array_index", "scenario_index", "scenario_id", "regime", "noise_rate",
    "rate_percent", "seed", "seed_block", "prepared_relpath", "blind_run_relpath",
}
if set(blind_manifest.columns) != expected_blind_columns or len(blind_manifest) != 24:
    raise RuntimeError("Blind manifest schema or row count is incorrect")
forbidden = [
    column for column in blind_manifest.columns
    if any(token in column.lower() for token in ["clean", "reference", "injected", "true_", "error"])
]
if forbidden:
    raise RuntimeError(f"Blind manifest contains outcome fields: {forbidden}")
if scenarios["scenario_id"].tolist() != [
    "clean", "symmetric_entry_r10", "symmetric_entry_r20", "symmetric_entry_r30"
]:
    raise RuntimeError("Scenario grid differs from the locked four anchors")
if sorted(seeds["seed"].tolist()) != [13, 42, 97, 123, 211, 307]:
    raise RuntimeError("Seed manifest is incomplete")

expected_errors = {0: 0, 10: 1800, 20: 3600, 30: 5400}
expected_per_label = {0: 0, 10: 300, 20: 600, 30: 900}
for row in manifest.itertuples(index=False):
    if int(row.injected_errors) != expected_errors[int(row.rate_percent)]:
        raise RuntimeError(f"Unexpected global error count: {row.scenario_id}, seed={row.seed}")
    if not np.isclose(float(row.true_quality), 1.0 - float(row.noise_rate)):
        raise RuntimeError(f"True-quality anchor is wrong: {row.scenario_id}, seed={row.seed}")
    prepared = root / row.prepared_relpath
    blind = pd.read_csv(prepared / "blind_noisy_cohort.csv")
    private = pd.read_csv(prepared / "private_reference.csv")
    counts = pd.read_csv(prepared / "corruption_counts.csv")
    support = pd.read_csv(prepared / "fold_support.csv")
    if len(blind) != 3000 or len(private) != 18000 or len(counts) != 6:
        raise RuntimeError(f"Prepared row counts failed: {row.scenario_id}, seed={row.seed}")
    if not (counts["injected_errors"] == expected_per_label[int(row.rate_percent)]).all():
        raise RuntimeError(f"Per-label error count failed: {row.scenario_id}, seed={row.seed}")
    if int(private["injected_error"].sum()) != int(row.injected_errors):
        raise RuntimeError(f"Private error count failed: {row.scenario_id}, seed={row.seed}")
    if support["positive_entries"].min() < 5 or support["negative_entries"].min() < 5:
        raise RuntimeError(f"Outer-fold support failed: {row.scenario_id}, seed={row.seed}")
    if (root / row.blind_run_relpath).exists():
        raise RuntimeError("Prepare transaction created a blind-run outcome directory")
summary = json.loads((root / "prepare_summary.json").read_text())
if summary["blind_runs"] != 24 or summary["scenarios_per_seed"] != 4:
    raise RuntimeError("Prepare summary does not match the locked grid")
if not (root / ".prepare_complete").is_file():
    raise RuntimeError("Root prepare marker is missing")
print("Global symmetric-noise prepare postflight passed")
PY

echo "Completed: $(date --iso-8601=seconds)"
