#!/bin/bash
#SBATCH --job-name=vindr-dir-prepare
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_dir_prepare_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_dir_prepare_%j.err
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
PROGRAM="${ROOT}/cxr_real_experiment/vindr_noise_direction_sensitivity.py"
PROTOCOL="${ROOT}/cxr_real_experiment/vindr_noise_direction_sensitivity_protocol_20260804.md"
OOF_ENGINE="${ROOT}/cxr_real_experiment/vindr_known_gt_cl_benchmark.py"
HELPER="${ROOT}/cxr_real_experiment/cxr_real_noise_validation_smoke.py"
TEST_FILE="${ROOT}/cxr_real_experiment/test_vindr_noise_direction_sensitivity.py"
DATA_ROOT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0"
PARENT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_known_gt_cl/20260804_v1"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_noise_direction_sensitivity/20260804_v2"
EXPECTED_PROGRAM_SHA="eabe258dd8ce0b860d458c451cd4064514bc1b7a8fa76679347b6c532602390e"
EXPECTED_PROTOCOL_SHA="2baeffb1bd82663332a75234e9c67013bf5ae6b0779c075a3618a7c379ce3ad0"
EXPECTED_OOF_SHA="f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83"
EXPECTED_HELPER_SHA="62da5b019f9809b4641b58f3c9bda56c1f987057fc765f9a76f32feade7046e0"
EXPECTED_TEST_SHA="15b7557edd445a29031a69698e514419bb736e114f9c9144cd1e76b7100f1bd6"
EXPECTED_FEATURE_SHA="d361d521e681ea8fdff94ba83c206680123032eb247a5b84ffd77d393a895327"

for item in \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${OOF_ENGINE}:${EXPECTED_OOF_SHA}" \
  "${HELPER}:${EXPECTED_HELPER_SHA}" \
  "${TEST_FILE}:${EXPECTED_TEST_SHA}" \
  "${PARENT_ROOT}/features/xrv_features.npz:${EXPECTED_FEATURE_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  if [[ "$(sha256sum "${path}" | cut -d' ' -f1)" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done
for marker in \
  "${DATA_ROOT}/.download_verified" \
  "${PARENT_ROOT}/.prepare_complete" \
  "${PARENT_ROOT}/features/.features_complete" \
  "${PARENT_ROOT}/.benchmark_verified"; do
  if [[ ! -e "${marker}" ]]; then
    echo "Required parent marker is missing: ${marker}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Sensitivity output root already exists; refusing to overwrite" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}/cxr_real_experiment:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"

echo "VinDr direction-sensitivity prepare"
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

import pandas as pd

root = Path(sys.argv[1])
manifest = pd.read_csv(root / "scenario_manifest.csv")
blind_manifest = pd.read_csv(root / "blind_run_manifest.csv")
scenarios = pd.read_csv(root / "scenarios.csv")
seeds = pd.read_csv(root / "seed_manifest.csv")
if len(manifest) != 60 or manifest[["scenario_id", "seed"]].duplicated().any():
    raise RuntimeError("Prepare manifest does not contain 60 unique runs")
expected_blind_columns = {
    "array_index",
    "scenario_index",
    "scenario_id",
    "regime",
    "noise_rate",
    "rate_percent",
    "seed",
    "seed_block",
    "prepared_relpath",
    "blind_run_relpath",
}
if set(blind_manifest.columns) != expected_blind_columns:
    raise RuntimeError(f"Blind manifest columns are not locked: {blind_manifest.columns.tolist()}")
if len(blind_manifest) != 60 or blind_manifest["array_index"].tolist() != list(range(60)):
    raise RuntimeError("Blind manifest must contain exactly the ordered array indices 0-59")
if blind_manifest[["scenario_id", "seed"]].duplicated().any():
    raise RuntimeError("Blind manifest contains duplicate scenario-seed runs")
forbidden = [
    column for column in blind_manifest.columns
    if any(token in column.lower() for token in ["clean", "reference", "injected", "true_", "error"])
]
if forbidden:
    raise RuntimeError(f"Blind manifest contains outcome fields: {forbidden}")
if len(scenarios) != 10 or scenarios["scenario_id"].nunique() != 10:
    raise RuntimeError("Scenario grid must contain ten unique scenarios")
if sorted(seeds["seed"].tolist()) != [13, 42, 97, 123, 211, 307]:
    raise RuntimeError("Seed manifest is incomplete")
expected_errors = {0: 0, 10: 188, 20: 372, 30: 560}
for row in manifest.itertuples(index=False):
    if int(row.injected_errors) != expected_errors[int(row.rate_percent)]:
        raise RuntimeError(f"Unexpected error count: {row.scenario_id}, seed={row.seed}")
    prepared = root / row.prepared_relpath
    if not (prepared / ".prepare_complete").is_file():
        raise RuntimeError(f"Prepare marker is missing: {prepared}")
    blind = pd.read_csv(prepared / "blind_noisy_cohort.csv")
    private = pd.read_csv(prepared / "private_reference.csv")
    counts = pd.read_csv(prepared / "corruption_counts.csv")
    if len(blind) != 3000 or len(private) != 18000 or len(counts) != 6:
        raise RuntimeError(f"Prepared row counts failed: {row.scenario_id}, seed={row.seed}")
    forbidden = [
        column for column in blind.columns
        if any(token in column.lower() for token in ["clean", "reference", "injected", "true_", "error"])
    ]
    if forbidden:
        raise RuntimeError(f"Blind outcome leakage: {forbidden}")
    if int(private["injected_error"].sum()) != int(row.injected_errors):
        raise RuntimeError(f"Private error count failed: {row.scenario_id}, seed={row.seed}")
    if (root / row.blind_run_relpath).exists():
        raise RuntimeError("Prepare transaction created a blind-run outcome directory")
summary = json.loads((root / "prepare_summary.json").read_text())
if summary["blind_runs"] != 60 or summary["scenarios_per_seed"] != 10:
    raise RuntimeError("Prepare summary does not match the locked grid")
if not summary.get("blind_run_manifest_sha256"):
    raise RuntimeError("Prepare summary is missing the blind manifest hash")
if not (root / ".prepare_complete").is_file():
    raise RuntimeError("Root prepare marker is missing")
print("Sensitivity prepare postflight passed")
PY

echo "Completed: $(date --iso-8601=seconds)"
