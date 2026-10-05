#!/bin/bash
#SBATCH --job-name=vindr-xrv-oof
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_xrv_oof_%A_%a.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_xrv_oof_%A_%a.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --array=0-5%3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
SCRIPT_DIR="${ROOT}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
ENGINE="${SCRIPT_DIR}/vindr_known_gt_cl_benchmark.py"
PROGRAM="${SCRIPT_DIR}/vindr_xrv_oof_comparison.py"
TEST_PROGRAM="${SCRIPT_DIR}/test_vindr_xrv_oof_comparison.py"
PROTOCOL="${SCRIPT_DIR}/vindr_xrv_oof_comparison_protocol_20260813.md"
PARENT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_detector_benchmark_v2/20260812_v1"
MANIFEST="${PARENT}/blind_run_manifest.csv"
FEATURE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_detector_benchmark_v2_smoke/20260812_v1/external_evidence"
FEATURES="${FEATURE_ROOT}/xrv_external_evidence.npz"
SMOKE="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_xrv_oof_comparison_smoke/20260813_v1"
OUTPUT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_xrv_oof_comparison/20260813_v1"
XRV_ROOT="${OUTPUT}/xrv_oof_runs"

EXPECTED_ENGINE_SHA="f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83"
EXPECTED_PROGRAM_SHA="1b3e02e01699f0d891928c3d9fc0cd3618770d4f0e5ed4323cebf119573cc976"
EXPECTED_TEST_SHA="347764d512d905ca7eeb7d2e892d98b3f8c5cff232ca63aec49d44bd7442021c"
EXPECTED_PROTOCOL_SHA="ebbdd154adc6abdafa280886335bb68aef728feb716ccad13fac6365387ae8ef"
EXPECTED_FEATURE_SHA="c03864a90dfcb339b13c76444ddc116f90be67520f56f12f1158097c62ed35c6"
EXPECTED_MANIFEST_SHA="305864db2ecc316860cdd9c7c2fd46512c1c08325d74c41413d48f530ff5adb6"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 5 )); then
  echo "A Slurm array task id in 0-5 is required" >&2
  exit 2
fi
for item in \
  "${ENGINE}:${EXPECTED_ENGINE_SHA}" \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${TEST_PROGRAM}:${EXPECTED_TEST_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${FEATURES}:${EXPECTED_FEATURE_SHA}" \
  "${MANIFEST}:${EXPECTED_MANIFEST_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  [[ "$(sha256sum "${path}" | cut -d' ' -f1)" == "${expected}" ]] || {
    echo "Hash mismatch: ${path}" >&2
    exit 2
  }
done
for marker in \
  "${PARENT}/.prepare_complete" \
  "${FEATURE_ROOT}/.external_evidence_complete" \
  "${SMOKE}/.smoke_complete"; do
  [[ -e "${marker}" ]] || { echo "Required marker is missing: ${marker}" >&2; exit 2; }
done

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "${XRV_ROOT}" "${MPLCONFIGDIR}"

run_one() {
  local run_index="$1"
  mapfile -t fields < <("${PYTHON}" - "${MANIFEST}" "${run_index}" <<'PY'
import csv
import sys
from pathlib import PurePosixPath

with open(sys.argv[1], newline="") as handle:
    rows = list(csv.DictReader(handle))
index = int(sys.argv[2])
if len(rows) != 60 or [int(row["array_index"]) for row in rows] != list(range(60)):
    raise RuntimeError("Manifest is not the ordered 60-run grid")
row = rows[index]
prepared = PurePosixPath(row["prepared_relpath"])
if prepared.is_absolute() or ".." in prepared.parts:
    raise RuntimeError("Unsafe prepared path")
for key in ["scenario_id", "seed", "prepared_relpath"]:
    print(row[key])
PY
  )
  [[ ${#fields[@]} -eq 3 ]] || { echo "Manifest lookup failed" >&2; exit 2; }
  local scenario="${fields[0]}"
  local seed="${fields[1]}"
  local prepared="${PARENT}/${fields[2]}"
  local run_root="${XRV_ROOT}/scenarios/${scenario}/seed_${seed}/blind_run"
  [[ -e "${prepared}/.prepare_complete" ]] || { echo "Prepared marker missing: ${prepared}" >&2; exit 2; }
  [[ ! -e "${run_root}" ]] || { echo "Refusing to overwrite: ${run_root}" >&2; exit 2; }
  mkdir -p "$(dirname "${run_root}")"
  echo "run=${run_index}; scenario=${scenario}; seed=${seed}; started=$(date --iso-8601=seconds)"
  "${PYTHON}" -u "${ENGINE}" run \
    --blind-cohort "${prepared}/blind_noisy_cohort.csv" \
    --features "${FEATURES}" \
    --output-dir "${run_root}" \
    --seed "${seed}" \
    --n-splits 4 \
    --epochs 50 \
    --early-stopping-patience 8 \
    --learning-rate 0.001 \
    --batch-size 128 \
    --device cpu
  "${PYTHON}" - "${run_root}" "${seed}" "${EXPECTED_ENGINE_SHA}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
seed = int(sys.argv[2])
expected = sys.argv[3]
summary = json.loads((root / "blind_run_summary.json").read_text())
evidence = pd.read_csv(root / "entry_evidence.csv")
oof = pd.read_csv(root / "oof_predictions.csv")
if len(evidence) != 18000 or evidence[["image_id", "label_name"]].duplicated().any():
    raise RuntimeError("XRV OOF evidence postflight failed")
probability_columns = [column for column in oof if column.startswith("probability__")]
if len(oof) != 3000 or len(probability_columns) != 6:
    raise RuntimeError("XRV OOF prediction postflight failed")
if not np.isfinite(oof[probability_columns].to_numpy()).all():
    raise RuntimeError("XRV OOF probabilities are non-finite")
if summary["seed"] != seed or summary["program_sha256"] != expected:
    raise RuntimeError("XRV OOF provenance failed")
if not summary["outcome_blind"] or not (root / ".blind_run_complete").is_file():
    raise RuntimeError("XRV OOF blind completion failed")
PY
}

echo "VinDr XRV OOF shard ${SLURM_ARRAY_TASK_ID}"
for offset in 0 6 12 18 24 30 36 42 48 54; do
  run_one "$((SLURM_ARRAY_TASK_ID + offset))"
done
echo "Shard completed: $(date --iso-8601=seconds)"
