#!/bin/bash
#SBATCH --job-name=vindr-dir-eval
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_dir_eval_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_dir_eval_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
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
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_noise_direction_sensitivity/20260804_v2"
PARENT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_known_gt_cl/20260804_v1"
BLIND_MANIFEST="${OUTPUT_ROOT}/blind_run_manifest.csv"
EVALUATION_ROOT="${OUTPUT_ROOT}/evaluation_formal_v1"
TMP_EVALUATION_ROOT="${EVALUATION_ROOT}.partial.${SLURM_JOB_ID:-manual}"
EXPECTED_PROGRAM_SHA="eabe258dd8ce0b860d458c451cd4064514bc1b7a8fa76679347b6c532602390e"
EXPECTED_PROTOCOL_SHA="2baeffb1bd82663332a75234e9c67013bf5ae6b0779c075a3618a7c379ce3ad0"
EXPECTED_OOF_SHA="f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83"
EXPECTED_HELPER_SHA="62da5b019f9809b4641b58f3c9bda56c1f987057fc765f9a76f32feade7046e0"
EXPECTED_FEATURE_SHA="d361d521e681ea8fdff94ba83c206680123032eb247a5b84ffd77d393a895327"
EXPECTED_BLIND_MANIFEST_SHA="a17fba9c02c8a8c9428c1cffa2afee3261980e3c59a4905a68be1bbb8d0da6c4"

for item in \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${OOF_ENGINE}:${EXPECTED_OOF_SHA}" \
  "${HELPER}:${EXPECTED_HELPER_SHA}" \
  "${PARENT_ROOT}/features/xrv_features.npz:${EXPECTED_FEATURE_SHA}" \
  "${BLIND_MANIFEST}:${EXPECTED_BLIND_MANIFEST_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  if [[ "$(sha256sum "${path}" | cut -d' ' -f1)" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done
if [[ ! -e "${OUTPUT_ROOT}/.prepare_complete" ]]; then
  echo "Sensitivity prepare marker is missing" >&2
  exit 2
fi
if [[ -e "${EVALUATION_ROOT}" || -e "${TMP_EVALUATION_ROOT}" ]]; then
  echo "Formal evaluation output already exists; refusing to overwrite" >&2
  exit 2
fi

"${PYTHON}" - "${OUTPUT_ROOT}" "${BLIND_MANIFEST}" <<'PY'
import csv
import sys
from pathlib import Path, PurePosixPath

root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
with manifest_path.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 60 or [int(row["array_index"]) for row in rows] != list(range(60)):
    raise RuntimeError("Blind manifest is not the locked 60-run grid")
for row in rows:
    for key in ["prepared_relpath", "blind_run_relpath"]:
        relpath = PurePosixPath(row[key])
        if relpath.is_absolute() or ".." in relpath.parts:
            raise RuntimeError(f"Unsafe blind manifest path: {key}")
    for relative, marker in [
        (row["prepared_relpath"], ".prepare_complete"),
        (row["blind_run_relpath"], ".blind_run_complete"),
    ]:
        if not (root / relative / marker).is_file():
            raise RuntimeError(f"Required formal marker is missing: {relative}/{marker}")
PY

export PYTHONPATH="${ROOT}/cxr_real_experiment:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"

trap 'status=$?; if (( status != 0 )); then rm -f "${OUTPUT_ROOT}/.benchmark_verified"; fi; exit "${status}"' EXIT

echo "VinDr noise-direction formal private evaluation"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Started: $(date --iso-8601=seconds)"

"${PYTHON}" -u "${PROGRAM}" evaluate \
  --experiment-root "${OUTPUT_ROOT}" \
  --output-dir "${TMP_EVALUATION_ROOT}" \
  --seeds "13,42,97,123,211,307" \
  --bootstrap-iterations 1000

"${PYTHON}" -u "${PROGRAM}" verify \
  --experiment-root "${OUTPUT_ROOT}" \
  --evaluation-dir "${TMP_EVALUATION_ROOT}" \
  --seeds "13,42,97,123,211,307"

"${PYTHON}" - "${OUTPUT_ROOT}" "${TMP_EVALUATION_ROOT}" "${EXPECTED_PROGRAM_SHA}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

experiment_root = Path(sys.argv[1])
evaluation = Path(sys.argv[2])
expected_program_sha = sys.argv[3]
expected_rows = {
    "overall_detection_metrics.csv": 180,
    "budget_metrics.csv": 810,
    "per_label_direction_metrics.csv": 3_240,
    "oof_label_aurocs.csv": 360,
    "quality_summary.csv": 60,
    "image_cluster_bootstrap.csv": 60_000,
    "dqs_calibration.csv": 18,
    "direction_gap_by_seed_rate.csv": 18,
    "direction_gap_by_seed.csv": 6,
    "method_contrasts_by_seed.csv": 6,
    "method_contrasts.csv": 2,
}
for name, expected in expected_rows.items():
    frame = pd.read_csv(evaluation / name)
    if len(frame) != expected:
        raise RuntimeError(f"{name} has {len(frame)} rows; expected {expected}")
    numeric = frame.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise RuntimeError(f"{name} contains infinite values")

summary = json.loads((evaluation / "evaluation_summary.json").read_text())
expected_seeds = [13, 42, 97, 123, 211, 307]
if summary["seeds"] != expected_seeds or summary["program_sha256"] != expected_program_sha:
    raise RuntimeError("Formal evaluation provenance does not match the locked protocol")
if summary["verification_status"] != "ANALYZED" or summary["blind_runs"] != 60:
    raise RuntimeError("Formal evaluation summary is incomplete")
direction = pd.read_csv(evaluation / "direction_gap_by_seed.csv")
if (direction["six_seed_exact_sign_flip_p"] < 0.03125 - 1e-12).any():
    raise RuntimeError("A six-seed exact p-value fell below its attainable minimum")
contrasts = pd.read_csv(evaluation / "method_contrasts.csv")
if (contrasts["exact_sign_flip_p"] < 0.03125 - 1e-12).any():
    raise RuntimeError("A six-seed method exact p-value fell below its attainable minimum")
if (contrasts["holm_p"] + 1e-12 < contrasts["exact_sign_flip_p"]).any():
    raise RuntimeError("Holm-adjusted p-values are invalid")
quality = pd.read_csv(evaluation / "quality_summary.csv")
if not quality["true_quality"].between(0, 1).all() or not quality["raw_entry_dqs"].between(0, 1).all():
    raise RuntimeError("Quality metrics are outside [0, 1]")
for plot in ["direction_recall_by_rate.png", "dqs_calibration.png", "auprc_by_regime_rate.png"]:
    if (evaluation / plot).stat().st_size <= 1_000:
        raise RuntimeError(f"Formal plot is missing or empty: {plot}")
if not (evaluation / ".evaluation_complete").is_file():
    raise RuntimeError("Formal evaluation marker is missing")
if not (experiment_root / ".benchmark_verified").is_file():
    raise RuntimeError("Benchmark verification marker is missing")
print(json.dumps({"formal_evaluation_postflight": "passed", "gates": summary["gates"]}, indent=2))
PY

mv "${TMP_EVALUATION_ROOT}" "${EVALUATION_ROOT}"
trap - EXIT
echo "Completed: $(date --iso-8601=seconds)"
