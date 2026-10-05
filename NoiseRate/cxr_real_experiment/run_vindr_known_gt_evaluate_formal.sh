#!/bin/bash
#SBATCH --job-name=vindr-gt-eval
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_gt_eval_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_gt_eval_%j.err
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
PROGRAM="${ROOT}/cxr_real_experiment/vindr_known_gt_cl_benchmark.py"
PROTOCOL="${ROOT}/cxr_real_experiment/vindr_known_gt_cl_protocol_20260804.md"
HELPER="${ROOT}/cxr_real_experiment/cxr_real_noise_validation_smoke.py"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_known_gt_cl/20260804_v1"
EVALUATION_ROOT="${OUTPUT_ROOT}/evaluation_formal_v1"
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
for seed in 13 42 97 123; do
  for marker in \
    "${OUTPUT_ROOT}/seed_${seed}/prepared/.prepare_complete" \
    "${OUTPUT_ROOT}/seed_${seed}/blind_run/.blind_run_complete"; do
    if [[ ! -e "${marker}" ]]; then
      echo "Required formal marker is missing: ${marker}" >&2
      exit 2
    fi
  done
done
if [[ -e "${EVALUATION_ROOT}" ]]; then
  echo "Formal evaluation output already exists; refusing to overwrite" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}/cxr_real_experiment:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"

echo "VinDr known-GT formal private evaluation"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Started: $(date --iso-8601=seconds)"

"${PYTHON}" -u "${PROGRAM}" evaluate \
  --experiment-root "${OUTPUT_ROOT}" \
  --output-dir "${EVALUATION_ROOT}" \
  --seeds "13,42,97,123" \
  --random-iterations 10000 \
  --bootstrap-iterations 1000 \
  --bootstrap-seed 20260804

"${PYTHON}" -u "${PROGRAM}" verify \
  --experiment-root "${OUTPUT_ROOT}" \
  --evaluation-dir "${EVALUATION_ROOT}" \
  --seeds "13,42,97,123"

"${PYTHON}" - "${OUTPUT_ROOT}" "${EVALUATION_ROOT}" "${EXPECTED_PROGRAM_SHA}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

experiment_root = Path(sys.argv[1])
evaluation = Path(sys.argv[2])
expected_program_sha = sys.argv[3]
expected_rows = {
    "overall_detection_metrics.csv": 12,
    "budget_metrics.csv": 60,
    "per_label_direction_metrics.csv": 216,
    "matched_random_replicates.csv": 200000,
    "oof_label_aurocs.csv": 24,
    "seed_quality_summary.csv": 4,
    "image_cluster_bootstrap.csv": 4000,
    "hierarchical_seed_image_bootstrap.csv": 1000,
    "hierarchical_bootstrap_ci.csv": 5,
    "paired_seed_contrasts.csv": 2,
}
for name, expected in expected_rows.items():
    frame = pd.read_csv(evaluation / name)
    if len(frame) != expected:
        raise RuntimeError(f"{name} has {len(frame)} rows; expected {expected}")
    numeric = frame.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise RuntimeError(f"{name} contains infinite values")

summary = json.loads((evaluation / "evaluation_summary.json").read_text())
if summary["seeds"] != [13, 42, 97, 123] or summary["program_sha256"] != expected_program_sha:
    raise RuntimeError("Formal evaluation provenance does not match the locked protocol")
if summary["verification_status"] != "ANALYZED":
    raise RuntimeError("Formal evaluation status is not ANALYZED")
contrasts = pd.read_csv(evaluation / "paired_seed_contrasts.csv")
if (contrasts["exact_sign_flip_p"] < 0.125 - 1e-12).any():
    raise RuntimeError("A four-seed exact p-value fell below its attainable minimum")
if (contrasts["holm_p"] + 1e-12 < contrasts["exact_sign_flip_p"]).any():
    raise RuntimeError("Holm-adjusted p-values are invalid")
for plot in ["injected_error_auprc_by_seed.png", "error_recall_by_budget.png"]:
    if (evaluation / plot).stat().st_size <= 1000:
        raise RuntimeError(f"Formal plot is missing or empty: {plot}")
if not (evaluation / ".evaluation_complete").is_file():
    raise RuntimeError("Formal evaluation marker is missing")
if not (experiment_root / ".benchmark_verified").is_file():
    raise RuntimeError("Benchmark verification marker is missing")
print(json.dumps({"formal_evaluation_postflight": "passed", "primary_gate_pass": summary["primary_gate_pass"]}, indent=2))
PY

echo "Completed: $(date --iso-8601=seconds)"
