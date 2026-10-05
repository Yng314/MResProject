#!/bin/bash
#SBATCH --job-name=vindr-iter-eval
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_iter_eval_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_iter_eval_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
PROGRAM="${ROOT}/cxr_real_experiment/vindr_iterative_oracle_cleaning.py"
TEST_PROGRAM="${ROOT}/cxr_real_experiment/test_vindr_iterative_oracle_cleaning.py"
PROTOCOL="${ROOT}/cxr_real_experiment/vindr_iterative_oracle_cleaning_protocol_20260805.md"
EXPERIMENT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_iterative_oracle_cleaning/20260805_symmetric_r20_v1"
AGGREGATE_ROOT="${EXPERIMENT_ROOT}/evaluation_formal_v1"

EXPECTED_PROGRAM_SHA="724e78b2e806c90147f2aac153f7c1561d9fd321aec5573cdc9e789026f22d0c"
EXPECTED_TEST_PROGRAM_SHA="3047f53bda9a6428034a66cc1e99d7458408697a143f04a683063f823f62cdff"
EXPECTED_PROTOCOL_SHA="69f5f0822059c64a0a4a33097d1ac7152d6251f49ba0214d9cb412a294170736"

for item in \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${TEST_PROGRAM}:${EXPECTED_TEST_PROGRAM_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  if [[ "$(sha256sum "${path}" | cut -d' ' -f1)" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${AGGREGATE_ROOT}" ]]; then
  echo "Aggregate output already exists; refusing to overwrite: ${AGGREGATE_ROOT}" >&2
  exit 2
fi
for seed in 13 42 97 123 211 307; do
  if [[ ! -e "${EXPERIMENT_ROOT}/seed_${seed}/.seed_evaluation_complete" ]]; then
    echo "Seed evaluation marker is missing: ${seed}" >&2
    exit 2
  fi
done

export PYTHONPATH="${ROOT}/cxr_real_experiment:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

echo "VinDr iterative oracle-cleaning aggregate evaluation"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Started: $(date --iso-8601=seconds)"
"${PYTHON}" -u "${PROGRAM}" aggregate \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --aggregate-output "${AGGREGATE_ROOT}" \
  --seeds "13,42,97,123,211,307"

"${PYTHON}" - "${AGGREGATE_ROOT}" <<'PY'
import json
import math
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
expected_rows = {
    "all_seed_trajectories.csv": 162,
    "seed_summaries.csv": 6,
    "paired_tests.csv": 2,
    "dynamic_loop_summary.csv": 9,
}
for name, expected in expected_rows.items():
    actual = len(pd.read_csv(root / name))
    if actual != expected:
        raise RuntimeError(f"{name} has {actual} rows; expected {expected}")
tests = pd.read_csv(root / "paired_tests.csv")
if not tests[["mean_difference", "exact_p_value", "holm_adjusted_p_value"]].applymap(math.isfinite).all().all():
    raise RuntimeError("Paired test results contain non-finite values")
if (tests["holm_adjusted_p_value"] + 1e-15 < tests["exact_p_value"]).any():
    raise RuntimeError("Holm-adjusted p-value is smaller than its raw p-value")
summary = json.loads((root / "aggregate_summary.json").read_text())
if summary["seeds"] != [13, 42, 97, 123, 211, 307] or summary["loops"] != 8:
    raise RuntimeError("Aggregate summary differs from the protocol")
for name in ["iterative_oracle_summary.png", ".aggregate_complete"]:
    if not (root / name).is_file():
        raise RuntimeError(f"Aggregate artifact is missing: {name}")
print(json.dumps({"aggregate_postflight": "passed", **summary}, indent=2))
PY

echo "Completed: $(date --iso-8601=seconds)"
