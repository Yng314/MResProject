#!/bin/bash
#SBATCH --job-name=vindr-gt-oof
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_gt_oof_%A_%a.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_gt_oof_%A_%a.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --array=0-3%3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
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
EXPECTED_PROGRAM_SHA="f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83"
EXPECTED_PROTOCOL_SHA="733af6031ec02bb5f3559085d97199d87cd34a0c9e0e025efac1765811a51622"
EXPECTED_HELPER_SHA="62da5b019f9809b4641b58f3c9bda56c1f987057fc765f9a76f32feade7046e0"
SEEDS=(13 42 97 123)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#SEEDS[@]} )); then
  echo "A valid Slurm array task id is required" >&2
  exit 2
fi
SEED="${SEEDS[SLURM_ARRAY_TASK_ID]}"
PREPARED_ROOT="${OUTPUT_ROOT}/seed_${SEED}/prepared"
RUN_ROOT="${OUTPUT_ROOT}/seed_${SEED}/blind_run"

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
for marker in \
  "${OUTPUT_ROOT}/.prepare_complete" \
  "${OUTPUT_ROOT}/features/.features_complete" \
  "${OUTPUT_ROOT}/smoke_seed13/.smoke_postflight_complete" \
  "${PREPARED_ROOT}/.prepare_complete"; do
  if [[ ! -e "${marker}" ]]; then
    echo "Required upstream marker is missing: ${marker}" >&2
    exit 2
  fi
done
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Formal blind-run output already exists; refusing to overwrite: ${RUN_ROOT}" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}/cxr_real_experiment:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"

echo "VinDr known-GT formal nested OOF"
echo "Job ID: ${SLURM_JOB_ID:-manual}; array task: ${SLURM_ARRAY_TASK_ID}; seed: ${SEED}"
echo "Started: $(date --iso-8601=seconds)"

"${PYTHON}" -u "${PROGRAM}" run \
  --blind-cohort "${PREPARED_ROOT}/blind_noisy_cohort.csv" \
  --features "${OUTPUT_ROOT}/features/xrv_features.npz" \
  --output-dir "${RUN_ROOT}" \
  --seed "${SEED}" \
  --n-splits 4 \
  --epochs 50 \
  --early-stopping-patience 8 \
  --learning-rate 0.001 \
  --batch-size 128 \
  --device cpu

"${PYTHON}" - "${RUN_ROOT}" "${SEED}" "${EXPECTED_PROGRAM_SHA}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

run_root = Path(sys.argv[1])
seed = int(sys.argv[2])
expected_program_sha = sys.argv[3]
evidence = pd.read_csv(run_root / "entry_evidence.csv")
oof = pd.read_csv(run_root / "oof_predictions.csv")
history = pd.read_csv(run_root / "training_history.csv")
support = pd.read_csv(run_root / "fold_support.csv")
summary = json.loads((run_root / "blind_run_summary.json").read_text())

if len(evidence) != 18000 or evidence[["image_id", "label_name"]].duplicated().any():
    raise RuntimeError("Formal entry-evidence postflight failed")
if len(oof) != 3000 or oof["image_id"].nunique() != 3000:
    raise RuntimeError("Formal OOF row-count postflight failed")
probability_columns = [column for column in oof.columns if column.startswith("probability__")]
if len(probability_columns) != 6 or not np.isfinite(oof[probability_columns].to_numpy()).all():
    raise RuntimeError("Formal OOF probabilities are incomplete")
if set(history["fold_id"]) != {0, 1, 2, 3} or history.groupby("fold_id").size().min() < 1:
    raise RuntimeError("Formal training history lacks one or more folds")
if int(history["epoch"].max()) > 50 or history["inner_split_seed"].isna().any():
    raise RuntimeError("Formal early-stopping history is invalid")
if set(support["fold_id"]) != {0, 1, 2, 3}:
    raise RuntimeError("Formal outer-fold support is incomplete")
if summary["seed"] != seed or summary["entries"] != 18000:
    raise RuntimeError("Formal blind-run summary does not match this task")
if summary["program_sha256"] != expected_program_sha or not summary["outcome_blind"]:
    raise RuntimeError("Formal blind-run provenance check failed")
if int(summary["cl_issue_entries"]) <= 0 or not (run_root / ".blind_run_complete").is_file():
    raise RuntimeError("Formal CL evidence or completion marker is missing")
print(json.dumps({"seed": seed, "formal_blind_postflight": "passed", "cl_issue_entries": summary["cl_issue_entries"]}, indent=2))
PY

echo "Completed: $(date --iso-8601=seconds)"
