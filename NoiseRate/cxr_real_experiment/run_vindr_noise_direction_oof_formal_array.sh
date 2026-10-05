#!/bin/bash
#SBATCH --job-name=vindr-dir-oof
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_dir_oof_%A_%a.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_dir_oof_%A_%a.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --array=0-5%3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
SENSITIVITY_PROGRAM="${ROOT}/cxr_real_experiment/vindr_noise_direction_sensitivity.py"
PROTOCOL="${ROOT}/cxr_real_experiment/vindr_noise_direction_sensitivity_protocol_20260804.md"
OOF_ENGINE="${ROOT}/cxr_real_experiment/vindr_known_gt_cl_benchmark.py"
HELPER="${ROOT}/cxr_real_experiment/cxr_real_noise_validation_smoke.py"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_noise_direction_sensitivity/20260804_v2"
PARENT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_known_gt_cl/20260804_v1"
BLIND_MANIFEST="${OUTPUT_ROOT}/blind_run_manifest.csv"
EXPECTED_SENSITIVITY_SHA="eabe258dd8ce0b860d458c451cd4064514bc1b7a8fa76679347b6c532602390e"
EXPECTED_PROTOCOL_SHA="2baeffb1bd82663332a75234e9c67013bf5ae6b0779c075a3618a7c379ce3ad0"
EXPECTED_OOF_SHA="f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83"
EXPECTED_HELPER_SHA="62da5b019f9809b4641b58f3c9bda56c1f987057fc765f9a76f32feade7046e0"
EXPECTED_FEATURE_SHA="d361d521e681ea8fdff94ba83c206680123032eb247a5b84ffd77d393a895327"
EXPECTED_BLIND_MANIFEST_SHA="a17fba9c02c8a8c9428c1cffa2afee3261980e3c59a4905a68be1bbb8d0da6c4"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 5 )); then
  echo "A valid Slurm array task id in 0-5 is required" >&2
  exit 2
fi

for item in \
  "${SENSITIVITY_PROGRAM}:${EXPECTED_SENSITIVITY_SHA}" \
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
for marker in \
  "${OUTPUT_ROOT}/.prepare_complete" \
  "${OUTPUT_ROOT}/smoke_fn_only_r30_seed211/.smoke_postflight_complete" \
  "${PARENT_ROOT}/features/.features_complete"; do
  if [[ ! -e "${marker}" ]]; then
    echo "Required upstream marker is missing: ${marker}" >&2
    exit 2
  fi
done

run_one() {
RUN_INDEX="$1"
mapfile -t RUN_FIELDS < <("${PYTHON}" - "${BLIND_MANIFEST}" "${RUN_INDEX}" <<'PY'
import csv
import re
import sys
from pathlib import PurePosixPath

path, task_id = sys.argv[1], int(sys.argv[2])
expected_columns = {
    "array_index", "scenario_index", "scenario_id", "regime", "noise_rate",
    "rate_percent", "seed", "seed_block", "prepared_relpath", "blind_run_relpath",
}
with open(path, newline="") as handle:
    reader = csv.DictReader(handle)
    if set(reader.fieldnames or []) != expected_columns:
        raise RuntimeError("Blind manifest columns differ from the locked schema")
    rows = list(reader)
if len(rows) != 60 or [int(row["array_index"]) for row in rows] != list(range(60)):
    raise RuntimeError("Blind manifest is not the ordered 60-run grid")
row = rows[task_id]
if int(row["array_index"]) != task_id:
    raise RuntimeError("Blind manifest task lookup failed")
if not re.fullmatch(r"[a-z0-9_]+", row["scenario_id"]):
    raise RuntimeError("Unsafe scenario identifier")
for key in ["prepared_relpath", "blind_run_relpath"]:
    relpath = PurePosixPath(row[key])
    if relpath.is_absolute() or ".." in relpath.parts:
        raise RuntimeError(f"Unsafe relative path in blind manifest: {key}")
for key in ["scenario_id", "regime", "noise_rate", "rate_percent", "seed", "seed_block", "prepared_relpath", "blind_run_relpath"]:
    print(row[key])
PY
)

if (( ${#RUN_FIELDS[@]} != 8 )); then
  echo "Blind manifest did not return the expected task fields" >&2
  exit 2
fi
SCENARIO_ID="${RUN_FIELDS[0]}"
REGIME="${RUN_FIELDS[1]}"
NOISE_RATE="${RUN_FIELDS[2]}"
RATE_PERCENT="${RUN_FIELDS[3]}"
SEED="${RUN_FIELDS[4]}"
SEED_BLOCK="${RUN_FIELDS[5]}"
PREPARED_ROOT="${OUTPUT_ROOT}/${RUN_FIELDS[6]}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_FIELDS[7]}"

if [[ ! -e "${PREPARED_ROOT}/.prepare_complete" ]]; then
  echo "Prepared input marker is missing: ${PREPARED_ROOT}" >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Formal blind-run output already exists; refusing to overwrite: ${RUN_ROOT}" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}/cxr_real_experiment:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"

echo "VinDr noise-direction formal nested OOF"
echo "Job ID: ${SLURM_JOB_ID:-manual}; array task=${SLURM_ARRAY_TASK_ID}; run index=${RUN_INDEX}; scenario=${SCENARIO_ID}; seed=${SEED}"
echo "Condition: regime=${REGIME}; noise_rate=${NOISE_RATE}; rate_percent=${RATE_PERCENT}; seed_block=${SEED_BLOCK}"
echo "Started: $(date --iso-8601=seconds)"

"${PYTHON}" -u "${OOF_ENGINE}" run \
  --blind-cohort "${PREPARED_ROOT}/blind_noisy_cohort.csv" \
  --features "${PARENT_ROOT}/features/xrv_features.npz" \
  --output-dir "${RUN_ROOT}" \
  --seed "${SEED}" \
  --n-splits 4 \
  --epochs 50 \
  --early-stopping-patience 8 \
  --learning-rate 0.001 \
  --batch-size 128 \
  --device cpu

"${PYTHON}" - "${PREPARED_ROOT}" "${RUN_ROOT}" "${SEED}" "${EXPECTED_OOF_SHA}" "${EXPECTED_FEATURE_SHA}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

prepared = Path(sys.argv[1])
run_root = Path(sys.argv[2])
seed = int(sys.argv[3])
expected_program_sha = sys.argv[4]
expected_feature_sha = sys.argv[5]
prepare_summary = json.loads((prepared / "prepare_summary.json").read_text())
summary = json.loads((run_root / "blind_run_summary.json").read_text())
evidence = pd.read_csv(run_root / "entry_evidence.csv")
oof = pd.read_csv(run_root / "oof_predictions.csv")
history = pd.read_csv(run_root / "training_history.csv")
support = pd.read_csv(run_root / "fold_support.csv")

if len(evidence) != 18_000 or evidence[["image_id", "label_name"]].duplicated().any():
    raise RuntimeError("Formal entry-evidence postflight failed")
if len(oof) != 3_000 or oof["image_id"].nunique() != 3_000:
    raise RuntimeError("Formal OOF row-count postflight failed")
probability_columns = [column for column in oof.columns if column.startswith("probability__")]
if len(probability_columns) != 6 or not np.isfinite(oof[probability_columns].to_numpy()).all():
    raise RuntimeError("Formal OOF probabilities are incomplete")
for frame, name in [(evidence.drop(columns=["cl_issue"], errors="ignore"), "evidence"), (oof, "OOF")]:
    forbidden = [column for column in frame.columns if any(token in column.lower() for token in ["clean", "reference", "injected", "true_", "error"])]
    if forbidden:
        raise RuntimeError(f"Outcome columns leaked into {name}: {forbidden}")
if set(history["fold_id"]) != {0, 1, 2, 3} or history.groupby("fold_id").size().min() < 1:
    raise RuntimeError("Formal training history lacks one or more folds")
if int(history["epoch"].max()) > 50 or history["inner_split_seed"].isna().any():
    raise RuntimeError("Formal early-stopping history is invalid")
if set(support["fold_id"]) != {0, 1, 2, 3}:
    raise RuntimeError("Formal outer-fold support is incomplete")
if summary["seed"] != seed or summary["entries"] != 18_000 or not summary["outcome_blind"]:
    raise RuntimeError("Formal blind-run summary does not match this task")
if summary["program_sha256"] != expected_program_sha or summary["features_sha256"] != expected_feature_sha:
    raise RuntimeError("Formal blind-run code or feature provenance failed")
if summary["blind_cohort_sha256"] != prepare_summary["blind_cohort_sha256"]:
    raise RuntimeError("Formal blind cohort differs from the prepared input")
if int(summary["cl_issue_entries"]) <= 0 or not (run_root / ".blind_run_complete").is_file():
    raise RuntimeError("Formal CL evidence or completion marker is missing")
print(json.dumps({"seed": seed, "formal_blind_postflight": "passed", "cl_issue_entries": summary["cl_issue_entries"]}, indent=2))
PY

echo "Completed: $(date --iso-8601=seconds)"
}

echo "Array shard ${SLURM_ARRAY_TASK_ID} will process ten run indices at offsets 0,6,...,54"
for offset in 0 6 12 18 24 30 36 42 48 54; do
  run_one "$((SLURM_ARRAY_TASK_ID + offset))"
done
echo "Array shard completed: $(date --iso-8601=seconds)"
