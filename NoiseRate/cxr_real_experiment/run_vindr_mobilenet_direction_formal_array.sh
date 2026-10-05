#!/bin/bash
#SBATCH --job-name=vindr-mnet-oof
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_mnet_oof_%A_%a.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_mnet_oof_%A_%a.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
ENGINE="${ROOT}/cxr_real_experiment/vindr_mobilenet_oof.py"
PROTOCOL="${ROOT}/cxr_real_experiment/vindr_mobilenet_direction_sensitivity_protocol_20260805.md"
TEST_PROGRAM="${ROOT}/cxr_real_experiment/test_vindr_mobilenet_oof.py"
DATASET_HELPER="${ROOT}/cxr_real_experiment/cxr_real_full_train_eval_cleanlab_xrv12.py"
CL_HELPER="${ROOT}/cxr_real_experiment/cxr_real_noise_validation_smoke.py"
BENCHMARK_HELPER="${ROOT}/cxr_real_experiment/vindr_known_gt_cl_benchmark.py"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_noise_direction_sensitivity_mobilenet/20260805_v1"
IMAGE_PARENT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/derived/png224_20260804_v1"
IMAGE_ROOT="${IMAGE_PARENT}/images"
BLIND_MANIFEST="${OUTPUT_ROOT}/blind_run_manifest.csv"

EXPECTED_ENGINE_SHA="9d1585d8d37b7c8043516889b9b3d6ddda984425f910a891d889b0c2ecd9b026"
EXPECTED_PROTOCOL_SHA="edb164ded4e17bc387a7822a121d22c62ddbb59178ec806dde22f84849189ffe"
EXPECTED_TEST_SHA="f6d866ad34f3268af6c1ab854a1e3adf88cd1ed6a6f3e715d993228dc3dacf91"
EXPECTED_DATASET_HELPER_SHA="f9b1eb32aaa7d6df9e1d1ebd3ba66fd2ef637bde47676fda4c7b629a7c7f452d"
EXPECTED_CL_HELPER_SHA="62da5b019f9809b4641b58f3c9bda56c1f987057fc765f9a76f32feade7046e0"
EXPECTED_BENCHMARK_HELPER_SHA="f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83"
EXPECTED_BLIND_MANIFEST_SHA="a17fba9c02c8a8c9428c1cffa2afee3261980e3c59a4905a68be1bbb8d0da6c4"
EXPECTED_IMAGE_INDEX_SHA="645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2"
EXPECTED_REPLAY_SUMMARY_SHA="c15023748103e3d117b4d670ccd43d8ea65791440a6427e436e786aff6aa412c"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 5 )); then
  echo "A valid Slurm array task id in 0-5 is required" >&2
  exit 2
fi

for item in \
  "${ENGINE}:${EXPECTED_ENGINE_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${TEST_PROGRAM}:${EXPECTED_TEST_SHA}" \
  "${DATASET_HELPER}:${EXPECTED_DATASET_HELPER_SHA}" \
  "${CL_HELPER}:${EXPECTED_CL_HELPER_SHA}" \
  "${BENCHMARK_HELPER}:${EXPECTED_BENCHMARK_HELPER_SHA}" \
  "${BLIND_MANIFEST}:${EXPECTED_BLIND_MANIFEST_SHA}" \
  "${OUTPUT_ROOT}/image_index.csv:${EXPECTED_IMAGE_INDEX_SHA}" \
  "${OUTPUT_ROOT}/replay_prepare_summary.json:${EXPECTED_REPLAY_SUMMARY_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  if [[ "$(sha256sum "${path}" | cut -d' ' -f1)" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done
for marker in \
  "${OUTPUT_ROOT}/.prepare_complete" \
  "${OUTPUT_ROOT}/smoke_fn_only_r30_seed211_v2/.smoke_postflight_complete" \
  "${IMAGE_PARENT}/.conversion_complete"; do
  if [[ ! -e "${marker}" ]]; then
    echo "Required marker is missing: ${marker}" >&2
    exit 2
  fi
done

source /vol/cuda/12.5.0/setup.sh
export PYTHONPATH="${ROOT}/cxr_real_experiment:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}"

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
for key in [
    "scenario_id", "regime", "noise_rate", "rate_percent", "seed",
    "seed_block", "prepared_relpath", "blind_run_relpath",
]:
    print(row[key])
PY
  )

  if (( ${#RUN_FIELDS[@]} != 8 )); then
    echo "Blind manifest did not return the expected run fields" >&2
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

  echo "VinDr MobileNet formal nested OOF"
  echo "Job ID: ${SLURM_JOB_ID:-manual}; array task=${SLURM_ARRAY_TASK_ID}; run index=${RUN_INDEX}; scenario=${SCENARIO_ID}; seed=${SEED}"
  echo "Condition: regime=${REGIME}; noise_rate=${NOISE_RATE}; rate_percent=${RATE_PERCENT}; seed_block=${SEED_BLOCK}"
  echo "Started: $(date --iso-8601=seconds)"

  "${PYTHON}" -u "${ENGINE}" \
    --blind-cohort "${PREPARED_ROOT}/blind_noisy_cohort.csv" \
    --image-index "${OUTPUT_ROOT}/image_index.csv" \
    --image-root "${IMAGE_ROOT}" \
    --output-dir "${RUN_ROOT}" \
    --seed "${SEED}" \
    --n-splits 4 \
    --epochs 100 \
    --early-stopping-patience 10 \
    --learning-rate 0.001 \
    --batch-size 32 \
    --num-workers 4 \
    --device cuda

  "${PYTHON}" - "${PREPARED_ROOT}" "${RUN_ROOT}" "${SEED}" "${EXPECTED_ENGINE_SHA}" "${EXPECTED_IMAGE_INDEX_SHA}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

prepared = Path(sys.argv[1])
run_root = Path(sys.argv[2])
seed = int(sys.argv[3])
expected_engine_sha = sys.argv[4]
expected_image_index_sha = sys.argv[5]
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
    forbidden = [
        column for column in frame.columns
        if any(token in column.lower() for token in ["clean", "reference", "injected", "true_", "error"])
    ]
    if forbidden:
        raise RuntimeError(f"Outcome columns leaked into {name}: {forbidden}")
if set(history["fold_id"]) != {0, 1, 2, 3} or history.groupby("fold_id").size().min() < 1:
    raise RuntimeError("Formal training history lacks one or more folds")
if int(history["epoch"].max()) > 100 or history["inner_split_seed"].isna().any():
    raise RuntimeError("Formal early-stopping history is invalid")
if set(support["fold_id"]) != {0, 1, 2, 3}:
    raise RuntimeError("Formal outer-fold support is incomplete")
if summary["seed"] != seed or summary["entries"] != 18_000 or not summary["outcome_blind"]:
    raise RuntimeError("Formal blind-run summary does not match this run")
if summary["program_sha256"] != expected_engine_sha or summary["image_index_sha256"] != expected_image_index_sha:
    raise RuntimeError("Formal code or image-index provenance failed")
if summary["blind_cohort_sha256"] != prepare_summary["blind_cohort_sha256"]:
    raise RuntimeError("Formal blind cohort differs from the prepared input")
if summary["epochs_max"] != 100 or summary["early_stopping_patience"] != 10:
    raise RuntimeError("Formal training limits differ from the protocol")
if summary["batch_size"] != 32 or summary["learning_rate"] != 0.001:
    raise RuntimeError("Formal optimizer configuration differs from the protocol")
if not summary["model"].startswith("torchvision mobilenet_v3_small scratch"):
    raise RuntimeError("Formal model does not match the MobileNet protocol")
if int(summary["cl_issue_entries"]) <= 0 or not (run_root / ".blind_run_complete").is_file():
    raise RuntimeError("Formal CL evidence or completion marker is missing")
print(json.dumps({
    "seed": seed,
    "scenario": run_root.parent.parent.name,
    "formal_blind_postflight": "passed",
    "cl_issue_entries": summary["cl_issue_entries"],
    "best_epochs": summary["best_epochs"],
}, indent=2))
PY
  echo "Completed run index ${RUN_INDEX}: $(date --iso-8601=seconds)"
}

echo "MobileNet array shard ${SLURM_ARRAY_TASK_ID} will process ten run indices at offsets 0,6,...,54"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for offset in 0 6 12 18 24 30 36 42 48 54; do
  run_one "$((SLURM_ARRAY_TASK_ID + offset))"
done
echo "Array shard completed: $(date --iso-8601=seconds)"
