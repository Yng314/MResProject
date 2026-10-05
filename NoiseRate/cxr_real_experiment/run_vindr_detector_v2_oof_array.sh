#!/bin/bash
#SBATCH --job-name=vindr-det-v2-oof
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_det_v2_oof_%A_%a.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_det_v2_oof_%A_%a.err
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
SCRIPT_DIR="${ROOT}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
ENGINE="${SCRIPT_DIR}/vindr_mobilenet_oof.py"
TEST_ENGINE="${SCRIPT_DIR}/test_vindr_mobilenet_oof.py"
DATASET_HELPER="${SCRIPT_DIR}/cxr_real_full_train_eval_cleanlab_xrv12.py"
CL_HELPER="${SCRIPT_DIR}/cxr_real_noise_validation_smoke.py"
BENCHMARK_HELPER="${SCRIPT_DIR}/vindr_known_gt_cl_benchmark.py"
PROTOCOL="${SCRIPT_DIR}/vindr_detector_benchmark_v2_protocol_20260812.md"
OUTPUT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_detector_benchmark_v2/20260812_v1"
IMAGE_PARENT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/derived/png224_20260804_v1"
IMAGE_ROOT="${IMAGE_PARENT}/images"
MANIFEST="${OUTPUT}/blind_run_manifest.csv"

EXPECTED_ENGINE_SHA="9d1585d8d37b7c8043516889b9b3d6ddda984425f910a891d889b0c2ecd9b026"
EXPECTED_TEST_SHA="f6d866ad34f3268af6c1ab854a1e3adf88cd1ed6a6f3e715d993228dc3dacf91"
EXPECTED_DATASET_SHA="f9b1eb32aaa7d6df9e1d1ebd3ba66fd2ef637bde47676fda4c7b629a7c7f452d"
EXPECTED_CL_SHA="62da5b019f9809b4641b58f3c9bda56c1f987057fc765f9a76f32feade7046e0"
EXPECTED_BENCHMARK_SHA="f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83"
EXPECTED_PROTOCOL_SHA="c4524cb159d5f8ff7a6f18a5bdf444aa15f77407a0b8731077dc21cd661c9de7"
EXPECTED_INDEX_SHA="645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 5 )); then
  echo "A Slurm array task id in 0-5 is required" >&2
  exit 2
fi
for item in \
  "${ENGINE}:${EXPECTED_ENGINE_SHA}" \
  "${TEST_ENGINE}:${EXPECTED_TEST_SHA}" \
  "${DATASET_HELPER}:${EXPECTED_DATASET_SHA}" \
  "${CL_HELPER}:${EXPECTED_CL_SHA}" \
  "${BENCHMARK_HELPER}:${EXPECTED_BENCHMARK_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${OUTPUT}/image_index.csv:${EXPECTED_INDEX_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  [[ "$(sha256sum "${path}" | cut -d' ' -f1)" == "${expected}" ]] || {
    echo "Hash mismatch: ${path}" >&2
    exit 2
  }
done
[[ -e "${OUTPUT}/.prepare_complete" ]] || { echo "Prepare marker is missing" >&2; exit 2; }
[[ -e "${IMAGE_PARENT}/.conversion_complete" ]] || { echo "Image marker is missing" >&2; exit 2; }

"${PYTHON}" - "${OUTPUT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root / "prepare_summary.json").read_text())
actual = hashlib.sha256((root / "blind_run_manifest.csv").read_bytes()).hexdigest()
if actual != summary["blind_run_manifest_sha256"]:
    raise RuntimeError("Blind manifest differs from the prepared hash")
PY

source /vol/cuda/12.5.0/setup.sh
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}"

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
    raise RuntimeError("Blind manifest is not the ordered 60-run grid")
row = rows[index]
for key in ["prepared_relpath", "blind_run_relpath"]:
    path = PurePosixPath(row[key])
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"Unsafe path in manifest: {key}")
for key in ["scenario_id", "seed", "prepared_relpath", "blind_run_relpath"]:
    print(row[key])
PY
  )
  if (( ${#fields[@]} != 4 )); then
    echo "Manifest lookup did not return four fields" >&2
    exit 2
  fi
  local scenario="${fields[0]}"
  local seed="${fields[1]}"
  local prepared="${OUTPUT}/${fields[2]}"
  local run_root="${OUTPUT}/${fields[3]}"
  [[ -e "${prepared}/.prepare_complete" ]] || { echo "Missing prepared marker: ${prepared}" >&2; exit 2; }
  [[ ! -e "${run_root}" ]] || { echo "Refusing to overwrite: ${run_root}" >&2; exit 2; }
  echo "run=${run_index}; scenario=${scenario}; seed=${seed}; started=$(date --iso-8601=seconds)"
  "${PYTHON}" -u "${ENGINE}" \
    --blind-cohort "${prepared}/blind_noisy_cohort.csv" \
    --image-index "${OUTPUT}/image_index.csv" \
    --image-root "${IMAGE_ROOT}" \
    --output-dir "${run_root}" \
    --seed "${seed}" \
    --n-splits 4 \
    --epochs 100 \
    --early-stopping-patience 10 \
    --learning-rate 0.001 \
    --batch-size 32 \
    --num-workers 4 \
    --device cuda
  "${PYTHON}" - "${prepared}" "${run_root}" "${seed}" "${EXPECTED_ENGINE_SHA}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

prepared, run = Path(sys.argv[1]), Path(sys.argv[2])
seed, expected_engine = int(sys.argv[3]), sys.argv[4]
prepare_summary = json.loads((prepared / "prepare_summary.json").read_text())
summary = json.loads((run / "blind_run_summary.json").read_text())
evidence = pd.read_csv(run / "entry_evidence.csv")
oof = pd.read_csv(run / "oof_predictions.csv")
history = pd.read_csv(run / "training_history.csv")
if len(evidence) != 18_000 or evidence[["image_id", "label_name"]].duplicated().any():
    raise RuntimeError("Entry-evidence postflight failed")
probability_columns = [column for column in oof if column.startswith("probability__")]
if len(oof) != 3_000 or len(probability_columns) != 6:
    raise RuntimeError("OOF postflight failed")
if not np.isfinite(oof[probability_columns].to_numpy()).all():
    raise RuntimeError("OOF probabilities are non-finite")
if set(history["fold_id"]) != {0, 1, 2, 3}:
    raise RuntimeError("Training history is missing a fold")
if summary["seed"] != seed or summary["program_sha256"] != expected_engine:
    raise RuntimeError("Engine provenance failed")
if summary["blind_cohort_sha256"] != prepare_summary["blind_cohort_sha256"]:
    raise RuntimeError("Blind cohort provenance failed")
if not (run / ".blind_run_complete").is_file():
    raise RuntimeError("Blind completion marker is missing")
PY
}

echo "Detector v2 OOF shard ${SLURM_ARRAY_TASK_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for offset in 0 6 12 18 24 30 36 42 48 54; do
  run_one "$((SLURM_ARRAY_TASK_ID + offset))"
done
echo "Shard completed: $(date --iso-8601=seconds)"
