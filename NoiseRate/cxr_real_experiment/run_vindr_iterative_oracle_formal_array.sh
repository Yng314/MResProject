#!/bin/bash
#SBATCH --job-name=vindr-iter-oracle
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_iter_oracle_%A_%a.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_iter_oracle_%A_%a.err
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
PROGRAM="${ROOT}/cxr_real_experiment/vindr_iterative_oracle_cleaning.py"
TEST_PROGRAM="${ROOT}/cxr_real_experiment/test_vindr_iterative_oracle_cleaning.py"
PROTOCOL="${ROOT}/cxr_real_experiment/vindr_iterative_oracle_cleaning_protocol_20260805.md"
ENGINE="${ROOT}/cxr_real_experiment/vindr_mobilenet_oof.py"
ENGINE_TEST="${ROOT}/cxr_real_experiment/test_vindr_mobilenet_oof.py"
BENCHMARK_HELPER="${ROOT}/cxr_real_experiment/vindr_known_gt_cl_benchmark.py"
DATASET_HELPER="${ROOT}/cxr_real_experiment/cxr_real_full_train_eval_cleanlab_xrv12.py"
CL_HELPER="${ROOT}/cxr_real_experiment/cxr_real_noise_validation_smoke.py"
SOURCE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_global_symmetric_noise_mobilenet/20260805_v1"
IMAGE_PARENT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/derived/png224_20260804_v1"
IMAGE_ROOT="${IMAGE_PARENT}/images"
BASE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_iterative_oracle_cleaning"
SMOKE_ROOT="${BASE_ROOT}/smoke_20260805_seed211_loop1_3epoch"
EXPERIMENT_ROOT="${BASE_ROOT}/20260805_symmetric_r20_v1"

EXPECTED_PROGRAM_SHA="724e78b2e806c90147f2aac153f7c1561d9fd321aec5573cdc9e789026f22d0c"
EXPECTED_TEST_PROGRAM_SHA="3047f53bda9a6428034a66cc1e99d7458408697a143f04a683063f823f62cdff"
EXPECTED_PROTOCOL_SHA="69f5f0822059c64a0a4a33097d1ac7152d6251f49ba0214d9cb412a294170736"
EXPECTED_ENGINE_SHA="9d1585d8d37b7c8043516889b9b3d6ddda984425f910a891d889b0c2ecd9b026"
EXPECTED_ENGINE_TEST_SHA="f6d866ad34f3268af6c1ab854a1e3adf88cd1ed6a6f3e715d993228dc3dacf91"
EXPECTED_BENCHMARK_SHA="f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83"
EXPECTED_DATASET_SHA="f9b1eb32aaa7d6df9e1d1ebd3ba66fd2ef637bde47676fda4c7b629a7c7f452d"
EXPECTED_CL_SHA="62da5b019f9809b4641b58f3c9bda56c1f987057fc765f9a76f32feade7046e0"
EXPECTED_SCENARIO_MANIFEST_SHA="cdff8ca8ec86f0a061f9c4856ed0888573c4defe68a31a1784484f7652eb0f17"
EXPECTED_BLIND_MANIFEST_SHA="201ac89c578a1227a8651d3ed942d1727dd8a543fb72c0a59adcc8c8f023745f"
EXPECTED_IMAGE_INDEX_SHA="645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 5 )); then
  echo "A valid Slurm array task id in 0-5 is required" >&2
  exit 2
fi
for item in \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${TEST_PROGRAM}:${EXPECTED_TEST_PROGRAM_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${ENGINE}:${EXPECTED_ENGINE_SHA}" \
  "${ENGINE_TEST}:${EXPECTED_ENGINE_TEST_SHA}" \
  "${BENCHMARK_HELPER}:${EXPECTED_BENCHMARK_SHA}" \
  "${DATASET_HELPER}:${EXPECTED_DATASET_SHA}" \
  "${CL_HELPER}:${EXPECTED_CL_SHA}" \
  "${SOURCE_ROOT}/scenario_manifest.csv:${EXPECTED_SCENARIO_MANIFEST_SHA}" \
  "${SOURCE_ROOT}/blind_run_manifest.csv:${EXPECTED_BLIND_MANIFEST_SHA}" \
  "${SOURCE_ROOT}/image_index.csv:${EXPECTED_IMAGE_INDEX_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  if [[ "$(sha256sum "${path}" | cut -d' ' -f1)" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done
for marker in \
  "${SOURCE_ROOT}/.prepare_complete" \
  "${SMOKE_ROOT}/.smoke_postflight_complete" \
  "${IMAGE_PARENT}/.conversion_complete"; do
  if [[ ! -e "${marker}" ]]; then
    echo "Required marker is missing: ${marker}" >&2
    exit 2
  fi
done

SEEDS=(13 42 97 123 211 307)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
PREPARED="${SOURCE_ROOT}/scenarios/symmetric_entry_r20/seed_${SEED}/prepared"
SOURCE_BLIND="${SOURCE_ROOT}/scenarios/symmetric_entry_r20/seed_${SEED}/blind_run"
SEED_ROOT="${EXPERIMENT_ROOT}/seed_${SEED}"
for marker in "${PREPARED}/.prepare_complete" "${SOURCE_BLIND}/.blind_run_complete"; do
  if [[ ! -e "${marker}" ]]; then
    echo "Source seed marker is missing: ${marker}" >&2
    exit 2
  fi
done
if [[ -e "${SEED_ROOT}" ]]; then
  echo "Formal seed output already exists; refusing to overwrite: ${SEED_ROOT}" >&2
  exit 2
fi

source /vol/cuda/12.5.0/setup.sh
export PYTHONPATH="${ROOT}/cxr_real_experiment:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}"

COMMON=(
  --output-dir "${SEED_ROOT}"
  --source-prepared "${PREPARED}"
  --source-blind-run "${SOURCE_BLIND}"
  --seed "${SEED}"
  --loops 8
  --top-fraction 0.20
)

echo "VinDr iterative CL oracle-cleaning formal run"
echo "Job=${SLURM_JOB_ID:-manual}; task=${SLURM_ARRAY_TASK_ID}; seed=${SEED}; loops=8"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"${PYTHON}" -u "${PROGRAM}" initialize "${COMMON[@]}"
for LOOP_ID in $(seq 1 8); do
  LOOP_PADDED="$(printf '%02d' "${LOOP_ID}")"
  LOOP_ROOT="${SEED_ROOT}/loop_${LOOP_PADDED}"
  echo "Seed ${SEED}: starting loop ${LOOP_ID} at $(date --iso-8601=seconds)"
  "${PYTHON}" -u "${PROGRAM}" select "${COMMON[@]}" --loop-id "${LOOP_ID}"
  "${PYTHON}" -u "${PROGRAM}" oracle-update "${COMMON[@]}" \
    --loop-id "${LOOP_ID}" --private-reference "${PREPARED}/private_reference.csv"
  "${PYTHON}" -u "${ENGINE}" \
    --blind-cohort "${LOOP_ROOT}/cohort_after_action_blind.csv" \
    --image-index "${SOURCE_ROOT}/image_index.csv" \
    --image-root "${IMAGE_ROOT}" \
    --output-dir "${LOOP_ROOT}/oof_after_action" \
    --seed "${SEED}" \
    --n-splits 4 \
    --epochs 100 \
    --early-stopping-patience 10 \
    --learning-rate 0.001 \
    --batch-size 32 \
    --num-workers 4 \
    --device cuda
  "${PYTHON}" - "${LOOP_ROOT}" "${SEED}" "${EXPECTED_ENGINE_SHA}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
seed = int(sys.argv[2])
expected_engine_sha = sys.argv[3]
blind = pd.read_csv(root / "selected_entries_blind.csv")
evidence = pd.read_csv(root / "oof_after_action/entry_evidence.csv")
oof = pd.read_csv(root / "oof_after_action/oof_predictions.csv")
history = pd.read_csv(root / "oof_after_action/training_history.csv")
summary = json.loads((root / "oof_after_action/blind_run_summary.json").read_text())
forbidden = [
    column for column in blind.columns
    if any(token in column.lower() for token in ["clean", "reference", "injected", "true_", "error"])
]
if forbidden:
    raise RuntimeError(f"Private columns leaked into blind selection: {forbidden}")
if blind.empty or blind["entry_key"].duplicated().any():
    raise RuntimeError("Formal loop selected no entries or duplicate entries")
if len(evidence) != 18_000 or evidence[["image_id", "label_name"]].duplicated().any():
    raise RuntimeError("Formal loop evidence is invalid")
if len(oof) != 3_000 or oof["image_id"].nunique() != 3_000:
    raise RuntimeError("Formal loop OOF predictions are invalid")
probabilities = [column for column in oof if column.startswith("probability__")]
if len(probabilities) != 6 or not np.isfinite(oof[probabilities].to_numpy()).all():
    raise RuntimeError("Formal loop OOF probabilities are incomplete")
if set(history["fold_id"]) != {0, 1, 2, 3} or history.groupby("fold_id").size().min() < 1:
    raise RuntimeError("Formal loop training history lacks a fold")
if summary["seed"] != seed or summary["entries"] != 18_000 or not summary["outcome_blind"]:
    raise RuntimeError("Formal loop blind summary is invalid")
if summary["program_sha256"] != expected_engine_sha or summary["epochs_max"] != 100:
    raise RuntimeError("Formal loop engine provenance or epoch limit failed")
if summary["early_stopping_patience"] != 10 or summary["batch_size"] != 32:
    raise RuntimeError("Formal loop training configuration differs from protocol")
PY
  "${PYTHON}" -u "${PROGRAM}" finalize-loop "${COMMON[@]}" --loop-id "${LOOP_ID}"
  echo "Seed ${SEED}: completed loop ${LOOP_ID} at $(date --iso-8601=seconds)"
done

"${PYTHON}" -u "${PROGRAM}" evaluate "${COMMON[@]}" \
  --private-reference "${PREPARED}/private_reference.csv" \
  --random-replicates 10000

"${PYTHON}" - "${SEED_ROOT}" "${SEED}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
seed = int(sys.argv[2])
blind_history = pd.read_csv(root / "state/review_history_blind.csv")
private_history = pd.read_csv(root / "state/review_history_private.csv")
trajectory = pd.read_csv(root / "iterative_trajectory.csv")
summary = json.loads((root / "seed_evaluation_summary.json").read_text())
if blind_history.empty or len(blind_history) != len(private_history):
    raise RuntimeError("Formal cumulative review histories differ")
if blind_history["entry_key"].duplicated().any() or private_history["entry_key"].duplicated().any():
    raise RuntimeError("An entry was reviewed more than once")
if len(trajectory) != 27 or set(trajectory["method"]) != {"dynamic_cl", "frozen_loop0_cl", "random_review"}:
    raise RuntimeError("Formal trajectory does not contain the expected 27 rows")
dynamic = trajectory[trajectory["method"] == "dynamic_cl"].sort_values("loop")
if dynamic["loop"].tolist() != list(range(9)):
    raise RuntimeError("Formal dynamic trajectory does not cover loops 0-8")
if not dynamic["cumulative_reviews"].is_monotonic_increasing or dynamic["cumulative_reviews"].duplicated().any():
    raise RuntimeError("Formal review budget is not strictly increasing")
if (np.diff(dynamic["true_quality"]) < -1e-12).any() or (np.diff(dynamic["cumulative_recall"]) < -1e-12).any():
    raise RuntimeError("Oracle quality or error recall decreased")
if summary["seed"] != seed or summary["loops"] != 8 or summary["final_dynamic_quality"] <= 0.80:
    raise RuntimeError("Formal seed evaluation summary is invalid")
for loop_id in range(1, 9):
    if not (root / f"loop_{loop_id:02d}/.loop_complete").is_file():
        raise RuntimeError(f"Loop marker missing: {loop_id}")
if not (root / ".seed_evaluation_complete").is_file():
    raise RuntimeError("Seed evaluation marker missing")
print(json.dumps({
    "seed": seed,
    "formal_seed_postflight": "passed",
    "cumulative_reviews": len(blind_history),
    "corrected_errors": int(private_history["true_issue"].sum()),
    "final_quality": summary["final_dynamic_quality"],
}, indent=2))
PY

echo "Seed ${SEED} completed: $(date --iso-8601=seconds)"
