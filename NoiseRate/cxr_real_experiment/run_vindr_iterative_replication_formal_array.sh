#!/bin/bash
#SBATCH --job-name=vindr-repl-iter
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_repl_iter_%A_%a.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_repl_iter_%A_%a.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2
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
BASE_PROTOCOL="${ROOT}/cxr_real_experiment/vindr_iterative_oracle_cleaning_protocol_20260805.md"
REPLICATION_PROTOCOL="${ROOT}/cxr_real_experiment/vindr_iterative_oracle_replication_protocol_20260806.md"
ENGINE="${ROOT}/cxr_real_experiment/vindr_mobilenet_oof.py"
BENCHMARK_HELPER="${ROOT}/cxr_real_experiment/vindr_known_gt_cl_benchmark.py"
SOURCE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_global_symmetric_noise_mobilenet/20260806_replication_seeds509_701_v1"
IMAGE_PARENT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/derived/png224_20260804_v1"
IMAGE_ROOT="${IMAGE_PARENT}/images"
EXPERIMENT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_iterative_oracle_cleaning/20260806_replication_seeds509_701_v1"

EXPECTED_PROGRAM_SHA="724e78b2e806c90147f2aac153f7c1561d9fd321aec5573cdc9e789026f22d0c"
EXPECTED_TEST_PROGRAM_SHA="3047f53bda9a6428034a66cc1e99d7458408697a143f04a683063f823f62cdff"
EXPECTED_BASE_PROTOCOL_SHA="69f5f0822059c64a0a4a33097d1ac7152d6251f49ba0214d9cb412a294170736"
EXPECTED_REPLICATION_PROTOCOL_SHA="873287d99a76fc95f12959b468dc7dfb1e4d28b24177f41ad296561ebec47118"
EXPECTED_ENGINE_SHA="9d1585d8d37b7c8043516889b9b3d6ddda984425f910a891d889b0c2ecd9b026"
EXPECTED_BENCHMARK_SHA="f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83"
EXPECTED_IMAGE_INDEX_SHA="645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 1 )); then
  echo "A valid array task id in 0-1 is required" >&2
  exit 2
fi
for item in \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${TEST_PROGRAM}:${EXPECTED_TEST_PROGRAM_SHA}" \
  "${BASE_PROTOCOL}:${EXPECTED_BASE_PROTOCOL_SHA}" \
  "${REPLICATION_PROTOCOL}:${EXPECTED_REPLICATION_PROTOCOL_SHA}" \
  "${ENGINE}:${EXPECTED_ENGINE_SHA}" \
  "${BENCHMARK_HELPER}:${EXPECTED_BENCHMARK_SHA}" \
  "${SOURCE_ROOT}/image_index.csv:${EXPECTED_IMAGE_INDEX_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  if [[ "$(sha256sum "${path}" | cut -d' ' -f1)" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done
for marker in "${SOURCE_ROOT}/.prepare_complete" "${IMAGE_PARENT}/.conversion_complete"; do
  if [[ ! -e "${marker}" ]]; then
    echo "Required marker is missing: ${marker}" >&2
    exit 2
  fi
done

SEEDS=(509 701)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
PREPARED="${SOURCE_ROOT}/scenarios/symmetric_entry_r20/seed_${SEED}/prepared"
SOURCE_BLIND="${SOURCE_ROOT}/scenarios/symmetric_entry_r20/seed_${SEED}/blind_run"
SEED_ROOT="${EXPERIMENT_ROOT}/seed_${SEED}"
for marker in "${PREPARED}/.prepare_complete" "${SOURCE_BLIND}/.blind_run_complete"; do
  if [[ ! -e "${marker}" ]]; then
    echo "Replication source marker is missing: ${marker}" >&2
    exit 2
  fi
done
if [[ -e "${SEED_ROOT}" ]]; then
  echo "Replication iterative output already exists: ${SEED_ROOT}" >&2
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

echo "VinDr fixed two-seed iterative oracle replication"
echo "Job=${SLURM_JOB_ID:-manual}; task=${SLURM_ARRAY_TASK_ID}; seed=${SEED}"
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
forbidden = [column for column in blind if any(token in column.lower() for token in ["clean", "reference", "injected", "true_", "error"])]
if forbidden or blind.empty or blind["entry_key"].duplicated().any():
    raise RuntimeError("Replication loop blind selection failed")
if len(evidence) != 18000 or evidence[["image_id", "label_name"]].duplicated().any():
    raise RuntimeError("Replication loop evidence failed")
if len(oof) != 3000 or oof["image_id"].nunique() != 3000:
    raise RuntimeError("Replication loop OOF rows failed")
probabilities = [column for column in oof if column.startswith("probability__")]
if len(probabilities) != 6 or not np.isfinite(oof[probabilities].to_numpy()).all():
    raise RuntimeError("Replication loop OOF probabilities failed")
if set(history["fold_id"]) != {0, 1, 2, 3} or history.groupby("fold_id").size().min() < 1:
    raise RuntimeError("Replication loop training history lacks a fold")
if summary["seed"] != seed or summary["entries"] != 18000 or not summary["outcome_blind"]:
    raise RuntimeError("Replication loop blind summary failed")
if summary["program_sha256"] != expected_engine_sha or summary["epochs_max"] != 100 or summary["early_stopping_patience"] != 10:
    raise RuntimeError("Replication loop engine provenance/configuration failed")
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
    raise RuntimeError("Replication review histories differ")
if blind_history["entry_key"].duplicated().any() or private_history["entry_key"].duplicated().any():
    raise RuntimeError("Replication reviewed an entry more than once")
if len(trajectory) != 27 or set(trajectory["method"]) != {"dynamic_cl", "frozen_loop0_cl", "random_review"}:
    raise RuntimeError("Replication trajectory rows/methods failed")
dynamic = trajectory[trajectory["method"] == "dynamic_cl"].sort_values("loop")
if dynamic["loop"].tolist() != list(range(9)) or dynamic["cumulative_reviews"].duplicated().any():
    raise RuntimeError("Replication dynamic trajectory loops/budgets failed")
if (np.diff(dynamic["true_quality"]) < -1e-12).any() or (np.diff(dynamic["cumulative_recall"]) < -1e-12).any():
    raise RuntimeError("Replication oracle quality or recall decreased")
if summary["seed"] != seed or summary["loops"] != 8 or summary["final_dynamic_quality"] <= 0.80:
    raise RuntimeError("Replication seed evaluation failed")
for loop_id in range(1, 9):
    if not (root / f"loop_{loop_id:02d}/.loop_complete").is_file():
        raise RuntimeError(f"Replication loop marker missing: {loop_id}")
if not (root / ".seed_evaluation_complete").is_file():
    raise RuntimeError("Replication seed evaluation marker is missing")
print(json.dumps({
    "seed": seed,
    "replication_seed_postflight": "passed",
    "cumulative_reviews": len(blind_history),
    "corrected_errors": int(private_history["true_issue"].sum()),
    "final_quality": summary["final_dynamic_quality"],
}, indent=2))
PY

echo "Seed ${SEED} completed: $(date --iso-8601=seconds)"
