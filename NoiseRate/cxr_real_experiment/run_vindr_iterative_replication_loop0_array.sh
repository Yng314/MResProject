#!/bin/bash
#SBATCH --job-name=vindr-repl-loop0
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_repl_loop0_%A_%a.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_repl_loop0_%A_%a.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
ENGINE="${ROOT}/cxr_real_experiment/vindr_mobilenet_oof.py"
ENGINE_TEST="${ROOT}/cxr_real_experiment/test_vindr_mobilenet_oof.py"
REPLICATION_PROTOCOL="${ROOT}/cxr_real_experiment/vindr_iterative_oracle_replication_protocol_20260806.md"
SOURCE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_global_symmetric_noise_mobilenet/20260806_replication_seeds509_701_v1"
IMAGE_PARENT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/derived/png224_20260804_v1"
IMAGE_ROOT="${IMAGE_PARENT}/images"

EXPECTED_ENGINE_SHA="9d1585d8d37b7c8043516889b9b3d6ddda984425f910a891d889b0c2ecd9b026"
EXPECTED_ENGINE_TEST_SHA="f6d866ad34f3268af6c1ab854a1e3adf88cd1ed6a6f3e715d993228dc3dacf91"
EXPECTED_REPLICATION_PROTOCOL_SHA="873287d99a76fc95f12959b468dc7dfb1e4d28b24177f41ad296561ebec47118"
EXPECTED_IMAGE_INDEX_SHA="645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 1 )); then
  echo "A valid array task id in 0-1 is required" >&2
  exit 2
fi
for item in \
  "${ENGINE}:${EXPECTED_ENGINE_SHA}" \
  "${ENGINE_TEST}:${EXPECTED_ENGINE_TEST_SHA}" \
  "${REPLICATION_PROTOCOL}:${EXPECTED_REPLICATION_PROTOCOL_SHA}" \
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
RUN_ROOT="${SOURCE_ROOT}/scenarios/symmetric_entry_r20/seed_${SEED}/blind_run"
if [[ ! -e "${PREPARED}/.prepare_complete" ]]; then
  echo "Prepared seed marker is missing: ${SEED}" >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Loop-0 output already exists: ${RUN_ROOT}" >&2
  exit 2
fi

source /vol/cuda/12.5.0/setup.sh
export PYTHONPATH="${ROOT}/cxr_real_experiment:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}"

echo "VinDr replication Loop-0 OOF"
echo "Job=${SLURM_JOB_ID:-manual}; task=${SLURM_ARRAY_TASK_ID}; seed=${SEED}"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"${PYTHON}" -u "${ENGINE}" \
  --blind-cohort "${PREPARED}/blind_noisy_cohort.csv" \
  --image-index "${SOURCE_ROOT}/image_index.csv" \
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

"${PYTHON}" - "${PREPARED}" "${RUN_ROOT}" "${SEED}" "${EXPECTED_ENGINE_SHA}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

prepared, run = map(Path, sys.argv[1:3])
seed = int(sys.argv[3])
expected_engine_sha = sys.argv[4]
prepare = json.loads((prepared / "prepare_summary.json").read_text())
summary = json.loads((run / "blind_run_summary.json").read_text())
evidence = pd.read_csv(run / "entry_evidence.csv")
oof = pd.read_csv(run / "oof_predictions.csv")
history = pd.read_csv(run / "training_history.csv")
if len(evidence) != 18000 or evidence[["image_id", "label_name"]].duplicated().any():
    raise RuntimeError("Replication Loop-0 evidence failed")
if len(oof) != 3000 or oof["image_id"].nunique() != 3000:
    raise RuntimeError("Replication Loop-0 OOF rows failed")
probabilities = [column for column in oof if column.startswith("probability__")]
if len(probabilities) != 6 or not np.isfinite(oof[probabilities].to_numpy()).all():
    raise RuntimeError("Replication Loop-0 probabilities failed")
for frame, name in [(evidence.drop(columns=["cl_issue"], errors="ignore"), "evidence"), (oof, "OOF")]:
    forbidden = [column for column in frame if any(token in column.lower() for token in ["clean", "reference", "injected", "true_", "error"])]
    if forbidden:
        raise RuntimeError(f"Outcome fields leaked into {name}: {forbidden}")
if set(history["fold_id"]) != {0, 1, 2, 3} or history.groupby("fold_id").size().min() < 1:
    raise RuntimeError("Replication Loop-0 history lacks a fold")
if summary["seed"] != seed or summary["entries"] != 18000 or not summary["outcome_blind"]:
    raise RuntimeError("Replication Loop-0 summary failed")
if summary["program_sha256"] != expected_engine_sha or summary["epochs_max"] != 100 or summary["early_stopping_patience"] != 10:
    raise RuntimeError("Replication Loop-0 provenance/configuration failed")
if summary["blind_cohort_sha256"] != prepare["blind_cohort_sha256"]:
    raise RuntimeError("Replication Loop-0 cohort differs from prepared input")
if int(summary["cl_issue_entries"]) <= 0 or not (run / ".blind_run_complete").is_file():
    raise RuntimeError("Replication Loop-0 CL evidence or marker is missing")
print(json.dumps({"seed": seed, "loop0_postflight": "passed", "cl_issue_entries": summary["cl_issue_entries"]}, indent=2))
PY

echo "Completed: $(date --iso-8601=seconds)"
