#!/bin/bash
#SBATCH --job-name=vindr-evid-loop
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_evid_loop_%A_%a.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_evid_loop_%A_%a.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
SCRIPT_DIR="${ROOT}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
ENGINE="${SCRIPT_DIR}/vindr_densenet_oof.py"
DRIVER="${SCRIPT_DIR}/vindr_iterative_evidence_improvement.py"
PROTOCOL="${SCRIPT_DIR}/vindr_iterative_evidence_improvement_protocol_20260813.md"
SOURCE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_global_symmetric_noise_mobilenet/20260805_v1"
SCENARIO_ROOT="${SOURCE_ROOT}/scenarios/symmetric_entry_r20"
IMAGE_INDEX="${SOURCE_ROOT}/image_index.csv"
IMAGE_ROOT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/derived/png224_20260804_v1/images"
XRV_CACHE="/vol/gpudata/yz3522-llmtest/.cache/torchxrayvision"
XRV_WEIGHT="${XRV_CACHE}/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
SMOKE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_iterative_evidence_improvement_smoke/20260813_v2"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_iterative_evidence_improvement/20260813_v2"

EXPECTED_ENGINE_SHA="fd7a8a3141680d27fe20955e17e7f77437ee0d9a6f0d18a45b16be628c9627bc"
EXPECTED_DRIVER_SHA="ad42522d355beaebaa1a19f441b7c281926c4936bb0a8ec52b9edef48489e6a6"
EXPECTED_PROTOCOL_SHA="be5a0db22997a91afb13e9588148afa4aaacb901d704b0fe87acab548707c807"
EXPECTED_IMAGE_INDEX_SHA="645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2"
EXPECTED_WEIGHT_SHA="56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899"

SEEDS=(13 42 97 123 211 307)
BLIND_SHAS=(
  685f33adb62786da2ad0b84725e70d5b88f1aab78ed6773fd6b9c626353e46c5
  32f0b895b6401c1952840613b680cc70ca3514a7ff3a8569ec0aba1dec64e93f
  63ed7c1899a298f21bbdd563257904f1e35c4d68f013a09b75f79dc657c59425
  ecc7b3623ec8ac841e41dbbb4162458e91e8bf6317abe257c7df1e497fc692f4
  1f9b728404106e34f6e1d4fbf98a0fe93f7bd418b808c229efb53fb6c7f51e0a
  c5a82e479668d116104f446e06b5c024053e30580e986da4a8c82c5bd769184e
)
PRIVATE_SHAS=(
  e8b2e3c467824326e61116ad3bce487da8a0818712c457f122dd054f66959082
  c0af8b5d44a6cbca40d260616cd3403e59ab2ebd1d1bf081981df41b47d3c4d1
  bbaae3612894a4a7712667c4db0306eb835eafc1a5cae16ee072b657219fd657
  16208ab0edd114b563619baf5bfb742937a413d64b05580a63225977445a2e64
  39ebf8c3df186018569e0d9931ca5a8c03cfea213faf59252f4093f140e0e3f6
  4e0e5c3742bf571d3a15c94718b8a79bd923bb90a3f1ca76f8a5568422f26473
)

[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-2]$ ]] || { echo "Worker id 0-2 is required" >&2; exit 2; }
for smoke_branch in scratch xrv_pretrained; do
  [[ -e "${SMOKE_ROOT}/${smoke_branch}/seed_13/.smoke_branch_complete" ]] || {
    echo "Smoke branch is incomplete: ${smoke_branch}" >&2
    exit 2
  }
done
for item in \
  "${ENGINE}:${EXPECTED_ENGINE_SHA}" \
  "${DRIVER}:${EXPECTED_DRIVER_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${IMAGE_INDEX}:${EXPECTED_IMAGE_INDEX_SHA}" \
  "${XRV_WEIGHT}:${EXPECTED_WEIGHT_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  [[ "$(sha256sum "${path}" | cut -d' ' -f1)" == "${expected}" ]] || {
    echo "Hash mismatch: ${path}" >&2
    exit 2
  }
done

source /vol/cuda/12.5.0/setup.sh
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
mkdir -p "${OUTPUT_ROOT}" "${MPLCONFIGDIR}"

run_one() {
  local run_index="$1"
  local seed_index=$((run_index % 6))
  local seed="${SEEDS[$seed_index]}"
  local branch learning_rate
  if (( run_index < 6 )); then
    branch="scratch"
    learning_rate="0.001"
  else
    branch="xrv_pretrained"
    learning_rate="0.0001"
  fi
  local prepared="${SCENARIO_ROOT}/seed_${seed}/prepared"
  local output="${OUTPUT_ROOT}/${branch}/seed_${seed}"
  [[ "$(sha256sum "${prepared}/blind_noisy_cohort.csv" | cut -d' ' -f1)" == "${BLIND_SHAS[$seed_index]}" ]] || {
    echo "Blind cohort hash mismatch: seed ${seed}" >&2; exit 2;
  }
  [[ "$(sha256sum "${prepared}/private_reference.csv" | cut -d' ' -f1)" == "${PRIVATE_SHAS[$seed_index]}" ]] || {
    echo "Private reference hash mismatch: seed ${seed}" >&2; exit 2;
  }
  [[ -e "${prepared}/.prepare_complete" ]] || { echo "Prepared marker missing: seed ${seed}" >&2; exit 2; }
  [[ ! -e "${output}" ]] || { echo "Refusing to overwrite ${output}" >&2; exit 2; }
  mkdir -p "$(dirname "${output}")"

  local driver_args=(
    --output-dir "${output}"
    --source-prepared "${prepared}"
    --private-reference "${prepared}/private_reference.csv"
    --seed "${seed}"
    --initialization "${branch}"
    --loops 8
    --review-budget 360
    --probe-review-budget 72
  )
  run_oof() {
    local cohort="$1"
    local target="$2"
    "${PYTHON}" -u "${ENGINE}" \
      --blind-cohort "${cohort}" \
      --split-reference-cohort "${prepared}/blind_noisy_cohort.csv" \
      --image-index "${IMAGE_INDEX}" \
      --image-root "${IMAGE_ROOT}" \
      --output-dir "${target}" \
      --seed "${seed}" \
      --initialization "${branch}" \
      --xrv-cache-dir "${XRV_CACHE}" \
      --n-splits 4 \
      --epochs 50 \
      --early-stopping-patience 8 \
      --learning-rate "${learning_rate}" \
      --batch-size 32 \
      --num-workers 4 \
      --device cuda
  }

  echo "Run index=${run_index}; branch=${branch}; seed=${seed}; start=$(date --iso-8601=seconds)"
  "${PYTHON}" -u "${DRIVER}" initialize "${driver_args[@]}"
  run_oof "${prepared}/blind_noisy_cohort.csv" "${output}/loop_00/oof_evidence"
  for loop_id in $(seq 1 8); do
    "${PYTHON}" -u "${DRIVER}" select "${driver_args[@]}" --loop-id "${loop_id}"
    "${PYTHON}" -u "${DRIVER}" oracle-update "${driver_args[@]}" --loop-id "${loop_id}"
    run_oof "${output}/loop_$(printf '%02d' "${loop_id}")/cohort_after_action_blind.csv" \
      "${output}/loop_$(printf '%02d' "${loop_id}")/oof_after_action"
    "${PYTHON}" -u "${DRIVER}" finalize-loop "${driver_args[@]}" --loop-id "${loop_id}"
  done
  "${PYTHON}" -u "${DRIVER}" evaluate "${driver_args[@]}"

  "${PYTHON}" - "${output}" "${branch}" "${seed}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root, branch, seed = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
history = pd.read_csv(root / "state/review_history_private.csv")
probe = set(pd.read_csv(root / "state/probe_images_blind.csv")["image_id"].astype(str))
summary = json.loads((root / "seed_evaluation_summary.json").read_text())
if len(history) != 2880 or history["entry_key"].duplicated().any():
    raise RuntimeError("Formal review history failed")
if set(history["image_id"].astype(str)) & probe:
    raise RuntimeError("Formal run selected a fixed-probe entry")
if sorted(history["loop"].unique()) != list(range(1, 9)):
    raise RuntimeError("Formal loop coverage failed")
if summary["initialization"] != branch or summary["seed"] != seed or summary["loops"] != 8:
    raise RuntimeError("Formal summary provenance failed")
for loop_id in range(9):
    evidence = root / f"loop_{loop_id:02d}" / ("oof_evidence" if loop_id == 0 else "oof_after_action")
    frame = pd.read_csv(evidence / "entry_evidence.csv")
    if len(frame) != 18000 or frame[["image_id", "label_name"]].duplicated().any():
        raise RuntimeError(f"Formal evidence failed at Loop {loop_id}")
print("Formal postflight passed")
PY
  printf 'job_id=%s\nworker=%s\nbranch=%s\nseed=%s\n' \
    "${SLURM_JOB_ID:-manual}" "${SLURM_ARRAY_TASK_ID}" "${branch}" "${seed}" \
    > "${output}/.formal_run_complete"
  echo "Run index=${run_index} completed: $(date --iso-8601=seconds)"
}

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
for run_index in \
  "${SLURM_ARRAY_TASK_ID}" \
  "$((SLURM_ARRAY_TASK_ID + 3))" \
  "$((SLURM_ARRAY_TASK_ID + 6))" \
  "$((SLURM_ARRAY_TASK_ID + 9))"; do
  run_one "${run_index}"
done
printf 'job_id=%s\nworker=%s\n' "${SLURM_JOB_ID:-manual}" "${SLURM_ARRAY_TASK_ID}" \
  > "${OUTPUT_ROOT}/.worker_${SLURM_ARRAY_TASK_ID}_complete"

all_complete=true
for worker in 0 1 2; do
  [[ -e "${OUTPUT_ROOT}/.worker_${worker}_complete" ]] || all_complete=false
done
if [[ "${all_complete}" == true ]] && mkdir "${OUTPUT_ROOT}/.aggregate_lock" 2>/dev/null; then
  "${PYTHON}" -u "${DRIVER}" aggregate \
    --experiment-root "${OUTPUT_ROOT}" \
    --aggregate-output "${OUTPUT_ROOT}/aggregate" \
    --seeds 13,42,97,123,211,307
  [[ -e "${OUTPUT_ROOT}/aggregate/.aggregate_complete" ]] || { echo "Aggregate marker missing" >&2; exit 2; }
  printf 'job_id=%s\n' "${SLURM_JOB_ID:-manual}" > "${OUTPUT_ROOT}/.complete"
fi
echo "Worker ${SLURM_ARRAY_TASK_ID} completed: $(date --iso-8601=seconds)"
