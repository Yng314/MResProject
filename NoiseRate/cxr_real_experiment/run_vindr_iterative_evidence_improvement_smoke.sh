#!/bin/bash
#SBATCH --job-name=vindr-evid-smoke
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_evid_smoke_%A_%a.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_evid_smoke_%A_%a.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
SCRIPT_DIR="${ROOT}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
ENGINE="${SCRIPT_DIR}/vindr_densenet_oof.py"
DRIVER="${SCRIPT_DIR}/vindr_iterative_evidence_improvement.py"
TEST="${SCRIPT_DIR}/test_vindr_densenet_oof.py"
PROTOCOL="${SCRIPT_DIR}/vindr_iterative_evidence_improvement_protocol_20260813.md"
SOURCE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_global_symmetric_noise_mobilenet/20260805_v1"
PREPARED="${SOURCE_ROOT}/scenarios/symmetric_entry_r20/seed_13/prepared"
IMAGE_INDEX="${SOURCE_ROOT}/image_index.csv"
IMAGE_ROOT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/derived/png224_20260804_v1/images"
XRV_CACHE="/vol/gpudata/yz3522-llmtest/.cache/torchxrayvision"
XRV_WEIGHT="${XRV_CACHE}/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_iterative_evidence_improvement_smoke/20260813_v2"

EXPECTED_ENGINE_SHA="fd7a8a3141680d27fe20955e17e7f77437ee0d9a6f0d18a45b16be628c9627bc"
EXPECTED_DRIVER_SHA="ad42522d355beaebaa1a19f441b7c281926c4936bb0a8ec52b9edef48489e6a6"
EXPECTED_TEST_SHA="3d08171fb507cc391e9389fc94770c6c3623afb8db043cd64e2767236ad63088"
EXPECTED_PROTOCOL_SHA="be5a0db22997a91afb13e9588148afa4aaacb901d704b0fe87acab548707c807"
EXPECTED_IMAGE_INDEX_SHA="645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2"
EXPECTED_WEIGHT_SHA="56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899"
EXPECTED_BLIND_SHA="685f33adb62786da2ad0b84725e70d5b88f1aab78ed6773fd6b9c626353e46c5"
EXPECTED_PRIVATE_SHA="e8b2e3c467824326e61116ad3bce487da8a0818712c457f122dd054f66959082"

[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[01]$ ]] || { echo "Array task 0 or 1 is required" >&2; exit 2; }
BRANCHES=(scratch xrv_pretrained)
LEARNING_RATES=(0.001 0.0001)
BRANCH="${BRANCHES[$SLURM_ARRAY_TASK_ID]}"
LEARNING_RATE="${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}"
OUTPUT="${OUTPUT_ROOT}/${BRANCH}/seed_13"

for item in \
  "${ENGINE}:${EXPECTED_ENGINE_SHA}" \
  "${DRIVER}:${EXPECTED_DRIVER_SHA}" \
  "${TEST}:${EXPECTED_TEST_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${IMAGE_INDEX}:${EXPECTED_IMAGE_INDEX_SHA}" \
  "${XRV_WEIGHT}:${EXPECTED_WEIGHT_SHA}" \
  "${PREPARED}/blind_noisy_cohort.csv:${EXPECTED_BLIND_SHA}" \
  "${PREPARED}/private_reference.csv:${EXPECTED_PRIVATE_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  [[ "$(sha256sum "${path}" | cut -d' ' -f1)" == "${expected}" ]] || {
    echo "Hash mismatch: ${path}" >&2
    exit 2
  }
done
[[ -e "${PREPARED}/.prepare_complete" ]] || { echo "Prepared marker missing" >&2; exit 2; }
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite ${OUTPUT}" >&2; exit 2; }

source /vol/cuda/12.5.0/setup.sh
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
mkdir -p "$(dirname "${OUTPUT}")" "${MPLCONFIGDIR}"

driver_args=(
  --output-dir "${OUTPUT}"
  --source-prepared "${PREPARED}"
  --private-reference "${PREPARED}/private_reference.csv"
  --seed 13
  --initialization "${BRANCH}"
  --loops 1
  --review-budget 360
  --probe-review-budget 72
)

run_oof() {
  local cohort="$1"
  local target="$2"
  "${PYTHON}" -u "${ENGINE}" \
    --blind-cohort "${cohort}" \
    --split-reference-cohort "${PREPARED}/blind_noisy_cohort.csv" \
    --image-index "${IMAGE_INDEX}" \
    --image-root "${IMAGE_ROOT}" \
    --output-dir "${target}" \
    --seed 13 \
    --initialization "${BRANCH}" \
    --xrv-cache-dir "${XRV_CACHE}" \
    --n-splits 4 \
    --epochs 2 \
    --early-stopping-patience 2 \
    --learning-rate "${LEARNING_RATE}" \
    --batch-size 32 \
    --num-workers 4 \
    --device cuda
}

echo "Smoke branch=${BRANCH}; job=${SLURM_JOB_ID:-manual}; start=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
"${PYTHON}" -m unittest -v test_vindr_densenet_oof
"${PYTHON}" -u "${DRIVER}" initialize "${driver_args[@]}"
run_oof "${PREPARED}/blind_noisy_cohort.csv" "${OUTPUT}/loop_00/oof_evidence"
"${PYTHON}" -u "${DRIVER}" select "${driver_args[@]}" --loop-id 1
"${PYTHON}" -u "${DRIVER}" oracle-update "${driver_args[@]}" --loop-id 1
run_oof "${OUTPUT}/loop_01/cohort_after_action_blind.csv" "${OUTPUT}/loop_01/oof_after_action"
"${PYTHON}" -u "${DRIVER}" finalize-loop "${driver_args[@]}" --loop-id 1
"${PYTHON}" -u "${DRIVER}" evaluate "${driver_args[@]}"

"${PYTHON}" - "${OUTPUT}" "${BRANCH}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root, branch = Path(sys.argv[1]), sys.argv[2]
history = pd.read_csv(root / "state/review_history_private.csv")
probe = set(pd.read_csv(root / "state/probe_images_blind.csv")["image_id"].astype(str))
summary = json.loads((root / "seed_evaluation_summary.json").read_text())
if len(history) != 360 or history["entry_key"].duplicated().any():
    raise RuntimeError("Smoke review history failed")
if set(history["image_id"].astype(str)) & probe:
    raise RuntimeError("Smoke selected a fixed-probe entry")
if summary["initialization"] != branch or summary["loops"] != 1:
    raise RuntimeError("Smoke summary provenance failed")
for loop_id in range(2):
    evidence = root / f"loop_{loop_id:02d}" / ("oof_evidence" if loop_id == 0 else "oof_after_action")
    if not (evidence / ".blind_run_complete").is_file():
        raise RuntimeError(f"Smoke OOF marker missing: {evidence}")
print("Smoke postflight passed")
PY
printf 'job_id=%s\nbranch=%s\n' "${SLURM_JOB_ID:-manual}" "${BRANCH}" > "${OUTPUT}/.smoke_branch_complete"
echo "Smoke completed: $(date --iso-8601=seconds)"
