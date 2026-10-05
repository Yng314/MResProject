#!/bin/bash
#SBATCH --job-name=vindr-fullpool
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_fullpool_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_fullpool_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
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
DRIVER="${SCRIPT_DIR}/vindr_full_issue_pool_iteration.py"
ENGINE="${SCRIPT_DIR}/vindr_densenet_sentinel_oof.py"
BASE_ENGINE="${SCRIPT_DIR}/vindr_densenet_oof.py"
PROTOCOL="${SCRIPT_DIR}/vindr_full_issue_pool_iteration_protocol_20260819.md"
PREPARED="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_self_optimization_stress_prepared/20260814_v1/scenarios/hard_r30/seed_11003/prepared"
INITIAL_EVIDENCE="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_self_optimization_stress/20260814_v1/scratch/hard_r30/seed_11003/loop_00/oof_evidence"
IMAGE_INDEX="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_self_optimization_stress_prepared/20260814_v1/image_index.csv"
IMAGE_ROOT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/derived/png224_20260804_v1/images"
XRV_CACHE="/vol/gpudata/yz3522-llmtest/.cache/torchxrayvision"
XRV_WEIGHT="${XRV_CACHE}/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_full_issue_pool_iteration/20260819_v1"
OUTPUT="${OUTPUT_ROOT}/seed_11003"

EXPECTED_DRIVER_SHA="d55f354fd58791e249e74f7ea985b6ef9b415f79fe65289bba1f78ab01112386"
EXPECTED_ENGINE_SHA="faf2e31edaef6150c0600a1c07d0633e0292ef995dd3bca74179518a596470d4"
EXPECTED_BASE_ENGINE_SHA="fd7a8a3141680d27fe20955e17e7f77437ee0d9a6f0d18a45b16be628c9627bc"
EXPECTED_PROTOCOL_SHA="0a02f7409e0f2ef3ba44190b6963fc94ce34614221d68cbb08fa08941135b5d2"
EXPECTED_IMAGE_INDEX_SHA="645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2"
EXPECTED_INITIAL_EVIDENCE_SHA="c40ece2e1120bb3d849f1fc0dd8c50a67edf7e0af09e25f58ccd45bfac26cdf8"
EXPECTED_WEIGHT_SHA="56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899"

for item in \
  "${DRIVER}:${EXPECTED_DRIVER_SHA}" \
  "${ENGINE}:${EXPECTED_ENGINE_SHA}" \
  "${BASE_ENGINE}:${EXPECTED_BASE_ENGINE_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${IMAGE_INDEX}:${EXPECTED_IMAGE_INDEX_SHA}" \
  "${INITIAL_EVIDENCE}/entry_evidence.csv:${EXPECTED_INITIAL_EVIDENCE_SHA}" \
  "${XRV_WEIGHT}:${EXPECTED_WEIGHT_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  [[ "$(sha256sum "${path}" | cut -d' ' -f1)" == "${expected}" ]] || {
    echo "Hash mismatch: ${path}" >&2
    exit 2
  }
done
[[ -e "${PREPARED}/.prepare_complete" ]] || { echo "Prepared marker missing" >&2; exit 2; }
[[ -e "${INITIAL_EVIDENCE}/.blind_run_complete" ]] || { echo "Initial evidence marker missing" >&2; exit 2; }
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Refusing to overwrite ${OUTPUT_ROOT}" >&2; exit 2; }

source /vol/cuda/12.5.0/setup.sh
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
mkdir -p "$(dirname "${OUTPUT_ROOT}")" "${MPLCONFIGDIR}"

COMMON=(
  --output-dir "${OUTPUT}"
  --source-prepared "${PREPARED}"
  --private-reference "${PREPARED}/private_reference.csv"
  --sentinel-private-reference "${PREPARED}/sentinel_private_reference.csv"
  --initial-evidence-dir "${INITIAL_EVIDENCE}"
  --seed 11003
  --loops 5
)
run_oof() {
  local cohort="$1"
  local target="$2"
  "${PYTHON}" -u "${ENGINE}" \
    --blind-cohort "${cohort}" \
    --sentinel-cohort "${PREPARED}/sentinel_blind_cohort.csv" \
    --split-reference-cohort "${PREPARED}/blind_noisy_cohort.csv" \
    --image-index "${IMAGE_INDEX}" \
    --image-root "${IMAGE_ROOT}" \
    --output-dir "${target}" \
    --seed 11003 \
    --initialization scratch \
    --xrv-cache-dir "${XRV_CACHE}" \
    --n-splits 4 \
    --epochs 50 \
    --early-stopping-patience 8 \
    --learning-rate 0.001 \
    --batch-size 32 \
    --num-workers 4 \
    --device cuda
}

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
"${PYTHON}" -u "${DRIVER}" initialize "${COMMON[@]}"
for loop_id in $(seq 1 5); do
  "${PYTHON}" -u "${DRIVER}" select "${COMMON[@]}" --loop-id "${loop_id}"
  "${PYTHON}" -u "${DRIVER}" oracle-update "${COMMON[@]}" --loop-id "${loop_id}"
  run_oof \
    "${OUTPUT}/loop_$(printf '%02d' "${loop_id}")/cohort_after_action_blind.csv" \
    "${OUTPUT}/loop_$(printf '%02d' "${loop_id}")/oof_after_action"
  "${PYTHON}" -u "${DRIVER}" finalize-loop "${COMMON[@]}" --loop-id "${loop_id}"
done
"${PYTHON}" -u "${DRIVER}" evaluate "${COMMON[@]}"
"${PYTHON}" -u "${DRIVER}" aggregate \
  --experiment-root "${OUTPUT_ROOT}" \
  --aggregate-output "${OUTPUT_ROOT}/aggregate" \
  --seeds 11003

"${PYTHON}" - "${OUTPUT}" "${OUTPUT_ROOT}" <<'PY'
from pathlib import Path
import json
import pandas as pd
import sys

run, root = Path(sys.argv[1]), Path(sys.argv[2])
history = pd.read_csv(run / "state/review_history_private.csv")
manifest = json.loads((run / "state/initialization_manifest.json").read_text())
loop1 = pd.read_csv(run / "loop_01/selected_entries_private.csv")
if len(loop1) != int(manifest["initial_issue_pool"]):
    raise RuntimeError("Loop 1 did not review the complete initial issue pool")
if history["entry_key"].duplicated().any():
    raise RuntimeError("A reviewed entry re-entered a later issue pool")
if sorted(history["loop"].unique()) != [1, 2, 3, 4, 5]:
    raise RuntimeError("Loop history is incomplete")
for loop_id in range(1, 6):
    selection = pd.read_csv(run / f"loop_{loop_id:02d}/selected_entries_blind.csv")
    if not selection["cl_issue"].astype(bool).all():
        raise RuntimeError(f"Loop {loop_id} selection escaped the issue pool")
    evidence = run / f"loop_{loop_id:02d}/oof_after_action"
    if len(pd.read_csv(evidence / "entry_evidence.csv")) != 14400:
        raise RuntimeError(f"Loop {loop_id} action evidence failed")
    if len(pd.read_csv(evidence / "sentinel_entry_evidence.csv")) != 3600:
        raise RuntimeError(f"Loop {loop_id} sentinel evidence failed")
if not (run / ".seed_evaluation_complete").is_file():
    raise RuntimeError("Seed evaluation marker missing")
if not (root / "aggregate/.aggregate_complete").is_file():
    raise RuntimeError("Aggregate marker missing")
print("Full issue-pool pilot postflight passed")
PY
printf 'job_id=%s\nseed=11003\n' "${SLURM_JOB_ID:-manual}" > "${OUTPUT_ROOT}/.complete"
echo "Single-seed full issue-pool pilot completed"
