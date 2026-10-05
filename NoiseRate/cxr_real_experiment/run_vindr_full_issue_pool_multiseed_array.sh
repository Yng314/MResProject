#!/bin/bash
#SBATCH --job-name=vfull-ms
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_fullpool_ms_%A_%a.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_fullpool_ms_%A_%a.err
#SBATCH --partition=a30,a40,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-6%3
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
PROTOCOL="${SCRIPT_DIR}/vindr_full_issue_pool_multiseed_protocol_20260819.md"
PREP_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_self_optimization_stress_prepared/20260814_v1/scenarios/hard_r30"
EVIDENCE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_self_optimization_stress/20260814_v1/scratch/hard_r30"
IMAGE_INDEX="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_self_optimization_stress_prepared/20260814_v1/image_index.csv"
IMAGE_ROOT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/derived/png224_20260804_v1/images"
XRV_CACHE="/vol/gpudata/yz3522-llmtest/.cache/torchxrayvision"
XRV_WEIGHT="${XRV_CACHE}/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_full_issue_pool_iteration/20260819_multiseed_v1"

SEEDS=(13007 17011 19001 23003 27011 31013 37003)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}]}"
PREPARED="${PREP_ROOT}/seed_${SEED}/prepared"
INITIAL_EVIDENCE="${EVIDENCE_ROOT}/seed_${SEED}/loop_00/oof_evidence"
OUTPUT="${OUTPUT_ROOT}/seed_${SEED}"

EXPECTED_DRIVER_SHA="d55f354fd58791e249e74f7ea985b6ef9b415f79fe65289bba1f78ab01112386"
EXPECTED_ENGINE_SHA="faf2e31edaef6150c0600a1c07d0633e0292ef995dd3bca74179518a596470d4"
EXPECTED_BASE_ENGINE_SHA="fd7a8a3141680d27fe20955e17e7f77437ee0d9a6f0d18a45b16be628c9627bc"
EXPECTED_PROTOCOL_SHA="9958b597297ab92102246d046aa9c0369b9646f9e963035ceb0ea2208c083e02"
EXPECTED_IMAGE_INDEX_SHA="645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2"
EXPECTED_WEIGHT_SHA="56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899"

declare -A EXPECTED_COHORT_SHA=(
  [13007]="10ac3250f1b7a307c2ab78af32f50513da09c0b2efa4f0b48c9accd4da74ca39"
  [17011]="d3b296d5baca316a3222967f91381887e62e646ecbf52f029202686a2ca1aa7c"
  [19001]="d488fec48889572e6c8bfdd1093f6736d7e2d47fce23cfe4982ec7cc1ef0f7ad"
  [23003]="7381956ccefb3f9979bf3aaf5245763a29ea4de6ffee1286cd181a7a9c30fdb5"
  [27011]="7422d504e53520da9e5def161d98137fec338154b879990387bbc66ead131403"
  [31013]="2c41b1a6ff9b13c9a05358a6d96c61ef221e4219c7b842f6f0627fbbd0a2a89a"
  [37003]="ea231c87b346d2ed0510c5d6bc496d4298723d1fc21f790f9b58ee37dedac23b"
)
declare -A EXPECTED_PRIVATE_SHA=(
  [13007]="dcc53d484f95556b1c4f8ca0a791449f6914e20562d753e63aa371e67f03850b"
  [17011]="c40d550567e8894e0bf6d61148a09f4e8f56091b92b2d452d77b14e61de5ab6d"
  [19001]="9d53826fce82df2d1afd1131c9fa87559ba5bdd0e37a5df9b91b43b465bcfd37"
  [23003]="371888fc082afd49076ebea7184a951405bc5a3bab92804b47e4643ba9fabb95"
  [27011]="82ef79e5952980b2aae6c268084031cd92b1e6101a83417dd2d996c91e39caed"
  [31013]="1010fd1a60e40a6790a83f02970db1f21999b0ac3ecfd25487429993cff68b16"
  [37003]="d86af7ac670e26edf162114ec7170dcff48002b20de0c19188180bcac2ecfb1c"
)
declare -A EXPECTED_SENTINEL_SHA=(
  [13007]="7c0293c8943a61186d0ad79db5660e5ce55570827e7c992b02ac6c9f193de77e"
  [17011]="01855d3a1cb1480aa006865639852562f196e686b264df7d579f7a78b8ee224f"
  [19001]="a24423849bae9c37534764a23f4aa7dad83b38d2e689fc948c467808fc99e122"
  [23003]="741c1af0f4e56f0c1fbffc87b28513d132790a214871daf7ea0e38d2dabaf859"
  [27011]="f54127d8abf56bebe3c5623100cb697ff1bb86adeca846203395ccefd5540b5c"
  [31013]="38052b24c23983ee39b7ebd4c2189e1ab38249eeeb5d8d218be46e7d90c46425"
  [37003]="a67f8f0d762b4cdb10f21c473af3cde3c5bb53c7972dcae7f58e408c86caf2a2"
)
declare -A EXPECTED_EVIDENCE_SHA=(
  [13007]="32fdd026b47030cc757c1b72ed338d41ffc52ea93e036b93bedd172d06627102"
  [17011]="4fd78fa2ac1b7d1757d4d291519fed10fb6c75941f56735ef0245bc36e075a41"
  [19001]="129ec11fadc8ab55c3e7016b2c6ad1e190e1067a66b6df4e237d72e333e48674"
  [23003]="10c3521ce01a1ed82ff85fb83382a9637faeb211472b87f8f609d2b07f444973"
  [27011]="c84613efb6bcc870812e1b6b0a76eef8f494f81ddd8f35faf38dc286e5a51170"
  [31013]="ede2a7930c3f55614c0f2c437b9cdae92a522909268f5a169bd1267fc55273a1"
  [37003]="a79ad341bc72b10809306cce87738b1ce0ddf524db1dbf79b3857e124d6ec728"
)

check_hash() {
  local path="$1"
  local expected="$2"
  [[ "$(sha256sum "${path}" | cut -d' ' -f1)" == "${expected}" ]] || {
    echo "Hash mismatch: ${path}" >&2
    exit 2
  }
}

check_hash "${DRIVER}" "${EXPECTED_DRIVER_SHA}"
check_hash "${ENGINE}" "${EXPECTED_ENGINE_SHA}"
check_hash "${BASE_ENGINE}" "${EXPECTED_BASE_ENGINE_SHA}"
check_hash "${PROTOCOL}" "${EXPECTED_PROTOCOL_SHA}"
check_hash "${IMAGE_INDEX}" "${EXPECTED_IMAGE_INDEX_SHA}"
check_hash "${XRV_WEIGHT}" "${EXPECTED_WEIGHT_SHA}"
check_hash "${PREPARED}/blind_noisy_cohort.csv" "${EXPECTED_COHORT_SHA[${SEED}]}"
check_hash "${PREPARED}/private_reference.csv" "${EXPECTED_PRIVATE_SHA[${SEED}]}"
check_hash "${PREPARED}/sentinel_private_reference.csv" "${EXPECTED_SENTINEL_SHA[${SEED}]}"
check_hash "${INITIAL_EVIDENCE}/entry_evidence.csv" "${EXPECTED_EVIDENCE_SHA[${SEED}]}"
[[ -e "${PREPARED}/.prepare_complete" ]] || { echo "Prepared marker missing" >&2; exit 2; }
[[ -e "${INITIAL_EVIDENCE}/.blind_run_complete" ]] || { echo "Initial evidence marker missing" >&2; exit 2; }
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite ${OUTPUT}" >&2; exit 2; }

source /vol/cuda/12.5.0/setup.sh
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
mkdir -p "${OUTPUT_ROOT}" "${MPLCONFIGDIR}"

COMMON=(
  --output-dir "${OUTPUT}"
  --source-prepared "${PREPARED}"
  --private-reference "${PREPARED}/private_reference.csv"
  --sentinel-private-reference "${PREPARED}/sentinel_private_reference.csv"
  --initial-evidence-dir "${INITIAL_EVIDENCE}"
  --seed "${SEED}"
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
    --seed "${SEED}" \
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

echo "Starting full-issue-pool extension seed=${SEED} task=${SLURM_ARRAY_TASK_ID}"
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

"${PYTHON}" - "${OUTPUT}" "${SEED}" <<'PY'
from pathlib import Path
import json
import pandas as pd
import sys

run, seed = Path(sys.argv[1]), int(sys.argv[2])
history = pd.read_csv(run / "state/review_history_private.csv")
manifest = json.loads((run / "state/initialization_manifest.json").read_text())
loop1 = pd.read_csv(run / "loop_01/selected_entries_private.csv")
summary = json.loads((run / "seed_evaluation_summary.json").read_text())
if summary["seed"] != seed:
    raise RuntimeError("Seed provenance failed")
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
    action = pd.read_csv(evidence / "entry_evidence.csv")
    sentinel = pd.read_csv(evidence / "sentinel_entry_evidence.csv")
    if len(action) != 14400 or action[["image_id", "label_name"]].duplicated().any():
        raise RuntimeError(f"Loop {loop_id} action evidence failed")
    if len(sentinel) != 3600 or sentinel[["image_id", "label_name"]].duplicated().any():
        raise RuntimeError(f"Loop {loop_id} sentinel evidence failed")
    if set(action["image_id"].astype(str)) & set(sentinel["image_id"].astype(str)):
        raise RuntimeError("Action and sentinel evidence overlap")
if not (run / ".seed_evaluation_complete").is_file():
    raise RuntimeError("Seed evaluation marker missing")
print(f"Full issue-pool extension postflight passed for seed {seed}")
PY
printf 'job_id=%s\narray_task_id=%s\nseed=%s\n' \
  "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}" "${SLURM_ARRAY_TASK_ID}" "${SEED}" \
  > "${OUTPUT}/.worker_complete"
echo "Full-issue-pool extension seed=${SEED} completed"
