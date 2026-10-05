#!/bin/bash
#SBATCH --job-name=vindr-selfopt
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_selfopt_%A_%a.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_selfopt_%A_%a.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --array=1-2%2
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
ENGINE="${SCRIPT_DIR}/vindr_densenet_sentinel_oof.py"
BASE_ENGINE="${SCRIPT_DIR}/vindr_densenet_oof.py"
DRIVER="${SCRIPT_DIR}/vindr_self_optimization_stress.py"
PROTOCOL="${SCRIPT_DIR}/vindr_self_optimization_stress_protocol_20260814.md"
PREP_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_self_optimization_stress_prepared/20260814_v1"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_self_optimization_stress/20260814_v1"
SMOKE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_self_optimization_stress_smoke/20260814_v1"
RUN_MANIFEST="${PREP_ROOT}/formal_run_manifest.csv"
IMAGE_INDEX="${PREP_ROOT}/image_index.csv"
IMAGE_ROOT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/derived/png224_20260804_v1/images"
XRV_CACHE="/vol/gpudata/yz3522-llmtest/.cache/torchxrayvision"
XRV_WEIGHT="${XRV_CACHE}/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"

EXPECTED_ENGINE_SHA="faf2e31edaef6150c0600a1c07d0633e0292ef995dd3bca74179518a596470d4"
EXPECTED_BASE_ENGINE_SHA="fd7a8a3141680d27fe20955e17e7f77437ee0d9a6f0d18a45b16be628c9627bc"
EXPECTED_DRIVER_SHA="96f47a284e1092c610a9327c4f646af4c067be87cfde70b01ca60f98bccda331"
EXPECTED_PROTOCOL_SHA="a943d01cf8cd76c6eaad10280a2b74b5ca7aa40b63d02d92545d4469435b9bd1"
EXPECTED_IMAGE_INDEX_SHA="645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2"
EXPECTED_RUN_MANIFEST_SHA="368f534ba7f1290e642b18fbccc573675212b6f6213dabd48996cf1e508fc858"
EXPECTED_WEIGHT_SHA="56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899"

WORKER_ID="${SELFOPT_WORKER_ID:-${SLURM_ARRAY_TASK_ID:-}}"
[[ "${WORKER_ID}" =~ ^[0-2]$ ]] || { echo "Worker id 0-2 is required" >&2; exit 2; }
[[ -e "${SMOKE_ROOT}/.smoke_complete" ]] || { echo "Smoke marker is missing" >&2; exit 2; }
[[ -e "${PREP_ROOT}/.prepare_complete" ]] || { echo "Preparation marker is missing" >&2; exit 2; }

for item in \
  "${ENGINE}:${EXPECTED_ENGINE_SHA}" \
  "${BASE_ENGINE}:${EXPECTED_BASE_ENGINE_SHA}" \
  "${DRIVER}:${EXPECTED_DRIVER_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${IMAGE_INDEX}:${EXPECTED_IMAGE_INDEX_SHA}" \
  "${RUN_MANIFEST}:${EXPECTED_RUN_MANIFEST_SHA}" \
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
  local row
  row="$(${PYTHON} - "${RUN_MANIFEST}" "${run_index}" <<'PY'
import pandas as pd
import sys

frame = pd.read_csv(sys.argv[1])
row = frame.loc[frame["run_index"].eq(int(sys.argv[2]))]
if len(row) != 1:
    raise RuntimeError("Formal manifest index is missing or duplicated")
row = row.iloc[0]
print("\t".join(map(str, [
    row.initialization, row.scenario_id, row.structure,
    row.noise_rate, int(row.seed), row.prepared_path,
])))
PY
)"
  local initialization scenario structure noise_rate seed prepared
  IFS=$'\t' read -r initialization scenario structure noise_rate seed prepared <<< "${row}"
  local output="${OUTPUT_ROOT}/${initialization}/${scenario}/seed_${seed}"
  if [[ -e "${output}/.formal_run_complete" ]]; then
    echo "Run ${run_index} already complete; skipping"
    return
  fi
  [[ ! -e "${output}" ]] || { echo "Incomplete pre-existing run requires audit: ${output}" >&2; exit 2; }
  [[ -e "${prepared}/.prepare_complete" ]] || { echo "Prepared marker missing: ${prepared}" >&2; exit 2; }
  mkdir -p "$(dirname "${output}")"
  local learning_rate="0.001"
  [[ "${initialization}" == "scratch" ]] || learning_rate="0.0001"
  local common=(
    --output-dir "${output}"
    --source-prepared "${prepared}"
    --private-reference "${prepared}/private_reference.csv"
    --sentinel-private-reference "${prepared}/sentinel_private_reference.csv"
    --scenario-id "${scenario}"
    --structure "${structure}"
    --noise-rate "${noise_rate}"
    --seed "${seed}"
    --initialization "${initialization}"
    --loops 5
    --review-budget 288
    --sentinel-review-budget 72
  )
  run_oof() {
    local cohort="$1"
    local target="$2"
    "${PYTHON}" -u "${ENGINE}" \
      --blind-cohort "${cohort}" \
      --sentinel-cohort "${prepared}/sentinel_blind_cohort.csv" \
      --split-reference-cohort "${prepared}/blind_noisy_cohort.csv" \
      --image-index "${IMAGE_INDEX}" \
      --image-root "${IMAGE_ROOT}" \
      --output-dir "${target}" \
      --seed "${seed}" \
      --initialization "${initialization}" \
      --xrv-cache-dir "${XRV_CACHE}" \
      --n-splits 4 \
      --epochs 50 \
      --early-stopping-patience 8 \
      --learning-rate "${learning_rate}" \
      --batch-size 32 \
      --num-workers 4 \
      --device cuda
  }

  echo "Formal run ${run_index}: ${initialization} ${scenario} seed=${seed}"
  "${PYTHON}" -u "${DRIVER}" initialize "${common[@]}"
  run_oof "${prepared}/blind_noisy_cohort.csv" "${output}/loop_00/oof_evidence"
  for loop_id in $(seq 1 5); do
    "${PYTHON}" -u "${DRIVER}" select "${common[@]}" --loop-id "${loop_id}"
    "${PYTHON}" -u "${DRIVER}" oracle-update "${common[@]}" --loop-id "${loop_id}"
    run_oof \
      "${output}/loop_$(printf '%02d' "${loop_id}")/cohort_after_action_blind.csv" \
      "${output}/loop_$(printf '%02d' "${loop_id}")/oof_after_action"
    "${PYTHON}" -u "${DRIVER}" finalize-loop "${common[@]}" --loop-id "${loop_id}"
  done
  "${PYTHON}" -u "${DRIVER}" evaluate "${common[@]}"
  "${PYTHON}" - "${output}" "${initialization}" "${scenario}" "${seed}" <<'PY'
import json
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
initialization, scenario, seed = sys.argv[2], sys.argv[3], int(sys.argv[4])
history = pd.read_csv(root / "state/review_history_private.csv")
summary = json.loads((root / "seed_evaluation_summary.json").read_text())
if len(history) != 1440 or history["entry_key"].duplicated().any():
    raise RuntimeError("Formal no-repeat review history failed")
if sorted(history["loop"].unique()) != [1, 2, 3, 4, 5]:
    raise RuntimeError("Formal loop coverage failed")
if summary["initialization"] != initialization or summary["scenario_id"] != scenario or summary["seed"] != seed:
    raise RuntimeError("Formal summary provenance failed")
for loop_id in range(6):
    evidence = root / f"loop_{loop_id:02d}" / ("oof_evidence" if loop_id == 0 else "oof_after_action")
    action = pd.read_csv(evidence / "entry_evidence.csv")
    sentinel = pd.read_csv(evidence / "sentinel_entry_evidence.csv")
    if len(action) != 14400 or action[["image_id", "label_name"]].duplicated().any():
        raise RuntimeError(f"Action evidence failed at Loop {loop_id}")
    if len(sentinel) != 3600 or sentinel[["image_id", "label_name"]].duplicated().any():
        raise RuntimeError(f"Sentinel evidence failed at Loop {loop_id}")
    if set(action["image_id"].astype(str)) & set(sentinel["image_id"].astype(str)):
        raise RuntimeError("Action and sentinel evidence overlap")
print("Formal postflight passed")
PY
  printf 'job_id=%s\nworker=%s\nrun_index=%s\n' \
    "${SLURM_JOB_ID:-manual}" "${WORKER_ID}" "${run_index}" > "${output}/.formal_run_complete"
}

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
TOTAL_RUNS="$(${PYTHON} -c "import pandas as pd; print(len(pd.read_csv('${RUN_MANIFEST}')))" )"
for ((run_index=WORKER_ID; run_index<TOTAL_RUNS; run_index+=3)); do
  run_one "${run_index}"
done
printf 'job_id=%s\nworker=%s\n' "${SLURM_JOB_ID:-manual}" "${WORKER_ID}" \
  > "${OUTPUT_ROOT}/.formal_worker_${WORKER_ID}_complete"

all_complete=true
for worker in 0 1 2; do
  [[ -e "${OUTPUT_ROOT}/.formal_worker_${worker}_complete" ]] || all_complete=false
done
if [[ "${all_complete}" == true ]] && mkdir "${OUTPUT_ROOT}/.aggregate_lock" 2>/dev/null; then
  "${PYTHON}" -u "${DRIVER}" aggregate \
    --experiment-root "${OUTPUT_ROOT}" \
    --aggregate-output "${OUTPUT_ROOT}/aggregate" \
    --run-manifest "${RUN_MANIFEST}"
  [[ -e "${OUTPUT_ROOT}/aggregate/.aggregate_complete" ]] || { echo "Aggregate marker missing" >&2; exit 2; }
  printf 'job_id=%s\n' "${SLURM_JOB_ID:-manual}" > "${OUTPUT_ROOT}/.complete"
fi
echo "Worker ${WORKER_ID} complete"
