#!/bin/bash
#SBATCH --job-name=vfull-mnet
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_fullpool_mobilenet_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_fullpool_mobilenet_%j.err
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 SEED" >&2
  exit 2
fi

SEED="$1"
case "${SEED}" in
  11003|13007|17011|19001|23003|27011|31013|37003) ;;
  *) echo "Seed is outside the locked eight-seed design: ${SEED}" >&2; exit 2 ;;
esac

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
SCRIPT_DIR="${ROOT}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
DRIVER="${SCRIPT_DIR}/vindr_full_issue_pool_iteration.py"
ENGINE="${SCRIPT_DIR}/vindr_mobilenet_sentinel_oof.py"
MOBILENET_HELPER="${SCRIPT_DIR}/vindr_mobilenet_oof.py"
KNOWN_GT_HELPER="${SCRIPT_DIR}/vindr_known_gt_cl_benchmark.py"
LOSS_HELPER="${SCRIPT_DIR}/cxr_real_noise_validation_smoke.py"
DATASET_HELPER="${SCRIPT_DIR}/cxr_real_full_train_eval_cleanlab_xrv12.py"
RANK_HELPER="${SCRIPT_DIR}/vindr_iterative_evidence_improvement.py"
STRESS_HELPER="${SCRIPT_DIR}/vindr_self_optimization_stress.py"
ENTRY_HELPER="${SCRIPT_DIR}/vindr_iterative_oracle_cleaning.py"
FOLD_HELPER="${SCRIPT_DIR}/vindr_noise_direction_sensitivity.py"
PROTOCOL="${SCRIPT_DIR}/vindr_full_issue_pool_mobilenet_protocol_20260824.md"
PREP_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_self_optimization_stress_prepared/20260814_v1/scenarios/hard_r30"
PREPARED="${PREP_ROOT}/seed_${SEED}/prepared"
IMAGE_INDEX="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_self_optimization_stress_prepared/20260814_v1/image_index.csv"
IMAGE_ROOT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/derived/png224_20260804_v1/images"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_full_issue_pool_iteration_mobilenet/20260824_multiseed_v1"
INITIAL_EVIDENCE="${OUTPUT_ROOT}/loop0_evidence/seed_${SEED}"
OUTPUT="${OUTPUT_ROOT}/seed_${SEED}"

EXPECTED_DRIVER_SHA="0d777cd3567a5cb20a82b5c4c4ab52a8be6ac6bba12f8a91cfddb8e0a78f3c88"
EXPECTED_ENGINE_SHA="706c38a8af84a740a06261b459c3b733e6a88eff21bd9017ae62dec04b0a3f04"
EXPECTED_MOBILENET_HELPER_SHA="9d1585d8d37b7c8043516889b9b3d6ddda984425f910a891d889b0c2ecd9b026"
EXPECTED_KNOWN_GT_HELPER_SHA="f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83"
EXPECTED_LOSS_HELPER_SHA="62da5b019f9809b4641b58f3c9bda56c1f987057fc765f9a76f32feade7046e0"
EXPECTED_DATASET_HELPER_SHA="f9b1eb32aaa7d6df9e1d1ebd3ba66fd2ef637bde47676fda4c7b629a7c7f452d"
EXPECTED_RANK_HELPER_SHA="ad42522d355beaebaa1a19f441b7c281926c4936bb0a8ec52b9edef48489e6a6"
EXPECTED_STRESS_HELPER_SHA="96f47a284e1092c610a9327c4f646af4c067be87cfde70b01ca60f98bccda331"
EXPECTED_ENTRY_HELPER_SHA="724e78b2e806c90147f2aac153f7c1561d9fd321aec5573cdc9e789026f22d0c"
EXPECTED_FOLD_HELPER_SHA="eabe258dd8ce0b860d458c451cd4064514bc1b7a8fa76679347b6c532602390e"
EXPECTED_PROTOCOL_SHA="a00e5bee0d9ac2403e193b6b2958b648014b363e9f580d2df27bd6178bcb602d"
EXPECTED_IMAGE_INDEX_SHA="645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2"

declare -A EXPECTED_ACTION_SHA=(
  [11003]="e9917c719fa58b7ec4e9b4bb547b46f1e77a76431b790e89c4133556bcb10e8b"
  [13007]="10ac3250f1b7a307c2ab78af32f50513da09c0b2efa4f0b48c9accd4da74ca39"
  [17011]="d3b296d5baca316a3222967f91381887e62e646ecbf52f029202686a2ca1aa7c"
  [19001]="d488fec48889572e6c8bfdd1093f6736d7e2d47fce23cfe4982ec7cc1ef0f7ad"
  [23003]="7381956ccefb3f9979bf3aaf5245763a29ea4de6ffee1286cd181a7a9c30fdb5"
  [27011]="7422d504e53520da9e5def161d98137fec338154b879990387bbc66ead131403"
  [31013]="2c41b1a6ff9b13c9a05358a6d96c61ef221e4219c7b842f6f0627fbbd0a2a89a"
  [37003]="ea231c87b346d2ed0510c5d6bc496d4298723d1fc21f790f9b58ee37dedac23b"
)
declare -A EXPECTED_PRIVATE_SHA=(
  [11003]="80e909c5b814db36a1058e9ec5088a5c2bccf2e9f81ea92a276a34858a748373"
  [13007]="dcc53d484f95556b1c4f8ca0a791449f6914e20562d753e63aa371e67f03850b"
  [17011]="c40d550567e8894e0bf6d61148a09f4e8f56091b92b2d452d77b14e61de5ab6d"
  [19001]="9d53826fce82df2d1afd1131c9fa87559ba5bdd0e37a5df9b91b43b465bcfd37"
  [23003]="371888fc082afd49076ebea7184a951405bc5a3bab92804b47e4643ba9fabb95"
  [27011]="82ef79e5952980b2aae6c268084031cd92b1e6101a83417dd2d996c91e39caed"
  [31013]="1010fd1a60e40a6790a83f02970db1f21999b0ac3ecfd25487429993cff68b16"
  [37003]="d86af7ac670e26edf162114ec7170dcff48002b20de0c19188180bcac2ecfb1c"
)
declare -A EXPECTED_SENTINEL_BLIND_SHA=(
  [11003]="7d5ea06947d40015d46e7a1b2ea56ab71db9676dddbdc5f24caa500cb2a0cf62"
  [13007]="adcdf2ab19cc1407ac927fb5c78041f237ea41cf32da5f03f435c00ee6ebc12a"
  [17011]="15d45218ffa9996569d06aefa53b0f9b6d07115e7a2ecd24f44f4706ca291879"
  [19001]="34cbf4da5394ea32465ae892d98a0faa445bbbb7711bc4c40059a5bed9042e00"
  [23003]="7288e945feb001108534454c6ecccfabafb791317c89871d04be08f1d92e35e1"
  [27011]="6da14a5b655e7d997f8d314658280c55e7d9b053c6aab18eadb9d3210554d210"
  [31013]="1c77efc6997413020782d6d35def34a8d856944f22617d019e436c8ac6e49c45"
  [37003]="8d3ff748c21a08d9064fa3bd2def2bc9dfc7a66247656df369fa0c9608ad61d9"
)
declare -A EXPECTED_SENTINEL_PRIVATE_SHA=(
  [11003]="d6f7e18ddbb7bf2654f5589f685bc2eab541c093b0035ae93fe5d212df4b8919"
  [13007]="7c0293c8943a61186d0ad79db5660e5ce55570827e7c992b02ac6c9f193de77e"
  [17011]="01855d3a1cb1480aa006865639852562f196e686b264df7d579f7a78b8ee224f"
  [19001]="a24423849bae9c37534764a23f4aa7dad83b38d2e689fc948c467808fc99e122"
  [23003]="741c1af0f4e56f0c1fbffc87b28513d132790a214871daf7ea0e38d2dabaf859"
  [27011]="f54127d8abf56bebe3c5623100cb697ff1bb86adeca846203395ccefd5540b5c"
  [31013]="38052b24c23983ee39b7ebd4c2189e1ab38249eeeb5d8d218be46e7d90c46425"
  [37003]="a67f8f0d762b4cdb10f21c473af3cde3c5bb53c7972dcae7f58e408c86caf2a2"
)

check_hash() {
  local path="$1"
  local expected="$2"
  [[ "$(sha256sum "${path}" | cut -d' ' -f1)" == "${expected}" ]] || {
    echo "Hash mismatch: ${path}" >&2
    exit 2
  }
}

for item in \
  "${DRIVER}:${EXPECTED_DRIVER_SHA}" \
  "${ENGINE}:${EXPECTED_ENGINE_SHA}" \
  "${MOBILENET_HELPER}:${EXPECTED_MOBILENET_HELPER_SHA}" \
  "${KNOWN_GT_HELPER}:${EXPECTED_KNOWN_GT_HELPER_SHA}" \
  "${LOSS_HELPER}:${EXPECTED_LOSS_HELPER_SHA}" \
  "${DATASET_HELPER}:${EXPECTED_DATASET_HELPER_SHA}" \
  "${RANK_HELPER}:${EXPECTED_RANK_HELPER_SHA}" \
  "${STRESS_HELPER}:${EXPECTED_STRESS_HELPER_SHA}" \
  "${ENTRY_HELPER}:${EXPECTED_ENTRY_HELPER_SHA}" \
  "${FOLD_HELPER}:${EXPECTED_FOLD_HELPER_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${IMAGE_INDEX}:${EXPECTED_IMAGE_INDEX_SHA}"; do
  check_hash "${item%%:*}" "${item##*:}"
done
check_hash "${PREPARED}/blind_noisy_cohort.csv" "${EXPECTED_ACTION_SHA[${SEED}]}"
check_hash "${PREPARED}/private_reference.csv" "${EXPECTED_PRIVATE_SHA[${SEED}]}"
check_hash "${PREPARED}/sentinel_blind_cohort.csv" "${EXPECTED_SENTINEL_BLIND_SHA[${SEED}]}"
check_hash "${PREPARED}/sentinel_private_reference.csv" "${EXPECTED_SENTINEL_PRIVATE_SHA[${SEED}]}"
[[ -e "${PREPARED}/.prepare_complete" ]] || { echo "Prepared marker missing" >&2; exit 2; }
if [[ -e "${OUTPUT}/.worker_complete" ]]; then
  echo "VinDr MobileNet full-issue-pool seed=${SEED} is already complete; skipping."
  exit 0
fi
if [[ -e "${INITIAL_EVIDENCE}" || -e "${OUTPUT}" ]]; then
  [[ -e "${INITIAL_EVIDENCE}/.blind_run_complete" && -e "${OUTPUT}/state/.initialized" ]] || {
    echo "Existing seed state is not safely resumable: ${SEED}" >&2
    exit 2
  }
else
  FRESH_SEED=1
fi

"${PYTHON}" - "${PREPARED}" "${IMAGE_INDEX}" <<'PY'
from pathlib import Path
import pandas as pd
import sys

prepared, image_index_path = Path(sys.argv[1]), Path(sys.argv[2])
labels = [
    "Atelectasis", "Cardiomegaly", "Consolidation",
    "Lung Opacity", "Pleural effusion", "Pneumonia",
]
action = pd.read_csv(prepared / "blind_noisy_cohort.csv")
sentinel = pd.read_csv(prepared / "sentinel_blind_cohort.csv")
image_index = pd.read_csv(image_index_path)
if len(action) != 2400 or action["image_id"].nunique() != 2400:
    raise RuntimeError("Action cohort structure failed")
if len(sentinel) != 600 or sentinel["image_id"].nunique() != 600:
    raise RuntimeError("Sentinel cohort structure failed")
if sorted(action["fold_id"].unique()) != [0, 1, 2, 3]:
    raise RuntimeError("Action folds are not locked to 0..3")
if not sentinel["fold_id"].eq(-1).all():
    raise RuntimeError("Sentinel folds are not locked to -1")
if not action[labels].isin([0, 1]).all().all() or not sentinel[labels].isin([0, 1]).all().all():
    raise RuntimeError("Action or sentinel labels are not binary")
if len(image_index) != 3000 or image_index["image_id"].nunique() != 3000:
    raise RuntimeError("Image index is not one-to-one over 3,000 images")
combined = pd.concat(
    [action[["image_id", "image_path"]], sentinel[["image_id", "image_path"]]],
    ignore_index=True,
)
if combined["image_id"].duplicated().any():
    raise RuntimeError("Action and sentinel images overlap")
aligned = combined.merge(
    image_index[["image_id", "image_path"]], on="image_id",
    how="outer", suffixes=("_cohort", "_index"), indicator=True,
)
if len(aligned) != 3000 or not aligned["_merge"].eq("both").all():
    raise RuntimeError("Cohorts do not exactly cover the locked image index")
if not aligned["image_path_cohort"].eq(aligned["image_path_index"]).all():
    raise RuntimeError("Cohort paths do not match the locked image index")
print("Locked VinDr input postflight passed")
PY

source /vol/cuda/12.5.0/setup.sh
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
mkdir -p "${OUTPUT_ROOT}/loop0_evidence" "${MPLCONFIGDIR}"

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
    --n-splits 4 \
    --epochs 50 \
    --early-stopping-patience 8 \
    --learning-rate 0.001 \
    --batch-size 32 \
    --num-workers 4 \
    --device cuda
}

COMMON=(
  --output-dir "${OUTPUT}"
  --source-prepared "${PREPARED}"
  --private-reference "${PREPARED}/private_reference.csv"
  --sentinel-private-reference "${PREPARED}/sentinel_private_reference.csv"
  --initial-evidence-dir "${INITIAL_EVIDENCE}"
  --seed "${SEED}"
  --loops 5
)

echo "Starting VinDr MobileNet full-issue-pool iteration seed=${SEED}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
if [[ "${FRESH_SEED:-0}" -eq 1 ]]; then
  run_oof "${PREPARED}/blind_noisy_cohort.csv" "${INITIAL_EVIDENCE}"
  "${PYTHON}" -u "${DRIVER}" initialize "${COMMON[@]}"
else
  echo "Resuming existing seed state for seed=${SEED}"
fi

FINAL_LOOP=0
for loop_id in $(seq 1 5); do
  LOOP_DIR="${OUTPUT}/loop_$(printf '%02d' "${loop_id}")"
  if [[ -e "${LOOP_DIR}/.loop_complete" ]]; then
    FINAL_LOOP="${loop_id}"
    echo "Loop ${loop_id} already complete; skipping."
    continue
  fi
  NEW_ISSUES="$("${PYTHON}" -u "${DRIVER}" inspect-next "${COMMON[@]}" --loop-id "${loop_id}" | tail -n 1)"
  if [[ "${NEW_ISSUES}" -eq 0 ]]; then
    FINAL_LOOP="$((loop_id - 1))"
    echo "Natural stop after Loop ${FINAL_LOOP}: no new unreviewed issue entries remain."
    break
  fi
  [[ ! -e "${LOOP_DIR}" ]] || {
    echo "Incomplete pre-existing loop directory cannot be resumed safely: ${LOOP_DIR}" >&2
    exit 2
  }
  "${PYTHON}" -u "${DRIVER}" select "${COMMON[@]}" --loop-id "${loop_id}"
  "${PYTHON}" -u "${DRIVER}" oracle-update "${COMMON[@]}" --loop-id "${loop_id}"
  run_oof \
    "${LOOP_DIR}/cohort_after_action_blind.csv" \
    "${LOOP_DIR}/oof_after_action"
  "${PYTHON}" -u "${DRIVER}" finalize-loop "${COMMON[@]}" --loop-id "${loop_id}"
  FINAL_LOOP="${loop_id}"
done
[[ "${FINAL_LOOP}" -ge 1 ]] || {
  echo "No full-pool cleaning loop completed for seed=${SEED}" >&2
  exit 2
}
EVALUATE_COMMON=("${COMMON[@]}")
EVALUATE_COMMON[-1]="${FINAL_LOOP}"
"${PYTHON}" -u "${DRIVER}" evaluate "${EVALUATE_COMMON[@]}"

"${PYTHON}" - "${OUTPUT}" "${INITIAL_EVIDENCE}" "${SEED}" "${FINAL_LOOP}" <<'PY'
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys

run, initial_evidence, seed, final_loop = (
    Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
)
history = pd.read_csv(run / "state/review_history_private.csv")
manifest = json.loads((run / "state/initialization_manifest.json").read_text())
loop1 = pd.read_csv(run / "loop_01/selected_entries_private.csv")
summary = json.loads((run / "seed_evaluation_summary.json").read_text())
if summary["seed"] != seed or manifest["seed"] != seed:
    raise RuntimeError("Seed provenance failed")
if manifest["protocol"] != "vindr_full_issue_pool_iteration_v1":
    raise RuntimeError("Workflow driver protocol provenance failed")
if len(loop1) != int(manifest["initial_issue_pool"]):
    raise RuntimeError("Loop 1 did not review the complete initial MobileNet issue pool")
if history["entry_key"].duplicated().any():
    raise RuntimeError("A reviewed entry re-entered a later issue pool")
if sorted(history["loop"].unique()) != list(range(1, final_loop + 1)):
    raise RuntimeError("Loop history is incomplete")
for loop_id, evidence in [(0, initial_evidence)] + [
    (index, run / f"loop_{index:02d}/oof_after_action")
    for index in range(1, final_loop + 1)
]:
    evidence_summary = json.loads((evidence / "blind_run_summary.json").read_text())
    if evidence_summary["protocol"] != "vindr_full_issue_pool_mobilenet_v1":
        raise RuntimeError(f"Loop {loop_id} used the wrong evidence protocol")
    if evidence_summary["seed"] != seed:
        raise RuntimeError(f"Loop {loop_id} seed provenance failed")
    if "MobileNetV3-small" not in evidence_summary["architecture"]:
        raise RuntimeError(f"Loop {loop_id} is not MobileNetV3-small")
    if evidence_summary["initialization"] != "scratch":
        raise RuntimeError(f"Loop {loop_id} did not use scratch initialization")
    if evidence_summary["sentinel_excluded_from_all_training_and_inner_validation"] is not True:
        raise RuntimeError(f"Loop {loop_id} sentinel isolation failed")
    best_epochs = evidence_summary["best_epochs"]
    if len(best_epochs) != 4 or not all(1 <= int(value) <= 50 for value in best_epochs):
        raise RuntimeError(f"Loop {loop_id} best-epoch provenance failed")
    action = pd.read_csv(evidence / "entry_evidence.csv")
    sentinel = pd.read_csv(evidence / "sentinel_entry_evidence.csv")
    if len(action) != 14400 or action[["image_id", "label_name"]].duplicated().any():
        raise RuntimeError(f"Loop {loop_id} action evidence failed")
    if len(sentinel) != 3600 or sentinel[["image_id", "label_name"]].duplicated().any():
        raise RuntimeError(f"Loop {loop_id} sentinel evidence failed")
    if set(action["image_id"].astype(str)) & set(sentinel["image_id"].astype(str)):
        raise RuntimeError("Action and sentinel evidence overlap")
    for frame, name in ((action, "action"), (sentinel, "sentinel")):
        probability = pd.to_numeric(frame["oof_probability"], errors="coerce")
        if probability.isna().any() or not probability.between(0.0, 1.0).all():
            raise RuntimeError(f"Loop {loop_id} {name} probabilities failed")
for loop_id in range(1, final_loop + 1):
    selection = pd.read_csv(run / f"loop_{loop_id:02d}/selected_entries_blind.csv")
    if not selection["cl_issue"].astype(bool).all():
        raise RuntimeError(f"Loop {loop_id} selection escaped the issue pool")
    selection_manifest = json.loads(
        (run / f"loop_{loop_id:02d}/selection_manifest_blind.json").read_text()
    )
    if len(selection) != int(selection_manifest["new_unreviewed_issue_entries"]):
        raise RuntimeError(f"Loop {loop_id} did not review the full new issue pool")
    oracle = json.loads((run / f"loop_{loop_id:02d}/oracle_summary_private.json").read_text())
    if oracle["quality_after"] + 1e-12 < oracle["quality_before"]:
        raise RuntimeError(f"Loop {loop_id} oracle quality decreased")
trajectory = pd.read_csv(run / "full_issue_iteration_trajectory_private.csv")
if len(trajectory) != final_loop + 1 or trajectory["loop"].tolist() != list(range(final_loop + 1)):
    raise RuntimeError("Trajectory does not match the completed loop range")
if trajectory[["seed", "loop"]].duplicated().any() or not trajectory["seed"].eq(seed).all():
    raise RuntimeError("Trajectory seed-loop provenance failed")
numeric = [
    "true_quality", "remaining_errors", "raw_dqs", "action_clean_macro_auroc",
    "sentinel_clean_macro_auroc", "sentinel_error_auprc", "sentinel_error_auroc",
]
numeric_values = trajectory[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
if not np.isfinite(numeric_values).all():
    raise RuntimeError("Trajectory contains non-finite required outcomes")
for loop_id in range(1, final_loop + 1):
    before = trajectory.loc[trajectory["loop"].eq(loop_id - 1)].iloc[0]
    after = trajectory.loc[trajectory["loop"].eq(loop_id)].iloc[0]
    corrected = int(after["new_true_errors"])
    if int(before["remaining_errors"]) - int(after["remaining_errors"]) != corrected:
        raise RuntimeError(f"Loop {loop_id} error-count accounting failed")
    if float(after["true_quality"]) + 1e-12 < float(before["true_quality"]):
        raise RuntimeError(f"Loop {loop_id} true quality decreased")
if not (run / ".seed_evaluation_complete").is_file():
    raise RuntimeError("Seed evaluation marker missing")
if int(summary["loops"]) != final_loop:
    raise RuntimeError("Summary does not record the natural stopping loop")
print(f"VinDr MobileNet full-issue-pool postflight passed for seed {seed}")
PY
printf 'job_id=%s\narray_job_id=%s\narray_task_id=%s\nseed=%s\n' \
  "${SLURM_JOB_ID:-manual}" "${SLURM_ARRAY_JOB_ID:-none}" "${SLURM_ARRAY_TASK_ID:-none}" "${SEED}" \
  > "${OUTPUT}/.worker_complete"
echo "VinDr MobileNet full-issue-pool iteration seed=${SEED} completed"
