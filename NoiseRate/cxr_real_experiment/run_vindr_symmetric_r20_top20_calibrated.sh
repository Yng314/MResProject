#!/bin/bash
#SBATCH --job-name=v-r20-cal-top20
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_r20_cal_top20_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_r20_cal_top20_%j.err
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=44G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

PROJECT="/vol/gpudata/yz3522-llmtest/MResProject"
ROOT="${PROJECT}/NoiseRate"
SCRIPT_DIR="${ROOT}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
PREPARE="${SCRIPT_DIR}/vindr_top20_calibrated_prepare.py"
PREPARE_TEST="${SCRIPT_DIR}/test_vindr_top20_calibrated_prepare.py"
DRIVER="${SCRIPT_DIR}/vindr_top20_calibrated_refinement.py"
DRIVER_TEST="${SCRIPT_DIR}/test_vindr_top20_calibrated_refinement.py"
SCORE_ENGINE="${SCRIPT_DIR}/vindr_dqs_calibration_transfer.py"
OOF_ENGINE="${SCRIPT_DIR}/vindr_mobilenet_sentinel_oof.py"
PROTOCOL="${SCRIPT_DIR}/vindr_symmetric_r20_top20_calibrated_protocol_20260827.md"
MOBILENET_HELPER="${SCRIPT_DIR}/vindr_mobilenet_oof.py"
KNOWN_GT_HELPER="${SCRIPT_DIR}/vindr_known_gt_cl_benchmark.py"
LOSS_HELPER="${SCRIPT_DIR}/cxr_real_noise_validation_smoke.py"
DATASET_HELPER="${SCRIPT_DIR}/cxr_real_full_train_eval_cleanlab_xrv12.py"
SYMMETRIC_HELPER="${SCRIPT_DIR}/vindr_global_symmetric_noise.py"
DIRECTION_HELPER="${SCRIPT_DIR}/vindr_noise_direction_sensitivity.py"
STRESS_HELPER="${SCRIPT_DIR}/vindr_self_optimization_stress.py"

SOURCE_MAIN="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_global_symmetric_noise_mobilenet/20260805_v1"
SOURCE_REPLICATION="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_global_symmetric_noise_mobilenet/20260806_replication_seeds509_701_v1"
IMAGE_ROOT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/derived/png224_20260804_v1/images"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_symmetric_r20_top20_calibrated/20260827_v1"
PREPARED_ROOT="${OUTPUT_ROOT}/prepared"
SCORE_ROOT="${OUTPUT_ROOT}/loop0_scores"
INITIAL_EVIDENCE_ROOT="${OUTPUT_ROOT}/loop0_evidence"
CALIBRATION_ROOT="${OUTPUT_ROOT}/calibration"
REFINEMENT_ROOT="${OUTPUT_ROOT}/refinement"
AGGREGATE_ROOT="${OUTPUT_ROOT}/aggregate"
SEEDS=(13 42 97 123 211 307 509 701)
SEEDS_CSV="13,42,97,123,211,307,509,701"

# Filled and checked by the submission preflight after the implementation is frozen.
EXPECTED_PREPARE_SHA="b2e38abd35dab2198b1ad0d1b6c01825805d8d146caff4da3f3e85468e2036a4"
EXPECTED_PREPARE_TEST_SHA="bc7c27c392d7390d138bc0e1b38476c3ce188cb3d7d891fee74e4b2c03f32e46"
EXPECTED_DRIVER_SHA="ae04bae9525820ea4c68a89993d9dd3e30d14537799c29d77014f955e6a68621"
EXPECTED_DRIVER_TEST_SHA="02f2ff7c12af447d4d8ef34a5dcdc98bda43c123fa790c5ee85a47a1fb1dec84"
EXPECTED_SCORE_ENGINE_SHA="279f9e1f99a9e773755575006abf7fb30a3812f62672e5fbcc1c58aadfd2247e"
EXPECTED_OOF_ENGINE_SHA="706c38a8af84a740a06261b459c3b733e6a88eff21bd9017ae62dec04b0a3f04"
EXPECTED_PROTOCOL_SHA="71b6b06b85e1793856e62b6149ea8301f7d013c871ce4bfe36f7afd787788f0a"
EXPECTED_MOBILENET_HELPER_SHA="9d1585d8d37b7c8043516889b9b3d6ddda984425f910a891d889b0c2ecd9b026"
EXPECTED_KNOWN_GT_HELPER_SHA="f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83"
EXPECTED_LOSS_HELPER_SHA="62da5b019f9809b4641b58f3c9bda56c1f987057fc765f9a76f32feade7046e0"
EXPECTED_DATASET_HELPER_SHA="f9b1eb32aaa7d6df9e1d1ebd3ba66fd2ef637bde47676fda4c7b629a7c7f452d"
EXPECTED_SYMMETRIC_HELPER_SHA="80316491dd61e7f9aa13d52a44bf76448968889bf84a2a33e7aa2a1b70431404"
EXPECTED_DIRECTION_HELPER_SHA="eabe258dd8ce0b860d458c451cd4064514bc1b7a8fa76679347b6c532602390e"
EXPECTED_STRESS_HELPER_SHA="96f47a284e1092c610a9327c4f646af4c067be87cfde70b01ca60f98bccda331"

check_hash() {
  local path="$1"
  local expected="$2"
  [[ "$(sha256sum "${path}" | cut -d' ' -f1)" == "${expected}" ]] || {
    echo "Hash mismatch: ${path}" >&2
    exit 2
  }
}

for item in \
  "${PREPARE}:${EXPECTED_PREPARE_SHA}" \
  "${PREPARE_TEST}:${EXPECTED_PREPARE_TEST_SHA}" \
  "${DRIVER}:${EXPECTED_DRIVER_SHA}" \
  "${DRIVER_TEST}:${EXPECTED_DRIVER_TEST_SHA}" \
  "${SCORE_ENGINE}:${EXPECTED_SCORE_ENGINE_SHA}" \
  "${OOF_ENGINE}:${EXPECTED_OOF_ENGINE_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${MOBILENET_HELPER}:${EXPECTED_MOBILENET_HELPER_SHA}" \
  "${KNOWN_GT_HELPER}:${EXPECTED_KNOWN_GT_HELPER_SHA}" \
  "${LOSS_HELPER}:${EXPECTED_LOSS_HELPER_SHA}" \
  "${DATASET_HELPER}:${EXPECTED_DATASET_HELPER_SHA}" \
  "${SYMMETRIC_HELPER}:${EXPECTED_SYMMETRIC_HELPER_SHA}" \
  "${DIRECTION_HELPER}:${EXPECTED_DIRECTION_HELPER_SHA}" \
  "${STRESS_HELPER}:${EXPECTED_STRESS_HELPER_SHA}"; do
  check_hash "${item%%:*}" "${item##*:}"
done

source /vol/cuda/12.5.0/setup.sh
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
mkdir -p "${PROJECT}/slurm_logs" "${MPLCONFIGDIR}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
[[ "${GPU_NAME}" == *"A16"* ]] || {
  echo "Expected NVIDIA A16; found ${GPU_NAME}" >&2
  exit 2
}
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

"${PYTHON}" -m py_compile "${PREPARE}" "${PREPARE_TEST}" "${DRIVER}" "${DRIVER_TEST}"
"${PYTHON}" "${PREPARE_TEST}"
"${PYTHON}" "${DRIVER_TEST}"

if [[ -e "${PREPARED_ROOT}/.prepare_complete" ]]; then
  echo "Preparation already complete; reusing ${PREPARED_ROOT}"
else
  [[ ! -e "${PREPARED_ROOT}" ]] || {
    echo "Incomplete preparation directory requires audit: ${PREPARED_ROOT}" >&2
    exit 2
  }
  "${PYTHON}" -u "${PREPARE}" prepare \
    --source-root "${SOURCE_MAIN}" \
    --source-root "${SOURCE_REPLICATION}" \
    --output-root "${PREPARED_ROOT}" \
    --seeds "${SEEDS_CSV}"
fi

IMAGE_INDEX="${PREPARED_ROOT}/image_index.csv"

run_loop0_score() {
  local seed="$1"
  local prepared="${PREPARED_ROOT}/seed_${seed}/prepared"
  local final="${SCORE_ROOT}/seed_${seed}"
  if [[ -e "${final}/.score_complete" ]]; then
    echo "Loop-0 score already complete for seed ${seed}; skipping"
    return
  fi
  [[ ! -e "${final}" ]] || {
    echo "Incomplete Loop-0 score requires audit: ${final}" >&2
    exit 2
  }
  mkdir -p "${SCORE_ROOT}"
  local partial="${final}.partial_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}"
  [[ ! -e "${partial}" ]] || {
    echo "Current-attempt Loop-0 score directory exists: ${partial}" >&2
    exit 2
  }
  "${PYTHON}" -u "${SCORE_ENGINE}" score \
    --blind-cohort "${prepared}/blind_noisy_cohort.csv" \
    --sentinel-cohort "${prepared}/sentinel_blind_cohort.csv" \
    --split-reference-cohort "${prepared}/blind_noisy_cohort.csv" \
    --image-index "${IMAGE_INDEX}" \
    --image-root "${IMAGE_ROOT}" \
    --output-dir "${partial}" \
    --seed "${seed}" \
    --n-splits 4 \
    --epochs 100 \
    --early-stopping-patience 10 \
    --learning-rate 0.001 \
    --batch-size 32 \
    --num-workers 4 \
    --device cuda
  mv "${partial}" "${final}"
}

materialize_loop0() {
  local seed="$1"
  local prepared="${PREPARED_ROOT}/seed_${seed}/prepared"
  local final="${INITIAL_EVIDENCE_ROOT}/seed_${seed}"
  if [[ -e "${final}/.blind_run_complete" ]]; then
    echo "Loop-0 entry evidence already complete for seed ${seed}; skipping"
    return
  fi
  [[ ! -e "${final}" ]] || {
    echo "Incomplete Loop-0 evidence requires audit: ${final}" >&2
    exit 2
  }
  mkdir -p "${INITIAL_EVIDENCE_ROOT}"
  local partial="${final}.partial_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}"
  "${PYTHON}" -u "${PREPARE}" materialize-evidence \
    --blind-cohort "${prepared}/blind_noisy_cohort.csv" \
    --sentinel-cohort "${prepared}/sentinel_blind_cohort.csv" \
    --score-dir "${SCORE_ROOT}/seed_${seed}" \
    --output-dir "${partial}" \
    --seed "${seed}"
  mv "${partial}" "${final}"
}

# Complete all outcome-blind Loop-0 scoring before fitting any threshold.
for seed in "${SEEDS[@]}"; do
  run_loop0_score "${seed}"
  materialize_loop0 "${seed}"
done

if [[ -e "${CALIBRATION_ROOT}/.calibration_frozen" ]]; then
  echo "Calibration thresholds already frozen; reusing them"
else
  [[ ! -e "${CALIBRATION_ROOT}" ]] || {
    echo "Incomplete calibration directory requires audit: ${CALIBRATION_ROOT}" >&2
    exit 2
  }
  local_calibration="${CALIBRATION_ROOT}.partial_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}"
  "${PYTHON}" -u "${PREPARE}" calibrate \
    --score-root "${SCORE_ROOT}" \
    --prepared-root "${PREPARED_ROOT}" \
    --output-dir "${local_calibration}" \
    --seeds "${SEEDS_CSV}"
  mv "${local_calibration}" "${CALIBRATION_ROOT}"
fi

THRESHOLDS="${CALIBRATION_ROOT}/calibration_thresholds_private.csv"
THRESHOLD_SHA="$(sha256sum "${THRESHOLDS}" | cut -d' ' -f1)"
[[ "$(tr -d '\n' < "${CALIBRATION_ROOT}/.calibration_frozen")" == "${THRESHOLD_SHA}" ]] || {
  echo "Frozen threshold marker disagrees with the threshold table" >&2
  exit 2
}

run_post_action_oof() {
  local seed="$1"
  local cohort="$2"
  local prepared="$3"
  local final="$4"
  if [[ -e "${final}/.blind_run_complete" ]]; then
    echo "Post-action OOF already complete: ${final}"
    return
  fi
  [[ ! -e "${final}" ]] || {
    echo "Incomplete final OOF directory requires audit: ${final}" >&2
    exit 2
  }
  local partial="${final}.partial_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}"
  [[ ! -e "${partial}" ]] || {
    echo "Current-attempt OOF directory exists: ${partial}" >&2
    exit 2
  }
  "${PYTHON}" -u "${OOF_ENGINE}" \
    --blind-cohort "${cohort}" \
    --sentinel-cohort "${prepared}/sentinel_blind_cohort.csv" \
    --split-reference-cohort "${prepared}/blind_noisy_cohort.csv" \
    --image-index "${IMAGE_INDEX}" \
    --image-root "${IMAGE_ROOT}" \
    --output-dir "${partial}" \
    --seed "${seed}" \
    --n-splits 4 \
    --epochs 100 \
    --early-stopping-patience 10 \
    --learning-rate 0.001 \
    --batch-size 32 \
    --num-workers 4 \
    --device cuda
  mv "${partial}" "${final}"
}

# Action-set references are first used only after all thresholds are frozen.
for seed in "${SEEDS[@]}"; do
  prepared="${PREPARED_ROOT}/seed_${seed}/prepared"
  seed_output="${REFINEMENT_ROOT}/seed_${seed}"
  "${PYTHON}" -u "${DRIVER}" initialize \
    --output-dir "${seed_output}" \
    --initial-cohort "${prepared}/blind_noisy_cohort.csv" \
    --private-reference "${prepared}/private_reference.csv" \
    --initial-oof-dir "${INITIAL_EVIDENCE_ROOT}/seed_${seed}" \
    --thresholds "${THRESHOLDS}" \
    --seed "${seed}"

  for loop_id in $(seq 1 5); do
    loop_pad="$(printf '%02d' "${loop_id}")"
    loop_root="${seed_output}/loop_${loop_pad}"
    "${PYTHON}" -u "${DRIVER}" select \
      --output-dir "${seed_output}" --loop-id "${loop_id}" --seed "${seed}"
    "${PYTHON}" -u "${DRIVER}" oracle-update \
      --output-dir "${seed_output}" --loop-id "${loop_id}" --seed "${seed}" \
      --private-reference "${prepared}/private_reference.csv"
    run_post_action_oof \
      "${seed}" \
      "${loop_root}/cohort_after_action_blind.csv" \
      "${prepared}" \
      "${loop_root}/oof_after_action"
    "${PYTHON}" -u "${DRIVER}" finalize-loop \
      --output-dir "${seed_output}" --loop-id "${loop_id}" --seed "${seed}" \
      --post-oof-dir "${loop_root}/oof_after_action"
  done
  "${PYTHON}" -u "${DRIVER}" evaluate \
    --output-dir "${seed_output}" \
    --private-reference "${prepared}/private_reference.csv" \
    --seed "${seed}"
done

"${PYTHON}" -u "${DRIVER}" aggregate \
  --experiment-root "${REFINEMENT_ROOT}" \
  --aggregate-output "${AGGREGATE_ROOT}" \
  --seeds "${SEEDS_CSV}"

"${PYTHON}" - "${OUTPUT_ROOT}" "${THRESHOLD_SHA}" <<'PY'
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys

root, threshold_sha = Path(sys.argv[1]), sys.argv[2]
calibration = root / "calibration"
aggregate = root / "aggregate"
thresholds = pd.read_csv(calibration / "calibration_thresholds_private.csv")
if len(thresholds) != 32 or thresholds[["seed", "fold_id"]].duplicated().any():
    raise RuntimeError("Threshold postflight failed")
if not thresholds["calibration_true_quality"].eq(0.8).all():
    raise RuntimeError("Calibration anchor is not exactly 0.80")
manifest = json.loads((calibration / "calibration_manifest_private.json").read_text())
if manifest["thresholds_sha256"] != threshold_sha or manifest["action_private_reference_accessed"] is not False:
    raise RuntimeError("Calibration provenance postflight failed")
trajectory = pd.read_csv(aggregate / "all_seed_dqs_trajectories_private.csv")
if len(trajectory) != 48 or trajectory[["seed", "loop"]].duplicated().any():
    raise RuntimeError("Expected eight complete six-state trajectories")
if not trajectory.loc[trajectory["loop"].eq(0), "known_true_quality"].eq(0.8).all():
    raise RuntimeError("Action quality anchor is not exactly 0.80")
for _, frame in trajectory.groupby("seed"):
    if np.any(np.diff(frame.sort_values("loop")["known_true_quality"].to_numpy(float)) < -1e-12):
        raise RuntimeError("Known quality decreased under oracle correction")
if not (aggregate / ".aggregate_complete").is_file():
    raise RuntimeError("Aggregate marker is missing")
print("VinDr symmetric-r20 calibrated top-20% postflight passed")
PY

printf 'job_id=%s\nthresholds_sha256=%s\n' "${SLURM_JOB_ID:-manual}" "${THRESHOLD_SHA}" \
  > "${OUTPUT_ROOT}/.job_complete"
echo "VinDr symmetric-r20 calibrated top-20% experiment completed"
