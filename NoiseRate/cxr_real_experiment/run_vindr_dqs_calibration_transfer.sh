#!/bin/bash
#SBATCH --job-name=vindr-dqs-cal
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_dqs_calibration_transfer_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_dqs_calibration_transfer_%j.err
#SBATCH --partition=a16
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
PROGRAM="${SCRIPT_DIR}/vindr_dqs_calibration_transfer.py"
TEST="${SCRIPT_DIR}/test_vindr_dqs_calibration_transfer.py"
PROTOCOL="${SCRIPT_DIR}/vindr_dqs_calibration_transfer_protocol_20260827.md"
ENGINE="${SCRIPT_DIR}/vindr_mobilenet_sentinel_oof.py"
MOBILENET_HELPER="${SCRIPT_DIR}/vindr_mobilenet_oof.py"
KNOWN_GT_HELPER="${SCRIPT_DIR}/vindr_known_gt_cl_benchmark.py"
LOSS_HELPER="${SCRIPT_DIR}/cxr_real_noise_validation_smoke.py"
DATASET_HELPER="${SCRIPT_DIR}/cxr_real_full_train_eval_cleanlab_xrv12.py"
STRESS_HELPER="${SCRIPT_DIR}/vindr_self_optimization_stress.py"
RANK_HELPER="${SCRIPT_DIR}/vindr_iterative_evidence_improvement.py"
FOLD_HELPER="${SCRIPT_DIR}/vindr_noise_direction_sensitivity.py"
ENTRY_HELPER="${SCRIPT_DIR}/vindr_iterative_oracle_cleaning.py"

PREP_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_self_optimization_stress_prepared/20260814_v1/scenarios/hard_r30"
IMAGE_INDEX="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_self_optimization_stress_prepared/20260814_v1/image_index.csv"
IMAGE_ROOT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/derived/png224_20260804_v1/images"
HARDNESS="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_detector_benchmark_v2/20260812_v1/scenarios/clean/seed_887/blind_run/entry_evidence.csv"
TRAJECTORY_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_full_issue_pool_iteration_mobilenet/20260824_multiseed_v1"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_dqs_calibration_transfer/20260827_v1"
SCORE_ROOT="${OUTPUT_ROOT}/scores"
CALIBRATION_DIR="${OUTPUT_ROOT}/calibration"
EVALUATION_DIR="${OUTPUT_ROOT}/evaluation"
SEEDS=(11003 13007 17011 19001 23003 27011 31013 37003)
SEEDS_CSV="11003,13007,17011,19001,23003,27011,31013,37003"

EXPECTED_PROGRAM_SHA="279f9e1f99a9e773755575006abf7fb30a3812f62672e5fbcc1c58aadfd2247e"
EXPECTED_TEST_SHA="59fd3dbc52c22149a135e5af293e256cd49bd554c17506033947a111237a59fd"
EXPECTED_PROTOCOL_SHA="a40dc9e2e8970b241f7384084fbec8ae5e0f3874a9d36ce105e9506f4c1b01be"
EXPECTED_ENGINE_SHA="706c38a8af84a740a06261b459c3b733e6a88eff21bd9017ae62dec04b0a3f04"
EXPECTED_MOBILENET_HELPER_SHA="9d1585d8d37b7c8043516889b9b3d6ddda984425f910a891d889b0c2ecd9b026"
EXPECTED_KNOWN_GT_HELPER_SHA="f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83"
EXPECTED_LOSS_HELPER_SHA="62da5b019f9809b4641b58f3c9bda56c1f987057fc765f9a76f32feade7046e0"
EXPECTED_DATASET_HELPER_SHA="f9b1eb32aaa7d6df9e1d1ebd3ba66fd2ef637bde47676fda4c7b629a7c7f452d"
EXPECTED_STRESS_HELPER_SHA="96f47a284e1092c610a9327c4f646af4c067be87cfde70b01ca60f98bccda331"
EXPECTED_RANK_HELPER_SHA="ad42522d355beaebaa1a19f441b7c281926c4936bb0a8ec52b9edef48489e6a6"
EXPECTED_FOLD_HELPER_SHA="eabe258dd8ce0b860d458c451cd4064514bc1b7a8fa76679347b6c532602390e"
EXPECTED_ENTRY_HELPER_SHA="724e78b2e806c90147f2aac153f7c1561d9fd321aec5573cdc9e789026f22d0c"
EXPECTED_IMAGE_INDEX_SHA="645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2"
EXPECTED_HARDNESS_SHA="52c2bd7fbf8dd7666d9659ee9964a855bd2bdb50d0b4dbfe663d73d0ff8543d7"

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
declare -A EXPECTED_PREPARE_SUMMARY_SHA=(
  [11003]="fb05a63e28e65844d4add84d85c24139adda2c7f14e4d8e04f4c6dc17777fac8"
  [13007]="bc77ff62456b35f6d23cd3f60ad84bc3b9e4287bfc80a79d22f6f2337be2d8d3"
  [17011]="c79c442316efb45dd3ab5464dcf8aa57454d68923a1dee8642cbc967fab9e962"
  [19001]="3a832a36f40b7b05541b073bdfbcc23fcb1b43f2feca2b299abd8e1b44cc53f6"
  [23003]="d5474aaa27c2fc612ffa6aa412b4758dd20c6ed0c84917ee1a91752c23d8562d"
  [27011]="5cf11fedae9a11dfee1200cb5cdf9e31e9a30b0cab1302fee92f500246c6da97"
  [31013]="f394e6adf9cbec8d17bde7fbe3803e01c6ea2fd98f401a047d10a241a6ec3f9a"
  [37003]="5c93b53a349648ca56d39a74f60e893026fb17ed6d86794e6f1cc334458b17cd"
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
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${TEST}:${EXPECTED_TEST_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${ENGINE}:${EXPECTED_ENGINE_SHA}" \
  "${MOBILENET_HELPER}:${EXPECTED_MOBILENET_HELPER_SHA}" \
  "${KNOWN_GT_HELPER}:${EXPECTED_KNOWN_GT_HELPER_SHA}" \
  "${LOSS_HELPER}:${EXPECTED_LOSS_HELPER_SHA}" \
  "${DATASET_HELPER}:${EXPECTED_DATASET_HELPER_SHA}" \
  "${STRESS_HELPER}:${EXPECTED_STRESS_HELPER_SHA}" \
  "${RANK_HELPER}:${EXPECTED_RANK_HELPER_SHA}" \
  "${FOLD_HELPER}:${EXPECTED_FOLD_HELPER_SHA}" \
  "${ENTRY_HELPER}:${EXPECTED_ENTRY_HELPER_SHA}" \
  "${IMAGE_INDEX}:${EXPECTED_IMAGE_INDEX_SHA}" \
  "${HARDNESS}:${EXPECTED_HARDNESS_SHA}"; do
  check_hash "${item%%:*}" "${item##*:}"
done

for seed in "${SEEDS[@]}"; do
  prepared="${PREP_ROOT}/seed_${seed}/prepared"
  [[ -e "${prepared}/.prepare_complete" ]] || { echo "Prepare marker missing: ${seed}" >&2; exit 2; }
  check_hash "${prepared}/blind_noisy_cohort.csv" "${EXPECTED_ACTION_SHA[${seed}]}"
  check_hash "${prepared}/sentinel_blind_cohort.csv" "${EXPECTED_SENTINEL_BLIND_SHA[${seed}]}"
  check_hash "${prepared}/sentinel_private_reference.csv" "${EXPECTED_SENTINEL_PRIVATE_SHA[${seed}]}"
done

source /vol/cuda/12.5.0/setup.sh
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
mkdir -p "${SCORE_ROOT}" "${MPLCONFIGDIR}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
[[ "${GPU_NAME}" == *"A16"* ]] || { echo "Expected NVIDIA A16; found ${GPU_NAME}" >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

"${PYTHON}" -m py_compile "${PROGRAM}" "${TEST}"
"${PYTHON}" "${TEST}"

"${PYTHON}" - "${PREP_ROOT}" "${HARDNESS}" "${SEEDS_CSV}" <<'PY'
from pathlib import Path
import pandas as pd
import sys

from vindr_dqs_calibration_transfer import reconstruct_matched_sentinel

prep_root, hardness_path, seeds_text = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
hardness = pd.read_csv(
    hardness_path,
    usecols=["image_id", "label_name", "label_quality_self_confidence"],
)
for seed in [int(value) for value in seeds_text.split(",")]:
    prepared = prep_root / f"seed_{seed}" / "prepared"
    cohort = pd.read_csv(prepared / "sentinel_blind_cohort.csv")
    private = pd.read_csv(prepared / "sentinel_private_reference.csv")
    matched, summary = reconstruct_matched_sentinel(
        private, cohort["image_id"], hardness, seed
    )
    if summary["archived_r20_exact_reproduction"] is not True:
        raise RuntimeError(f"Seed {seed} failed archived r20 reconstruction")
    if len(matched) != 3600 or int(matched["injected_error"].sum()) != 1080:
        raise RuntimeError(f"Seed {seed} failed matched-r30 reconstruction")
print("All eight sentinel r20/r30 reconstruction preflights passed")
PY

for seed in "${SEEDS[@]}"; do
  prepared="${PREP_ROOT}/seed_${seed}/prepared"
  score_dir="${SCORE_ROOT}/seed_${seed}"
  if [[ -e "${score_dir}/.score_complete" ]]; then
    echo "Fold-matched score already complete for seed ${seed}; skipping."
    continue
  fi
  [[ ! -e "${score_dir}" ]] || {
    echo "Incomplete score directory cannot be resumed safely: ${score_dir}" >&2
    exit 2
  }
  score_partial="${score_dir}.partial_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}"
  [[ ! -e "${score_partial}" ]] || {
    echo "Current-attempt score directory already exists: ${score_partial}" >&2
    exit 2
  }
  echo "Starting Round-0 fold-matched replay for seed ${seed}"
  "${PYTHON}" -u "${PROGRAM}" score \
    --blind-cohort "${prepared}/blind_noisy_cohort.csv" \
    --sentinel-cohort "${prepared}/sentinel_blind_cohort.csv" \
    --split-reference-cohort "${prepared}/blind_noisy_cohort.csv" \
    --image-index "${IMAGE_INDEX}" \
    --image-root "${IMAGE_ROOT}" \
    --output-dir "${score_partial}" \
    --seed "${seed}" \
    --n-splits 4 \
    --epochs 50 \
    --early-stopping-patience 8 \
    --learning-rate 0.001 \
    --batch-size 32 \
    --num-workers 4 \
    --device cuda
  mv "${score_partial}" "${score_dir}"
done

if [[ -e "${CALIBRATION_DIR}/.calibration_frozen" ]]; then
  echo "Calibration is already frozen; reusing its locked thresholds."
else
  [[ ! -e "${CALIBRATION_DIR}" ]] || {
    echo "Incomplete calibration directory cannot be resumed safely" >&2
    exit 2
  }
  calibration_partial="${CALIBRATION_DIR}.partial_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}"
  [[ ! -e "${calibration_partial}" ]] || {
    echo "Current-attempt calibration directory already exists" >&2
    exit 2
  }
  "${PYTHON}" -u "${PROGRAM}" calibrate \
    --score-root "${SCORE_ROOT}" \
    --prepared-root "${PREP_ROOT}" \
    --trajectory-root "${TRAJECTORY_ROOT}" \
    --hardness-evidence "${HARDNESS}" \
    --output-dir "${calibration_partial}" \
    --seeds "${SEEDS_CSV}"
  mv "${calibration_partial}" "${CALIBRATION_DIR}"
fi

THRESHOLDS="${CALIBRATION_DIR}/calibration_thresholds_private.csv"
THRESHOLD_SHA="$(sha256sum "${THRESHOLDS}" | cut -d' ' -f1)"
[[ "$(tr -d '\n' < "${CALIBRATION_DIR}/.calibration_frozen")" == "${THRESHOLD_SHA}" ]] || {
  echo "Frozen-threshold marker disagrees with threshold file" >&2
  exit 2
}

# Action-set references are intentionally not read until the thresholds above are frozen.
for seed in "${SEEDS[@]}"; do
  prepared="${PREP_ROOT}/seed_${seed}/prepared"
  check_hash "${prepared}/private_reference.csv" "${EXPECTED_PRIVATE_SHA[${seed}]}"
  check_hash "${prepared}/prepare_summary.json" "${EXPECTED_PREPARE_SUMMARY_SHA[${seed}]}"
done

if [[ -e "${EVALUATION_DIR}/.analysis_complete" ]]; then
  echo "Transfer evaluation is already complete; skipping."
else
  [[ ! -e "${EVALUATION_DIR}" ]] || {
    echo "Incomplete evaluation directory cannot be resumed safely" >&2
    exit 2
  }
  evaluation_partial="${EVALUATION_DIR}.partial_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}"
  [[ ! -e "${evaluation_partial}" ]] || {
    echo "Current-attempt evaluation directory already exists" >&2
    exit 2
  }
  "${PYTHON}" -u "${PROGRAM}" evaluate \
    --thresholds "${THRESHOLDS}" \
    --expected-thresholds-sha256 "${THRESHOLD_SHA}" \
    --trajectory-root "${TRAJECTORY_ROOT}" \
    --output-dir "${evaluation_partial}" \
    --seeds "${SEEDS_CSV}"
  mv "${evaluation_partial}" "${EVALUATION_DIR}"
fi

"${PYTHON}" - "${OUTPUT_ROOT}" "${THRESHOLD_SHA}" <<'PY'
from pathlib import Path
import json
import pandas as pd
import sys

root, threshold_sha = Path(sys.argv[1]), sys.argv[2]
calibration = root / "calibration"
evaluation = root / "evaluation"
if not (calibration / ".calibration_frozen").is_file():
    raise RuntimeError("Calibration marker is missing")
if not (evaluation / ".analysis_complete").is_file():
    raise RuntimeError("Evaluation marker is missing")
thresholds = pd.read_csv(calibration / "calibration_thresholds_private.csv")
if len(thresholds) != 32 or thresholds[["seed", "fold_id"]].duplicated().any():
    raise RuntimeError("Threshold table postflight failed")
manifest = json.loads((calibration / "calibration_manifest_private.json").read_text())
if manifest["thresholds_sha256"] != threshold_sha or manifest["action_reference_accessed"] is not False:
    raise RuntimeError("Calibration freeze provenance failed")
summary = json.loads((evaluation / "action_transfer_summary_private.json").read_text())
if summary["thresholds_sha256"] != threshold_sha or len(summary["seeds"]) != 8:
    raise RuntimeError("Evaluation provenance failed")
print("VinDr DQS calibration-transfer postflight passed")
PY

printf 'job_id=%s\nthresholds_sha256=%s\n' "${SLURM_JOB_ID:-manual}" "${THRESHOLD_SHA}" \
  > "${OUTPUT_ROOT}/.job_complete"
echo "VinDr DQS calibration-transfer simulation completed"
