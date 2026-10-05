#!/bin/bash
#SBATCH --job-name=vindr-oof-aum
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_oof_aum_%A_%a.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_oof_aum_%A_%a.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 1 )); then
  echo "A Slurm array task id in 0-1 is required" >&2
  exit 2
fi

PROJECT_ROOT="/vol/gpudata/yz3522-llmtest/MResProject"
NOISE_ROOT="${PROJECT_ROOT}/NoiseRate"
SCRIPT_DIR="${NOISE_ROOT}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
PROGRAM="${SCRIPT_DIR}/vindr_oof_aum_screen.py"
TEST_PROGRAM="${SCRIPT_DIR}/test_vindr_oof_aum_screen.py"
DETECTOR_TEST="${SCRIPT_DIR}/test_vindr_detector_alternatives_benchmark.py"
PROTOCOL="${SCRIPT_DIR}/vindr_oof_aum_screen_protocol_20260812.md"
MOBILENET_HELPER="${SCRIPT_DIR}/vindr_mobilenet_oof.py"
DETECTOR_HELPER="${SCRIPT_DIR}/vindr_detector_alternatives_benchmark.py"
DATASET_HELPER="${SCRIPT_DIR}/cxr_real_full_train_eval_cleanlab_xrv12.py"
CL_HELPER="${SCRIPT_DIR}/cxr_real_noise_validation_smoke.py"
BENCHMARK_HELPER="${SCRIPT_DIR}/vindr_known_gt_cl_benchmark.py"
PARENT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_global_symmetric_noise_mobilenet/20260805_v1"
IMAGE_ROOT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/derived/png224_20260804_v1/images"
OUTPUT_ROOT="${VINDR_OOF_AUM_OUTPUT_OVERRIDE:-/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_oof_aum_screen/20260812_v1}"

SEEDS=(13 211)
BLIND_HASHES=(
  685f33adb62786da2ad0b84725e70d5b88f1aab78ed6773fd6b9c626353e46c5
  1f9b728404106e34f6e1d4fbf98a0fe93f7bd418b808c229efb53fb6c7f51e0a
)
PRIVATE_HASHES=(
  e8b2e3c467824326e61116ad3bce487da8a0818712c457f122dd054f66959082
  39ebf8c3df186018569e0d9931ca5a8c03cfea213faf59252f4093f140e0e3f6
)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
PREPARED="${PARENT_ROOT}/scenarios/symmetric_entry_r20/seed_${SEED}/prepared"
BLIND="${PREPARED}/blind_noisy_cohort.csv"
PRIVATE="${PREPARED}/private_reference.csv"
OUTPUT="${OUTPUT_ROOT}/seed_${SEED}"

for item in \
  "${PROGRAM}:f9fbb79d256bf66dc0c8720f3eb87e878cab78613816678daa7e338224ab01f7" \
  "${TEST_PROGRAM}:6e82c02d7f88a0dfd427b12bcfa94d2de0c576f1e308067b064a15e7ee2e38e4" \
  "${DETECTOR_TEST}:94b48002db1179ffe923025fe84101dc94a77d8d47f8ed44591d354d548c6507" \
  "${PROTOCOL}:64cecd91655483bdbf06e1fbd3f34d9978282162b0d4086a3ed841f6a4f57300" \
  "${MOBILENET_HELPER}:9d1585d8d37b7c8043516889b9b3d6ddda984425f910a891d889b0c2ecd9b026" \
  "${DETECTOR_HELPER}:f872eb109a3d1711409e5ed18fc74d204ed644ce9da8b05a31374d7ad5eabcda" \
  "${DATASET_HELPER}:f9b1eb32aaa7d6df9e1d1ebd3ba66fd2ef637bde47676fda4c7b629a7c7f452d" \
  "${CL_HELPER}:62da5b019f9809b4641b58f3c9bda56c1f987057fc765f9a76f32feade7046e0" \
  "${BENCHMARK_HELPER}:f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83" \
  "${PARENT_ROOT}/image_index.csv:645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2" \
  "${BLIND}:${BLIND_HASHES[${SLURM_ARRAY_TASK_ID}]}" \
  "${PRIVATE}:${PRIVATE_HASHES[${SLURM_ARRAY_TASK_ID}]}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  if [[ "$(sha256sum "${path}" | cut -d' ' -f1)" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done
for marker in "${PARENT_ROOT}/.benchmark_verified" "${PREPARED}/.prepare_complete"; do
  if [[ ! -f "${marker}" ]]; then
    echo "Required marker is missing: ${marker}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to overwrite AUM screen output: ${OUTPUT}" >&2
  exit 2
fi

source /vol/cuda/12.5.0/setup.sh
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${PROJECT_ROOT}/slurm_logs" "${OUTPUT_ROOT}" "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}"

echo "VinDr OOF AUM-style screen"
echo "Job=${SLURM_JOB_ID:-manual}; task=${SLURM_ARRAY_TASK_ID}; seed=${SEED}"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"${PYTHON}" -m unittest -v test_vindr_oof_aum_screen test_vindr_detector_alternatives_benchmark
"${PYTHON}" -u "${PROGRAM}" run-blind \
  --blind-cohort "${BLIND}" \
  --image-index "${PARENT_ROOT}/image_index.csv" \
  --image-root "${IMAGE_ROOT}" \
  --output-dir "${OUTPUT}" \
  --seed "${SEED}" \
  --n-splits 4 \
  --epochs 100 \
  --early-stopping-patience 10 \
  --learning-rate 0.001 \
  --batch-size 32 \
  --num-workers 4 \
  --device cuda
"${PYTHON}" -u "${PROGRAM}" evaluate \
  --output-dir "${OUTPUT}" \
  --private-reference "${PRIVATE}" \
  --seed "${SEED}"

if [[ ! -f "${OUTPUT}/.benchmark_verified" ]] || [[ ! -f "${OUTPUT}/private_evaluation/.evaluation_complete" ]]; then
  echo "AUM screen postflight marker is missing" >&2
  exit 2
fi
echo "Completed: $(date --iso-8601=seconds)"
