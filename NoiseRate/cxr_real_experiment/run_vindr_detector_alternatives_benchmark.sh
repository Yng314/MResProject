#!/bin/bash
#SBATCH --job-name=vindr-det-alt
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_det_alt_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_det_alt_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

PROJECT_ROOT="/vol/gpudata/yz3522-llmtest/MResProject"
NOISE_ROOT="${PROJECT_ROOT}/NoiseRate"
SCRIPT_DIR="${NOISE_ROOT}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
PROGRAM="${SCRIPT_DIR}/vindr_detector_alternatives_benchmark.py"
TEST_PROGRAM="${SCRIPT_DIR}/test_vindr_detector_alternatives_benchmark.py"
PROTOCOL="${SCRIPT_DIR}/vindr_detector_alternatives_protocol_20260812.md"
PARENT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_global_symmetric_noise_mobilenet/20260805_v1"
FEATURES="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_known_gt_cl/20260804_v1/features/xrv_features.npz"
OUTPUT_ROOT="${VINDR_DETECTOR_ALTERNATIVES_OUTPUT_OVERRIDE:-/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_detector_alternatives/20260812_v1}"

EXPECTED_PROGRAM_SHA="f872eb109a3d1711409e5ed18fc74d204ed644ce9da8b05a31374d7ad5eabcda"
EXPECTED_TEST_SHA="94b48002db1179ffe923025fe84101dc94a77d8d47f8ed44591d354d548c6507"
EXPECTED_PROTOCOL_SHA="8399b79fc418b5278559c1c697cfa54bff5df11f471a54146f0190341be974d8"
EXPECTED_PARENT_MANIFEST_SHA="cdff8ca8ec86f0a061f9c4856ed0888573c4defe68a31a1784484f7652eb0f17"
EXPECTED_FEATURES_SHA="d361d521e681ea8fdff94ba83c206680123032eb247a5b84ffd77d393a895327"

for item in \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${TEST_PROGRAM}:${EXPECTED_TEST_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${PARENT_ROOT}/scenario_manifest.csv:${EXPECTED_PARENT_MANIFEST_SHA}" \
  "${FEATURES}:${EXPECTED_FEATURES_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  actual="$(sha256sum "${path}" | cut -d' ' -f1)"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done

for marker in \
  "${PARENT_ROOT}/.benchmark_verified" \
  "$(dirname "${FEATURES}")/.features_complete"; do
  if [[ ! -f "${marker}" ]]; then
    echo "Required marker is missing: ${marker}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite formal output: ${OUTPUT_ROOT}" >&2
  exit 2
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "${PROJECT_ROOT}/slurm_logs" "${MPLCONFIGDIR}" "$(dirname "${OUTPUT_ROOT}")"

echo "VinDr detector alternatives benchmark"
echo "Job: ${SLURM_JOB_ID:-manual}"
echo "Output: ${OUTPUT_ROOT}"
echo "Started: $(date --iso-8601=seconds)"

"${PYTHON}" -m unittest -v test_vindr_detector_alternatives_benchmark
"${PYTHON}" -u "${PROGRAM}" score \
  --parent-root "${PARENT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --features-npz "${FEATURES}" \
  --k-values 10,20,50 \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}"
"${PYTHON}" -u "${PROGRAM}" evaluate \
  --parent-root "${PARENT_ROOT}" \
  --output-root "${OUTPUT_ROOT}"
"${PYTHON}" -u "${PROGRAM}" verify \
  --parent-root "${PARENT_ROOT}" \
  --output-root "${OUTPUT_ROOT}"

echo "Completed: $(date --iso-8601=seconds)"
