#!/bin/bash
#SBATCH --job-name=vindr-det-v2-smoke
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_det_v2_smoke_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_det_v2_smoke_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
SCRIPT_DIR="${ROOT}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
PROGRAM="${SCRIPT_DIR}/vindr_detector_benchmark_v2.py"
TEST_PROGRAM="${SCRIPT_DIR}/test_vindr_detector_benchmark_v2.py"
PROTOCOL="${SCRIPT_DIR}/vindr_detector_benchmark_v2_protocol_20260812.md"
PARENT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_noise_direction_sensitivity_mobilenet/20260805_v1"
IMAGE_PARENT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/derived/png224_20260804_v1"
IMAGE_ROOT="${IMAGE_PARENT}/images"
XRV_CACHE="/vol/gpudata/yz3522-llmtest/.cache/torchxrayvision"
XRV_WEIGHTS="${XRV_CACHE}/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
OUTPUT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_detector_benchmark_v2_smoke/20260812_v1"
EXTERNAL="${OUTPUT}/external_evidence"
SCORES="${OUTPUT}/blind_detector_scores"

EXPECTED_PROGRAM_SHA="b77e5f3db073782cc321a71eed91e2da2d5177ab5cd641b169535ea711b5b870"
EXPECTED_TEST_SHA="bcd7562fd121c9a64f6bccef9331c4e3d9fd3f272a488ada0219f9908a3adf67"
EXPECTED_PROTOCOL_SHA="c4524cb159d5f8ff7a6f18a5bdf444aa15f77407a0b8731077dc21cd661c9de7"
EXPECTED_INDEX_SHA="645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2"
EXPECTED_MANIFEST_SHA="a17fba9c02c8a8c9428c1cffa2afee3261980e3c59a4905a68be1bbb8d0da6c4"
EXPECTED_WEIGHTS_SHA="56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899"

for item in \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${TEST_PROGRAM}:${EXPECTED_TEST_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${PARENT}/image_index.csv:${EXPECTED_INDEX_SHA}" \
  "${PARENT}/blind_run_manifest.csv:${EXPECTED_MANIFEST_SHA}" \
  "${XRV_WEIGHTS}:${EXPECTED_WEIGHTS_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  if [[ "$(sha256sum "${path}" | cut -d' ' -f1)" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done
for marker in \
  "${PARENT}/.benchmark_verified" \
  "${IMAGE_PARENT}/.conversion_complete"; do
  [[ -e "${marker}" ]] || { echo "Required marker is missing: ${marker}" >&2; exit 2; }
done
if [[ -e "${OUTPUT}" ]]; then
  echo "Smoke output already exists; refusing to overwrite: ${OUTPUT}" >&2
  exit 2
fi

source /vol/cuda/12.5.0/setup.sh
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "${OUTPUT}" "${MPLCONFIGDIR}" "${ROOT%/NoiseRate}/slurm_logs"

echo "VinDr detector benchmark v2 smoke"
echo "Job: ${SLURM_JOB_ID:-manual}"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"${PYTHON}" -m unittest -v test_vindr_detector_benchmark_v2
"${PYTHON}" -u "${PROGRAM}" extract-external \
  --image-index "${PARENT}/image_index.csv" \
  --image-root "${IMAGE_ROOT}" \
  --output-dir "${EXTERNAL}" \
  --xrv-cache-dir "${XRV_CACHE}" \
  --device cuda \
  --batch-size 32 \
  --num-workers 4 \
  --seed 13
"${PYTHON}" -u "${PROGRAM}" score \
  --parent-root "${PARENT}" \
  --output-root "${SCORES}" \
  --external-evidence "${EXTERNAL}/xrv_external_evidence.npz" \
  --seeds 211 \
  --scenarios balanced_r20
"${PYTHON}" -u "${PROGRAM}" evaluate \
  --parent-root "${PARENT}" \
  --output-root "${SCORES}"
"${PYTHON}" -u "${PROGRAM}" verify \
  --parent-root "${PARENT}" \
  --output-root "${SCORES}" \
  --expected-runs 1
printf 'smoke_job_id=%s\n' "${SLURM_JOB_ID:-manual}" > "${OUTPUT}/.smoke_complete"
echo "Completed: $(date --iso-8601=seconds)"
