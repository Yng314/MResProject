#!/bin/bash
#SBATCH --job-name=medpalm-cl-detect
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/medpalm_cl_detect_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/medpalm_cl_detect_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 077

PROJECT_ROOT="/vol/gpudata/yz3522-llmtest/MResProject"
NOISE_ROOT="${PROJECT_ROOT}/NoiseRate"
SCRIPT_DIR="${NOISE_ROOT}/cxr_real_experiment"
SCRIPT="${SCRIPT_DIR}/medpalm_cl_detection_benchmark.py"
PROTOCOL="${SCRIPT_DIR}/medpalm_cl_detection_protocol_20260803.md"
TRAINING_SCRIPT="${SCRIPT_DIR}/cxr_real_full_train_eval_cleanlab_xrv12.py"
DATA_SCRIPT="${SCRIPT_DIR}/cxr_real_noise_validation_smoke.py"

DATA_ROOT="${PROJECT_ROOT}/MedSoul/datasets"
IMAGE_ROOT="${DATA_ROOT}/mimic-cxr-jpg-224"
CHEXPERT_CSV="${IMAGE_ROOT}/mimic-cxr-2.0.0-chexpert.csv"
SPLIT_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-split.csv.gz"
METADATA_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz"

RESULT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment"
MODEL_ROOT="${RESULT_ROOT}/results_mobilenetv3_noes50_clean_3seed/20260707_123320"
MEDPALM_ROOT="${RESULT_ROOT}/medpalm_external_entry_benchmark/20260730_binary_external_no_oof"
OUTPUT_DIR="${MEDPALM_CL_OUTPUT_OVERRIDE:-${RESULT_ROOT}/medpalm_cl_detection_benchmark/20260803_frozen4seed}"

EXPECTED_SCRIPT_SHA256="624bc89537f95d2cf19440ff5c94f9f2c5b49d7fd859737df5304135640a0bef"
EXPECTED_PROTOCOL_SHA256="0605e809a281b6d455558da9a063a75cf7d486bd25eb4dd415483b0a16efa825"
EXPECTED_TRAINING_SCRIPT_SHA256="f9b1eb32aaa7d6df9e1d1ebd3ba66fd2ef637bde47676fda4c7b629a7c7f452d"
EXPECTED_DATA_SCRIPT_SHA256="62da5b019f9809b4641b58f3c9bda56c1f987057fc765f9a76f32feade7046e0"

check_hash() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "${path}" | cut -d' ' -f1)"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "Hash mismatch for ${path}: expected ${expected}, found ${actual}" >&2
    exit 2
  fi
}

check_hash "${SCRIPT}" "${EXPECTED_SCRIPT_SHA256}"
check_hash "${PROTOCOL}" "${EXPECTED_PROTOCOL_SHA256}"
check_hash "${TRAINING_SCRIPT}" "${EXPECTED_TRAINING_SCRIPT_SHA256}"
check_hash "${DATA_SCRIPT}" "${EXPECTED_DATA_SCRIPT_SHA256}"

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${OUTPUT_DIR}/source_snapshot"
source /vol/cuda/12.5.0/setup.sh
source /vol/gpudata/yz3522-llmtest/venv/bin/activate
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"

cp "${SCRIPT}" "${PROTOCOL}" "${TRAINING_SCRIPT}" "${DATA_SCRIPT}" "${OUTPUT_DIR}/source_snapshot/"

echo "=========================================================="
echo "Med-PaLM confident-learning detection benchmark"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Output: ${OUTPUT_DIR}"
echo "Frozen seeds: 13, 42, 97, 123"
echo "Iterative loops: none"
echo "LLM API calls: none"
echo "Started: $(date --iso-8601=seconds)"
echo "=========================================================="

if [[ ! -s "${OUTPUT_DIR}/.inference_complete" ]]; then
  python -u "${SCRIPT}" infer \
    --output-dir "${OUTPUT_DIR}" \
    --image-root "${IMAGE_ROOT}" \
    --chexpert-csv "${CHEXPERT_CSV}" \
    --split-csv "${SPLIT_CSV}" \
    --metadata-csv "${METADATA_CSV}" \
    --checkpoint "13=${MODEL_ROOT}/seed_13/baseline_no_clean/baseline_model.pt" \
    --checkpoint "42=${MODEL_ROOT}/seed_42/baseline_no_clean/baseline_model.pt" \
    --checkpoint "97=${MODEL_ROOT}/seed_97/baseline_no_clean/baseline_model.pt" \
    --checkpoint "123=${MODEL_ROOT}/seed_123/baseline_no_clean/baseline_model.pt" \
    --device cuda \
    --batch-size 64 \
    --num-workers 8
else
  echo "[resume] Inference is already complete."
fi

if [[ ! -s "${OUTPUT_DIR}/.scores_complete" ]]; then
  python -u "${SCRIPT}" score --output-dir "${OUTPUT_DIR}"
else
  echo "[resume] Outcome-blind CL scoring is already complete."
fi

if [[ ! -s "${OUTPUT_DIR}/.evaluation_complete" ]]; then
  python -u "${SCRIPT}" evaluate \
    --output-dir "${OUTPUT_DIR}" \
    --benchmark-blinded "${MEDPALM_ROOT}/benchmark_entries_blinded.csv" \
    --benchmark-reference "${MEDPALM_ROOT}/benchmark_reference_private.csv" \
    --llm-evaluation "${MEDPALM_ROOT}/row_level_evaluation_private.csv" \
    --bootstrap-replicates 2000 \
    --bootstrap-seed 20260803 \
    --synthetic-replicates 100
else
  echo "[resume] Private-reference evaluation is already complete."
fi

test -s "${OUTPUT_DIR}/.inference_complete"
test -s "${OUTPUT_DIR}/.scores_complete"
test -s "${OUTPUT_DIR}/.evaluation_complete"

echo "Completed: $(date --iso-8601=seconds)"
