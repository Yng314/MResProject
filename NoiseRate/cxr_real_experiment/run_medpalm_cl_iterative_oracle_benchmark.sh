#!/bin/bash
#SBATCH --job-name=medpalm-cl-loop
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/medpalm_cl_loop_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/medpalm_cl_loop_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 077

PROJECT_ROOT="/vol/gpudata/yz3522-llmtest/MResProject"
NOISE_ROOT="${PROJECT_ROOT}/NoiseRate"
SCRIPT_DIR="${NOISE_ROOT}/cxr_real_experiment"
SCRIPT="${SCRIPT_DIR}/medpalm_cl_iterative_oracle_benchmark.py"
PROTOCOL="${SCRIPT_DIR}/medpalm_cl_iterative_oracle_protocol_20260803.md"
FULL_TRAIN_SCRIPT="${SCRIPT_DIR}/cxr_real_full_train_eval_cleanlab_xrv12.py"
OOF_SCRIPT="${SCRIPT_DIR}/cxr_real_oof_cleanlab_smoke.py"
DATA_SCRIPT="${SCRIPT_DIR}/cxr_real_noise_validation_smoke.py"
DETECTION_SCRIPT="${SCRIPT_DIR}/medpalm_cl_detection_benchmark.py"

DATA_ROOT="${PROJECT_ROOT}/MedSoul/datasets"
IMAGE_ROOT="${DATA_ROOT}/mimic-cxr-jpg-224"
CHEXPERT_CSV="${IMAGE_ROOT}/mimic-cxr-2.0.0-chexpert.csv"
SPLIT_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-split.csv.gz"
METADATA_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz"

RESULT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment"
MEDPALM_ROOT="${RESULT_ROOT}/medpalm_external_entry_benchmark/20260730_binary_external_no_oof"
PROFILE="${1:-formal}"
case "${PROFILE}" in
  formal)
    DEFAULT_OUTPUT="${RESULT_ROOT}/medpalm_cl_iterative_oracle_benchmark/20260803_seed13_dynamic5loop"
    DEFAULT_LOOPS=5
    DEFAULT_REVIEW_BUDGET=100
    DEFAULT_N_SPLITS=4
    DEFAULT_EPOCHS=100
    DEFAULT_RANDOM_REPLICATES=10000
    ;;
  smoke)
    DEFAULT_OUTPUT="${RESULT_ROOT}/medpalm_cl_iterative_oracle_benchmark/smoke_20260803_1loop_2fold_1epoch"
    DEFAULT_LOOPS=1
    DEFAULT_REVIEW_BUDGET=498
    DEFAULT_N_SPLITS=2
    DEFAULT_EPOCHS=1
    DEFAULT_RANDOM_REPLICATES=100
    ;;
  *)
    echo "Profile must be formal or smoke; received ${PROFILE}." >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${MEDPALM_ITER_OUTPUT_OVERRIDE:-${DEFAULT_OUTPUT}}"
LOOPS="${MEDPALM_ITER_LOOPS_OVERRIDE:-${DEFAULT_LOOPS}}"
REVIEW_BUDGET="${MEDPALM_ITER_REVIEW_BUDGET_OVERRIDE:-${DEFAULT_REVIEW_BUDGET}}"
SEED="${MEDPALM_ITER_SEED_OVERRIDE:-13}"
N_SPLITS="${MEDPALM_ITER_N_SPLITS_OVERRIDE:-${DEFAULT_N_SPLITS}}"
EPOCHS="${MEDPALM_ITER_EPOCHS_OVERRIDE:-${DEFAULT_EPOCHS}}"
BATCH_SIZE="${MEDPALM_ITER_BATCH_SIZE_OVERRIDE:-32}"
RANDOM_REPLICATES="${MEDPALM_ITER_RANDOM_REPLICATES_OVERRIDE:-${DEFAULT_RANDOM_REPLICATES}}"

EXPECTED_SCRIPT_SHA256="cdc607e124210d5947c466fed677db7abd735e5d153ee888520ef23f98abe26b"
EXPECTED_PROTOCOL_SHA256="bd4e9cba0d9cd03d503287f1ba85eb207aade7d00efb0efac325ae1c2042aa10"
EXPECTED_FULL_TRAIN_SHA256="f9b1eb32aaa7d6df9e1d1ebd3ba66fd2ef637bde47676fda4c7b629a7c7f452d"
EXPECTED_OOF_SHA256="dc6b2d577985bbeec4d1c9e71911fca78419f63126c7c4eaa1d2397943c29a7b"
EXPECTED_DATA_SHA256="62da5b019f9809b4641b58f3c9bda56c1f987057fc765f9a76f32feade7046e0"
EXPECTED_DETECTION_SHA256="624bc89537f95d2cf19440ff5c94f9f2c5b49d7fd859737df5304135640a0bef"

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
check_hash "${FULL_TRAIN_SCRIPT}" "${EXPECTED_FULL_TRAIN_SHA256}"
check_hash "${OOF_SCRIPT}" "${EXPECTED_OOF_SHA256}"
check_hash "${DATA_SCRIPT}" "${EXPECTED_DATA_SHA256}"
check_hash "${DETECTION_SCRIPT}" "${EXPECTED_DETECTION_SHA256}"

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${OUTPUT_DIR}/source_snapshot"
source /vol/cuda/12.5.0/setup.sh
source /vol/gpudata/yz3522-llmtest/venv/bin/activate
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"

cp \
  "${SCRIPT}" \
  "${PROTOCOL}" \
  "${FULL_TRAIN_SCRIPT}" \
  "${OOF_SCRIPT}" \
  "${DATA_SCRIPT}" \
  "${DETECTION_SCRIPT}" \
  "${OUTPUT_DIR}/source_snapshot/"

COMMON_ARGS=(
  --output-dir "${OUTPUT_DIR}"
  --loops "${LOOPS}"
  --review-budget "${REVIEW_BUDGET}"
  --seed "${SEED}"
  --n-splits "${N_SPLITS}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers 4
  --learning-rate 0.001
  --early-stopping-patience 10
)
DATA_ARGS=(
  --image-root "${IMAGE_ROOT}"
  --chexpert-csv "${CHEXPERT_CSV}"
  --split-csv "${SPLIT_CSV}"
  --metadata-csv "${METADATA_CSV}"
  --benchmark-blinded "${MEDPALM_ROOT}/benchmark_entries_blinded.csv"
)

echo "=========================================================="
echo "Med-PaLM dynamic-OOF iterative CL oracle benchmark"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Profile: ${PROFILE}"
echo "Output: ${OUTPUT_DIR}"
echo "Loops: ${LOOPS}; review budget: ${REVIEW_BUDGET}"
echo "Seed: ${SEED}; folds: ${N_SPLITS}; max epochs: ${EPOCHS}"
echo "LLM API calls: none"
echo "Started: $(date --iso-8601=seconds)"
echo "=========================================================="

python -u "${SCRIPT}" preflight "${COMMON_ARGS[@]}" "${DATA_ARGS[@]}"
python -u "${SCRIPT}" initialize "${COMMON_ARGS[@]}" \
  --benchmark-blinded "${MEDPALM_ROOT}/benchmark_entries_blinded.csv"

for LOOP_ID in $(seq 1 "${LOOPS}"); do
  echo ""
  echo "================ Blind dynamic OOF Loop ${LOOP_ID}/${LOOPS} ================"
  python -u "${SCRIPT}" run-loop \
    "${COMMON_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    --loop-id "${LOOP_ID}" \
    --device cuda

  echo "================ Selected-only oracle update ${LOOP_ID}/${LOOPS} ================"
  python -u "${SCRIPT}" oracle-update \
    "${COMMON_ARGS[@]}" \
    --loop-id "${LOOP_ID}" \
    --benchmark-reference "${MEDPALM_ROOT}/benchmark_reference_private.csv"
done

python -u "${SCRIPT}" evaluate \
  "${COMMON_ARGS[@]}" \
  --benchmark-reference "${MEDPALM_ROOT}/benchmark_reference_private.csv" \
  --random-replicates "${RANDOM_REPLICATES}" \
  --random-seed 20260803

test -s "${OUTPUT_DIR}/.evaluation_complete"
echo "Completed: $(date --iso-8601=seconds)"
