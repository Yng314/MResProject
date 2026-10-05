#!/bin/bash
#SBATCH --job-name=cl-vs-unc
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/cl_vs_unc_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/cl_vs_unc_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 077

PROJECT_ROOT="/vol/gpudata/yz3522-llmtest/MResProject"
NOISE_ROOT="${PROJECT_ROOT}/NoiseRate"
SCRIPT_DIR="${NOISE_ROOT}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
SCRIPT="${SCRIPT_DIR}/cl_vs_uncertainty_ablation.py"
PROTOCOL="${SCRIPT_DIR}/cl_vs_uncertainty_ablation_protocol_20260803.md"
HELPER="${SCRIPT_DIR}/dual_reference_cl_detection_benchmark.py"

RESULT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment"
FROZEN_ROOT="${RESULT_ROOT}/medpalm_cl_detection_benchmark/20260803_frozen4seed"
MEDPALM_ROOT="${RESULT_ROOT}/medpalm_external_entry_benchmark/20260730_binary_external_no_oof"
DATA_ROOT="${PROJECT_ROOT}/MedSoul/datasets"
OUTPUT_DIR="${CL_VS_UNCERTAINTY_OUTPUT_OVERRIDE:-${RESULT_ROOT}/cl_vs_uncertainty_ablation/20260803_frozen4seed_equal_entry_budget}"

SCORE_CSV="${FROZEN_ROOT}/full_test_cl_entry_scores.csv"
SCORE_MANIFEST="${FROZEN_ROOT}/scoring_manifest.json"
MIMIC_REFERENCE="${DATA_ROOT}/mimic-cxr-2.1.0-test-set-labeled.csv"
MEDPALM_BLINDED="${MEDPALM_ROOT}/benchmark_entries_blinded.csv"
MEDPALM_REFERENCE="${MEDPALM_ROOT}/benchmark_reference_private.csv"

EXPECTED_SCRIPT_SHA256="39e93049ed3ab6337be3111de906d70aeada15081d7d1fa3396542be2139b6da"
EXPECTED_PROTOCOL_SHA256="2c2340bc8ef0034b38b6f5d671567463bff25d8d9d71be099f194e9fbfa0ecd5"
EXPECTED_HELPER_SHA256="bd0b639cf137bcfe0dc449dcbcb7282f0b8e046938460d93bee6ff7780010a90"

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
check_hash "${HELPER}" "${EXPECTED_HELPER_SHA256}"
for required in "${SCORE_CSV}" "${SCORE_MANIFEST}" "${MIMIC_REFERENCE}" "${MEDPALM_BLINDED}" "${MEDPALM_REFERENCE}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing or empty input: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${OUTPUT_DIR}/source_snapshot"
cp "${SCRIPT}" "${PROTOCOL}" "${HELPER}" "${OUTPUT_DIR}/source_snapshot/"

echo "CL versus uncertainty equal-budget ablation"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Output: ${OUTPUT_DIR}"
echo "New model training: none"
echo "LLM API calls: none"
echo "Started: $(date --iso-8601=seconds)"

cd "${SCRIPT_DIR}"
COMMON_ARGS=(
  --score-csv "${SCORE_CSV}"
  --score-manifest "${SCORE_MANIFEST}"
  --output-dir "${OUTPUT_DIR}"
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-5000}"
  --random-replicates "${RANDOM_REPLICATES:-10000}"
  --random-seed 20260803
)

if [[ ! -e "${OUTPUT_DIR}/.selection_complete" ]]; then
  "${PYTHON}" -u "${SCRIPT}" select "${COMMON_ARGS[@]}"
fi
if [[ ! -e "${OUTPUT_DIR}/.evaluation_complete" ]]; then
  "${PYTHON}" -u "${SCRIPT}" evaluate "${COMMON_ARGS[@]}" \
    --mimic-reference-csv "${MIMIC_REFERENCE}" \
    --medpalm-blinded-csv "${MEDPALM_BLINDED}" \
    --medpalm-reference-csv "${MEDPALM_REFERENCE}"
fi
"${PYTHON}" -u "${SCRIPT}" verify "${COMMON_ARGS[@]}"
echo "Completed: $(date --iso-8601=seconds)"
