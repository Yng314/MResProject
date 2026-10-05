#!/bin/bash
#SBATCH --job-name=mimcal-eval
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mimic_dqs_cal_eval_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mimic_dqs_cal_eval_%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 077

PROJECT_ROOT="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_ROOT}/NoiseRate/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
PROGRAM="${SCRIPT_DIR}/mimic_dqs_calibration_transfer.py"
TEST="${SCRIPT_DIR}/test_mimic_dqs_calibration_transfer.py"
PROTOCOL="${SCRIPT_DIR}/mimic_dqs_calibration_transfer_protocol_20260827.md"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/mimic_dqs_calibration_transfer/20260827_v1"
SCORE_ROOT="${OUTPUT_ROOT}/scores"
CALIBRATION_DIR="${OUTPUT_ROOT}/calibration"
EVALUATION_DIR="${OUTPUT_ROOT}/evaluation"
REFERENCE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/dual_reference_cl_detection_benchmark/20260803_frozen4seed_exact_top20"
RADIOLOGIST_REFERENCE="${REFERENCE_ROOT}/mimic_reference_private.csv"
MEDPALM_REFERENCE="${REFERENCE_ROOT}/medpalm_reference_private.csv"
SEEDS="7,13,42,97,123"

check_hash() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(sha256sum "${path}" | cut -d' ' -f1)"
  [[ "${observed}" == "${expected}" ]] || {
    echo "Hash mismatch for ${path}: ${observed} != ${expected}" >&2
    exit 2
  }
}

check_hash "${PROGRAM}" "1930bea8ba54811c540982a34fd171b96120db75d489f41129014884bf6b3678"
check_hash "${TEST}" "302a47b0e3c601532a455d21007adb631b27466ab9f6ad3879186d24eae16a44"
check_hash "${PROTOCOL}" "19b307b802993f239c3aae12063c78895e39d91126f2220d880ecc5f90273a0b"

for seed in 7 13 42 97 123; do
  [[ -s "${SCORE_ROOT}/seed_${seed}/.score_complete" ]] || {
    echo "Outcome-blind score marker is missing for seed ${seed}" >&2
    exit 2
  }
done
for required in "${RADIOLOGIST_REFERENCE}" "${MEDPALM_REFERENCE}"; do
  [[ -s "${required}" ]] || { echo "Missing reference: ${required}" >&2; exit 2; }
done

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${PROJECT_ROOT}/slurm_logs" "${OUTPUT_ROOT}/source_snapshot" "${MPLCONFIGDIR}"
"${PYTHON}" -m py_compile "${PROGRAM}" "${TEST}"
"${PYTHON}" "${TEST}"
cp "${PROGRAM}" "${TEST}" "${PROTOCOL}" "${OUTPUT_ROOT}/source_snapshot/"

echo "MIMIC-CXR DQS calibration transfer"
echo "Job: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURMD_NODENAME:-manual}"
echo "New model training: none"
echo "LLM API calls: none"

if [[ ! -e "${CALIBRATION_DIR}/.calibration_frozen" ]]; then
  [[ ! -e "${CALIBRATION_DIR}" ]] || {
    echo "Incomplete calibration directory already exists: ${CALIBRATION_DIR}" >&2
    exit 2
  }
  "${PYTHON}" -u "${PROGRAM}" calibrate \
    --score-root "${SCORE_ROOT}" \
    --radiologist-reference "${RADIOLOGIST_REFERENCE}" \
    --medpalm-reference "${MEDPALM_REFERENCE}" \
    --output-dir "${CALIBRATION_DIR}" \
    --seeds "${SEEDS}"
fi

THRESHOLD_HASH="$(tr -d '[:space:]' < "${CALIBRATION_DIR}/.calibration_frozen")"
[[ "${THRESHOLD_HASH}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "Frozen threshold hash is invalid" >&2
  exit 2
}

if [[ ! -e "${EVALUATION_DIR}/.evaluation_complete" ]]; then
  [[ ! -e "${EVALUATION_DIR}" ]] || {
    echo "Incomplete evaluation directory already exists: ${EVALUATION_DIR}" >&2
    exit 2
  }
  "${PYTHON}" -u "${PROGRAM}" evaluate \
    --score-root "${SCORE_ROOT}" \
    --thresholds "${CALIBRATION_DIR}/calibration_thresholds_private.csv" \
    --expected-thresholds-sha256 "${THRESHOLD_HASH}" \
    --output-dir "${EVALUATION_DIR}" \
    --seeds "${SEEDS}"
fi

"${PYTHON}" -u "${PROGRAM}" verify \
  --calibration-dir "${CALIBRATION_DIR}" \
  --evaluation-dir "${EVALUATION_DIR}"
printf 'job_id=%s\ncompleted_at=%s\n' "${SLURM_JOB_ID:-manual}" "$(date --iso-8601=seconds)" \
  > "${OUTPUT_ROOT}/.job_complete"
echo "MIMIC-CXR calibration-transfer analysis completed"
