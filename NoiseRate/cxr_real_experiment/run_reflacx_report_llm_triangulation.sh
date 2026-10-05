#!/bin/bash
#SBATCH --job-name=reflacx-report-llm
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/reflacx_report_llm_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/reflacx_report_llm_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail

PROJECT_ROOT="/vol/gpudata/yz3522-llmtest/MResProject"
NOISERATE_ROOT="${PROJECT_ROOT}/NoiseRate"
SCRIPT_DIR="${NOISERATE_ROOT}/cxr_real_experiment"
SCRIPT="${SCRIPT_DIR}/reflacx_report_llm_triangulation.py"
PROTOCOL="${SCRIPT_DIR}/reflacx_report_llm_triangulation_protocol_20260801.md"
PILOT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/reflacx_phase3_cl_pilot/20260730_seed13"
BLIND_COHORT="${PILOT_ROOT}/prepared/blind_cohort.csv"
PRIVATE_REFERENCE="${PILOT_ROOT}/prepared/private_reference.csv"
ENTRY_EVIDENCE="${PILOT_ROOT}/blind_run/entry_evidence.csv"
CHEXPERT="${PROJECT_ROOT}/MedSoul/datasets/mimic-cxr-2.0.0-chexpert.csv.gz"
REPORT_ARCHIVE="${PROJECT_ROOT}/MedSoul/datasets/mimic-cxr-reports.tar.gz"
OUTPUT_DIR="${1:-/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/reflacx_report_llm_triangulation/20260801_gpt54_independent}"
EXPECTED_MODEL="gpt-5.4"
EXPECTED_SCRIPT_SHA256="d2b7bd2491a5e2d323645556f4d9e5a744165eb20e675fd475941a4f7ad5c541"
EXPECTED_PROTOCOL_SHA256="cc527343747cb67f8dddef79e6818143658263ec73fd60f4cd07e20b42c591c1"

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${OUTPUT_DIR}/protocol_snapshot"
source /vol/gpudata/yz3522-llmtest/venv/bin/activate
source ~/.llm_review_env

if [[ -e "${OUTPUT_DIR}/.evaluation_complete" ]]; then
  echo "Refusing to overwrite completed evaluation: ${OUTPUT_DIR}" >&2
  exit 2
fi
if [[ "$(sha256sum "${SCRIPT}" | cut -d' ' -f1)" != "${EXPECTED_SCRIPT_SHA256}" ]]; then
  echo "Triangulation execution script changed after protocol lock." >&2
  exit 2
fi
if [[ "$(sha256sum "${PROTOCOL}" | cut -d' ' -f1)" != "${EXPECTED_PROTOCOL_SHA256}" ]]; then
  echo "Triangulation protocol changed after lock." >&2
  exit 2
fi
if [[ "${OPENAI_MODEL:-}" != "${EXPECTED_MODEL}" ]]; then
  echo "OPENAI_MODEL must be ${EXPECTED_MODEL}; found ${OPENAI_MODEL:-unset}." >&2
  exit 2
fi

cp "${SCRIPT}" "${PROTOCOL}" "$0" "${OUTPUT_DIR}/protocol_snapshot/"
sha256sum "${OUTPUT_DIR}"/protocol_snapshot/* > "${OUTPUT_DIR}/protocol_snapshot/SHA256SUMS.txt"
python -m py_compile "${SCRIPT}"

echo "============================================================"
echo "REFLACX report-only LLM triangulation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Model: ${OPENAI_MODEL}"
echo "Full GPU requested: none (one scheduler shard)"
echo "Output: ${OUTPUT_DIR}"
echo "Started: $(date --iso-8601=seconds)"
echo "============================================================"

python -u "${SCRIPT}" prepare \
  --blind-cohort "${BLIND_COHORT}" \
  --private-reference "${PRIVATE_REFERENCE}" \
  --entry-evidence "${ENTRY_EVIDENCE}" \
  --chexpert-csv "${CHEXPERT}" \
  --report-archive "${REPORT_ARCHIVE}" \
  --output-dir "${OUTPUT_DIR}"

for pass in 1 2; do
  echo ""
  echo "[Review pass ${pass}/2]"
  python -u "${SCRIPT}" review \
    --output-dir "${OUTPUT_DIR}" \
    --llm-mode real \
    --resume \
    --batch-size 100 \
    --concurrency 10 \
    --retry-limit 2 \
    --request-timeout-seconds 180 \
    --temperature 0 \
    --max-completion-tokens 400
done

python -u "${SCRIPT}" verify --output-dir "${OUTPUT_DIR}"
python -u "${SCRIPT}" evaluate \
  --output-dir "${OUTPUT_DIR}" \
  --bootstrap-replicates 2000 \
  --bootstrap-seed 20260801

echo "Completed: $(date --iso-8601=seconds)"
