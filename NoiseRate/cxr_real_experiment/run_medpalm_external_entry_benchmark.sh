#!/bin/bash
#SBATCH --job-name=medpalm-ext-review
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/medpalm_ext_review_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/medpalm_ext_review_%j.err
#SBATCH --partition=training
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail

PROJECT_ROOT="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_ROOT}/NoiseRate/cxr_real_experiment"
SCRIPT="${SCRIPT_DIR}/medpalm_external_entry_benchmark.py"
PROTOCOL="${SCRIPT_DIR}/medpalm_external_entry_benchmark_protocol_20260730.md"
GROUND_TRUTH="${MEDPALM_GT_OVERRIDE:-/homes/yz3522/.codex/attachments/e3a26137-0995-43bc-a5eb-902bb045f17c/external_medpalm2_ground_truth_labels_v1.csv}"
CHEXPERT="${PROJECT_ROOT}/MedSoul/datasets/mimic-cxr-jpg-224/mimic-cxr-2.0.0-chexpert.csv"
SPLIT="${PROJECT_ROOT}/MedSoul/datasets/mimic-cxr-2.0.0-split.csv.gz"
REPORT_ARCHIVE="${PROJECT_ROOT}/MedSoul/datasets/mimic-cxr-reports.tar.gz"
OUTPUT_DIR="${MEDPALM_OUTPUT_OVERRIDE:-/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/medpalm_external_entry_benchmark/20260730_binary_external_no_oof}"
EXPECTED_MODEL="gpt-5.4"
EXPECTED_SCRIPT_SHA256="f79b498d58bceff5992e7dd55a4f82284673239ea89f0a4dd28f71e2dba62a4a"
EXPECTED_PROTOCOL_SHA256="77072cfe6fdd2613754da0488250d0c3a592a8bf16da3ad79842720973339e60"

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${OUTPUT_DIR}"
source /vol/gpudata/yz3522-llmtest/venv/bin/activate
source ~/.llm_review_env

if [[ "$(sha256sum "${SCRIPT}" | cut -d' ' -f1)" != "${EXPECTED_SCRIPT_SHA256}" ]]; then
  echo "Benchmark execution script changed after protocol lock." >&2
  exit 2
fi
if [[ "$(sha256sum "${PROTOCOL}" | cut -d' ' -f1)" != "${EXPECTED_PROTOCOL_SHA256}" ]]; then
  echo "Benchmark protocol changed after lock." >&2
  exit 2
fi
if [[ "${OPENAI_MODEL:-}" != "${EXPECTED_MODEL}" ]]; then
  echo "OPENAI_MODEL must be ${EXPECTED_MODEL}; found ${OPENAI_MODEL:-unset}." >&2
  exit 2
fi

echo "============================================================"
echo "Med-PaLM external-suspicion binary reviewer benchmark"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Model: ${OPENAI_MODEL}"
echo "GPU requested: none"
echo "Output: ${OUTPUT_DIR}"
echo "Started: $(date --iso-8601=seconds)"
echo "============================================================"

python -u "${SCRIPT}" prepare \
  --ground-truth-csv "${GROUND_TRUTH}" \
  --chexpert-csv "${CHEXPERT}" \
  --split-csv "${SPLIT}" \
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
  --bootstrap-replicates 5000 \
  --bootstrap-seed 20260730

echo ""
echo "Completed: $(date --iso-8601=seconds)"
