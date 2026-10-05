#!/bin/bash
#SBATCH --job-name=entry-llm-mock
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/entry_llm_mock_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/entry_llm_mock_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

PROJECT_ROOT="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_ROOT}/NoiseRate/cxr_real_experiment"
OUTPUT_DIR="${SCRIPT_DIR}/meeting_followup_20260520/entry_llm_mock_test/slurm_${SLURM_JOB_ID}"
INPUT_CSV="${SCRIPT_DIR}/results_appa_pipeline/slurm_236304/01_baseline/train_cleanlab_entry_issues_only.csv"
REPORT_ARCHIVE="/vol/gpudata/yz3522-llmtest/MResProject/MedSoul/datasets/mimic-cxr-reports.tar.gz"

mkdir -p "${PROJECT_ROOT}/slurm_logs"
mkdir -p "${OUTPUT_DIR}"

source /vol/gpudata/yz3522-llmtest/venv/bin/activate

echo "=============================================="
echo "Entry-Level LLM Mock Smoke Test"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Time: $(date)"
echo "Output Dir: ${OUTPUT_DIR}"
echo "=============================================="

echo ""
echo "[Phase 1] Retry + error path"
python -u "${SCRIPT_DIR}/run_entry_level_llm_review.py" \
  --input-csv "${INPUT_CSV}" \
  --report-archive "${REPORT_ARCHIVE}" \
  --output-dir "${OUTPUT_DIR}" \
  --top-k 10 \
  --batch-size 10 \
  --concurrency 5 \
  --retry-limit 2 \
  --llm-mode mock \
  --mock-latency-ms 25 \
  --mock-fail-once-indexes 2,5 \
  --mock-always-fail-indexes 8

echo ""
echo "[Phase 2] Resume remaining failed rows"
python -u "${SCRIPT_DIR}/run_entry_level_llm_review.py" \
  --input-csv "${INPUT_CSV}" \
  --report-archive "${REPORT_ARCHIVE}" \
  --output-dir "${OUTPUT_DIR}" \
  --top-k 10 \
  --batch-size 10 \
  --concurrency 5 \
  --retry-limit 2 \
  --resume \
  --llm-mode mock \
  --mock-latency-ms 25

echo ""
echo "[Verification]"
export OUTPUT_DIR
python - <<'PY'
import os
from pathlib import Path
import pandas as pd

output_dir = Path(os.environ["OUTPUT_DIR"])
results_csv = output_dir / "results.csv"
errors_csv = output_dir / "errors.csv"

results = pd.read_csv(results_csv)
errors = pd.read_csv(errors_csv)

print(f"results rows: {len(results)}")
print(f"errors rows: {len(errors)}")
print("mismatch_type counts:")
print(results["mismatch_type"].value_counts(dropna=False).to_string())
print("")
print("recommended_action counts:")
print(results["recommended_action"].value_counts(dropna=False).to_string())
print("")
print("attempt_count distribution:")
print(results["attempt_count"].value_counts(dropna=False).sort_index().to_string())
print("")
print("sample result rows:")
cols = [
    "entry_key",
    "label_name",
    "raw_label",
    "pred_prob",
    "mismatch_type",
    "recommended_action",
    "attempt_count",
]
print(results[cols].head(5).to_string(index=False))
PY

echo ""
echo "Completed at $(date)"
