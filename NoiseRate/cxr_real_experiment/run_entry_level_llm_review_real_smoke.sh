#!/bin/bash
#SBATCH --job-name=entry-llm-real
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/entry_llm_real_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/entry_llm_real_%j.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

PROJECT_ROOT="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_ROOT}/NoiseRate/cxr_real_experiment"
OUTPUT_DIR="${SCRIPT_DIR}/meeting_followup_20260520/entry_llm_real_smoke/slurm_${SLURM_JOB_ID}"
INPUT_CSV="${SCRIPT_DIR}/results_appa_pipeline/slurm_236304/01_baseline/train_cleanlab_entry_issues_only.csv"
REPORT_ARCHIVE="${PROJECT_ROOT}/MedSoul/datasets/mimic-cxr-reports.tar.gz"
TOP_K="${TOP_K_OVERRIDE:-10}"
BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-${TOP_K}}"
CONCURRENCY="${CONCURRENCY_OVERRIDE:-5}"
RETRY_LIMIT="${RETRY_LIMIT_OVERRIDE:-2}"

mkdir -p "${PROJECT_ROOT}/slurm_logs"
mkdir -p "${OUTPUT_DIR}"

source /vol/gpudata/yz3522-llmtest/venv/bin/activate
source ~/.llm_review_env
if [[ -n "${OPENAI_MODEL_OVERRIDE:-}" ]]; then
  export OPENAI_MODEL="${OPENAI_MODEL_OVERRIDE}"
fi

echo "=============================================="
echo "Entry-Level LLM Real Smoke Test"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Time: $(date)"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Model: ${OPENAI_MODEL:-unset}"
echo "=============================================="

python -u "${SCRIPT_DIR}/run_entry_level_llm_review.py" \
  --input-csv "${INPUT_CSV}" \
  --report-archive "${REPORT_ARCHIVE}" \
  --output-dir "${OUTPUT_DIR}" \
  --top-k "${TOP_K}" \
  --batch-size "${BATCH_SIZE}" \
  --concurrency "${CONCURRENCY}" \
  --retry-limit "${RETRY_LIMIT}" \
  --llm-mode real \
  --request-timeout-seconds 120 \
  --temperature 0 \
  --max-completion-tokens 400

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

results = pd.read_csv(results_csv) if results_csv.exists() else pd.DataFrame()
errors = pd.read_csv(errors_csv) if errors_csv.exists() else pd.DataFrame()

print(f"results rows: {len(results)}")
print(f"errors rows: {len(errors)}")
if not results.empty:
    print("mismatch_type counts:")
    print(results["mismatch_type"].value_counts(dropna=False).to_string())
    print("")
    print("recommended_action counts:")
    print(results["recommended_action"].value_counts(dropna=False).to_string())
    print("")
    print("sample rows:")
    cols = [
        "entry_key",
        "label_name",
        "raw_label",
        "pred_prob",
        "mismatch_type",
        "recommended_action",
    ]
    print(results[cols].head(5).to_string(index=False))
if not errors.empty:
    print("")
    print("error sample:")
    print(errors[["entry_key", "label_name", "error_message", "attempt_count"]].head(5).to_string(index=False))
PY

echo ""
echo "Completed at $(date)"
