#!/bin/bash
#SBATCH --job-name=sample05-entry-llm
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/sample05_entry_llm_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/sample05_entry_llm_%j.err
#SBATCH --partition=a16
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G

set -euo pipefail

PROJECT_ROOT="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_ROOT}/NoiseRate/cxr_real_experiment"
PIPELINE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_xrv_k4_pipeline_formal/20260615_131836"
SAMPLE_ISSUES_CSV="${PIPELINE_ROOT}/01_oof_job_250159/train_cleanlab_sample_issues_only.csv"
REPORT_ARCHIVE="${PROJECT_ROOT}/MedSoul/datasets/mimic-cxr-reports.tar.gz"
TOP_FRACTION="${TOP_FRACTION_OVERRIDE:-0.05}"
FRACTION_TAG="${FRACTION_TAG_OVERRIDE:-sample_top05pct}"
OUTPUT_PARENT="${OUTPUT_PARENT_OVERRIDE:-/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/${FRACTION_TAG}_entry_llm_real}"
OUTPUT_DIR="${OUTPUT_PARENT}/slurm_${SLURM_JOB_ID}"
EXPANDED_CSV="${OUTPUT_DIR}/${FRACTION_TAG}_expanded_entries.csv"
LLM_OUTPUT_DIR="${OUTPUT_DIR}/llm_review"
BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-100}"
CONCURRENCY="${CONCURRENCY_OVERRIDE:-10}"
RETRY_LIMIT="${RETRY_LIMIT_OVERRIDE:-2}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT_OVERRIDE:-180}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS_OVERRIDE:-400}"

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${OUTPUT_DIR}" "${LLM_OUTPUT_DIR}"

source /vol/gpudata/yz3522-llmtest/venv/bin/activate
source ~/.llm_review_env
if [[ -n "${OPENAI_MODEL_OVERRIDE:-}" ]]; then
  export OPENAI_MODEL="${OPENAI_MODEL_OVERRIDE}"
fi

TOP_N_SAMPLES="$(python - <<PY
import pandas as pd
df = pd.read_csv(r'''${SAMPLE_ISSUES_CSV}''')
print(max(1, int(round(len(df) * float(r'''${TOP_FRACTION}''')))))
PY
)"

echo "=============================================="
echo "${FRACTION_TAG} -> Entry LLM Real"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Time: $(date)"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Model: ${OPENAI_MODEL:-unset}"
echo "Top Fraction: ${TOP_FRACTION}"
echo "Top N Samples: ${TOP_N_SAMPLES}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Concurrency: ${CONCURRENCY}"
echo "=============================================="

python -u "${SCRIPT_DIR}/export_sample_topk_entries_for_llm.py" \
  --sample-issues-csv "${SAMPLE_ISSUES_CSV}" \
  --output-csv "${EXPANDED_CSV}" \
  --top-n-samples "${TOP_N_SAMPLES}"

python -u "${SCRIPT_DIR}/run_entry_level_llm_review.py" \
  --input-csv "${EXPANDED_CSV}" \
  --report-archive "${REPORT_ARCHIVE}" \
  --output-dir "${LLM_OUTPUT_DIR}" \
  --top-k 100000000 \
  --batch-size "${BATCH_SIZE}" \
  --concurrency "${CONCURRENCY}" \
  --retry-limit "${RETRY_LIMIT}" \
  --llm-mode real \
  --request-timeout-seconds "${REQUEST_TIMEOUT}" \
  --temperature 0 \
  --max-completion-tokens "${MAX_COMPLETION_TOKENS}"

export OUTPUT_DIR EXPANDED_CSV LLM_OUTPUT_DIR
python - <<'PY'
import os
from pathlib import Path
import pandas as pd

output_dir = Path(os.environ["OUTPUT_DIR"])
expanded_csv = Path(os.environ["EXPANDED_CSV"])
llm_output_dir = Path(os.environ["LLM_OUTPUT_DIR"])

expanded = pd.read_csv(expanded_csv)
results = pd.read_csv(llm_output_dir / "results.csv") if (llm_output_dir / "results.csv").exists() else pd.DataFrame()
errors_path = llm_output_dir / "errors.csv"
errors = pd.read_csv(errors_path) if errors_path.exists() and errors_path.stat().st_size > 0 else pd.DataFrame()

sample_summary = pd.DataFrame()
if not results.empty:
    sample_summary = (
        results.sort_values(["sample_selected_rank", "entry_rank_within_sample", "label_index"])
        .groupby(
            [
                "sample_selected_rank",
                "pool_row_id",
                "study_id",
                "sample_est_issue_entry_count",
                "sample_flagged_label_names",
            ],
            as_index=False,
        )
        .agg(
            review_entry_count=("entry_key", "count"),
            reviewed_labels=("label_name", lambda s: "|".join(s.astype(str).tolist())),
            mismatch_types=("mismatch_type", lambda s: "|".join(s.astype(str).tolist())),
            recommended_actions=("recommended_action", lambda s: "|".join(s.astype(str).tolist())),
        )
    )
sample_summary.to_csv(output_dir / "sample_to_entry_review_summary.csv", index=False)

print(f"expanded entry rows: {len(expanded)}")
print(f"review result rows: {len(results)}")
print(f"review error rows: {len(errors)}")
if not results.empty:
    print("")
    print("mismatch_type counts:")
    print(results["mismatch_type"].value_counts(dropna=False).to_string())
    print("")
    print("recommended_action counts:")
    print(results["recommended_action"].value_counts(dropna=False).to_string())
PY

echo ""
echo "Completed at $(date)"
