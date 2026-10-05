#!/bin/bash
#SBATCH --job-name=sample100-entry-llm
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/sample100_entry_llm_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/sample100_entry_llm_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

PROJECT_ROOT="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_ROOT}/NoiseRate/cxr_real_experiment"
PIPELINE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_xrv_k4_pipeline_formal/20260615_131836"
SAMPLE_ISSUES_CSV="${PIPELINE_ROOT}/01_oof_job_250159/train_cleanlab_sample_issues_only.csv"
REPORT_ARCHIVE="${PROJECT_ROOT}/MedSoul/datasets/mimic-cxr-reports.tar.gz"
OUTPUT_DIR="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/sample_top100_entry_llm_mock_smoke/slurm_${SLURM_JOB_ID}"
EXPANDED_CSV="${OUTPUT_DIR}/sample_top100_expanded_entries.csv"
LLM_OUTPUT_DIR="${OUTPUT_DIR}/llm_review"
TOP_N_SAMPLES="${TOP_N_SAMPLES_OVERRIDE:-100}"

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${OUTPUT_DIR}" "${LLM_OUTPUT_DIR}"

source /vol/gpudata/yz3522-llmtest/venv/bin/activate

echo "=============================================="
echo "Sample Top100 -> Entry LLM Mock Smoke"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Time: $(date)"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Top N Samples: ${TOP_N_SAMPLES}"
echo "=============================================="

python -u "${SCRIPT_DIR}/export_sample_topk_entries_for_llm.py" \
  --sample-issues-csv "${SAMPLE_ISSUES_CSV}" \
  --output-csv "${EXPANDED_CSV}" \
  --top-n-samples "${TOP_N_SAMPLES}"

python -u "${SCRIPT_DIR}/run_entry_level_llm_review.py" \
  --input-csv "${EXPANDED_CSV}" \
  --report-archive "${REPORT_ARCHIVE}" \
  --output-dir "${LLM_OUTPUT_DIR}" \
  --top-k 100000 \
  --batch-size 25 \
  --concurrency 5 \
  --retry-limit 2 \
  --llm-mode mock \
  --mock-latency-ms 10

export OUTPUT_DIR EXPANDED_CSV LLM_OUTPUT_DIR
python - <<'PY'
import os
from pathlib import Path
import pandas as pd

output_dir = Path(os.environ["OUTPUT_DIR"])
expanded_csv = Path(os.environ["EXPANDED_CSV"])
llm_output_dir = Path(os.environ["LLM_OUTPUT_DIR"])

expanded = pd.read_csv(expanded_csv)
results = pd.read_csv(llm_output_dir / "results.csv")
errors_path = llm_output_dir / "errors.csv"
errors = pd.read_csv(errors_path) if errors_path.exists() and errors_path.stat().st_size > 0 else pd.DataFrame()

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

preview_cols = [
    "sample_selected_rank",
    "pool_row_id",
    "study_id",
    "label_name",
    "raw_label",
    "pred_prob",
    "mismatch_type",
    "recommended_action",
]

print(f"expanded entry rows: {len(expanded)}")
print(f"review result rows: {len(results)}")
print(f"review error rows: {len(errors)}")
print("")
print("mismatch_type counts:")
print(results["mismatch_type"].value_counts(dropna=False).to_string())
print("")
print("recommended_action counts:")
print(results["recommended_action"].value_counts(dropna=False).to_string())
print("")
print("entry-level preview:")
print(results[preview_cols].head(12).to_string(index=False))
print("")
print("sample-level grouped preview:")
print(sample_summary.head(8).to_string(index=False))
PY

echo ""
echo "Completed at $(date)"
