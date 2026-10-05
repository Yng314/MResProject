#!/bin/bash
#SBATCH --job-name=mbv3-dq-fix
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mbv3_dq_fix_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mbv3_dq_fix_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
NOISERATE_DIR="${PROJECT_DIR}/NoiseRate"
SCRIPT_DIR="${NOISERATE_DIR}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_noes50_clean_3seed/20260707_123320"
REFINE_SOURCE="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_xrv_iterative_sample20_dual_loop/20260624_150533/llm_refine"
OUT_DIR="${ROOT}/evaluation_20260713_data_quality_sample_denominator_corrected"
FIGURE_DIR="${SCRIPT_DIR}/evaluation_followup_20260713/real_data_quality_figures"

mkdir -p "${OUT_DIR}" "${FIGURE_DIR}" "${PROJECT_DIR}/slurm_logs"

echo "Corrected evaluable-sample denominator analysis"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Output: ${OUT_DIR}"
echo "Started: $(date --iso-8601=seconds)"

cd "${NOISERATE_DIR}"

"${PYTHON}" "${SCRIPT_DIR}/evaluate_mobilenet_data_quality_5seed.py" \
  --root "${ROOT}" \
  --refine-source "${REFINE_SOURCE}" \
  --out-dir "${OUT_DIR}" \
  --seeds 7 13 42 97 123 \
  --loops 1 2 3 4 5 \
  --workers 5

"${PYTHON}" "${SCRIPT_DIR}/build_evaluation_followup_quality_figures.py" \
  --input "${OUT_DIR}/oof_quality_metrics_per_seed.csv" \
  --output-dir "${FIGURE_DIR}"

"${PYTHON}" "${SCRIPT_DIR}/build_evaluation_followup_deck_20260713.py"

"${PYTHON}" - "${OUT_DIR}" "${FIGURE_DIR}" <<'PY'
from pathlib import Path
import json
import sys

out_dir = Path(sys.argv[1])
figure_dir = Path(sys.argv[2])
required = [
    out_dir / "oof_quality_metrics_per_seed.csv",
    figure_dir / "sample_issue_free_rate_five_seed.png",
    figure_dir / "quality_figure_metadata.json",
]
missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise SystemExit(f"Missing corrected sample-quality outputs: {missing}")
metadata = json.loads((figure_dir / "quality_figure_metadata.json").read_text(encoding="utf-8"))
definition = metadata["definitions"]["sample_coverage"]
if "zero-valid samples are not counted as issue-free" not in definition:
    raise SystemExit("Corrected sample denominator definition is missing")
print("Corrected sample-quality and deck checks passed.")
PY

echo "Completed: $(date --iso-8601=seconds)"
