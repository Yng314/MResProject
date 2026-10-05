#!/bin/bash
#SBATCH --job-name=mechanism-4s
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mechanism_4s_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mechanism_4s_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_DIR}/NoiseRate/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
REFINE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_own_top20_binary_v5_llm_refine_5seed/20260714_065539"
OUTPUT_DIR="${1:-${REFINE_ROOT}/evaluation_loop_mechanism_audit_4seed_excluding_seed7_20260723}"

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUTPUT_DIR}"

if [[ -f "${OUTPUT_DIR}/.audit_complete" ]]; then
  echo "Refusing to overwrite completed audit: ${OUTPUT_DIR}" >&2
  exit 2
fi

for seed in 13 42 97 123; do
  for loop in 01 02 03 04 05 06 07 08; do
    loop_dir="${REFINE_ROOT}/seed_${seed}/llm_refine/loop_${loop}"
    for marker in \
      .oof_complete \
      .selection_complete \
      .entry_expansion_complete \
      .llm_review_complete \
      .action_tables_complete \
      .train_eval_complete; do
      if [[ ! -e "${loop_dir}/${marker}" ]]; then
        echo "Missing transaction marker: ${loop_dir}/${marker}" >&2
        exit 2
      fi
    done
  done
done

"${PYTHON}" -m py_compile \
  "${SCRIPT_DIR}/analyze_own_top20_loop_mechanisms.py"

echo "Four-seed Loop1-8 own-top20 mechanism audit"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Included seeds: 13 42 97 123"
echo "Output: ${OUTPUT_DIR}"
echo "Started: $(date --iso-8601=seconds)"

cd "${PROJECT_DIR}/NoiseRate"

"${PYTHON}" "${SCRIPT_DIR}/analyze_own_top20_loop_mechanisms.py" \
  --refine-root "${REFINE_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --slurm-log-dir "${PROJECT_DIR}/slurm_logs" \
  --seeds 13 42 97 123 \
  --loops 1 2 3 4 5 6 7 8

"${PYTHON}" - "${OUTPUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
metadata = json.loads((output_dir / "audit_metadata.json").read_text())
checks = metadata["checks"]
if metadata["status"] != "passed" or len(checks) != 9:
    raise SystemExit("Mechanism audit metadata did not pass all nine checks")
if any(item["status"] != "passed" for item in checks):
    raise SystemExit("One or more mechanism audit checks failed")
if metadata["seeds"] != [13, 42, 97, 123]:
    raise SystemExit(f"Unexpected audit seeds: {metadata['seeds']}")
(output_dir / ".audit_complete").touch()
PY

echo "Completed: $(date --iso-8601=seconds)"
