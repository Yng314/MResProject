#!/bin/bash
#SBATCH --job-name=own-ref-3eval
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/own_ref_3eval_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/own_ref_3eval_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
NOISERATE_DIR="${PROJECT_DIR}/NoiseRate"
SCRIPT_DIR="${NOISERATE_DIR}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
BASE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_noes50_clean_3seed/20260707_123320"
REFINE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_own_top20_binary_v5_llm_refine_5seed/20260714_065539"
OUT_DIR="${REFINE_ROOT}/evaluation_interim_3seed_loop5_loop8_20260717"
SEEDS=(7 13 42)
MODE="${1:-full}"

case "${MODE}" in
  full)
    REQUIRED_LOOPS=(05 08)
    REFINE_LOOPS=(1 2 3 4 5 6 7 8)
    EXTENSION_LOOP=8
    EXTRA_ARGS=()
    ;;
  checkpoint)
    OUT_DIR="${REFINE_ROOT}/evaluation_interim_3seed_loop5_only_20260716"
    REQUIRED_LOOPS=(05)
    REFINE_LOOPS=(1 2 3 4 5)
    EXTENSION_LOOP=5
    EXTRA_ARGS=(--checkpoint-only)
    ;;
  *)
    echo "Mode must be 'full' or 'checkpoint'; received ${MODE}." >&2
    exit 2
    ;;
esac

mkdir -p "${PROJECT_DIR}/slurm_logs" "${OUT_DIR}/performance" "${OUT_DIR}/quality"

for seed in "${SEEDS[@]}"; do
  for loop in "${REQUIRED_LOOPS[@]}"; do
    loop_dir="${REFINE_ROOT}/seed_${seed}/llm_refine/loop_${loop}"
    if [[ ! -f "${loop_dir}/.train_eval_complete" ]]; then
      echo "Interim three-seed evaluation requires: ${loop_dir}/.train_eval_complete" >&2
      exit 2
    fi
    for required in \
      "${loop_dir}/train_eval/baseline_run_summary.csv" \
      "${loop_dir}/train_eval/test_study_predictions.csv"; do
      if [[ ! -s "${required}" ]]; then
        echo "Interim three-seed evaluation requires: ${required}" >&2
        exit 2
      fi
    done
  done
done

echo "Interim three-seed own-top20 evaluation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Seeds: ${SEEDS[*]}"
echo "Mode: ${MODE}"
echo "Output: ${OUT_DIR}"
echo "Started: $(date --iso-8601=seconds)"

cd "${NOISERATE_DIR}"

"${PYTHON}" "${SCRIPT_DIR}/evaluate_own_top20_refinement_endpoints_interim.py" \
  --base-root "${BASE_ROOT}" \
  --refine-root "${REFINE_ROOT}" \
  --out-dir "${OUT_DIR}/performance" \
  --seeds "${SEEDS[@]}" \
  --remove-loops 1 2 3 4 5 \
  --refine-loops "${REFINE_LOOPS[@]}" \
  --checkpoint-loop 5 \
  --extension-loop "${EXTENSION_LOOP}" \
  --remove-comparator-loop 5 \
  --bootstrap-iters 1000 \
  --bootstrap-seed 20260717 \
  --workers 3 \
  --refine-layout own \
  "${EXTRA_ARGS[@]}"

"${PYTHON}" "${SCRIPT_DIR}/evaluate_own_top20_refinement_quality_interim.py" \
  --base-root "${BASE_ROOT}" \
  --refine-root "${REFINE_ROOT}" \
  --out-dir "${OUT_DIR}/quality" \
  --seeds "${SEEDS[@]}" \
  --loops "${REFINE_LOOPS[@]}" \
  --checkpoint-loop 5 \
  --extension-loop "${EXTENSION_LOOP}" \
  --workers 3 \
  --refine-layout own \
  "${EXTRA_ARGS[@]}"

"${PYTHON}" - "${OUT_DIR}" <<'PY'
from pathlib import Path
import json
import sys

out_dir = Path(sys.argv[1])
required = [
    out_dir / "performance" / "interim_seed_paired_statistics.csv",
    out_dir / "performance" / "interim_hierarchical_bootstrap.csv",
    out_dir / "quality" / "own_top20_quality_summary.csv",
    out_dir / "quality" / "own_top20_quality_prelocked_statistics.csv",
]
missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise SystemExit(f"Missing interim outputs: {missing}")
for metadata_path in [
    out_dir / "performance" / "evaluation_metadata.json",
    out_dir / "quality" / "own_top20_quality_metadata.json",
]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "passed":
        raise SystemExit(f"Interim evaluator did not pass: {metadata_path}")
    if metadata.get("analysis_status") != "interim_descriptive_seed_subset":
        raise SystemExit(f"Interim status label missing: {metadata_path}")
    if metadata.get("seeds") != [7, 13, 42]:
        raise SystemExit(f"Unexpected interim seeds: {metadata_path}")
print("Interim three-seed evaluation outputs verified.")
PY

echo "Completed: $(date --iso-8601=seconds)"
