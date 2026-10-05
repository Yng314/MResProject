#!/bin/bash
#SBATCH --job-name=vindr-gt-smoke
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_gt_smoke_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_gt_smoke_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
PROGRAM="${ROOT}/cxr_real_experiment/vindr_known_gt_cl_benchmark.py"
PROTOCOL="${ROOT}/cxr_real_experiment/vindr_known_gt_cl_protocol_20260804.md"
HELPER="${ROOT}/cxr_real_experiment/cxr_real_noise_validation_smoke.py"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_known_gt_cl/20260804_v1"
SMOKE_ROOT="${OUTPUT_ROOT}/smoke_seed13"
EXPECTED_PROGRAM_SHA="f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83"
EXPECTED_PROTOCOL_SHA="733af6031ec02bb5f3559085d97199d87cd34a0c9e0e025efac1765811a51622"
EXPECTED_HELPER_SHA="62da5b019f9809b4641b58f3c9bda56c1f987057fc765f9a76f32feade7046e0"

for item in \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${HELPER}:${EXPECTED_HELPER_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  if [[ "$(sha256sum "${path}" | cut -d' ' -f1)" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done
for marker in \
  "${OUTPUT_ROOT}/.prepare_complete" \
  "${OUTPUT_ROOT}/features/.features_complete" \
  "${OUTPUT_ROOT}/seed_13/prepared/.prepare_complete"; do
  if [[ ! -e "${marker}" ]]; then
    echo "Required upstream marker is missing: ${marker}" >&2
    exit 2
  fi
done
if [[ -e "${SMOKE_ROOT}" ]]; then
  echo "Smoke output already exists; refusing to overwrite" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}/cxr_real_experiment:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"

echo "VinDr known-GT nested-OOF smoke"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Started: $(date --iso-8601=seconds)"

"${PYTHON}" -u "${PROGRAM}" run \
  --blind-cohort "${OUTPUT_ROOT}/seed_13/prepared/blind_noisy_cohort.csv" \
  --features "${OUTPUT_ROOT}/features/xrv_features.npz" \
  --output-dir "${SMOKE_ROOT}" \
  --seed 13 \
  --n-splits 4 \
  --epochs 3 \
  --early-stopping-patience 3 \
  --learning-rate 0.001 \
  --batch-size 128 \
  --device cpu

"${PYTHON}" - "${ROOT}" "${OUTPUT_ROOT}" "${SMOKE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(sys.argv[1]) / "cxr_real_experiment"))
from vindr_known_gt_cl_benchmark import evaluate_seed

output_root = Path(sys.argv[2])
smoke_root = Path(sys.argv[3])
evidence = pd.read_csv(smoke_root / "entry_evidence.csv")
oof = pd.read_csv(smoke_root / "oof_predictions.csv")
history = pd.read_csv(smoke_root / "training_history.csv")
if len(evidence) != 18000 or evidence[["image_id", "label_name"]].duplicated().any():
    raise RuntimeError("Smoke entry evidence postflight failed")
if len(oof) != 3000 or oof["image_id"].nunique() != 3000:
    raise RuntimeError("Smoke OOF row-count postflight failed")
probability_columns = [column for column in oof.columns if column.startswith("probability__")]
if len(probability_columns) != 6 or not np.isfinite(oof[probability_columns].to_numpy()).all():
    raise RuntimeError("Smoke OOF probabilities are incomplete")
if set(history["fold_id"]) != {0, 1, 2, 3} or int(history["epoch"].max()) != 3:
    raise RuntimeError("Smoke did not execute all four folds and three epochs")
if int(evidence["cl_issue"].sum()) <= 0:
    raise RuntimeError("Smoke produced no CL issue entries")
if not (smoke_root / ".blind_run_complete").is_file():
    raise RuntimeError("Blind-run completion marker is missing")

overall, budgets, labels, random_frame, aurocs, summary = evaluate_seed(
    13,
    output_root / "seed_13" / "prepared",
    smoke_root,
    random_iterations=100,
)
private_smoke = {
    "seed": 13,
    "status": "smoke_only_not_formal",
    "overall_rows": len(overall),
    "budget_rows": len(budgets),
    "label_rows": len(labels),
    "random_rows": len(random_frame),
    "auroc_rows": len(aurocs),
    "true_errors": summary["true_errors"],
    "raw_entry_dqs": summary["raw_entry_dqs"],
    "true_quality": summary["true_quality"],
}
(smoke_root / "smoke_private_metrics.json").write_text(json.dumps(private_smoke, indent=2))
(smoke_root / ".smoke_postflight_complete").write_text("complete\n")
print(json.dumps(private_smoke, indent=2))
PY

echo "Completed: $(date --iso-8601=seconds)"
