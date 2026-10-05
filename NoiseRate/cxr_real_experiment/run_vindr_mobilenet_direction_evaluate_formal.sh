#!/bin/bash
#SBATCH --job-name=vindr-mnet-eval
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_mnet_eval_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_mnet_eval_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
ENGINE="${ROOT}/cxr_real_experiment/vindr_mobilenet_oof.py"
PROTOCOL="${ROOT}/cxr_real_experiment/vindr_mobilenet_direction_sensitivity_protocol_20260805.md"
EVALUATOR="${ROOT}/cxr_real_experiment/vindr_noise_direction_sensitivity.py"
COMPARATOR="${ROOT}/cxr_real_experiment/compare_vindr_mobilenet_frozen_xrv.py"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_noise_direction_sensitivity_mobilenet/20260805_v1"
EVALUATION="${OUTPUT_ROOT}/evaluation_formal_v1"
COMPARISON="${OUTPUT_ROOT}/paired_comparison_formal_v1"
FROZEN_EVALUATION="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_noise_direction_sensitivity/20260804_v2/evaluation_formal_v1"

EXPECTED_ENGINE_SHA="9d1585d8d37b7c8043516889b9b3d6ddda984425f910a891d889b0c2ecd9b026"
EXPECTED_PROTOCOL_SHA="edb164ded4e17bc387a7822a121d22c62ddbb59178ec806dde22f84849189ffe"
EXPECTED_EVALUATOR_SHA="eabe258dd8ce0b860d458c451cd4064514bc1b7a8fa76679347b6c532602390e"
EXPECTED_COMPARATOR_SHA="79c3bdd00788ab0306b83ea610a522691aac47e97a759e76c2e5024e27f235b0"
EXPECTED_SCENARIO_MANIFEST_SHA="a5f3fe54d623e2ff3201ed444be8fc8f4ec9dd01be1b0a51f3abb0bb80f1c721"
EXPECTED_BLIND_MANIFEST_SHA="a17fba9c02c8a8c9428c1cffa2afee3261980e3c59a4905a68be1bbb8d0da6c4"

for item in \
  "${ENGINE}:${EXPECTED_ENGINE_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${EVALUATOR}:${EXPECTED_EVALUATOR_SHA}" \
  "${COMPARATOR}:${EXPECTED_COMPARATOR_SHA}" \
  "${OUTPUT_ROOT}/scenario_manifest.csv:${EXPECTED_SCENARIO_MANIFEST_SHA}" \
  "${OUTPUT_ROOT}/blind_run_manifest.csv:${EXPECTED_BLIND_MANIFEST_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  if [[ "$(sha256sum "${path}" | cut -d' ' -f1)" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done
for marker in \
  "${OUTPUT_ROOT}/.prepare_complete" \
  "${OUTPUT_ROOT}/smoke_fn_only_r30_seed211_v2/.smoke_postflight_complete" \
  "${FROZEN_EVALUATION}/.evaluation_complete"; do
  if [[ ! -e "${marker}" ]]; then
    echo "Required marker is missing: ${marker}" >&2
    exit 2
  fi
done
if [[ -e "${EVALUATION}" || -e "${COMPARISON}" ]]; then
  echo "Formal evaluation or paired-comparison output already exists" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}/cxr_real_experiment:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON}" - "${OUTPUT_ROOT}" "${EXPECTED_ENGINE_SHA}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
expected_engine_sha = sys.argv[2]
manifest = pd.read_csv(root / "blind_run_manifest.csv")
if len(manifest) != 60:
    raise RuntimeError("Blind manifest does not contain 60 runs")
for row in manifest.itertuples(index=False):
    run = root / row.blind_run_relpath
    if not (run / ".blind_run_complete").is_file():
        raise RuntimeError(f"Blind completion marker is missing: {run}")
    summary = json.loads((run / "blind_run_summary.json").read_text())
    if summary["program_sha256"] != expected_engine_sha or not summary["outcome_blind"]:
        raise RuntimeError(f"Blind provenance failed: {run}")
    if summary["seed"] != int(row.seed) or summary["entries"] != 18_000:
        raise RuntimeError(f"Blind summary does not match manifest: {run}")
print(json.dumps({"blind_runs": 60, "pre_private_gate": "passed"}, indent=2))
PY

echo "VinDr MobileNet formal private evaluation"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Started: $(date --iso-8601=seconds)"

"${PYTHON}" -u "${EVALUATOR}" evaluate \
  --experiment-root "${OUTPUT_ROOT}" \
  --output-dir "${EVALUATION}" \
  --seeds 13,42,97,123,211,307 \
  --bootstrap-iterations 1000

"${PYTHON}" -u "${EVALUATOR}" verify \
  --experiment-root "${OUTPUT_ROOT}" \
  --evaluation-dir "${EVALUATION}" \
  --seeds 13,42,97,123,211,307

"${PYTHON}" -u "${COMPARATOR}" \
  --mobilenet-evaluation "${EVALUATION}" \
  --frozen-evaluation "${FROZEN_EVALUATION}" \
  --output-dir "${COMPARISON}"

"${PYTHON}" - "${OUTPUT_ROOT}" "${EVALUATION}" "${COMPARISON}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root, evaluation, comparison = map(Path, sys.argv[1:])
for marker in [
    root / ".benchmark_verified",
    evaluation / ".evaluation_complete",
    comparison / ".paired_comparison_complete",
]:
    if not marker.is_file():
        raise RuntimeError(f"Formal completion marker is missing: {marker}")
expected_rows = {
    "overall_detection_metrics.csv": 180,
    "budget_metrics.csv": 810,
    "per_label_direction_metrics.csv": 3240,
    "oof_label_aurocs.csv": 360,
    "quality_summary.csv": 60,
    "image_cluster_bootstrap.csv": 60000,
    "dqs_calibration.csv": 18,
    "direction_gap_by_seed.csv": 6,
}
for name, rows in expected_rows.items():
    actual = len(pd.read_csv(evaluation / name))
    if actual != rows:
        raise RuntimeError(f"{name} has {actual} rows; expected {rows}")
for name, rows in {
    "paired_scenario_quality.csv": 60,
    "paired_label_auroc.csv": 360,
    "primary_fn_recall_by_seed.csv": 6,
    "clean_reference_auroc_by_seed.csv": 6,
}.items():
    actual = len(pd.read_csv(comparison / name))
    if actual != rows:
        raise RuntimeError(f"{name} has {actual} rows; expected {rows}")
summary = json.loads((comparison / "paired_comparison_summary.json").read_text())
if summary["seeds"] != [13, 42, 97, 123, 211, 307]:
    raise RuntimeError("Paired comparison seed order differs from the protocol")
print(json.dumps({
    "formal_evaluation_postflight": "passed",
    "model_limitation_explanation_supported": summary["primary"]["model_limitation_explanation_supported"],
}, indent=2))
PY

echo "Completed: $(date --iso-8601=seconds)"
