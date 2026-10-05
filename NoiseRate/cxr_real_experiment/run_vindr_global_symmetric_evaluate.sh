#!/bin/bash
#SBATCH --job-name=vindr-global-eval
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_global_eval_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_global_eval_%j.err
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
PROGRAM="${ROOT}/cxr_real_experiment/vindr_global_symmetric_noise.py"
PROTOCOL="${ROOT}/cxr_real_experiment/vindr_global_symmetric_noise_protocol_20260805.md"
DIRECTION_HELPER="${ROOT}/cxr_real_experiment/vindr_noise_direction_sensitivity.py"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_global_symmetric_noise_mobilenet/20260805_v1"
EVALUATION="${OUTPUT_ROOT}/evaluation_formal_v1"

EXPECTED_ENGINE_SHA="9d1585d8d37b7c8043516889b9b3d6ddda984425f910a891d889b0c2ecd9b026"
EXPECTED_PROGRAM_SHA="80316491dd61e7f9aa13d52a44bf76448968889bf84a2a33e7aa2a1b70431404"
EXPECTED_PROTOCOL_SHA="033c08d7a93eb06076f35245c7bb2fd36753abf206970cf6e6e78ab87fd4d3da"
EXPECTED_DIRECTION_HELPER_SHA="eabe258dd8ce0b860d458c451cd4064514bc1b7a8fa76679347b6c532602390e"
EXPECTED_SCENARIO_MANIFEST_SHA="cdff8ca8ec86f0a061f9c4856ed0888573c4defe68a31a1784484f7652eb0f17"
EXPECTED_BLIND_MANIFEST_SHA="201ac89c578a1227a8651d3ed942d1727dd8a543fb72c0a59adcc8c8f023745f"

for item in \
  "${ENGINE}:${EXPECTED_ENGINE_SHA}" \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${DIRECTION_HELPER}:${EXPECTED_DIRECTION_HELPER_SHA}" \
  "${OUTPUT_ROOT}/scenario_manifest.csv:${EXPECTED_SCENARIO_MANIFEST_SHA}" \
  "${OUTPUT_ROOT}/blind_run_manifest.csv:${EXPECTED_BLIND_MANIFEST_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  if [[ "$(sha256sum "${path}" | cut -d' ' -f1)" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done
if [[ ! -e "${OUTPUT_ROOT}/.prepare_complete" ]]; then
  echo "Prepare marker is missing" >&2
  exit 2
fi
if [[ -e "${EVALUATION}" ]]; then
  echo "Evaluation output already exists; refusing to overwrite" >&2
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
if len(manifest) != 24:
    raise RuntimeError("Blind manifest does not contain 24 runs")
for row in manifest.itertuples(index=False):
    run = root / row.blind_run_relpath
    if not (run / ".blind_run_complete").is_file():
        raise RuntimeError(f"Blind completion marker is missing: {run}")
    summary = json.loads((run / "blind_run_summary.json").read_text())
    if summary["program_sha256"] != expected_engine_sha or not summary["outcome_blind"]:
        raise RuntimeError(f"Blind provenance failed: {run}")
    if summary["seed"] != int(row.seed) or summary["entries"] != 18_000:
        raise RuntimeError(f"Blind summary does not match manifest: {run}")
print(json.dumps({"blind_runs": 24, "pre_private_gate": "passed"}, indent=2))
PY

echo "VinDr exact global symmetric-noise private evaluation"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Started: $(date --iso-8601=seconds)"
"${PYTHON}" -u "${PROGRAM}" evaluate \
  --experiment-root "${OUTPUT_ROOT}" \
  --output-dir "${EVALUATION}" \
  --seeds "13,42,97,123,211,307" \
  --bootstrap-iterations 1000
"${PYTHON}" -u "${PROGRAM}" verify \
  --experiment-root "${OUTPUT_ROOT}" \
  --evaluation-dir "${EVALUATION}" \
  --seeds "13,42,97,123,211,307"

"${PYTHON}" - "${OUTPUT_ROOT}" "${EVALUATION}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root, evaluation = map(Path, sys.argv[1:])
for marker in [root / ".benchmark_verified", evaluation / ".evaluation_complete"]:
    if not marker.is_file():
        raise RuntimeError(f"Completion marker is missing: {marker}")
expected_rows = {
    "overall_detection_metrics.csv": 72,
    "budget_metrics.csv": 270,
    "per_label_direction_metrics.csv": 1296,
    "oof_label_aurocs.csv": 144,
    "quality_summary.csv": 24,
    "image_cluster_bootstrap.csv": 24000,
    "corruption_prevalence.csv": 144,
    "dqs_calibration.csv": 6,
    "detection_lift_by_seed.csv": 6,
}
for name, rows in expected_rows.items():
    actual = len(pd.read_csv(evaluation / name))
    if actual != rows:
        raise RuntimeError(f"{name} has {actual} rows; expected {rows}")
summary = json.loads((evaluation / "evaluation_summary.json").read_text())
if summary["known_true_quality_anchors"] != [1.0, 0.9, 0.8, 0.7]:
    raise RuntimeError("Known quality anchors differ from protocol")
print(json.dumps({"formal_evaluation_postflight": "passed", "gates": summary["gates"]}, indent=2))
PY
echo "Completed: $(date --iso-8601=seconds)"
