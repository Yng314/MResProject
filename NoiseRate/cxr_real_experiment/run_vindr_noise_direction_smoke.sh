#!/bin/bash
#SBATCH --job-name=vindr-dir-smoke
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_dir_smoke_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_dir_smoke_%j.err
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
PROGRAM="${ROOT}/cxr_real_experiment/vindr_noise_direction_sensitivity.py"
PROTOCOL="${ROOT}/cxr_real_experiment/vindr_noise_direction_sensitivity_protocol_20260804.md"
OOF_ENGINE="${ROOT}/cxr_real_experiment/vindr_known_gt_cl_benchmark.py"
HELPER="${ROOT}/cxr_real_experiment/cxr_real_noise_validation_smoke.py"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_noise_direction_sensitivity/20260804_v2"
PARENT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_known_gt_cl/20260804_v1"
PREPARED="${OUTPUT_ROOT}/scenarios/fn_only_r30/seed_211/prepared"
SMOKE_ROOT="${OUTPUT_ROOT}/smoke_fn_only_r30_seed211"
EXPECTED_PROGRAM_SHA="eabe258dd8ce0b860d458c451cd4064514bc1b7a8fa76679347b6c532602390e"
EXPECTED_PROTOCOL_SHA="2baeffb1bd82663332a75234e9c67013bf5ae6b0779c075a3618a7c379ce3ad0"
EXPECTED_OOF_SHA="f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83"
EXPECTED_HELPER_SHA="62da5b019f9809b4641b58f3c9bda56c1f987057fc765f9a76f32feade7046e0"
EXPECTED_FEATURE_SHA="d361d521e681ea8fdff94ba83c206680123032eb247a5b84ffd77d393a895327"

for item in \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${OOF_ENGINE}:${EXPECTED_OOF_SHA}" \
  "${HELPER}:${EXPECTED_HELPER_SHA}" \
  "${PARENT_ROOT}/features/xrv_features.npz:${EXPECTED_FEATURE_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  if [[ "$(sha256sum "${path}" | cut -d' ' -f1)" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done
for marker in "${OUTPUT_ROOT}/.prepare_complete" "${PREPARED}/.prepare_complete"; do
  if [[ ! -e "${marker}" ]]; then
    echo "Required prepare marker is missing: ${marker}" >&2
    exit 2
  fi
done
if [[ -e "${SMOKE_ROOT}" ]]; then
  echo "Smoke output already exists; refusing to overwrite" >&2
  exit 2
fi

export PYTHONPATH="${ROOT}/cxr_real_experiment:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"

"${PYTHON}" - "${OUTPUT_ROOT}/blind_run_manifest.csv" "${PREPARED}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

manifest_path = Path(sys.argv[1])
prepared = Path(sys.argv[2])
manifest = pd.read_csv(manifest_path)
row = manifest[(manifest["scenario_id"] == "fn_only_r30") & (manifest["seed"] == 211)]
if len(row) != 1:
    raise RuntimeError("Smoke target is not unique in blind manifest")
expected = manifest_path.parent / row.iloc[0]["prepared_relpath"]
if expected.resolve() != prepared.resolve():
    raise RuntimeError("Smoke prepared path does not match blind manifest")
PY

echo "VinDr direction-sensitivity high-risk smoke"
echo "Job ID: ${SLURM_JOB_ID:-manual}; scenario=fn_only_r30; seed=211"
echo "Started: $(date --iso-8601=seconds)"

"${PYTHON}" -u "${OOF_ENGINE}" run \
  --blind-cohort "${PREPARED}/blind_noisy_cohort.csv" \
  --features "${PARENT_ROOT}/features/xrv_features.npz" \
  --output-dir "${SMOKE_ROOT}" \
  --seed 211 \
  --n-splits 4 \
  --epochs 3 \
  --early-stopping-patience 3 \
  --learning-rate 0.001 \
  --batch-size 128 \
  --device cpu

"${PYTHON}" - "${ROOT}" "${PREPARED}" "${SMOKE_ROOT}" <<'PY'
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]) / "cxr_real_experiment"))
from vindr_noise_direction_sensitivity import bootstrap_scenario, merge_private, scenario_metrics

prepared = Path(sys.argv[2])
smoke = Path(sys.argv[3])
merged = merge_private(prepared, smoke)
metadata = {
    "scenario_id": "fn_only_r30",
    "regime": "fn_only",
    "noise_rate": 0.30,
    "rate_percent": 30,
    "seed": 211,
    "seed_block": "new_replication",
}
overall, budgets, labels, aurocs, quality = scenario_metrics(merged, metadata)
bootstrap = bootstrap_scenario(merged, metadata, 10)
if len(overall) != 3 or len(budgets) != 15 or len(labels) != 54 or len(aurocs) != 6:
    raise RuntimeError("Smoke metric row counts are incomplete")
if quality["true_errors"] != 560 or not math.isnan(quality["hard_recall_0_to_1"]):
    raise RuntimeError("Smoke private direction scope is wrong")
if not math.isfinite(quality["hard_recall_1_to_0"]):
    raise RuntimeError("Smoke false-negative recall is not finite")
if len(bootstrap) != 10 or bootstrap["hard_recall_1_to_0"].isna().any():
    raise RuntimeError("Smoke bootstrap path failed")
payload = {
    "status": "smoke_only_not_formal",
    "scenario_id": "fn_only_r30",
    "seed": 211,
    "true_errors": quality["true_errors"],
    "cl_issue_entries": quality["cl_issue_entries"],
    "hard_recall_1_to_0": quality["hard_recall_1_to_0"],
}
(smoke / "smoke_private_metrics.json").write_text(json.dumps(payload, indent=2))
(smoke / ".smoke_postflight_complete").write_text("complete\n")
print(json.dumps(payload, indent=2))
PY

echo "Completed: $(date --iso-8601=seconds)"
