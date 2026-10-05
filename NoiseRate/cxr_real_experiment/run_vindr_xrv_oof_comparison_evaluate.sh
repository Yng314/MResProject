#!/bin/bash
#SBATCH --job-name=vindr-xrv-oof-eval
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_xrv_oof_eval_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_xrv_oof_eval_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
SCRIPT_DIR="${ROOT}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
PROGRAM="${SCRIPT_DIR}/vindr_xrv_oof_comparison.py"
TEST_PROGRAM="${SCRIPT_DIR}/test_vindr_xrv_oof_comparison.py"
PROTOCOL="${SCRIPT_DIR}/vindr_xrv_oof_comparison_protocol_20260813.md"
PARENT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_detector_benchmark_v2/20260812_v1"
MOBILE="${PARENT}/blind_detector_scores_v2"
OUTPUT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_xrv_oof_comparison/20260813_v1"
XRV_ROOT="${OUTPUT}/xrv_oof_runs"
COMPARISON="${OUTPUT}/comparison"

EXPECTED_PROGRAM_SHA="1b3e02e01699f0d891928c3d9fc0cd3618770d4f0e5ed4323cebf119573cc976"
EXPECTED_TEST_SHA="347764d512d905ca7eeb7d2e892d98b3f8c5cff232ca63aec49d44bd7442021c"
EXPECTED_PROTOCOL_SHA="ebbdd154adc6abdafa280886335bb68aef728feb716ccad13fac6365387ae8ef"

for item in \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${TEST_PROGRAM}:${EXPECTED_TEST_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  [[ "$(sha256sum "${path}" | cut -d' ' -f1)" == "${expected}" ]] || {
    echo "Hash mismatch: ${path}" >&2
    exit 2
  }
done
for marker in "${PARENT}/.prepare_complete" "${MOBILE}/.benchmark_verified"; do
  [[ -e "${marker}" ]] || { echo "Required marker is missing: ${marker}" >&2; exit 2; }
done
[[ ! -e "${COMPARISON}" ]] || { echo "Comparison output exists: ${COMPARISON}" >&2; exit 2; }

"${PYTHON}" - "${PARENT}" "${XRV_ROOT}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

parent, xrv = Path(sys.argv[1]), Path(sys.argv[2])
manifest = pd.read_csv(parent / "blind_run_manifest.csv")
if len(manifest) != 60:
    raise RuntimeError("Formal manifest must contain 60 runs")
for row in manifest.itertuples(index=False):
    run = xrv / "scenarios" / str(row.scenario_id) / f"seed_{int(row.seed)}" / "blind_run"
    if not (run / ".blind_run_complete").is_file():
        raise RuntimeError(f"XRV OOF run is incomplete: {run}")
print("All 60 XRV OOF runs passed dependency postflight")
PY

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON}" -m unittest -v test_vindr_xrv_oof_comparison
"${PYTHON}" -u "${PROGRAM}" score \
  --parent-root "${PARENT}" \
  --mobile-score-root "${MOBILE}" \
  --xrv-oof-root "${XRV_ROOT}" \
  --output-root "${COMPARISON}"
"${PYTHON}" -u "${PROGRAM}" evaluate \
  --parent-root "${PARENT}" \
  --output-root "${COMPARISON}"
"${PYTHON}" -u "${PROGRAM}" verify \
  --parent-root "${PARENT}" \
  --output-root "${COMPARISON}" \
  --expected-runs 60
printf 'evaluation_job_id=%s\n' "${SLURM_JOB_ID:-manual}" > "${OUTPUT}/.complete"
echo "Completed: $(date --iso-8601=seconds)"
