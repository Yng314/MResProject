#!/bin/bash
#SBATCH --job-name=vindr-det-v2-eval
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_det_v2_eval_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_det_v2_eval_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
SCRIPT_DIR="${ROOT}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
PROGRAM="${SCRIPT_DIR}/vindr_detector_benchmark_v2.py"
TEST_PROGRAM="${SCRIPT_DIR}/test_vindr_detector_benchmark_v2.py"
PROTOCOL="${SCRIPT_DIR}/vindr_detector_benchmark_v2_protocol_20260812.md"
PARENT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_detector_benchmark_v2/20260812_v1"
SMOKE="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_detector_benchmark_v2_smoke/20260812_v1"
EXTERNAL="${SMOKE}/external_evidence/xrv_external_evidence.npz"
OUTPUT="${PARENT}/blind_detector_scores_v2"

EXPECTED_PROGRAM_SHA="b77e5f3db073782cc321a71eed91e2da2d5177ab5cd641b169535ea711b5b870"
EXPECTED_TEST_SHA="bcd7562fd121c9a64f6bccef9331c4e3d9fd3f272a488ada0219f9908a3adf67"
EXPECTED_PROTOCOL_SHA="c4524cb159d5f8ff7a6f18a5bdf444aa15f77407a0b8731077dc21cd661c9de7"

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
for marker in \
  "${PARENT}/.prepare_complete" \
  "${SMOKE}/.smoke_complete" \
  "${SMOKE}/external_evidence/.external_evidence_complete"; do
  [[ -e "${marker}" ]] || { echo "Required marker is missing: ${marker}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "Formal score output already exists: ${OUTPUT}" >&2; exit 2; }

"${PYTHON}" - "${PARENT}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
manifest = pd.read_csv(root / "blind_run_manifest.csv")
if len(manifest) != 60:
    raise RuntimeError("Formal blind manifest does not contain 60 runs")
for row in manifest.itertuples(index=False):
    run = root / row.blind_run_relpath
    if not (run / ".blind_run_complete").is_file():
        raise RuntimeError(f"Formal OOF run is incomplete: {run}")
    summary = json.loads((run / "blind_run_summary.json").read_text())
    if not summary["outcome_blind"]:
        raise RuntimeError(f"Formal OOF run is not outcome blind: {run}")
print("All 60 formal OOF runs passed dependency postflight")
PY

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON}" -m unittest -v test_vindr_detector_benchmark_v2
"${PYTHON}" -u "${PROGRAM}" score \
  --parent-root "${PARENT}" \
  --output-root "${OUTPUT}" \
  --external-evidence "${EXTERNAL}"
"${PYTHON}" -u "${PROGRAM}" evaluate \
  --parent-root "${PARENT}" \
  --output-root "${OUTPUT}"
"${PYTHON}" -u "${PROGRAM}" verify \
  --parent-root "${PARENT}" \
  --output-root "${OUTPUT}" \
  --expected-runs 60
echo "Completed: $(date --iso-8601=seconds)"
