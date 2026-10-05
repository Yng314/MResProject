#!/bin/bash
#SBATCH --job-name=vfull-agg
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_fullpool_aggregate_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_fullpool_aggregate_%j.err
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
SCRIPT_DIR="${ROOT}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
DRIVER="${SCRIPT_DIR}/vindr_full_issue_pool_iteration.py"
PROTOCOL="${SCRIPT_DIR}/vindr_full_issue_pool_multiseed_protocol_20260819.md"
PILOT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_full_issue_pool_iteration/20260819_v1"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_full_issue_pool_iteration/20260819_multiseed_v1"
SEEDS="11003,13007,17011,19001,23003,27011,31013,37003"

EXPECTED_DRIVER_SHA="d55f354fd58791e249e74f7ea985b6ef9b415f79fe65289bba1f78ab01112386"
EXPECTED_PROTOCOL_SHA="9958b597297ab92102246d046aa9c0369b9646f9e963035ceb0ea2208c083e02"
[[ "$(sha256sum "${DRIVER}" | cut -d' ' -f1)" == "${EXPECTED_DRIVER_SHA}" ]] || exit 2
[[ "$(sha256sum "${PROTOCOL}" | cut -d' ' -f1)" == "${EXPECTED_PROTOCOL_SHA}" ]] || exit 2
[[ -e "${PILOT_ROOT}/.complete" && -e "${PILOT_ROOT}/seed_11003/.seed_evaluation_complete" ]] || {
  echo "Completed seed-11003 pilot is unavailable" >&2
  exit 2
}
mkdir -p "${OUTPUT_ROOT}"
if [[ ! -e "${OUTPUT_ROOT}/seed_11003" ]]; then
  ln -s "${PILOT_ROOT}/seed_11003" "${OUTPUT_ROOT}/seed_11003"
fi
[[ "$(readlink -f "${OUTPUT_ROOT}/seed_11003")" == "$(readlink -f "${PILOT_ROOT}/seed_11003")" ]] || {
  echo "Seed-11003 link points to the wrong pilot" >&2
  exit 2
}
for seed in 13007 17011 19001 23003 27011 31013 37003; do
  [[ -e "${OUTPUT_ROOT}/seed_${seed}/.seed_evaluation_complete" ]] || {
    echo "Incomplete seed evaluation: ${seed}" >&2
    exit 2
  }
  [[ -e "${OUTPUT_ROOT}/seed_${seed}/.worker_complete" ]] || {
    echo "Missing worker postflight: ${seed}" >&2
    exit 2
  }
done
[[ ! -e "${OUTPUT_ROOT}/aggregate" && ! -e "${OUTPUT_ROOT}/.complete" ]] || {
  echo "Refusing to overwrite completed aggregate" >&2
  exit 2
}

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"
"${PYTHON}" -u "${DRIVER}" aggregate \
  --experiment-root "${OUTPUT_ROOT}" \
  --aggregate-output "${OUTPUT_ROOT}/aggregate" \
  --seeds "${SEEDS}"

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
from pathlib import Path
import json
import pandas as pd
import sys

root = Path(sys.argv[1])
summary = pd.read_csv(root / "aggregate/seed_summaries.csv")
trajectory = pd.read_csv(root / "aggregate/all_trajectories.csv")
tests = pd.read_csv(root / "aggregate/paired_tests.csv")
expected = {11003, 13007, 17011, 19001, 23003, 27011, 31013, 37003}
if set(summary["seed"].astype(int)) != expected or len(summary) != 8:
    raise RuntimeError("Eight-seed summary coverage failed")
if len(trajectory) != 48 or set(trajectory["loop"].astype(int)) != set(range(6)):
    raise RuntimeError("Eight-seed trajectory coverage failed")
if len(tests) != 3 or not (tests["paired_seeds"] == 8).all():
    raise RuntimeError("Paired-test coverage failed")
aggregate = json.loads((root / "aggregate/aggregate_summary.json").read_text())
if aggregate["runs"] != 8:
    raise RuntimeError("Aggregate run count failed")
if not (root / "aggregate/.aggregate_complete").is_file():
    raise RuntimeError("Aggregate marker missing")
print("Eight-seed full issue-pool aggregate postflight passed")
PY
printf 'job_id=%s\nseeds=%s\n' "${SLURM_JOB_ID:-manual}" "${SEEDS}" > "${OUTPUT_ROOT}/.complete"
echo "Eight-seed full issue-pool extension completed"
