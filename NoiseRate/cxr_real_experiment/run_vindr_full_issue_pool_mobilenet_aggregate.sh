#!/bin/bash
#SBATCH --job-name=vfull-mnet-agg
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_fullpool_mobilenet_aggregate_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_fullpool_mobilenet_aggregate_%j.err
#SBATCH --partition=training
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
PROTOCOL="${SCRIPT_DIR}/vindr_full_issue_pool_mobilenet_protocol_20260824.md"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_full_issue_pool_iteration_mobilenet/20260824_multiseed_v1"
SEEDS="11003,13007,17011,19001,23003,27011,31013,37003"

[[ "$(sha256sum "${DRIVER}" | cut -d' ' -f1)" == "0d777cd3567a5cb20a82b5c4c4ab52a8be6ac6bba12f8a91cfddb8e0a78f3c88" ]] || exit 2
[[ "$(sha256sum "${PROTOCOL}" | cut -d' ' -f1)" == "a00e5bee0d9ac2403e193b6b2958b648014b363e9f580d2df27bd6178bcb602d" ]] || exit 2
for seed in 11003 13007 17011 19001 23003 27011 31013 37003; do
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
export MPLBACKEND=Agg
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"
"${PYTHON}" -u "${DRIVER}" aggregate \
  --experiment-root "${OUTPUT_ROOT}" \
  --aggregate-output "${OUTPUT_ROOT}/aggregate" \
  --seeds "${SEEDS}"

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys

root = Path(sys.argv[1])
summary = pd.read_csv(root / "aggregate/seed_summaries.csv")
trajectory = pd.read_csv(root / "aggregate/all_trajectories.csv")
tests = pd.read_csv(root / "aggregate/paired_tests.csv")
expected = {11003, 13007, 17011, 19001, 23003, 27011, 31013, 37003}
if set(summary["seed"].astype(int)) != expected or len(summary) != 8:
    raise RuntimeError("Eight-seed summary coverage failed")
if trajectory[["seed", "loop"]].duplicated().any():
    raise RuntimeError("Eight-seed trajectory coverage failed")
for seed in expected:
    loops = trajectory.loc[trajectory["seed"].eq(seed), "loop"].astype(int).tolist()
    if not 1 <= max(loops) <= 5 or loops != list(range(max(loops) + 1)):
        raise RuntimeError(f"Seed {seed} does not contain a contiguous natural-stop trajectory")
if len(tests) != 3 or not (tests["paired_seeds"] == 8).all():
    raise RuntimeError("Paired-test coverage failed")
required_summary = [
    "iterative_extra_quality", "post_one_shot_action_auroc_change",
    "post_one_shot_sentinel_auroc_change",
]
if not np.isfinite(summary[required_summary].to_numpy(dtype=float)).all():
    raise RuntimeError("Seed-level paired outcomes are non-finite")
test_values = tests[["mean_difference", "exact_sign_flip_p"]].to_numpy(dtype=float)
if not np.isfinite(test_values).all() or not tests["exact_sign_flip_p"].between(0.0, 1.0).all():
    raise RuntimeError("Paired-test values are invalid")
aggregate = json.loads((root / "aggregate/aggregate_summary.json").read_text())
if aggregate["runs"] != 8 or set(map(int, aggregate["seeds"])) != expected:
    raise RuntimeError("Aggregate run count failed")
if not (root / "aggregate/.aggregate_complete").is_file():
    raise RuntimeError("Aggregate marker missing")
print("Eight-seed VinDr MobileNet full-issue-pool aggregate postflight passed")
PY
printf 'job_id=%s\nseeds=%s\n' "${SLURM_JOB_ID:-manual}" "${SEEDS}" > "${OUTPUT_ROOT}/.complete"
echo "Eight-seed VinDr MobileNet full-issue-pool aggregate completed"
