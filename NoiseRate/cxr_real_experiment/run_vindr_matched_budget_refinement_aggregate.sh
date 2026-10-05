#!/bin/bash
#SBATCH --job-name=vindr-match-agg
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_matched_budget_aggregate_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_matched_budget_aggregate_%j.err
#SBATCH --partition=training
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

SCRIPT_DIR="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
DRIVER="${SCRIPT_DIR}/vindr_matched_budget_refinement.py"
PROTOCOL="${SCRIPT_DIR}/vindr_matched_budget_refinement_protocol_20260824.md"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_matched_budget_refinement_mobilenet/20260824_v1"
SEEDS="11003,13007,17011,19001,23003,27011,31013,37003"

[[ "$(sha256sum "${DRIVER}" | cut -d' ' -f1)" == "82aef55e4e099f280590d0d6e72eb9346bfd376eb8aac4892364d8139c7a0367" ]] || exit 2
[[ "$(sha256sum "${PROTOCOL}" | cut -d' ' -f1)" == "d29179f28fc44ac49503c763171ddbd4081771df0b0c58562b263357fe5fd9a5" ]] || exit 2
for seed in 11003 13007 17011 19001 23003 27011 31013 37003; do
  [[ -e "${OUTPUT_ROOT}/seed_${seed}/.worker_complete" ]] || {
    echo "Incomplete matched-budget seed: ${seed}" >&2
    exit 2
  }
done
[[ ! -e "${OUTPUT_ROOT}/aggregate" && ! -e "${OUTPUT_ROOT}/.complete" ]] || {
  echo "Refusing to overwrite completed matched-budget aggregate" >&2
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
evidence = pd.read_csv(root / "aggregate/all_dynamic_evidence_trajectories.csv")
discovery = pd.read_csv(root / "aggregate/all_matched_discovery_trajectories.csv")
tests = pd.read_csv(root / "aggregate/paired_tests.csv")
expected = {11003, 13007, 17011, 19001, 23003, 27011, 31013, 37003}
if len(summary) != 8 or set(summary["seed"].astype(int)) != expected:
    raise RuntimeError("Eight-seed summary coverage failed")
if len(evidence) != 48 or evidence[["seed", "loop"]].duplicated().any():
    raise RuntimeError("Dynamic evidence trajectory coverage failed")
if len(discovery) != 144 or discovery[["seed", "method", "loop"]].duplicated().any():
    raise RuntimeError("Matched discovery trajectory coverage failed")
if len(tests) != 4 or not tests["paired_seeds"].eq(8).all():
    raise RuntimeError("Paired-test coverage failed")
if not np.isfinite(tests[["mean_difference", "exact_sign_flip_p"]].to_numpy(dtype=float)).all():
    raise RuntimeError("Paired-test values are non-finite")
if not tests["exact_sign_flip_p"].between(0.0, 1.0).all():
    raise RuntimeError("Paired-test p-values are invalid")
if not summary["frozen_endpoint_matches_one_shot_keys"].astype(bool).all():
    raise RuntimeError("Frozen/one-shot endpoint equality failed")
aggregate = json.loads((root / "aggregate/aggregate_summary.json").read_text())
if aggregate["runs"] != 8 or set(map(int, aggregate["seeds"])) != expected:
    raise RuntimeError("Aggregate provenance failed")
if not (root / "aggregate/.aggregate_complete").is_file():
    raise RuntimeError("Aggregate marker missing")
print("Eight-seed VinDr matched-budget aggregate postflight passed")
PY
printf 'job_id=%s\nseeds=%s\n' "${SLURM_JOB_ID:-manual}" "${SEEDS}" > "${OUTPUT_ROOT}/.complete"
echo "Eight-seed VinDr matched-budget aggregate completed"
