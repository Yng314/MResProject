#!/bin/bash
#SBATCH --job-name=vindr-match
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_matched_budget_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_matched_budget_%j.err
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 SEED" >&2
  exit 2
fi
SEED="$1"
case "${SEED}" in
  11003|13007|17011|19001|23003|27011|31013|37003) ;;
  *) echo "Seed is outside the locked eight-seed design: ${SEED}" >&2; exit 2 ;;
esac

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
SCRIPT_DIR="${ROOT}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
DRIVER="${SCRIPT_DIR}/vindr_matched_budget_refinement.py"
ENGINE="${SCRIPT_DIR}/vindr_mobilenet_sentinel_oof.py"
RANK_HELPER="${SCRIPT_DIR}/vindr_iterative_evidence_improvement.py"
STRESS_HELPER="${SCRIPT_DIR}/vindr_self_optimization_stress.py"
PROTOCOL="${SCRIPT_DIR}/vindr_matched_budget_refinement_protocol_20260824.md"
PREP_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_self_optimization_stress_prepared/20260814_v1/scenarios/hard_r30"
PREPARED="${PREP_ROOT}/seed_${SEED}/prepared"
IMAGE_INDEX="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_self_optimization_stress_prepared/20260814_v1/image_index.csv"
IMAGE_ROOT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/derived/png224_20260804_v1/images"
FULL_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_full_issue_pool_iteration_mobilenet/20260824_multiseed_v1"
FULL_SEED="${FULL_ROOT}/seed_${SEED}"
INITIAL_EVIDENCE="${FULL_ROOT}/loop0_evidence/seed_${SEED}"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_matched_budget_refinement_mobilenet/20260824_v1"
OUTPUT="${OUTPUT_ROOT}/seed_${SEED}"

check_hash() {
  local path="$1"
  local expected="$2"
  [[ "$(sha256sum "${path}" | cut -d' ' -f1)" == "${expected}" ]] || {
    echo "Hash mismatch: ${path}" >&2
    exit 2
  }
}
check_hash "${DRIVER}" "82aef55e4e099f280590d0d6e72eb9346bfd376eb8aac4892364d8139c7a0367"
check_hash "${ENGINE}" "706c38a8af84a740a06261b459c3b733e6a88eff21bd9017ae62dec04b0a3f04"
check_hash "${RANK_HELPER}" "ad42522d355beaebaa1a19f441b7c281926c4936bb0a8ec52b9edef48489e6a6"
check_hash "${STRESS_HELPER}" "96f47a284e1092c610a9327c4f646af4c067be87cfde70b01ca60f98bccda331"
check_hash "${PROTOCOL}" "d29179f28fc44ac49503c763171ddbd4081771df0b0c58562b263357fe5fd9a5"
[[ -e "${PREPARED}/.prepare_complete" ]] || { echo "Prepared marker missing" >&2; exit 2; }
[[ -e "${INITIAL_EVIDENCE}/.blind_run_complete" ]] || { echo "Loop-0 evidence missing" >&2; exit 2; }
[[ -e "${FULL_SEED}/.worker_complete" ]] || { echo "Full-pool seed is incomplete" >&2; exit 2; }
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite ${OUTPUT}" >&2; exit 2; }

source /vol/cuda/12.5.0/setup.sh
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
mkdir -p "${OUTPUT_ROOT}" "${MPLCONFIGDIR}"

run_oof() {
  local cohort="$1"
  local target="$2"
  "${PYTHON}" -u "${ENGINE}" \
    --blind-cohort "${cohort}" \
    --sentinel-cohort "${PREPARED}/sentinel_blind_cohort.csv" \
    --split-reference-cohort "${PREPARED}/blind_noisy_cohort.csv" \
    --image-index "${IMAGE_INDEX}" \
    --image-root "${IMAGE_ROOT}" \
    --output-dir "${target}" \
    --seed "${SEED}" \
    --n-splits 4 \
    --epochs 50 \
    --early-stopping-patience 8 \
    --learning-rate 0.001 \
    --batch-size 32 \
    --num-workers 4 \
    --device cuda
}

COMMON=(
  --output-dir "${OUTPUT}"
  --source-prepared "${PREPARED}"
  --private-reference "${PREPARED}/private_reference.csv"
  --sentinel-private-reference "${PREPARED}/sentinel_private_reference.csv"
  --initial-evidence-dir "${INITIAL_EVIDENCE}"
  --full-seed-dir "${FULL_SEED}"
  --seed "${SEED}"
  --loops 5
)

echo "Starting VinDr matched-budget refinement seed=${SEED}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
"${PYTHON}" -u "${DRIVER}" initialize "${COMMON[@]}"
for loop_id in $(seq 1 5); do
  "${PYTHON}" -u "${DRIVER}" select "${COMMON[@]}" --loop-id "${loop_id}"
  "${PYTHON}" -u "${DRIVER}" oracle-update "${COMMON[@]}" --loop-id "${loop_id}"
  LOOP_DIR="${OUTPUT}/loop_$(printf '%02d' "${loop_id}")"
  run_oof "${LOOP_DIR}/cohort_after_action_blind.csv" "${LOOP_DIR}/oof_after_action"
  "${PYTHON}" -u "${DRIVER}" finalize-loop "${COMMON[@]}" --loop-id "${loop_id}"
done
"${PYTHON}" -u "${DRIVER}" evaluate "${COMMON[@]}"

"${PYTHON}" - "${OUTPUT}" "${INITIAL_EVIDENCE}" "${FULL_SEED}" "${SEED}" <<'PY'
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys

run, initial_evidence, full_seed, seed = (
    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
)
manifest = json.loads((run / "state/initialization_manifest.json").read_text())
summary = json.loads((run / "seed_evaluation_summary.json").read_text())
endpoints = list(map(int, manifest["cumulative_review_endpoints"]))
budgets = list(map(int, manifest["per_loop_review_budgets"]))
if manifest["seed"] != seed or summary["seed"] != seed:
    raise RuntimeError("Seed provenance failed")
if endpoints[0] != 0 or endpoints[-1] != manifest["initial_issue_pool"]:
    raise RuntimeError("Matched review endpoints failed")
if sum(budgets) != manifest["initial_issue_pool"] or any(value <= 0 for value in budgets):
    raise RuntimeError("Per-loop budget partition failed")
history = pd.read_csv(run / "state/review_history_private.csv")
if len(history) != manifest["initial_issue_pool"] or history["entry_key"].duplicated().any():
    raise RuntimeError("Dynamic review history failed")
if sorted(history["loop"].unique()) != [1, 2, 3, 4, 5]:
    raise RuntimeError("Dynamic loop history failed")
for loop_id, budget in enumerate(budgets, start=1):
    target = run / f"loop_{loop_id:02d}"
    selection = pd.read_csv(target / "selected_entries_blind.csv")
    selection_manifest = json.loads((target / "selection_manifest_blind.json").read_text())
    if len(selection) != budget or selection_manifest["review_budget"] != budget:
        raise RuntimeError(f"Loop {loop_id} budget failed")
    if not (target / ".loop_complete").is_file():
        raise RuntimeError(f"Loop {loop_id} completion marker missing")
    evidence_summary = json.loads((target / "oof_after_action/blind_run_summary.json").read_text())
    if "MobileNetV3-small" not in evidence_summary["architecture"]:
        raise RuntimeError(f"Loop {loop_id} architecture failed")
    if evidence_summary["initialization"] != "scratch":
        raise RuntimeError(f"Loop {loop_id} initialization failed")
initial = pd.read_csv(initial_evidence / "entry_evidence.csv")
frozen = pd.read_csv(run / "state/frozen_loop0_ranking_blind.csv")
full = pd.read_csv(full_seed / "loop_01/selected_entries_blind.csv")
if set(frozen.head(manifest["initial_issue_pool"])["entry_key"].astype(str)) != set(full["entry_key"].astype(str)):
    raise RuntimeError("Frozen endpoint and one-shot pool differ")
if int(initial["cl_issue"].sum()) != manifest["initial_issue_pool"]:
    raise RuntimeError("Initial issue-pool count failed")
trajectory = pd.read_csv(run / "dynamic_evidence_trajectory_private.csv")
discovery = pd.read_csv(run / "matched_discovery_trajectory_private.csv")
if trajectory["loop"].tolist() != list(range(6)) or len(discovery) != 18:
    raise RuntimeError("Matched trajectories are incomplete")
required = [
    "dynamic_minus_frozen_audc", "dynamic_final_quality", "frozen_final_quality",
    "dynamic_final_action_auroc", "one_shot_action_auroc",
    "dynamic_final_sentinel_auroc", "one_shot_sentinel_auroc",
]
if not np.isfinite([float(summary[key]) for key in required]).all():
    raise RuntimeError("Matched outcomes contain non-finite values")
if summary["frozen_endpoint_matches_one_shot_keys"] is not True:
    raise RuntimeError("Endpoint equality flag failed")
if not (run / ".seed_evaluation_complete").is_file():
    raise RuntimeError("Seed evaluation marker missing")
print(f"VinDr matched-budget postflight passed for seed {seed}")
PY
printf 'job_id=%s\nseed=%s\n' "${SLURM_JOB_ID:-manual}" "${SEED}" > "${OUTPUT}/.worker_complete"
echo "VinDr matched-budget refinement seed=${SEED} completed"
