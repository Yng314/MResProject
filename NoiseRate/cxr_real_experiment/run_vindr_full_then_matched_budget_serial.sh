#!/bin/bash
#SBATCH --job-name=vfull-then-match
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_full_then_matched_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_full_then_matched_%j.err
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=44G
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

SCRIPT_DIR="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/cxr_real_experiment"
FULL_SERIAL="${SCRIPT_DIR}/run_vindr_full_issue_pool_mobilenet_remaining_serial.sh"
MATCHED_RUNNER="${SCRIPT_DIR}/run_vindr_matched_budget_refinement_seed.sh"
MATCHED_AGGREGATE="${SCRIPT_DIR}/run_vindr_matched_budget_refinement_aggregate.sh"
FULL_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_full_issue_pool_iteration_mobilenet/20260824_multiseed_v1"
MATCHED_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_matched_budget_refinement_mobilenet/20260824_v1"
SEEDS=(11003 13007 17011 19001 23003 27011 31013 37003)

bash -n "${FULL_SERIAL}" "${MATCHED_RUNNER}" "${MATCHED_AGGREGATE}"

if [[ -e "${FULL_ROOT}/.complete" && -e "${FULL_ROOT}/aggregate/.aggregate_complete" ]]; then
  echo "Full-issue-pool experiment and aggregate already complete; skipping."
else
  echo "Resuming and completing the full-issue-pool experiment at $(date --iso-8601=seconds)"
  bash "${FULL_SERIAL}"
fi

echo "Starting the matched-budget experiment after full-pool completion at $(date --iso-8601=seconds)"
for seed in "${SEEDS[@]}"; do
  if [[ -e "${MATCHED_ROOT}/seed_${seed}/.worker_complete" ]]; then
    echo "Matched-budget seed ${seed} already complete; skipping."
    continue
  fi
  [[ ! -e "${MATCHED_ROOT}/seed_${seed}" ]] || {
    echo "Incomplete matched-budget output requires audit before resume: seed ${seed}" >&2
    exit 2
  }
  bash "${MATCHED_RUNNER}" "${seed}"
done

bash "${MATCHED_AGGREGATE}"
echo "Full-pool continuation and matched-budget comparison completed at $(date --iso-8601=seconds)"
