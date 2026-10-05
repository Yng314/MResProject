#!/bin/bash
#SBATCH --job-name=vfull-mnet-serial
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_fullpool_mobilenet_serial_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_fullpool_mobilenet_serial_%j.err
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

SCRIPT_DIR="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/cxr_real_experiment"
RUNNER="${SCRIPT_DIR}/run_vindr_full_issue_pool_mobilenet_seed.sh"
AGGREGATE="${SCRIPT_DIR}/run_vindr_full_issue_pool_mobilenet_aggregate.sh"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_full_issue_pool_iteration_mobilenet/20260824_multiseed_v1"
SEEDS=(13007 17011 19001 23003 27011 31013 37003)

[[ -e "${OUTPUT_ROOT}/seed_11003/.worker_complete" ]] || {
  echo "Seed-11003 pilot has not passed postflight" >&2
  exit 2
}
bash -n "${RUNNER}" "${AGGREGATE}"

for seed in "${SEEDS[@]}"; do
  if [[ -e "${OUTPUT_ROOT}/seed_${seed}/.worker_complete" ]]; then
    echo "Seed ${seed} already complete; skipping."
    continue
  fi
  echo "Starting remaining VinDr MobileNet seed ${seed} at $(date --iso-8601=seconds)"
  bash "${RUNNER}" "${seed}"
done

echo "All remaining seeds completed; starting aggregate at $(date --iso-8601=seconds)"
bash "${AGGREGATE}"
echo "Serial VinDr MobileNet extension completed at $(date --iso-8601=seconds)"
