#!/bin/bash
#SBATCH --job-name=vfull-mnet-ms
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_fullpool_mobilenet_%A_%a.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_fullpool_mobilenet_%A_%a.err
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-6%1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail

SCRIPT_DIR="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/cxr_real_experiment"
RUNNER="${SCRIPT_DIR}/run_vindr_full_issue_pool_mobilenet_seed.sh"
SEEDS=(13007 17011 19001 23003 27011 31013 37003)
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
[[ "${TASK_ID}" =~ ^[0-6]$ ]] || { echo "Unexpected array task: ${TASK_ID}" >&2; exit 2; }
exec bash "${RUNNER}" "${SEEDS[${TASK_ID}]}"
