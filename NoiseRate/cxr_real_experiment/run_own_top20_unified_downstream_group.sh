#!/bin/bash
#SBATCH --job-name=mb-top20-group
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mb_top20_group_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mb_top20_group_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail

END_LOOP="5"
if [[ "${1:-}" == "--end-loop" ]]; then
  END_LOOP="${2:?--end-loop requires a value}"
  shift 2
fi

EXPERIMENT_ROOT="${1:?experiment root is required}"
shift
if (( $# == 0 )); then
  echo "At least one seed is required." >&2
  exit 2
fi

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
RUNNER="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/run_mobilenetv3_unified_downstream_seed.sh"

if ! [[ "${END_LOOP}" =~ ^[1-8]$ ]]; then
  echo "end loop must be an integer from 1 to 8: ${END_LOOP}" >&2
  exit 2
fi

for seed in "$@"; do
  case "${seed}" in
    7|13|42|97|123) ;;
    *) echo "Unsupported own-top20 seed: ${seed}" >&2; exit 2 ;;
  esac
  echo "Starting own-top20 unified downstream evaluation for seed ${seed}."
  bash "${RUNNER}" "${EXPERIMENT_ROOT}" "${seed}" 0.20 1 "${END_LOOP}" 1
done

echo "Requested own-top20 seed group completed unified downstream evaluation through Loop ${END_LOOP}."
