#!/bin/bash
#SBATCH --job-name=mb-full-u100
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mb_full_u100_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mb_full_u100_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail

EXPERIMENT_ROOT="${1:?experiment root is required}"
SEED="${2:?seed is required}"
SOURCE_LOOP="${3:?source Loop 1 path is required}"
COMPLETED_OLD_LOOPS="${4:-0}"
TARGET_LOOPS="${5:-8}"
PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_DIR}/NoiseRate/cxr_real_experiment"

if ! [[ "${TARGET_LOOPS}" =~ ^[1-8]$ ]]; then
  echo "target loops must be an integer from 1 to 8: ${TARGET_LOOPS}" >&2
  exit 2
fi
if (( COMPLETED_OLD_LOOPS > TARGET_LOOPS )); then
  COMPLETED_OLD_LOOPS="${TARGET_LOOPS}"
fi

if (( COMPLETED_OLD_LOOPS > 0 )); then
  bash "${SCRIPT_DIR}/run_mobilenetv3_unified_downstream_seed.sh" \
    "${EXPERIMENT_ROOT}" "${SEED}" 1.0 1 "${COMPLETED_OLD_LOOPS}" 1
else
  bash "${SCRIPT_DIR}/run_mobilenetv3_unified_downstream_seed.sh" \
    "${EXPERIMENT_ROOT}" "${SEED}" 1.0 1 0 1
fi

exec bash "${SCRIPT_DIR}/run_mobilenetv3_full_issue_pool_no_repeat_seed.sh" \
  "${EXPERIMENT_ROOT}" "${SEED}" "${SOURCE_LOOP}" "${TARGET_LOOPS}"
