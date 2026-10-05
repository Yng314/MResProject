#!/bin/bash
#SBATCH --job-name=mbv3-eval
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mbv3_eval_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mbv3_eval_%j.err
#SBATCH --partition=a30
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_DIR}/NoiseRate/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"

ROOT="${ROOT_OVERRIDE:-/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_noes50_clean_3seed/20260707_123320}"
OUT_DIR="${OUT_DIR_OVERRIDE:-${ROOT}/evaluation_$(date +%Y%m%d_%H%M%S)}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS_OVERRIDE:-1000}"
SEEDS="${SEEDS_OVERRIDE:-}"

mkdir -p "${OUT_DIR}" "${PROJECT_DIR}/slurm_logs"

echo "MobileNet noES evaluation job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Root: ${ROOT}"
echo "Out dir: ${OUT_DIR}"
echo "Bootstrap iters: ${BOOTSTRAP_ITERS}"
echo "Seeds override: ${SEEDS:-auto}"
echo "Started: $(date)"

cd "${PROJECT_DIR}/NoiseRate" || exit 1

SEED_ARGS=()
if [[ -n "${SEEDS}" ]]; then
  # shellcheck disable=SC2206
  SEED_ARGS=(--seeds ${SEEDS})
fi

"${PYTHON}" "${SCRIPT_DIR}/evaluate_mobilenet_noes_seed_robustness.py" \
  --root "${ROOT}" \
  --out-dir "${OUT_DIR}" \
  --bootstrap-iters "${BOOTSTRAP_ITERS}" \
  "${SEED_ARGS[@]}"

echo "Completed: $(date)"
