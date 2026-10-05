#!/bin/bash
#SBATCH --job-name=vindr-aum-eval
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_aum_eval_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_aum_eval_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

PROJECT_ROOT="/vol/gpudata/yz3522-llmtest/MResProject"
NOISE_ROOT="${PROJECT_ROOT}/NoiseRate"
SCRIPT_DIR="${NOISE_ROOT}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
PROGRAM="${SCRIPT_DIR}/aggregate_vindr_oof_aum_screen.py"
TEST_PROGRAM="${SCRIPT_DIR}/test_aggregate_vindr_oof_aum_screen.py"
ROOT="${VINDR_OOF_AUM_OUTPUT_OVERRIDE:-/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_oof_aum_screen/20260812_v1}"

for item in \
  "${PROGRAM}:a92fadcbe511a902550abd5894ccb8598a3a575288138c870e78c447af12842b" \
  "${TEST_PROGRAM}:0d49f64d27ef30f057212ef3714719d9addae1161b6361b0c5e9cc610f63d3da"; do
  path="${item%%:*}"
  expected="${item##*:}"
  if [[ "$(sha256sum "${path}" | cut -d' ' -f1)" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done
for seed in 13 211; do
  for marker in \
    "${ROOT}/seed_${seed}/.benchmark_verified" \
    "${ROOT}/seed_${seed}/private_evaluation/.evaluation_complete"; do
    if [[ ! -f "${marker}" ]]; then
      echo "Required marker is missing: ${marker}" >&2
      exit 2
    fi
  done
done
if [[ -e "${ROOT}/aggregate" ]]; then
  echo "Refusing to overwrite aggregate output" >&2
  exit 2
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${PROJECT_ROOT}/slurm_logs" "${MPLCONFIGDIR}"

echo "VinDr OOF AUM-style screen aggregate"
echo "Job: ${SLURM_JOB_ID:-manual}"
echo "Started: $(date --iso-8601=seconds)"
"${PYTHON}" -m unittest -v test_aggregate_vindr_oof_aum_screen
"${PYTHON}" -u "${PROGRAM}" --root "${ROOT}"
if [[ ! -f "${ROOT}/aggregate/.aggregate_complete" ]]; then
  echo "Aggregate completion marker is missing" >&2
  exit 2
fi
echo "Completed: $(date --iso-8601=seconds)"
