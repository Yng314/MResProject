#!/bin/bash
#SBATCH --job-name=vindr-evid-eval
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_evid_eval_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_evid_eval_%j.err
#SBATCH --partition=a16
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
SCRIPT_DIR="${ROOT}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
DRIVER="${SCRIPT_DIR}/vindr_iterative_evidence_improvement.py"
PROTOCOL="${SCRIPT_DIR}/vindr_iterative_evidence_improvement_protocol_20260813.md"
EXPERIMENT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_iterative_evidence_improvement/20260813_v2"
OUTPUT="${EXPERIMENT}/aggregate"
EXPECTED_DRIVER_SHA="ad42522d355beaebaa1a19f441b7c281926c4936bb0a8ec52b9edef48489e6a6"
EXPECTED_PROTOCOL_SHA="be5a0db22997a91afb13e9588148afa4aaacb901d704b0fe87acab548707c807"

for item in "${DRIVER}:${EXPECTED_DRIVER_SHA}" "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  [[ "$(sha256sum "${path}" | cut -d' ' -f1)" == "${expected}" ]] || {
    echo "Hash mismatch: ${path}" >&2
    exit 2
  }
done
for branch in scratch xrv_pretrained; do
  for seed in 13 42 97 123 211 307; do
    [[ -e "${EXPERIMENT}/${branch}/seed_${seed}/.formal_run_complete" ]] || {
      echo "Formal run incomplete: ${branch} seed ${seed}" >&2
      exit 2
    }
  done
done
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite ${OUTPUT}" >&2; exit 2; }

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"
"${PYTHON}" -u "${DRIVER}" aggregate \
  --experiment-root "${EXPERIMENT}" \
  --aggregate-output "${OUTPUT}" \
  --seeds 13,42,97,123,211,307
[[ -e "${OUTPUT}/.aggregate_complete" ]] || { echo "Aggregate marker missing" >&2; exit 2; }
printf 'job_id=%s\n' "${SLURM_JOB_ID:-manual}" > "${EXPERIMENT}/.complete"
echo "Aggregate completed: $(date --iso-8601=seconds)"
