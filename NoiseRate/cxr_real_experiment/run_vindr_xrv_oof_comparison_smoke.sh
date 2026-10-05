#!/bin/bash
#SBATCH --job-name=vindr-xrv-oof-smoke
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_xrv_oof_smoke_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_xrv_oof_smoke_%j.err
#SBATCH --partition=training
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
SCRIPT_DIR="${ROOT}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
ENGINE="${SCRIPT_DIR}/vindr_known_gt_cl_benchmark.py"
PROGRAM="${SCRIPT_DIR}/vindr_xrv_oof_comparison.py"
TEST_PROGRAM="${SCRIPT_DIR}/test_vindr_xrv_oof_comparison.py"
PROTOCOL="${SCRIPT_DIR}/vindr_xrv_oof_comparison_protocol_20260813.md"
PARENT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_detector_benchmark_v2/20260812_v1"
MOBILE="${PARENT}/blind_detector_scores_v2"
FEATURE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_detector_benchmark_v2_smoke/20260812_v1/external_evidence"
FEATURES="${FEATURE_ROOT}/xrv_external_evidence.npz"
OUTPUT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_xrv_oof_comparison_smoke/20260813_v1"
XRV_ROOT="${OUTPUT}/xrv_oof_runs"
XRV_RUN="${XRV_ROOT}/scenarios/balanced_r20/seed_887/blind_run"
COMPARISON="${OUTPUT}/comparison"

EXPECTED_ENGINE_SHA="f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83"
EXPECTED_PROGRAM_SHA="1b3e02e01699f0d891928c3d9fc0cd3618770d4f0e5ed4323cebf119573cc976"
EXPECTED_TEST_SHA="347764d512d905ca7eeb7d2e892d98b3f8c5cff232ca63aec49d44bd7442021c"
EXPECTED_PROTOCOL_SHA="ebbdd154adc6abdafa280886335bb68aef728feb716ccad13fac6365387ae8ef"
EXPECTED_FEATURE_SHA="c03864a90dfcb339b13c76444ddc116f90be67520f56f12f1158097c62ed35c6"
EXPECTED_MANIFEST_SHA="305864db2ecc316860cdd9c7c2fd46512c1c08325d74c41413d48f530ff5adb6"

for item in \
  "${ENGINE}:${EXPECTED_ENGINE_SHA}" \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${TEST_PROGRAM}:${EXPECTED_TEST_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${FEATURES}:${EXPECTED_FEATURE_SHA}" \
  "${PARENT}/blind_run_manifest.csv:${EXPECTED_MANIFEST_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  [[ "$(sha256sum "${path}" | cut -d' ' -f1)" == "${expected}" ]] || {
    echo "Hash mismatch: ${path}" >&2
    exit 2
  }
done
for marker in \
  "${PARENT}/.prepare_complete" \
  "${MOBILE}/.benchmark_verified" \
  "${FEATURE_ROOT}/.external_evidence_complete" \
  "${PARENT}/scenarios/balanced_r20/seed_887/prepared/.prepare_complete"; do
  [[ -e "${marker}" ]] || { echo "Required marker is missing: ${marker}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "Smoke output exists: ${OUTPUT}" >&2; exit 2; }

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$(dirname "${XRV_RUN}")" "${MPLCONFIGDIR}"

echo "VinDr XRV OOF comparison smoke"
echo "Job: ${SLURM_JOB_ID:-manual}; started: $(date --iso-8601=seconds)"
"${PYTHON}" -m unittest -v test_vindr_xrv_oof_comparison
"${PYTHON}" -u "${ENGINE}" run \
  --blind-cohort "${PARENT}/scenarios/balanced_r20/seed_887/prepared/blind_noisy_cohort.csv" \
  --features "${FEATURES}" \
  --output-dir "${XRV_RUN}" \
  --seed 887 \
  --n-splits 4 \
  --epochs 50 \
  --early-stopping-patience 8 \
  --learning-rate 0.001 \
  --batch-size 128 \
  --device cpu
"${PYTHON}" -u "${PROGRAM}" score \
  --parent-root "${PARENT}" \
  --mobile-score-root "${MOBILE}" \
  --xrv-oof-root "${XRV_ROOT}" \
  --output-root "${COMPARISON}" \
  --seeds 887 \
  --scenarios balanced_r20
"${PYTHON}" -u "${PROGRAM}" evaluate \
  --parent-root "${PARENT}" \
  --output-root "${COMPARISON}" \
  --smoke-mode
"${PYTHON}" -u "${PROGRAM}" verify \
  --parent-root "${PARENT}" \
  --output-root "${COMPARISON}" \
  --expected-runs 1 \
  --smoke-mode
printf 'smoke_job_id=%s\n' "${SLURM_JOB_ID:-manual}" > "${OUTPUT}/.smoke_complete"
echo "Completed: $(date --iso-8601=seconds)"
