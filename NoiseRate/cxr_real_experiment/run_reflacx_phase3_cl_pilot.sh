#!/bin/bash
#SBATCH --job-name=reflacx-cl-pilot
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/reflacx_cl_pilot_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/reflacx_cl_pilot_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail

OUTPUT_ROOT="${1:?output root is required}"
SEED="${2:-13}"
PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT="${PROJECT_DIR}/NoiseRate/cxr_real_experiment/reflacx_phase3_cl_pilot.py"
DATA_ROOT="${PROJECT_DIR}/MedSoul/datasets"
IMAGE_ROOT="${DATA_ROOT}/mimic-cxr-jpg-224"
PHASE3_CSV="${DATA_ROOT}/reflacx/main_data/metadata_phase_3.csv"
CHEXPERT_CSV="${DATA_ROOT}/mimic-cxr-2.0.0-chexpert.csv.gz"
PREP_DIR="${OUTPUT_ROOT}/prepared"
BLIND_DIR="${OUTPUT_ROOT}/blind_run"
EVAL_DIR="${OUTPUT_ROOT}/private_evaluation"

mkdir -p "${PROJECT_DIR}/slurm_logs" "${PREP_DIR}" "${BLIND_DIR}" "${EVAL_DIR}"
source /vol/cuda/12.5.0/setup.sh
source /vol/gpudata/yz3522-llmtest/venv/bin/activate
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"

echo "=========================================================="
echo "REFLACX Phase-3 confident-learning pilot"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Seed: ${SEED}"
echo "Output: ${OUTPUT_ROOT}"
echo "Reference is unavailable to the blind run transaction."
echo "Started: $(date --iso-8601=seconds)"
echo "=========================================================="

python -u "${SCRIPT}" prepare \
  --phase3-csv "${PHASE3_CSV}" \
  --chexpert-csv "${CHEXPERT_CSV}" \
  --image-root "${IMAGE_ROOT}" \
  --output-dir "${PREP_DIR}" \
  --seed "${SEED}" \
  --n-splits 4

python -u "${SCRIPT}" run \
  --blind-cohort "${PREP_DIR}/blind_cohort.csv" \
  --image-root "${IMAGE_ROOT}" \
  --output-dir "${BLIND_DIR}" \
  --seed "${SEED}" \
  --n-splits 4 \
  --epochs 50 \
  --batch-size 32 \
  --image-size 224 \
  --learning-rate 0.001 \
  --early-stopping-patience 10 \
  --num-workers 4 \
  --top-fraction 0.20 \
  --device cuda

test -s "${BLIND_DIR}/.blind_run_complete"

python -u "${SCRIPT}" evaluate \
  --blind-cohort "${PREP_DIR}/blind_cohort.csv" \
  --private-reference "${PREP_DIR}/private_reference.csv" \
  --entry-evidence "${BLIND_DIR}/entry_evidence.csv" \
  --output-dir "${EVAL_DIR}" \
  --bootstrap-iterations 1000 \
  --bootstrap-seed 20260730

test -s "${EVAL_DIR}/.evaluation_complete"
echo "Completed: $(date --iso-8601=seconds)"
