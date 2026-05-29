#!/bin/bash
#SBATCH --job-name=cxr_real_smoke
#SBATCH --output=slurm_logs/cxr_real_smoke_%j.log
#SBATCH --error=slurm_logs/cxr_real_smoke_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -e

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MRes"
cd "${PROJECT_DIR}" || exit 1

mkdir -p slurm_logs

echo "=============================================="
echo "Real CXR Noise Smoke"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=============================================="

source /vol/cuda/12.5.0/setup.sh
source /vol/gpudata/yz3522-llmtest/venv/bin/activate

echo "[GPU]"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

python -u NoiseRate/cxr_real_experiment/cxr_real_noise_validation_smoke.py \
  --image-root /vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-jpg-224 \
  --chexpert-csv /vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-jpg-224/mimic-cxr-2.0.0-chexpert.csv \
  --split-csv /vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-2.0.0-split.csv.gz \
  --split train \
  --num-samples 1000 \
  --n-splits 3 \
  --epochs 2 \
  --batch-size 32 \
  --image-size 224 \
  --learning-rate 0.001 \
  --early-stopping-patience 2 \
  --model-backbone lightcnn \
  --device cuda \
  --output-dir /vol/gpudata/yz3522-llmtest/MRes/NoiseRate/cxr_real_experiment/results_smoke/slurm_${SLURM_JOB_ID}

echo ""
echo "Completed at $(date)"
