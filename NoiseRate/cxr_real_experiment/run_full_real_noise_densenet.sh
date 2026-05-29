#!/bin/bash
#SBATCH --job-name=cxr_real_dense
#SBATCH --output=slurm_logs/cxr_real_dense_%j.log
#SBATCH --error=slurm_logs/cxr_real_dense_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -e

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MRes"
cd "${PROJECT_DIR}" || exit 1

mkdir -p slurm_logs

echo "=============================================="
echo "Real CXR DenseNet Full-Train Trial"
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
  --num-samples -1 \
  --n-splits 3 \
  --epochs 1 \
  --batch-size 16 \
  --image-size 224 \
  --learning-rate 0.001 \
  --early-stopping-patience 1 \
  --model-backbone densenet121_pretrained \
  --device cuda \
  --output-dir /vol/gpudata/yz3522-llmtest/MRes/NoiseRate/cxr_real_experiment/results_full/slurm_${SLURM_JOB_ID}

echo ""
echo "Completed at $(date)"
