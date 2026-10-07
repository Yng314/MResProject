#!/bin/bash
#SBATCH --job-name=medsoul_noise
#SBATCH --output=slurm_logs/noise_rate_%j.log
#SBATCH --error=slurm_logs/noise_rate_%j.err
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yz3522

# ============================================================
# MedSoul Noise Rate Estimation Script for Slurm
# ============================================================

set -e

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MedSoul"
cd "${PROJECT_DIR}" || exit 1

mkdir -p slurm_logs

echo "=============================================="
echo "MedSoul Noise Rate Estimation"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Time: $(date)"
echo "=============================================="

# 1. Load CUDA
source /vol/cuda/12.5.0/setup.sh

# 2. Activate virtual environment
source /vol/gpudata/yz3522-llmtest/venv/bin/activate

# 3. Run noise rate estimation
echo "Running noise rate estimation..."
python estimate_noise_rate.py --config slurm_jobs/config_slurm.yaml

echo ""
echo "Completed at $(date)"
