#!/bin/bash
#SBATCH --job-name=medsoul_wsl
#SBATCH --output=slurm_logs/wsl_%j.log
#SBATCH --error=slurm_logs/wsl_%j.err
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yz3522

# ============================================================
# MedSoul WSL Training Script for Slurm
# ============================================================

# Exit on any error
set -e

# Project directory
PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MedSoul"
cd "${PROJECT_DIR}" || exit 1

# Create log directory if not exists
mkdir -p slurm_logs

echo "=============================================="
echo "MedSoul WSL Training Job Started"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=============================================="

# 1. Load CUDA
echo "[1/4] Loading CUDA..."
source /vol/cuda/12.5.0/setup.sh

# 2. Activate virtual environment
echo "[2/4] Activating virtual environment..."
source /vol/gpudata/yz3522-llmtest/venv/bin/activate

# 3. Check GPU
echo "[3/4] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# 4. Run training
echo "[4/4] Starting WSL training..."
echo "Config: slurm_jobs/config_slurm.yaml"
echo ""

python train_wsl.py --config slurm_jobs/config_slurm.yaml

echo ""
echo "=============================================="
echo "Training completed at $(date)"
echo "=============================================="
