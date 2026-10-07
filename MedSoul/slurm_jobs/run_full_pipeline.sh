#!/bin/bash
#SBATCH --job-name=medsoul_full
#SBATCH --output=slurm_logs/full_%j.log
#SBATCH --error=slurm_logs/full_%j.err
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yz3522

# ============================================================
# MedSoul Full Pipeline (WSL + Noise Rate Estimation)
# ============================================================

set -e

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MedSoul"
cd "${PROJECT_DIR}" || exit 1

mkdir -p slurm_logs

echo "=============================================="
echo "MedSoul Full Pipeline"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Time: $(date)"
echo "=============================================="

# 1. Load CUDA
echo "[Setup] Loading CUDA..."
source /vol/cuda/12.5.0/setup.sh

# 2. Activate virtual environment
echo "[Setup] Activating virtual environment..."
source /vol/gpudata/yz3522-llmtest/venv/bin/activate

# 3. Check GPU
echo "[Setup] GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# 4. Phase 1: WSL Training
echo ""
echo "=============================================="
echo "Phase 1: WSL Training"
echo "=============================================="
python train_wsl.py --config slurm_jobs/config_slurm.yaml

# 5. Phase 2: Noise Rate Estimation
echo ""
echo "=============================================="
echo "Phase 2: Noise Rate Estimation"
echo "=============================================="
python estimate_noise_rate.py --config slurm_jobs/config_slurm.yaml

echo ""
echo "=============================================="
echo "All phases completed at $(date)"
echo "=============================================="
