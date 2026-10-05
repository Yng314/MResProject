#!/bin/bash
#SBATCH --job-name=entry-llm-55
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/entry_llm_55_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/entry_llm_55_%j.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

export OPENAI_MODEL_OVERRIDE="gpt-5.5"
exec /bin/bash /vol/gpudata/yz3522-llmtest/MResProject/NoiseRate/cxr_real_experiment/run_entry_level_llm_review_real_smoke.sh
