#!/bin/bash
#SBATCH --job-name=report-fa-scan
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MRes/slurm_logs/report_false_alarm_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MRes/slurm_logs/report_false_alarm_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

source /vol/gpudata/yz3522-llmtest/venv/bin/activate

python /vol/gpudata/yz3522-llmtest/MRes/NoiseRate/cxr_real_experiment/find_false_alarm_candidates_from_reports.py \
  --entry-details-csv /vol/gpudata/yz3522-llmtest/MRes/NoiseRate/cxr_real_experiment/results_appa_xrv12_linearhead_pipeline/slurm_236337/01_baseline/train_cleanlab_entry_details.csv \
  --report-archive /vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-reports.tar.gz \
  --output-dir /vol/gpudata/yz3522-llmtest/MRes/NoiseRate/cxr_real_experiment/meeting_followup_20260520/report_false_alarm_scan \
  --top-k-per-label 20 \
  --min-pred-prob 0.95
