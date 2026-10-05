#!/bin/bash
#SBATCH --job-name=mim-dqs-cal
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mimic_dqs_cal_score_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/mimic_dqs_cal_score_%j.err
#SBATCH --partition=a16,a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

GROUP_NAME="${1:?group name is required}"
shift
(( $# > 0 )) || { echo "At least one seed is required" >&2; exit 2; }
SEEDS=("$@")

PROJECT_ROOT="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_ROOT}/NoiseRate/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
PROGRAM="${SCRIPT_DIR}/mimic_dqs_calibration_score.py"
TEST="${SCRIPT_DIR}/test_mimic_dqs_calibration_score.py"
PROTOCOL="${SCRIPT_DIR}/mimic_dqs_calibration_transfer_protocol_20260827.md"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/mimic_dqs_calibration_transfer/20260827_v1"
SCORE_ROOT="${OUTPUT_ROOT}/scores"
DATA_ROOT="${PROJECT_ROOT}/MedSoul/datasets"
IMAGE_ROOT="${DATA_ROOT}/mimic-cxr-jpg-224"
CHEXPERT="${IMAGE_ROOT}/mimic-cxr-2.0.0-chexpert.csv"
SPLIT="${DATA_ROOT}/mimic-cxr-2.0.0-split.csv.gz"
METADATA="${DATA_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz"
ARCHIVE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_noes50_clean_3seed/20260707_123320"

check_hash() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(sha256sum "${path}" | cut -d' ' -f1)"
  [[ "${observed}" == "${expected}" ]] || {
    echo "Hash mismatch for ${path}: ${observed} != ${expected}" >&2
    exit 2
  }
}

check_hash "${PROGRAM}" "90e90cebb4f61e0427cea3b6b347fb0f1ce2b9a739ec7b9b6565f508a1f0ea7f"
check_hash "${TEST}" "747cc3ce96a6adcb349d3bad7fe5520be14e3052ade7d4d73c08caefb6462095"
check_hash "${PROTOCOL}" "19b307b802993f239c3aae12063c78895e39d91126f2220d880ecc5f90273a0b"
check_hash "${SCRIPT_DIR}/cxr_real_oof_cleanlab_smoke.py" "dc6b2d577985bbeec4d1c9e71911fca78419f63126c7c4eaa1d2397943c29a7b"
check_hash "${SCRIPT_DIR}/cxr_real_noise_validation_smoke.py" "62da5b019f9809b4641b58f3c9bda56c1f987057fc765f9a76f32feade7046e0"
check_hash "${SCRIPT_DIR}/cxr_real_full_train_eval_cleanlab_xrv12.py" "f9b1eb32aaa7d6df9e1d1ebd3ba66fd2ef637bde47676fda4c7b629a7c7f452d"

source /vol/cuda/12.5.0/setup.sh
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "${SCORE_ROOT}" "${PROJECT_ROOT}/slurm_logs" "${MPLCONFIGDIR}"

"${PYTHON}" -m py_compile "${PROGRAM}" "${TEST}"
"${PYTHON}" "${TEST}"

echo "MIMIC-CXR DQS outcome-blind calibration scoring"
echo "Job: ${SLURM_JOB_ID:-manual}"
echo "Group: ${GROUP_NAME}"
echo "Seeds: ${SEEDS[*]}"
echo "Node: ${SLURMD_NODENAME:-manual}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

for seed in "${SEEDS[@]}"; do
  case "${seed}" in
    7|13|42|97|123) ;;
    *) echo "Unexpected seed: ${seed}" >&2; exit 2 ;;
  esac
  final="${SCORE_ROOT}/seed_${seed}"
  if [[ -e "${final}/.score_complete" ]]; then
    echo "Seed ${seed} already complete; skipping"
    continue
  fi
  [[ ! -e "${final}" ]] || { echo "Incomplete final directory exists: ${final}" >&2; exit 2; }
  partial="${final}.partial_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}"
  [[ ! -e "${partial}" ]] || { echo "Attempt directory already exists: ${partial}" >&2; exit 2; }
  archived="${ARCHIVE_ROOT}/seed_${seed}/sample20_remove_loop/remove_only/loop_01/oof/train_cleanlab_sample_details.csv"
  [[ -s "${archived}" ]] || { echo "Missing archived OOF details: ${archived}" >&2; exit 2; }
  "${PYTHON}" -u "${PROGRAM}" \
    --image-root "${IMAGE_ROOT}" \
    --chexpert-csv "${CHEXPERT}" \
    --split-csv "${SPLIT}" \
    --metadata-csv "${METADATA}" \
    --archived-detail-csv "${archived}" \
    --output-dir "${partial}" \
    --seed "${seed}" \
    --n-splits 4 \
    --epochs 100 \
    --batch-size 32 \
    --image-size 224 \
    --learning-rate 0.001 \
    --early-stopping-patience 10 \
    --num-workers 4 \
    --device cuda
  mv "${partial}" "${final}"
done

printf 'job_id=%s\ngroup=%s\nseeds=%s\ncompleted_at=%s\n' \
  "${SLURM_JOB_ID:-manual}" "${GROUP_NAME}" "${SEEDS[*]}" "$(date --iso-8601=seconds)" \
  > "${SCORE_ROOT}/.group_${GROUP_NAME}_complete"
echo "MIMIC-CXR calibration score group ${GROUP_NAME} completed"
