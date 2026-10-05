#!/bin/bash
#SBATCH --job-name=vindr-selfopt-smoke
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_selfopt_smoke_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_selfopt_smoke_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
SCRIPT_DIR="${ROOT}/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
ENGINE="${SCRIPT_DIR}/vindr_densenet_sentinel_oof.py"
BASE_ENGINE="${SCRIPT_DIR}/vindr_densenet_oof.py"
DRIVER="${SCRIPT_DIR}/vindr_self_optimization_stress.py"
TEST="${SCRIPT_DIR}/test_vindr_self_optimization_stress.py"
PROTOCOL="${SCRIPT_DIR}/vindr_self_optimization_stress_protocol_20260814.md"
FORMAL="${SCRIPT_DIR}/run_vindr_self_optimization_formal.sh"
LABELS="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/annotations/image_labels_test.csv"
SOURCE_IMAGE_INDEX="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_known_gt_cl/20260804_v1/image_index.csv"
HARDNESS="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_detector_benchmark_v2/20260812_v1/scenarios/clean/seed_887/blind_run/entry_evidence.csv"
PREP_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_self_optimization_stress_prepared/20260814_v1"
SMOKE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_self_optimization_stress_smoke/20260814_v1"
RUN_MANIFEST="${PREP_ROOT}/formal_run_manifest.csv"
IMAGE_ROOT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/derived/png224_20260804_v1/images"
XRV_CACHE="/vol/gpudata/yz3522-llmtest/.cache/torchxrayvision"

EXPECTED_ENGINE_SHA="faf2e31edaef6150c0600a1c07d0633e0292ef995dd3bca74179518a596470d4"
EXPECTED_BASE_ENGINE_SHA="fd7a8a3141680d27fe20955e17e7f77437ee0d9a6f0d18a45b16be628c9627bc"
EXPECTED_DRIVER_SHA="96f47a284e1092c610a9327c4f646af4c067be87cfde70b01ca60f98bccda331"
EXPECTED_TEST_SHA="a6d4d2389d27ca1f12266dea367d7d3ec00b1c574376955d66ae5a5b59ba56a3"
EXPECTED_PROTOCOL_SHA="a943d01cf8cd76c6eaad10280a2b74b5ca7aa40b63d02d92545d4469435b9bd1"
EXPECTED_FORMAL_SHA="c2eaf56b96d9ac3b997f317eb226f4cd465604a6298b5043521506d93593ae32"
EXPECTED_LABELS_SHA="9874c665991c5098db571082f9d9d096fe80c40ac3c2b2007a76d8d226c87a02"
EXPECTED_IMAGE_INDEX_SHA="645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2"
EXPECTED_HARDNESS_SHA="52c2bd7fbf8dd7666d9659ee9964a855bd2bdb50d0b4dbfe663d73d0ff8543d7"

for item in \
  "${ENGINE}:${EXPECTED_ENGINE_SHA}" \
  "${BASE_ENGINE}:${EXPECTED_BASE_ENGINE_SHA}" \
  "${DRIVER}:${EXPECTED_DRIVER_SHA}" \
  "${TEST}:${EXPECTED_TEST_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${FORMAL}:${EXPECTED_FORMAL_SHA}" \
  "${LABELS}:${EXPECTED_LABELS_SHA}" \
  "${SOURCE_IMAGE_INDEX}:${EXPECTED_IMAGE_INDEX_SHA}" \
  "${HARDNESS}:${EXPECTED_HARDNESS_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  [[ "$(sha256sum "${path}" | cut -d' ' -f1)" == "${expected}" ]] || {
    echo "Hash mismatch: ${path}" >&2
    exit 2
  }
done

source /vol/cuda/12.5.0/setup.sh
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
mkdir -p "$(dirname "${PREP_ROOT}")" "$(dirname "${SMOKE_ROOT}")" "${MPLCONFIGDIR}"

"${PYTHON}" "${TEST}" -v
if [[ ! -e "${PREP_ROOT}" ]]; then
  "${PYTHON}" -u "${DRIVER}" prepare \
    --labels-csv "${LABELS}" \
    --image-index "${SOURCE_IMAGE_INDEX}" \
    --hardness-evidence "${HARDNESS}" \
    --output-root "${PREP_ROOT}" \
    --seeds 11003,13007,17011,19001,23003,27011,31013,37003 \
    --rates 0.2,0.3,0.4 \
    --structures uniform,hard \
    --sentinel-noise-rate 0.2 \
    --n-splits 4
fi
[[ -e "${PREP_ROOT}/.prepare_complete" ]] || { echo "Preparation is incomplete" >&2; exit 2; }
[[ "$(sha256sum "${PREP_ROOT}/image_index.csv" | cut -d' ' -f1)" == "${EXPECTED_IMAGE_INDEX_SHA}" ]] || {
  echo "Prepared image index hash mismatch" >&2; exit 2;
}

if [[ ! -e "${RUN_MANIFEST}" ]]; then
  "${PYTHON}" - "${PREP_ROOT}" "${RUN_MANIFEST}" <<'PY'
from pathlib import Path
import sys
import pandas as pd

root, destination = Path(sys.argv[1]), Path(sys.argv[2])
seeds = [11003, 13007, 17011, 19001, 23003, 27011, 31013, 37003]
conditions = [
    ("scratch", "uniform_r20", "uniform", 0.20),
    ("scratch", "hard_r30", "hard", 0.30),
    ("xrv_pretrained", "hard_r40", "hard", 0.40),
    ("scratch", "uniform_r30", "uniform", 0.30),
    ("scratch", "hard_r20", "hard", 0.20),
    ("xrv_pretrained", "uniform_r20", "uniform", 0.20),
    ("scratch", "uniform_r40", "uniform", 0.40),
    ("scratch", "hard_r40", "hard", 0.40),
]
rows = []
for seed in seeds:
    for initialization, scenario, structure, rate in conditions:
        prepared = root / "scenarios" / scenario / f"seed_{seed}" / "prepared"
        if not (prepared / ".prepare_complete").is_file():
            raise RuntimeError(f"Missing prepared run: {prepared}")
        rows.append(
            {
                "initialization": initialization,
                "scenario_id": scenario,
                "structure": structure,
                "noise_rate": rate,
                "seed": seed,
                "prepared_path": str(prepared),
            }
        )
frame = pd.DataFrame(rows)
if len(frame) != 64 or len(frame[frame.initialization.eq("scratch")]) != 48:
    raise RuntimeError("Formal run manifest does not have the locked 48+16 design")
frame.insert(0, "run_index", range(len(frame)))
temporary = destination.with_suffix(destination.suffix + ".tmp")
frame.to_csv(temporary, index=False)
temporary.replace(destination)
PY
fi

"${PYTHON}" - "${PREP_ROOT}" "${RUN_MANIFEST}" <<'PY'
from pathlib import Path
import pandas as pd
import sys

root, manifest_path = Path(sys.argv[1]), Path(sys.argv[2])
prepared = pd.read_csv(root / "prepared_manifest.csv")
formal = pd.read_csv(manifest_path)
if len(prepared) != 48 or len(formal) != 64:
    raise RuntimeError("Prepared or formal manifest row count failed")
if formal["run_index"].tolist() != list(range(64)):
    raise RuntimeError("Formal run indices are not contiguous")
if set(formal["prepared_path"]) != set(prepared["prepared_path"]):
    raise RuntimeError("Formal Scratch design does not cover every prepared condition")
for row in formal.itertuples(index=False):
    if not (Path(row.prepared_path) / ".prepare_complete").is_file():
        raise RuntimeError("Formal manifest references incomplete preparation")
print("Preparation and formal manifest postflight passed")
PY

PREPARED="${PREP_ROOT}/scenarios/hard_r30/seed_11003/prepared"
OUTPUT="${SMOKE_ROOT}/scratch/hard_r30/seed_11003"
[[ ! -e "${SMOKE_ROOT}" ]] || { echo "Smoke root already exists" >&2; exit 2; }
mkdir -p "$(dirname "${OUTPUT}")"
COMMON=(
  --output-dir "${OUTPUT}"
  --source-prepared "${PREPARED}"
  --private-reference "${PREPARED}/private_reference.csv"
  --sentinel-private-reference "${PREPARED}/sentinel_private_reference.csv"
  --scenario-id hard_r30
  --structure hard
  --noise-rate 0.3
  --seed 11003
  --initialization scratch
  --loops 1
  --review-budget 288
  --sentinel-review-budget 72
)
run_smoke_oof() {
  local cohort="$1"
  local target="$2"
  "${PYTHON}" -u "${ENGINE}" \
    --blind-cohort "${cohort}" \
    --sentinel-cohort "${PREPARED}/sentinel_blind_cohort.csv" \
    --split-reference-cohort "${PREPARED}/blind_noisy_cohort.csv" \
    --image-index "${PREP_ROOT}/image_index.csv" \
    --image-root "${IMAGE_ROOT}" \
    --output-dir "${target}" \
    --seed 11003 \
    --initialization scratch \
    --xrv-cache-dir "${XRV_CACHE}" \
    --n-splits 4 \
    --epochs 3 \
    --early-stopping-patience 2 \
    --learning-rate 0.001 \
    --batch-size 32 \
    --num-workers 4 \
    --device cuda
}

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
"${PYTHON}" -u "${DRIVER}" initialize "${COMMON[@]}"
run_smoke_oof "${PREPARED}/blind_noisy_cohort.csv" "${OUTPUT}/loop_00/oof_evidence"
"${PYTHON}" -u "${DRIVER}" select "${COMMON[@]}" --loop-id 1
"${PYTHON}" -u "${DRIVER}" oracle-update "${COMMON[@]}" --loop-id 1
run_smoke_oof "${OUTPUT}/loop_01/cohort_after_action_blind.csv" "${OUTPUT}/loop_01/oof_after_action"
"${PYTHON}" -u "${DRIVER}" finalize-loop "${COMMON[@]}" --loop-id 1
"${PYTHON}" -u "${DRIVER}" evaluate "${COMMON[@]}"

"${PYTHON}" - "${OUTPUT}" <<'PY'
from pathlib import Path
import pandas as pd
import sys

root = Path(sys.argv[1])
history = pd.read_csv(root / "state/review_history_private.csv")
if len(history) != 288 or history["entry_key"].duplicated().any():
    raise RuntimeError("Smoke review history failed")
for loop_id in (0, 1):
    evidence = root / f"loop_{loop_id:02d}" / ("oof_evidence" if loop_id == 0 else "oof_after_action")
    action = pd.read_csv(evidence / "entry_evidence.csv")
    sentinel = pd.read_csv(evidence / "sentinel_entry_evidence.csv")
    if len(action) != 14400 or len(sentinel) != 3600:
        raise RuntimeError("Smoke action/sentinel evidence size failed")
    if set(action.image_id.astype(str)) & set(sentinel.image_id.astype(str)):
        raise RuntimeError("Smoke sentinel leakage detected")
print("GPU smoke postflight passed")
PY
printf 'job_id=%s\n' "${SLURM_JOB_ID:-manual}" > "${SMOKE_ROOT}/.smoke_complete"

FORMAL_JOB_ID="$(sbatch --parsable --array=1-2%2 "${FORMAL}")"
printf 'smoke_job_id=%s\nformal_array_job_id=%s\n' \
  "${SLURM_JOB_ID:-manual}" "${FORMAL_JOB_ID}" > "${PREP_ROOT}/slurm_submission_manifest.txt"
echo "Smoke passed; submitted formal workers 1-2 as ${FORMAL_JOB_ID}; continuing as worker 0"
SELFOPT_WORKER_ID=0 bash "${FORMAL}"
