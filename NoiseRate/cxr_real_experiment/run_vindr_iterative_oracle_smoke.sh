#!/bin/bash
#SBATCH --job-name=vindr-iter-smoke
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_iter_smoke_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_iter_smoke_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
PROGRAM="${ROOT}/cxr_real_experiment/vindr_iterative_oracle_cleaning.py"
TEST_PROGRAM="${ROOT}/cxr_real_experiment/test_vindr_iterative_oracle_cleaning.py"
PROTOCOL="${ROOT}/cxr_real_experiment/vindr_iterative_oracle_cleaning_protocol_20260805.md"
ENGINE="${ROOT}/cxr_real_experiment/vindr_mobilenet_oof.py"
ENGINE_TEST="${ROOT}/cxr_real_experiment/test_vindr_mobilenet_oof.py"
BENCHMARK_HELPER="${ROOT}/cxr_real_experiment/vindr_known_gt_cl_benchmark.py"
DATASET_HELPER="${ROOT}/cxr_real_experiment/cxr_real_full_train_eval_cleanlab_xrv12.py"
CL_HELPER="${ROOT}/cxr_real_experiment/cxr_real_noise_validation_smoke.py"
SOURCE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_global_symmetric_noise_mobilenet/20260805_v1"
PREPARED="${SOURCE_ROOT}/scenarios/symmetric_entry_r20/seed_211/prepared"
SOURCE_BLIND="${SOURCE_ROOT}/scenarios/symmetric_entry_r20/seed_211/blind_run"
IMAGE_PARENT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/derived/png224_20260804_v1"
IMAGE_ROOT="${IMAGE_PARENT}/images"
SMOKE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_iterative_oracle_cleaning/smoke_20260805_seed211_loop1_3epoch"

EXPECTED_PROGRAM_SHA="724e78b2e806c90147f2aac153f7c1561d9fd321aec5573cdc9e789026f22d0c"
EXPECTED_TEST_PROGRAM_SHA="3047f53bda9a6428034a66cc1e99d7458408697a143f04a683063f823f62cdff"
EXPECTED_PROTOCOL_SHA="69f5f0822059c64a0a4a33097d1ac7152d6251f49ba0214d9cb412a294170736"
EXPECTED_ENGINE_SHA="9d1585d8d37b7c8043516889b9b3d6ddda984425f910a891d889b0c2ecd9b026"
EXPECTED_ENGINE_TEST_SHA="f6d866ad34f3268af6c1ab854a1e3adf88cd1ed6a6f3e715d993228dc3dacf91"
EXPECTED_BENCHMARK_SHA="f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83"
EXPECTED_DATASET_SHA="f9b1eb32aaa7d6df9e1d1ebd3ba66fd2ef637bde47676fda4c7b629a7c7f452d"
EXPECTED_CL_SHA="62da5b019f9809b4641b58f3c9bda56c1f987057fc765f9a76f32feade7046e0"
EXPECTED_SCENARIO_MANIFEST_SHA="cdff8ca8ec86f0a061f9c4856ed0888573c4defe68a31a1784484f7652eb0f17"
EXPECTED_BLIND_MANIFEST_SHA="201ac89c578a1227a8651d3ed942d1727dd8a543fb72c0a59adcc8c8f023745f"
EXPECTED_IMAGE_INDEX_SHA="645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2"
EXPECTED_SOURCE_COHORT_SHA="1f9b728404106e34f6e1d4fbf98a0fe93f7bd418b808c229efb53fb6c7f51e0a"
EXPECTED_SOURCE_EVIDENCE_SHA="783a918e6f0681d4196ad683afa1ad965d49a4ed54cf7750f7e804776a869e9f"

for item in \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${TEST_PROGRAM}:${EXPECTED_TEST_PROGRAM_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${ENGINE}:${EXPECTED_ENGINE_SHA}" \
  "${ENGINE_TEST}:${EXPECTED_ENGINE_TEST_SHA}" \
  "${BENCHMARK_HELPER}:${EXPECTED_BENCHMARK_SHA}" \
  "${DATASET_HELPER}:${EXPECTED_DATASET_SHA}" \
  "${CL_HELPER}:${EXPECTED_CL_SHA}" \
  "${SOURCE_ROOT}/scenario_manifest.csv:${EXPECTED_SCENARIO_MANIFEST_SHA}" \
  "${SOURCE_ROOT}/blind_run_manifest.csv:${EXPECTED_BLIND_MANIFEST_SHA}" \
  "${SOURCE_ROOT}/image_index.csv:${EXPECTED_IMAGE_INDEX_SHA}" \
  "${PREPARED}/blind_noisy_cohort.csv:${EXPECTED_SOURCE_COHORT_SHA}" \
  "${SOURCE_BLIND}/entry_evidence.csv:${EXPECTED_SOURCE_EVIDENCE_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  if [[ "$(sha256sum "${path}" | cut -d' ' -f1)" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done
for marker in \
  "${SOURCE_ROOT}/.prepare_complete" \
  "${PREPARED}/.prepare_complete" \
  "${SOURCE_BLIND}/.blind_run_complete" \
  "${IMAGE_PARENT}/.conversion_complete"; do
  if [[ ! -e "${marker}" ]]; then
    echo "Required marker is missing: ${marker}" >&2
    exit 2
  fi
done
if [[ -e "${SMOKE_ROOT}" ]]; then
  echo "Smoke output already exists; refusing to overwrite: ${SMOKE_ROOT}" >&2
  exit 2
fi

source /vol/cuda/12.5.0/setup.sh
export PYTHONPATH="${ROOT}/cxr_real_experiment:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}"

COMMON=(
  --output-dir "${SMOKE_ROOT}"
  --source-prepared "${PREPARED}"
  --source-blind-run "${SOURCE_BLIND}"
  --seed 211
  --loops 1
  --top-fraction 0.20
)

echo "VinDr iterative oracle-cleaning smoke"
echo "Job ID: ${SLURM_JOB_ID:-manual}; seed=211; loops=1; epochs=3"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"${PYTHON}" -u "${PROGRAM}" initialize "${COMMON[@]}"
"${PYTHON}" -u "${PROGRAM}" select "${COMMON[@]}" --loop-id 1
"${PYTHON}" -u "${PROGRAM}" oracle-update "${COMMON[@]}" \
  --loop-id 1 --private-reference "${PREPARED}/private_reference.csv"
"${PYTHON}" -u "${ENGINE}" \
  --blind-cohort "${SMOKE_ROOT}/loop_01/cohort_after_action_blind.csv" \
  --image-index "${SOURCE_ROOT}/image_index.csv" \
  --image-root "${IMAGE_ROOT}" \
  --output-dir "${SMOKE_ROOT}/loop_01/oof_after_action" \
  --seed 211 \
  --n-splits 4 \
  --epochs 3 \
  --early-stopping-patience 3 \
  --learning-rate 0.001 \
  --batch-size 32 \
  --num-workers 4 \
  --device cuda
"${PYTHON}" -u "${PROGRAM}" finalize-loop "${COMMON[@]}" --loop-id 1
"${PYTHON}" -u "${PROGRAM}" evaluate "${COMMON[@]}" \
  --private-reference "${PREPARED}/private_reference.csv" \
  --random-replicates 100

"${PYTHON}" - "${SMOKE_ROOT}" "${EXPECTED_ENGINE_SHA}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
expected_engine_sha = sys.argv[2]
loop = root / "loop_01"
selection = json.loads((loop / "selection_manifest_blind.json").read_text())
oracle = json.loads((loop / "oracle_summary_private.json").read_text())
metrics = json.loads((loop / "loop_metrics_private.json").read_text())
engine = json.loads((loop / "oof_after_action/blind_run_summary.json").read_text())
evaluation = json.loads((root / "seed_evaluation_summary.json").read_text())
blind = pd.read_csv(loop / "selected_entries_blind.csv")
history = pd.read_csv(root / "state/review_history_blind.csv")
evidence = pd.read_csv(loop / "oof_after_action/entry_evidence.csv")
training = pd.read_csv(loop / "oof_after_action/training_history.csv")
forbidden = [
    column for column in blind.columns
    if any(token in column.lower() for token in ["clean", "reference", "injected", "true_", "error"])
]
if forbidden:
    raise RuntimeError(f"Private columns leaked into blind selection: {forbidden}")
if not 0 < len(blind) == len(history) == selection["selected_entries"]:
    raise RuntimeError("Smoke selection/history row counts differ")
if history["entry_key"].duplicated().any() or len(evidence) != 18_000:
    raise RuntimeError("Smoke review history or evidence is invalid")
if evidence[["image_id", "label_name"]].duplicated().any():
    raise RuntimeError("Smoke evidence contains duplicate entries")
if set(training["fold_id"]) != {0, 1, 2, 3} or training.groupby("fold_id").size().min() != 3:
    raise RuntimeError("Smoke did not run three epochs in every fold")
if engine["program_sha256"] != expected_engine_sha or engine["epochs_max"] != 3:
    raise RuntimeError("Smoke engine provenance or configuration failed")
if oracle["selected_true_issues"] <= 0 or oracle["true_quality_after_action"] <= 0.80:
    raise RuntimeError("Smoke oracle action did not improve known quality")
if metrics["raw_dqs_after_action"] != 1.0 - metrics["cl_issue_entries_after_action"] / 18_000:
    raise RuntimeError("Smoke post-action DQS is inconsistent")
for field in [
    "final_dynamic_recall", "final_dynamic_quality", "final_dynamic_dqs",
    "dynamic_recall_audc", "frozen_recall_audc", "random_recall_audc",
]:
    if not isinstance(evaluation[field], (int, float)):
        raise RuntimeError(f"Smoke evaluation field is not numeric: {field}")
for marker in [loop / ".selection_complete", loop / ".oracle_update_complete", loop / ".loop_complete", root / ".seed_evaluation_complete"]:
    if not marker.is_file():
        raise RuntimeError(f"Smoke marker missing: {marker}")
payload = {
    "status": "smoke_only_not_formal",
    "selected_entries": len(blind),
    "selected_true_issues": oracle["selected_true_issues"],
    "quality_before": oracle["true_quality_before_action"],
    "quality_after": oracle["true_quality_after_action"],
    "post_action_raw_dqs": metrics["raw_dqs_after_action"],
}
(root / "smoke_postflight_summary.json").write_text(json.dumps(payload, indent=2))
(root / ".smoke_postflight_complete").write_text("complete\n")
print(json.dumps(payload, indent=2))
PY

echo "Completed: $(date --iso-8601=seconds)"
