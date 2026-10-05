#!/bin/bash
#SBATCH --job-name=vindr-global-smoke
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_global_smoke_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_global_smoke_%j.err
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
ENGINE="${ROOT}/cxr_real_experiment/vindr_mobilenet_oof.py"
PROGRAM="${ROOT}/cxr_real_experiment/vindr_global_symmetric_noise.py"
PROTOCOL="${ROOT}/cxr_real_experiment/vindr_global_symmetric_noise_protocol_20260805.md"
TEST_PROGRAM="${ROOT}/cxr_real_experiment/test_vindr_mobilenet_oof.py"
TEST_GLOBAL="${ROOT}/cxr_real_experiment/test_vindr_global_symmetric_noise.py"
DATASET_HELPER="${ROOT}/cxr_real_experiment/cxr_real_full_train_eval_cleanlab_xrv12.py"
CL_HELPER="${ROOT}/cxr_real_experiment/cxr_real_noise_validation_smoke.py"
BENCHMARK_HELPER="${ROOT}/cxr_real_experiment/vindr_known_gt_cl_benchmark.py"
DIRECTION_HELPER="${ROOT}/cxr_real_experiment/vindr_noise_direction_sensitivity.py"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_global_symmetric_noise_mobilenet/20260805_v1"
IMAGE_PARENT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/derived/png224_20260804_v1"
IMAGE_ROOT="${IMAGE_PARENT}/images"
PREPARED="${OUTPUT_ROOT}/scenarios/symmetric_entry_r30/seed_211/prepared"
SMOKE_ROOT="${OUTPUT_ROOT}/smoke_symmetric_entry_r30_seed211"

EXPECTED_ENGINE_SHA="9d1585d8d37b7c8043516889b9b3d6ddda984425f910a891d889b0c2ecd9b026"
EXPECTED_PROGRAM_SHA="80316491dd61e7f9aa13d52a44bf76448968889bf84a2a33e7aa2a1b70431404"
EXPECTED_PROTOCOL_SHA="033c08d7a93eb06076f35245c7bb2fd36753abf206970cf6e6e78ab87fd4d3da"
EXPECTED_TEST_PROGRAM_SHA="f6d866ad34f3268af6c1ab854a1e3adf88cd1ed6a6f3e715d993228dc3dacf91"
EXPECTED_TEST_GLOBAL_SHA="4e26804183020f650dd9e74cb35ebaf4bb06512dff4bf93a6d7f33c900ca79fb"
EXPECTED_DATASET_HELPER_SHA="f9b1eb32aaa7d6df9e1d1ebd3ba66fd2ef637bde47676fda4c7b629a7c7f452d"
EXPECTED_CL_HELPER_SHA="62da5b019f9809b4641b58f3c9bda56c1f987057fc765f9a76f32feade7046e0"
EXPECTED_BENCHMARK_HELPER_SHA="f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83"
EXPECTED_DIRECTION_HELPER_SHA="eabe258dd8ce0b860d458c451cd4064514bc1b7a8fa76679347b6c532602390e"
EXPECTED_BLIND_MANIFEST_SHA="201ac89c578a1227a8651d3ed942d1727dd8a543fb72c0a59adcc8c8f023745f"
EXPECTED_IMAGE_INDEX_SHA="645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2"
EXPECTED_BLIND_COHORT_SHA="473cc745085455e126b65522b409163dbd7dc905f01bc19d177fc5ff64847c54"

for item in \
  "${ENGINE}:${EXPECTED_ENGINE_SHA}" \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${TEST_PROGRAM}:${EXPECTED_TEST_PROGRAM_SHA}" \
  "${TEST_GLOBAL}:${EXPECTED_TEST_GLOBAL_SHA}" \
  "${DATASET_HELPER}:${EXPECTED_DATASET_HELPER_SHA}" \
  "${CL_HELPER}:${EXPECTED_CL_HELPER_SHA}" \
  "${BENCHMARK_HELPER}:${EXPECTED_BENCHMARK_HELPER_SHA}" \
  "${DIRECTION_HELPER}:${EXPECTED_DIRECTION_HELPER_SHA}" \
  "${OUTPUT_ROOT}/blind_run_manifest.csv:${EXPECTED_BLIND_MANIFEST_SHA}" \
  "${OUTPUT_ROOT}/image_index.csv:${EXPECTED_IMAGE_INDEX_SHA}" \
  "${PREPARED}/blind_noisy_cohort.csv:${EXPECTED_BLIND_COHORT_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  if [[ "$(sha256sum "${path}" | cut -d' ' -f1)" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done
for marker in \
  "${OUTPUT_ROOT}/.prepare_complete" \
  "${PREPARED}/.prepare_complete" \
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

echo "VinDr exact global symmetric-noise MobileNet smoke"
echo "Job ID: ${SLURM_JOB_ID:-manual}; scenario=symmetric_entry_r30; seed=211"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"${PYTHON}" -u "${ENGINE}" \
  --blind-cohort "${PREPARED}/blind_noisy_cohort.csv" \
  --image-index "${OUTPUT_ROOT}/image_index.csv" \
  --image-root "${IMAGE_ROOT}" \
  --output-dir "${SMOKE_ROOT}" \
  --seed 211 \
  --n-splits 4 \
  --epochs 3 \
  --early-stopping-patience 3 \
  --learning-rate 0.001 \
  --batch-size 32 \
  --num-workers 4 \
  --device cuda

"${PYTHON}" - "${ROOT}" "${PREPARED}" "${SMOKE_ROOT}" "${EXPECTED_ENGINE_SHA}" "${EXPECTED_IMAGE_INDEX_SHA}" <<'PY'
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(sys.argv[1]) / "cxr_real_experiment"))
from vindr_global_symmetric_noise import bootstrap_scenario, merge_private, scenario_metrics

prepared = Path(sys.argv[2])
smoke = Path(sys.argv[3])
expected_engine_sha = sys.argv[4]
expected_image_index_sha = sys.argv[5]
summary = json.loads((smoke / "blind_run_summary.json").read_text())
prepare_summary = json.loads((prepared / "prepare_summary.json").read_text())
evidence = pd.read_csv(smoke / "entry_evidence.csv")
oof = pd.read_csv(smoke / "oof_predictions.csv")
history = pd.read_csv(smoke / "training_history.csv")
support = pd.read_csv(smoke / "fold_support.csv")
if len(evidence) != 18_000 or evidence[["image_id", "label_name"]].duplicated().any():
    raise RuntimeError("Smoke entry-evidence postflight failed")
if len(oof) != 3_000 or oof["image_id"].nunique() != 3_000:
    raise RuntimeError("Smoke OOF row-count postflight failed")
probability_columns = [column for column in oof.columns if column.startswith("probability__")]
if len(probability_columns) != 6 or not np.isfinite(oof[probability_columns].to_numpy()).all():
    raise RuntimeError("Smoke OOF probabilities are incomplete")
for frame, name in [(evidence.drop(columns=["cl_issue"], errors="ignore"), "evidence"), (oof, "OOF")]:
    forbidden = [
        column for column in frame.columns
        if any(token in column.lower() for token in ["clean", "reference", "injected", "true_", "error"])
    ]
    if forbidden:
        raise RuntimeError(f"Outcome columns leaked into {name}: {forbidden}")
if set(history["fold_id"]) != {0, 1, 2, 3} or history.groupby("fold_id").size().min() != 3:
    raise RuntimeError("Smoke history does not contain three epochs for every fold")
if set(support["fold_id"]) != {0, 1, 2, 3}:
    raise RuntimeError("Smoke fold support is incomplete")
if summary["seed"] != 211 or summary["entries"] != 18_000 or not summary["outcome_blind"]:
    raise RuntimeError("Smoke blind-run summary is invalid")
if summary["program_sha256"] != expected_engine_sha or summary["image_index_sha256"] != expected_image_index_sha:
    raise RuntimeError("Smoke engine or image-index provenance failed")
if summary["blind_cohort_sha256"] != prepare_summary["blind_cohort_sha256"]:
    raise RuntimeError("Smoke blind cohort differs from prepared input")
if summary["epochs_max"] != 3 or summary["batch_size"] != 32:
    raise RuntimeError("Smoke training configuration differs from the locked invocation")
if not (smoke / ".blind_run_complete").is_file():
    raise RuntimeError("Smoke blind marker is missing")

merged = merge_private(prepared, smoke)
metadata = {
    "scenario_id": "symmetric_entry_r30",
    "regime": "symmetric_entry",
    "noise_rate": 0.30,
    "rate_percent": 30,
    "seed": 211,
    "seed_block": "new_replication",
}
overall, budgets, labels, aurocs, quality = scenario_metrics(merged, metadata)
bootstrap = bootstrap_scenario(merged, metadata, 10)
if len(overall) != 3 or len(budgets) != 15 or len(labels) != 54 or len(aurocs) != 6:
    raise RuntimeError("Smoke private metric row counts are incomplete")
if quality["true_errors"] != 5_400 or not math.isfinite(quality["hard_recall_0_to_1"]) or not math.isfinite(quality["hard_recall_1_to_0"]):
    raise RuntimeError("Smoke known truth or direction metrics are invalid")
if len(bootstrap) != 10 or bootstrap["hard_recall"].isna().any():
    raise RuntimeError("Smoke bootstrap path failed")

payload = {
    "status": "smoke_only_not_formal",
    "scenario_id": "symmetric_entry_r30",
    "seed": 211,
    "true_errors": quality["true_errors"],
    "true_quality": quality["true_quality"],
    "cl_issue_entries": quality["cl_issue_entries"],
    "hard_recall": quality["hard_recall"],
    "clean_reference_macro_auroc": float(pd.DataFrame(aurocs)["clean_label_auroc"].mean()),
}
(smoke / "smoke_private_metrics.json").write_text(json.dumps(payload, indent=2))
(smoke / ".smoke_postflight_complete").write_text("complete\n")
print(json.dumps(payload, indent=2))
PY

echo "Completed: $(date --iso-8601=seconds)"
