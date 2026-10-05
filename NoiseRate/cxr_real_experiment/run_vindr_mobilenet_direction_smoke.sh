#!/bin/bash
#SBATCH --job-name=vindr-mnet-smoke
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_mnet_smoke_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_mnet_smoke_%j.err
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
PROTOCOL="${ROOT}/cxr_real_experiment/vindr_mobilenet_direction_sensitivity_protocol_20260805.md"
TEST_PROGRAM="${ROOT}/cxr_real_experiment/test_vindr_mobilenet_oof.py"
DATASET_HELPER="${ROOT}/cxr_real_experiment/cxr_real_full_train_eval_cleanlab_xrv12.py"
CL_HELPER="${ROOT}/cxr_real_experiment/cxr_real_noise_validation_smoke.py"
BENCHMARK_HELPER="${ROOT}/cxr_real_experiment/vindr_known_gt_cl_benchmark.py"
EVALUATOR="${ROOT}/cxr_real_experiment/vindr_noise_direction_sensitivity.py"
SOURCE_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_noise_direction_sensitivity/20260804_v2"
IMAGE_PARENT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0/derived/png224_20260804_v1"
IMAGE_ROOT="${IMAGE_PARENT}/images"
PREPARED="${SOURCE_ROOT}/scenarios/fn_only_r30/seed_211/prepared"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_noise_direction_sensitivity_mobilenet/20260805_v1"
SMOKE_ROOT="${OUTPUT_ROOT}/smoke_fn_only_r30_seed211_v2"

EXPECTED_ENGINE_SHA="9d1585d8d37b7c8043516889b9b3d6ddda984425f910a891d889b0c2ecd9b026"
EXPECTED_PROTOCOL_SHA="edb164ded4e17bc387a7822a121d22c62ddbb59178ec806dde22f84849189ffe"
EXPECTED_TEST_SHA="f6d866ad34f3268af6c1ab854a1e3adf88cd1ed6a6f3e715d993228dc3dacf91"
EXPECTED_DATASET_HELPER_SHA="f9b1eb32aaa7d6df9e1d1ebd3ba66fd2ef637bde47676fda4c7b629a7c7f452d"
EXPECTED_CL_HELPER_SHA="62da5b019f9809b4641b58f3c9bda56c1f987057fc765f9a76f32feade7046e0"
EXPECTED_BENCHMARK_HELPER_SHA="f724a9ee6fd6ce6f5b139a9aef56e5f137843bc0a96afcd5ab6fa5ba34edaa83"
EXPECTED_EVALUATOR_SHA="eabe258dd8ce0b860d458c451cd4064514bc1b7a8fa76679347b6c532602390e"
EXPECTED_BLIND_MANIFEST_SHA="a17fba9c02c8a8c9428c1cffa2afee3261980e3c59a4905a68be1bbb8d0da6c4"
EXPECTED_IMAGE_INDEX_SHA="645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2"
EXPECTED_BLIND_COHORT_SHA="962d84307f7f74d267f420dc73a331b5452aaf3a50e9ee2a8bb1bba9ad60dc26"

for item in \
  "${ENGINE}:${EXPECTED_ENGINE_SHA}" \
  "${PROTOCOL}:${EXPECTED_PROTOCOL_SHA}" \
  "${TEST_PROGRAM}:${EXPECTED_TEST_SHA}" \
  "${DATASET_HELPER}:${EXPECTED_DATASET_HELPER_SHA}" \
  "${CL_HELPER}:${EXPECTED_CL_HELPER_SHA}" \
  "${BENCHMARK_HELPER}:${EXPECTED_BENCHMARK_HELPER_SHA}" \
  "${EVALUATOR}:${EXPECTED_EVALUATOR_SHA}" \
  "${SOURCE_ROOT}/blind_run_manifest.csv:${EXPECTED_BLIND_MANIFEST_SHA}" \
  "${SOURCE_ROOT}/image_index.csv:${EXPECTED_IMAGE_INDEX_SHA}" \
  "${PREPARED}/blind_noisy_cohort.csv:${EXPECTED_BLIND_COHORT_SHA}"; do
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

mkdir -p "${OUTPUT_ROOT}"
source /vol/cuda/12.5.0/setup.sh
export PYTHONPATH="${ROOT}/cxr_real_experiment:${PYTHONPATH:-}"
export XDG_CACHE_HOME="/vol/gpudata/yz3522-llmtest/.cache"
export TORCH_HOME="/vol/gpudata/yz3522-llmtest/.cache/torch"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}"

"${PYTHON}" - "${SOURCE_ROOT}/blind_run_manifest.csv" "${PREPARED}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

manifest_path = Path(sys.argv[1])
prepared = Path(sys.argv[2])
manifest = pd.read_csv(manifest_path)
forbidden = [
    column for column in manifest.columns
    if any(token in column.lower() for token in ["clean", "reference", "injected", "true_", "error"])
]
if forbidden:
    raise RuntimeError(f"Outcome columns leaked into blind manifest: {forbidden}")
row = manifest[(manifest["scenario_id"] == "fn_only_r30") & (manifest["seed"] == 211)]
if len(row) != 1:
    raise RuntimeError("Smoke target is not unique in blind manifest")
expected = manifest_path.parent / row.iloc[0]["prepared_relpath"]
if expected.resolve() != prepared.resolve():
    raise RuntimeError("Smoke prepared path does not match blind manifest")
PY

echo "VinDr MobileNet direction-sensitivity smoke"
echo "Job ID: ${SLURM_JOB_ID:-manual}; scenario=fn_only_r30; seed=211"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"${PYTHON}" -u "${ENGINE}" \
  --blind-cohort "${PREPARED}/blind_noisy_cohort.csv" \
  --image-index "${SOURCE_ROOT}/image_index.csv" \
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
from vindr_noise_direction_sensitivity import bootstrap_scenario, merge_private, scenario_metrics

prepared = Path(sys.argv[2])
smoke = Path(sys.argv[3])
expected_engine_sha = sys.argv[4]
expected_image_index_sha = sys.argv[5]

summary = json.loads((smoke / "blind_run_summary.json").read_text())
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
    raise RuntimeError("Smoke training history does not contain three epochs for every fold")
if set(support["fold_id"]) != {0, 1, 2, 3}:
    raise RuntimeError("Smoke fold support is incomplete")
if summary["seed"] != 211 or summary["entries"] != 18_000 or not summary["outcome_blind"]:
    raise RuntimeError("Smoke blind-run summary is invalid")
if summary["program_sha256"] != expected_engine_sha:
    raise RuntimeError("Smoke engine provenance failed")
if summary["image_index_sha256"] != expected_image_index_sha:
    raise RuntimeError("Smoke image-index provenance failed")
if summary["epochs_max"] != 3 or summary["batch_size"] != 32:
    raise RuntimeError("Smoke training configuration differs from the locked invocation")
if not (smoke / ".blind_run_complete").is_file():
    raise RuntimeError("Smoke blind completion marker is missing")

merged = merge_private(prepared, smoke)
metadata = {
    "scenario_id": "fn_only_r30",
    "regime": "fn_only",
    "noise_rate": 0.30,
    "rate_percent": 30,
    "seed": 211,
    "seed_block": "new_replication",
}
overall, budgets, labels, aurocs, quality = scenario_metrics(merged, metadata)
bootstrap = bootstrap_scenario(merged, metadata, 10)
if len(overall) != 3 or len(budgets) != 15 or len(labels) != 54 or len(aurocs) != 6:
    raise RuntimeError("Smoke private metric row counts are incomplete")
if quality["true_errors"] != 560 or not math.isnan(quality["hard_recall_0_to_1"]):
    raise RuntimeError("Smoke private direction scope is wrong")
if not math.isfinite(quality["hard_recall_1_to_0"]):
    raise RuntimeError("Smoke false-negative recall is not finite")
if len(bootstrap) != 10 or bootstrap["hard_recall_1_to_0"].isna().any():
    raise RuntimeError("Smoke bootstrap path failed")

payload = {
    "status": "smoke_only_not_formal",
    "scenario_id": "fn_only_r30",
    "seed": 211,
    "true_errors": quality["true_errors"],
    "cl_issue_entries": quality["cl_issue_entries"],
    "hard_recall_1_to_0": quality["hard_recall_1_to_0"],
    "clean_reference_macro_auroc": float(pd.DataFrame(aurocs)["clean_label_auroc"].mean()),
}
(smoke / "smoke_private_metrics.json").write_text(json.dumps(payload, indent=2))
(smoke / ".smoke_postflight_complete").write_text("complete\n")
print(json.dumps(payload, indent=2))
PY

echo "Completed: $(date --iso-8601=seconds)"
