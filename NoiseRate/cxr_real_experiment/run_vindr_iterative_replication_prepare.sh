#!/bin/bash
#SBATCH --job-name=vindr-repl-prepare
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_repl_prepare_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_repl_prepare_%j.err
#SBATCH --partition=a16
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

ROOT="/vol/gpudata/yz3522-llmtest/MResProject/NoiseRate"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
PROGRAM="${ROOT}/cxr_real_experiment/vindr_global_symmetric_noise.py"
TEST_FILE="${ROOT}/cxr_real_experiment/test_vindr_global_symmetric_noise.py"
BASE_PROTOCOL="${ROOT}/cxr_real_experiment/vindr_global_symmetric_noise_protocol_20260805.md"
REPLICATION_PROTOCOL="${ROOT}/cxr_real_experiment/vindr_iterative_oracle_replication_protocol_20260806.md"
DATA_ROOT="/vol/bitbucket/yz3522/datasets/vindr-cxr/1.0.0"
PARENT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_known_gt_cl/20260804_v1"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_global_symmetric_noise_mobilenet/20260806_replication_seeds509_701_v1"

EXPECTED_PROGRAM_SHA="80316491dd61e7f9aa13d52a44bf76448968889bf84a2a33e7aa2a1b70431404"
EXPECTED_TEST_SHA="4e26804183020f650dd9e74cb35ebaf4bb06512dff4bf93a6d7f33c900ca79fb"
EXPECTED_BASE_PROTOCOL_SHA="033c08d7a93eb06076f35245c7bb2fd36753abf206970cf6e6e78ab87fd4d3da"
EXPECTED_REPLICATION_PROTOCOL_SHA="873287d99a76fc95f12959b468dc7dfb1e4d28b24177f41ad296561ebec47118"
EXPECTED_LABELS_SHA="9874c665991c5098db571082f9d9d096fe80c40ac3c2b2007a76d8d226c87a02"
EXPECTED_PARENT_INDEX_SHA="645f3c48ee1414e97476a3462ca038c3e2674a3df320b9f460ed246052b924e2"

for item in \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${TEST_FILE}:${EXPECTED_TEST_SHA}" \
  "${BASE_PROTOCOL}:${EXPECTED_BASE_PROTOCOL_SHA}" \
  "${REPLICATION_PROTOCOL}:${EXPECTED_REPLICATION_PROTOCOL_SHA}" \
  "${DATA_ROOT}/annotations/image_labels_test.csv:${EXPECTED_LABELS_SHA}" \
  "${PARENT_ROOT}/image_index.csv:${EXPECTED_PARENT_INDEX_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  if [[ "$(sha256sum "${path}" | cut -d' ' -f1)" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done
for marker in "${DATA_ROOT}/.download_verified" "${PARENT_ROOT}/.prepare_complete"; do
  if [[ ! -e "${marker}" ]]; then
    echo "Required marker is missing: ${marker}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Replication source output already exists: ${OUTPUT_ROOT}" >&2
  exit 2
fi
for seed in 509 701; do
  if find /vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment \
      -type d -name "seed_${seed}" -print -quit 2>/dev/null | grep -q .; then
    echo "Seed ${seed} appeared in a result directory after protocol freeze" >&2
    exit 2
  fi
done

export PYTHONPATH="${ROOT}/cxr_real_experiment:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

echo "VinDr two-seed replication preparation"
echo "Job ID: ${SLURM_JOB_ID:-manual}; seeds=509,701"
echo "Started: $(date --iso-8601=seconds)"
"${PYTHON}" -u "${PROGRAM}" prepare \
  --labels-csv "${DATA_ROOT}/annotations/image_labels_test.csv" \
  --parent-image-index "${PARENT_ROOT}/image_index.csv" \
  --output-root "${OUTPUT_ROOT}" \
  --seeds "509,701" \
  --rates "0.10,0.20,0.30" \
  --n-splits 4

"${PYTHON}" - "${OUTPUT_ROOT}" "${EXPECTED_PROGRAM_SHA}" <<'PY'
import json
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
expected_program_sha = sys.argv[2]
manifest = pd.read_csv(root / "scenario_manifest.csv")
blind_manifest = pd.read_csv(root / "blind_run_manifest.csv")
seed_manifest = pd.read_csv(root / "seed_manifest.csv")
if len(manifest) != 8 or manifest["array_index"].tolist() != list(range(8)):
    raise RuntimeError("Replication manifest is not the ordered eight-run grid")
if manifest[["scenario_id", "seed"]].duplicated().any():
    raise RuntimeError("Replication manifest contains duplicate runs")
if sorted(seed_manifest["seed"].tolist()) != [509, 701]:
    raise RuntimeError("Replication seed manifest differs from the frozen seeds")
if set(seed_manifest["seed_block"]) != {"new_replication"}:
    raise RuntimeError("Replication seeds are not marked new_replication")
expected_scenarios = {"clean": 0, "symmetric_entry_r10": 1800, "symmetric_entry_r20": 3600, "symmetric_entry_r30": 5400}
if set(manifest["scenario_id"]) != set(expected_scenarios):
    raise RuntimeError("Replication scenario grid differs from the parent experiment")
expected_blind_columns = {
    "array_index", "scenario_index", "scenario_id", "regime", "noise_rate",
    "rate_percent", "seed", "seed_block", "prepared_relpath", "blind_run_relpath",
}
if len(blind_manifest) != 8 or set(blind_manifest.columns) != expected_blind_columns:
    raise RuntimeError("Replication blind manifest schema or rows are invalid")
forbidden = [column for column in blind_manifest if any(token in column.lower() for token in ["clean", "reference", "injected", "true_", "error"])]
if forbidden:
    raise RuntimeError(f"Outcome fields leaked into blind manifest: {forbidden}")
for row in manifest.itertuples(index=False):
    if int(row.injected_errors) != expected_scenarios[row.scenario_id]:
        raise RuntimeError(f"Injected-error count differs: {row.scenario_id}, seed={row.seed}")
    if not np.isclose(float(row.true_quality), 1.0 - float(row.noise_rate)):
        raise RuntimeError("Known quality differs from the locked anchor")
    for field in [row.prepared_relpath, row.blind_run_relpath]:
        path = PurePosixPath(str(field))
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError("Unsafe relative path in replication manifest")
    prepared = root / row.prepared_relpath
    blind = pd.read_csv(prepared / "blind_noisy_cohort.csv")
    private = pd.read_csv(prepared / "private_reference.csv")
    counts = pd.read_csv(prepared / "corruption_counts.csv")
    support = pd.read_csv(prepared / "fold_support.csv")
    if len(blind) != 3000 or len(private) != 18000 or len(counts) != 6:
        raise RuntimeError("Replication prepared row counts failed")
    if int(private["injected_error"].sum()) != expected_scenarios[row.scenario_id]:
        raise RuntimeError("Replication private error count failed")
    if support[["positive_entries", "negative_entries"]].min().min() < 5:
        raise RuntimeError("Replication fold support failed")
    if (root / row.blind_run_relpath).exists():
        raise RuntimeError("Prepare stage created a blind-run outcome directory")
summary = json.loads((root / "prepare_summary.json").read_text())
if summary["seeds"] != [509, 701] or summary["blind_runs"] != 8 or summary["program_sha256"] != expected_program_sha:
    raise RuntimeError("Replication prepare summary failed")
if not (root / ".prepare_complete").is_file():
    raise RuntimeError("Replication prepare marker is missing")
print(json.dumps({"replication_prepare_postflight": "passed", "seeds": [509, 701], "prepared_runs": 8}, indent=2))
PY

echo "Completed: $(date --iso-8601=seconds)"
