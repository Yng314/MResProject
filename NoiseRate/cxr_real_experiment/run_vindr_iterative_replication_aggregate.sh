#!/bin/bash
#SBATCH --job-name=vindr-repl-eval
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_repl_eval_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/vindr_repl_eval_%j.err
#SBATCH --partition=a16
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
BASE_PROTOCOL="${ROOT}/cxr_real_experiment/vindr_iterative_oracle_cleaning_protocol_20260805.md"
REPLICATION_PROTOCOL="${ROOT}/cxr_real_experiment/vindr_iterative_oracle_replication_protocol_20260806.md"
OLD_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_iterative_oracle_cleaning/20260805_symmetric_r20_v1"
NEW_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_iterative_oracle_cleaning/20260806_replication_seeds509_701_v1"
COMBINED_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/vindr_iterative_oracle_cleaning/20260806_combined_8seed_v1"
EVALUATION_ROOT="${COMBINED_ROOT}/evaluation_formal_v1"

EXPECTED_PROGRAM_SHA="724e78b2e806c90147f2aac153f7c1561d9fd321aec5573cdc9e789026f22d0c"
EXPECTED_BASE_PROTOCOL_SHA="69f5f0822059c64a0a4a33097d1ac7152d6251f49ba0214d9cb412a294170736"
EXPECTED_REPLICATION_PROTOCOL_SHA="873287d99a76fc95f12959b468dc7dfb1e4d28b24177f41ad296561ebec47118"

for item in \
  "${PROGRAM}:${EXPECTED_PROGRAM_SHA}" \
  "${BASE_PROTOCOL}:${EXPECTED_BASE_PROTOCOL_SHA}" \
  "${REPLICATION_PROTOCOL}:${EXPECTED_REPLICATION_PROTOCOL_SHA}"; do
  path="${item%%:*}"
  expected="${item##*:}"
  if [[ "$(sha256sum "${path}" | cut -d' ' -f1)" != "${expected}" ]]; then
    echo "Hash mismatch: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${COMBINED_ROOT}" ]]; then
  echo "Combined eight-seed output already exists: ${COMBINED_ROOT}" >&2
  exit 2
fi
for seed in 13 42 97 123 211 307; do
  if [[ ! -e "${OLD_ROOT}/seed_${seed}/.seed_evaluation_complete" ]]; then
    echo "Original seed marker is missing: ${seed}" >&2
    exit 2
  fi
done
for seed in 509 701; do
  if [[ ! -e "${NEW_ROOT}/seed_${seed}/.seed_evaluation_complete" ]]; then
    echo "Replication seed marker is missing: ${seed}" >&2
    exit 2
  fi
done

export PYTHONPATH="${ROOT}/cxr_real_experiment:${PYTHONPATH:-}"
export MPLCONFIGDIR="/vol/gpudata/yz3522-llmtest/.cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}" "${COMBINED_ROOT}"
for seed in 13 42 97 123 211 307; do
  ln -s "${OLD_ROOT}/seed_${seed}" "${COMBINED_ROOT}/seed_${seed}"
done
for seed in 509 701; do
  ln -s "${NEW_ROOT}/seed_${seed}" "${COMBINED_ROOT}/seed_${seed}"
done

"${PYTHON}" - "${COMBINED_ROOT}" "${OLD_ROOT}" "${NEW_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

combined, old, new = map(Path, sys.argv[1:])
records = []
for seed in [13, 42, 97, 123, 211, 307, 509, 701]:
    block = "original_six" if seed in [13, 42, 97, 123, 211, 307] else "fixed_replication_two"
    source = (old if block == "original_six" else new) / f"seed_{seed}"
    link = combined / f"seed_{seed}"
    if not link.is_symlink() or link.resolve() != source.resolve():
        raise RuntimeError(f"Combined seed link failed: {seed}")
    summary = source / "seed_evaluation_summary.json"
    digest = hashlib.sha256(summary.read_bytes()).hexdigest()
    records.append({"seed": seed, "analysis_block": block, "source_seed_root": str(source), "seed_summary_sha256": digest})
pd.DataFrame(records).to_csv(combined / "combined_input_manifest.csv", index=False)
(combined / ".combined_inputs_frozen").write_text("complete\n")
print(json.dumps({"combined_inputs": len(records), "replication_seeds": [509, 701]}, indent=2))
PY

echo "VinDr post-result fixed two-seed precision extension"
echo "Job ID: ${SLURM_JOB_ID:-manual}; combined seeds=8"
echo "Started: $(date --iso-8601=seconds)"
"${PYTHON}" -u "${PROGRAM}" aggregate \
  --experiment-root "${COMBINED_ROOT}" \
  --aggregate-output "${EVALUATION_ROOT}" \
  --seeds "13,42,97,123,211,307,509,701"

"${PYTHON}" - "${EVALUATION_ROOT}" <<'PY'
import json
import math
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
summaries = pd.read_csv(root / "seed_summaries.csv")
replication = summaries[summaries["seed"].isin([509, 701])].copy()
if sorted(replication["seed"].tolist()) != [509, 701]:
    raise RuntimeError("Replication block is incomplete in combined summaries")
replication.to_csv(root / "replication_block_seed_summaries.csv", index=False)
records = []
for row in replication.itertuples(index=False):
    records.extend([
        {"seed": int(row.seed), "contrast": "dynamic_minus_frozen_recall_audc", "difference": float(row.dynamic_recall_audc - row.frozen_recall_audc)},
        {"seed": int(row.seed), "contrast": "dynamic_minus_random_recall_audc", "difference": float(row.dynamic_recall_audc - row.random_recall_audc)},
    ])
effects = pd.DataFrame(records)
effects.to_csv(root / "replication_block_effects.csv", index=False)
expected_rows = {
    "all_seed_trajectories.csv": 216,
    "seed_summaries.csv": 8,
    "paired_tests.csv": 2,
    "dynamic_loop_summary.csv": 9,
    "replication_block_seed_summaries.csv": 2,
    "replication_block_effects.csv": 4,
}
for name, expected in expected_rows.items():
    actual = len(pd.read_csv(root / name))
    if actual != expected:
        raise RuntimeError(f"{name} has {actual} rows; expected {expected}")
tests = pd.read_csv(root / "paired_tests.csv")
for value in tests[["mean_difference", "exact_p_value", "holm_adjusted_p_value"]].to_numpy().ravel():
    if not math.isfinite(float(value)):
        raise RuntimeError("Combined paired test contains a non-finite value")
if (tests["holm_adjusted_p_value"] + 1e-15 < tests["exact_p_value"]).any():
    raise RuntimeError("Combined Holm p-value is below its raw p-value")
summary = json.loads((root / "aggregate_summary.json").read_text())
if summary["seeds"] != [13, 42, 97, 123, 211, 307, 509, 701] or summary["loops"] != 8:
    raise RuntimeError("Combined aggregate summary differs from the locked eight seeds")
for name in ["iterative_oracle_summary.png", ".aggregate_complete"]:
    if not (root / name).is_file():
        raise RuntimeError(f"Combined artifact is missing: {name}")
payload = {
    "combined_postflight": "passed",
    "interpretation": "post_result_fixed_size_precision_extension",
    "replication_effects": records,
    "combined_summary": summary,
}
(root / "replication_extension_summary.json").write_text(json.dumps(payload, indent=2))
(root / ".replication_extension_complete").write_text("complete\n")
print(json.dumps(payload, indent=2))
PY

echo "Completed: $(date --iso-8601=seconds)"
