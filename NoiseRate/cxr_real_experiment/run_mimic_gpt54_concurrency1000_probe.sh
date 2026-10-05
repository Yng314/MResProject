#!/bin/bash
#SBATCH --job-name=gpt54-c1000
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/gpt54_c1000_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/gpt54_c1000_%j.err
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yz3522@ic.ac.uk

set -euo pipefail
umask 022

PROJECT_DIR="/vol/gpudata/yz3522-llmtest/MResProject"
SCRIPT_DIR="${PROJECT_DIR}/NoiseRate/cxr_real_experiment"
PYTHON="/vol/gpudata/yz3522-llmtest/venv/bin/python"
REVIEW_SCRIPT="${SCRIPT_DIR}/run_entry_level_llm_review.py"
REPORT_ARCHIVE="${PROJECT_DIR}/MedSoul/datasets/mimic-cxr-reports.tar.gz"
RUN_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/results_mobilenetv3_full_issue_pool_no_repeat_gpt54/20260820_seed13_v1/seed_13/llm_refine/loop_01"
EXPANDED="${RUN_ROOT}/sample_top_fraction_expanded_entries.csv"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/mimic_gpt54_concurrency_benchmark/20260821_c1000_probe"
INPUT_CSV="${OUTPUT_ROOT}/input_c1000.csv"
REVIEW_DIR="${OUTPUT_ROOT}/review"
ROWS=1000
CONCURRENCY=1000

EXPECTED_REVIEW_SHA="4c42d95eb13d756ab76da721c67ce3228eb96cea394f6157e311764ccd7205d1"
[[ "$(sha256sum "${REVIEW_SCRIPT}" | cut -d' ' -f1)" == "${EXPECTED_REVIEW_SHA}" ]] || {
  echo "Review script hash mismatch" >&2
  exit 2
}
[[ -s "${EXPANDED}" && -s "${REPORT_ARCHIVE}" ]] || {
  echo "Missing expanded entries or report archive" >&2
  exit 2
}
[[ ! -e "${OUTPUT_ROOT}" ]] || {
  echo "Refusing to overwrite existing probe output: ${OUTPUT_ROOT}" >&2
  exit 2
}

mkdir -p "${OUTPUT_ROOT}" "${PROJECT_DIR}/slurm_logs"
source ~/.llm_review_env
export OPENAI_MODEL="gpt-5.4"

# One socket per in-flight request requires a descriptor limit above 1000.
ulimit -n 4096

"${PYTHON}" - "${EXPANDED}" "${INPUT_CSV}" "${ROWS}" <<'PY'
from pathlib import Path
import sys

import pandas as pd

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
rows = int(sys.argv[3])

frame = pd.read_csv(source)
if len(frame) < rows:
    raise SystemExit(f"Insufficient real entries: {len(frame)} < {rows}")
selected = frame.head(rows).copy()
keys = selected["pool_row_id"].astype(str) + "::" + selected["label_index"].astype(str)
if keys.duplicated().any():
    raise SystemExit("Probe input contains duplicate entry keys")
selected.to_csv(destination, index=False)
print(f"Frozen {len(selected)} real entries for the concurrency-1000 probe")
PY

echo "soft_fd_limit=$(ulimit -Sn)"
echo "hard_fd_limit=$(ulimit -Hn)"
echo "probe_start=$(date --iso-8601=seconds)"
start_epoch="$(date +%s)"

"${PYTHON}" -u "${REVIEW_SCRIPT}" \
  --input-csv "${INPUT_CSV}" \
  --report-archive "${REPORT_ARCHIVE}" \
  --output-dir "${REVIEW_DIR}" \
  --top-k "${ROWS}" \
  --batch-size "${ROWS}" \
  --concurrency "${CONCURRENCY}" \
  --retry-limit 0 \
  --llm-mode real \
  --request-timeout-seconds 180 \
  --temperature 0 \
  --max-completion-tokens 400

end_epoch="$(date +%s)"
elapsed="$((end_epoch - start_epoch))"
printf '%s\n' "${elapsed}" > "${OUTPUT_ROOT}/elapsed_seconds.txt"

"${PYTHON}" - "${REVIEW_DIR}" "${OUTPUT_ROOT}" "${ROWS}" "${elapsed}" <<'PY'
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import sys

review_dir = Path(sys.argv[1])
output_root = Path(sys.argv[2])
expected = int(sys.argv[3])
elapsed = int(sys.argv[4])


def rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


successes = rows(review_dir / "results.csv")
errors = rows(review_dir / "errors.csv")
if len(successes) + len(errors) != expected:
    raise SystemExit(
        f"Incomplete probe accounting: {len(successes)} + {len(errors)} != {expected}"
    )

error_classes: dict[str, int] = {}
for row in errors:
    message = row.get("error_message", "")
    status = re.search(r"status (\d+)", message)
    if "Too many open files" in message:
        key = "too_many_open_files"
    elif status:
        key = f"http_{status.group(1)}"
    elif "timeout" in message.lower() or "timed out" in message.lower():
        key = "timeout"
    else:
        key = "other"
    error_classes[key] = error_classes.get(key, 0) + 1

summary = {
    "concurrency": 1000,
    "input_entries": expected,
    "success_entries": len(successes),
    "error_entries": len(errors),
    "success_rate": len(successes) / expected,
    "elapsed_seconds": elapsed,
    "successful_entries_per_minute": 60 * len(successes) / elapsed if elapsed else None,
    "error_classes": error_classes,
    "merged_into_main_checkpoint": False,
}
(output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(output_root / ".complete").touch()
print(json.dumps(summary, indent=2))
PY

echo "probe_end=$(date --iso-8601=seconds)"
