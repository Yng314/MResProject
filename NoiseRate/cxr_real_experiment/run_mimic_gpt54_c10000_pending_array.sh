#!/bin/bash
#SBATCH --job-name=gpt54-c10k
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/gpt54_c10k_%A_%a.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/gpt54_c10k_%A_%a.err
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --array=0-2%3
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
ACTIVE_REVIEW_DIR="${RUN_ROOT}/llm_review"
ACTIVE_RESULTS="${ACTIVE_REVIEW_DIR}/results.csv"
ACTIVE_ERRORS="${ACTIVE_REVIEW_DIR}/errors.csv"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/mimic_gpt54_concurrency_benchmark/20260821_c10000_pending"
SHARD_ID="${SLURM_ARRAY_TASK_ID:?array task id is required}"
INPUT_CSV="${OUTPUT_ROOT}/input_shard_${SHARD_ID}.csv"
REVIEW_DIR="${OUTPUT_ROOT}/review_shard_${SHARD_ID}"
COMPLETE_MARKER="${OUTPUT_ROOT}/.shard_${SHARD_ID}_complete"

EXPECTED_REVIEW_SHA="4c42d95eb13d756ab76da721c67ce3228eb96cea394f6157e311764ccd7205d1"
[[ "$(sha256sum "${REVIEW_SCRIPT}" | cut -d' ' -f1)" == "${EXPECTED_REVIEW_SHA}" ]] || {
  echo "Review script hash mismatch" >&2
  exit 2
}
[[ -s "${INPUT_CSV}" && -s "${REPORT_ARCHIVE}" && -s "${ACTIVE_RESULTS}" ]] || {
  echo "Missing locked shard, report archive or active results" >&2
  exit 2
}
[[ ! -e "${COMPLETE_MARKER}" && ! -e "${REVIEW_DIR}" ]] || {
  echo "Refusing to overwrite existing shard ${SHARD_ID}" >&2
  exit 2
}

source ~/.llm_review_env
export OPENAI_MODEL="gpt-5.4"
ulimit -n 4096

rows="$(${PYTHON} - "${INPUT_CSV}" <<'PY'
import pandas as pd
import sys
print(len(pd.read_csv(sys.argv[1], usecols=["pool_row_id"])))
PY
)"

echo "shard_id=${SHARD_ID}"
echo "rows=${rows}"
echo "soft_fd_limit=$(ulimit -Sn)"
echo "hard_fd_limit=$(ulimit -Hn)"
echo "review_start=$(date --iso-8601=seconds)"
start_epoch="$(date +%s)"

"${PYTHON}" -u "${REVIEW_SCRIPT}" \
  --input-csv "${INPUT_CSV}" \
  --report-archive "${REPORT_ARCHIVE}" \
  --output-dir "${REVIEW_DIR}" \
  --top-k "${rows}" \
  --batch-size "${rows}" \
  --concurrency "${rows}" \
  --retry-limit 0 \
  --llm-mode real \
  --request-timeout-seconds 180 \
  --temperature 0 \
  --max-completion-tokens 400

end_epoch="$(date +%s)"
printf '%s\n' "$((end_epoch - start_epoch))" > "${REVIEW_DIR}/elapsed_seconds.txt"
touch "${COMPLETE_MARKER}"
echo "review_end=$(date --iso-8601=seconds)"

if [[ "${SHARD_ID}" != "0" ]]; then
  exit 0
fi

for _ in $(seq 1 180); do
  if [[ -f "${OUTPUT_ROOT}/.shard_0_complete" && \
        -f "${OUTPUT_ROOT}/.shard_1_complete" && \
        -f "${OUTPUT_ROOT}/.shard_2_complete" ]]; then
    break
  fi
  sleep 10
done
[[ -f "${OUTPUT_ROOT}/.shard_0_complete" && \
   -f "${OUTPUT_ROOT}/.shard_1_complete" && \
   -f "${OUTPUT_ROOT}/.shard_2_complete" ]] || {
  echo "Timed out waiting for all shard completion markers" >&2
  exit 3
}

"${PYTHON}" - "${OUTPUT_ROOT}" "${ACTIVE_RESULTS}" "${ACTIVE_ERRORS}" <<'PY'
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
active_results = Path(sys.argv[2])
active_errors = Path(sys.argv[3])


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file() or not path.stat().st_size:
        return [], []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


result_fields, merged_results = read_rows(active_results)
if not result_fields or "entry_key" not in result_fields:
    raise SystemExit("Active result schema is invalid")
seen = {row["entry_key"] for row in merged_results}
success_before = len(seen)
shard_summaries = []

for shard_id in range(3):
    review_dir = root / f"review_shard_{shard_id}"
    fields, successes = read_rows(review_dir / "results.csv")
    error_fields, errors = read_rows(review_dir / "errors.csv")
    if fields and set(fields) != set(result_fields):
        raise SystemExit(f"Result schema mismatch in shard {shard_id}")
    added = 0
    for row in successes:
        key = row["entry_key"]
        if key not in seen:
            seen.add(key)
            merged_results.append({field: row.get(field, "") for field in result_fields})
            added += 1
    elapsed = int((review_dir / "elapsed_seconds.txt").read_text().strip())
    shard_summaries.append(
        {
            "shard_id": shard_id,
            "success_entries": len(successes),
            "error_entries": len(errors),
            "successes_added": added,
            "elapsed_seconds": elapsed,
        }
    )

tmp_results = active_results.with_suffix(".csv.c10000_merge.tmp")
with tmp_results.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=result_fields)
    writer.writeheader()
    writer.writerows(merged_results)
os.replace(tmp_results, active_results)

active_error_fields, existing_errors = read_rows(active_errors)
if not active_error_fields:
    for shard_id in range(3):
        active_error_fields, _ = read_rows(root / f"review_shard_{shard_id}" / "errors.csv")
        if active_error_fields:
            break
if not active_error_fields or "entry_key" not in active_error_fields:
    raise SystemExit("Could not determine error schema")

errors_by_key = {
    row["entry_key"]: row for row in existing_errors if row["entry_key"] not in seen
}
for shard_id in range(3):
    fields, errors = read_rows(root / f"review_shard_{shard_id}" / "errors.csv")
    if fields and set(fields) != set(active_error_fields):
        raise SystemExit(f"Error schema mismatch in shard {shard_id}")
    for row in errors:
        if row["entry_key"] not in seen:
            errors_by_key[row["entry_key"]] = {
                field: row.get(field, "") for field in active_error_fields
            }

tmp_errors = active_errors.with_suffix(".csv.c10000_merge.tmp")
with tmp_errors.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=active_error_fields)
    writer.writeheader()
    writer.writerows(errors_by_key.values())
os.replace(tmp_errors, active_errors)

error_classes: dict[str, int] = {}
for row in errors_by_key.values():
    message = row.get("error_message", "")
    status = re.search(r"status (\d+)", message)
    if status:
        key = f"http_{status.group(1)}"
    elif "mismatch_type" in message:
        key = "invalid_mismatch_type"
    elif "timeout" in message.lower() or "timed out" in message.lower():
        key = "timeout"
    else:
        key = "other"
    error_classes[key] = error_classes.get(key, 0) + 1

attempted = sum(row["success_entries"] + row["error_entries"] for row in shard_summaries)
added = len(seen) - success_before
summary = {
    "requested_total_concurrency": 10000,
    "actual_pending_entries_and_max_concurrency": attempted,
    "active_success_rows_before": success_before,
    "successful_entries_added": added,
    "active_success_rows_after": len(seen),
    "remaining_error_entries": len(errors_by_key),
    "success_rate": added / attempted if attempted else None,
    "wall_time_seconds": max(row["elapsed_seconds"] for row in shard_summaries),
    "successful_entries_per_minute": (
        60 * added / max(row["elapsed_seconds"] for row in shard_summaries)
        if attempted else None
    ),
    "error_classes": error_classes,
    "shards": shard_summaries,
    "merged_into_main_checkpoint": True,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(root / ".complete").touch()
print(json.dumps(summary, indent=2))
PY
