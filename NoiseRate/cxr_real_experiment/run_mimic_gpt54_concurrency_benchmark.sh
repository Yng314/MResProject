#!/bin/bash
#SBATCH --job-name=gpt54-conc
#SBATCH --output=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/gpt54_conc_%j.out
#SBATCH --error=/vol/gpudata/yz3522-llmtest/MResProject/slurm_logs/gpt54_conc_%j.err
#SBATCH --partition=t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
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
ACTIVE_REVIEW_DIR="${RUN_ROOT}/llm_review"
ACTIVE_RESULTS="${ACTIVE_REVIEW_DIR}/results.csv"
ACTIVE_ERRORS="${ACTIVE_REVIEW_DIR}/errors.csv"
OUTPUT_ROOT="/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/mimic_gpt54_concurrency_benchmark/20260821_v1"
CONCURRENCIES=(10 20 40 80 120)
ROWS_PER_LEVEL=160

EXPECTED_REVIEW_SHA="4c42d95eb13d756ab76da721c67ce3228eb96cea394f6157e311764ccd7205d1"
[[ "$(sha256sum "${REVIEW_SCRIPT}" | cut -d' ' -f1)" == "${EXPECTED_REVIEW_SHA}" ]] || {
  echo "Review script hash mismatch" >&2
  exit 2
}
[[ -s "${EXPANDED}" && -s "${ACTIVE_RESULTS}" ]] || {
  echo "Missing active expansion or successful-result checkpoint" >&2
  exit 2
}

source ~/.llm_review_env
export OPENAI_MODEL="gpt-5.4"
mkdir -p "${OUTPUT_ROOT}" "${PROJECT_DIR}/slurm_logs"

"${PYTHON}" - "${EXPANDED}" "${ACTIVE_RESULTS}" "${ACTIVE_ERRORS}" "${OUTPUT_ROOT}" "${ROWS_PER_LEVEL}" "${CONCURRENCIES[@]}" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

expanded_path = Path(sys.argv[1])
results_path = Path(sys.argv[2])
errors_path = Path(sys.argv[3])
output_root = Path(sys.argv[4])
rows_per_level = int(sys.argv[5])
concurrencies = [int(value) for value in sys.argv[6:]]
manifest_path = output_root / "selection_manifest.csv"
metadata_path = output_root / "selection_metadata.json"

expanded = pd.read_csv(expanded_path)
success = pd.read_csv(results_path, usecols=["entry_key"])
errors = (
    pd.read_csv(errors_path, usecols=["entry_key"])
    if errors_path.is_file() and errors_path.stat().st_size
    else pd.DataFrame(columns=["entry_key"])
)
for frame in (expanded, success, errors):
    frame["entry_key"] = frame["entry_key"].astype(str)

if expanded["entry_key"].duplicated().any() or success["entry_key"].duplicated().any():
    raise SystemExit("Duplicate entry keys in active checkpoint")
blocked = set(success["entry_key"]) | set(errors["entry_key"])
pending = expanded[~expanded["entry_key"].isin(blocked)].copy()
needed = rows_per_level * len(concurrencies)
if len(pending) < needed:
    raise SystemExit(f"Insufficient untouched pending rows: {len(pending)} < {needed}")

selected = pending.iloc[:needed].copy().reset_index(drop=True)
selected["concurrency"] = [
    concurrency
    for concurrency in concurrencies
    for _ in range(rows_per_level)
]
if selected["entry_key"].duplicated().any():
    raise SystemExit("Benchmark selection contains duplicate entry keys")

if manifest_path.exists():
    old = pd.read_csv(manifest_path, usecols=["entry_key", "concurrency"])
    old["entry_key"] = old["entry_key"].astype(str)
    current = selected[["entry_key", "concurrency"]]
    if not old.equals(current):
        raise SystemExit("Existing benchmark selection differs from locked selection")
else:
    selected[["entry_key", "concurrency"]].to_csv(manifest_path, index=False)

for concurrency in concurrencies:
    shard = selected[selected["concurrency"].eq(concurrency)].drop(columns=["concurrency"])
    shard_path = output_root / f"input_c{concurrency}.csv"
    if shard_path.exists():
        existing = pd.read_csv(shard_path, usecols=["entry_key"])
        if existing["entry_key"].astype(str).tolist() != shard["entry_key"].tolist():
            raise SystemExit(f"Existing c{concurrency} shard differs from locked selection")
    else:
        shard.to_csv(shard_path, index=False)

digest = hashlib.sha256("\n".join(selected["entry_key"]).encode()).hexdigest()
metadata = {
    "protocol": "mimic_gpt54_real_entry_concurrency_sweep_v1",
    "rows_per_level": rows_per_level,
    "concurrencies": concurrencies,
    "selection_entry_key_sha256": digest,
    "active_success_rows_at_selection": len(success),
    "active_error_rows_excluded": len(errors),
    "untouched_pending_rows_before_selection": len(pending),
    "retry_limit": 0,
    "request_timeout_seconds": 180,
    "model": "gpt-5.4",
}
metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
print(json.dumps(metadata, indent=2))
PY

for concurrency in "${CONCURRENCIES[@]}"; do
  level_dir="${OUTPUT_ROOT}/c${concurrency}"
  complete_marker="${level_dir}/.complete"
  mkdir -p "${level_dir}"
  if [[ -f "${complete_marker}" ]]; then
    echo "Concurrency ${concurrency} already complete; reusing."
    continue
  fi
  start_epoch="$(date +%s)"
  "${PYTHON}" -u "${REVIEW_SCRIPT}" \
    --input-csv "${OUTPUT_ROOT}/input_c${concurrency}.csv" \
    --report-archive "${REPORT_ARCHIVE}" \
    --output-dir "${level_dir}" \
    --top-k "${ROWS_PER_LEVEL}" \
    --batch-size "${ROWS_PER_LEVEL}" \
    --concurrency "${concurrency}" \
    --retry-limit 0 \
    --llm-mode real \
    --request-timeout-seconds 180 \
    --temperature 0 \
    --max-completion-tokens 400
  end_epoch="$(date +%s)"
  printf '%s\n' "$((end_epoch - start_epoch))" > "${level_dir}/elapsed_seconds.txt"
  touch "${complete_marker}"
done

"${PYTHON}" - "${OUTPUT_ROOT}" "${ACTIVE_RESULTS}" "${ACTIVE_ERRORS}" "${ROWS_PER_LEVEL}" "${CONCURRENCIES[@]}" <<'PY'
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import sys

import pandas as pd

root = Path(sys.argv[1])
active_results = Path(sys.argv[2])
active_errors = Path(sys.argv[3])
rows_per_level = int(sys.argv[4])
concurrencies = [int(value) for value in sys.argv[5:]]


def count_rows(path: Path) -> int:
    if not path.is_file() or not path.stat().st_size:
        return 0
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def error_classes(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.is_file() or not path.stat().st_size:
        return counts
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            message = row.get("error_message", "")
            status = re.search(r"status (\d+)", message)
            if "model_not_found" in message:
                key = "model_not_found"
            elif status:
                key = f"http_{status.group(1)}"
            elif "timeout" in message.lower() or "timed out" in message.lower():
                key = "timeout"
            elif "non-JSON" in message:
                key = "non_json"
            else:
                key = "other"
            counts[key] = counts.get(key, 0) + 1
    return counts


summary_rows = []
for concurrency in concurrencies:
    level = root / f"c{concurrency}"
    elapsed = int((level / "elapsed_seconds.txt").read_text().strip())
    success = count_rows(level / "results.csv")
    errors = count_rows(level / "errors.csv")
    if success + errors != rows_per_level:
        raise SystemExit(
            f"c{concurrency} incomplete: success={success}, errors={errors}, expected={rows_per_level}"
        )
    classes = error_classes(level / "errors.csv")
    summary_rows.append(
        {
            "concurrency": concurrency,
            "input_entries": rows_per_level,
            "success_entries": success,
            "error_entries": errors,
            "success_rate": success / rows_per_level,
            "elapsed_seconds": elapsed,
            "attempted_entries_per_second": rows_per_level / elapsed,
            "successful_entries_per_second": success / elapsed,
            "successful_entries_per_minute": 60 * success / elapsed,
            **{f"errors_{key}": value for key, value in classes.items()},
        }
    )
summary = pd.DataFrame(summary_rows).fillna(0)
summary.to_csv(root / "concurrency_summary.csv", index=False)

with active_results.open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames
    if not fieldnames or "entry_key" not in fieldnames:
        raise SystemExit("Active results schema is invalid")
    merged_rows = list(reader)
seen = {row["entry_key"] for row in merged_rows}
added = 0
for concurrency in concurrencies:
    path = root / f"c{concurrency}" / "results.csv"
    if not path.is_file() or not path.stat().st_size:
        continue
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fieldnames:
            raise SystemExit(f"Result schema mismatch for c{concurrency}")
        for row in reader:
            key = row["entry_key"]
            if key in seen:
                continue
            seen.add(key)
            merged_rows.append(row)
            added += 1

tmp_results = active_results.with_suffix(".csv.concurrency_merge.tmp")
with tmp_results.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(merged_rows)
os.replace(tmp_results, active_results)

if active_errors.is_file() and active_errors.stat().st_size:
    with active_errors.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        error_fields = reader.fieldnames
        error_rows = [row for row in reader if row["entry_key"] not in seen]
    tmp_errors = active_errors.with_suffix(".csv.concurrency_merge.tmp")
    with tmp_errors.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=error_fields)
        writer.writeheader()
        writer.writerows(error_rows)
    os.replace(tmp_errors, active_errors)

merge = {
    "active_success_rows_after_merge": len(merged_rows),
    "benchmark_success_rows_added": added,
    "active_error_rows_after_merge": count_rows(active_errors),
}
(root / "merge_summary.json").write_text(json.dumps(merge, indent=2) + "\n")
(root / ".benchmark_complete").touch()
print(summary.to_string(index=False))
print(json.dumps(merge, indent=2))
PY

echo "Concurrency benchmark completed: ${OUTPUT_ROOT}"
