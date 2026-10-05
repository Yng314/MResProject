#!/usr/bin/env python3
"""Run one resumable LLM-review pass across local worker processes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

from run_entry_level_llm_review import entry_key, filter_and_rank_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--report-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-script", type=Path, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--total-concurrency", type=int, required=True)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--retry-limit", type=int, default=0)
    parser.add_argument("--llm-mode", choices=["mock", "real"], default="real")
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=400)
    parser.add_argument("--mock-latency-ms", type=int, default=0)
    parser.add_argument("--mock-fail-once-indexes", default="")
    parser.add_argument("--mock-always-fail-indexes", default="")
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file() or not path.stat().st_size:
        return [], []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def atomic_write_rows(
    path: Path, fieldnames: list[str], rows: list[dict[str, str]], suffix: str
) -> None:
    temporary = path.with_suffix(path.suffix + suffix)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fieldnames} for row in rows
        )
    os.replace(temporary, path)


def selected_with_keys(input_csv: Path, top_k: int) -> pd.DataFrame:
    selected = filter_and_rank_entries(pd.read_csv(input_csv), top_k=top_k).copy()
    computed = selected.apply(entry_key, axis=1).astype(str)
    if "entry_key" in selected.columns:
        existing = selected["entry_key"].astype(str)
        if not existing.equals(computed):
            raise RuntimeError("Input entry_key values do not match the review key contract")
    else:
        selected["entry_key"] = computed
    if selected["entry_key"].duplicated().any():
        raise RuntimeError("Selected review entries contain duplicate keys")
    return selected


def choose_pass_dir(workspace: Path) -> tuple[int, Path, bool]:
    existing: list[tuple[int, Path]] = []
    for path in workspace.glob("pass_*"):
        try:
            pass_id = int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        existing.append((pass_id, path))
    if not existing:
        return 1, workspace / "pass_001", False
    pass_id, latest = max(existing)
    if not (latest / ".complete").is_file():
        return pass_id, latest, True
    return pass_id + 1, workspace / f"pass_{pass_id + 1:03d}", False


def prepare_pass(
    *,
    selected: pd.DataFrame,
    successful_keys: set[str],
    pass_dir: Path,
    requested_shards: int,
    resume_incomplete: bool,
) -> tuple[list[Path], set[str]]:
    manifest_path = pass_dir / "manifest.csv"
    metadata_path = pass_dir / "metadata.json"
    if resume_incomplete:
        if not manifest_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"Incomplete pass lacks locked metadata: {pass_dir}")
        manifest = pd.read_csv(manifest_path)
        required = {"entry_key", "shard_id"}
        if not required.issubset(manifest.columns):
            raise RuntimeError("Locked pass manifest has an invalid schema")
        locked_keys = set(manifest["entry_key"].astype(str))
        current_keys = set(selected["entry_key"].astype(str))
        if not locked_keys.issubset(current_keys):
            raise RuntimeError("Locked pass contains keys absent from the current input")
        shard_paths = sorted(pass_dir.glob("input_shard_*.csv"))
        if not shard_paths:
            raise RuntimeError("Incomplete pass has no locked shard inputs")
        return shard_paths, locked_keys

    pending = selected[~selected["entry_key"].isin(successful_keys)].copy()
    if pending.empty:
        return [], set()
    shard_count = min(requested_shards, len(pending))
    pass_dir.mkdir(parents=True, exist_ok=False)
    pending["shard_id"] = [index % shard_count for index in range(len(pending))]
    pending[["entry_key", "shard_id"]].to_csv(manifest_path, index=False)
    shard_paths = []
    for shard_id in range(shard_count):
        shard = pending[pending["shard_id"].eq(shard_id)].drop(columns=["shard_id"])
        path = pass_dir / f"input_shard_{shard_id}.csv"
        shard.to_csv(path, index=False)
        shard_paths.append(path)
    digest = hashlib.sha256(
        "\n".join(pending["entry_key"].astype(str)).encode()
    ).hexdigest()
    metadata_path.write_text(
        json.dumps(
            {
                "selected_total": len(selected),
                "success_rows_before_pass": len(successful_keys),
                "pending_rows": len(pending),
                "requested_shards": requested_shards,
                "actual_shards": shard_count,
                "shard_rows": [
                    int((pending["shard_id"] == shard_id).sum())
                    for shard_id in range(shard_count)
                ],
                "entry_key_sha256": digest,
            },
            indent=2,
        )
        + "\n"
    )
    return shard_paths, set(pending["entry_key"].astype(str))


def launch_shards(args: argparse.Namespace, pass_dir: Path, shard_paths: list[Path]) -> None:
    concurrency_per_shard = math.ceil(args.total_concurrency / len(shard_paths))
    processes: list[tuple[int, subprocess.Popen[bytes], object, object]] = []
    for shard_id, input_path in enumerate(shard_paths):
        rows = len(pd.read_csv(input_path, usecols=["entry_key"]))
        review_dir = pass_dir / f"review_shard_{shard_id}"
        review_dir.mkdir(exist_ok=True)
        command = [
            sys.executable,
            "-u",
            str(args.review_script),
            "--input-csv",
            str(input_path),
            "--report-archive",
            str(args.report_archive),
            "--output-dir",
            str(review_dir),
            "--top-k",
            str(rows),
            "--batch-size",
            str(rows),
            "--concurrency",
            str(min(rows, concurrency_per_shard)),
            "--retry-limit",
            str(args.retry_limit),
            "--resume",
            "--llm-mode",
            args.llm_mode,
            "--request-timeout-seconds",
            str(args.request_timeout_seconds),
            "--temperature",
            str(args.temperature),
            "--max-completion-tokens",
            str(args.max_completion_tokens),
        ]
        if args.llm_mode == "mock":
            command.extend(["--mock-latency-ms", str(args.mock_latency_ms)])
            if args.mock_fail_once_indexes:
                command.extend(
                    ["--mock-fail-once-indexes", args.mock_fail_once_indexes]
                )
            if args.mock_always_fail_indexes:
                command.extend(
                    ["--mock-always-fail-indexes", args.mock_always_fail_indexes]
                )
        stdout_handle = (review_dir / "worker.out").open("ab")
        stderr_handle = (review_dir / "worker.err").open("ab")
        process = subprocess.Popen(
            command, stdout=stdout_handle, stderr=stderr_handle, start_new_session=True
        )
        processes.append((shard_id, process, stdout_handle, stderr_handle))

    failures = []
    for shard_id, process, stdout_handle, stderr_handle in processes:
        return_code = process.wait()
        stdout_handle.close()
        stderr_handle.close()
        if return_code:
            failures.append((shard_id, return_code))
    if failures:
        raise RuntimeError(f"Sharded reviewers failed: {failures}")


def merge_pass(
    *,
    output_dir: Path,
    pass_dir: Path,
    shard_paths: list[Path],
    pass_keys: set[str],
) -> dict[str, object]:
    results_path = output_dir / "results.csv"
    errors_path = output_dir / "errors.csv"
    result_fields, existing_results = read_rows(results_path)
    error_fields, existing_errors = read_rows(errors_path)

    shard_payloads = []
    for shard_id, input_path in enumerate(shard_paths):
        input_keys = set(pd.read_csv(input_path, usecols=["entry_key"])["entry_key"].astype(str))
        review_dir = pass_dir / f"review_shard_{shard_id}"
        shard_result_fields, successes = read_rows(review_dir / "results.csv")
        shard_error_fields, errors = read_rows(review_dir / "errors.csv")
        output_keys = {row["entry_key"] for row in successes} | {
            row["entry_key"] for row in errors
        }
        if output_keys != input_keys:
            raise RuntimeError(
                f"Shard {shard_id} result accounting mismatch: "
                f"input={len(input_keys)}, output={len(output_keys)}"
            )
        shard_payloads.append(
            (
                shard_id,
                shard_result_fields,
                successes,
                shard_error_fields,
                errors,
            )
        )

    if not result_fields:
        for _, fields, successes, _, _ in shard_payloads:
            if successes:
                result_fields = fields
                break
    if not error_fields:
        for _, _, _, fields, errors in shard_payloads:
            if fields:
                error_fields = fields
                break
    if not result_fields or "entry_key" not in result_fields:
        raise RuntimeError("Could not determine result schema")
    if not error_fields or "entry_key" not in error_fields:
        raise RuntimeError("Could not determine error schema")

    seen = {row["entry_key"] for row in existing_results}
    success_before = len(seen)
    pass_error_rows: list[dict[str, str]] = []
    shard_summaries = []
    for shard_id, fields, successes, shard_error_fields, errors in shard_payloads:
        if fields and set(fields) != set(result_fields):
            raise RuntimeError(f"Result schema mismatch in shard {shard_id}")
        if shard_error_fields and set(shard_error_fields) != set(error_fields):
            raise RuntimeError(f"Error schema mismatch in shard {shard_id}")
        added = 0
        for row in successes:
            key = row["entry_key"]
            if key not in seen:
                seen.add(key)
                existing_results.append(row)
                added += 1
        pass_error_rows.extend(errors)
        shard_summaries.append(
            {
                "shard_id": shard_id,
                "success_entries": len(successes),
                "error_entries": len(errors),
                "successes_added": added,
            }
        )

    errors_by_key = {
        row["entry_key"]: row for row in existing_errors if row["entry_key"] not in seen
    }
    for row in pass_error_rows:
        if row["entry_key"] not in seen:
            errors_by_key[row["entry_key"]] = row
    atomic_write_rows(results_path, result_fields, existing_results, ".sharded.tmp")
    atomic_write_rows(
        errors_path, error_fields, list(errors_by_key.values()), ".sharded.tmp"
    )

    added = len(seen) - success_before
    pass_output_keys = {
        row["entry_key"]
        for _, _, successes, _, errors in shard_payloads
        for row in successes + errors
    }
    if pass_output_keys != pass_keys:
        raise RuntimeError("Merged pass does not account for every locked pass key")
    return {
        "pass_id": int(pass_dir.name.split("_", 1)[1]),
        "attempted_entries": len(pass_keys),
        "successful_entries_added": added,
        "remaining_error_entries": len(errors_by_key),
        "active_success_rows_after": len(seen),
        "shards": shard_summaries,
    }


def main() -> None:
    args = parse_args()
    if args.shards < 1 or args.total_concurrency < 1:
        raise ValueError("shards and total concurrency must be positive")
    if not args.review_script.is_file() or not args.report_archive.is_file():
        raise FileNotFoundError("Review script or report archive is missing")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    workspace = args.output_dir / ".sharded_review"
    workspace.mkdir(exist_ok=True)
    selected = selected_with_keys(args.input_csv, args.top_k)
    _, existing_results = read_rows(args.output_dir / "results.csv")
    successful_keys = {row["entry_key"] for row in existing_results}
    pass_id, pass_dir, resume_incomplete = choose_pass_dir(workspace)
    shard_paths, pass_keys = prepare_pass(
        selected=selected,
        successful_keys=successful_keys,
        pass_dir=pass_dir,
        requested_shards=args.shards,
        resume_incomplete=resume_incomplete,
    )
    if not shard_paths:
        print(json.dumps({"pending_entries": 0, "active_success_rows": len(successful_keys)}))
        return

    print(
        json.dumps(
            {
                "pass_id": pass_id,
                "resume_incomplete": resume_incomplete,
                "pending_entries": len(pass_keys),
                "shards": len(shard_paths),
                "requested_total_concurrency": args.total_concurrency,
            },
            indent=2,
        ),
        flush=True,
    )
    start = time.monotonic()
    launch_shards(args, pass_dir, shard_paths)
    summary = merge_pass(
        output_dir=args.output_dir,
        pass_dir=pass_dir,
        shard_paths=shard_paths,
        pass_keys=pass_keys,
    )
    summary["elapsed_seconds"] = time.monotonic() - start
    summary["requested_total_concurrency"] = args.total_concurrency
    (pass_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (pass_dir / ".complete").touch()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
