#!/usr/bin/env python3
"""Paired GPT-5.4 versus GPT-5.6-Luna Med-PaLM review bake-off."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests

from medpalm_external_entry_benchmark import (
    ACTION_BY_MISMATCH,
    EXPECTED_RECOMMENDATION,
    SYSTEM_PROMPT,
    build_user_prompt,
    validate_payload,
)


DEFAULT_SOURCE = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "medpalm_external_entry_benchmark/20260730_binary_external_no_oof"
)
SAMPLE_SEED = 20260820
PER_ACTION = 50
EXPECTED_MODEL = "gpt-5.6-luna"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "review", "evaluate", "verify"])
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--retry-limit", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_order(frame: pd.DataFrame, salt: str) -> pd.DataFrame:
    out = frame.copy()
    out["_sample_order"] = out["entry_key"].astype(str).map(
        lambda key: hashlib.sha256(f"{SAMPLE_SEED}:{salt}:{key}".encode()).hexdigest()
    )
    return out.sort_values("_sample_order", kind="stable").drop(columns="_sample_order")


def select_unique_subjects(frame: pd.DataFrame, n: int, excluded: set[int], salt: str) -> pd.DataFrame:
    candidates = stable_order(frame[~frame["subject_id"].isin(excluded)], salt)
    selected = candidates.drop_duplicates("subject_id", keep="first").head(n).copy()
    if len(selected) != n:
        raise ValueError(f"Could only select {len(selected)}/{n} unique-subject rows for {salt}")
    return selected


def prepare(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    row_path = args.source_dir / "row_level_evaluation_private.csv"
    blinded_path = args.source_dir / "benchmark_entries_blinded.csv"
    review_path = args.source_dir / "review_results_blinded.csv"
    for path in [row_path, blinded_path, review_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = pd.read_csv(row_path)
    blinded = pd.read_csv(blinded_path)
    reviews = pd.read_csv(review_path)
    if len(rows) != 498 or len(blinded) != 498 or len(reviews) != 498:
        raise ValueError("Canonical Med-PaLM benchmark must contain 498 entries")

    relabel = select_unique_subjects(
        rows[rows["expected_binary_action"].eq("relabel")], PER_ACTION, set(), "relabel"
    )
    used_subjects = set(relabel["subject_id"].astype(int))
    keep = select_unique_subjects(
        rows[rows["expected_binary_action"].eq("keep")], PER_ACTION, used_subjects, "keep"
    )
    sample = pd.concat([relabel, keep], ignore_index=True)
    sample = stable_order(sample, "final")
    if len(sample) != 100 or sample["entry_key"].duplicated().any():
        raise AssertionError("The paired sample must contain 100 unique entries")
    if sample["subject_id"].nunique() != 100:
        raise AssertionError("Every paired entry must come from a different subject")

    keys = sample[["entry_key"]]
    sample_blinded = keys.merge(blinded, on="entry_key", how="left", validate="one_to_one")
    prohibited = {"reader_label", "reader_agreement", "reference_binary_label", "expected_binary_action"}
    if prohibited.intersection(sample_blinded.columns) or sample_blinded.isna().any().any():
        raise AssertionError("Blinded sample contains reference leakage or missing values")

    reference_columns = [
        "entry_key",
        "subject_id",
        "study_id",
        "finding",
        "reader_label",
        "reader_agreement",
        "reference_binary_label",
        "expected_binary_action",
    ]
    sample_reference = sample[reference_columns].copy()
    gpt54_columns = [
        "entry_key",
        "mismatch_type",
        "analysis_note",
        "recommended_action",
        "attempt_count",
    ]
    gpt54 = keys.merge(reviews[gpt54_columns], on="entry_key", how="left", validate="one_to_one")
    if gpt54.isna().any().any():
        raise AssertionError("Archived GPT-5.4 results are incomplete for the paired sample")

    sample_blinded.to_csv(args.output_dir / "sample_entries_blinded.csv", index=False)
    sample_reference.to_csv(args.output_dir / "sample_reference_private.csv", index=False)
    gpt54.to_csv(args.output_dir / "gpt54_results_blinded.csv", index=False)
    key_digest = hashlib.sha256("\n".join(sample["entry_key"]).encode()).hexdigest()
    metadata = {
        "protocol": "medpalm_gpt54_vs_gpt56luna_paired_n100_v1",
        "sample_seed": SAMPLE_SEED,
        "sample_size": 100,
        "unique_subjects": 100,
        "expected_keep": int(sample["expected_binary_action"].eq("keep").sum()),
        "expected_relabel": int(sample["expected_binary_action"].eq("relabel").sum()),
        "selected_key_sha256": key_digest,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "source_row_evaluation_sha256": sha256_file(row_path),
        "source_blinded_sha256": sha256_file(blinded_path),
        "source_gpt54_review_sha256": sha256_file(review_path),
    }
    (args.output_dir / "preparation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (args.output_dir / ".prepared").touch()
    print(json.dumps(metadata, indent=2))


def request_review(row: dict[str, Any], timeout: float) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    model = os.environ.get("OPENAI_MODEL", "").strip()
    if not api_key or not base_url or model != EXPECTED_MODEL:
        raise RuntimeError("The configured API credentials/model do not match the locked Luna run")
    payload = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": build_user_prompt(row),
        "temperature": 0,
        "max_output_tokens": 400,
    }
    started = time.monotonic()
    response = requests.post(
        urljoin(base_url.rstrip("/") + "/", "responses"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    elapsed = time.monotonic() - started
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text.strip()[:500]}")
    data = response.json()
    content = "".join(
        part.get("text", "")
        for item in data.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )
    if not content:
        raise RuntimeError(f"Responses API returned no output_text: {json.dumps(data)[:500]}")
    result = validate_payload(json.loads(content))
    usage = data.get("usage") or {}
    details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    return {
        **result,
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_prompt_tokens": details.get("cached_tokens", 0),
        "reasoning_tokens": output_details.get("reasoning_tokens", 0),
        "latency_seconds": elapsed,
    }


def review_one(row: dict[str, Any], retry_limit: int, timeout: float) -> tuple[bool, dict[str, Any]]:
    last_error = ""
    for attempt in range(1, retry_limit + 2):
        try:
            return True, {**row, **request_review(row, timeout), "attempt_count": attempt}
        except Exception as exc:  # network/provider errors are recorded per entry
            last_error = str(exc)
            if attempt <= retry_limit:
                time.sleep(0.5 * attempt)
    return False, {**row, "error_message": last_error, "attempt_count": retry_limit + 1}


def append_csv(path: Path, row: dict[str, Any], fields: list[str]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def review(args: argparse.Namespace) -> None:
    blinded_path = args.output_dir / "sample_entries_blinded.csv"
    blinded = pd.read_csv(blinded_path)
    if len(blinded) != 100 or blinded["entry_key"].duplicated().any():
        raise ValueError("Prepared blinded sample is invalid")
    prohibited = {"reader_label", "reader_agreement", "reference_binary_label", "expected_binary_action"}
    if prohibited.intersection(blinded.columns):
        raise AssertionError("Reference leaked into Luna input")

    result_path = args.output_dir / "gpt56_luna_results_blinded.csv"
    error_path = args.output_dir / "gpt56_luna_errors.csv"
    completed: set[str] = set()
    if args.resume and result_path.exists() and result_path.stat().st_size:
        old = pd.read_csv(result_path)
        if old["entry_key"].duplicated().any():
            raise ValueError("Duplicate Luna review results")
        completed = set(old["entry_key"].astype(str))
    pending = [row.to_dict() for _, row in blinded.iterrows() if str(row["entry_key"]) not in completed]
    print(f"Completed={len(completed)} pending={len(pending)}")
    if not pending:
        return

    result_fields = list(blinded.columns) + [
        "mismatch_type",
        "analysis_note",
        "recommended_action",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_prompt_tokens",
        "reasoning_tokens",
        "latency_seconds",
        "attempt_count",
    ]
    error_fields = list(blinded.columns) + ["error_message", "attempt_count"]
    errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {
            executor.submit(review_one, row, args.retry_limit, args.timeout): row["entry_key"]
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            success, output = future.result()
            if success:
                append_csv(result_path, output, result_fields)
            else:
                errors.append(output)
            if index % 10 == 0 or index == len(pending):
                print(f"Processed {index}/{len(pending)}", flush=True)
    pd.DataFrame(errors, columns=error_fields).to_csv(error_path, index=False)
    result = pd.read_csv(result_path)
    metadata = {
        "model": EXPECTED_MODEL,
        "temperature": 0,
        "completed": int(result["entry_key"].nunique()),
        "remaining_errors": len(errors),
        "prompt_tokens": int(pd.to_numeric(result["prompt_tokens"], errors="coerce").sum()),
        "completion_tokens": int(pd.to_numeric(result["completion_tokens"], errors="coerce").sum()),
        "cached_prompt_tokens": int(pd.to_numeric(result["cached_prompt_tokens"], errors="coerce").sum()),
        "reasoning_tokens": int(pd.to_numeric(result["reasoning_tokens"], errors="coerce").sum()),
    }
    (args.output_dir / "luna_review_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def exact_mcnemar_p(gpt54_only: int, luna_only: int) -> float:
    discordant = gpt54_only + luna_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(gpt54_only, luna_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def add_actions(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    consistent = out[f"{prefix}_recommended_action"].eq(
        out[f"{prefix}_mismatch_type"].map(EXPECTED_RECOMMENDATION)
    )
    out[f"{prefix}_action"] = out[f"{prefix}_mismatch_type"].map(ACTION_BY_MISMATCH)
    out.loc[~consistent, f"{prefix}_action"] = "mask"
    out[f"{prefix}_correct"] = out[f"{prefix}_action"].eq(out["expected_binary_action"])
    return out


def model_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    expected_keep = frame["expected_binary_action"].eq("keep")
    expected_relabel = frame["expected_binary_action"].eq("relabel")
    action = frame[f"{prefix}_action"]
    actual_relabel = action.eq("relabel")
    return {
        "model": "gpt-5.4" if prefix == "gpt54" else EXPECTED_MODEL,
        "n_entries": len(frame),
        "action_accuracy": float(frame[f"{prefix}_correct"].mean()),
        "keep_accuracy": float((action[expected_keep] == "keep").mean()),
        "relabel_recall": float((action[expected_relabel] == "relabel").mean()),
        "relabel_precision_balanced_sample": float(
            (expected_relabel & actual_relabel).sum() / actual_relabel.sum()
        ) if actual_relabel.sum() else float("nan"),
        "harmful_flip_rate": float((action[expected_keep] == "relabel").mean()),
        "mask_rate": float(action.eq("mask").mean()),
    }


def evaluate(args: argparse.Namespace) -> None:
    reference = pd.read_csv(args.output_dir / "sample_reference_private.csv")
    gpt54 = pd.read_csv(args.output_dir / "gpt54_results_blinded.csv")
    luna = pd.read_csv(args.output_dir / "gpt56_luna_results_blinded.csv")
    if len(reference) != 100 or len(gpt54) != 100 or len(luna) != 100:
        raise ValueError("Paired evaluation requires 100 complete results per model")
    if luna["entry_key"].duplicated().any():
        raise ValueError("Luna results contain duplicate entries")
    gpt54 = gpt54.rename(
        columns={
            "mismatch_type": "gpt54_mismatch_type",
            "recommended_action": "gpt54_recommended_action",
            "analysis_note": "gpt54_analysis_note",
        }
    )
    luna = luna.rename(
        columns={
            "mismatch_type": "luna_mismatch_type",
            "recommended_action": "luna_recommended_action",
            "analysis_note": "luna_analysis_note",
        }
    )
    paired = reference.merge(gpt54, on="entry_key", validate="one_to_one").merge(
        luna[
            [
                "entry_key",
                "luna_mismatch_type",
                "luna_recommended_action",
                "luna_analysis_note",
                "prompt_tokens",
                "completion_tokens",
                "cached_prompt_tokens",
                "reasoning_tokens",
                "latency_seconds",
            ]
        ],
        on="entry_key",
        validate="one_to_one",
    )
    paired = add_actions(paired, "gpt54")
    paired = add_actions(paired, "luna")
    paired.to_csv(args.output_dir / "paired_row_evaluation_private.csv", index=False)
    summary = pd.DataFrame([model_metrics(paired, "gpt54"), model_metrics(paired, "luna")])
    summary.to_csv(args.output_dir / "model_quality_summary.csv", index=False)

    gpt54_only = int((paired["gpt54_correct"] & ~paired["luna_correct"]).sum())
    luna_only = int((~paired["gpt54_correct"] & paired["luna_correct"]).sum())
    comparison = {
        "protocol": "medpalm_gpt54_vs_gpt56luna_paired_n100_v1",
        "sample_design": "50 expert-expected keep + 50 expert-expected relabel; 100 unique subjects",
        "gpt54_correct_luna_wrong": gpt54_only,
        "gpt54_wrong_luna_correct": luna_only,
        "both_correct": int((paired["gpt54_correct"] & paired["luna_correct"]).sum()),
        "both_wrong": int((~paired["gpt54_correct"] & ~paired["luna_correct"]).sum()),
        "exact_mcnemar_p_two_sided": exact_mcnemar_p(gpt54_only, luna_only),
        "action_agreement": float(paired["gpt54_action"].eq(paired["luna_action"]).mean()),
        "luna_prompt_tokens": int(pd.to_numeric(paired["prompt_tokens"]).sum()),
        "luna_completion_tokens": int(pd.to_numeric(paired["completion_tokens"]).sum()),
        "luna_cached_prompt_tokens": int(pd.to_numeric(paired["cached_prompt_tokens"]).sum()),
        "luna_reasoning_tokens": int(pd.to_numeric(paired["reasoning_tokens"]).sum()),
        "luna_mean_latency_seconds": float(pd.to_numeric(paired["latency_seconds"]).mean()),
    }
    (args.output_dir / "paired_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    (args.output_dir / ".evaluated").touch()
    print(summary.to_string(index=False))
    print(json.dumps(comparison, indent=2))


def verify(args: argparse.Namespace) -> None:
    required = [
        ".prepared",
        ".evaluated",
        "preparation_metadata.json",
        "sample_entries_blinded.csv",
        "sample_reference_private.csv",
        "gpt54_results_blinded.csv",
        "gpt56_luna_results_blinded.csv",
        "gpt56_luna_errors.csv",
        "luna_review_metadata.json",
        "model_quality_summary.csv",
        "paired_comparison.json",
        "paired_row_evaluation_private.csv",
    ]
    missing = [name for name in required if not (args.output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing bake-off outputs: {missing}")
    blinded = pd.read_csv(args.output_dir / "sample_entries_blinded.csv")
    luna = pd.read_csv(args.output_dir / "gpt56_luna_results_blinded.csv")
    errors = pd.read_csv(args.output_dir / "gpt56_luna_errors.csv")
    if len(blinded) != 100 or len(luna) != 100 or len(errors) != 0:
        raise ValueError("Bake-off did not finish with 100 results and zero errors")
    if blinded["subject_id"].nunique() != 100 or luna["entry_key"].duplicated().any():
        raise ValueError("Bake-off uniqueness verification failed")
    (args.output_dir / ".complete").touch()
    print("Med-PaLM GPT-5.4 versus GPT-5.6-Luna bake-off verification passed.")


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "review":
        review(args)
    elif args.mode == "evaluate":
        evaluate(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
