#!/usr/bin/env python3
"""Compare GPT-5.6-Terra default reasoning with archived GPT-5.4 actions."""

from __future__ import annotations

import argparse
import csv
import json
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
from medpalm_luna_effort_bakeoff import exact_mcnemar_p


MODEL = "gpt-5.6-terra"
MAX_OUTPUT_TOKENS = 1200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["review", "evaluate", "verify"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--retry-limit", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def response_text(data: dict[str, Any]) -> str:
    return "".join(
        part.get("text", "")
        for item in data.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )


def request_review(row: dict[str, Any], timeout: float) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    model = os.environ.get("OPENAI_MODEL", "").strip()
    if not api_key or not base_url or model != MODEL:
        raise RuntimeError("The configured API credentials/model do not match the Terra run")

    # Deliberately omit the reasoning field to measure the provider's default.
    payload = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": build_user_prompt(row),
        "temperature": 0,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
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
    text = response_text(data)
    if not text:
        raise RuntimeError(f"Responses API returned no output_text: {json.dumps(data)[:500]}")
    result = validate_payload(json.loads(text))
    usage = data.get("usage") or {}
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    returned_reasoning = data.get("reasoning") or {}
    return {
        **result,
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_prompt_tokens": input_details.get("cached_tokens", 0),
        "reasoning_tokens": output_details.get("reasoning_tokens", 0),
        "reasoning_parameter_sent": False,
        "response_reasoning_effort": returned_reasoning.get("effort"),
        "response_temperature": data.get("temperature"),
        "latency_seconds": elapsed,
    }


def review_one(
    row: dict[str, Any], retry_limit: int, timeout: float
) -> tuple[bool, dict[str, Any]]:
    last_error = ""
    for attempt in range(1, retry_limit + 2):
        try:
            output = request_review(row, timeout)
            return True, {**row, **output, "attempt_count": attempt}
        except Exception as exc:
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
    blinded = pd.read_csv(args.output_dir / "sample_entries_blinded.csv")
    if len(blinded) != 100 or blinded["entry_key"].duplicated().any():
        raise ValueError("Prepared blinded sample is invalid")
    prohibited = {"reader_label", "reader_agreement", "reference_binary_label", "expected_binary_action"}
    if prohibited.intersection(blinded.columns):
        raise AssertionError("Reference leaked into Terra input")

    result_path = args.output_dir / "gpt56_terra_default_results_blinded.csv"
    error_path = args.output_dir / "gpt56_terra_default_errors.csv"
    completed: set[str] = set()
    if args.resume and result_path.exists() and result_path.stat().st_size:
        old = pd.read_csv(result_path)
        if old["entry_key"].duplicated().any():
            raise ValueError("Duplicate Terra review results")
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
        "reasoning_parameter_sent",
        "response_reasoning_effort",
        "response_temperature",
        "latency_seconds",
        "attempt_count",
    ]
    error_fields = list(blinded.columns) + ["error_message", "attempt_count"]

    canary_success, canary = review_one(pending[0], args.retry_limit, args.timeout)
    if not canary_success:
        pd.DataFrame([canary], columns=error_fields).to_csv(error_path, index=False)
        raise RuntimeError(f"Terra canary failed; batch not started: {canary['error_message']}")
    append_csv(result_path, canary, result_fields)
    pending = pending[1:]

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
    metadata: dict[str, Any] = {
        "model": MODEL,
        "reasoning_parameter_sent": False,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "completed": int(result["entry_key"].nunique()),
        "remaining_errors": len(errors),
    }
    for column in ["prompt_tokens", "completion_tokens", "cached_prompt_tokens", "reasoning_tokens"]:
        metadata[column] = int(pd.to_numeric(result[column], errors="coerce").sum())
    metadata["returned_reasoning_efforts"] = sorted(
        result["response_reasoning_effort"].dropna().astype(str).unique().tolist()
    )
    metadata["returned_temperatures"] = sorted(
        pd.to_numeric(result["response_temperature"], errors="coerce").dropna().unique().tolist()
    )
    metadata["mean_latency_seconds"] = float(
        pd.to_numeric(result["latency_seconds"], errors="coerce").mean()
    )
    (args.output_dir / "terra_default_review_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def add_actions(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    consistent = out[f"{prefix}_recommended_action"].eq(
        out[f"{prefix}_mismatch_type"].map(EXPECTED_RECOMMENDATION)
    )
    out[f"{prefix}_action"] = out[f"{prefix}_mismatch_type"].map(ACTION_BY_MISMATCH)
    out.loc[~consistent, f"{prefix}_action"] = "mask"
    out[f"{prefix}_correct"] = out[f"{prefix}_action"].eq(out["expected_binary_action"])
    return out


def model_metrics(frame: pd.DataFrame, prefix: str, name: str) -> dict[str, Any]:
    expected_keep = frame["expected_binary_action"].eq("keep")
    expected_relabel = frame["expected_binary_action"].eq("relabel")
    action = frame[f"{prefix}_action"]
    actual_relabel = action.eq("relabel")
    return {
        "model": name,
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
    terra = pd.read_csv(args.output_dir / "gpt56_terra_default_results_blinded.csv")
    if len(reference) != 100 or len(gpt54) != 100 or len(terra) != 100:
        raise ValueError("Paired evaluation requires 100 complete results per model")
    if terra["entry_key"].duplicated().any():
        raise ValueError("Terra results contain duplicate entries")

    gpt54 = gpt54.rename(columns={
        "mismatch_type": "gpt54_mismatch_type",
        "recommended_action": "gpt54_recommended_action",
        "analysis_note": "gpt54_analysis_note",
    })
    terra = terra.rename(columns={
        "mismatch_type": "terra_mismatch_type",
        "recommended_action": "terra_recommended_action",
        "analysis_note": "terra_analysis_note",
    })
    terra_columns = [
        "entry_key",
        "terra_mismatch_type",
        "terra_recommended_action",
        "terra_analysis_note",
        "prompt_tokens",
        "completion_tokens",
        "cached_prompt_tokens",
        "reasoning_tokens",
        "response_reasoning_effort",
        "response_temperature",
        "latency_seconds",
    ]
    paired = reference.merge(gpt54, on="entry_key", validate="one_to_one").merge(
        terra[terra_columns], on="entry_key", validate="one_to_one"
    )
    paired = add_actions(paired, "gpt54")
    paired = add_actions(paired, "terra")
    paired.to_csv(args.output_dir / "terra_default_paired_row_evaluation_private.csv", index=False)

    summary = pd.DataFrame([
        model_metrics(paired, "gpt54", "gpt-5.4"),
        model_metrics(paired, "terra", "gpt-5.6-terra-default"),
    ])
    summary.to_csv(args.output_dir / "terra_default_model_quality_summary.csv", index=False)

    gpt54_only = int((paired["gpt54_correct"] & ~paired["terra_correct"]).sum())
    terra_only = int((~paired["gpt54_correct"] & paired["terra_correct"]).sum())
    comparison = {
        "protocol": "medpalm_gpt54_vs_gpt56terra_default_n100_v1",
        "sample_design": "same locked 50 keep + 50 relabel entries from 100 unique subjects",
        "reasoning_parameter_sent": False,
        "gpt54_correct_terra_wrong": gpt54_only,
        "gpt54_wrong_terra_correct": terra_only,
        "both_correct": int((paired["gpt54_correct"] & paired["terra_correct"]).sum()),
        "both_wrong": int((~paired["gpt54_correct"] & ~paired["terra_correct"]).sum()),
        "exact_mcnemar_p_two_sided": exact_mcnemar_p(gpt54_only, terra_only),
        "action_agreement": float(paired["gpt54_action"].eq(paired["terra_action"]).mean()),
        "prompt_tokens": int(pd.to_numeric(paired["prompt_tokens"]).sum()),
        "completion_tokens": int(pd.to_numeric(paired["completion_tokens"]).sum()),
        "cached_prompt_tokens": int(pd.to_numeric(paired["cached_prompt_tokens"]).sum()),
        "reasoning_tokens": int(pd.to_numeric(paired["reasoning_tokens"]).sum()),
        "mean_latency_seconds": float(pd.to_numeric(paired["latency_seconds"]).mean()),
        "returned_reasoning_efforts": sorted(
            paired["response_reasoning_effort"].dropna().astype(str).unique().tolist()
        ),
        "returned_temperatures": sorted(
            pd.to_numeric(paired["response_temperature"], errors="coerce").dropna().unique().tolist()
        ),
    }
    (args.output_dir / "terra_default_paired_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    (args.output_dir / ".terra_default_evaluated").touch()
    print(summary.to_string(index=False))
    print(json.dumps(comparison, indent=2))


def verify(args: argparse.Namespace) -> None:
    required = [
        ".prepared",
        ".terra_default_evaluated",
        "sample_entries_blinded.csv",
        "sample_reference_private.csv",
        "gpt54_results_blinded.csv",
        "gpt56_terra_default_results_blinded.csv",
        "gpt56_terra_default_errors.csv",
        "terra_default_review_metadata.json",
        "terra_default_model_quality_summary.csv",
        "terra_default_paired_comparison.json",
        "terra_default_paired_row_evaluation_private.csv",
    ]
    missing = [name for name in required if not (args.output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Terra bake-off outputs: {missing}")
    result = pd.read_csv(args.output_dir / "gpt56_terra_default_results_blinded.csv")
    errors = pd.read_csv(args.output_dir / "gpt56_terra_default_errors.csv")
    if len(result) != 100 or result["entry_key"].duplicated().any() or len(errors):
        raise ValueError("Terra run did not finish with 100 unique results and zero errors")
    if result["reasoning_parameter_sent"].astype(str).str.lower().ne("false").any():
        raise ValueError("Terra default run unexpectedly recorded an explicit reasoning parameter")
    (args.output_dir / ".complete").touch()
    print("Med-PaLM GPT-5.4 versus GPT-5.6-Terra default bake-off verification passed.")


def main() -> None:
    args = parse_args()
    if args.mode == "review":
        review(args)
    elif args.mode == "evaluate":
        evaluate(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
