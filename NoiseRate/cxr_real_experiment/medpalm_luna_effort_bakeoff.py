#!/usr/bin/env python3
"""Compare explicit GPT-5.6-Luna reasoning efforts on a locked Med-PaLM sample."""

from __future__ import annotations

import argparse
import csv
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


MODEL = "gpt-5.6-luna"
EFFORTS = ("medium", "max")
MAX_OUTPUT_TOKENS = 1200
LUNA_INPUT_PRICE = 0.03
LUNA_OUTPUT_PRICE = 0.18
LUNA_CACHED_INPUT_PRICE = 0.003


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["review", "evaluate", "verify"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--effort", choices=EFFORTS)
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


def request_review(row: dict[str, Any], effort: str, timeout: float) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    model = os.environ.get("OPENAI_MODEL", "").strip()
    if not api_key or not base_url or model != MODEL:
        raise RuntimeError("The configured API credentials/model do not match the Luna effort run")
    payload = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": build_user_prompt(row),
        "reasoning": {"effort": effort},
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
    reasoning = data.get("reasoning") or {}
    return {
        **result,
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_prompt_tokens": input_details.get("cached_tokens", 0),
        "reasoning_tokens": output_details.get("reasoning_tokens", 0),
        "requested_reasoning_effort": effort,
        "response_reasoning_effort": reasoning.get("effort"),
        "response_temperature": data.get("temperature"),
        "latency_seconds": elapsed,
    }


def review_one(
    row: dict[str, Any], effort: str, retry_limit: int, timeout: float
) -> tuple[bool, dict[str, Any]]:
    last_error = ""
    for attempt in range(1, retry_limit + 2):
        try:
            result = request_review(row, effort, timeout)
            return True, {**row, **result, "attempt_count": attempt}
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
    if args.effort not in EFFORTS:
        raise ValueError("--effort is required for review mode")
    blinded = pd.read_csv(args.output_dir / "sample_entries_blinded.csv")
    if len(blinded) != 100 or blinded["entry_key"].duplicated().any():
        raise ValueError("Prepared blinded sample is invalid")
    prohibited = {"reader_label", "reader_agreement", "reference_binary_label", "expected_binary_action"}
    if prohibited.intersection(blinded.columns):
        raise AssertionError("Reference leaked into Luna input")

    prefix = f"gpt56_luna_{args.effort}"
    result_path = args.output_dir / f"{prefix}_results_blinded.csv"
    error_path = args.output_dir / f"{prefix}_errors.csv"
    completed: set[str] = set()
    if args.resume and result_path.exists() and result_path.stat().st_size:
        old = pd.read_csv(result_path)
        if old["entry_key"].duplicated().any():
            raise ValueError(f"Duplicate {args.effort} review results")
        completed = set(old["entry_key"].astype(str))
    pending = [row.to_dict() for _, row in blinded.iterrows() if str(row["entry_key"]) not in completed]
    print(f"Effort={args.effort} completed={len(completed)} pending={len(pending)}")
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
        "requested_reasoning_effort",
        "response_reasoning_effort",
        "response_temperature",
        "latency_seconds",
        "attempt_count",
    ]
    error_fields = list(blinded.columns) + ["error_message", "attempt_count"]
    errors: list[dict[str, Any]] = []

    # Probe the provider with one request before opening the concurrent batch.
    # The proxy can temporarily expose the model with no available distributor.
    canary_success, canary_output = review_one(
        pending[0], args.effort, args.retry_limit, args.timeout
    )
    if not canary_success:
        pd.DataFrame([canary_output], columns=error_fields).to_csv(error_path, index=False)
        raise RuntimeError(
            f"Luna {args.effort} canary failed; concurrent review was not started: "
            f"{canary_output['error_message']}"
        )
    append_csv(result_path, canary_output, result_fields)
    pending = pending[1:]
    if not pending:
        print("Canary completed the final pending entry.", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {
            executor.submit(
                review_one, row, args.effort, args.retry_limit, args.timeout
            ): row["entry_key"]
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
    result = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
    metadata = {
        "model": MODEL,
        "requested_reasoning_effort": args.effort,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "completed": int(result["entry_key"].nunique()) if len(result) else 0,
        "remaining_errors": len(errors),
    }
    if len(result):
        for column in [
            "prompt_tokens",
            "completion_tokens",
            "cached_prompt_tokens",
            "reasoning_tokens",
        ]:
            metadata[column] = int(pd.to_numeric(result[column], errors="coerce").sum())
        metadata["returned_reasoning_efforts"] = sorted(
            result["response_reasoning_effort"].dropna().astype(str).unique().tolist()
        )
        metadata["returned_temperatures"] = sorted(
            pd.to_numeric(result["response_temperature"], errors="coerce").dropna().unique().tolist()
        )
    (args.output_dir / f"{prefix}_review_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def exact_mcnemar_p(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(first_only, second_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def add_model(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    consistent = out[f"{prefix}_recommended_action"].eq(
        out[f"{prefix}_mismatch_type"].map(EXPECTED_RECOMMENDATION)
    )
    out[f"{prefix}_action"] = out[f"{prefix}_mismatch_type"].map(ACTION_BY_MISMATCH)
    out.loc[~consistent, f"{prefix}_action"] = "mask"
    out[f"{prefix}_correct"] = out[f"{prefix}_action"].eq(out["expected_binary_action"])
    return out


def metrics(frame: pd.DataFrame, prefix: str, model_name: str) -> dict[str, Any]:
    expected_keep = frame["expected_binary_action"].eq("keep")
    expected_relabel = frame["expected_binary_action"].eq("relabel")
    action = frame[f"{prefix}_action"]
    actual_relabel = action.eq("relabel")
    return {
        "model": model_name,
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


def pairwise(frame: pd.DataFrame, first: str, second: str) -> dict[str, Any]:
    first_only = int((frame[f"{first}_correct"] & ~frame[f"{second}_correct"]).sum())
    second_only = int((~frame[f"{first}_correct"] & frame[f"{second}_correct"]).sum())
    return {
        "first": first,
        "second": second,
        "first_correct_second_wrong": first_only,
        "first_wrong_second_correct": second_only,
        "both_correct": int((frame[f"{first}_correct"] & frame[f"{second}_correct"]).sum()),
        "both_wrong": int((~frame[f"{first}_correct"] & ~frame[f"{second}_correct"]).sum()),
        "exact_mcnemar_p_two_sided": exact_mcnemar_p(first_only, second_only),
        "action_agreement": float(frame[f"{first}_action"].eq(frame[f"{second}_action"]).mean()),
    }


def usage_summary(frame: pd.DataFrame, prefix: str, effort: str) -> dict[str, Any]:
    input_tokens = int(pd.to_numeric(frame[f"{prefix}_prompt_tokens"]).sum())
    output_tokens = int(pd.to_numeric(frame[f"{prefix}_completion_tokens"]).sum())
    cached_tokens = int(pd.to_numeric(frame[f"{prefix}_cached_prompt_tokens"]).sum())
    cost = (
        (input_tokens - cached_tokens) / 1e6 * LUNA_INPUT_PRICE
        + cached_tokens / 1e6 * LUNA_CACHED_INPUT_PRICE
        + output_tokens / 1e6 * LUNA_OUTPUT_PRICE
    )
    return {
        "effort": effort,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": int(pd.to_numeric(frame[f"{prefix}_reasoning_tokens"]).sum()),
        "cached_input_tokens": cached_tokens,
        "mean_latency_seconds": float(pd.to_numeric(frame[f"{prefix}_latency_seconds"]).mean()),
        "successful_response_cost_usd": cost,
    }


def evaluate(args: argparse.Namespace) -> None:
    reference = pd.read_csv(args.output_dir / "sample_reference_private.csv")
    gpt54 = pd.read_csv(args.output_dir / "gpt54_results_blinded.csv")
    if len(reference) != 100 or len(gpt54) != 100:
        raise ValueError("Locked reference/GPT-5.4 sample is incomplete")
    gpt54 = gpt54.rename(
        columns={
            "mismatch_type": "gpt54_mismatch_type",
            "recommended_action": "gpt54_recommended_action",
            "analysis_note": "gpt54_analysis_note",
        }
    )
    paired = reference.merge(gpt54, on="entry_key", validate="one_to_one")
    for effort in EFFORTS:
        prefix = f"luna_{effort}"
        result = pd.read_csv(args.output_dir / f"gpt56_luna_{effort}_results_blinded.csv")
        if len(result) != 100 or result["entry_key"].duplicated().any():
            raise ValueError(f"Luna {effort} results are incomplete")
        rename = {
            "mismatch_type": f"{prefix}_mismatch_type",
            "recommended_action": f"{prefix}_recommended_action",
            "analysis_note": f"{prefix}_analysis_note",
            "prompt_tokens": f"{prefix}_prompt_tokens",
            "completion_tokens": f"{prefix}_completion_tokens",
            "cached_prompt_tokens": f"{prefix}_cached_prompt_tokens",
            "reasoning_tokens": f"{prefix}_reasoning_tokens",
            "latency_seconds": f"{prefix}_latency_seconds",
            "response_reasoning_effort": f"{prefix}_response_reasoning_effort",
            "response_temperature": f"{prefix}_response_temperature",
        }
        columns = ["entry_key", *rename]
        paired = paired.merge(
            result[columns].rename(columns=rename), on="entry_key", validate="one_to_one"
        )

    paired = add_model(paired, "gpt54")
    for effort in EFFORTS:
        paired = add_model(paired, f"luna_{effort}")
    paired.to_csv(args.output_dir / "effort_paired_row_evaluation_private.csv", index=False)

    summary = pd.DataFrame(
        [
            metrics(paired, "gpt54", "gpt-5.4"),
            metrics(paired, "luna_medium", "gpt-5.6-luna-medium"),
            metrics(paired, "luna_max", "gpt-5.6-luna-max"),
        ]
    )
    summary.to_csv(args.output_dir / "effort_model_quality_summary.csv", index=False)
    comparison = {
        "protocol": "medpalm_gpt54_vs_gpt56luna_explicit_effort_n100_v1",
        "sample_design": "same locked 50 keep + 50 relabel entries from 100 unique subjects",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "pairwise": [
            pairwise(paired, "gpt54", "luna_medium"),
            pairwise(paired, "gpt54", "luna_max"),
            pairwise(paired, "luna_medium", "luna_max"),
        ],
        "usage": [
            usage_summary(paired, "luna_medium", "medium"),
            usage_summary(paired, "luna_max", "max"),
        ],
        "returned_effort": {
            effort: sorted(
                paired[f"luna_{effort}_response_reasoning_effort"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            for effort in EFFORTS
        },
        "returned_temperature": {
            effort: sorted(
                pd.to_numeric(
                    paired[f"luna_{effort}_response_temperature"], errors="coerce"
                )
                .dropna()
                .unique()
                .tolist()
            )
            for effort in EFFORTS
        },
    }
    (args.output_dir / "effort_paired_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    (args.output_dir / ".effort_evaluated").touch()
    print(summary.to_string(index=False))
    print(json.dumps(comparison, indent=2))


def verify(args: argparse.Namespace) -> None:
    required = [
        ".prepared",
        ".effort_evaluated",
        "sample_entries_blinded.csv",
        "sample_reference_private.csv",
        "gpt54_results_blinded.csv",
        "effort_model_quality_summary.csv",
        "effort_paired_comparison.json",
        "effort_paired_row_evaluation_private.csv",
    ]
    for effort in EFFORTS:
        required.extend(
            [
                f"gpt56_luna_{effort}_results_blinded.csv",
                f"gpt56_luna_{effort}_errors.csv",
                f"gpt56_luna_{effort}_review_metadata.json",
            ]
        )
    missing = [name for name in required if not (args.output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing effort bake-off outputs: {missing}")
    blinded = pd.read_csv(args.output_dir / "sample_entries_blinded.csv")
    if len(blinded) != 100 or blinded["subject_id"].nunique() != 100:
        raise ValueError("Locked effort sample is invalid")
    for effort in EFFORTS:
        result = pd.read_csv(args.output_dir / f"gpt56_luna_{effort}_results_blinded.csv")
        errors = pd.read_csv(args.output_dir / f"gpt56_luna_{effort}_errors.csv")
        if len(result) != 100 or result["entry_key"].duplicated().any() or len(errors):
            raise ValueError(f"Luna {effort} did not finish with 100 unique results and zero errors")
        if not result["requested_reasoning_effort"].eq(effort).all():
            raise ValueError(f"Luna {effort} request metadata is inconsistent")
        returned = set(result["response_reasoning_effort"].dropna().astype(str))
        if returned != {effort}:
            raise ValueError(f"Luna {effort} response metadata returned {returned}")
    (args.output_dir / ".complete").touch()
    print("Explicit Luna medium/max Med-PaLM bake-off verification passed.")


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
