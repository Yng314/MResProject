#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests


LABEL_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
]
LABEL_TO_INDEX = {name: index for index, name in enumerate(LABEL_NAMES)}

SYSTEM_PROMPT = """\
You are reviewing a suspicious binary chest X-ray label entry. An external method identified
this specific entry as suspicious, but its proposed replacement label and confidence are not
provided. The decision target is the current binary label used by the existing refinement and
model-training pipeline, not the raw four-state CheXpert output. Under the fixed U-Ones
projection, raw -1 (uncertain) and raw 1 both map to binary 1, while raw 0 maps to binary 0.
The raw value is supplied only as provenance.

Your job is to judge, using only the original radiology report text, whether the suspicious
entry is more likely:
1. a case where the current binary label is wrong and should be flipped,
2. a case where the current binary label is likely correct and should be kept,
3. or a case where the report is too ambiguous or internally conflicting to decide.

You must only use the original report text as evidence.
Do not use outside medical knowledge.
Do not assume the external method is correct merely because it selected this entry.
If the report is ambiguous, internally conflicting, or does not provide enough stable evidence
for the target label, choose report_ambiguous.

Return valid JSON only, with exactly these keys:
- mismatch_type
- analysis_note
- recommended_action

Allowed values for mismatch_type:
- chexpert_wrong_cl_right
- chexpert_right_cl_wrong
- report_ambiguous

The historical words "chexpert" and "cl" in these output values are retained for compatibility
with the existing refinement pipeline. Here they mean that the current derived binary label is
wrong or right; no confident-learning probability is supplied.

Allowed values for recommended_action:
- keep_chexpert
- review_or_relabel
- mark_ambiguous

Requirements for analysis_note:
- Write 2 to 4 sentences.
- State whether the report supports the current binary label, supports flipping it, or is ambiguous.
- Quote the key report text directly.
- If there is a conflict, explicitly say which statement conflicts with which statement.
- Briefly explain why the quoted text supports the final mismatch_type.
""".strip()

EXPECTED_RECOMMENDATION = {
    "chexpert_wrong_cl_right": "review_or_relabel",
    "chexpert_right_cl_wrong": "keep_chexpert",
    "report_ambiguous": "mark_ambiguous",
}
ACTION_BY_MISMATCH = {
    "chexpert_wrong_cl_right": "relabel",
    "chexpert_right_cl_wrong": "keep",
    "report_ambiguous": "mask",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Med-PaLM externally flagged entry-level LLM benchmark."
    )
    parser.add_argument("mode", choices=["prepare", "review", "evaluate", "verify"])
    parser.add_argument("--ground-truth-csv", type=Path)
    parser.add_argument("--chexpert-csv", type=Path)
    parser.add_argument("--split-csv", type=Path)
    parser.add_argument("--report-archive", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm-mode", choices=["real", "mock"], default="real")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--retry-limit", type=int, default=2)
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=400)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    return parser.parse_args()


def require_path(value: Path | None, name: str) -> Path:
    if value is None:
        raise ValueError(f"{name} is required for this mode")
    if not value.exists():
        raise FileNotFoundError(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_ids(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ["subject_id", "study_id"]:
        out[column] = pd.to_numeric(out[column], errors="raise").astype("int64")
    return out


def load_report_map(report_archive: Path, study_ids: list[int]) -> dict[int, str]:
    reports: dict[int, str] = {}
    with zipfile.ZipFile(report_archive) as archive:
        members = {
            Path(name).name: name for name in archive.namelist() if name.endswith(".txt")
        }
        for study_id in study_ids:
            key = f"s{study_id}.txt"
            if key not in members:
                raise ValueError(f"Missing report for study {study_id}")
            report = archive.read(members[key]).decode("utf-8", errors="replace").strip()
            if not report:
                raise ValueError(f"Empty report for study {study_id}")
            reports[study_id] = report
    return reports


def prepare_benchmark(args: argparse.Namespace) -> None:
    ground_truth_path = require_path(args.ground_truth_csv, "--ground-truth-csv")
    chexpert_path = require_path(args.chexpert_csv, "--chexpert-csv")
    split_path = require_path(args.split_csv, "--split-csv")
    report_archive = require_path(args.report_archive, "--report-archive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = normalize_ids(pd.read_csv(ground_truth_path))
    chexpert = normalize_ids(pd.read_csv(chexpert_path))
    split = normalize_ids(pd.read_csv(split_path))

    required_gt = {
        "subject_id",
        "study_id",
        "finding",
        "reader_label",
        "reader_agreement",
    }
    if not required_gt.issubset(ground_truth.columns):
        raise ValueError(f"Ground truth columns missing: {sorted(required_gt - set(ground_truth))}")
    if ground_truth.duplicated(["subject_id", "study_id", "finding"]).any():
        raise ValueError("Ground truth contains duplicate study-finding entries")
    if not set(LABEL_NAMES).issubset(chexpert.columns):
        raise ValueError("CheXpert table does not contain the current 12-label schema")

    split_per_study = split[["subject_id", "study_id", "split"]].drop_duplicates()
    if split_per_study.duplicated(["subject_id", "study_id"]).any():
        raise ValueError("A study has inconsistent split assignments")

    merged = ground_truth.merge(
        chexpert,
        on=["subject_id", "study_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        split_per_study,
        on=["subject_id", "study_id"],
        how="left",
        validate="many_to_one",
    )
    merged["raw_label"] = [
        row[row["finding"]] if row["finding"] in chexpert.columns else np.nan
        for _, row in merged.iterrows()
    ]
    merged["in_current_schema"] = merged["finding"].isin(LABEL_NAMES)
    merged["raw_valid"] = merged["raw_label"].isin([-1.0, 0.0, 1.0])
    eligible = merged[merged["in_current_schema"] & merged["raw_valid"]].copy()
    eligible = eligible.sort_values(
        ["subject_id", "study_id", "finding"], kind="stable"
    ).reset_index(drop=True)

    if len(ground_truth) != 1378:
        raise ValueError(f"Expected 1,378 reference entries, found {len(ground_truth)}")
    if len(eligible) != 498:
        raise ValueError(f"Expected 498 protocol-compatible entries, found {len(eligible)}")
    if not eligible["split"].eq("test").all():
        raise ValueError("Every benchmark entry must belong to the official test split")
    if not eligible["reader_label"].isin([-1.0, 0.0, 1.0]).all():
        raise ValueError("Protocol-compatible entries contain unsupported expert labels")

    eligible["label_index"] = eligible["finding"].map(LABEL_TO_INDEX).astype(int)
    eligible["current_binary_label"] = eligible["raw_label"].map(
        {-1.0: 1, 0.0: 0, 1.0: 1}
    ).astype(int)
    eligible["reference_binary_label"] = eligible["reader_label"].map(
        {-1.0: 1, 0.0: 0, 1.0: 1}
    ).astype(int)
    eligible["expected_binary_action"] = np.where(
        eligible["current_binary_label"].eq(eligible["reference_binary_label"]),
        "keep",
        "relabel",
    )
    eligible["entry_key"] = (
        eligible["subject_id"].astype(str)
        + "::"
        + eligible["study_id"].astype(str)
        + "::"
        + eligible["finding"].str.replace(" ", "_", regex=False)
    )
    if eligible["entry_key"].duplicated().any():
        raise ValueError("Prepared benchmark contains duplicate entry keys")

    reports = load_report_map(report_archive, sorted(eligible["study_id"].unique()))
    eligible["original_report_full"] = eligible["study_id"].map(reports)

    blinded_columns = [
        "entry_key",
        "subject_id",
        "study_id",
        "finding",
        "label_index",
        "raw_label",
        "current_binary_label",
        "original_report_full",
    ]
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
    blinded = eligible[blinded_columns].copy()
    reference = eligible[reference_columns].copy()
    if any(column.startswith("reader_") for column in blinded.columns):
        raise AssertionError("Reader reference leaked into blinded inputs")
    if "reference_binary_label" in blinded.columns:
        raise AssertionError("Binary reference leaked into blinded inputs")

    blinded_path = args.output_dir / "benchmark_entries_blinded.csv"
    reference_path = args.output_dir / "benchmark_reference_private.csv"
    blinded.to_csv(blinded_path, index=False)
    reference.to_csv(reference_path, index=False)

    metadata = {
        "protocol": "medpalm_external_suspicion_binary_uones_no_probability_v1",
        "ground_truth_sha256": sha256_file(ground_truth_path),
        "chexpert_sha256": sha256_file(chexpert_path),
        "split_sha256": sha256_file(split_path),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "total_reference_entries": int(len(ground_truth)),
        "protocol_compatible_entries": int(len(eligible)),
        "unique_studies": int(eligible["study_id"].nunique()),
        "unique_subjects": int(eligible["subject_id"].nunique()),
        "reference_binary_positive": int(eligible["reference_binary_label"].sum()),
        "reference_binary_negative": int(
            len(eligible) - eligible["reference_binary_label"].sum()
        ),
        "expected_keep": int(eligible["expected_binary_action"].eq("keep").sum()),
        "expected_relabel": int(eligible["expected_binary_action"].eq("relabel").sum()),
        "unanimous_entries": int(eligible["reader_agreement"].eq(1).sum()),
        "excluded_support_devices": int(
            merged["finding"].eq("Support Devices").sum()
        ),
        "excluded_raw_missing_current_schema": int(
            (merged["in_current_schema"] & ~merged["raw_valid"]).sum()
        ),
        "blinded_columns": blinded_columns,
        "reference_columns": reference_columns,
    }
    (args.output_dir / "preparation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def build_user_prompt(row: dict[str, Any]) -> str:
    return f"""\
Target label: {row['finding']}
Current binary label (decision target): {int(row['current_binary_label'])}
Binary meaning: 1 = positive under U-Ones; 0 = negative
Raw CheXpert label (provenance only): {row['raw_label']}
Fixed projection: raw -1 or 1 -> binary 1; raw 0 -> binary 0

Original report:
{row['original_report_full']}
""".strip()


def validate_payload(payload: dict[str, Any]) -> dict[str, str]:
    mismatch = str(payload.get("mismatch_type", "")).strip()
    note = str(payload.get("analysis_note", "")).strip()
    recommendation = str(payload.get("recommended_action", "")).strip()
    if mismatch not in EXPECTED_RECOMMENDATION:
        raise ValueError(f"Invalid mismatch_type: {mismatch!r}")
    if recommendation not in set(EXPECTED_RECOMMENDATION.values()):
        raise ValueError(f"Invalid recommended_action: {recommendation!r}")
    if not note:
        raise ValueError("analysis_note is empty")
    return {
        "mismatch_type": mismatch,
        "analysis_note": note,
        "recommended_action": recommendation,
    }


def real_review(
    row: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
    temperature: float,
    max_completion_tokens: int,
) -> dict[str, str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
    }
    response = requests.post(
        urljoin(base_url.rstrip("/") + "/", "chat/completions"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if response.status_code >= 400:
        message = response.text.strip()[:500] or f"HTTP {response.status_code}"
        raise RuntimeError(f"LLM request failed with status {response.status_code}: {message}")
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected LLM response: {json.dumps(data)[:500]}") from exc
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    if not isinstance(content, str):
        raise RuntimeError(f"Unexpected response content type: {type(content).__name__}")
    return validate_payload(json.loads(content))


def mock_review(row: dict[str, Any], **_: Any) -> dict[str, str]:
    return {
        "mismatch_type": "chexpert_right_cl_wrong",
        "analysis_note": (
            f'The mock reviewer keeps {row["finding"]}. '
            '"Mock report evidence" is used only for pipeline validation.'
        ),
        "recommended_action": "keep_chexpert",
    }


def process_review(
    row: dict[str, Any],
    review_fn: Callable[..., dict[str, str]],
    review_kwargs: dict[str, Any],
    retry_limit: int,
) -> tuple[bool, dict[str, Any]]:
    last_error = ""
    for attempt in range(1, retry_limit + 2):
        try:
            result = review_fn(row, **review_kwargs)
            return True, {**row, **result, "attempt_count": attempt}
        except Exception as exc:
            last_error = str(exc)
            if attempt <= retry_limit:
                time.sleep(0.2 * attempt)
    return False, {**row, "error_message": last_error, "attempt_count": retry_limit + 1}


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_errors(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def review_benchmark(args: argparse.Namespace) -> None:
    blinded_path = args.output_dir / "benchmark_entries_blinded.csv"
    if not blinded_path.exists():
        raise FileNotFoundError("Run prepare before review")
    blinded = pd.read_csv(blinded_path)
    if len(blinded) != 498 or blinded["entry_key"].duplicated().any():
        raise ValueError("Blinded benchmark must contain 498 unique entries")
    prohibited = {"reader_label", "reader_agreement", "reference_binary_label"}
    if prohibited.intersection(blinded.columns):
        raise ValueError("Reference columns leaked into blinded review input")

    results_path = args.output_dir / "review_results_blinded.csv"
    errors_path = args.output_dir / "review_errors.csv"
    completed: set[str] = set()
    if args.resume and results_path.exists() and results_path.stat().st_size:
        existing = pd.read_csv(results_path)
        if existing["entry_key"].duplicated().any():
            raise ValueError("Existing result file contains duplicate entry keys")
        completed = set(existing["entry_key"].astype(str))

    errors_by_key: dict[str, dict[str, Any]] = {}
    if args.resume and errors_path.exists() and errors_path.stat().st_size:
        previous_errors = pd.read_csv(errors_path)
        errors_by_key = {
            str(row["entry_key"]): row.to_dict()
            for _, row in previous_errors.iterrows()
            if str(row["entry_key"]) not in completed
        }

    pending = [
        row.to_dict()
        for _, row in blinded.iterrows()
        if str(row["entry_key"]) not in completed
    ]
    print(f"Benchmark entries: {len(blinded)}")
    print(f"Completed via resume: {len(completed)}")
    print(f"Pending: {len(pending)}")
    if not pending:
        return

    if args.llm_mode == "real":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        model = os.environ.get("OPENAI_MODEL", "").strip()
        if not api_key or not base_url or not model:
            raise RuntimeError("OPENAI_API_KEY, OPENAI_BASE_URL and OPENAI_MODEL are required")
        review_fn = real_review
        review_kwargs = {
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
            "timeout": args.request_timeout_seconds,
            "temperature": args.temperature,
            "max_completion_tokens": args.max_completion_tokens,
        }
    else:
        model = "mock"
        review_fn = mock_review
        review_kwargs = {}

    result_fields = list(blinded.columns) + [
        "mismatch_type",
        "analysis_note",
        "recommended_action",
        "attempt_count",
    ]
    error_fields = list(blinded.columns) + ["error_message", "attempt_count"]
    for start in range(0, len(pending), max(1, args.batch_size)):
        batch = pending[start : start + max(1, args.batch_size)]
        print(
            f"Processing batch {start // max(1, args.batch_size) + 1}/"
            f"{(len(pending) + max(1, args.batch_size) - 1) // max(1, args.batch_size)}"
        )
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            futures = {
                executor.submit(
                    process_review,
                    row,
                    review_fn,
                    review_kwargs,
                    args.retry_limit,
                ): str(row["entry_key"])
                for row in batch
            }
            for future in as_completed(futures):
                success, output = future.result()
                key = str(output["entry_key"])
                if success:
                    append_csv(results_path, output, result_fields)
                    completed.add(key)
                    errors_by_key.pop(key, None)
                else:
                    errors_by_key[key] = output
        write_errors(errors_path, list(errors_by_key.values()), error_fields)

    run_metadata = {
        "model": model,
        "temperature": args.temperature,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "completed": len(completed),
        "remaining_errors": len(errors_by_key),
    }
    (args.output_dir / "review_run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_metadata, indent=2))


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def metric_row(frame: pd.DataFrame, cohort: str) -> dict[str, Any]:
    n = len(frame)
    baseline_correct = frame["baseline_correct"].astype(bool)
    retained = frame["actual_action"].ne("mask")
    post_correct = frame["post_correct"].astype(bool)
    expected_relabel = frame["expected_binary_action"].eq("relabel")
    expected_keep = frame["expected_binary_action"].eq("keep")
    actual_relabel = frame["actual_action"].eq("relabel")
    actual_mask = frame["actual_action"].eq("mask")
    correctly_relabelled = expected_relabel & actual_relabel
    harmful_flip = expected_keep & actual_relabel
    return {
        "cohort": cohort,
        "n_entries": n,
        "n_subjects": int(frame["subject_id"].nunique()),
        "baseline_binary_accuracy": baseline_correct.mean(),
        "expected_relabel_count": int(expected_relabel.sum()),
        "actual_keep_count": int(frame["actual_action"].eq("keep").sum()),
        "actual_relabel_count": int(actual_relabel.sum()),
        "actual_mask_count": int(actual_mask.sum()),
        "coverage": retained.mean(),
        "retained_binary_accuracy": safe_div(post_correct.sum(), retained.sum()),
        "coverage_adjusted_correct_mass": post_correct.mean(),
        "delta_adjusted_vs_baseline": post_correct.mean() - baseline_correct.mean(),
        "binary_action_accuracy": frame["binary_action_correct"].mean(),
        "relabel_precision": safe_div(correctly_relabelled.sum(), actual_relabel.sum()),
        "relabel_recall": safe_div(correctly_relabelled.sum(), expected_relabel.sum()),
        "harmful_flip_rate": safe_div(harmful_flip.sum(), expected_keep.sum()),
        "mask_rate": actual_mask.mean(),
        "mask_rate_expected_keep": safe_div(
            (actual_mask & expected_keep).sum(), expected_keep.sum()
        ),
        "mask_rate_expected_relabel": safe_div(
            (actual_mask & expected_relabel).sum(), expected_relabel.sum()
        ),
        "response_field_conflicts": int((~frame["response_consistent"]).sum()),
    }


def bootstrap_intervals(
    frame: pd.DataFrame,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    metrics = [
        "baseline_binary_accuracy",
        "coverage",
        "retained_binary_accuracy",
        "coverage_adjusted_correct_mass",
        "delta_adjusted_vs_baseline",
        "binary_action_accuracy",
        "relabel_precision",
        "relabel_recall",
        "harmful_flip_rate",
        "mask_rate",
    ]
    subjects = frame["subject_id"].drop_duplicates().to_numpy()
    groups = {subject: group for subject, group in frame.groupby("subject_id", sort=False)}
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in metrics}
    for _ in range(replicates):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        replicate = pd.concat([groups[subject] for subject in sampled], ignore_index=True)
        row = metric_row(replicate, "bootstrap")
        for metric in metrics:
            value = row[metric]
            if not pd.isna(value):
                values[metric].append(float(value))
    point = metric_row(frame, "all")
    rows = []
    for metric in metrics:
        distribution = np.asarray(values[metric], dtype=float)
        lower = float("nan")
        upper = float("nan")
        if len(distribution):
            lower = float(np.quantile(distribution, 0.025))
            upper = float(np.quantile(distribution, 0.975))
        rows.append(
            {
                "metric": metric,
                "point_estimate": point[metric],
                "ci_lower_95": lower,
                "ci_upper_95": upper,
                "bootstrap_valid_replicates": len(distribution),
                "cluster": "subject_id",
            }
        )
    return pd.DataFrame(rows)


def evaluate_benchmark(args: argparse.Namespace) -> None:
    blinded_path = args.output_dir / "benchmark_entries_blinded.csv"
    reference_path = args.output_dir / "benchmark_reference_private.csv"
    results_path = args.output_dir / "review_results_blinded.csv"
    errors_path = args.output_dir / "review_errors.csv"
    for path in [blinded_path, reference_path, results_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    blinded = pd.read_csv(blinded_path)
    reference = pd.read_csv(reference_path)
    results = pd.read_csv(results_path)
    if len(results) != 498 or results["entry_key"].duplicated().any():
        raise ValueError(f"Evaluation requires 498 unique results; found {len(results)}")
    if errors_path.exists() and errors_path.stat().st_size:
        errors = pd.read_csv(errors_path)
        if len(errors):
            raise ValueError(f"Evaluation blocked by {len(errors)} unresolved review errors")

    result_columns = [
        "entry_key",
        "mismatch_type",
        "recommended_action",
        "analysis_note",
        "attempt_count",
    ]
    evaluation = blinded.merge(
        results[result_columns],
        on="entry_key",
        how="left",
        validate="one_to_one",
    ).merge(
        reference,
        on=["entry_key", "subject_id", "study_id", "finding"],
        how="left",
        validate="one_to_one",
    )
    if evaluation.isna().any().any():
        missing_columns = evaluation.columns[evaluation.isna().any()].tolist()
        raise ValueError(f"Evaluation contains missing values in: {missing_columns}")

    evaluation["response_consistent"] = evaluation["recommended_action"].eq(
        evaluation["mismatch_type"].map(EXPECTED_RECOMMENDATION)
    )
    evaluation["actual_action"] = evaluation["mismatch_type"].map(ACTION_BY_MISMATCH)
    evaluation.loc[~evaluation["response_consistent"], "actual_action"] = "mask"
    evaluation["baseline_correct"] = evaluation["current_binary_label"].eq(
        evaluation["reference_binary_label"]
    )
    evaluation["post_binary_label"] = evaluation["current_binary_label"].astype(float)
    relabel = evaluation["actual_action"].eq("relabel")
    evaluation.loc[relabel, "post_binary_label"] = (
        1 - evaluation.loc[relabel, "current_binary_label"]
    )
    evaluation.loc[evaluation["actual_action"].eq("mask"), "post_binary_label"] = np.nan
    evaluation["post_correct"] = (
        evaluation["post_binary_label"].notna()
        & evaluation["post_binary_label"].eq(evaluation["reference_binary_label"])
    )
    evaluation["binary_action_correct"] = evaluation["actual_action"].eq(
        evaluation["expected_binary_action"]
    )

    summary = pd.DataFrame(
        [
            metric_row(evaluation, "all_adjudicated"),
            metric_row(
                evaluation[evaluation["reader_agreement"].eq(1)].copy(),
                "three_reader_unanimous",
            ),
        ]
    )
    per_finding = pd.DataFrame(
        [
            metric_row(group, finding)
            for finding, group in evaluation.groupby("finding", sort=True)
        ]
    )
    confusion = pd.crosstab(
        evaluation["expected_binary_action"],
        evaluation["actual_action"],
        rownames=["expected_binary_action"],
        colnames=["actual_action"],
        dropna=False,
    )
    intervals = bootstrap_intervals(
        evaluation,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )

    evaluation.to_csv(args.output_dir / "row_level_evaluation_private.csv", index=False)
    summary.to_csv(args.output_dir / "benchmark_summary.csv", index=False)
    per_finding.to_csv(args.output_dir / "benchmark_per_finding.csv", index=False)
    confusion.to_csv(args.output_dir / "benchmark_action_confusion.csv")
    intervals.to_csv(args.output_dir / "benchmark_subject_cluster_bootstrap_ci.csv", index=False)

    payload = {
        "protocol": "medpalm_external_suspicion_binary_uones_no_probability_v1",
        "summary": summary.to_dict(orient="records"),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "inference_cluster": "subject_id",
    }
    (args.output_dir / "benchmark_evaluation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\nAction confusion:")
    print(confusion.to_string())


def verify_benchmark(args: argparse.Namespace) -> None:
    metadata_path = args.output_dir / "preparation_metadata.json"
    results_path = args.output_dir / "review_results_blinded.csv"
    errors_path = args.output_dir / "review_errors.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["protocol_compatible_entries"] != 498:
        raise ValueError("Preparation metadata does not describe 498 entries")
    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    errors = pd.read_csv(errors_path) if errors_path.exists() and errors_path.stat().st_size else pd.DataFrame()
    if len(results) != 498 or (not results.empty and results["entry_key"].duplicated().any()):
        raise ValueError(f"Expected 498 unique review results, found {len(results)}")
    if len(errors):
        raise ValueError(f"Review still has {len(errors)} unresolved errors")
    if hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest() != metadata["system_prompt_sha256"]:
        raise ValueError("System prompt differs from preparation metadata")
    print("Benchmark verification passed: 498 unique results, zero unresolved errors.")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        prepare_benchmark(args)
    elif args.mode == "review":
        review_benchmark(args)
    elif args.mode == "evaluate":
        evaluate_benchmark(args)
    else:
        verify_benchmark(args)


if __name__ == "__main__":
    main()
