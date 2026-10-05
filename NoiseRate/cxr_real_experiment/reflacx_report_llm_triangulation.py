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
    "Lung Lesion",
    "Pneumothorax",
]
EXPECTED_ENTRIES = 3172
EXPECTED_STUDIES = 1646
EXPECTED_SUBJECTS = 1386

SYSTEM_PROMPT = """\
You are independently extracting one target chest X-ray finding from an original radiology
report. You are not given the current dataset label, any model prediction, or any image-reader
reference. Use only the supplied report text.

Return one of three report-level judgments:
- report_positive: the report supports that the target finding is present;
- report_negative: the report explicitly supports that the target finding is absent;
- report_ambiguous: the report is uncertain, conflicting, or does not provide enough stable
  evidence to assign positive or negative.

Do not infer the target from unrelated findings or outside medical knowledge. If positive and
negative statements conflict, or the wording is explicitly uncertain/equivocal, use
report_ambiguous. The task is to interpret what the report says, not to diagnose the image.

Return valid JSON only, with exactly these keys:
- report_label
- analysis_note

Requirements for analysis_note:
- Write 2 to 4 sentences.
- Quote the key report text directly.
- Explain briefly why that text supports the selected report_label.
""".strip()

ALLOWED_REPORT_LABELS = {
    "report_positive": 1.0,
    "report_negative": 0.0,
    "report_ambiguous": np.nan,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blind report-only LLM triangulation for the REFLACX Phase-3 cohort."
    )
    parser.add_argument("mode", choices=["prepare", "review", "verify", "evaluate"])
    parser.add_argument("--blind-cohort", type=Path)
    parser.add_argument("--private-reference", type=Path)
    parser.add_argument("--entry-evidence", type=Path)
    parser.add_argument("--chexpert-csv", type=Path)
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
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    return parser.parse_args()


def require_path(value: Path | None, name: str) -> Path:
    if value is None:
        raise ValueError(f"{name} is required")
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
    frame = frame.copy()
    for column in ["subject_id", "study_id"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    return frame


def load_reports(report_archive: Path, study_ids: list[int]) -> dict[int, str]:
    reports: dict[int, str] = {}
    with zipfile.ZipFile(report_archive) as archive:
        members = {
            Path(name).name: name for name in archive.namelist() if name.endswith(".txt")
        }
        for study_id in study_ids:
            basename = f"s{study_id}.txt"
            if basename not in members:
                raise ValueError(f"Missing report for study {study_id}")
            report = archive.read(members[basename]).decode("utf-8", errors="replace").strip()
            if not report:
                raise ValueError(f"Empty report for study {study_id}")
            reports[study_id] = report
    return reports


def prepare(args: argparse.Namespace) -> None:
    blind_path = require_path(args.blind_cohort, "--blind-cohort")
    reference_path = require_path(args.private_reference, "--private-reference")
    evidence_path = require_path(args.entry_evidence, "--entry-evidence")
    chexpert_path = require_path(args.chexpert_csv, "--chexpert-csv")
    report_path = require_path(args.report_archive, "--report-archive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    blind = normalize_ids(pd.read_csv(blind_path))
    reference = normalize_ids(pd.read_csv(reference_path))
    evidence = normalize_ids(pd.read_csv(evidence_path))
    chexpert = normalize_ids(pd.read_csv(chexpert_path))

    if len(blind) != EXPECTED_STUDIES or blind["subject_id"].nunique() != EXPECTED_SUBJECTS:
        raise ValueError("Unexpected REFLACX blind cohort size")
    if not set(LABEL_NAMES).issubset(chexpert.columns):
        raise ValueError("CheXpert table is missing one or more strict six labels")

    valid = evidence[evidence["valid_label"].eq(1)].copy()
    if len(valid) != EXPECTED_ENTRIES:
        raise ValueError(f"Expected {EXPECTED_ENTRIES} valid entries, found {len(valid)}")
    if valid.duplicated(["pool_row_id", "label_name"]).any():
        raise ValueError("Duplicate valid evidence entry")
    if set(valid["label_name"].unique()) != set(LABEL_NAMES):
        raise ValueError("Evidence labels do not match the strict six-label scope")

    raw_long = chexpert[["subject_id", "study_id", *LABEL_NAMES]].melt(
        id_vars=["subject_id", "study_id"],
        value_vars=LABEL_NAMES,
        var_name="label_name",
        value_name="current_raw_label",
    )
    if raw_long.duplicated(["subject_id", "study_id", "label_name"]).any():
        raise ValueError("CheXpert contains duplicate study-label rows")

    certainty_columns = [f"certainty__{label}" for label in LABEL_NAMES]
    if not set(certainty_columns).issubset(reference.columns):
        raise ValueError("Private REFLACX table is missing certainty columns")
    ref_long = reference[
        ["pool_row_id", "subject_id", "study_id", "dicom_id", *certainty_columns]
    ].melt(
        id_vars=["pool_row_id", "subject_id", "study_id", "dicom_id"],
        value_vars=certainty_columns,
        var_name="certainty_name",
        value_name="reflacx_certainty",
    )
    ref_long["label_name"] = ref_long["certainty_name"].str.removeprefix("certainty__")
    ref_long = ref_long.drop(columns="certainty_name")

    private = valid[
        [
            "pool_row_id",
            "subject_id",
            "study_id",
            "dicom_id",
            "label_index",
            "label_name",
            "binary_label",
            "pred_probability",
            "est_issue_entry",
            "selected_top20_candidate",
        ]
    ].merge(
        raw_long,
        on=["subject_id", "study_id", "label_name"],
        how="left",
        validate="many_to_one",
    ).merge(
        ref_long,
        on=["pool_row_id", "subject_id", "study_id", "dicom_id", "label_name"],
        how="left",
        validate="one_to_one",
    )
    if private[["current_raw_label", "reflacx_certainty"]].isna().any().any():
        raise ValueError("Private comparison table contains missing source labels")
    if not private["current_raw_label"].isin([-1.0, 0.0, 1.0]).all():
        raise ValueError("Current raw labels must be -1, 0, or 1")
    projected = private["current_raw_label"].map({-1.0: 1, 0.0: 0, 1.0: 1})
    if not projected.eq(private["binary_label"].astype(int)).all():
        raise ValueError("Current evidence does not match the fixed U-Ones projection")
    private = private.rename(columns={"binary_label": "current_binary_label"})
    private["current_binary_label"] = private["current_binary_label"].astype(int)
    private["reflacx_certainty"] = private["reflacx_certainty"].astype(int)
    private["entry_key"] = (
        private["subject_id"].astype(str)
        + "::"
        + private["study_id"].astype(str)
        + "::"
        + private["dicom_id"].astype(str)
        + "::"
        + private["label_name"].str.replace(" ", "_", regex=False)
    )
    if private["entry_key"].duplicated().any():
        raise ValueError("Duplicate entry keys")

    reports = load_reports(report_path, sorted(private["study_id"].unique()))
    api_inputs = private[
        ["entry_key", "subject_id", "study_id", "dicom_id", "label_name"]
    ].copy()
    api_inputs["original_report_full"] = api_inputs["study_id"].map(reports)
    if api_inputs["original_report_full"].isna().any():
        raise ValueError("Missing report after report join")

    prohibited_fragments = ["label", "certainty", "probability", "issue", "selected"]
    safe_bookkeeping = {"entry_key", "label_name"}
    leaked = [
        column
        for column in api_inputs.columns
        if column not in safe_bookkeeping
        and any(fragment in column.lower() for fragment in prohibited_fragments)
    ]
    if leaked:
        raise AssertionError(f"Private comparison fields leaked into API input: {leaked}")

    api_path = args.output_dir / "report_review_inputs_blinded.csv"
    private_path = args.output_dir / "three_source_comparison_private.csv"
    api_inputs.to_csv(api_path, index=False)
    private.to_csv(private_path, index=False)

    metadata = {
        "protocol": "reflacx_report_llm_triangulation_v1",
        "entries": int(len(private)),
        "studies": int(private["study_id"].nunique()),
        "subjects": int(private["subject_id"].nunique()),
        "labels": LABEL_NAMES,
        "api_input_columns": list(api_inputs.columns),
        "private_columns": list(private.columns),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "blind_cohort_sha256": sha256_file(blind_path),
        "private_reference_sha256": sha256_file(reference_path),
        "entry_evidence_sha256": sha256_file(evidence_path),
        "chexpert_sha256": sha256_file(chexpert_path),
        "api_inputs_sha256": sha256_file(api_path),
        "private_comparison_sha256": sha256_file(private_path),
    }
    (args.output_dir / "preparation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def build_user_prompt(row: dict[str, Any]) -> str:
    return f"""\
Target finding: {row['label_name']}

Original report:
{row['original_report_full']}
""".strip()


def validate_payload(payload: dict[str, Any]) -> dict[str, str]:
    report_label = str(payload.get("report_label", "")).strip()
    note = str(payload.get("analysis_note", "")).strip()
    if report_label not in ALLOWED_REPORT_LABELS:
        raise ValueError(f"Invalid report_label: {report_label!r}")
    if not note:
        raise ValueError("analysis_note is empty")
    return {"report_label": report_label, "analysis_note": note}


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
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
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
        "report_label": "report_ambiguous",
        "analysis_note": (
            f'The mock reviewer does not decide {row["label_name"]}. '
            '"Mock report evidence" is used only for pipeline validation.'
        ),
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


def review(args: argparse.Namespace) -> None:
    inputs_path = args.output_dir / "report_review_inputs_blinded.csv"
    metadata_path = args.output_dir / "preparation_metadata.json"
    if not inputs_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Run prepare before review")
    inputs = pd.read_csv(inputs_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if len(inputs) != EXPECTED_ENTRIES or inputs["entry_key"].duplicated().any():
        raise ValueError("Blinded review inputs must contain 3,172 unique entries")
    prohibited = {
        "current_raw_label",
        "current_binary_label",
        "reflacx_certainty",
        "pred_probability",
        "est_issue_entry",
        "selected_top20_candidate",
    }
    if prohibited.intersection(inputs.columns):
        raise ValueError("Private comparison columns leaked into API inputs")
    if sha256_file(inputs_path) != metadata["api_inputs_sha256"]:
        raise ValueError("Blinded API input changed after preparation")

    results_path = args.output_dir / "report_review_results_blinded.csv"
    errors_path = args.output_dir / "report_review_errors.csv"
    completed: set[str] = set()
    if args.resume and results_path.exists() and results_path.stat().st_size:
        existing = pd.read_csv(results_path)
        if existing["entry_key"].duplicated().any():
            raise ValueError("Existing review results contain duplicate keys")
        completed = set(existing["entry_key"].astype(str))

    errors_by_key: dict[str, dict[str, Any]] = {}
    if args.resume and errors_path.exists() and errors_path.stat().st_size:
        prior_errors = pd.read_csv(errors_path)
        errors_by_key = {
            str(row["entry_key"]): row.to_dict()
            for _, row in prior_errors.iterrows()
            if str(row["entry_key"]) not in completed
        }

    pending = [
        row.to_dict()
        for _, row in inputs.iterrows()
        if str(row["entry_key"]) not in completed
    ]
    print(f"Review entries: {len(inputs)}")
    print(f"Completed via resume: {len(completed)}")
    print(f"Pending: {len(pending)}")
    if not pending:
        return

    model = "mock"
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
        review_fn = mock_review
        review_kwargs = {}

    result_fields = list(inputs.columns) + ["report_label", "analysis_note", "attempt_count"]
    error_fields = list(inputs.columns) + ["error_message", "attempt_count"]
    batch_size = max(1, args.batch_size)
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        print(f"Processing batch {start // batch_size + 1}/{(len(pending) + batch_size - 1) // batch_size}")
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            futures = {
                executor.submit(
                    process_review, row, review_fn, review_kwargs, args.retry_limit
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
        "protocol": "reflacx_report_llm_triangulation_v1",
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


def verify(args: argparse.Namespace) -> None:
    inputs_path = args.output_dir / "report_review_inputs_blinded.csv"
    private_path = args.output_dir / "three_source_comparison_private.csv"
    metadata_path = args.output_dir / "preparation_metadata.json"
    results_path = args.output_dir / "report_review_results_blinded.csv"
    errors_path = args.output_dir / "report_review_errors.csv"
    for path in [inputs_path, private_path, metadata_path, results_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    results = pd.read_csv(results_path)
    errors = (
        pd.read_csv(errors_path)
        if errors_path.exists() and errors_path.stat().st_size
        else pd.DataFrame()
    )
    if len(results) != EXPECTED_ENTRIES or results["entry_key"].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_ENTRIES} unique results, found {len(results)}")
    if len(errors):
        raise ValueError(f"Review has {len(errors)} unresolved errors")
    if sha256_file(inputs_path) != metadata["api_inputs_sha256"]:
        raise ValueError("Blinded inputs changed after preparation")
    if sha256_file(private_path) != metadata["private_comparison_sha256"]:
        raise ValueError("Private comparison table changed after preparation")
    if hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest() != metadata["system_prompt_sha256"]:
        raise ValueError("System prompt changed after preparation")
    (args.output_dir / ".review_complete").write_text("complete\n", encoding="utf-8")
    print("Review verification passed: 3,172 unique results and zero unresolved errors.")


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def add_scope_labels(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    out = frame.copy()
    if scope == "certainty_ge3":
        out["reflacx_binary_label"] = out["reflacx_certainty"].ge(3).astype(int)
    elif scope == "certainty_ge4_excluding3":
        out = out[out["reflacx_certainty"].ne(3)].copy()
        out["reflacx_binary_label"] = out["reflacx_certainty"].ge(4).astype(int)
    else:
        raise ValueError(scope)
    return out


def classify_pattern(row: pd.Series) -> str:
    if pd.isna(row["llm_binary_label"]):
        return "llm_report_ambiguous"
    llm = int(row["llm_binary_label"])
    current = int(row["current_binary_label"])
    reference = int(row["reflacx_binary_label"])
    if llm == current == reference:
        return "all_three_agree"
    if llm == reference and current != reference:
        return "llm_reflacx_agree_chexpert_differs"
    if llm == current and reference != current:
        return "llm_chexpert_agree_reflacx_differs"
    if current == reference and llm != current:
        return "chexpert_reflacx_agree_llm_differs"
    raise AssertionError("Unexpected binary three-way pattern")


def metric_row(frame: pd.DataFrame, cohort: str) -> dict[str, Any]:
    n = len(frame)
    definite = frame["llm_binary_label"].notna()
    current_ref = frame["current_binary_label"].eq(frame["reflacx_binary_label"])
    llm_current = definite & frame["llm_binary_label"].eq(frame["current_binary_label"])
    llm_ref = definite & frame["llm_binary_label"].eq(frame["reflacx_binary_label"])
    disagreement = ~current_ref
    dis_n = int(disagreement.sum())
    patterns = frame["three_way_pattern"].value_counts()
    return {
        "cohort": cohort,
        "n_entries": n,
        "n_subjects": int(frame["subject_id"].nunique()),
        "chexpert_reflacx_agreement": current_ref.mean(),
        "llm_definite_coverage": definite.mean(),
        "llm_chexpert_agreement_retained": safe_div(llm_current.sum(), definite.sum()),
        "llm_chexpert_adjusted_agreement": llm_current.mean(),
        "llm_reflacx_accuracy_retained": safe_div(llm_ref.sum(), definite.sum()),
        "llm_reflacx_adjusted_correct_mass": llm_ref.mean(),
        "delta_llm_adjusted_vs_chexpert_reference": llm_ref.mean() - current_ref.mean(),
        "chexpert_reflacx_disagreement_n": dis_n,
        "llm_sides_with_reflacx_on_disagreement_n": int(
            patterns.get("llm_reflacx_agree_chexpert_differs", 0)
        ),
        "llm_sides_with_reflacx_on_disagreement_rate": safe_div(
            patterns.get("llm_reflacx_agree_chexpert_differs", 0), dis_n
        ),
        "llm_sides_with_chexpert_on_disagreement_n": int(
            patterns.get("llm_chexpert_agree_reflacx_differs", 0)
        ),
        "llm_sides_with_chexpert_on_disagreement_rate": safe_div(
            patterns.get("llm_chexpert_agree_reflacx_differs", 0), dis_n
        ),
        "llm_ambiguous_on_disagreement_n": int(
            (disagreement & ~definite).sum()
        ),
        "llm_ambiguous_on_disagreement_rate": safe_div(
            (disagreement & ~definite).sum(), dis_n
        ),
        "all_three_agree_n": int(patterns.get("all_three_agree", 0)),
        "llm_reflacx_agree_chexpert_differs_n": int(
            patterns.get("llm_reflacx_agree_chexpert_differs", 0)
        ),
        "llm_chexpert_agree_reflacx_differs_n": int(
            patterns.get("llm_chexpert_agree_reflacx_differs", 0)
        ),
        "chexpert_reflacx_agree_llm_differs_n": int(
            patterns.get("chexpert_reflacx_agree_llm_differs", 0)
        ),
        "llm_report_ambiguous_n": int(patterns.get("llm_report_ambiguous", 0)),
    }


BOOTSTRAP_METRICS = [
    "chexpert_reflacx_agreement",
    "llm_definite_coverage",
    "llm_chexpert_agreement_retained",
    "llm_reflacx_accuracy_retained",
    "llm_reflacx_adjusted_correct_mass",
    "delta_llm_adjusted_vs_chexpert_reference",
    "llm_sides_with_reflacx_on_disagreement_rate",
    "llm_sides_with_chexpert_on_disagreement_rate",
    "llm_ambiguous_on_disagreement_rate",
]


def bootstrap_intervals(frame: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    subjects = frame["subject_id"].drop_duplicates().to_numpy()
    groups = {subject: group for subject, group in frame.groupby("subject_id", sort=False)}
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(replicates):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        replicate = pd.concat([groups[subject] for subject in sampled], ignore_index=True)
        row = metric_row(replicate, "bootstrap")
        for metric in BOOTSTRAP_METRICS:
            value = row[metric]
            if not pd.isna(value):
                values[metric].append(float(value))
    point = metric_row(frame, "certainty_ge3")
    rows = []
    for metric in BOOTSTRAP_METRICS:
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


def evaluate(args: argparse.Namespace) -> None:
    marker = args.output_dir / ".review_complete"
    private_path = args.output_dir / "three_source_comparison_private.csv"
    results_path = args.output_dir / "report_review_results_blinded.csv"
    if not marker.exists():
        raise FileNotFoundError("Review verification marker is missing")
    private = pd.read_csv(private_path)
    results = pd.read_csv(results_path)
    result_columns = ["entry_key", "report_label", "analysis_note", "attempt_count"]
    evaluation = private.merge(
        results[result_columns], on="entry_key", how="left", validate="one_to_one"
    )
    if len(evaluation) != EXPECTED_ENTRIES or evaluation[result_columns].isna().any().any():
        raise ValueError("Incomplete private evaluation join")
    if not evaluation["report_label"].isin(ALLOWED_REPORT_LABELS).all():
        raise ValueError("Unexpected report labels")
    evaluation["llm_binary_label"] = evaluation["report_label"].map(ALLOWED_REPORT_LABELS)
    evaluation["derived_refinement_action"] = "mask"
    definite = evaluation["llm_binary_label"].notna()
    evaluation.loc[
        definite & evaluation["llm_binary_label"].eq(evaluation["current_binary_label"]),
        "derived_refinement_action",
    ] = "keep"
    evaluation.loc[
        definite & evaluation["llm_binary_label"].ne(evaluation["current_binary_label"]),
        "derived_refinement_action",
    ] = "relabel"

    scoped_frames: dict[str, pd.DataFrame] = {}
    summary_rows = []
    per_label_rows = []
    pattern_rows = []
    cl_rows = []
    for scope in ["certainty_ge3", "certainty_ge4_excluding3"]:
        scoped = add_scope_labels(evaluation, scope)
        scoped["three_way_pattern"] = scoped.apply(classify_pattern, axis=1)
        scoped_frames[scope] = scoped
        summary_rows.append(metric_row(scoped, scope))
        for label, group in scoped.groupby("label_name", sort=True):
            per_label_rows.append(metric_row(group, f"{scope}::{label}"))
        counts = scoped["three_way_pattern"].value_counts()
        for pattern, count in counts.items():
            pattern_rows.append(
                {
                    "scope": scope,
                    "three_way_pattern": pattern,
                    "count": int(count),
                    "fraction": float(count / len(scoped)),
                }
            )
        for name, mask in {
            "all_entries": pd.Series(True, index=scoped.index),
            "cl_issue_flagged": scoped["est_issue_entry"].eq(1),
            "cl_issue_not_flagged": scoped["est_issue_entry"].eq(0),
            "selected_top20_candidate": scoped["selected_top20_candidate"].eq(1),
        }.items():
            if mask.any():
                cl_rows.append(metric_row(scoped[mask].copy(), f"{scope}::{name}"))

    primary = scoped_frames["certainty_ge3"]
    intervals = bootstrap_intervals(
        primary, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed
    )
    primary.to_csv(args.output_dir / "row_level_triangulation_private.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "triangulation_summary.csv", index=False)
    pd.DataFrame(per_label_rows).to_csv(args.output_dir / "triangulation_per_label.csv", index=False)
    pd.DataFrame(pattern_rows).to_csv(args.output_dir / "three_way_pattern_counts.csv", index=False)
    pd.DataFrame(cl_rows).to_csv(args.output_dir / "triangulation_by_cl_status.csv", index=False)
    intervals.to_csv(args.output_dir / "subject_cluster_bootstrap_ci.csv", index=False)

    payload = {
        "protocol": "reflacx_report_llm_triangulation_v1",
        "interpretation": {
            "llm_reflacx_agree_chexpert_differs": (
                "evidence consistent with a CheXpert report-extraction mismatch"
            ),
            "llm_chexpert_agree_reflacx_differs": (
                "evidence consistent with report/image-reader mismatch"
            ),
            "warning": "Three-way agreement does not adjudicate definitive truth.",
        },
        "summary": summary_rows,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "inference_cluster": "subject_id",
    }
    (args.output_dir / "triangulation_evaluation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (args.output_dir / ".evaluation_complete").write_text("complete\n", encoding="utf-8")
    print(pd.DataFrame(summary_rows).to_string(index=False))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "review":
        review(args)
    elif args.mode == "verify":
        verify(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
