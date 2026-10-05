#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import textwrap
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin
from typing import Any

import pandas as pd
import requests


SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are reviewing a suspicious binary chest X-ray label entry. The decision target is the
    current binary label used by confident-learning detection and model training, not the raw
    four-state CheXpert output. Under the fixed U-Ones projection, raw -1 (uncertain) and raw 1
    both map to binary 1, while raw 0 maps to binary 0. The raw value is supplied only as provenance.

    Your job is to judge, using only the original radiology report text, whether the suspicious
    entry is more likely:
    1. a case where the current binary label is wrong and the confident-learning suspicion is useful,
    2. a case where the current binary label is likely correct and the confident-learning suspicion is wrong,
    3. or a case where the report is too ambiguous or internally conflicting to decide.

    You must only use the original report text as evidence.
    Do not use outside medical knowledge.
    Do not assume the model is correct just because its predicted probability is high.
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

    The historical word "chexpert" in these output values refers to the current derived binary
    label. It does not mean that a raw -1 value should be judged or excluded as a separate class.

    Allowed values for recommended_action:
    - keep_chexpert
    - review_or_relabel
    - mark_ambiguous

    Requirements for analysis_note:
    - Write 2 to 4 sentences.
    - State whether the report supports the current binary label, supports the suspicious-model side,
      or is ambiguous.
    - Quote the key report text directly.
    - If there is a conflict, explicitly say which statement conflicts with which statement, such as
      a findings sentence versus an impression sentence.
    - Briefly explain why the quoted text supports the final mismatch_type.
    """
).strip()


MISMATCH_TYPES = {
    "chexpert_wrong_cl_right",
    "chexpert_right_cl_wrong",
    "report_ambiguous",
}
RECOMMENDED_ACTIONS = {
    "keep_chexpert",
    "review_or_relabel",
    "mark_ambiguous",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run entry-level suspicious-label review with a mock LLM pipeline."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--report-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--retry-limit", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--llm-mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--mock-latency-ms", type=int, default=50)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=400)
    parser.add_argument(
        "--mock-fail-once-indexes",
        type=str,
        default="",
        help="Comma-separated 0-based selected-row indexes that fail once before succeeding.",
    )
    parser.add_argument(
        "--mock-always-fail-indexes",
        type=str,
        default="",
        help="Comma-separated 0-based selected-row indexes that always fail.",
    )
    return parser.parse_args()


def parse_index_list(value: str) -> set[int]:
    if not value.strip():
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def entry_key(row: pd.Series | dict[str, Any]) -> str:
    return f"{row['pool_row_id']}::{row['label_index']}"


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def extract_report_map(report_archive: Path, study_ids: list[int]) -> dict[int, str]:
    reports: dict[int, str] = {}
    with zipfile.ZipFile(report_archive, "r") as zf:
        names = zf.namelist()
        index = {name.rsplit("/", 1)[-1]: name for name in names if name.endswith(".txt")}
        for study_id in study_ids:
            key = f"s{study_id}.txt"
            if key in index:
                reports[study_id] = zf.read(index[key]).decode("utf-8", errors="replace")
            else:
                reports[study_id] = ""
    return reports


def normalize_report_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def split_into_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def extract_section_text(report_text: str, section_name: str) -> str:
    pattern = re.compile(
        rf"{section_name}\s*:\s*(.*?)(?=\n[A-Z][A-Z /\-]+:|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(report_text)
    if not match:
        return ""
    return normalize_report_text(match.group(1))


def quote_text(text: str, max_len: int = 160) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def build_user_prompt(row: dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""\
        Target label: {row['target_label']}
        Current binary label (decision target): {row['current_binary_label']}
        Binary meaning: 1 = positive under U-Ones; 0 = negative
        Raw CheXpert label (provenance only): {row['chexpert_raw_label']}
        Fixed projection: raw -1 or 1 -> binary 1; raw 0 -> binary 0
        Model OOF probability: {row['model_oof_prob']}

        Original report:
        {row['original_report_full']}
        """
    ).strip()


def validate_review_payload(payload: dict[str, Any]) -> dict[str, str]:
    mismatch_type = str(payload.get("mismatch_type", "")).strip()
    analysis_note = str(payload.get("analysis_note", "")).strip()
    recommended_action = str(payload.get("recommended_action", "")).strip()

    if mismatch_type not in MISMATCH_TYPES:
        raise ValueError(f"Invalid mismatch_type: {mismatch_type!r}")
    if recommended_action not in RECOMMENDED_ACTIONS:
        raise ValueError(f"Invalid recommended_action: {recommended_action!r}")
    if not analysis_note:
        raise ValueError("analysis_note is empty")

    return {
        "mismatch_type": mismatch_type,
        "analysis_note": analysis_note,
        "recommended_action": recommended_action,
    }


@dataclass(frozen=True)
class ReviewTask:
    selected_index: int
    source_row: dict[str, Any]
    llm_input: dict[str, Any]

    @property
    def key(self) -> str:
        return entry_key(self.source_row)


class MockReviewClient:
    def __init__(
        self,
        *,
        fail_once_indexes: set[int],
        always_fail_indexes: set[int],
        latency_ms: int,
    ) -> None:
        self.fail_once_indexes = fail_once_indexes
        self.always_fail_indexes = always_fail_indexes
        self.latency_ms = latency_ms

    def review(self, task: ReviewTask, attempt_number: int) -> dict[str, str]:
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

        if task.selected_index in self.always_fail_indexes:
            raise RuntimeError(
                f"Mock permanent failure for selected_index={task.selected_index}"
            )
        if task.selected_index in self.fail_once_indexes and attempt_number == 1:
            raise RuntimeError(
                f"Mock transient failure for selected_index={task.selected_index}"
            )

        row = task.llm_input
        target_label = str(row["target_label"])
        current_binary = int(float(row["current_binary_label"]))
        model_prob = float(row["model_oof_prob"])
        report_text = str(row["original_report_full"])

        findings_text = extract_section_text(report_text, "findings")
        impression_text = extract_section_text(report_text, "impression")
        findings_sentence = split_into_sentences(findings_text)
        impression_sentence = split_into_sentences(impression_text)

        findings_quote = quote_text(findings_sentence[0]) if findings_sentence else ""
        impression_quote = quote_text(impression_sentence[0]) if impression_sentence else ""
        fallback_sentence = split_into_sentences(report_text)
        fallback_quote = quote_text(fallback_sentence[0]) if fallback_sentence else ""

        if 0.2 < model_prob < 0.8:
            mismatch_type = "report_ambiguous"
            analysis_note = (
                f'Mock review: the report is treated as ambiguous for {target_label}. '
                f'Findings excerpt: "{findings_quote or fallback_quote}". '
                f'Impression excerpt: "{impression_quote or fallback_quote}". '
                "Because the mock pipeline uses a mid-range model probability, it marks this entry as ambiguous."
            )
            recommended_action = "mark_ambiguous"
        else:
            suspicious_model_side = (
                (current_binary == 0 and model_prob >= 0.8)
                or (current_binary == 1 and model_prob <= 0.2)
            )
            if suspicious_model_side:
                mismatch_type = "chexpert_wrong_cl_right"
                recommended_action = "review_or_relabel"
                analysis_note = (
                    f'Mock review: the report is treated as supporting the suspicious-model side for {target_label}. '
                    f'Quoted report text: "{findings_quote or impression_quote or fallback_quote}". '
                    f'The mock rule combines current binary label {current_binary} with model probability '
                    f'{model_prob:.4f}, so this entry is marked as a likely binary-label error.'
                )
            else:
                mismatch_type = "chexpert_right_cl_wrong"
                recommended_action = "keep_chexpert"
                analysis_note = (
                    f'Mock review: the report is treated as supporting the current binary label for {target_label}. '
                    f'Quoted report text: "{impression_quote or findings_quote or fallback_quote}". '
                    f'The mock rule combines current binary label {current_binary} with model probability {model_prob:.4f}, '
                    "so this entry is marked as a likely false alarm from the suspicious-entry pipeline."
                )

        return validate_review_payload(
            {
                "mismatch_type": mismatch_type,
                "analysis_note": analysis_note,
                "recommended_action": recommended_action,
            }
        )


class RealReviewClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        request_timeout_seconds: float,
        temperature: float,
        max_completion_tokens: int,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.api_key = api_key
        self.model = model
        self.request_timeout_seconds = request_timeout_seconds
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

    def review(self, task: ReviewTask, attempt_number: int) -> dict[str, str]:
        del attempt_number
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(task.llm_input)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
        }

        response = requests.post(
            urljoin(self.base_url, "chat/completions"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.request_timeout_seconds,
        )

        if response.status_code >= 400:
            error_text = response.text.strip()
            error_text = error_text[:500] if error_text else f"HTTP {response.status_code}"
            raise RuntimeError(f"LLM request failed with status {response.status_code}: {error_text}")

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response shape: {json.dumps(data)[:500]}") from exc

        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected content type from LLM: {type(content).__name__}")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM returned non-JSON content: {content[:500]}") from exc

        return validate_review_payload(parsed)


def filter_and_rank_entries(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    required = {
        "pool_row_id",
        "label_index",
        "raw_label",
        "binary_label",
        "valid_label",
        "est_issue_entry",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing binary-review columns: {missing}")

    checked = df.copy()
    for column in ["raw_label", "binary_label", "valid_label", "est_issue_entry"]:
        checked[column] = pd.to_numeric(checked[column], errors="coerce")
    suspicious = checked[checked["est_issue_entry"].eq(1)].copy()
    if suspicious.empty:
        raise ValueError("No entry-level suspicious rows were found.")
    if not suspicious["valid_label"].eq(1).all():
        raise ValueError("Suspicious entry rows contain invalid targets (valid_label != 1).")
    if not suspicious["raw_label"].isin([-1.0, 0.0, 1.0]).all():
        raise ValueError("Suspicious entry rows contain unsupported raw labels.")
    if not suspicious["binary_label"].isin([0.0, 1.0]).all():
        raise ValueError("Suspicious entry rows contain non-binary decision targets.")
    expected_binary = suspicious["raw_label"].map({-1.0: 1.0, 0.0: 0.0, 1.0: 1.0})
    if not suspicious["binary_label"].eq(expected_binary).all():
        raise ValueError("Suspicious entry rows violate the fixed U-Ones binary projection.")

    filtered = suspicious
    if filtered.empty:
        raise ValueError("No valid binary suspicious rows remain after filtering.")

    sort_cols: list[str] = []
    ascending: list[bool] = []
    if "entry_issue_rank_self_confidence" in filtered.columns:
        sort_cols.append("entry_issue_rank_self_confidence")
        ascending.append(True)
    if "entry_quality_self_confidence" in filtered.columns:
        sort_cols.append("entry_quality_self_confidence")
        ascending.append(True)
    if "pred_prob" in filtered.columns:
        sort_cols.append("pred_prob")
        ascending.append(False)
    if not sort_cols:
        raise ValueError("No ranking columns found in the input CSV.")

    ranked = filtered.sort_values(sort_cols, ascending=ascending, kind="stable").reset_index(drop=True)
    ranked = ranked.head(top_k).copy()
    ranked_keys = ranked.apply(entry_key, axis=1)
    if ranked_keys.duplicated().any():
        raise ValueError("Selected review rows contain duplicate entry keys.")
    return ranked


def prepare_tasks(df: pd.DataFrame, report_archive: Path) -> list[ReviewTask]:
    study_ids = sorted(df["study_id"].astype(int).unique().tolist())
    report_map = extract_report_map(report_archive, study_ids)

    tasks: list[ReviewTask] = []
    for idx, (_, row) in enumerate(df.iterrows()):
        source_row = row.to_dict()
        report_text = normalize_report_text(report_map.get(int(row["study_id"]), ""))
        source_row["original_report_full"] = report_text
        source_row["entry_key"] = entry_key(source_row)
        llm_input = {
            "target_label": source_row["label_name"],
            "current_binary_label": source_row["binary_label"],
            "chexpert_raw_label": source_row["raw_label"],
            "model_oof_prob": source_row["pred_prob"],
            "original_report_full": source_row["original_report_full"],
        }
        tasks.append(ReviewTask(selected_index=idx, source_row=source_row, llm_input=llm_input))
    return tasks


def write_result_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_errors_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_existing_results(path: Path) -> set[str]:
    df = load_csv_if_exists(path)
    if df.empty or "entry_key" not in df.columns:
        return set()
    return set(df["entry_key"].astype(str).tolist())


def load_existing_errors(path: Path) -> dict[str, dict[str, Any]]:
    df = load_csv_if_exists(path)
    if df.empty or "entry_key" not in df.columns:
        return {}
    return {str(row["entry_key"]): row.to_dict() for _, row in df.iterrows()}


def process_task(
    task: ReviewTask,
    client: MockReviewClient | RealReviewClient,
    retry_limit: int,
) -> tuple[str, dict[str, Any]]:
    last_error = ""
    for attempt_number in range(1, retry_limit + 2):
        try:
            response = client.review(task, attempt_number=attempt_number)
            return "success", {
                "task": task,
                "response": response,
                "attempt_count": attempt_number,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": build_user_prompt(task.llm_input),
            }
        except Exception as exc:  # pragma: no cover - exercised in CLI test runs
            last_error = str(exc)
            if attempt_number > retry_limit:
                break
            time.sleep(0.1 * attempt_number)

    return "error", {
        "task": task,
        "error_message": last_error,
        "attempt_count": retry_limit + 1,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": build_user_prompt(task.llm_input),
    }


def chunked(items: list[ReviewTask], size: int) -> list[list[ReviewTask]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = args.output_dir / "results.csv"
    errors_csv = args.output_dir / "errors.csv"

    df = pd.read_csv(args.input_csv)
    selected = filter_and_rank_entries(df, top_k=args.top_k)
    tasks = prepare_tasks(selected, args.report_archive)

    existing_success = load_existing_results(results_csv) if args.resume else set()
    error_rows_by_key = load_existing_errors(errors_csv) if args.resume else {}
    pending_tasks = [task for task in tasks if task.key not in existing_success]

    print(f"Loaded input rows: {len(df)}")
    print(
        "Filtered suspicious entry rows (valid post-binarization targets, including raw -1): "
        f"{len(selected)} selected for this run"
    )
    print(f"Already completed rows skipped via resume: {len(existing_success.intersection({t.key for t in tasks}))}")
    print(f"Pending rows: {len(pending_tasks)}")

    if not pending_tasks:
        print("No pending rows left to process.")
        return

    if args.llm_mode == "mock":
        client: MockReviewClient | RealReviewClient = MockReviewClient(
            fail_once_indexes=parse_index_list(args.mock_fail_once_indexes),
            always_fail_indexes=parse_index_list(args.mock_always_fail_indexes),
            latency_ms=args.mock_latency_ms,
        )
    else:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        model = os.getenv("OPENAI_MODEL", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        if not base_url:
            raise RuntimeError("OPENAI_BASE_URL is not set in the environment.")
        if not model:
            raise RuntimeError("OPENAI_MODEL is not set in the environment.")

        client = RealReviewClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            request_timeout_seconds=args.request_timeout_seconds,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
        )

    result_fieldnames = list(tasks[0].source_row.keys()) + [
        "mismatch_type",
        "analysis_note",
        "recommended_action",
        "attempt_count",
    ]
    error_fieldnames = list(tasks[0].source_row.keys()) + [
        "error_message",
        "attempt_count",
    ]

    success_count = 0
    failure_count = 0
    batches = chunked(pending_tasks, max(1, args.batch_size))
    for batch_index, batch in enumerate(batches, start=1):
        print(f"Processing batch {batch_index}/{len(batches)} with {len(batch)} rows...")
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_map = {
                executor.submit(process_task, task, client, args.retry_limit): task for task in batch
            }
            for future in as_completed(future_map):
                status, payload = future.result()
                task = payload["task"]
                if status == "success":
                    result_row = dict(task.source_row)
                    result_row.update(payload["response"])
                    result_row["attempt_count"] = payload["attempt_count"]
                    write_result_row(results_csv, result_row, result_fieldnames)
                    existing_success.add(task.key)
                    error_rows_by_key.pop(task.key, None)
                    success_count += 1
                else:
                    error_row = dict(task.source_row)
                    error_row["error_message"] = payload["error_message"]
                    error_row["attempt_count"] = payload["attempt_count"]
                    error_rows_by_key[task.key] = error_row
                    failure_count += 1

        write_errors_csv(errors_csv, list(error_rows_by_key.values()), error_fieldnames)

    print(json.dumps(
        {
            "results_csv": str(results_csv),
            "errors_csv": str(errors_csv),
            "success_count_this_run": success_count,
            "failure_count_this_run": failure_count,
            "selected_total": len(tasks),
            "completed_total_after_run": len(existing_success.intersection({t.key for t in tasks})),
            "remaining_error_total": len(error_rows_by_key),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
