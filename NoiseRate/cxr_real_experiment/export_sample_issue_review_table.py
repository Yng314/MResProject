#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


def parse_pipe_floats(value: str) -> list[float]:
    out: list[float] = []
    for token in str(value).split("|"):
        token = token.strip()
        if token == "nan" or token == "":
            out.append(math.nan)
        else:
            out.append(float(token))
    return out


def parse_pipe_ints(value: str) -> list[int]:
    return [int(token.strip()) for token in str(value).split("|")]


def cleanlab_support_score(binary_label: float, pred_prob: float) -> float:
    if math.isnan(binary_label) or math.isnan(pred_prob):
        return math.nan
    return pred_prob if binary_label == 1.0 else 1.0 - pred_prob


def format_label_summary(
    label_name: str,
    raw_label: float,
    binary_label: float,
    pred_prob: float,
    issue_flag: int,
) -> str:
    raw_str = "nan" if math.isnan(raw_label) else f"{raw_label:.0f}"
    bin_str = "nan" if math.isnan(binary_label) else f"{binary_label:.0f}"
    gap_str = "nan" if math.isnan(binary_label) or math.isnan(pred_prob) else f"{pred_prob - binary_label:+.2f}"
    marker = "*" if issue_flag == 1 else ""
    return f"{marker}{label_name}: raw={raw_str}, bin={bin_str}, oof={pred_prob:.3f}, gap={gap_str}"


def extract_report_map(report_archive: Path, study_ids: list[int]) -> dict[int, str]:
    reports: dict[int, str] = {}
    with zipfile.ZipFile(report_archive, "r") as zf:
        name_index = {
            name.rsplit("/", 1)[-1]: name
            for name in zf.namelist()
            if name.endswith(".txt")
        }
        for study_id in study_ids:
            key = f"s{study_id}.txt"
            path = name_index.get(key)
            if path is None:
                reports[study_id] = ""
                continue
            reports[study_id] = zf.read(path).decode("utf-8", errors="replace")
    return reports


def shorten_report(text: str, max_chars: int = 400) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 4] + " ..."


def build_review_table(sample_df: pd.DataFrame, report_map: dict[int, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in sample_df.iterrows():
        raw_vals = parse_pipe_floats(row["raw_labels_4class"])
        bin_vals = parse_pipe_floats(row["binary_labels_for_detection"])
        valid_mask = parse_pipe_ints(row["valid_label_mask"])
        pred_probs = parse_pipe_floats(row["pred_probs"])
        issue_mask = parse_pipe_ints(row["issue_entry_mask"])

        flagged_labels = [
            LABEL_NAMES[idx]
            for idx, flag in enumerate(issue_mask)
            if idx < len(LABEL_NAMES) and flag == 1
        ]

        mismatch_rows = []
        flagged_summaries = []
        for idx, label_name in enumerate(LABEL_NAMES):
            raw_label = raw_vals[idx]
            binary_label = bin_vals[idx]
            pred_prob = pred_probs[idx]
            issue_flag = issue_mask[idx]
            if valid_mask[idx] == 1 and not math.isnan(binary_label):
                mismatch_rows.append(
                    (
                        abs(pred_prob - binary_label),
                        issue_flag,
                        format_label_summary(label_name, raw_label, binary_label, pred_prob, issue_flag),
                    )
                )
            if issue_flag == 1:
                flagged_summaries.append(
                    format_label_summary(label_name, raw_label, binary_label, pred_prob, issue_flag)
                )

        mismatch_rows.sort(key=lambda x: x[0], reverse=True)
        filtered_mismatches = [
            summary for gap, issue_flag, summary in mismatch_rows if issue_flag == 1 or gap >= 0.25
        ]
        if not filtered_mismatches:
            filtered_mismatches = [summary for _, _, summary in mismatch_rows[:5]]
        top_mismatch_labels = " | ".join(filtered_mismatches[:5])
        flagged_label_details = " | ".join(flagged_summaries)

        out: dict[str, object] = {
            "sample_local_index": row["sample_local_index"],
            "pool_row_id": row["pool_row_id"],
            "subject_id": row["subject_id"],
            "study_id": row["study_id"],
            "dicom_id": row["dicom_id"],
            "image_path": row["image_path"],
            "est_issue_sample": row["est_issue_sample"],
            "est_issue_entry_count": row["est_issue_entry_count"],
            "issue_rank_self_confidence": row["issue_rank_self_confidence"],
            "issue_rank_normalized_margin": row["issue_rank_normalized_margin"],
            "issue_rank_confidence_weighted_entropy": row["issue_rank_confidence_weighted_entropy"],
            "sample_quality_self_confidence": row["sample_quality_self_confidence"],
            "sample_quality_normalized_margin": row["sample_quality_normalized_margin"],
            "sample_quality_confidence_weighted_entropy": row["sample_quality_confidence_weighted_entropy"],
            "flagged_label_names": "|".join(flagged_labels),
            "flagged_label_details": flagged_label_details,
            "top_mismatch_labels": top_mismatch_labels,
            "report_text": report_map.get(int(row["study_id"]), ""),
            "report_snippet": shorten_report(report_map.get(int(row["study_id"]), "")),
        }

        for idx, label_name in enumerate(LABEL_NAMES):
            safe = label_name.lower().replace(" ", "_")
            raw_label = raw_vals[idx]
            binary_label = bin_vals[idx]
            pred_prob = pred_probs[idx]
            issue_flag = issue_mask[idx]
            out[f"raw__{safe}"] = raw_label
            out[f"chexpert_bin__{safe}"] = binary_label
            out[f"valid__{safe}"] = valid_mask[idx]
            out[f"oof_prob__{safe}"] = pred_prob
            out[f"issue_flag__{safe}"] = issue_flag
            out[f"prob_minus_label__{safe}"] = (
                math.nan if math.isnan(binary_label) else pred_prob - binary_label
            )
            out[f"support_score__{safe}"] = cleanlab_support_score(binary_label, pred_prob)
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-issues-csv", type=Path, required=True)
    parser.add_argument("--report-archive", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--full-output-csv", type=Path, default=None)
    args = parser.parse_args()

    sample_df = pd.read_csv(args.sample_issues_csv)
    sample_df = sample_df.sort_values("issue_rank_self_confidence", na_position="last").reset_index(drop=True)

    top_df = sample_df.head(args.top_n).copy()
    top_report_map = extract_report_map(
        args.report_archive,
        sorted(top_df["study_id"].astype(int).unique().tolist()),
    )
    top_review_df = build_review_table(top_df, top_report_map)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    top_review_df.to_csv(args.output_csv, index=False)
    print(f"Saved top-{args.top_n} review CSV: {args.output_csv}")

    if args.full_output_csv is not None:
        full_report_map = extract_report_map(
            args.report_archive,
            sorted(sample_df["study_id"].astype(int).unique().tolist()),
        )
        full_review_df = build_review_table(sample_df, full_report_map)
        args.full_output_csv.parent.mkdir(parents=True, exist_ok=True)
        full_review_df.to_csv(args.full_output_csv, index=False)
        print(f"Saved full review CSV: {args.full_output_csv}")


if __name__ == "__main__":
    main()
