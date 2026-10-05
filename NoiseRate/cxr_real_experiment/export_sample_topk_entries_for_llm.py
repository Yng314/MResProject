#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

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
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
]


def parse_pipe_floats(value: str) -> list[float]:
    out: list[float] = []
    for token in str(value).split("|"):
        token = token.strip()
        if token in {"", "nan"}:
            out.append(math.nan)
        else:
            out.append(float(token))
    return out


def parse_pipe_ints(value: str) -> list[int]:
    return [int(token.strip()) for token in str(value).split("|")]


def support_score(binary_label: float, pred_prob: float) -> float:
    if math.isnan(binary_label) or math.isnan(pred_prob):
        return math.nan
    return pred_prob if binary_label == 1.0 else 1.0 - pred_prob


def expected_binary_label(raw_label: float) -> float:
    if raw_label == 0.0:
        return 0.0
    if raw_label in {-1.0, 1.0}:
        return 1.0
    raise ValueError(f"Unsupported valid raw label: {raw_label!r}")


def flagged_label_names(issue_mask: list[int]) -> str:
    return "|".join(
        label_name
        for label_name, flag in zip(LABEL_NAMES, issue_mask)
        if flag == 1
    )


def rank_samples(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    if "est_issue_sample" in ranked.columns:
        ranked = ranked[ranked["est_issue_sample"] == 1].copy()

    sort_cols: list[str] = []
    ascending: list[bool] = []
    if "sample_quality_self_confidence" in ranked.columns:
        sort_cols.append("sample_quality_self_confidence")
        ascending.append(True)
    if "issue_rank_self_confidence" in ranked.columns:
        sort_cols.append("issue_rank_self_confidence")
        ascending.append(True)
    if "est_issue_entry_count" in ranked.columns:
        sort_cols.append("est_issue_entry_count")
        ascending.append(False)
    if not sort_cols:
        raise ValueError("No ranking columns found in sample issue CSV.")

    ranked = ranked.sort_values(sort_cols, ascending=ascending, kind="stable").reset_index(drop=True)
    ranked["sample_selected_rank"] = ranked.index + 1
    return ranked


def expand_sample_rows(sample_df: pd.DataFrame) -> pd.DataFrame:
    if sample_df.empty:
        raise ValueError("The selected sample-level pool is empty.")
    if sample_df["pool_row_id"].duplicated().any():
        raise ValueError("Selected sample-level rows contain duplicate pool_row_id values.")

    rows: list[dict[str, object]] = []
    for _, row in sample_df.iterrows():
        raw_vals = parse_pipe_floats(row["raw_labels_4class"])
        bin_vals = parse_pipe_floats(row["binary_labels_for_detection"])
        valid_mask = parse_pipe_ints(row["valid_label_mask"])
        pred_probs = parse_pipe_floats(row["pred_probs"])
        issue_mask = parse_pipe_ints(row["issue_entry_mask"])
        vector_lengths = {
            "raw_labels_4class": len(raw_vals),
            "binary_labels_for_detection": len(bin_vals),
            "valid_label_mask": len(valid_mask),
            "pred_probs": len(pred_probs),
            "issue_entry_mask": len(issue_mask),
        }
        if any(length != len(LABEL_NAMES) for length in vector_lengths.values()):
            raise ValueError(
                f"Sample {row['pool_row_id']} has non-12-label vectors: {vector_lengths}"
            )
        sample_flagged_names = flagged_label_names(issue_mask)

        local_rows: list[dict[str, object]] = []
        for label_index, label_name in enumerate(LABEL_NAMES):
            if label_index >= len(issue_mask) or issue_mask[label_index] != 1:
                continue
            raw_label = raw_vals[label_index]
            binary_label = bin_vals[label_index]
            valid_label = valid_mask[label_index]
            pred_prob = pred_probs[label_index]
            if valid_label != 1:
                raise ValueError(
                    f"Flagged entry {row['pool_row_id']}::{label_index} is not a valid label."
                )
            if math.isnan(raw_label) or raw_label not in {-1.0, 0.0, 1.0}:
                raise ValueError(
                    f"Flagged entry {row['pool_row_id']}::{label_index} has invalid raw label "
                    f"{raw_label!r}."
                )
            if math.isnan(binary_label) or binary_label not in {0.0, 1.0}:
                raise ValueError(
                    f"Flagged entry {row['pool_row_id']}::{label_index} has invalid binary label "
                    f"{binary_label!r}."
                )
            expected_binary = expected_binary_label(raw_label)
            if binary_label != expected_binary:
                raise ValueError(
                    f"Flagged entry {row['pool_row_id']}::{label_index} violates the U-Ones "
                    f"projection: raw={raw_label}, binary={binary_label}, expected={expected_binary}."
                )

            sample_support = support_score(binary_label, pred_prob)
            local_rows.append(
                {
                    "sample_selected_rank": int(row["sample_selected_rank"]),
                    "sample_est_issue_entry_count": int(row["est_issue_entry_count"]),
                    "sample_issue_rank_self_confidence": float(row["issue_rank_self_confidence"]),
                    "sample_issue_rank_normalized_margin": float(row["issue_rank_normalized_margin"]),
                    "sample_issue_rank_confidence_weighted_entropy": float(
                        row["issue_rank_confidence_weighted_entropy"]
                    ),
                    "sample_quality_self_confidence": float(row["sample_quality_self_confidence"]),
                    "sample_quality_normalized_margin": float(row["sample_quality_normalized_margin"]),
                    "sample_quality_confidence_weighted_entropy": float(
                        row["sample_quality_confidence_weighted_entropy"]
                    ),
                    "sample_flagged_label_names": sample_flagged_names,
                    "sample_local_index": int(row["sample_local_index"]),
                    "pool_row_id": int(row["pool_row_id"]),
                    "subject_id": int(row["subject_id"]),
                    "study_id": int(row["study_id"]),
                    "dicom_id": str(row["dicom_id"]),
                    "image_path": str(row["image_path"]),
                    "label_index": int(label_index),
                    "label_name": label_name,
                    "raw_label": float(raw_label),
                    "binary_label": float(binary_label),
                    "valid_label": int(valid_label),
                    "pred_prob": float(pred_prob),
                    "prob_minus_label": float(pred_prob - binary_label),
                    "entry_quality_self_confidence": float(sample_support),
                    "est_issue_entry": 1,
                }
            )

        local_rows.sort(
            key=lambda item: (
                item["entry_quality_self_confidence"],
                -abs(item["prob_minus_label"]),
                item["label_index"],
            )
        )
        for entry_rank, item in enumerate(local_rows, start=1):
            item["entry_rank_within_sample"] = entry_rank
            rows.append(item)

    expanded = pd.DataFrame(rows)
    if expanded.empty:
        raise ValueError("No valid binary suspicious entries found inside the selected sample-level pool.")

    selected_ids = set(sample_df["pool_row_id"].astype(int))
    represented_ids = set(expanded["pool_row_id"].astype(int))
    missing_ids = sorted(selected_ids - represented_ids)
    if missing_ids:
        preview = missing_ids[:10]
        raise ValueError(
            "Binary-label expansion did not cover every selected sample: "
            f"missing={len(missing_ids)}/{len(selected_ids)}, examples={preview}"
        )

    expanded = expanded.sort_values(
        [
            "sample_selected_rank",
            "entry_quality_self_confidence",
            "entry_rank_within_sample",
            "label_index",
        ],
        ascending=[True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    expanded["entry_issue_rank_self_confidence"] = expanded.index + 1
    expanded["entry_key"] = (
        expanded["pool_row_id"].astype(str) + "::" + expanded["label_index"].astype(str)
    )
    if expanded["entry_key"].duplicated().any():
        raise ValueError("Expanded entry rows contain duplicate entry_key values.")
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand sample-level suspicious rows into entry-level LLM review rows."
    )
    parser.add_argument("--sample-issues-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--top-n-samples", type=int, default=100)
    args = parser.parse_args()

    sample_df = pd.read_csv(args.sample_issues_csv)
    ranked = rank_samples(sample_df)
    selected = ranked.head(args.top_n_samples).copy()
    expanded = expand_sample_rows(selected)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_csv(args.output_csv, index=False)

    print(f"Loaded suspicious sample rows: {len(ranked)}")
    print(f"Selected top samples: {len(selected)}")
    print(f"Expanded suspicious entry rows: {len(expanded)}")
    print(
        "Selected-sample binary review coverage: "
        f"{expanded['pool_row_id'].nunique()}/{len(selected)} (100.00%)"
    )
    print(f"Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()
