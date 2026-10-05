#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_ACTION_BY_MISMATCH = {
    "chexpert_wrong_cl_right": "review_or_relabel",
    "chexpert_right_cl_wrong": "keep_chexpert",
    "report_ambiguous": "mark_ambiguous",
}


def build_refinement_tables(
    review_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = review_rows.copy()
    required = {
        "pool_row_id",
        "label_index",
        "label_name",
        "raw_label",
        "binary_label",
        "valid_label",
        "mismatch_type",
        "recommended_action",
        "sample_selected_rank",
        "study_id",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"LLM results are missing required columns: {missing}")
    unknown_mismatch = sorted(
        set(df["mismatch_type"].dropna()) - set(EXPECTED_ACTION_BY_MISMATCH)
    )
    if unknown_mismatch:
        raise ValueError(f"Unknown mismatch_type values: {unknown_mismatch}")
    if df.duplicated(subset=["pool_row_id", "label_index"]).any():
        raise ValueError("LLM results contain duplicate entry keys")

    for column in ["raw_label", "binary_label", "valid_label"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if not df["valid_label"].eq(1).all():
        raise ValueError("Refinement actions require valid_label == 1")
    if not df["raw_label"].isin([-1.0, 0.0, 1.0]).all():
        raise ValueError("Refinement actions contain unsupported raw_label values")
    if not df["binary_label"].isin([0.0, 1.0]).all():
        raise ValueError("Refinement actions require binary_label values in {0,1}")
    expected_binary = df["raw_label"].map({-1.0: 1.0, 0.0: 0.0, 1.0: 1.0})
    if not df["binary_label"].eq(expected_binary).all():
        raise ValueError("Refinement actions violate the fixed U-Ones binary projection")

    expected_action = df["mismatch_type"].map(EXPECTED_ACTION_BY_MISMATCH)
    df["response_consistent"] = df["recommended_action"].eq(expected_action)
    df["action_resolution"] = "keep_initial_label"
    df.loc[
        df["response_consistent"]
        & df["mismatch_type"].eq("chexpert_wrong_cl_right"),
        "action_resolution",
    ] = "relabel_flip"
    df.loc[
        df["response_consistent"] & df["mismatch_type"].eq("report_ambiguous"),
        "action_resolution",
    ] = "mask_report_ambiguous"
    df.loc[~df["response_consistent"], "action_resolution"] = (
        "mask_response_field_conflict"
    )

    relabel_df = df[df["action_resolution"].eq("relabel_flip")].copy()
    relabel_df["new_binary_label"] = 1.0 - relabel_df["binary_label"]
    relabel_df["new_raw_label"] = relabel_df["new_binary_label"]
    relabel_df["action_label_basis"] = "binary_label"
    relabel_df = relabel_df[
        [
            "pool_row_id",
            "label_index",
            "label_name",
            "raw_label",
            "binary_label",
            "new_binary_label",
            "new_raw_label",
            "valid_label",
            "mismatch_type",
            "recommended_action",
            "response_consistent",
            "action_resolution",
            "action_label_basis",
            "sample_selected_rank",
            "study_id",
        ]
    ].rename(
        columns={
            "raw_label": "old_raw_label",
            "binary_label": "old_binary_label",
        }
    )

    mask_df = df[df["action_resolution"].str.startswith("mask_")].copy()
    mask_df["est_issue_entry"] = 1
    mask_df["action_label_basis"] = "binary_label"
    mask_df = mask_df[
        [
            "pool_row_id",
            "label_index",
            "label_name",
            "raw_label",
            "binary_label",
            "valid_label",
            "mismatch_type",
            "recommended_action",
            "response_consistent",
            "action_resolution",
            "action_label_basis",
            "sample_selected_rank",
            "study_id",
            "est_issue_entry",
        ]
    ]

    keep_rows = int(df["action_resolution"].eq("keep_initial_label").sum())
    conflict_rows = int((~df["response_consistent"]).sum())
    if len(relabel_df) + len(mask_df) + keep_rows != len(df):
        raise AssertionError("Resolved refinement actions do not partition all review rows")

    summary_df = pd.DataFrame(
        [
            {
                "llm_review_rows": int(len(df)),
                "relabel_rows": int(len(relabel_df)),
                "mask_rows": int(len(mask_df)),
                "keep_rows": keep_rows,
                "response_consistent_rows": int(df["response_consistent"].sum()),
                "response_conflict_rows": conflict_rows,
                "conflict_resolution": "mask_entry",
                "action_label_basis": "post_binarization_binary_label",
                "uncertain_projection": "raw_-1_to_binary_1",
                "raw_uncertain_review_rows": int(df["raw_label"].eq(-1.0).sum()),
                "raw_explicit_review_rows": int(df["raw_label"].isin([0.0, 1.0]).sum()),
                "raw_uncertain_relabel_rows": int(
                    relabel_df["old_raw_label"].eq(-1.0).sum()
                ),
                "raw_uncertain_mask_rows": int(mask_df["raw_label"].eq(-1.0).sum()),
            }
        ]
    )

    return relabel_df, mask_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build relabel/mask tables from entry-level LLM review results."
    )
    parser.add_argument("--llm-results-csv", type=Path, required=True)
    parser.add_argument("--relabel-csv", type=Path, required=True)
    parser.add_argument("--mask-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.llm_results_csv)
    relabel_df, mask_df, summary_df = build_refinement_tables(df)
    summary = summary_df.iloc[0]
    keep_rows = int(summary["keep_rows"])
    conflict_rows = int(summary["response_conflict_rows"])

    for path in [args.relabel_csv, args.mask_csv, args.summary_csv]:
        path.parent.mkdir(parents=True, exist_ok=True)

    relabel_df.to_csv(args.relabel_csv, index=False)
    mask_df.to_csv(args.mask_csv, index=False)
    summary_df.to_csv(args.summary_csv, index=False)

    print(f"LLM review rows: {len(df)}")
    print(f"Relabel rows: {len(relabel_df)}")
    print(f"Mask rows: {len(mask_df)}")
    print(f"Keep rows: {keep_rows}")
    print(f"Reviewed raw -1 rows under binary-positive target: {int(summary['raw_uncertain_review_rows'])}")
    print(f"Response-field conflicts conservatively masked: {conflict_rows}")
    print(f"Wrote relabel CSV: {args.relabel_csv}")
    print(f"Wrote mask CSV: {args.mask_csv}")
    print(f"Wrote summary CSV: {args.summary_csv}")


if __name__ == "__main__":
    main()
