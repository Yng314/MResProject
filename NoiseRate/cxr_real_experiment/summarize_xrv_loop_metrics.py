#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def study_weighted_auroc(study_csv: Path) -> float:
    df = pd.read_csv(study_csv)
    values = df["study_auroc_binary"].to_numpy(dtype=float)
    weights = df["study_valid_count"].to_numpy(dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Append one iterative-loop metric row.")
    parser.add_argument("--branch", required=True)
    parser.add_argument("--loop-id", type=int, required=True)
    parser.add_argument("--top-fraction", type=float, required=True)
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--selected-sample-csv", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--cumulative-sample-csv", type=Path, default=None)
    parser.add_argument("--cumulative-relabel-csv", type=Path, default=None)
    parser.add_argument("--cumulative-mask-csv", type=Path, default=None)
    parser.add_argument("--llm-results-csv", type=Path, default=None)
    parser.add_argument("--llm-errors-csv", type=Path, default=None)
    args = parser.parse_args()

    train_summary = pd.read_csv(args.train_dir / "baseline_run_summary.csv").iloc[0].to_dict()
    oof_summary = pd.read_csv(args.oof_dir / "oof_cleanlab_smoke_summary.csv").iloc[0].to_dict()
    selected_df = read_csv_if_exists(args.selected_sample_csv)
    cumulative_sample_df = read_csv_if_exists(args.cumulative_sample_csv) if args.cumulative_sample_csv else pd.DataFrame()
    cumulative_relabel_df = read_csv_if_exists(args.cumulative_relabel_csv) if args.cumulative_relabel_csv else pd.DataFrame()
    cumulative_mask_df = read_csv_if_exists(args.cumulative_mask_csv) if args.cumulative_mask_csv else pd.DataFrame()
    llm_results_df = read_csv_if_exists(args.llm_results_csv) if args.llm_results_csv else pd.DataFrame()
    llm_errors_df = read_csv_if_exists(args.llm_errors_csv) if args.llm_errors_csv else pd.DataFrame()

    row = {
        "branch": args.branch,
        "loop_id": int(args.loop_id),
        "top_fraction": float(args.top_fraction),
        "oof_train_samples": int(oof_summary.get("n_samples", 0)),
        "oof_sample_issue_rate": float(oof_summary.get("estimated_noise_rate_sample", np.nan)),
        "oof_entry_issue_rate": float(oof_summary.get("estimated_noise_rate_entry", np.nan)),
        "selected_samples_this_loop": int(len(selected_df)),
        "cumulative_removed_samples": int(cumulative_sample_df["pool_row_id"].nunique()) if "pool_row_id" in cumulative_sample_df else 0,
        "llm_review_rows_this_loop": int(len(llm_results_df)),
        "llm_error_rows_this_loop": int(len(llm_errors_df)),
        "cumulative_relabel_entries": int(
            cumulative_relabel_df[["pool_row_id", "label_index"]].drop_duplicates().shape[0]
        )
        if {"pool_row_id", "label_index"}.issubset(cumulative_relabel_df.columns)
        else 0,
        "cumulative_mask_entries": int(
            cumulative_mask_df[["pool_row_id", "label_index"]].drop_duplicates().shape[0]
        )
        if {"pool_row_id", "label_index"}.issubset(cumulative_mask_df.columns)
        else 0,
        "train_samples": int(train_summary.get("train_samples", 0)),
        "test_image_macro_auroc": float(train_summary.get("test_image_macro_auroc_binary", np.nan)),
        "test_study_macro_auroc": float(train_summary.get("test_study_macro_auroc_binary", np.nan)),
        "test_study_weighted_auroc": study_weighted_auroc(args.train_dir / "test_study_auroc_summary.csv"),
        "best_epoch": int(train_summary.get("best_epoch", -1)),
        "best_val_loss": float(train_summary.get("best_val_loss", np.nan)),
        "seed": int(train_summary.get("seed", -1)),
    }

    out_df = pd.DataFrame([row])
    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.metrics_csv.exists() and args.metrics_csv.stat().st_size > 0:
        prev_df = pd.read_csv(args.metrics_csv)
        if {"branch", "loop_id"}.issubset(prev_df.columns):
            keep = ~(
                prev_df["branch"].astype(str).eq(str(args.branch))
                & prev_df["loop_id"].astype(int).eq(int(args.loop_id))
            )
            prev_df = prev_df.loc[keep]
        out_df = pd.concat([prev_df, out_df], ignore_index=True)
    out_df = out_df.sort_values(["branch", "loop_id"], kind="stable").reset_index(drop=True)
    temp_path = args.metrics_csv.with_suffix(args.metrics_csv.suffix + ".tmp")
    out_df.to_csv(temp_path, index=False)
    temp_path.replace(args.metrics_csv)
    print(out_df.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
