#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TEST_LABEL_MAP = {
    "Lung Opacity": "Airspace Opacity",
}


ACTION_MAP = {
    "chexpert_right_cl_wrong": "keep_initial_label",
    "chexpert_wrong_cl_right": "relabel_flip",
    "report_ambiguous": "mask_entry",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def safe_rate(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace({0: np.nan})
    return numer / denom


def add_refinement_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["action"] = out["mismatch_type"].map(ACTION_MAP).fillna("unknown")
    out["old_label"] = out["raw_label"].astype(float)
    out["new_label"] = np.nan
    relabel_mask = out["action"] == "relabel_flip"
    out.loc[relabel_mask, "new_label"] = 1.0 - out.loc[relabel_mask, "old_label"]
    out["label_change"] = out["old_label"].map(lambda x: f"{int(x)}->same" if pd.notna(x) else "NA")
    out.loc[relabel_mask, "label_change"] = (
        out.loc[relabel_mask, "old_label"].astype(int).astype(str)
        + "->"
        + out.loc[relabel_mask, "new_label"].astype(int).astype(str)
    )
    out.loc[out["action"] == "mask_entry", "label_change"] = (
        out.loc[out["action"] == "mask_entry", "old_label"].astype(int).astype(str) + "->mask"
    )
    out["model_side_label"] = (out["pred_prob"].astype(float) >= 0.5).astype(int)
    return out


def summarize_llm(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    total = len(df)
    by_action = (
        df.groupby(["mismatch_type", "action"], dropna=False)
        .size()
        .reset_index(name="count")
        .assign(fraction=lambda x: x["count"] / max(1, total))
        .sort_values("count", ascending=False)
    )

    by_label_action = (
        df.groupby(["label_name", "mismatch_type", "action"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    by_label_total = df.groupby("label_name").size().rename("label_review_rows").reset_index()
    by_label_action = by_label_action.merge(by_label_total, on="label_name", how="left")
    by_label_action["within_label_fraction"] = by_label_action["count"] / by_label_action["label_review_rows"]
    by_label_action = by_label_action.sort_values(["label_name", "count"], ascending=[True, False])

    by_relabel_direction = (
        df[df["action"] == "relabel_flip"]
        .groupby(["label_name", "label_change"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["label_name", "label_change"])
    )

    by_label_raw_action = (
        df.groupby(["label_name", "old_label", "action"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["label_name", "old_label", "count"], ascending=[True, True, False])
    )

    prob_summary = (
        df.groupby(["label_name", "action"], dropna=False)
        .agg(
            count=("entry_key", "count"),
            pred_prob_mean=("pred_prob", "mean"),
            pred_prob_median=("pred_prob", "median"),
            pred_prob_min=("pred_prob", "min"),
            pred_prob_max=("pred_prob", "max"),
            sample_rank_median=("sample_selected_rank", "median"),
            entry_quality_mean=("entry_quality_self_confidence", "mean"),
        )
        .reset_index()
        .sort_values(["label_name", "action"])
    )

    return {
        "llm_category_summary": by_action,
        "llm_category_by_label": by_label_action,
        "relabel_direction_by_label": by_relabel_direction,
        "raw_label_by_action": by_label_raw_action,
        "probability_by_label_action": prob_summary,
    }


def summarize_test_distribution(test_df: pd.DataFrame, metric_df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows = []
    for label_name in labels:
        test_col = TEST_LABEL_MAP.get(label_name, label_name)
        if test_col not in test_df.columns:
            continue
        s = test_df[test_col]
        metric_row = metric_df[metric_df["label_name"] == label_name]
        rows.append(
            {
                "label_name": label_name,
                "test_column": test_col,
                "raw_test_non_missing": int(s.notna().sum()),
                "raw_test_positive_1": int((s == 1).sum()),
                "raw_test_negative_0": int((s == 0).sum()),
                "raw_test_uncertain_minus1": int((s == -1).sum()),
                "raw_test_positive_rate_among_non_missing": float((s == 1).sum() / max(1, s.notna().sum())),
                "metric_valid_count": int(metric_row["study_valid_count"].iloc[0]) if not metric_row.empty else np.nan,
                "metric_positive_count": int(metric_row["study_positive_count"].iloc[0]) if not metric_row.empty else np.nan,
                "metric_negative_count": int(metric_row["study_negative_count"].iloc[0]) if not metric_row.empty else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["metric_positive_rate"] = safe_rate(out["metric_positive_count"], out["metric_valid_count"])
    return out


def metric_comparison(
    baseline_df: pd.DataFrame,
    remove_df: pd.DataFrame,
    refine_df: pd.DataFrame,
    label_action_df: pd.DataFrame,
    test_dist_df: pd.DataFrame,
) -> pd.DataFrame:
    out = baseline_df[["label_name", "study_auroc_binary"]].rename(
        columns={"study_auroc_binary": "baseline_study_auroc"}
    )
    out = out.merge(
        remove_df[["label_name", "study_auroc_binary"]].rename(
            columns={"study_auroc_binary": "topk_remove_study_auroc"}
        ),
        on="label_name",
        how="outer",
    )
    out = out.merge(
        refine_df[["label_name", "study_auroc_binary"]].rename(
            columns={"study_auroc_binary": "llm_refine_study_auroc"}
        ),
        on="label_name",
        how="outer",
    )
    action_wide = (
        label_action_df.pivot_table(
            index="label_name",
            columns="action",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    out = out.merge(action_wide, on="label_name", how="left")
    out = out.merge(test_dist_df, on="label_name", how="left")
    for col in ["keep_initial_label", "relabel_flip", "mask_entry"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = out[col].fillna(0).astype(int)
    out["delta_refine_vs_remove"] = out["llm_refine_study_auroc"] - out["topk_remove_study_auroc"]
    out["delta_remove_vs_baseline"] = out["topk_remove_study_auroc"] - out["baseline_study_auroc"]
    out["delta_refine_vs_baseline"] = out["llm_refine_study_auroc"] - out["baseline_study_auroc"]
    return out.sort_values("delta_refine_vs_remove")


def choose_manual_review_cases(df: pd.DataFrame, metric_df: pd.DataFrame, per_group: int) -> pd.DataFrame:
    selected_parts: list[pd.DataFrame] = []
    worst_labels = metric_df.sort_values("delta_refine_vs_remove").head(4)["label_name"].dropna().tolist()
    high_change_labels = (
        df[df["action"].isin(["relabel_flip", "mask_entry"])]
        .groupby("label_name")
        .size()
        .sort_values(ascending=False)
        .head(4)
        .index.tolist()
    )
    labels = list(dict.fromkeys(worst_labels + high_change_labels))
    for label_name in labels:
        for action in ["relabel_flip", "mask_entry", "keep_initial_label"]:
            subset = df[(df["label_name"] == label_name) & (df["action"] == action)].copy()
            if subset.empty:
                continue
            subset = subset.sort_values(
                ["sample_selected_rank", "entry_quality_self_confidence", "pred_prob"],
                ascending=[True, True, False],
            ).head(per_group)
            subset["case_reason"] = f"{label_name}::{action}"
            selected_parts.append(subset)
    if not selected_parts:
        return pd.DataFrame()
    cols = [
        "case_reason",
        "sample_selected_rank",
        "pool_row_id",
        "study_id",
        "label_name",
        "old_label",
        "new_label",
        "label_change",
        "pred_prob",
        "mismatch_type",
        "action",
        "analysis_note",
        "original_report_full",
    ]
    return pd.concat(selected_parts, ignore_index=True)[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audit tables for LLM label-refinement results.")
    parser.add_argument("--llm-results-csv", type=Path, required=True)
    parser.add_argument("--test-labels-csv", type=Path, required=True)
    parser.add_argument("--baseline-study-auroc-csv", type=Path, required=True)
    parser.add_argument("--remove-study-auroc-csv", type=Path, required=True)
    parser.add_argument("--refine-study-auroc-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manual-review-cases-per-group", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    llm_df = add_refinement_columns(read_csv(args.llm_results_csv))
    test_df = read_csv(args.test_labels_csv)
    baseline_df = read_csv(args.baseline_study_auroc_csv)
    remove_df = read_csv(args.remove_study_auroc_csv)
    refine_df = read_csv(args.refine_study_auroc_csv)

    audit_cols = [
        "sample_selected_rank",
        "pool_row_id",
        "study_id",
        "dicom_id",
        "label_index",
        "label_name",
        "old_label",
        "new_label",
        "label_change",
        "pred_prob",
        "prob_minus_label",
        "entry_quality_self_confidence",
        "mismatch_type",
        "action",
        "recommended_action",
        "analysis_note",
    ]
    llm_df[audit_cols].to_csv(args.output_dir / "llm_refinement_entry_audit.csv", index=False)

    summaries = summarize_llm(llm_df)
    for name, summary_df in summaries.items():
        summary_df.to_csv(args.output_dir / f"{name}.csv", index=False)

    test_dist_df = summarize_test_distribution(test_df, baseline_df, sorted(llm_df["label_name"].unique()))
    test_dist_df.to_csv(args.output_dir / "test_set_distribution_by_label.csv", index=False)

    metric_df = metric_comparison(
        baseline_df=baseline_df,
        remove_df=remove_df,
        refine_df=refine_df,
        label_action_df=summaries["llm_category_by_label"],
        test_dist_df=test_dist_df,
    )
    metric_df.to_csv(args.output_dir / "metric_delta_with_llm_actions_by_label.csv", index=False)

    manual_cases_df = choose_manual_review_cases(
        llm_df,
        metric_df,
        per_group=args.manual_review_cases_per_group,
    )
    manual_cases_df.to_csv(args.output_dir / "manual_review_cases_with_reports.csv", index=False)

    print(f"Audit rows: {len(llm_df)}")
    print(f"Output dir: {args.output_dir}")
    print("Wrote:")
    for path in sorted(args.output_dir.glob("*.csv")):
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
