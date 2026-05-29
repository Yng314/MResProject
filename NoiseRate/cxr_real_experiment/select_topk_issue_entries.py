#!/usr/bin/env python3
"""Select the top-k most suspicious entry-level issues from a cleanlab CSV."""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Select top-k suspicious entry issues.")
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-fraction", type=float, default=None)
    parser.add_argument("--issue-col", type=str, default="est_issue_entry")
    args = parser.parse_args()

    if args.top_k is None and args.top_fraction is None:
        raise ValueError("One of --top-k or --top-fraction must be provided.")
    if args.top_k is not None and args.top_fraction is not None:
        raise ValueError("Use only one of --top-k or --top-fraction.")

    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    if args.issue_col in df.columns:
        df = df[df[args.issue_col] == 1].copy()
    required_cols = {"pool_row_id", "label_index"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV missing required columns {sorted(missing_cols)}: {in_path}")

    sort_cols = []
    ascending = []
    if "entry_quality_self_confidence" in df.columns:
        sort_cols.append("entry_quality_self_confidence")
        ascending.append(True)
    if "entry_issue_rank_self_confidence" in df.columns:
        sort_cols.append("entry_issue_rank_self_confidence")
        ascending.append(True)
    if "pred_prob" in df.columns:
        sort_cols.append("pred_prob")
        ascending.append(True)
    if not sort_cols:
        raise ValueError("No ranking columns found in issue CSV.")

    ranked = df.sort_values(sort_cols, ascending=ascending, kind="stable").reset_index(drop=True)
    n_issue = len(ranked)
    if n_issue == 0:
        raise ValueError(f"No issue entries found in {in_path}")

    if args.top_k is not None:
        n_keep = max(1, min(int(args.top_k), n_issue))
    else:
        if not (0.0 < float(args.top_fraction) <= 1.0):
            raise ValueError("--top-fraction must be in (0, 1].")
        n_keep = max(1, min(n_issue, int(round(n_issue * float(args.top_fraction)))))

    selected = ranked.iloc[:n_keep].copy()
    selected.to_csv(out_path, index=False)

    print(f"Input issue rows: {n_issue}")
    print(f"Selected rows: {len(selected)}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
