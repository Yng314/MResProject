#!/usr/bin/env python3
"""Select a full sample-level top-fraction budget with unseen review entries only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def rank_issue_samples(df: pd.DataFrame, issue_col: str) -> pd.DataFrame:
    ranked = df.copy()
    if issue_col in ranked.columns:
        ranked = ranked[ranked[issue_col] == 1].copy()
    if "pool_row_id" not in ranked.columns:
        raise ValueError("Issue table is missing pool_row_id.")
    if ranked["pool_row_id"].duplicated().any():
        raise ValueError("Issue table contains duplicate pool_row_id values.")

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
        raise ValueError("No sample-level ranking columns were found.")

    return ranked.sort_values(
        sort_cols,
        ascending=ascending,
        kind="stable",
    ).reset_index(drop=True)


def parse_issue_mask(value: object) -> list[int]:
    mask = [int(float(token.strip())) for token in str(value).split("|")]
    if not mask or any(flag not in {0, 1} for flag in mask):
        raise ValueError(f"Invalid issue_entry_mask: {value!r}")
    return mask


def load_review_history(
    history_run_root: Path,
    current_loop: int,
    history_source: str = "expanded",
) -> tuple[set[str], list[dict[str, int]]]:
    if history_source not in {"expanded", "successful-results"}:
        raise ValueError(f"Unsupported history source: {history_source}")

    seen: set[str] = set()
    per_loop: list[dict[str, int]] = []
    for loop_id in range(1, current_loop):
        loop_dir = history_run_root / f"loop_{loop_id:02d}"
        if history_source == "successful-results":
            path = loop_dir / "llm_review" / "results.csv"
        else:
            path = loop_dir / "sample_top_fraction_expanded_entries.csv"
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(
                f"Missing {history_source} review history for Loop {loop_id}: {path}"
            )
        entries = pd.read_csv(path, usecols=["entry_key"])
        keys = entries["entry_key"].astype(str)
        if keys.duplicated().any():
            raise ValueError(f"Loop {loop_id} history contains duplicate entry_key values.")
        before = len(seen)
        seen.update(keys)
        per_loop.append(
            {
                "loop_id": loop_id,
                "history_rows": len(keys),
                "new_unique_entries": len(seen) - before,
                "repeated_entries": len(keys) - (len(seen) - before),
            }
        )
    return seen, per_loop


def selected_entry_keys(selected: pd.DataFrame) -> list[str]:
    keys: list[str] = []
    for row in selected.itertuples(index=False):
        mask = parse_issue_mask(row.issue_entry_mask)
        keys.extend(
            f"{int(row.pool_row_id)}::{label_index}"
            for label_index, flag in enumerate(mask)
            if flag == 1
        )
    return keys


def select_unreviewed_samples(
    issue_df: pd.DataFrame,
    *,
    history_keys: set[str],
    top_fraction: float,
    issue_col: str = "est_issue_sample",
    require_full_budget: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not (0.0 < top_fraction <= 1.0):
        raise ValueError("top_fraction must be in (0, 1].")

    ranked = rank_issue_samples(issue_df, issue_col)
    if ranked.empty:
        raise ValueError("No suspicious samples were found.")
    target = max(1, min(len(ranked), int(round(len(ranked) * top_fraction))))

    selected_rows: list[dict[str, object]] = []
    candidate_samples = 0
    scanned_samples = 0
    original_entries_selected = 0
    filtered_history_entries = 0

    for original_rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        scanned_samples = original_rank
        original_mask = parse_issue_mask(row["issue_entry_mask"])
        original_indices = [
            label_index
            for label_index, flag in enumerate(original_mask)
            if flag == 1
        ]
        expected_count = int(row["est_issue_entry_count"])
        if expected_count != len(original_indices):
            raise ValueError(
                f"Sample {int(row['pool_row_id'])} issue count mismatch: "
                f"{expected_count} != {len(original_indices)}"
            )

        unseen_indices = [
            label_index
            for label_index in original_indices
            if f"{int(row['pool_row_id'])}::{label_index}" not in history_keys
        ]
        if not unseen_indices:
            continue

        candidate_samples += 1
        unseen_mask = [0] * len(original_mask)
        for label_index in unseen_indices:
            unseen_mask[label_index] = 1

        output_row = row.to_dict()
        output_row["original_sample_rank"] = original_rank
        output_row["original_est_issue_entry_count"] = len(original_indices)
        output_row["seen_issue_entry_count"] = len(original_indices) - len(unseen_indices)
        output_row["unseen_issue_entry_count"] = len(unseen_indices)
        output_row["est_issue_entry_count"] = len(unseen_indices)
        output_row["issue_entry_mask"] = "|".join(str(flag) for flag in unseen_mask)
        selected_rows.append(output_row)
        original_entries_selected += len(original_indices)
        filtered_history_entries += len(original_indices) - len(unseen_indices)
        if len(selected_rows) == target:
            break

    if require_full_budget and len(selected_rows) != target:
        raise ValueError(
            "Unable to fill the requested sample budget with unseen entries: "
            f"selected={len(selected_rows)}, target={target}"
        )
    if not selected_rows:
        raise ValueError("No suspicious samples with unseen entries remain.")

    selected = pd.DataFrame(selected_rows)
    keys = selected_entry_keys(selected)
    if len(keys) != len(set(keys)):
        raise ValueError("Selected unseen entry keys are not unique.")
    overlap = sorted(set(keys) & history_keys)
    if overlap:
        raise ValueError(f"Selected entries overlap review history: {overlap[:10]}")

    key_digest = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()
    audit: dict[str, object] = {
        "selection_protocol": "sample_top_fraction_with_global_unseen_entry_backfill_v1",
        "history_definition": (
            "All entry_key values exported for LLM review in every earlier loop, "
            "including rows that later ended in an API error."
        ),
        "top_fraction": top_fraction,
        "total_issue_samples": len(ranked),
        "target_selected_samples": target,
        "selected_samples": len(selected),
        "full_budget_met": len(selected) == target,
        "history_unique_entries": len(history_keys),
        "samples_scanned_from_original_ranking": scanned_samples,
        "backfill_samples_scanned_beyond_original_cutoff": max(0, scanned_samples - target),
        "candidate_samples_with_unseen_entries_encountered": candidate_samples,
        "original_flagged_entries_in_selected_samples": original_entries_selected,
        "history_entries_filtered_inside_selected_samples": filtered_history_entries,
        "unseen_entries_selected_for_review": len(keys),
        "selected_entry_key_sha256": key_digest,
        "history_overlap_entries": 0,
    }
    return selected, audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select the original sample-level top-fraction budget while excluding "
            "every entry reviewed in any prior loop and backfilling lower-ranked samples."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--history-run-root", type=Path, required=True)
    parser.add_argument("--current-loop", type=int, required=True)
    parser.add_argument("--top-fraction", type=float, required=True)
    parser.add_argument("--issue-col", type=str, default="est_issue_sample")
    parser.add_argument(
        "--history-source",
        choices=["expanded", "successful-results"],
        default="expanded",
        help=(
            "Use exported review attempts (legacy behavior) or only successful "
            "LLM result rows when excluding previously reviewed entries."
        ),
    )
    parser.add_argument("--allow-budget-shortfall", action="store_true")
    args = parser.parse_args()

    if args.current_loop < 1:
        raise ValueError("current_loop must be >= 1.")

    issue_df = pd.read_csv(args.input_csv)
    history_keys, per_loop_history = load_review_history(
        args.history_run_root,
        args.current_loop,
        history_source=args.history_source,
    )
    selected, audit = select_unreviewed_samples(
        issue_df,
        history_keys=history_keys,
        top_fraction=args.top_fraction,
        issue_col=args.issue_col,
        require_full_budget=not args.allow_budget_shortfall,
    )
    audit["current_loop"] = args.current_loop
    audit["history_source"] = args.history_source
    if args.history_source == "successful-results":
        audit["selection_protocol"] = "full_or_fractional_issue_pool_successful_history_v2"
        audit["history_definition"] = (
            "Unique entry_key values with successful LLM results in earlier loops; "
            "API failures remain eligible for later review."
        )
    audit["history_loops"] = per_loop_history

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_csv, index=False)
    args.audit_json.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Input issue samples: {audit['total_issue_samples']}")
    print(f"Target selected samples: {audit['target_selected_samples']}")
    print(f"Selected samples with unseen entries: {audit['selected_samples']}")
    print(f"Prior unique reviewed entries: {audit['history_unique_entries']}")
    print(f"Unseen entries selected for review: {audit['unseen_entries_selected_for_review']}")
    print(
        "Original ranking scanned: "
        f"{audit['samples_scanned_from_original_ranking']} samples "
        f"(backfill +{audit['backfill_samples_scanned_beyond_original_cutoff']})"
    )
    print(f"Wrote: {args.output_csv}")
    print(f"Audit: {args.audit_json}")


if __name__ == "__main__":
    main()
