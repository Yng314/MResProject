#!/usr/bin/env python3
"""Forensic audit of systematic Loop 4-8 AUROC oscillation.

This analysis is read-only with respect to experiment artifacts. It reconstructs
entry-level review and effective-label state histories, quantifies cross-seed
agreement, and decomposes held-out AUROC changes to labels and studies.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DEFAULT_SEEDS = [13, 42, 97, 123]
DEFAULT_LOOPS = list(range(1, 9))
FOCUS_LOOPS = [5, 6, 7, 8]
KEY_COLUMNS = ["pool_row_id", "label_index"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refine-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--loops", type=int, nargs="+", default=DEFAULT_LOOPS)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(require_file(path), **kwargs)


def read_optional_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    try:
        return read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def entry_key(frame: pd.DataFrame) -> pd.Series:
    if "entry_key" in frame.columns:
        return frame["entry_key"].astype(str)
    return (
        frame["pool_row_id"].astype(int).astype(str)
        + "::"
        + frame["label_index"].astype(int).astype(str)
    )


def key_set(frame: pd.DataFrame) -> set[str]:
    return set(entry_key(frame)) if not frame.empty else set()


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else float("nan")


def safe_fraction(numerator: float | int, denominator: float | int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def state_text(value: int | str) -> str:
    return str(int(value)) if value in (0, 1, 0.0, 1.0) else str(value)


def parse_pipe_matrix(series: pd.Series) -> np.ndarray:
    rows: list[list[float]] = []
    width: int | None = None
    for value in series.astype(str):
        row = [
            float("nan") if token.strip().lower() == "nan" else float(token)
            for token in value.split("|")
        ]
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError("Inconsistent pipe-delimited matrix width")
        rows.append(row)
    return np.asarray(rows, dtype=float)


def weighted_auroc(per_label: pd.DataFrame, support_ge5: bool = False) -> float:
    frame = per_label.copy()
    if support_ge5:
        minority = frame[["study_positive_count", "study_negative_count"]].min(axis=1)
        frame = frame[minority >= 5]
    frame = frame[
        frame["study_auroc_binary"].notna() & (frame["study_valid_count"] > 0)
    ]
    if frame.empty:
        return float("nan")
    return float(
        np.average(
            frame["study_auroc_binary"].astype(float),
            weights=frame["study_valid_count"].astype(float),
        )
    )


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    condition: bool,
    details: str,
) -> None:
    checks.append(
        {
            "check": name,
            "status": "passed" if condition else "failed",
            "details": details,
        }
    )


def classify_direction(before: int, after: int | str) -> str:
    return f"{before}->{state_text(after)}"


def load_loop_events(
    loop_dir: Path,
    seed: int,
    loop: int,
    state: dict[str, int | str],
    original_state: dict[str, int],
    last_review_action: dict[str, str],
    last_review_loop: dict[str, int],
    review_count: Counter[str],
    relabel_count: Counter[str],
    last_change_loop: dict[str, int],
    last_change_before: dict[str, int],
    all_relabel_keys: set[str],
    all_mask_keys: set[str],
    last_relabel_after: dict[str, int],
    checks: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    expanded = read_csv(
        loop_dir / "sample_top_fraction_expanded_entries.csv",
        usecols=[
            "sample_selected_rank",
            "pool_row_id",
            "subject_id",
            "study_id",
            "label_index",
            "label_name",
            "raw_label",
            "binary_label",
            "pred_prob",
            "entry_quality_self_confidence",
            "entry_issue_rank_self_confidence",
            "entry_key",
        ],
    )
    expanded["entry_key"] = entry_key(expanded)
    if expanded["entry_key"].duplicated().any():
        raise ValueError(f"Duplicate expanded entry keys: seed={seed}, loop={loop}")

    results = read_csv(
        loop_dir / "llm_review" / "results.csv",
        usecols=["entry_key", "mismatch_type", "recommended_action"],
    )
    results["entry_key"] = entry_key(results)
    if results["entry_key"].duplicated().any():
        raise ValueError(f"Duplicate LLM result keys: seed={seed}, loop={loop}")

    errors = read_optional_csv(loop_dir / "llm_review" / "errors.csv")
    if not errors.empty:
        errors["entry_key"] = entry_key(errors)

    relabel = read_csv(loop_dir / "llm_relabel_entries.csv")
    mask = read_csv(loop_dir / "llm_mask_entries.csv")
    relabel["entry_key"] = entry_key(relabel)
    mask["entry_key"] = entry_key(mask)

    expanded_keys = key_set(expanded)
    result_keys = key_set(results)
    error_keys = key_set(errors)
    relabel_keys = key_set(relabel)
    mask_keys = key_set(mask)
    if relabel_keys & mask_keys:
        raise ValueError(f"Relabel/mask overlap: seed={seed}, loop={loop}")
    if (result_keys | error_keys) != expanded_keys:
        raise ValueError(
            f"LLM accounting mismatch: seed={seed}, loop={loop}, "
            f"expanded={len(expanded_keys)}, results={len(result_keys)}, errors={len(error_keys)}"
        )
    if (relabel_keys | mask_keys) - result_keys:
        raise ValueError(f"Action key missing successful result: seed={seed}, loop={loop}")

    relabel_lookup = relabel.set_index("entry_key").to_dict("index")
    mask_lookup = mask.set_index("entry_key").to_dict("index")
    result_lookup = results.set_index("entry_key").to_dict("index")

    rows: list[dict[str, Any]] = []
    review_keys: set[str] = set()
    keep_keys: set[str] = set()
    change_keys: set[str] = set()
    direction_sets: dict[str, set[str]] = {
        "0->1": set(),
        "1->0": set(),
        "0->mask": set(),
        "1->mask": set(),
    }

    for row in expanded.itertuples(index=False):
        key = str(row.entry_key)
        before = int(row.binary_label)
        if key in state:
            if state[key] == "mask":
                raise ValueError(
                    f"Masked entry re-entered review: seed={seed}, loop={loop}, key={key}"
                )
            if int(state[key]) != before:
                raise ValueError(
                    f"Pre-action state mismatch: seed={seed}, loop={loop}, key={key}, "
                    f"ledger={state[key]}, expanded={before}"
                )
        else:
            state[key] = before
            original_state[key] = before

        if key in relabel_keys:
            action = "relabel"
            action_row = relabel_lookup[key]
            after: int | str = int(action_row["new_binary_label"])
            if int(action_row["old_binary_label"]) != before or after != 1 - before:
                raise ValueError(
                    f"Invalid relabel transition: seed={seed}, loop={loop}, key={key}"
                )
        elif key in mask_keys:
            action = "mask"
            action_row = mask_lookup[key]
            after = "mask"
            if int(action_row["binary_label"]) != before:
                raise ValueError(
                    f"Invalid mask transition: seed={seed}, loop={loop}, key={key}"
                )
        elif key in error_keys:
            action = "error_no_action"
            after = before
        else:
            action = "keep"
            after = before

        repeated_review = review_count[key] > 0
        previous_action = last_review_action.get(key, "not_previously_reviewed")
        previous_review_loop = last_review_loop.get(key)
        prior_relabels = relabel_count[key]
        state_change = after != before
        binary_reversal = action == "relabel" and prior_relabels > 0
        immediate_reversal = (
            binary_reversal
            and last_change_loop.get(key) == loop - 1
            and after == last_change_before.get(key)
        )
        returns_to_original = (
            action == "relabel"
            and prior_relabels > 0
            and after == original_state[key]
        )
        direction = classify_direction(before, after) if state_change else "no_change"

        result_row = result_lookup.get(key, {})
        rows.append(
            {
                "seed": seed,
                "loop": loop,
                "transition": f"L{loop - 1}->L{loop}",
                "entry_key": key,
                "pool_row_id": int(row.pool_row_id),
                "subject_id": int(row.subject_id),
                "study_id": int(row.study_id),
                "label_index": int(row.label_index),
                "label_name": str(row.label_name),
                "sample_selected_rank": int(row.sample_selected_rank),
                "raw_label_before": float(row.raw_label),
                "binary_state_before": before,
                "binary_state_after": state_text(after),
                "state_change_direction": direction,
                "oof_pred_prob": float(row.pred_prob),
                "entry_quality_self_confidence": float(row.entry_quality_self_confidence),
                "entry_issue_rank_self_confidence": float(
                    row.entry_issue_rank_self_confidence
                ),
                "llm_action": action,
                "mismatch_type": result_row.get("mismatch_type"),
                "recommended_action": result_row.get("recommended_action"),
                "state_changed": bool(state_change),
                "repeated_review": bool(repeated_review),
                "previous_review_action": previous_action,
                "previous_review_loop": previous_review_loop,
                "prior_review_count": int(review_count[key]),
                "prior_relabel_count": int(prior_relabels),
                "binary_reversal": bool(binary_reversal),
                "immediate_reversal": bool(immediate_reversal),
                "returns_to_original": bool(returns_to_original),
                "changed_in_previous_loop": bool(last_change_loop.get(key) == loop - 1),
            }
        )

        review_keys.add(key)
        if action == "keep":
            keep_keys.add(key)
        if state_change:
            change_keys.add(key)
            direction_sets[direction].add(key)
            last_change_loop[key] = loop
            last_change_before[key] = before
        if action == "relabel":
            all_relabel_keys.add(key)
            last_relabel_after[key] = int(after)
            relabel_count[key] += 1
        if action == "mask":
            all_mask_keys.add(key)

        state[key] = after
        review_count[key] += 1
        last_review_action[key] = action
        last_review_loop[key] = loop

    applied_relabel = read_csv(loop_dir / "applied_relabel_entries.csv")
    applied_mask = read_csv(loop_dir / "applied_mask_entries.csv")
    applied_relabel["entry_key"] = entry_key(applied_relabel)
    applied_mask["entry_key"] = entry_key(applied_mask)
    cumulative_keys_ok = (
        key_set(applied_relabel) == all_relabel_keys
        and key_set(applied_mask) == all_mask_keys
    )
    add_check(
        checks,
        f"seed{seed}_loop{loop}_cumulative_action_keys",
        cumulative_keys_ok,
        f"relabel={len(all_relabel_keys)}, mask={len(all_mask_keys)}",
    )
    applied_relabel_lookup = applied_relabel.set_index("entry_key")["new_binary_label"]
    relabel_state_ok = all(
        int(applied_relabel_lookup.loc[key]) == value
        for key, value in last_relabel_after.items()
    )
    add_check(
        checks,
        f"seed{seed}_loop{loop}_cumulative_relabel_state",
        relabel_state_ok,
        f"checked={len(last_relabel_after)}",
    )

    sets = {
        "review": review_keys,
        "keep": keep_keys,
        "relabel": relabel_keys,
        "mask": mask_keys,
        "change": change_keys,
        **direction_sets,
    }
    return pd.DataFrame(rows), sets


def make_entry_sequences(events: pd.DataFrame) -> pd.DataFrame:
    change_counts = events.groupby(["seed", "entry_key"])["state_changed"].sum()
    eligible = set(change_counts[change_counts >= 2].index)
    if not eligible:
        return pd.DataFrame(
            columns=[
                "seed",
                "entry_key",
                "pool_row_id",
                "study_id",
                "label_index",
                "label_name",
                "review_count",
                "state_change_count",
                "relabel_count",
                "mask_count",
                "binary_reversal_count",
                "immediate_reversal_count",
                "returns_to_original_count",
                "review_sequence",
                "state_sequence",
            ]
        )
    pair_index = pd.MultiIndex.from_frame(events[["seed", "entry_key"]])
    filtered = events[pair_index.isin(eligible)]
    rows: list[dict[str, Any]] = []
    for (seed, key), group in filtered.groupby(["seed", "entry_key"], sort=False):
        group = group.sort_values("loop")
        change_rows = group[group["state_changed"]]
        rows.append(
            {
                "seed": int(seed),
                "entry_key": key,
                "pool_row_id": int(group.iloc[0]["pool_row_id"]),
                "study_id": int(group.iloc[0]["study_id"]),
                "label_index": int(group.iloc[0]["label_index"]),
                "label_name": group.iloc[0]["label_name"],
                "review_count": int(len(group)),
                "state_change_count": int(group["state_changed"].sum()),
                "relabel_count": int(group["llm_action"].eq("relabel").sum()),
                "mask_count": int(group["llm_action"].eq("mask").sum()),
                "binary_reversal_count": int(group["binary_reversal"].sum()),
                "immediate_reversal_count": int(group["immediate_reversal"].sum()),
                "returns_to_original_count": int(group["returns_to_original"].sum()),
                "review_sequence": " | ".join(
                    f"L{int(row.loop)}:{row.llm_action}" for row in group.itertuples()
                ),
                "state_sequence": " | ".join(
                    f"L{int(row.loop)}:{row.binary_state_before}->{row.binary_state_after}"
                    for row in change_rows.itertuples()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["binary_reversal_count", "state_change_count", "review_count"],
        ascending=[False, False, False],
    )


def make_transition_summary(
    events: pd.DataFrame,
    performance: pd.DataFrame,
    action_sets: dict[tuple[int, int], dict[str, set[str]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    focused = events[events["loop"].isin(FOCUS_LOOPS)]
    for (seed, loop), group in focused.groupby(["seed", "loop"], sort=True):
        current_changes = action_sets[(int(seed), int(loop))]["change"]
        previous_changes = action_sets[(int(seed), int(loop) - 1)]["change"]
        perf = performance[
            (performance["seed"] == seed) & (performance["loop"] == loop)
        ].iloc[0]
        rows.append(
            {
                "seed": int(seed),
                "from_loop": int(loop) - 1,
                "to_loop": int(loop),
                "transition": f"L{int(loop) - 1}->L{int(loop)}",
                "weighted_auroc_previous": float(perf["weighted_auroc_previous"]),
                "weighted_auroc_current": float(perf["weighted_auroc_current"]),
                "weighted_auroc_delta": float(perf["weighted_auroc_delta"]),
                "support_ge5_auroc_delta": float(perf["support_ge5_auroc_delta"]),
                "reviewed_entries": int(len(group)),
                "reviewed_samples": int(group["pool_row_id"].nunique()),
                "first_time_reviews": int((~group["repeated_review"]).sum()),
                "repeat_reviews": int(group["repeated_review"].sum()),
                "keep_count": int(group["llm_action"].eq("keep").sum()),
                "relabel_count": int(group["llm_action"].eq("relabel").sum()),
                "mask_count": int(group["llm_action"].eq("mask").sum()),
                "error_count": int(group["llm_action"].eq("error_no_action").sum()),
                "actual_state_changes": int(group["state_changed"].sum()),
                "changed_samples": int(
                    group.loc[group["state_changed"], "pool_row_id"].nunique()
                ),
                "zero_to_one": int(group["state_change_direction"].eq("0->1").sum()),
                "one_to_zero": int(group["state_change_direction"].eq("1->0").sum()),
                "zero_to_mask": int(
                    group["state_change_direction"].eq("0->mask").sum()
                ),
                "one_to_mask": int(
                    group["state_change_direction"].eq("1->mask").sum()
                ),
                "changes_on_repeated_reviews": int(
                    (group["state_changed"] & group["repeated_review"]).sum()
                ),
                "binary_reversals": int(group["binary_reversal"].sum()),
                "immediate_reversals": int(group["immediate_reversal"].sum()),
                "returns_to_original": int(group["returns_to_original"].sum()),
                "changed_again_from_previous_loop": int(
                    group["changed_in_previous_loop"].sum()
                ),
                "change_set_previous_loop_overlap": int(
                    len(current_changes & previous_changes)
                ),
                "change_set_previous_loop_jaccard": jaccard(
                    current_changes, previous_changes
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["seed", "to_loop"])


def load_performance(
    refine_root: Path,
    seeds: list[int],
    loops: list[int],
    checks: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[int, int], pd.DataFrame]]:
    stage_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[int, int], pd.DataFrame] = {}

    for seed in seeds:
        previous_summary: pd.DataFrame | None = None
        previous_all = float("nan")
        previous_supported = float("nan")
        for loop in loops:
            loop_dir = refine_root / f"seed_{seed}" / "llm_refine" / f"loop_{loop:02d}"
            summary = read_csv(loop_dir / "train_eval" / "test_study_auroc_summary.csv")
            all_auc = weighted_auroc(summary)
            supported_auc = weighted_auroc(summary, support_ge5=True)
            pred = read_csv(loop_dir / "train_eval" / "test_study_predictions.csv")
            predictions[(seed, loop)] = pred.sort_values("study_id").reset_index(drop=True)

            if previous_summary is not None:
                merged = previous_summary.merge(
                    summary,
                    on=["label_index", "label_name"],
                    suffixes=("_previous", "_current"),
                    validate="one_to_one",
                )
                total_weight = float(merged["study_valid_count_current"].sum())
                for row in merged.itertuples(index=False):
                    delta = float(
                        row.study_auroc_binary_current
                        - row.study_auroc_binary_previous
                    )
                    label_rows.append(
                        {
                            "seed": seed,
                            "from_loop": loop - 1,
                            "to_loop": loop,
                            "transition": f"L{loop - 1}->L{loop}",
                            "label_index": int(row.label_index),
                            "label_name": row.label_name,
                            "study_valid_count": int(row.study_valid_count_current),
                            "study_positive_count": int(row.study_positive_count_current),
                            "study_negative_count": int(row.study_negative_count_current),
                            "minority_support": int(
                                min(
                                    row.study_positive_count_current,
                                    row.study_negative_count_current,
                                )
                            ),
                            "support_ge5": bool(
                                min(
                                    row.study_positive_count_current,
                                    row.study_negative_count_current,
                                )
                                >= 5
                            ),
                            "label_auroc_previous": float(
                                row.study_auroc_binary_previous
                            ),
                            "label_auroc_current": float(
                                row.study_auroc_binary_current
                            ),
                            "label_auroc_delta": delta,
                            "weighted_auroc_contribution": float(
                                row.study_valid_count_current / total_weight * delta
                            ),
                        }
                    )
                stage_rows.append(
                    {
                        "seed": seed,
                        "from_loop": loop - 1,
                        "loop": loop,
                        "transition": f"L{loop - 1}->L{loop}",
                        "weighted_auroc_previous": previous_all,
                        "weighted_auroc_current": all_auc,
                        "weighted_auroc_delta": all_auc - previous_all,
                        "support_ge5_auroc_previous": previous_supported,
                        "support_ge5_auroc_current": supported_auc,
                        "support_ge5_auroc_delta": supported_auc - previous_supported,
                    }
                )
            previous_summary = summary
            previous_all = all_auc
            previous_supported = supported_auc

    performance = pd.DataFrame(stage_rows)
    labels = pd.DataFrame(label_rows)
    contribution_check = (
        labels.groupby(["seed", "to_loop"])["weighted_auroc_contribution"]
        .sum()
        .rename("recomputed")
        .reset_index()
        .merge(
            performance[["seed", "loop", "weighted_auroc_delta"]],
            left_on=["seed", "to_loop"],
            right_on=["seed", "loop"],
            validate="one_to_one",
        )
    )
    ok = np.allclose(
        contribution_check["recomputed"],
        contribution_check["weighted_auroc_delta"],
        atol=1e-12,
    )
    add_check(
        checks,
        "label_contributions_sum_to_weighted_auroc_delta",
        bool(ok),
        f"cells={len(contribution_check)}",
    )
    return performance, labels, predictions


def pair_score(pos_prob: np.ndarray, neg_prob: np.ndarray) -> np.ndarray:
    diff = pos_prob[:, None] - neg_prob[None, :]
    return (diff > 0).astype(float) + 0.5 * (diff == 0)


def decompose_heldout_predictions(
    predictions: dict[tuple[int, int], pd.DataFrame],
    label_performance: pd.DataFrame,
    seeds: list[int],
    checks: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    study_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    label_names = (
        label_performance[["label_index", "label_name"]]
        .drop_duplicates()
        .sort_values("label_index")
    )

    all_checks_ok = True
    max_stage_rounding_difference = 0.0
    max_transition_rounding_residual = 0.0
    for seed in seeds:
        for loop in FOCUS_LOOPS:
            previous = predictions[(seed, loop - 1)].copy()
            current = predictions[(seed, loop)].copy()
            if not previous["study_id"].equals(current["study_id"]):
                raise ValueError(f"Study order mismatch: seed={seed}, loop={loop}")

            prev_y = parse_pipe_matrix(previous["binary_labels_for_metric"])
            cur_y = parse_pipe_matrix(current["binary_labels_for_metric"])
            prev_valid = parse_pipe_matrix(previous["valid_label_mask"]).astype(bool)
            cur_valid = parse_pipe_matrix(current["valid_label_mask"]).astype(bool)
            prev_prob = parse_pipe_matrix(previous["pred_probs"])
            cur_prob = parse_pipe_matrix(current["pred_probs"])
            labels_equal = np.allclose(prev_y, cur_y, equal_nan=True)
            valid_equal = np.array_equal(prev_valid, cur_valid)
            all_checks_ok &= labels_equal and valid_equal

            total_valid = int(prev_valid.sum())
            for label_row in label_names.itertuples(index=False):
                label_index = int(label_row.label_index)
                valid = prev_valid[:, label_index]
                y = prev_y[valid, label_index].astype(int)
                p0 = prev_prob[valid, label_index]
                p1 = cur_prob[valid, label_index]
                study_ids = previous.loc[valid, "study_id"].astype(int).to_numpy()
                pos_local = np.flatnonzero(y == 1)
                neg_local = np.flatnonzero(y == 0)
                if not len(pos_local) or not len(neg_local):
                    continue

                score0 = pair_score(p0[pos_local], p0[neg_local])
                score1 = pair_score(p1[pos_local], p1[neg_local])
                pair_delta = score1 - score0
                pair_count = int(pair_delta.size)
                auc0 = float(score0.mean())
                auc1 = float(score1.mean())
                rounded_auc_delta = auc1 - auc0
                tied_pairs_previous = int((score0 == 0.5).sum())
                tied_pairs_current = int((score1 == 0.5).sum())

                reference = label_performance[
                    (label_performance["seed"] == seed)
                    & (label_performance["to_loop"] == loop)
                    & (label_performance["label_index"] == label_index)
                ].iloc[0]
                exact_auc_previous = float(reference["label_auroc_previous"])
                exact_auc_current = float(reference["label_auroc_current"])
                exact_auc_delta = float(reference["label_auroc_delta"])
                previous_rounding_bound = 0.5 * tied_pairs_previous / pair_count
                current_rounding_bound = 0.5 * tied_pairs_current / pair_count
                transition_rounding_bound = (
                    previous_rounding_bound + current_rounding_bound
                )
                transition_rounding_residual = (
                    exact_auc_delta - rounded_auc_delta
                )
                max_stage_rounding_difference = max(
                    max_stage_rounding_difference,
                    abs(auc0 - exact_auc_previous),
                    abs(auc1 - exact_auc_current),
                )
                max_transition_rounding_residual = max(
                    max_transition_rounding_residual,
                    abs(transition_rounding_residual),
                )
                all_checks_ok &= bool(
                    abs(auc0 - exact_auc_previous)
                    <= previous_rounding_bound + 1e-12
                    and abs(auc1 - exact_auc_current)
                    <= current_rounding_bound + 1e-12
                    and abs(transition_rounding_residual)
                    <= transition_rounding_bound + 1e-12
                )

                contributions = np.zeros(len(y), dtype=float)
                contributions[pos_local] = 0.5 * pair_delta.sum(axis=1) / pair_count
                contributions[neg_local] = 0.5 * pair_delta.sum(axis=0) / pair_count
                counterpart_count = np.zeros(len(y), dtype=int)
                counterpart_count[pos_local] = len(neg_local)
                counterpart_count[neg_local] = len(pos_local)
                counterpart_correct_previous = np.zeros(len(y), dtype=float)
                counterpart_correct_current = np.zeros(len(y), dtype=float)
                counterpart_correct_previous[pos_local] = score0.sum(axis=1)
                counterpart_correct_current[pos_local] = score1.sum(axis=1)
                counterpart_correct_previous[neg_local] = score0.sum(axis=0)
                counterpart_correct_current[neg_local] = score1.sum(axis=0)
                if not np.isclose(
                    contributions.sum(), rounded_auc_delta, atol=1e-12
                ):
                    raise ValueError(
                        f"Study contribution mismatch: seed={seed}, loop={loop}, "
                        f"label={label_row.label_name}"
                    )
                abs_total = float(np.abs(contributions).sum())
                sorted_abs = np.sort(np.abs(contributions))[::-1]
                changed_pairs = int(np.count_nonzero(pair_delta))
                label_rows.append(
                    {
                        "seed": seed,
                        "from_loop": loop - 1,
                        "to_loop": loop,
                        "transition": f"L{loop - 1}->L{loop}",
                        "label_index": label_index,
                        "label_name": label_row.label_name,
                        "pair_count": pair_count,
                        "changed_pair_count": changed_pairs,
                        "changed_pair_fraction": safe_fraction(
                            changed_pairs, pair_count
                        ),
                        "pairs_gained_equivalent": float(pair_delta[pair_delta > 0].sum()),
                        "pairs_lost_equivalent": float(-pair_delta[pair_delta < 0].sum()),
                        "auc_delta": exact_auc_delta,
                        "rounded_prediction_auc_delta": rounded_auc_delta,
                        "rounding_residual": transition_rounding_residual,
                        "rounding_residual_bound": transition_rounding_bound,
                        "tied_pair_fraction_previous": safe_fraction(
                            tied_pairs_previous, pair_count
                        ),
                        "tied_pair_fraction_current": safe_fraction(
                            tied_pairs_current, pair_count
                        ),
                        "study_attribution_status": (
                            "rounded_predictions_with_explicit_residual"
                            if abs(transition_rounding_residual) > 1e-15
                            else "exact_from_saved_predictions"
                        ),
                        "nonzero_study_count": int(np.count_nonzero(contributions)),
                        "top1_abs_study_share": safe_fraction(
                            sorted_abs[:1].sum(), abs_total
                        ),
                        "top3_abs_study_share": safe_fraction(
                            sorted_abs[:3].sum(), abs_total
                        ),
                    }
                )

                label_weight = int(valid.sum()) / total_valid
                for local_index, contribution in enumerate(contributions):
                    if contribution == 0:
                        continue
                    study_rows.append(
                        {
                            "seed": seed,
                            "from_loop": loop - 1,
                            "to_loop": loop,
                            "transition": f"L{loop - 1}->L{loop}",
                            "label_index": label_index,
                            "label_name": label_row.label_name,
                            "study_id": int(study_ids[local_index]),
                            "target": int(y[local_index]),
                            "probability_previous": float(p0[local_index]),
                            "probability_current": float(p1[local_index]),
                            "probability_delta": float(p1[local_index] - p0[local_index]),
                            "counterpart_count": int(
                                counterpart_count[local_index]
                            ),
                            "counterparts_correctly_ranked_previous": float(
                                counterpart_correct_previous[local_index]
                            ),
                            "counterparts_correctly_ranked_current": float(
                                counterpart_correct_current[local_index]
                            ),
                            "label_auc_contribution": float(contribution),
                            "weighted_auroc_contribution": float(
                                contribution * label_weight
                            ),
                        }
                    )
                if abs(transition_rounding_residual) > 1e-15:
                    study_rows.append(
                        {
                            "seed": seed,
                            "from_loop": loop - 1,
                            "to_loop": loop,
                            "transition": f"L{loop - 1}->L{loop}",
                            "label_index": label_index,
                            "label_name": label_row.label_name,
                            "study_id": -1,
                            "target": -1,
                            "probability_previous": float("nan"),
                            "probability_current": float("nan"),
                            "probability_delta": float("nan"),
                            "counterpart_count": 0,
                            "counterparts_correctly_ranked_previous": float("nan"),
                            "counterparts_correctly_ranked_current": float("nan"),
                            "label_auc_contribution": transition_rounding_residual,
                            "weighted_auroc_contribution": float(
                                transition_rounding_residual * label_weight
                            ),
                        }
                    )

    add_check(
        checks,
        "heldout_labels_validity_and_auc_reconstruction",
        bool(all_checks_ok),
        (
            f"seed_transitions={len(seeds) * len(FOCUS_LOOPS)}; "
            f"max_stage_rounding_difference={max_stage_rounding_difference:.8f}; "
            f"max_transition_rounding_residual={max_transition_rounding_residual:.8f}"
        ),
    )
    studies = pd.DataFrame(study_rows)
    label_decomposition = pd.DataFrame(label_rows)
    overall_check = (
        studies.groupby(["seed", "to_loop"])["weighted_auroc_contribution"]
        .sum()
        .rename("study_sum")
        .reset_index()
        .merge(
            label_performance.groupby(["seed", "to_loop"])[
                "weighted_auroc_contribution"
            ]
            .sum()
            .rename("label_sum")
            .reset_index(),
            on=["seed", "to_loop"],
            validate="one_to_one",
        )
    )
    add_check(
        checks,
        "study_contributions_sum_to_weighted_auroc_delta",
        bool(
            np.allclose(
                overall_check["study_sum"], overall_check["label_sum"], atol=1e-12
            )
        ),
        f"cells={len(overall_check)}",
    )
    return studies, label_decomposition


def build_training_label_state_trajectory(
    refine_root: Path,
    events: pd.DataFrame,
    seeds: list[int],
    loops: list[int],
    checks: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    count_alignment_ok = True
    for seed in seeds:
        initial_path = (
            refine_root
            / f"seed_{seed}"
            / "llm_refine"
            / "loop_01"
            / "oof"
            / "train_cleanlab_per_label_summary.csv"
        )
        initial = read_csv(initial_path).sort_values("label_index")
        positive_delta = Counter()
        valid_delta = Counter()
        seed_events = events[events["seed"] == seed]

        for loop in loops:
            current_events = seed_events[seed_events["loop"] == loop]
            directions = (
                current_events.groupby(["label_index", "state_change_direction"])
                .size()
                .rename("count")
                .reset_index()
            )
            direction_lookup = {
                (int(row.label_index), row.state_change_direction): int(row.count)
                for row in directions.itertuples(index=False)
            }
            for label_index in initial["label_index"].astype(int):
                positive_delta[label_index] += (
                    direction_lookup.get((label_index, "0->1"), 0)
                    - direction_lookup.get((label_index, "1->0"), 0)
                    - direction_lookup.get((label_index, "1->mask"), 0)
                )
                valid_delta[label_index] -= (
                    direction_lookup.get((label_index, "0->mask"), 0)
                    + direction_lookup.get((label_index, "1->mask"), 0)
                )

            observed = read_csv(
                refine_root
                / f"seed_{seed}"
                / "llm_refine"
                / f"loop_{loop:02d}"
                / "train_eval"
                / "train_cleanlab_per_label_summary.csv"
            ).set_index("label_index")
            for initial_row in initial.itertuples(index=False):
                label_index = int(initial_row.label_index)
                current_positive = int(
                    initial_row.positive_count + positive_delta[label_index]
                )
                current_valid = int(initial_row.valid_count + valid_delta[label_index])
                current_negative = current_valid - current_positive
                observed_row = observed.loc[label_index]
                count_alignment_ok &= bool(
                    current_positive == int(observed_row["positive_count"])
                    and current_negative == int(observed_row["negative_count"])
                    and current_valid == int(observed_row["valid_count"])
                )
                initial_minority_class = (
                    "positive"
                    if initial_row.positive_count < initial_row.negative_count
                    else "negative"
                )
                initial_minority_count = int(
                    min(initial_row.positive_count, initial_row.negative_count)
                )
                current_minority_count = int(
                    current_positive
                    if initial_minority_class == "positive"
                    else current_negative
                )
                rows.append(
                    {
                        "seed": seed,
                        "loop": loop,
                        "label_index": label_index,
                        "label_name": initial_row.label_name,
                        "initial_valid_count": int(initial_row.valid_count),
                        "initial_positive_count": int(initial_row.positive_count),
                        "initial_negative_count": int(initial_row.negative_count),
                        "initial_minority_class": initial_minority_class,
                        "initial_minority_count": initial_minority_count,
                        "cumulative_positive_delta": int(positive_delta[label_index]),
                        "cumulative_valid_delta": int(valid_delta[label_index]),
                        "current_valid_count": current_valid,
                        "current_positive_count": current_positive,
                        "current_negative_count": current_negative,
                        "current_positive_prevalence": safe_fraction(
                            current_positive, current_valid
                        ),
                        "current_original_minority_count": current_minority_count,
                        "original_minority_retention": safe_fraction(
                            current_minority_count, initial_minority_count
                        ),
                    }
                )
    add_check(
        checks,
        "reconstructed_training_label_counts_match_post_action_tables",
        bool(count_alignment_ok),
        f"cells={len(seeds) * len(loops) * 12}",
    )
    return pd.DataFrame(rows)


def build_label_exclusion_sensitivity(
    per_seed_label: pd.DataFrame,
) -> pd.DataFrame:
    low_support_labels = set(
        per_seed_label.loc[~per_seed_label["support_ge5"], "label_name"].unique()
    )
    triad = {"Atelectasis", "Lung Opacity", "Pneumothorax"}
    specifications = {
        "all_labels": set(),
        "without_atelectasis": {"Atelectasis"},
        "support_ge5": low_support_labels,
        "without_posthoc_driver_triad": triad,
        "without_low_support_and_posthoc_driver_triad": low_support_labels | triad,
    }
    rows: list[dict[str, Any]] = []
    for (seed, transition), group in per_seed_label.groupby(
        ["seed", "transition"], sort=False
    ):
        for name, excluded in specifications.items():
            frame = group[~group["label_name"].isin(excluded)]
            previous = float(
                np.average(
                    frame["label_auroc_previous"],
                    weights=frame["study_valid_count"],
                )
            )
            current = float(
                np.average(
                    frame["label_auroc_current"],
                    weights=frame["study_valid_count"],
                )
            )
            rows.append(
                {
                    "specification": name,
                    "excluded_labels": " | ".join(sorted(excluded)),
                    "seed": int(seed),
                    "transition": transition,
                    "from_loop": int(group["from_loop"].iloc[0]),
                    "to_loop": int(group["to_loop"].iloc[0]),
                    "included_label_count": int(len(frame)),
                    "weighted_auroc_delta": current - previous,
                }
            )
    return pd.DataFrame(rows)


def build_prediction_delta_similarity(
    predictions: dict[tuple[int, int], pd.DataFrame],
    label_performance: pd.DataFrame,
    seeds: list[int],
) -> pd.DataFrame:
    label_names = (
        label_performance[["label_index", "label_name"]]
        .drop_duplicates()
        .sort_values("label_index")
    )
    rows: list[dict[str, Any]] = []
    for loop in FOCUS_LOOPS:
        parsed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for seed in seeds:
            previous = predictions[(seed, loop - 1)]
            current = predictions[(seed, loop)]
            valid = parse_pipe_matrix(previous["valid_label_mask"]).astype(bool)
            delta = (
                parse_pipe_matrix(current["pred_probs"])
                - parse_pipe_matrix(previous["pred_probs"])
            )
            parsed[seed] = (valid, delta)

        scopes: list[tuple[int, str]] = [(-1, "All valid study-label entries")]
        scopes.extend(
            (int(row.label_index), row.label_name)
            for row in label_names.itertuples(index=False)
        )
        for label_index, label_name in scopes:
            for left, right in itertools.combinations(seeds, 2):
                valid_left, delta_left = parsed[left]
                valid_right, delta_right = parsed[right]
                if label_index == -1:
                    valid = valid_left & valid_right
                    x = delta_left[valid]
                    y = delta_right[valid]
                else:
                    valid = valid_left[:, label_index] & valid_right[:, label_index]
                    x = delta_left[valid, label_index]
                    y = delta_right[valid, label_index]
                if len(x) >= 3 and np.unique(x).size > 1 and np.unique(y).size > 1:
                    rho = float(spearmanr(x, y).statistic)
                else:
                    rho = float("nan")
                nonzero = (np.sign(x) != 0) & (np.sign(y) != 0)
                rows.append(
                    {
                        "loop": loop,
                        "transition": f"L{loop - 1}->L{loop}",
                        "label_index": label_index,
                        "label_name": label_name,
                        "seed_left": left,
                        "seed_right": right,
                        "n_values": int(len(x)),
                        "spearman_rho": rho,
                        "probability_delta_sign_agreement": float(
                            (np.sign(x[nonzero]) == np.sign(y[nonzero])).mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_label_evidence(
    events: pd.DataFrame,
    label_performance: pd.DataFrame,
    heldout_labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    focused = events[events["loop"].isin(FOCUS_LOOPS)].copy()
    action_rows: list[dict[str, Any]] = []
    for (seed, loop, label_index, label_name), group in focused.groupby(
        ["seed", "loop", "label_index", "label_name"], sort=True
    ):
        action_rows.append(
            {
                "seed": seed,
                "to_loop": loop,
                "label_index": label_index,
                "label_name": label_name,
                "review_count": len(group),
                "first_review_count": int((~group["repeated_review"]).sum()),
                "repeat_review_count": int(group["repeated_review"].sum()),
                "keep_count": int(group["llm_action"].eq("keep").sum()),
                "relabel_count": int(group["llm_action"].eq("relabel").sum()),
                "mask_count": int(group["llm_action"].eq("mask").sum()),
                "state_change_count": int(group["state_changed"].sum()),
                "zero_to_one": int(group["state_change_direction"].eq("0->1").sum()),
                "one_to_zero": int(group["state_change_direction"].eq("1->0").sum()),
                "zero_to_mask": int(
                    group["state_change_direction"].eq("0->mask").sum()
                ),
                "one_to_mask": int(
                    group["state_change_direction"].eq("1->mask").sum()
                ),
                "binary_reversal_count": int(group["binary_reversal"].sum()),
                "immediate_reversal_count": int(group["immediate_reversal"].sum()),
            }
        )
    actions = pd.DataFrame(action_rows)
    per_seed_label = (
        label_performance[label_performance["to_loop"].isin(FOCUS_LOOPS)]
        .merge(
            actions,
            on=["seed", "to_loop", "label_index", "label_name"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            heldout_labels,
            on=[
                "seed",
                "from_loop",
                "to_loop",
                "transition",
                "label_index",
                "label_name",
            ],
            how="left",
            validate="one_to_one",
            suffixes=("", "_heldout"),
        )
    )
    count_columns = [
        "review_count",
        "first_review_count",
        "repeat_review_count",
        "keep_count",
        "relabel_count",
        "mask_count",
        "state_change_count",
        "zero_to_one",
        "one_to_zero",
        "zero_to_mask",
        "one_to_mask",
        "binary_reversal_count",
        "immediate_reversal_count",
    ]
    per_seed_label[count_columns] = per_seed_label[count_columns].fillna(0).astype(int)

    aggregate_rows: list[dict[str, Any]] = []
    for (transition, label_index, label_name), group in per_seed_label.groupby(
        ["transition", "label_index", "label_name"], sort=False
    ):
        row: dict[str, Any] = {
            "transition": transition,
            "from_loop": int(group["from_loop"].iloc[0]),
            "to_loop": int(group["to_loop"].iloc[0]),
            "label_index": int(label_index),
            "label_name": label_name,
            "minority_support": int(group["minority_support"].iloc[0]),
            "support_ge5": bool(group["support_ge5"].iloc[0]),
            "mean_label_auroc_delta": float(group["label_auroc_delta"].mean()),
            "sd_label_auroc_delta": float(group["label_auroc_delta"].std(ddof=1)),
            "positive_delta_seeds": int((group["label_auroc_delta"] > 0).sum()),
            "negative_delta_seeds": int((group["label_auroc_delta"] < 0).sum()),
            "mean_weighted_auroc_contribution": float(
                group["weighted_auroc_contribution"].mean()
            ),
            "mean_top1_abs_study_share": float(
                group["top1_abs_study_share"].mean()
            ),
            "mean_top3_abs_study_share": float(
                group["top3_abs_study_share"].mean()
            ),
        }
        for column in count_columns:
            row[f"mean_{column}"] = float(group[column].mean())
            row[f"total_{column}"] = int(group[column].sum())
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    return per_seed_label, aggregate


def cross_seed_analysis(
    events: pd.DataFrame,
    action_sets: dict[tuple[int, int], dict[str, set[str]]],
    state_snapshots: dict[tuple[int, int], dict[str, int | str]],
    global_original: dict[str, int],
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    consensus_rows: list[dict[str, Any]] = []
    action_agreement_rows: list[dict[str, Any]] = []
    state_consensus_rows: list[dict[str, Any]] = []

    for loop in FOCUS_LOOPS:
        for kind in [
            "review",
            "change",
            "relabel",
            "mask",
            "0->1",
            "1->0",
            "0->mask",
            "1->mask",
        ]:
            sets = {seed: action_sets[(seed, loop)][kind] for seed in seeds}
            pair_values: list[float] = []
            for left, right in itertools.combinations(seeds, 2):
                value = jaccard(sets[left], sets[right])
                pair_values.append(value)
                pair_rows.append(
                    {
                        "loop": loop,
                        "transition": f"L{loop - 1}->L{loop}",
                        "set_type": kind,
                        "seed_left": left,
                        "seed_right": right,
                        "left_count": len(sets[left]),
                        "right_count": len(sets[right]),
                        "intersection_count": len(sets[left] & sets[right]),
                        "union_count": len(sets[left] | sets[right]),
                        "jaccard": value,
                    }
                )
            frequency = Counter(
                key for seed_set in sets.values() for key in seed_set
            )
            union = set(frequency)
            intersection = set.intersection(*sets.values()) if sets else set()
            consensus_rows.append(
                {
                    "loop": loop,
                    "transition": f"L{loop - 1}->L{loop}",
                    "set_type": kind,
                    "union_count": len(union),
                    "all_four_count": len(intersection),
                    "all_four_fraction_of_union": safe_fraction(
                        len(intersection), len(union)
                    ),
                    "seen_by_one_seed": sum(value == 1 for value in frequency.values()),
                    "seen_by_two_seeds": sum(value == 2 for value in frequency.values()),
                    "seen_by_three_seeds": sum(value == 3 for value in frequency.values()),
                    "seen_by_four_seeds": sum(value == 4 for value in frequency.values()),
                    "mean_pairwise_jaccard": float(np.nanmean(pair_values)),
                }
            )

        loop_events = events[events["loop"] == loop]
        for key, group in loop_events.groupby("entry_key"):
            if group["seed"].nunique() < 2:
                continue
            actions = group.set_index("seed")["llm_action"].to_dict()
            action_agreement_rows.append(
                {
                    "loop": loop,
                    "transition": f"L{loop - 1}->L{loop}",
                    "entry_key": key,
                    "label_name": group["label_name"].iloc[0],
                    "n_seeds_reviewed": int(group["seed"].nunique()),
                    "n_unique_actions": int(group["llm_action"].nunique()),
                    "all_actions_agree": bool(group["llm_action"].nunique() == 1),
                    "all_change_decisions_agree": bool(
                        group["state_changed"].nunique() == 1
                    ),
                    "actions": " | ".join(
                        f"{seed}:{actions[seed]}" for seed in sorted(actions)
                    ),
                }
            )

        relevant_keys = set().union(
            *[
                {
                    key
                    for prior_loop in range(1, loop + 1)
                    for key in action_sets[(seed, prior_loop)]["change"]
                }
                for seed in seeds
            ]
        )
        for key in relevant_keys:
            states = {
                seed: state_snapshots[(seed, loop)].get(key, global_original[key])
                for seed in seeds
            }
            state_consensus_rows.append(
                {
                    "loop": loop,
                    "entry_key": key,
                    "n_unique_states": len(set(states.values())),
                    "all_states_agree": len(set(states.values())) == 1,
                    "states": " | ".join(
                        f"{seed}:{state_text(states[seed])}" for seed in seeds
                    ),
                }
            )

    return (
        pd.DataFrame(pair_rows),
        pd.DataFrame(consensus_rows),
        pd.DataFrame(action_agreement_rows),
        pd.DataFrame(state_consensus_rows),
    )


def label_action_similarity(
    per_seed_label: pd.DataFrame, seeds: list[int]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = sorted(per_seed_label["label_name"].unique())
    for loop in FOCUS_LOOPS:
        subset = per_seed_label[per_seed_label["to_loop"] == loop]
        for metric in [
            "review_count",
            "state_change_count",
            "zero_to_one",
            "one_to_zero",
            "mask_count",
        ]:
            vectors: dict[int, np.ndarray] = {}
            for seed in seeds:
                lookup = (
                    subset[subset["seed"] == seed]
                    .set_index("label_name")[metric]
                    .to_dict()
                )
                vectors[seed] = np.asarray([lookup.get(label, 0) for label in labels])
            for left, right in itertools.combinations(seeds, 2):
                x, y = vectors[left], vectors[right]
                denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
                cosine = float(np.dot(x, y) / denominator) if denominator else float("nan")
                if np.unique(x).size > 1 and np.unique(y).size > 1:
                    result = spearmanr(x, y)
                    rho = float(result.statistic)
                else:
                    rho = float("nan")
                rows.append(
                    {
                        "loop": loop,
                        "transition": f"L{loop - 1}->L{loop}",
                        "metric": metric,
                        "seed_left": left,
                        "seed_right": right,
                        "cosine_similarity": cosine,
                        "spearman_rho": rho,
                    }
                )
    return pd.DataFrame(rows)


def top_heldout_studies(
    study_contributions: pd.DataFrame,
    seed_count: int,
    top_n: int = 20,
) -> pd.DataFrame:
    aggregate = (
        study_contributions.groupby(
            [
                "transition",
                "from_loop",
                "to_loop",
                "label_index",
                "label_name",
                "study_id",
                "target",
            ],
            as_index=False,
        )
        .agg(
            seeds_nonzero=("seed", "nunique"),
            sum_label_auc_contribution=("label_auc_contribution", "sum"),
            sum_weighted_auroc_contribution=(
                "weighted_auroc_contribution",
                "sum",
            ),
            mean_probability_delta=("probability_delta", "mean"),
            positive_contribution_seeds=(
                "weighted_auroc_contribution",
                lambda values: int((values > 0).sum()),
            ),
            negative_contribution_seeds=(
                "weighted_auroc_contribution",
                lambda values: int((values < 0).sum()),
            ),
        )
    )
    aggregate["mean_label_auc_contribution"] = (
        aggregate["sum_label_auc_contribution"] / seed_count
    )
    aggregate["mean_weighted_auroc_contribution"] = (
        aggregate["sum_weighted_auroc_contribution"] / seed_count
    )
    aggregate["abs_mean_weighted_contribution"] = aggregate[
        "mean_weighted_auroc_contribution"
    ].abs()
    return (
        aggregate.sort_values(
            ["transition", "abs_mean_weighted_contribution"],
            ascending=[True, False],
        )
        .groupby("transition", as_index=False, group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def make_figures(
    output_dir: Path,
    transition_summary: pd.DataFrame,
    aggregate_labels: pd.DataFrame,
    cross_seed_consensus: pd.DataFrame,
) -> None:
    transition_order = ["L4->L5", "L5->L6", "L6->L7", "L7->L8"]
    colors = {
        "L4->L5": "#c23b33",
        "L5->L6": "#2d8a59",
        "L6->L7": "#c23b33",
        "L7->L8": "#2d8a59",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    summary = (
        transition_summary.groupby("transition", as_index=False)
        .agg(
            auroc_mean=("weighted_auroc_delta", "mean"),
            auroc_sd=("weighted_auroc_delta", "std"),
            state_changes=("actual_state_changes", "mean"),
            reversals=("binary_reversals", "mean"),
            repeat_changes=("changes_on_repeated_reviews", "mean"),
        )
        .set_index("transition")
        .loc[transition_order]
        .reset_index()
    )
    x = np.arange(len(summary))
    axes[0, 0].bar(
        x,
        summary["auroc_mean"],
        yerr=summary["auroc_sd"],
        capsize=4,
        color=[colors[value] for value in summary["transition"]],
        alpha=0.85,
    )
    for index, transition in enumerate(transition_order):
        values = transition_summary.loc[
            transition_summary["transition"] == transition, "weighted_auroc_delta"
        ]
        axes[0, 0].scatter(
            np.full(len(values), index),
            values,
            color="#222222",
            s=24,
            zorder=3,
        )
    axes[0, 0].axhline(0, color="#444444", linewidth=1)
    axes[0, 0].set_xticks(x, summary["transition"])
    axes[0, 0].set_ylabel("Weighted AUROC change")
    axes[0, 0].set_title("All four seeds share the alternating direction")

    width = 0.26
    axes[0, 1].bar(
        x - width,
        summary["state_changes"],
        width,
        label="Actual state changes",
        color="#2673b8",
    )
    axes[0, 1].bar(
        x,
        summary["repeat_changes"],
        width,
        label="Changes after prior review",
        color="#e5a43b",
    )
    axes[0, 1].bar(
        x + width,
        summary["reversals"],
        width,
        label="Binary reversals",
        color="#c23b33",
    )
    axes[0, 1].set_xticks(x, summary["transition"])
    axes[0, 1].set_ylabel("Mean entries per seed")
    axes[0, 1].set_title("Actual training-state changes")
    axes[0, 1].legend(frameon=False)

    pivot = aggregate_labels.pivot(
        index="label_name",
        columns="transition",
        values="mean_weighted_auroc_contribution",
    )[transition_order]
    pivot = pivot.loc[pivot.abs().sum(axis=1).sort_values(ascending=False).index]
    limit = float(np.nanmax(np.abs(pivot.to_numpy())))
    image = axes[1, 0].imshow(
        pivot.to_numpy(),
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    axes[1, 0].set_xticks(np.arange(len(transition_order)), transition_order)
    axes[1, 0].set_yticks(np.arange(len(pivot)), pivot.index, fontsize=8)
    axes[1, 0].set_title("Mean label contribution to weighted AUROC change")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)

    consensus = cross_seed_consensus[
        cross_seed_consensus["set_type"].isin(["review", "change"])
    ].copy()
    for kind, color in [("review", "#8f9aa3"), ("change", "#2673b8")]:
        frame = (
            consensus[consensus["set_type"] == kind]
            .set_index("transition")
            .loc[transition_order]
        )
        axes[1, 1].plot(
            transition_order,
            frame["mean_pairwise_jaccard"],
            marker="o",
            linewidth=2,
            label=kind.capitalize(),
            color=color,
        )
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("Mean pairwise Jaccard")
    axes[1, 1].set_title("Cross-seed overlap of reviewed and changed entries")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(alpha=0.2)

    fig.suptitle("Loop 4-8 oscillation forensic audit", fontsize=15)
    fig.savefig(output_dir / "oscillation_forensic_overview.png", dpi=180)
    plt.close(fig)


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    checks: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    metadata = {
        "analysis_name": "loop_oscillation_forensics",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "refine_root": str(args.refine_root.resolve()),
        "seeds": args.seeds,
        "history_loops": args.loops,
        "focus_transitions": ["L4->L5", "L5->L6", "L6->L7", "L7->L8"],
        "review_event_rows": int(len(events)),
        "review_event_focus_rows": int(events["loop"].isin(FOCUS_LOOPS).sum()),
        "checks_total": int(len(checks)),
        "checks_passed": int(checks["status"].eq("passed").sum()),
        "checks_failed": int(checks["status"].eq("failed").sum()),
        "inference_scope": (
            "observational forensic decomposition of saved artifacts; "
            "does not causally isolate training randomness"
        ),
    }
    (output_dir / "audit_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.refine_root = args.refine_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.loops != DEFAULT_LOOPS:
        raise ValueError("This audit requires the complete Loop 1-8 review history")
    if sorted(args.seeds) != sorted(DEFAULT_SEEDS):
        raise ValueError(f"This audit is locked to seeds {DEFAULT_SEEDS}")

    checks: list[dict[str, Any]] = []
    event_frames: list[pd.DataFrame] = []
    action_sets: dict[tuple[int, int], dict[str, set[str]]] = {}
    state_snapshots: dict[tuple[int, int], dict[str, int | str]] = {}
    global_original: dict[str, int] = {}
    original_conflicts: list[str] = []

    for seed in args.seeds:
        state: dict[str, int | str] = {}
        original_state: dict[str, int] = {}
        last_review_action: dict[str, str] = {}
        last_review_loop: dict[str, int] = {}
        review_count: Counter[str] = Counter()
        relabel_count: Counter[str] = Counter()
        last_change_loop: dict[str, int] = {}
        last_change_before: dict[str, int] = {}
        all_relabel_keys: set[str] = set()
        all_mask_keys: set[str] = set()
        last_relabel_after: dict[str, int] = {}

        for loop in args.loops:
            loop_dir = (
                args.refine_root
                / f"seed_{seed}"
                / "llm_refine"
                / f"loop_{loop:02d}"
            )
            required_markers = [
                ".oof_complete",
                ".selection_complete",
                ".entry_expansion_complete",
                ".llm_review_complete",
                ".action_tables_complete",
                ".train_eval_complete",
            ]
            add_check(
                checks,
                f"seed{seed}_loop{loop}_stage_markers",
                all((loop_dir / marker).is_file() for marker in required_markers),
                str(loop_dir),
            )
            loop_events, sets = load_loop_events(
                loop_dir=loop_dir,
                seed=seed,
                loop=loop,
                state=state,
                original_state=original_state,
                last_review_action=last_review_action,
                last_review_loop=last_review_loop,
                review_count=review_count,
                relabel_count=relabel_count,
                last_change_loop=last_change_loop,
                last_change_before=last_change_before,
                all_relabel_keys=all_relabel_keys,
                all_mask_keys=all_mask_keys,
                last_relabel_after=last_relabel_after,
                checks=checks,
            )
            event_frames.append(loop_events)
            action_sets[(seed, loop)] = sets
            state_snapshots[(seed, loop)] = {
                key: value
                for key, value in state.items()
                if value == "mask" or int(value) != original_state[key]
            }

        for key, value in original_state.items():
            if key in global_original and global_original[key] != value:
                original_conflicts.append(key)
            global_original.setdefault(key, value)

    add_check(
        checks,
        "cross_seed_original_binary_state_consistency",
        not original_conflicts,
        f"conflicts={len(original_conflicts)}",
    )
    events = pd.concat(event_frames, ignore_index=True)
    events = events.sort_values(["seed", "loop", "sample_selected_rank", "entry_key"])

    performance, label_performance, predictions = load_performance(
        args.refine_root, args.seeds, args.loops, checks
    )
    transition_summary = make_transition_summary(events, performance, action_sets)
    sequences = make_entry_sequences(events)
    reversal_cases = sequences[sequences["binary_reversal_count"] > 0].copy()
    training_label_states = build_training_label_state_trajectory(
        args.refine_root,
        events,
        args.seeds,
        args.loops,
        checks,
    )
    training_label_state_aggregate = (
        training_label_states.groupby(["loop", "label_index", "label_name"], as_index=False)
        .agg(
            initial_valid_count=("initial_valid_count", "first"),
            initial_positive_count=("initial_positive_count", "first"),
            initial_negative_count=("initial_negative_count", "first"),
            initial_minority_class=("initial_minority_class", "first"),
            initial_minority_count=("initial_minority_count", "first"),
            mean_current_valid_count=("current_valid_count", "mean"),
            mean_current_positive_count=("current_positive_count", "mean"),
            mean_current_negative_count=("current_negative_count", "mean"),
            mean_current_positive_prevalence=("current_positive_prevalence", "mean"),
            mean_original_minority_retention=("original_minority_retention", "mean"),
            min_original_minority_retention=("original_minority_retention", "min"),
            max_original_minority_retention=("original_minority_retention", "max"),
        )
    )

    studies, heldout_labels = decompose_heldout_predictions(
        predictions, label_performance, args.seeds, checks
    )
    per_seed_label, aggregate_labels = aggregate_label_evidence(
        events, label_performance, heldout_labels
    )
    (
        cross_seed_pairs,
        cross_seed_consensus,
        action_agreement,
        state_consensus,
    ) = cross_seed_analysis(
        events,
        action_sets,
        state_snapshots,
        global_original,
        args.seeds,
    )
    label_similarity = label_action_similarity(per_seed_label, args.seeds)
    label_exclusion_sensitivity = build_label_exclusion_sensitivity(per_seed_label)
    label_exclusion_sensitivity_aggregate = (
        label_exclusion_sensitivity.groupby(
            ["specification", "excluded_labels", "transition", "from_loop", "to_loop"],
            as_index=False,
        )
        .agg(
            mean_weighted_auroc_delta=("weighted_auroc_delta", "mean"),
            sd_weighted_auroc_delta=("weighted_auroc_delta", "std"),
            positive_delta_seeds=(
                "weighted_auroc_delta",
                lambda values: int((values > 0).sum()),
            ),
            negative_delta_seeds=(
                "weighted_auroc_delta",
                lambda values: int((values < 0).sum()),
            ),
        )
    )
    prediction_delta_similarity = build_prediction_delta_similarity(
        predictions, label_performance, args.seeds
    )
    top_studies = top_heldout_studies(studies, seed_count=len(args.seeds))

    action_agreement_summary = (
        action_agreement.groupby("transition", as_index=False)
        .agg(
            shared_review_entries=("entry_key", "size"),
            all_action_agreement_rate=("all_actions_agree", "mean"),
            all_change_decision_agreement_rate=("all_change_decisions_agree", "mean"),
            mean_seeds_reviewing_same_entry=("n_seeds_reviewed", "mean"),
        )
    )
    state_consensus_summary = (
        state_consensus.groupby("loop", as_index=False)
        .agg(
            entries_changed_by_any_seed=("entry_key", "size"),
            all_four_effective_state_agreement_rate=("all_states_agree", "mean"),
        )
    )

    checks_frame = pd.DataFrame(checks)
    if checks_frame["status"].eq("failed").any():
        failed = checks_frame[checks_frame["status"].eq("failed")]
        failed.to_csv(args.output_dir / "audit_checks.csv", index=False)
        raise RuntimeError(f"Audit checks failed:\n{failed.to_string(index=False)}")

    events.to_csv(
        args.output_dir / "entry_review_state_ledger.csv.gz",
        index=False,
        compression="gzip",
    )
    sequences.to_csv(args.output_dir / "multi_change_entry_sequences.csv", index=False)
    reversal_cases.to_csv(args.output_dir / "binary_reversal_cases.csv", index=False)
    transition_summary.to_csv(args.output_dir / "transition_summary_per_seed.csv", index=False)
    performance.to_csv(args.output_dir / "performance_transitions.csv", index=False)
    training_label_states.to_csv(
        args.output_dir / "training_label_state_trajectory_per_seed.csv", index=False
    )
    training_label_state_aggregate.to_csv(
        args.output_dir / "training_label_state_trajectory_aggregate.csv", index=False
    )
    per_seed_label.to_csv(
        args.output_dir / "transition_label_evidence_per_seed.csv", index=False
    )
    aggregate_labels.to_csv(
        args.output_dir / "transition_label_evidence_aggregate.csv", index=False
    )
    studies.to_csv(
        args.output_dir / "heldout_study_auc_contributions.csv.gz",
        index=False,
        compression="gzip",
    )
    heldout_labels.to_csv(
        args.output_dir / "heldout_pairwise_decomposition_per_label.csv", index=False
    )
    top_studies.to_csv(args.output_dir / "top_heldout_study_drivers.csv", index=False)
    cross_seed_pairs.to_csv(args.output_dir / "cross_seed_pairwise_overlap.csv", index=False)
    cross_seed_consensus.to_csv(
        args.output_dir / "cross_seed_set_consensus.csv", index=False
    )
    action_agreement.to_csv(
        args.output_dir / "cross_seed_action_agreement_by_entry.csv.gz",
        index=False,
        compression="gzip",
    )
    action_agreement_summary.to_csv(
        args.output_dir / "cross_seed_action_agreement_summary.csv", index=False
    )
    state_consensus.to_csv(
        args.output_dir / "cross_seed_effective_state_consensus_by_entry.csv.gz",
        index=False,
        compression="gzip",
    )
    state_consensus_summary.to_csv(
        args.output_dir / "cross_seed_effective_state_consensus_summary.csv",
        index=False,
    )
    label_similarity.to_csv(
        args.output_dir / "cross_seed_label_composition_similarity.csv", index=False
    )
    label_exclusion_sensitivity.to_csv(
        args.output_dir / "label_exclusion_sensitivity_per_seed.csv", index=False
    )
    label_exclusion_sensitivity_aggregate.to_csv(
        args.output_dir / "label_exclusion_sensitivity_aggregate.csv", index=False
    )
    prediction_delta_similarity.to_csv(
        args.output_dir / "cross_seed_heldout_probability_delta_similarity.csv",
        index=False,
    )
    checks_frame.to_csv(args.output_dir / "audit_checks.csv", index=False)
    make_figures(
        args.output_dir,
        transition_summary,
        aggregate_labels,
        cross_seed_consensus,
    )
    write_metadata(args.output_dir, args, checks_frame, events)

    print(f"Output directory: {args.output_dir}")
    print(f"Review events: {len(events):,}")
    print(f"Binary reversal entries: {len(reversal_cases):,}")
    print(f"Held-out contribution rows: {len(studies):,}")
    print(f"Audit checks: {len(checks_frame)}/{len(checks_frame)} passed")


if __name__ == "__main__":
    main()
