#!/usr/bin/env python3
"""Audit per-loop mechanisms in an own-top20 refinement run."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DEFAULT_SEEDS = [7, 13, 42, 97, 123]
DEFAULT_LOOPS = list(range(1, 9))
TOP_FRACTION = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refine-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slurm-log-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--loops", type=int, nargs="+", default=DEFAULT_LOOPS)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(require_file(path), **kwargs)


def read_one(path: Path) -> pd.Series:
    frame = read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"Expected one row in {path}, found {len(frame)}")
    return frame.iloc[0]


def entry_keys(frame: pd.DataFrame) -> set[str]:
    if frame.empty:
        return set()
    if "entry_key" in frame.columns:
        return set(frame["entry_key"].astype(str))
    return set(
        frame["pool_row_id"].astype(int).astype(str)
        + "::"
        + frame["label_index"].astype(int).astype(str)
    )


def jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else float("nan")


def safe_fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def weighted_auroc(per_label: pd.DataFrame, supported_only: bool = False) -> float:
    frame = per_label.copy()
    if supported_only:
        minority = frame[["study_positive_count", "study_negative_count"]].min(axis=1)
        frame = frame[minority >= 5]
    valid = frame["study_auroc_binary"].notna() & (frame["study_valid_count"] > 0)
    frame = frame[valid]
    if frame.empty:
        return float("nan")
    return float(np.average(frame["study_auroc_binary"], weights=frame["study_valid_count"]))


def parse_final_epoch_diagnostics(log_dir: Path, refine_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    experiment_marker = f"Experiment root: {refine_root}"
    loop_pattern = re.compile(r"Loop (\d+)/8: full train/eval")
    seed_pattern = re.compile(r"^Seed:\s*(\d+)\s*$")
    train_pattern = re.compile(r"epoch 50/50 train_loss=([0-9.]+)")
    val_pattern = re.compile(r"epoch 50/50 val_loss=([0-9.]+)")

    for path in sorted(log_dir.glob("*.out")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if experiment_marker not in text:
            continue
        seed: int | None = None
        loop: int | None = None
        final_train_loss: float | None = None
        for line in text.splitlines():
            seed_match = seed_pattern.match(line)
            if seed_match:
                seed = int(seed_match.group(1))
                continue
            loop_match = loop_pattern.search(line)
            if loop_match:
                loop = int(loop_match.group(1))
                final_train_loss = None
                continue
            train_match = train_pattern.search(line)
            if train_match and seed is not None and loop is not None:
                final_train_loss = float(train_match.group(1))
                continue
            val_match = val_pattern.search(line)
            if val_match and seed is not None and loop is not None:
                rows.append(
                    {
                        "seed": seed,
                        "loop": loop,
                        "final_epoch_train_loss": final_train_loss,
                        "final_epoch_val_loss": float(val_match.group(1)),
                        "training_log": str(path.resolve()),
                    }
                )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No final-epoch diagnostics found in {log_dir}")
    duplicate = frame.duplicated(["seed", "loop"], keep=False)
    if duplicate.any():
        conflicts = (
            frame[duplicate]
            .groupby(["seed", "loop"])[["final_epoch_train_loss", "final_epoch_val_loss"]]
            .nunique(dropna=False)
        )
        if (conflicts > 1).any(axis=None):
            raise ValueError(f"Conflicting final-epoch diagnostics:\n{frame[duplicate].to_string(index=False)}")
        frame = frame.drop_duplicates(["seed", "loop"], keep="last")
    return frame.sort_values(["seed", "loop"]).reset_index(drop=True)


def flatten_summary(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for loop, group in frame.groupby("loop", sort=True):
        row: dict[str, Any] = {"loop": int(loop), "n_seeds": int(group["seed"].nunique())}
        for column in columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_sd"] = float(values.std(ddof=1))
        row["positive_auroc_delta_seeds"] = int((group["delta_study_weighted_auroc"] > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def add_check(checks: list[dict[str, str]], name: str, condition: bool, details: str) -> None:
    checks.append({"check": name, "status": "passed" if condition else "failed", "details": details})


def quality_row(quality: pd.DataFrame, seed: int, loop: int, evidence_mode: str) -> pd.Series:
    match = quality[
        (quality["seed"] == seed)
        & (quality["method"] == "refine")
        & (quality["loop"] == loop)
        & (quality["evidence_mode"] == evidence_mode)
    ]
    if len(match) != 1:
        raise ValueError(
            f"Expected one quality row for seed={seed}, loop={loop}, evidence={evidence_mode}; found {len(match)}"
        )
    return match.iloc[0]


def performance_row(performance: pd.DataFrame, seed: int, method: str, loop: int) -> pd.Series:
    match = performance[
        (performance["seed"] == seed)
        & (performance["method"] == method)
        & (performance["loop"] == loop)
    ]
    if len(match) != 1:
        raise ValueError(f"Expected one performance row for seed={seed}, method={method}, loop={loop}")
    return match.iloc[0]


def build_associations(mechanisms: pd.DataFrame) -> pd.DataFrame:
    predictors = [
        "issue_sample_count",
        "issue_sample_rate",
        "selection_quality_cutoff",
        "selection_quality_median",
        "relabel_rate",
        "mask_rate",
        "keep_rate",
        "action_rate",
        "selected_sample_adjacent_jaccard",
        "review_entry_adjacent_jaccard",
        "selected_sample_novel_fraction",
        "pre_dynamic_coverage_adjusted_dqs",
        "post_coverage_adjusted_dqs",
        "best_val_loss",
        "final_epoch_val_loss",
        "final_minus_best_val_loss",
        "best_epoch",
    ]
    frame = mechanisms[mechanisms["loop"] >= 2].sort_values(["seed", "loop"]).copy()
    outcome = frame["delta_study_weighted_auroc"]
    rows: list[dict[str, Any]] = []
    for predictor in predictors:
        for mode in ["level", "within_seed_delta"]:
            values = frame[predictor]
            if mode == "within_seed_delta":
                values = frame.groupby("seed", sort=False)[predictor].diff()
            valid = values.notna() & outcome.notna()
            x = values[valid].astype(float)
            y = outcome[valid].astype(float)
            if len(x) < 4 or x.nunique() < 2 or y.nunique() < 2:
                rho, p_value = float("nan"), float("nan")
            else:
                result = spearmanr(x, y)
                rho, p_value = float(result.statistic), float(result.pvalue)
            rows.append(
                {
                    "outcome": "delta_study_weighted_auroc",
                    "predictor": predictor,
                    "predictor_mode": mode,
                    "n_cells": int(valid.sum()),
                    "spearman_rho": rho,
                    "naive_p_value": p_value,
                    "inference_status": "exploratory_only; repeated seed-loop cells; no multiplicity correction",
                }
            )
    return pd.DataFrame(rows).sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)


def build_label_associations(label_mechanisms: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for predictor in ["review_count", "relabel_count", "mask_count", "action_count", "action_rate"]:
        valid = label_mechanisms[predictor].notna() & label_mechanisms["label_auroc_delta"].notna()
        x = label_mechanisms.loc[valid, predictor].astype(float)
        y = label_mechanisms.loc[valid, "label_auroc_delta"].astype(float)
        result = spearmanr(x, y)
        rows.append(
            {
                "outcome": "label_auroc_delta",
                "predictor": predictor,
                "n_cells": int(valid.sum()),
                "spearman_rho": float(result.statistic),
                "naive_p_value": float(result.pvalue),
                "inference_status": "exploratory_only; repeated seed-loop-label cells; no multiplicity correction",
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)


def errorbar(ax: plt.Axes, x: pd.Series, mean: pd.Series, sd: pd.Series, **kwargs: Any) -> None:
    ax.errorbar(x, mean, yerr=sd.fillna(0), marker="o", capsize=3, linewidth=2, **kwargs)


def make_overview_figure(
    mechanisms: pd.DataFrame,
    summary: pd.DataFrame,
    baseline: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    loops = summary["loop"]

    ax = axes[0, 0]
    errorbar(
        ax,
        loops,
        summary["study_weighted_auroc_mean"],
        summary["study_weighted_auroc_sd"],
        color="#c23b33",
        label="All 12 labels",
    )
    errorbar(
        ax,
        loops,
        summary["support_ge5_weighted_auroc_mean"],
        summary["support_ge5_weighted_auroc_sd"],
        color="#2673b8",
        label="Minority support >= 5",
    )
    ax.axhline(baseline["study_weighted_auroc"].mean(), color="#c23b33", linestyle="--", alpha=0.65)
    ax.axhline(baseline["support_ge5_weighted_auroc"].mean(), color="#2673b8", linestyle="--", alpha=0.65)
    ax.set_title("Held-out AUROC trajectory")
    ax.set_xlabel("Refinement loop")
    ax.set_ylabel("Study-weighted AUROC")
    ax.set_xticks(loops)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    ax.stackplot(
        loops,
        summary["relabel_rate_mean"],
        summary["mask_rate_mean"],
        summary["keep_rate_mean"],
        labels=["Relabel", "Mask", "Keep"],
        colors=["#c23b33", "#e5a43b", "#8f9aa3"],
        alpha=0.9,
    )
    ax.set_title("LLM action composition")
    ax.set_xlabel("Refinement loop")
    ax.set_ylabel("Fraction of reviewed entries")
    ax.set_ylim(0, 1)
    ax.set_xticks(loops)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1, 0]
    ax.plot(loops, summary["issue_sample_count_mean"], marker="o", linewidth=2, label="CL issue samples")
    ax.plot(loops, summary["selected_sample_count_mean"], marker="o", linewidth=2, label="Selected samples")
    ax.plot(loops, summary["review_entry_count_mean"], marker="o", linewidth=2, label="Reviewed entries")
    ax.set_title("Candidate and review volume")
    ax.set_xlabel("Refinement loop")
    ax.set_ylabel("Mean count across seeds")
    ax.set_xticks(loops)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    overlap = summary[summary["loop"] >= 2]
    errorbar(
        ax,
        overlap["loop"],
        overlap["selected_sample_adjacent_jaccard_mean"],
        overlap["selected_sample_adjacent_jaccard_sd"],
        color="#2673b8",
        label="Selected samples",
    )
    errorbar(
        ax,
        overlap["loop"],
        overlap["review_entry_adjacent_jaccard_mean"],
        overlap["review_entry_adjacent_jaccard_sd"],
        color="#2d8a59",
        label="Reviewed entries",
    )
    ax.set_title("Overlap with the previous loop")
    ax.set_xlabel("Current loop")
    ax.set_ylabel("Jaccard overlap")
    ax.set_ylim(0, 1)
    ax.set_xticks(overlap["loop"])
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    n_seeds = int(summary["n_seeds"].max())
    fig.suptitle(
        f"{n_seeds}-seed own-top20 refinement mechanism audit",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_diagnostic_figure(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    loops = summary["loop"]

    ax = axes[0, 0]
    for column, label, color in [
        ("post_dqs_flattened_mean", "Raw entry DQS", "#2673b8"),
        ("post_coverage_adjusted_dqs_mean", "Coverage-adjusted entry DQS", "#c23b33"),
        ("post_coverage_adjusted_sample_health_mean", "Coverage-adjusted sample DQS", "#2d8a59"),
    ]:
        ax.plot(loops, summary[column], marker="o", linewidth=2, label=label, color=color)
    ax.set_title("Frozen-evidence quality after each loop")
    ax.set_xlabel("Refinement loop")
    ax.set_ylabel("Score")
    ax.set_xticks(loops)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    errorbar(
        ax,
        loops,
        summary["selection_quality_cutoff_mean"],
        summary["selection_quality_cutoff_sd"],
        color="#c23b33",
        label="Top-20% cutoff",
    )
    errorbar(
        ax,
        loops,
        summary["selection_quality_median_mean"],
        summary["selection_quality_median_sd"],
        color="#2673b8",
        label="Selected median",
    )
    ax.set_title("OOF self-confidence among selected samples")
    ax.set_xlabel("Refinement loop")
    ax.set_ylabel("Self-confidence (lower is more suspicious)")
    ax.set_xticks(loops)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    errorbar(
        ax,
        loops,
        summary["best_val_loss_mean"],
        summary["best_val_loss_sd"],
        color="#e07a25",
        label="Best validation loss",
    )
    errorbar(
        ax,
        loops,
        summary["final_epoch_val_loss_mean"],
        summary["final_epoch_val_loss_sd"],
        color="#7b4fa3",
        label="Epoch-50 validation loss",
    )
    ax.set_title("Full-training validation diagnostic")
    ax.set_xlabel("Refinement loop")
    ax.set_ylabel("Validation loss")
    ax.set_xticks(loops)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    errorbar(
        ax,
        loops,
        summary["delta_study_weighted_auroc_mean"],
        summary["delta_study_weighted_auroc_sd"],
        color="#34495e",
        label="Change from previous stage",
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Loop-to-loop AUROC change")
    ax.set_xlabel("Current loop")
    ax.set_ylabel("AUROC delta")
    ax.set_xticks(loops)
    ax.grid(alpha=0.2)

    fig.suptitle("Quality, selection, and training diagnostics", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_label_contribution_figure(label_transitions: pd.DataFrame, output_path: Path) -> None:
    focus = label_transitions[label_transitions["to_loop"].isin([5, 6, 7, 8])]
    pivot = focus.pivot_table(
        index="label_name",
        columns="transition",
        values="weighted_auroc_contribution",
        aggfunc="mean",
    )
    columns = [name for name in ["L4->L5", "L5->L6", "L6->L7", "L7->L8"] if name in pivot.columns]
    pivot = pivot[columns]
    order = pivot.abs().sum(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    bound = max(0.001, float(np.nanmax(np.abs(pivot.to_numpy()))))
    image = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(np.arange(len(columns)), columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("Transition")
    ax.set_title("Mean label contribution to study-weighted AUROC change")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Weighted AUROC contribution")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_review_reuse_figure(summary: pd.DataFrame, output_path: Path) -> None:
    loops = summary["loop"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.bar(
        loops,
        summary["first_time_review_count_mean"],
        color="#2673b8",
        label="First-time entries",
    )
    ax.bar(
        loops,
        summary["rereview_count_mean"],
        bottom=summary["first_time_review_count_mean"],
        color="#8f9aa3",
        label="Previously reviewed entries",
    )
    ax.set_title("Review workload composition")
    ax.set_xlabel("Refinement loop")
    ax.set_ylabel("Mean entries across seeds")
    ax.set_xticks(loops)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(
        loops,
        summary["first_time_action_rate_per_attempt_mean"],
        marker="o",
        linewidth=2,
        color="#c23b33",
        label="First-time entries",
    )
    ax.plot(
        loops,
        summary["rereview_action_rate_per_attempt_mean"],
        marker="o",
        linewidth=2,
        color="#2673b8",
        label="Previously reviewed entries",
    )
    ax.set_title("Fraction resulting in relabel or mask")
    ax.set_xlabel("Refinement loop")
    ax.set_ylabel("Action rate per attempted review")
    ax.set_xticks(loops)
    ax.set_ylim(0, 0.35)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    fig.suptitle("Repeated keep decisions dominate later-loop review", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    refine_root = args.refine_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_dir = refine_root / "evaluation_prelocked_loop5_loop8"
    performance = read_csv(evaluation_dir / "performance" / "per_seed_stage_metrics.csv")
    quality = read_csv(evaluation_dir / "quality" / "own_top20_quality_per_seed.csv")
    manifest = read_csv(refine_root / "binary_v5_reuse_manifest.csv")
    manifest = manifest.set_index("seed", drop=False)
    training_diagnostics = (
        parse_final_epoch_diagnostics(args.slurm_log_dir.resolve(), refine_root)
        if args.slurm_log_dir is not None
        else pd.DataFrame()
    )
    if not training_diagnostics.empty:
        training_diagnostics = (
            training_diagnostics[
                training_diagnostics["seed"].isin(args.seeds)
                & training_diagnostics["loop"].isin(args.loops)
            ]
            .sort_values(["seed", "loop"])
            .reset_index(drop=True)
        )
    training_lookup = {
        (int(row.seed), int(row.loop)): row
        for row in training_diagnostics.itertuples(index=False)
    }

    checks: list[dict[str, str]] = []
    mechanism_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    label_action_rows: list[dict[str, Any]] = []
    action_transition_rows: list[dict[str, Any]] = []
    label_metric_rows: list[dict[str, Any]] = []
    label_transition_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []

    stage_markers_complete = True
    selection_budget_complete = True
    action_partition_complete = True
    llm_attempt_accounting_complete = True
    cumulative_action_complete = True
    metric_alignment_complete = True
    unique_keys_complete = True
    final_epoch_log_coverage_complete = True

    for seed in args.seeds:
        if seed not in manifest.index:
            raise ValueError(f"Missing seed {seed} in binary_v5_reuse_manifest.csv")
        source_loop = Path(str(manifest.loc[seed, "source_loop"]))
        seed_root = source_loop.parents[2]
        baseline_per_label = read_csv(seed_root / "baseline_no_clean" / "test_study_auroc_summary.csv")
        baseline_perf = performance_row(performance, seed, "baseline", 0)
        baseline_recomputed = weighted_auroc(baseline_per_label, supported_only=False)
        baseline_all = float(baseline_perf["study_weighted_auroc"])
        baseline_supported = weighted_auroc(baseline_per_label, supported_only=True)
        metric_alignment_complete &= bool(
            np.isclose(baseline_recomputed, baseline_all, atol=5e-4)
        )
        baseline_rows.append(
            {
                "seed": seed,
                "study_weighted_auroc": baseline_all,
                "support_ge5_weighted_auroc": baseline_supported,
            }
        )

        previous_selected_samples: set[int] | None = None
        previous_review_entries: set[str] | None = None
        previous_action_entries: set[str] | None = None
        last_action_by_entry: dict[str, str] = {}
        ever_selected_samples: set[int] = set()
        ever_review_entries: set[str] = set()
        cumulative_relabel_events = 0
        cumulative_mask_events = 0
        cumulative_relabel_entries: set[str] = set()
        cumulative_mask_entries: set[str] = set()
        previous_perf = baseline_perf
        previous_supported = baseline_supported
        previous_per_label = baseline_per_label

        for loop in args.loops:
            loop_dir = refine_root / f"seed_{seed}" / "llm_refine" / f"loop_{loop:02d}"
            markers = [
                ".oof_complete",
                ".selection_complete",
                ".entry_expansion_complete",
                ".llm_review_complete",
                ".action_tables_complete",
                ".train_eval_complete",
            ]
            stage_markers_complete &= all((loop_dir / marker).is_file() for marker in markers)

            selection_dir = source_loop if loop == 1 else loop_dir
            oof_dir = source_loop / "oof" if loop == 1 else loop_dir / "oof"
            selection = read_csv(
                selection_dir / "sample_top_fraction_issue_subset.csv",
                usecols=[
                    "pool_row_id",
                    "est_issue_entry_count",
                    "sample_quality_self_confidence",
                ],
            )
            oof_summary = read_one(oof_dir / "train_cleanlab_summary.csv")
            issue_count = int(round(float(oof_summary["n_samples"]) * float(oof_summary["estimated_noise_rate_sample"])))
            if loop == 1:
                issue_count = int(manifest.loc[seed, "issue_samples"])
            expected_selected = max(1, min(issue_count, int(round(issue_count * TOP_FRACTION))))
            selection_budget_complete &= len(selection) == expected_selected

            selected_samples = set(selection["pool_row_id"].astype(int))
            selected_quality = selection["sample_quality_self_confidence"].astype(float)
            unique_keys_complete &= len(selected_samples) == len(selection)

            expanded = read_csv(
                loop_dir / "sample_top_fraction_expanded_entries.csv",
                usecols=[
                    "pool_row_id",
                    "label_index",
                    "label_name",
                    "raw_label",
                    "binary_label",
                    "entry_key",
                ],
            )
            review_entries = entry_keys(expanded)
            unique_keys_complete &= len(review_entries) == len(expanded)
            unique_keys_complete &= set(expanded["pool_row_id"].astype(int)) == selected_samples

            llm_summary = read_one(loop_dir / "llm_refinement_summary.csv")
            relabel = read_csv(loop_dir / "llm_relabel_entries.csv")
            mask = read_csv(loop_dir / "llm_mask_entries.csv")
            relabel_entries = entry_keys(relabel)
            mask_entries = entry_keys(mask)
            action_entries = relabel_entries | mask_entries
            unique_keys_complete &= not (relabel_entries & mask_entries)
            unique_keys_complete &= action_entries <= review_entries

            try:
                errors = read_csv(loop_dir / "llm_review" / "errors.csv")
                error_count = len(errors)
            except pd.errors.EmptyDataError:
                errors = pd.DataFrame()
                error_count = 0
            error_entries = entry_keys(errors) if not errors.empty else set()

            review_count = int(llm_summary["llm_review_rows"])
            relabel_count = int(llm_summary["relabel_rows"])
            mask_count = int(llm_summary["mask_rows"])
            keep_count = int(llm_summary["keep_rows"])
            llm_attempt_accounting_complete &= review_count + error_count == len(expanded)
            action_partition_complete &= review_count == relabel_count + mask_count + keep_count
            action_partition_complete &= relabel_count == len(relabel_entries)
            action_partition_complete &= mask_count == len(mask_entries)
            unique_keys_complete &= error_entries <= review_entries
            unique_keys_complete &= not (error_entries & action_entries)

            cumulative_relabel_events += relabel_count
            cumulative_mask_events += mask_count
            cumulative_relabel_entries |= relabel_entries
            cumulative_mask_entries |= mask_entries
            applied_relabel = read_csv(loop_dir / "applied_relabel_entries.csv")
            applied_mask = read_csv(loop_dir / "applied_mask_entries.csv")
            cumulative_action_complete &= entry_keys(applied_relabel) == cumulative_relabel_entries
            cumulative_action_complete &= entry_keys(applied_mask) == cumulative_mask_entries

            action_lookup = pd.Series("keep", index=expanded["entry_key"].astype(str))
            action_lookup.loc[list(relabel_entries)] = "relabel"
            action_lookup.loc[list(mask_entries)] = "mask"
            action_lookup.loc[list(error_entries)] = "error_no_action"
            current_action_by_entry = action_lookup.to_dict()
            transition_counts: dict[tuple[str, str], int] = {}
            for key, current_action in current_action_by_entry.items():
                previous_action = last_action_by_entry.get(key, "not_previously_reviewed")
                pair = (previous_action, current_action)
                transition_counts[pair] = transition_counts.get(pair, 0) + 1
            for (previous_action, current_action), count in transition_counts.items():
                action_transition_rows.append(
                    {
                        "seed": seed,
                        "loop": loop,
                        "previous_action": previous_action,
                        "current_action": current_action,
                        "count": count,
                    }
                )
            repeated_action_keys = set(current_action_by_entry) & set(last_action_by_entry)
            first_time_action_keys = set(current_action_by_entry) - set(last_action_by_entry)
            rereview_same_action_count = sum(
                current_action_by_entry[key] == last_action_by_entry[key] for key in repeated_action_keys
            )
            rereview_prior_keep_count = sum(
                last_action_by_entry[key] == "keep" for key in repeated_action_keys
            )
            first_time_modified_count = len(first_time_action_keys & action_entries)
            rereview_modified_count = len(repeated_action_keys & action_entries)
            label_actions = expanded[["entry_key", "label_name", "raw_label", "binary_label"]].copy()
            label_actions["action"] = label_actions["entry_key"].astype(str).map(action_lookup)
            grouped_actions = (
                label_actions.groupby(["label_name", "action"], dropna=False)
                .size()
                .rename("count")
                .reset_index()
            )
            grouped_actions["seed"] = seed
            grouped_actions["loop"] = loop
            label_action_rows.extend(grouped_actions.to_dict("records"))

            perf = performance_row(performance, seed, "refine", loop)
            per_label = read_csv(loop_dir / "train_eval" / "test_study_auroc_summary.csv")
            recomputed_all_auroc = weighted_auroc(per_label, supported_only=False)
            all_auroc = float(perf["study_weighted_auroc"])
            supported_auroc = weighted_auroc(per_label, supported_only=True)
            metric_alignment_complete &= bool(
                np.isclose(recomputed_all_auroc, all_auroc, atol=5e-4)
            )

            label_frame = per_label.copy()
            label_frame["seed"] = seed
            label_frame["loop"] = loop
            label_frame["minority_support"] = label_frame[
                ["study_positive_count", "study_negative_count"]
            ].min(axis=1)
            label_frame["support_ge5"] = label_frame["minority_support"] >= 5
            label_metric_rows.extend(label_frame.to_dict("records"))

            merged_labels = previous_per_label.merge(
                per_label,
                on=["label_index", "label_name"],
                suffixes=("_previous", "_current"),
                validate="one_to_one",
            )
            total_weight = float(merged_labels["study_valid_count_current"].sum())
            for _, label_row in merged_labels.iterrows():
                delta = float(label_row["study_auroc_binary_current"] - label_row["study_auroc_binary_previous"])
                contribution = float(label_row["study_valid_count_current"] / total_weight * delta)
                label_transition_rows.append(
                    {
                        "seed": seed,
                        "from_loop": loop - 1,
                        "to_loop": loop,
                        "transition": f"L{loop - 1}->L{loop}",
                        "label_index": int(label_row["label_index"]),
                        "label_name": label_row["label_name"],
                        "minority_support": int(
                            min(label_row["study_positive_count_current"], label_row["study_negative_count_current"])
                        ),
                        "support_ge5": bool(
                            min(label_row["study_positive_count_current"], label_row["study_negative_count_current"])
                            >= 5
                        ),
                        "label_auroc_delta": delta,
                        "weighted_auroc_contribution": contribution,
                    }
                )

            frozen = quality_row(quality, seed, loop, "frozen_initial_oof_post_action")
            dynamic = quality_row(quality, seed, loop, "dynamic_iterative_oof_pre_action")
            train_summary = read_one(loop_dir / "train_eval" / "baseline_run_summary.csv")
            training_log_row = training_lookup.get((seed, loop))
            final_epoch_log_coverage_complete &= training_log_row is not None
            final_epoch_train_loss = (
                float(training_log_row.final_epoch_train_loss)
                if training_log_row is not None
                else float("nan")
            )
            final_epoch_val_loss = (
                float(training_log_row.final_epoch_val_loss)
                if training_log_row is not None
                else float("nan")
            )

            if previous_selected_samples is None:
                selected_jaccard = float("nan")
                review_jaccard = float("nan")
                action_jaccard = float("nan")
                selected_retained_fraction = float("nan")
                review_retained_fraction = float("nan")
                selected_novel_fraction = 1.0
                review_novel_fraction = 1.0
            else:
                selected_jaccard = jaccard(selected_samples, previous_selected_samples)
                review_jaccard = jaccard(review_entries, previous_review_entries or set())
                action_jaccard = jaccard(action_entries, previous_action_entries or set())
                selected_retained_fraction = safe_fraction(
                    len(selected_samples & previous_selected_samples), len(selected_samples)
                )
                review_retained_fraction = safe_fraction(
                    len(review_entries & (previous_review_entries or set())), len(review_entries)
                )
                selected_novel_fraction = safe_fraction(len(selected_samples - ever_selected_samples), len(selected_samples))
                review_novel_fraction = safe_fraction(len(review_entries - ever_review_entries), len(review_entries))

            overlap_rows.append(
                {
                    "seed": seed,
                    "loop": loop,
                    "selected_sample_adjacent_jaccard": selected_jaccard,
                    "review_entry_adjacent_jaccard": review_jaccard,
                    "modified_entry_adjacent_jaccard": action_jaccard,
                    "selected_sample_retained_from_previous_fraction": selected_retained_fraction,
                    "review_entry_retained_from_previous_fraction": review_retained_fraction,
                    "selected_sample_novel_fraction": selected_novel_fraction,
                    "review_entry_novel_fraction": review_novel_fraction,
                    "rereview_same_action_fraction": safe_fraction(
                        rereview_same_action_count, len(repeated_action_keys)
                    ),
                    "rereview_prior_keep_fraction": safe_fraction(
                        rereview_prior_keep_count, len(repeated_action_keys)
                    ),
                    "first_time_review_count": len(first_time_action_keys),
                    "first_time_modified_count": first_time_modified_count,
                    "first_time_action_rate_per_attempt": safe_fraction(
                        first_time_modified_count, len(first_time_action_keys)
                    ),
                    "rereview_count": len(repeated_action_keys),
                    "rereview_modified_count": rereview_modified_count,
                    "rereview_action_rate_per_attempt": safe_fraction(
                        rereview_modified_count, len(repeated_action_keys)
                    ),
                }
            )

            mechanism_rows.append(
                {
                    "seed": seed,
                    "loop": loop,
                    "issue_sample_count": issue_count,
                    "issue_sample_rate": float(oof_summary["estimated_noise_rate_sample"]),
                    "issue_entry_rate": float(oof_summary["estimated_noise_rate_entry"]),
                    "selected_sample_count": len(selection),
                    "selected_fraction_of_issue_pool": safe_fraction(len(selection), issue_count),
                    "selected_flagged_entry_count": int(selection["est_issue_entry_count"].sum()),
                    "selection_quality_cutoff": float(selected_quality.max()),
                    "selection_quality_mean": float(selected_quality.mean()),
                    "selection_quality_median": float(selected_quality.median()),
                    "selection_quality_q10": float(selected_quality.quantile(0.10)),
                    "selection_quality_q90": float(selected_quality.quantile(0.90)),
                    "review_entry_count": len(expanded),
                    "review_entries_per_selected_sample": safe_fraction(len(expanded), len(selection)),
                    "llm_success_count": review_count,
                    "llm_success_rate": safe_fraction(review_count, len(expanded)),
                    "relabel_count": relabel_count,
                    "mask_count": mask_count,
                    "keep_count": keep_count,
                    "action_count": relabel_count + mask_count,
                    "relabel_rate": safe_fraction(relabel_count, review_count),
                    "mask_rate": safe_fraction(mask_count, review_count),
                    "keep_rate": safe_fraction(keep_count, review_count),
                    "action_rate": safe_fraction(relabel_count + mask_count, review_count),
                    "response_conflict_count": int(llm_summary["response_conflict_rows"]),
                    "response_conflict_rate": safe_fraction(int(llm_summary["response_conflict_rows"]), review_count),
                    "raw_uncertain_review_count": int(llm_summary["raw_uncertain_review_rows"]),
                    "raw_uncertain_review_rate": safe_fraction(
                        int(llm_summary["raw_uncertain_review_rows"]), review_count
                    ),
                    "cumulative_relabel_event_count": cumulative_relabel_events,
                    "cumulative_mask_event_count": cumulative_mask_events,
                    "cumulative_unique_relabel_count": len(cumulative_relabel_entries),
                    "cumulative_unique_mask_count": len(cumulative_mask_entries),
                    "residual_llm_error_count": error_count,
                    "selected_sample_adjacent_jaccard": selected_jaccard,
                    "review_entry_adjacent_jaccard": review_jaccard,
                    "modified_entry_adjacent_jaccard": action_jaccard,
                    "selected_sample_retained_from_previous_fraction": selected_retained_fraction,
                    "review_entry_retained_from_previous_fraction": review_retained_fraction,
                    "selected_sample_novel_fraction": selected_novel_fraction,
                    "review_entry_novel_fraction": review_novel_fraction,
                    "rereview_same_action_fraction": safe_fraction(
                        rereview_same_action_count, len(repeated_action_keys)
                    ),
                    "rereview_prior_keep_fraction": safe_fraction(
                        rereview_prior_keep_count, len(repeated_action_keys)
                    ),
                    "first_time_review_count": len(first_time_action_keys),
                    "first_time_modified_count": first_time_modified_count,
                    "first_time_action_rate_per_attempt": safe_fraction(
                        first_time_modified_count, len(first_time_action_keys)
                    ),
                    "rereview_count": len(repeated_action_keys),
                    "rereview_modified_count": rereview_modified_count,
                    "rereview_action_rate_per_attempt": safe_fraction(
                        rereview_modified_count, len(repeated_action_keys)
                    ),
                    "pre_dynamic_dqs_flattened": float(dynamic["dqs_flattened"]),
                    "pre_dynamic_coverage_adjusted_dqs": float(dynamic["coverage_adjusted_dqs"]),
                    "pre_dynamic_sample_issue_free_rate": float(dynamic["sample_issue_free_rate"]),
                    "post_dqs_flattened": float(frozen["dqs_flattened"]),
                    "post_coverage_adjusted_dqs": float(frozen["coverage_adjusted_dqs"]),
                    "post_valid_entry_coverage": float(frozen["valid_entry_coverage"]),
                    "post_sample_issue_free_rate": float(frozen["sample_issue_free_rate"]),
                    "post_coverage_adjusted_sample_health": float(frozen["coverage_adjusted_sample_health"]),
                    "post_sample_coverage": float(frozen["sample_coverage"]),
                    "study_weighted_auroc": all_auroc,
                    "support_ge5_weighted_auroc": supported_auroc,
                    "low_support_auroc_gap": all_auroc - supported_auroc,
                    "study_macro_auroc": float(perf["study_macro_auroc"]),
                    "macro_average_precision": float(perf["macro_average_precision"]),
                    "micro_brier": float(perf["micro_brier"]),
                    "micro_nll": float(perf["micro_nll"]),
                    "delta_study_weighted_auroc": all_auroc - float(previous_perf["study_weighted_auroc"]),
                    "delta_support_ge5_weighted_auroc": supported_auroc - previous_supported,
                    "best_epoch": int(train_summary["best_epoch"]),
                    "best_val_loss": float(train_summary["best_val_loss"]),
                    "final_epoch_train_loss": final_epoch_train_loss,
                    "final_epoch_val_loss": final_epoch_val_loss,
                    "final_minus_best_val_loss": final_epoch_val_loss - float(train_summary["best_val_loss"]),
                    "fixed_train_epochs": int(train_summary["epochs"]),
                    "recover_best_weights": int(train_summary["recover_best_weights"]),
                }
            )

            ever_selected_samples |= selected_samples
            ever_review_entries |= review_entries
            previous_selected_samples = selected_samples
            previous_review_entries = review_entries
            previous_action_entries = action_entries
            last_action_by_entry.update(current_action_by_entry)
            previous_perf = perf
            previous_supported = supported_auroc
            previous_per_label = per_label

    mechanisms = pd.DataFrame(mechanism_rows).sort_values(["seed", "loop"])
    overlaps = pd.DataFrame(overlap_rows).sort_values(["seed", "loop"])
    label_actions = pd.DataFrame(label_action_rows).sort_values(["seed", "loop", "label_name", "action"])
    action_transitions = pd.DataFrame(action_transition_rows).sort_values(
        ["seed", "loop", "previous_action", "current_action"]
    )
    label_metrics = pd.DataFrame(label_metric_rows).sort_values(["seed", "loop", "label_index"])
    label_transitions = pd.DataFrame(label_transition_rows).sort_values(
        ["seed", "to_loop", "label_index"]
    )
    baseline = pd.DataFrame(baseline_rows).sort_values("seed")

    label_action_wide = (
        label_actions.pivot_table(
            index=["seed", "loop", "label_name"],
            columns="action",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for column in ["keep", "relabel", "mask", "error_no_action"]:
        if column not in label_action_wide:
            label_action_wide[column] = 0
    label_action_wide = label_action_wide.rename(
        columns={
            "keep": "keep_count",
            "relabel": "relabel_count",
            "mask": "mask_count",
            "error_no_action": "error_count",
        }
    )
    label_mechanisms = label_transitions.merge(
        label_action_wide,
        left_on=["seed", "to_loop", "label_name"],
        right_on=["seed", "loop", "label_name"],
        how="left",
        validate="one_to_one",
    )
    for column in ["keep_count", "relabel_count", "mask_count", "error_count"]:
        label_mechanisms[column] = label_mechanisms[column].fillna(0).astype(int)
    label_mechanisms["review_count"] = label_mechanisms[
        ["keep_count", "relabel_count", "mask_count", "error_count"]
    ].sum(axis=1)
    label_mechanisms["action_count"] = label_mechanisms["relabel_count"] + label_mechanisms["mask_count"]
    label_mechanisms["action_rate"] = label_mechanisms["action_count"] / label_mechanisms["review_count"]

    summary_columns = [
        "issue_sample_count",
        "issue_sample_rate",
        "issue_entry_rate",
        "selected_sample_count",
        "selected_flagged_entry_count",
        "selection_quality_cutoff",
        "selection_quality_median",
        "review_entry_count",
        "review_entries_per_selected_sample",
        "llm_success_count",
        "llm_success_rate",
        "relabel_count",
        "mask_count",
        "keep_count",
        "relabel_rate",
        "mask_rate",
        "keep_rate",
        "action_rate",
        "cumulative_unique_relabel_count",
        "cumulative_unique_mask_count",
        "response_conflict_rate",
        "raw_uncertain_review_rate",
        "residual_llm_error_count",
        "selected_sample_adjacent_jaccard",
        "review_entry_adjacent_jaccard",
        "selected_sample_retained_from_previous_fraction",
        "review_entry_retained_from_previous_fraction",
        "selected_sample_novel_fraction",
        "review_entry_novel_fraction",
        "rereview_same_action_fraction",
        "rereview_prior_keep_fraction",
        "first_time_review_count",
        "first_time_modified_count",
        "first_time_action_rate_per_attempt",
        "rereview_count",
        "rereview_modified_count",
        "rereview_action_rate_per_attempt",
        "pre_dynamic_coverage_adjusted_dqs",
        "post_dqs_flattened",
        "post_coverage_adjusted_dqs",
        "post_valid_entry_coverage",
        "post_sample_issue_free_rate",
        "post_coverage_adjusted_sample_health",
        "post_sample_coverage",
        "study_weighted_auroc",
        "support_ge5_weighted_auroc",
        "low_support_auroc_gap",
        "macro_average_precision",
        "micro_brier",
        "micro_nll",
        "delta_study_weighted_auroc",
        "delta_support_ge5_weighted_auroc",
        "best_epoch",
        "best_val_loss",
        "final_epoch_train_loss",
        "final_epoch_val_loss",
        "final_minus_best_val_loss",
    ]
    loop_summary = flatten_summary(mechanisms, summary_columns)
    associations = build_associations(mechanisms)
    label_associations = build_label_associations(label_mechanisms)

    expected_cells = len(args.seeds) * len(args.loops)
    add_check(
        checks,
        "expected_seed_loop_cells",
        len(mechanisms) == expected_cells,
        f"expected={expected_cells}, observed={len(mechanisms)}",
    )
    add_check(checks, "all_transaction_markers", stage_markers_complete, "6 markers per seed-loop")
    add_check(checks, "selection_budget_exact", selection_budget_complete, "selected=round(issue_pool*0.20)")
    add_check(checks, "unique_and_nested_keys", unique_keys_complete, "sample, review, relabel, and mask keys")
    add_check(checks, "llm_attempt_accounting", llm_attempt_accounting_complete, "expanded=success+residual_error")
    add_check(checks, "llm_action_partition", action_partition_complete, "success=relabel+mask+keep")
    add_check(checks, "cumulative_action_tables", cumulative_action_complete, "applied keys equal cumulative unique action keys")
    add_check(
        checks,
        "performance_metric_alignment",
        metric_alignment_complete,
        "per-label summary recomputation is within 5e-4 of prediction-artifact evaluator",
    )
    if args.slurm_log_dir is not None:
        add_check(
            checks,
            "final_epoch_training_log_coverage",
            final_epoch_log_coverage_complete and len(training_diagnostics) == len(mechanisms),
            f"expected={len(mechanisms)}, observed={len(training_diagnostics)}",
        )
    checks_frame = pd.DataFrame(checks)

    mechanisms.to_csv(output_dir / "per_seed_loop_mechanisms.csv", index=False)
    loop_summary.to_csv(output_dir / "loop_mechanism_summary.csv", index=False)
    overlaps.to_csv(output_dir / "adjacent_loop_overlap.csv", index=False)
    label_actions.to_csv(output_dir / "per_seed_loop_label_actions.csv", index=False)
    action_transitions.to_csv(output_dir / "cross_loop_action_transitions.csv", index=False)
    label_metrics.to_csv(output_dir / "per_seed_loop_label_auroc.csv", index=False)
    label_transitions.to_csv(output_dir / "per_seed_label_transition_contributions.csv", index=False)
    label_mechanisms.to_csv(output_dir / "per_seed_loop_label_mechanisms.csv", index=False)
    baseline.to_csv(output_dir / "baseline_support_sensitivity.csv", index=False)
    associations.to_csv(output_dir / "exploratory_transition_associations.csv", index=False)
    label_associations.to_csv(output_dir / "exploratory_label_action_associations.csv", index=False)
    checks_frame.to_csv(output_dir / "audit_checks.csv", index=False)
    if not training_diagnostics.empty:
        training_diagnostics.to_csv(output_dir / "final_epoch_training_diagnostics.csv", index=False)

    make_overview_figure(mechanisms, loop_summary, baseline, output_dir / "mechanism_overview.png")
    make_diagnostic_figure(loop_summary, output_dir / "quality_training_diagnostics.png")
    make_label_contribution_figure(label_transitions, output_dir / "label_transition_contributions.png")
    make_review_reuse_figure(loop_summary, output_dir / "review_reuse_diagnostics.png")

    metadata = {
        "status": "passed" if (checks_frame["status"] == "passed").all() else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "refine_root": str(refine_root),
        "evaluation_dir": str(evaluation_dir),
        "seeds": args.seeds,
        "loops": args.loops,
        "top_fraction": TOP_FRACTION,
        "slurm_log_dir": str(args.slurm_log_dir.resolve()) if args.slurm_log_dir else None,
        "analysis_scope": (
            "observational per-loop mechanism audit of the completed "
            f"{len(args.seeds)}-seed refinement trajectory"
        ),
        "excluded_default_seeds": sorted(set(DEFAULT_SEEDS) - set(args.seeds)),
        "training_randomness_limitation": (
            "Pipeline seed jointly changes OOF selection, LLM actions, and training. Existing runs cannot isolate "
            "training-only variance; fixed-cleaned-data retraining would be required for causal separation."
        ),
        "cleanlab_threshold_limitation": (
            "The pipeline stores issue sets and quality/rank scores, not Cleanlab's internal per-class confident "
            "thresholds. The audit reports issue-pool size and the operational top-20% self-confidence cutoff."
        ),
        "association_warning": (
            "Spearman associations use repeated seed-loop cells, are unadjusted and exploratory, and do not imply causality."
        ),
        "checks": checks,
    }
    (output_dir / "audit_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if metadata["status"] != "passed":
        failed = checks_frame[checks_frame["status"] != "passed"]
        raise RuntimeError(f"Audit checks failed:\n{failed.to_string(index=False)}")

    print(f"Audit passed: {len(checks)} checks")
    print(f"Mechanism cells: {len(mechanisms)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
