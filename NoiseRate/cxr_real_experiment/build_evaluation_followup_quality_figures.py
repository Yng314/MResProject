#!/usr/bin/env python3
"""Build denominator-aware entry- and sample-level five-seed quality figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_mobilenetv3_noes50_clean_3seed/20260707_123320/"
    "evaluation_20260713_data_quality_sample_denominator_corrected/"
    "oof_quality_metrics_per_seed.csv"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "evaluation_followup_20260713"
    / "real_data_quality_figures"
)
EVIDENCE_MODE = "frozen_initial_oof_post_action"
SEEDS = [7, 13, 42, 97, 123]
METHOD_LABELS = {"remove": "Simple removal", "refine": "LLM refinement"}
COLORS = {"remove": "#2676B8", "refine": "#C23B32"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "seed",
        "evidence_mode",
        "method",
        "loop",
        "sample_coverage",
        "valid_entry_coverage",
        "dqs_flattened",
        "coverage_adjusted_dqs",
        "sample_issue_rate",
        "sample_issue_count",
        "n_evaluable_samples",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame = frame.loc[frame["evidence_mode"].eq(EVIDENCE_MODE)].copy()
    frame = frame.loc[frame["seed"].isin(SEEDS)].copy()
    if sorted(frame["seed"].unique().tolist()) != SEEDS:
        raise ValueError("Expected exactly seeds 7, 13, 42, 97, 123")
    expected_rows = len(SEEDS) * (1 + 5 + 5)
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} frozen-evidence rows, found {len(frame)}")
    return frame


def add_sample_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["sample_issue_free_rate"] = 1.0 - result["sample_issue_rate"]
    result["coverage_adjusted_sample_health"] = (
        result["sample_issue_free_rate"] * result["sample_coverage"]
    )
    return result


def summarize(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    summary = (
        frame.groupby(["method", "loop"], as_index=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        column if isinstance(column, str) else "_".join(item for item in column if item)
        for column in summary.columns
    ]
    summary["n_seeds"] = len(SEEDS)
    return summary


def method_trajectory(
    summary: pd.DataFrame, baseline: pd.Series, method: str
) -> pd.DataFrame:
    rows = summary.loc[summary["method"].eq(method)].copy()
    baseline_row = {column: np.nan for column in rows.columns}
    baseline_row.update(baseline.to_dict())
    baseline_row["method"] = method
    baseline_row["loop"] = 0
    return pd.concat([pd.DataFrame([baseline_row]), rows], ignore_index=True).sort_values("loop")


def plot_metric_pair(
    *,
    per_seed: pd.DataFrame,
    summary: pd.DataFrame,
    raw_metric: str,
    adjusted_metric: str,
    raw_title: str,
    adjusted_title: str,
    ylabel: str,
    ylim: tuple[float, float] | None,
    output_path: Path,
) -> None:
    baseline = summary.loc[
        summary["method"].eq("baseline") & summary["loop"].eq(0)
    ].iloc[0]
    if ylim is None:
        plotted = per_seed[[raw_metric, adjusted_metric]].to_numpy(dtype=float)
        value_min = float(np.nanmin(plotted))
        value_max = float(np.nanmax(plotted))
        padding = max(0.02, 0.08 * (value_max - value_min))
        ylim = (max(0.0, value_min - padding), min(1.0, value_max + padding))
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.8), sharex=True, sharey=True)
    for axis, metric, title in zip(
        axes,
        [raw_metric, adjusted_metric],
        [raw_title, adjusted_title],
        strict=True,
    ):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        for method in ["remove", "refine"]:
            seed_rows = per_seed.loc[per_seed["method"].eq(method)]
            baseline_seed = per_seed.loc[per_seed["method"].eq("baseline")]
            for seed in SEEDS:
                values = pd.concat(
                    [
                        baseline_seed.loc[baseline_seed["seed"].eq(seed)],
                        seed_rows.loc[seed_rows["seed"].eq(seed)],
                    ]
                ).sort_values("loop")
                axis.plot(
                    values["loop"],
                    values[metric],
                    color=COLORS[method],
                    alpha=0.14,
                    linewidth=1.0,
                    marker="o",
                    markersize=2.5,
                    zorder=1,
                )

            trajectory = method_trajectory(summary, baseline, method)
            axis.errorbar(
                trajectory["loop"],
                trajectory[mean_col],
                yerr=trajectory[std_col].fillna(0.0),
                color="white",
                linewidth=5.0,
                elinewidth=5.0,
                capsize=7,
                capthick=5.0,
                zorder=2,
            )
            axis.errorbar(
                trajectory["loop"],
                trajectory[mean_col],
                yerr=trajectory[std_col].fillna(0.0),
                color=COLORS[method],
                linewidth=2.6,
                elinewidth=2.2,
                capsize=6,
                capthick=2.2,
                marker="o",
                markersize=6.5,
                label=METHOD_LABELS[method],
                zorder=3,
            )
            final = trajectory.loc[trajectory["loop"].eq(5)].iloc[0]
            axis.annotate(
                f"{final[mean_col]:.3f} +/- {final[std_col]:.3f}",
                xy=(5, final[mean_col]),
                xytext=(-6, 13 if method == "remove" else -18),
                textcoords="offset points",
                color=COLORS[method],
                fontsize=9,
                ha="right",
                fontweight="bold",
            )

        axis.set_title(title, fontsize=13, fontweight="bold")
        axis.set_xticks(range(6))
        axis.set_xlabel("Iteration (0 = baseline)")
        axis.set_ylim(*ylim)
        axis.grid(axis="y", color="#DDE3EA", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(
        "Five-seed mean +/- SD; faint lines show individual seeds",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def validate_anchors(frame: pd.DataFrame) -> dict[str, float]:
    for seed, rows in frame.groupby("seed"):
        baseline = rows.loc[rows["method"].eq("baseline") & rows["loop"].eq(0)]
        if len(baseline) != 1:
            raise ValueError(f"Expected one baseline row for seed {seed}")
        baseline_evaluable = int(baseline.iloc[0]["n_evaluable_samples"])
        expected_coverage = rows["n_evaluable_samples"] / baseline_evaluable
        expected_issue_rate = rows["sample_issue_count"] / rows["n_evaluable_samples"]
        if not np.allclose(rows["sample_coverage"], expected_coverage, atol=1e-12):
            raise AssertionError(f"Sample coverage denominator mismatch for seed {seed}")
        if not np.allclose(rows["sample_issue_rate"], expected_issue_rate, atol=1e-12):
            raise AssertionError(f"Sample issue-rate denominator mismatch for seed {seed}")
        if not np.isclose(float(baseline.iloc[0]["sample_coverage"]), 1.0, atol=1e-12):
            raise AssertionError(f"Baseline sample coverage is not one for seed {seed}")
    if not np.allclose(
        frame["coverage_adjusted_sample_health"],
        frame["sample_issue_free_rate"] * frame["sample_coverage"],
        atol=1e-12,
    ):
        raise AssertionError("Coverage-adjusted sample health identity failed")

    anchors: dict[str, float] = {}
    for method, loop in [("baseline", 0), ("remove", 5), ("refine", 5)]:
        rows = frame.loc[frame["method"].eq(method) & frame["loop"].eq(loop)]
        if len(rows) != len(SEEDS):
            raise ValueError(f"Expected five rows for {method} loop {loop}")
        prefix = f"{method}_l{loop}"
        anchors[f"{prefix}_raw_dqs"] = float(rows["dqs_flattened"].mean())
        anchors[f"{prefix}_adjusted_dqs"] = float(rows["coverage_adjusted_dqs"].mean())
        anchors[f"{prefix}_sample_issue_free_rate"] = float(
            rows["sample_issue_free_rate"].mean()
        )
        anchors[f"{prefix}_adjusted_sample_health"] = float(
            rows["coverage_adjusted_sample_health"].mean()
        )

    expected = {
        "baseline_l0_raw_dqs": 0.9277,
        "remove_l5_raw_dqs": 0.9745,
        "remove_l5_adjusted_dqs": 0.8097,
        "refine_l5_raw_dqs": 0.9374,
        "refine_l5_adjusted_dqs": 0.9334,
    }
    for key, target in expected.items():
        if abs(anchors[key] - target) > 0.002:
            raise AssertionError(f"Anchor {key}={anchors[key]:.6f}, expected near {target}")
    return anchors


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed = add_sample_metrics(load_metrics(args.input))
    metrics = [
        "dqs_flattened",
        "coverage_adjusted_dqs",
        "valid_entry_coverage",
        "sample_issue_free_rate",
        "coverage_adjusted_sample_health",
        "sample_coverage",
    ]
    summary = summarize(per_seed, metrics)
    anchors = validate_anchors(per_seed)

    per_seed.to_csv(args.output_dir / "quality_metrics_per_seed.csv", index=False)
    summary.to_csv(args.output_dir / "quality_metrics_five_seed_summary.csv", index=False)
    plot_metric_pair(
        per_seed=per_seed,
        summary=summary,
        raw_metric="dqs_flattened",
        adjusted_metric="coverage_adjusted_dqs",
        raw_title="Raw entry DQS (current valid entries)",
        adjusted_title="Coverage-adjusted entry DQS (original denominator)",
        ylabel="Entry-level quality score",
        ylim=(0.79, 0.99),
        output_path=args.output_dir / "entry_dqs_common_scale_five_seed.png",
    )
    plot_metric_pair(
        per_seed=per_seed,
        summary=summary,
        raw_metric="sample_issue_free_rate",
        adjusted_metric="coverage_adjusted_sample_health",
        raw_title="Raw sample DQS (current evaluable samples)",
        adjusted_title="Coverage-adjusted sample DQS (original denominator)",
        ylabel="Sample-level quality score",
        ylim=None,
        output_path=args.output_dir / "sample_issue_free_rate_five_seed.png",
    )

    metadata = {
        "status": "passed",
        "source": str(args.input),
        "evidence_mode": EVIDENCE_MODE,
        "seeds": SEEDS,
        "definitions": {
            "sample_issue_free_rate": "1 - sample_issue_rate among samples retaining at least one valid target label; a sample is problematic if any valid entry is flagged by confident learning",
            "sample_coverage": "evaluable samples divided by baseline evaluable samples; zero-valid samples are not counted as issue-free",
            "coverage_adjusted_sample_health": "sample_issue_free_rate * evaluable-sample coverage relative to baseline evaluable samples",
            "coverage_adjusted_dqs": "entry DQS * valid-entry coverage relative to baseline",
        },
        "status_notes": {
            "sample_metric": "custom strict sample-level diagnostic, not an official Cleanlab DQS",
            "adjusted_metrics": "custom denominator diagnostics, not official Cleanlab metrics",
            "error_bars": "sample standard deviation across five seeds",
        },
        "anchors": anchors,
    }
    (args.output_dir / "quality_figure_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
