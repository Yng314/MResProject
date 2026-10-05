#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_mobilenetv3_own_top20_binary_v5_refine_unseen_extension/"
    "20260726_seed42_to_loop15/seed_42/llm_refine/loop_metrics.csv"
)
DEFAULT_BASELINE = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_mobilenetv3_noes50_clean_3seed/20260707_123320/"
    "seed_42/baseline_no_clean/test_study_auroc_summary.csv"
)
DEFAULT_OUTPUT = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_mobilenetv3_own_top20_binary_v5_refine_unseen_extension/"
    "20260726_seed42_to_loop15/analysis/"
    "seed42_study_weighted_auroc_loop01_15.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot seed42 study-weighted AUROC over refinement Loops 1-15."
    )
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def weighted_auroc(
    path: Path,
    excluded_labels: set[str] | frozenset[str] = frozenset(),
) -> float:
    frame = pd.read_csv(path)
    if excluded_labels:
        observed = set(frame["label_name"].astype(str))
        missing = set(excluded_labels).difference(observed)
        if missing:
            raise ValueError(f"Labels not found in {path}: {sorted(missing)}")
        frame = frame.loc[~frame["label_name"].isin(excluded_labels)].copy()
    values = frame["study_auroc_binary"].to_numpy(dtype=float)
    weights = frame["study_valid_count"].to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        raise ValueError(f"No valid AUROC rows in {path}")
    return float(np.average(values[valid], weights=weights[valid]))


def main() -> None:
    args = parse_args()
    metrics = pd.read_csv(args.metrics_csv)
    required = {"loop_id", "test_study_weighted_auroc", "seed"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Missing columns in {args.metrics_csv}: {sorted(missing)}")

    metrics = metrics.sort_values("loop_id").reset_index(drop=True)
    observed_loops = metrics["loop_id"].astype(int).tolist()
    if observed_loops != list(range(1, 16)):
        raise ValueError(f"Expected contiguous Loops 1-15, found {observed_loops}")
    if set(metrics["seed"].astype(int)) != {42}:
        raise ValueError("This plot expects only seed 42.")

    loops = metrics["loop_id"].to_numpy(dtype=int)
    auroc = metrics["test_study_weighted_auroc"].to_numpy(dtype=float)
    baseline = weighted_auroc(args.baseline_csv)

    blue = "#2563A6"
    orange = "#D97706"
    ink = "#20242A"
    gray = "#6B7280"
    grid = "#D9DEE5"

    fig, ax = plt.subplots(figsize=(13.2, 7.2), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    first_phase = loops <= 8
    second_phase = loops >= 8
    ax.plot(
        loops[first_phase],
        auroc[first_phase],
        color=blue,
        linewidth=2.5,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=2,
        label="Loops 1-8: standard top-20 selection",
        zorder=3,
    )
    ax.plot(
        loops[second_phase],
        auroc[second_phase],
        color=orange,
        linewidth=2.5,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=2,
        label="Loops 9-15: unseen-entry selection",
        zorder=3,
    )
    ax.axhline(
        baseline,
        color=gray,
        linewidth=1.8,
        linestyle=(0, (5, 4)),
        label=f"No-clean baseline ({baseline:.3f})",
        zorder=1,
    )
    ax.axvline(8.5, color="#A7AFBA", linewidth=1.2, linestyle=(0, (2, 3)), zorder=1)
    ax.text(
        8.56,
        0.793,
        "selection rule changes",
        color=gray,
        fontsize=9,
        va="top",
    )

    annotation_loops = {1, 5, 8, 10, 11, 15}
    for loop_id, value in zip(loops, auroc, strict=True):
        if loop_id not in annotation_loops:
            continue
        offset = 9 if loop_id not in {5, 11, 15} else -15
        ax.annotate(
            f"{value:.3f}",
            (loop_id, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if offset > 0 else "top",
            color=ink,
            fontsize=9,
            fontweight="semibold" if loop_id in {8, 10, 15} else "normal",
        )

    ax.set_title(
        "Seed 42 study-weighted AUROC across 15 refinement loops",
        loc="left",
        fontsize=17,
        fontweight="semibold",
        color=ink,
        pad=22,
    )
    ax.text(
        0,
        1.025,
        "Held-out set: 605 studies | Focused y-axis | Single exploratory seed",
        transform=ax.transAxes,
        fontsize=10,
        color=gray,
        va="bottom",
    )
    ax.set_xlabel("Refinement loop", fontsize=11, color=ink, labelpad=10)
    ax.set_ylabel("Study-weighted AUROC", fontsize=11, color=ink, labelpad=10)
    ax.set_xticks(np.arange(1, 16))
    ax.set_xlim(0.6, 15.4)
    ax.set_ylim(0.70, 0.80)
    ax.set_yticks(np.arange(0.70, 0.801, 0.01))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.2f}")
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="both", colors=ink, labelsize=9)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#A7AFBA")
    ax.spines["bottom"].set_color("#A7AFBA")

    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=3,
    )
    for text in legend.get_texts():
        text.set_color(ink)

    fig.text(
        0.075,
        0.015,
        "Source: canonical seed42 loop_metrics.csv; weighted across valid label-study entries.",
        fontsize=8.5,
        color=gray,
    )
    fig.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.24)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
