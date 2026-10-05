#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_seed42_unseen_loop15_auroc import (
    DEFAULT_BASELINE,
    DEFAULT_METRICS,
    weighted_auroc,
)


EXCLUDED_LABELS = frozenset(
    {
        "Atelectasis",
        "Lung Opacity",
        "Pneumothorax",
    }
)
DEFAULT_OUTPUT = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_mobilenetv3_own_top20_binary_v5_refine_unseen_extension/"
    "20260726_seed42_to_loop15/analysis/"
    "seed42_loop01_15_auroc_driver_exclusion_comparison.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare seed42 Loop1-15 AUROC with and without three driver labels."
    )
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_trajectories(metrics_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metrics = pd.read_csv(metrics_path).sort_values("loop_id").reset_index(drop=True)
    observed = metrics["loop_id"].astype(int).tolist()
    if observed != list(range(1, 16)):
        raise ValueError(f"Expected contiguous Loops 1-15, found {observed}")
    if set(metrics["seed"].astype(int)) != {42}:
        raise ValueError("This plot expects only seed 42.")

    loops = metrics["loop_id"].to_numpy(dtype=int)
    all_labels = metrics["test_study_weighted_auroc"].to_numpy(dtype=float)
    without_drivers = []
    for loop_id in loops:
        summary_path = (
            metrics_path.parent
            / f"loop_{loop_id:02d}"
            / "train_eval"
            / "test_study_auroc_summary.csv"
        )
        without_drivers.append(weighted_auroc(summary_path, EXCLUDED_LABELS))

    return loops, all_labels, np.asarray(without_drivers, dtype=float)


def plot_phase_trajectory(
    ax: plt.Axes,
    loops: np.ndarray,
    values: np.ndarray,
    baseline: float,
    panel_title: str,
    annotation_loops: set[int],
) -> None:
    blue = "#2563A6"
    orange = "#D97706"
    ink = "#20242A"
    gray = "#6B7280"
    grid = "#D9DEE5"

    first_phase = loops <= 8
    second_phase = loops >= 8
    ax.plot(
        loops[first_phase],
        values[first_phase],
        color=blue,
        linewidth=2.3,
        marker="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.8,
        label="Loops 1-8: standard top-20",
        zorder=3,
    )
    ax.plot(
        loops[second_phase],
        values[second_phase],
        color=orange,
        linewidth=2.3,
        marker="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.8,
        label="Loops 9-15: unseen-entry",
        zorder=3,
    )
    ax.axhline(
        baseline,
        color=gray,
        linewidth=1.6,
        linestyle=(0, (5, 4)),
        label="Matched no-clean baseline",
        zorder=1,
    )
    ax.text(
        15.25,
        baseline + 0.002,
        f"baseline {baseline:.3f}",
        ha="right",
        va="bottom",
        color=gray,
        fontsize=8.5,
    )
    ax.axvline(8.5, color="#A7AFBA", linewidth=1.1, linestyle=(0, (2, 3)), zorder=1)

    for loop_id, value in zip(loops, values, strict=True):
        if loop_id not in annotation_loops:
            continue
        offset = -14 if loop_id in {5, 6, 11, 15} else 8
        ax.annotate(
            f"{value:.3f}",
            (loop_id, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="top" if offset < 0 else "bottom",
            color=ink,
            fontsize=8.5,
            fontweight="semibold" if loop_id in {4, 8, 15} else "normal",
        )

    ax.set_title(panel_title, loc="left", fontsize=12, fontweight="semibold", color=ink)
    ax.set_ylabel("Study-weighted AUROC", fontsize=10, color=ink)
    ax.set_xticks(np.arange(1, 16))
    ax.set_xlim(0.6, 15.4)
    ax.set_ylim(0.70, 0.83)
    ax.set_yticks(np.arange(0.70, 0.831, 0.02))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.2f}")
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.tick_params(axis="both", colors=ink, labelsize=8.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#A7AFBA")
    ax.spines["bottom"].set_color("#A7AFBA")


def main() -> None:
    args = parse_args()
    loops, all_labels, without_drivers = read_trajectories(args.metrics_csv)
    baseline_all = weighted_auroc(args.baseline_csv)
    baseline_without = weighted_auroc(args.baseline_csv, EXCLUDED_LABELS)

    fig, axes = plt.subplots(2, 1, figsize=(13.2, 10.0), dpi=180, sharex=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    plot_phase_trajectory(
        axes[0],
        loops,
        all_labels,
        baseline_all,
        "A. Original metric: all 12 labels",
        {1, 4, 5, 6, 7, 8, 15},
    )
    plot_phase_trajectory(
        axes[1],
        loops,
        without_drivers,
        baseline_without,
        "B. Post-hoc diagnostic: remaining 9 labels",
        {1, 4, 5, 6, 7, 8, 15},
    )
    axes[1].set_xlabel("Refinement loop", fontsize=10.5, color="#20242A", labelpad=8)

    fig.suptitle(
        "Seed 42 AUROC trajectory before and after excluding three driver labels",
        x=0.075,
        y=0.975,
        ha="left",
        fontsize=17,
        fontweight="semibold",
        color="#20242A",
    )
    fig.text(
        0.075,
        0.94,
        "Excluded: Atelectasis, Lung Opacity, Pneumothorax | "
        "Weights renormalized over remaining labels | Focused shared y-axis",
        fontsize=9.5,
        color="#6B7280",
    )

    handles, labels = axes[1].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=3,
    )
    for text in legend.get_texts():
        text.set_color("#20242A")

    fig.text(
        0.075,
        0.012,
        "Diagnostic exclusion was selected after inspecting outcomes; use it to localize "
        "instability, not as a replacement primary endpoint.",
        fontsize=8.3,
        color="#6B7280",
    )
    fig.subplots_adjust(left=0.075, right=0.98, top=0.90, bottom=0.13, hspace=0.30)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
