#!/usr/bin/env python3
"""Exploratory global self-confidence threshold sweep for VinDr Entry DQS."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "vindr_iterative_oracle_cleaning/20260806_combined_8seed_v1"
)
MANIFEST = ROOT / "combined_input_manifest.csv"
TRAJECTORIES = ROOT / "evaluation_formal_v1" / "all_seed_trajectories.csv"
SUMMARY = ROOT / "evaluation_formal_v1" / "dynamic_loop_summary.csv"
OUTPUT_DIR = HERE / "next_meeting_slides_20260807" / "extra_figures"
OUTPUT_PNG = OUTPUT_DIR / "vindr_entry_dqs_self_confidence_threshold_sweep.png"
OUTPUT_TRAJECTORY_PNG = OUTPUT_DIR / "vindr_entry_dqs_threshold_trajectories.png"
OUTPUT_SWEEP = OUTPUT_DIR / "vindr_entry_dqs_self_confidence_threshold_sweep.csv"
OUTPUT_CURVE = OUTPUT_DIR / "vindr_entry_dqs_best_threshold_curve.csv"
OUTPUT_ALL_CURVES = OUTPUT_DIR / "vindr_entry_dqs_all_threshold_curves.csv"
OUTPUT_TAU035_CURVE = OUTPUT_DIR / "vindr_entry_dqs_tau035_curve.csv"

FONT = "Times New Roman"
BLUE = "#0077BB"
RED = "#CC3311"
GREEN = "#009988"
GREY = "#667085"
GRID = "#D9DEE7"


def evidence_path(seed_root: Path, loop: int) -> Path:
    initialization = json.loads(
        (seed_root / "state" / "initialization_manifest.json").read_text()
    )
    if loop == 0:
        return Path(initialization["source_blind_run"]) / "entry_evidence.csv"
    return seed_root / f"loop_{loop:02d}" / "oof_after_action" / "entry_evidence.csv"


def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    thresholds = np.linspace(0.0, 1.0, 1001)
    checkpoint_rows: list[dict[str, int]] = []
    dqs_by_checkpoint: list[np.ndarray] = []

    for row in manifest.itertuples(index=False):
        seed_root = Path(row.source_seed_root)
        for loop in range(9):
            evidence = pd.read_csv(
                evidence_path(seed_root, loop),
                usecols=["label_quality_self_confidence"],
            )
            scores = evidence["label_quality_self_confidence"].to_numpy(dtype=float)
            if len(scores) != 18_000 or not np.isfinite(scores).all():
                raise ValueError(f"Invalid score vector for seed {row.seed}, loop {loop}")
            scores.sort()
            issue_counts = np.searchsorted(scores, thresholds, side="left")
            dqs_by_checkpoint.append(1.0 - issue_counts / len(scores))
            checkpoint_rows.append({"seed": int(row.seed), "loop": loop})

    checkpoint = pd.DataFrame(checkpoint_rows)
    dqs_matrix = np.vstack(dqs_by_checkpoint)
    truth = pd.read_csv(TRAJECTORIES)
    truth = truth[truth["method"].eq("dynamic_cl")][
        ["seed", "loop", "true_quality"]
    ]
    checkpoint = checkpoint.merge(truth, on=["seed", "loop"], validate="one_to_one")

    loop_means = np.vstack(
        [dqs_matrix[checkpoint["loop"].eq(loop).to_numpy()].mean(axis=0) for loop in range(9)]
    )
    true_means = (
        checkpoint.groupby("loop", sort=True)["true_quality"].mean().to_numpy(dtype=float)
    )
    errors = loop_means - true_means[:, None]
    mae = np.mean(np.abs(errors), axis=0)
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    bias = np.mean(errors, axis=0)
    best_index = int(np.argmin(mae))
    best_threshold = float(thresholds[best_index])

    sweep = pd.DataFrame(
        {
            "self_confidence_threshold": thresholds,
            "mae_across_9_mean_checkpoints": mae,
            "rmse_across_9_mean_checkpoints": rmse,
            "mean_signed_error": bias,
        }
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(OUTPUT_SWEEP, index=False)

    all_curves = pd.DataFrame(
        {
            "loop": np.repeat(np.arange(9), len(thresholds)),
            "self_confidence_threshold": np.tile(thresholds, 9),
            "mean_thresholded_dqs": loop_means.reshape(-1),
            "known_true_quality": np.repeat(true_means, len(thresholds)),
        }
    )
    all_curves.to_csv(OUTPUT_ALL_CURVES, index=False)

    default = pd.read_csv(SUMMARY).sort_values("loop")
    best_checkpoint_dqs = dqs_matrix[:, best_index]
    checkpoint["thresholded_dqs"] = best_checkpoint_dqs
    best_summary = checkpoint.groupby("loop", as_index=False).agg(
        true_quality_mean=("true_quality", "mean"),
        true_quality_sd=("true_quality", "std"),
        thresholded_dqs_mean=("thresholded_dqs", "mean"),
        thresholded_dqs_sd=("thresholded_dqs", "std"),
    )
    best_summary["raw_dqs_mean"] = default["raw_dqs_mean"]
    best_summary["raw_dqs_sd"] = default["raw_dqs_sd"]
    best_summary["selected_threshold"] = best_threshold
    best_summary.to_csv(OUTPUT_CURVE, index=False)

    tau035_index = 350
    tau035_checkpoint = checkpoint.copy()
    tau035_checkpoint["thresholded_dqs"] = dqs_matrix[:, tau035_index]
    tau035_summary = tau035_checkpoint.groupby("loop", as_index=False).agg(
        thresholded_dqs_mean=("thresholded_dqs", "mean"),
        thresholded_dqs_sd=("thresholded_dqs", "std"),
    )
    tau035_summary["self_confidence_threshold"] = thresholds[tau035_index]
    tau035_summary.to_csv(OUTPUT_TAU035_CURVE, index=False)

    raw_mae = float(
        np.mean(np.abs(default["raw_dqs_mean"].to_numpy() - true_means))
    )
    best_mae = float(mae[best_index])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FONT, "Liberation Serif", "DejaVu Serif"],
            "font.size": 11.5,
            "axes.titlesize": 14,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=220)

    ax = axes[0]
    ax.plot(thresholds, mae, color=GREEN, linewidth=2.2)
    ax.axvline(best_threshold, color=RED, linestyle="--", linewidth=1.7)
    ax.scatter([best_threshold], [best_mae], color=RED, s=42, zorder=3)
    ax.annotate(
        f"Best threshold = {best_threshold:.3f}\nMAE = {best_mae:.3f}",
        xy=(best_threshold, best_mae),
        xytext=(best_threshold + 0.06, best_mae + 0.035),
        arrowprops={"arrowstyle": "-", "color": RED, "linewidth": 1},
        color=RED,
        fontsize=10,
        fontweight="bold",
    )
    ax.set_title("Threshold sweep", loc="left", fontweight="bold")
    ax.set_xlabel("Self-confidence threshold")
    ax.set_ylabel("MAE versus known true quality")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    loops = best_summary["loop"].to_numpy()
    ax.errorbar(
        loops,
        best_summary["true_quality_mean"],
        yerr=best_summary["true_quality_sd"],
        color=BLUE,
        marker="o",
        linewidth=2.4,
        capsize=3,
        label="Known true entry quality",
    )
    ax.errorbar(
        loops,
        best_summary["raw_dqs_mean"],
        yerr=best_summary["raw_dqs_sd"],
        color=RED,
        marker="s",
        linestyle="--",
        linewidth=1.7,
        capsize=3,
        alpha=0.65,
        label=f"Default CL Entry DQS (MAE {raw_mae:.3f})",
    )
    ax.errorbar(
        loops,
        best_summary["thresholded_dqs_mean"],
        yerr=best_summary["thresholded_dqs_sd"],
        color=GREEN,
        marker="D",
        linestyle="-.",
        linewidth=2.2,
        capsize=3,
        label=f"Thresholded DQS, tau={best_threshold:.3f} (MAE {best_mae:.3f})",
    )
    ax.set_title("Best threshold on current data", loc="left", fontweight="bold")
    ax.set_xlabel("Cleaning loop")
    ax.set_ylabel("Entry-level quality score")
    ax.set_xticks(loops)
    ax.set_ylim(0.76, 0.98)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)

    fig.suptitle(
        "VinDr: Exploratory Entry-DQS Self-Confidence Threshold Sweep",
        x=0.07,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.93,
        0.955,
        "In-sample sweep over current 8 seeds and 9 mean checkpoints",
        ha="right",
        color=GREY,
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92], pad=1.0)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    representative = [0.20, 0.25, 0.30, best_threshold, 0.35, 0.40, 0.45]
    colors = ["#8C96C6", "#8C6BB1", "#88419D", GREEN, "#F4A261", "#E76F51", "#9D2A2A"]
    fig, ax = plt.subplots(figsize=(10.2, 5.3), dpi=220)
    ax.plot(
        np.arange(9),
        true_means,
        color=BLUE,
        marker="o",
        linewidth=3.0,
        markersize=6,
        label="Known true entry quality",
        zorder=5,
    )
    ax.plot(
        np.arange(9),
        default["raw_dqs_mean"],
        color=RED,
        marker="s",
        linestyle="--",
        linewidth=2.0,
        alpha=0.75,
        label="Default CL Raw Entry DQS",
        zorder=4,
    )
    for threshold, color in zip(representative, colors):
        threshold_index = int(round(threshold * 1000))
        is_best = np.isclose(threshold, best_threshold)
        ax.plot(
            np.arange(9),
            loop_means[:, threshold_index],
            color=color,
            linewidth=3.0 if is_best else 1.35,
            alpha=1.0 if is_best else 0.66,
            marker="D" if is_best else None,
            markersize=5 if is_best else 0,
            label=(
                f"tau={threshold:.3f} (closest on current benchmark)"
                if is_best
                else f"tau={threshold:.2f}"
            ),
            zorder=4 if is_best else 2,
        )
    ax.set_title("VinDr: Entry-DQS Threshold Sensitivity", loc="left", fontweight="bold")
    ax.text(
        1,
        1.02,
        "1,001 thresholds evaluated; representative trajectories shown",
        transform=ax.transAxes,
        ha="right",
        color=GREY,
        fontsize=9.5,
    )
    ax.set_xlabel("Cleaning loop")
    ax.set_ylabel("Entry-level quality score")
    ax.set_xticks(range(9))
    ax.set_ylim(0.56, 1.01)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.tight_layout(pad=0.9)
    fig.savefig(OUTPUT_TRAJECTORY_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(
        {
            "figure": str(OUTPUT_PNG),
            "trajectory_figure": str(OUTPUT_TRAJECTORY_PNG),
            "best_threshold": best_threshold,
            "default_raw_dqs_mae": raw_mae,
            "thresholded_dqs_mae": best_mae,
            "thresholded_dqs_rmse": float(rmse[best_index]),
            "thresholded_mean_bias": float(bias[best_index]),
        }
    )


if __name__ == "__main__":
    main()
