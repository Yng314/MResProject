#!/usr/bin/env python3
"""Exploratory in-sample affine calibration of VinDr raw entry DQS."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "vindr_iterative_oracle_cleaning/20260806_combined_8seed_v1/"
    "evaluation_formal_v1/dynamic_loop_summary.csv"
)
OUTPUT_DIR = HERE / "next_meeting_slides_20260807" / "extra_figures"
OUTPUT_PNG = OUTPUT_DIR / "vindr_entry_dqs_affine_calibration.png"
OUTPUT_CSV = OUTPUT_DIR / "vindr_entry_dqs_affine_calibration.csv"

FONT = "Times New Roman"
BLUE = "#0077BB"
RED = "#CC3311"
GREEN = "#009988"
GREY = "#667085"
GRID = "#D9DEE7"


def main() -> None:
    data = pd.read_csv(INPUT).sort_values("loop").reset_index(drop=True)
    raw = data["raw_dqs_mean"].to_numpy(dtype=float)
    truth = data["true_quality_mean"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(raw, truth, deg=1)
    calibrated = slope * raw + intercept
    calibrated_sd = abs(slope) * data["raw_dqs_sd"].to_numpy(dtype=float)

    raw_mae = float(np.mean(np.abs(raw - truth)))
    calibrated_mae = float(np.mean(np.abs(calibrated - truth)))
    raw_rmse = float(np.sqrt(np.mean((raw - truth) ** 2)))
    calibrated_rmse = float(np.sqrt(np.mean((calibrated - truth) ** 2)))

    result = data[
        ["loop", "true_quality_mean", "true_quality_sd", "raw_dqs_mean", "raw_dqs_sd"]
    ].copy()
    result["calibrated_dqs_mean"] = calibrated
    result["calibrated_dqs_sd"] = calibrated_sd
    result["calibration_slope"] = slope
    result["calibration_intercept"] = intercept
    result["raw_mae_across_mean_checkpoints"] = raw_mae
    result["calibrated_mae_across_mean_checkpoints"] = calibrated_mae
    result["raw_rmse_across_mean_checkpoints"] = raw_rmse
    result["calibrated_rmse_across_mean_checkpoints"] = calibrated_rmse
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FONT, "Liberation Serif", "DejaVu Serif"],
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(9.6, 5.0), dpi=220)
    loops = data["loop"].to_numpy()
    ax.errorbar(
        loops,
        truth,
        yerr=data["true_quality_sd"],
        color=BLUE,
        marker="o",
        linewidth=2.5,
        markersize=5.5,
        capsize=3,
        label="Known true entry quality",
    )
    ax.errorbar(
        loops,
        raw,
        yerr=data["raw_dqs_sd"],
        color=RED,
        marker="s",
        linestyle="--",
        linewidth=1.8,
        markersize=4.8,
        capsize=3,
        alpha=0.72,
        label=f"Raw entry DQS (MAE {raw_mae:.3f})",
    )
    ax.errorbar(
        loops,
        calibrated,
        yerr=calibrated_sd,
        color=GREEN,
        marker="D",
        linestyle="-.",
        linewidth=2.3,
        markersize=5.0,
        capsize=3,
        label=f"Affine-calibrated DQS (MAE {calibrated_mae:.3f})",
    )
    ax.set_title("VinDr: Entry-DQS Calibration", loc="left", fontweight="bold")
    ax.text(
        1,
        1.02,
        f"In-sample: calibrated = {slope:.3f} x raw {intercept:+.3f} (9 mean checkpoints)",
        transform=ax.transAxes,
        ha="right",
        color=GREY,
        fontsize=9.5,
    )
    ax.set_xlim(-0.15, 8.55)
    ax.set_ylim(0.77, 0.97)
    ax.set_xticks(loops)
    ax.set_xlabel("Cleaning loop")
    ax.set_ylabel("Entry-level quality score")
    ax.grid(axis="y", color=GRID, linewidth=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(pad=0.9)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(
        {
            "figure": str(OUTPUT_PNG),
            "slope": slope,
            "intercept": intercept,
            "raw_mae": raw_mae,
            "calibrated_mae": calibrated_mae,
            "raw_rmse": raw_rmse,
            "calibrated_rmse": calibrated_rmse,
        }
    )


if __name__ == "__main__":
    main()
