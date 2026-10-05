#!/usr/bin/env python3
"""Plot VinDr true sample quality against raw sample-level DQS."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
MANIFEST = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "vindr_iterative_oracle_cleaning/20260806_combined_8seed_v1/"
    "combined_input_manifest.csv"
)
OUTPUT_DIR = HERE / "next_meeting_slides_20260807" / "extra_figures"
OUTPUT_PNG = OUTPUT_DIR / "vindr_sample_level_quality_8seed.png"
OUTPUT_CSV = OUTPUT_DIR / "vindr_sample_level_quality_8seed.csv"

FONT = "Times New Roman"
BLUE = "#0077BB"
RED = "#CC3311"
GREY = "#667085"
GRID = "#D9DEE7"


def load_loop_evidence(seed_root: Path, loop: int) -> tuple[pd.DataFrame, Path]:
    initialization = json.loads(
        (seed_root / "state" / "initialization_manifest.json").read_text()
    )
    private_reference = Path(initialization["source_prepared"]) / "private_reference.csv"
    if loop == 0:
        evidence = Path(initialization["source_blind_run"]) / "entry_evidence.csv"
    else:
        evidence = seed_root / f"loop_{loop:02d}" / "oof_after_action" / "entry_evidence.csv"
    return pd.read_csv(evidence), private_reference


def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    records: list[dict[str, float | int]] = []
    for row in manifest.itertuples(index=False):
        seed_root = Path(row.source_seed_root)
        reference: pd.DataFrame | None = None
        for loop in range(9):
            evidence, reference_path = load_loop_evidence(seed_root, loop)
            if reference is None:
                reference = pd.read_csv(reference_path)[
                    ["image_id", "label_name", "clean_label"]
                ]
            merged = evidence.merge(
                reference,
                on=["image_id", "label_name"],
                how="inner",
                validate="one_to_one",
            )
            if len(merged) != 18_000 or merged["image_id"].nunique() != 3_000:
                raise ValueError(f"Unexpected VinDr dimensions for seed {row.seed}, loop {loop}")

            # Compute both sample metrics explicitly to avoid averaging entry scores.
            merged["entry_correct"] = merged["noisy_label"].eq(merged["clean_label"])
            per_sample = merged.groupby("image_id", sort=False).agg(
                true_sample_clean=("entry_correct", "all"),
                estimated_sample_clean=("cl_issue", lambda values: not values.astype(bool).any()),
            )
            records.append(
                {
                    "seed": int(row.seed),
                    "loop": loop,
                    "true_sample_quality": float(per_sample["true_sample_clean"].mean()),
                    "raw_sample_dqs": float(per_sample["estimated_sample_clean"].mean()),
                }
            )

    trajectories = pd.DataFrame(records)
    summary = trajectories.groupby("loop", as_index=False).agg(
        true_sample_quality_mean=("true_sample_quality", "mean"),
        true_sample_quality_sd=("true_sample_quality", "std"),
        raw_sample_dqs_mean=("raw_sample_dqs", "mean"),
        raw_sample_dqs_sd=("raw_sample_dqs", "std"),
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FONT, "Liberation Serif", "DejaVu Serif"],
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
        }
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=220)
    loops = summary["loop"]
    ax.errorbar(
        loops,
        summary["true_sample_quality_mean"],
        yerr=summary["true_sample_quality_sd"],
        color=BLUE,
        marker="o",
        linewidth=2.4,
        markersize=5.5,
        capsize=3,
        label="Known true sample quality",
    )
    ax.errorbar(
        loops,
        summary["raw_sample_dqs_mean"],
        yerr=summary["raw_sample_dqs_sd"],
        color=RED,
        marker="s",
        linestyle="--",
        linewidth=2.3,
        markersize=5.2,
        capsize=3,
        label="Raw sample DQS",
    )
    final = summary.iloc[-1]
    ax.text(
        8.08,
        final["true_sample_quality_mean"] - 0.008,
        f"True quality {final['true_sample_quality_mean']:.3f}",
        color=BLUE,
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    ax.text(
        8.08,
        final["raw_sample_dqs_mean"] + 0.008,
        f"Raw sample DQS {final['raw_sample_dqs_mean']:.3f}",
        color=RED,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )
    ax.set_title("VinDr: Sample-Level Dataset Quality", loc="left", fontweight="bold")
    ax.text(
        1,
        1.02,
        "Mean +/- SD across 8 seeds",
        transform=ax.transAxes,
        ha="right",
        color=GREY,
        fontsize=9.5,
    )
    ax.set_xlim(-0.15, 9.15)
    low = min(summary["true_sample_quality_mean"].min(), summary["raw_sample_dqs_mean"].min())
    high = max(summary["true_sample_quality_mean"].max(), summary["raw_sample_dqs_mean"].max())
    padding = max(0.04, (high - low) * 0.16)
    ax.set_ylim(max(0, low - padding), min(1, high + padding))
    ax.set_xticks(range(9))
    ax.set_xlabel("Cleaning loop")
    ax.set_ylabel("Sample-level quality score")
    ax.grid(axis="y", color=GRID, linewidth=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.tight_layout(pad=0.9)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
