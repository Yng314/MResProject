#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path("/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment")
RUN_ROOT = BASE / "results_xrv_iterative_sample20_dual_loop/20260624_150533"
OUT_DIR = RUN_ROOT / "progress_plots"
BASELINE_WEIGHTED = 0.8329938609999568


def load_branch(branch: str) -> pd.DataFrame:
    path = RUN_ROOT / branch / "loop_metrics.csv"
    if not path.exists():
        return pd.DataFrame(columns=["loop_id", "test_study_weighted_auroc"])
    return pd.read_csv(path)


def build_plot_df() -> pd.DataFrame:
    rows = [
        {
            "step": "Baseline",
            "step_order": 0,
            "simple_remove": BASELINE_WEIGHTED,
            "llm_refine": BASELINE_WEIGHTED,
        }
    ]
    remove_df = load_branch("remove_only")
    llm_df = load_branch("llm_refine")
    for loop_id in [1, 2]:
        remove_row = remove_df[remove_df["loop_id"] == loop_id]
        llm_row = llm_df[llm_df["loop_id"] == loop_id]
        rows.append(
            {
                "step": f"Loop {loop_id}",
                "step_order": loop_id,
                "simple_remove": float(remove_row["test_study_weighted_auroc"].iloc[0])
                if not remove_row.empty
                else None,
                "llm_refine": float(llm_row["test_study_weighted_auroc"].iloc[0])
                if not llm_row.empty
                else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_plot_df()
    csv_path = OUT_DIR / "iterative_sample20_weighted_progress.csv"
    df.to_csv(csv_path, index=False)

    plt.rcParams.update(
        {
            "figure.figsize": (8.5, 5.0),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 11,
        }
    )

    fig, ax = plt.subplots()
    x = df["step_order"].tolist()
    labels = df["step"].tolist()
    ax.plot(
        x,
        df["simple_remove"],
        marker="o",
        linewidth=2.4,
        color="#155EEF",
        label="Simple remove",
    )
    ax.plot(
        x,
        df["llm_refine"],
        marker="o",
        linewidth=2.4,
        color="#D92D20",
        label="LLM refine",
    )

    for _, row in df.iterrows():
        for col, color, dy in [
            ("simple_remove", "#155EEF", 8),
            ("llm_refine", "#D92D20", -16),
        ]:
            if pd.isna(row[col]):
                continue
            ax.annotate(
                f"{row[col]:.4f}",
                (row["step_order"], row[col]),
                textcoords="offset points",
                xytext=(0, dy),
                ha="center",
                fontsize=10,
                color=color,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Study weighted AUROC")
    ax.set_title("Iterative Sample20 Progress")
    ax.legend(frameon=False)
    ax.set_ylim(0.828, 0.855)

    fig.tight_layout()
    png_path = OUT_DIR / "iterative_sample20_weighted_progress.png"
    fig.savefig(png_path, dpi=180)
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote PNG: {png_path}")


if __name__ == "__main__":
    main()
