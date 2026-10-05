#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path("/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment")
OUTPUT_DIR = BASE / "sample20_refine_comparison"

RUNS = [
    (
        "baseline",
        "Baseline",
        BASE / "results_appa_xrv12_linearhead_pipeline/slurm_236337/01_baseline",
    ),
    (
        "sample20_remove",
        "Sample20 Remove",
        BASE / "results_xrv_k4_pipeline_formal/20260615_131836/02_sample_array_job_250160/top20_issue_fraction",
    ),
    (
        "sample20_llm_seed13",
        "Refine s13",
        BASE / "results_xrv_sample20_llm_refined/slurm_252886/refined_train",
    ),
    (
        "sample20_llm_seed97",
        "Refine s97",
        BASE / "results_xrv_sample20_llm_refined/slurm_252887/refined_train",
    ),
    (
        "sample20_llm_seed123",
        "Refine s123",
        BASE / "results_xrv_sample20_llm_refined/slurm_252888/refined_train",
    ),
]


def study_weighted(study_csv: Path) -> float:
    with study_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    aucs = [float(row["study_auroc_binary"]) for row in rows]
    weights = [int(row["study_valid_count"]) for row in rows]
    return sum(a * w for a, w in zip(aucs, weights)) / sum(weights)


def load_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_id, display_name, root in RUNS:
        summary = pd.read_csv(root / "baseline_run_summary.csv").iloc[0]
        rows.append(
            {
                "run_id": run_id,
                "display_name": display_name,
                "image_macro": float(summary["test_image_macro_auroc_binary"]),
                "study_macro": float(summary["test_study_macro_auroc_binary"]),
                "study_weighted": float(study_weighted(root / "test_study_auroc_summary.csv")),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_rows()
    df.to_csv(OUTPUT_DIR / "sample20_refine_comparison.csv", index=False)

    x = list(range(len(df)))
    labels = df["display_name"].tolist()

    plt.rcParams.update(
        {
            "figure.figsize": (10, 5.5),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "font.size": 11,
        }
    )

    fig, ax = plt.subplots()
    ax.plot(x, df["study_weighted"], marker="o", linewidth=2.5, color="#155EEF", label="Study Weighted")
    ax.plot(x, df["study_macro"], marker="o", linewidth=2.0, color="#12B76A", alpha=0.9, label="Study Macro")

    for idx, row in df.iterrows():
        ax.annotate(
            f"{row['study_weighted']:.4f}",
            (idx, row["study_weighted"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            color="#155EEF",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("Sample20 Refinement: Before vs After LLM Refinement")
    ax.legend(frameon=False)
    ax.set_ylim(0.79, 0.86)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "sample20_refine_comparison.png"
    fig.savefig(out_path, dpi=180)
    print(f"Wrote CSV: {OUTPUT_DIR / 'sample20_refine_comparison.csv'}")
    print(f"Wrote PNG: {out_path}")


if __name__ == "__main__":
    main()
