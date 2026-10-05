#!/usr/bin/env python3
"""Aggregate the two-seed VinDr OOF AUM-style screening experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vindr_known_gt_cl_benchmark import atomic_write_csv, atomic_write_text, sha256_file


EXPECTED_SEEDS = (13, 211)
EXPECTED_METHODS = (
    "oof_label_incompatibility",
    "active_label_cleaning",
    "oof_aum",
)


def aggregate(args: argparse.Namespace) -> None:
    root = Path(args.root)
    output = root / "aggregate"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite aggregate output: {output}")
    frames = []
    input_hashes = {}
    for seed in EXPECTED_SEEDS:
        seed_root = root / f"seed_{seed}"
        for marker in [
            seed_root / ".benchmark_verified",
            seed_root / ".blind_run_complete",
            seed_root / "private_evaluation" / ".evaluation_complete",
        ]:
            if not marker.is_file():
                raise FileNotFoundError(marker)
        metrics_path = seed_root / "private_evaluation" / "detector_metrics.csv"
        frame = pd.read_csv(metrics_path)
        if set(frame["method"]) != set(EXPECTED_METHODS) or set(frame["seed"]) != {seed}:
            raise RuntimeError(f"Seed {seed} metric grid is invalid")
        frames.append(frame)
        input_hashes[f"seed_{seed}"] = sha256_file(metrics_path)
    metrics = pd.concat(frames, ignore_index=True)

    means = metrics.groupby("method")[["auprc", "recall_at_error_count"]].agg(
        ["mean", "std"]
    )
    means.columns = ["_".join(column) for column in means.columns]
    means = means.reset_index()
    pivot_auprc = metrics.pivot(index="seed", columns="method", values="auprc")
    pivot_recall = metrics.pivot(index="seed", columns="method", values="recall_at_error_count")
    contrasts = pd.DataFrame(
        {
            "seed": list(EXPECTED_SEEDS),
            "aum_minus_oof_auprc": (
                pivot_auprc["oof_aum"] - pivot_auprc["oof_label_incompatibility"]
            ).reindex(EXPECTED_SEEDS).to_numpy(),
            "aum_minus_oof_recall_at_error_count": (
                pivot_recall["oof_aum"] - pivot_recall["oof_label_incompatibility"]
            ).reindex(EXPECTED_SEEDS).to_numpy(),
        }
    )

    # This is a predeclared screening gate, not an inferential significance test.
    expand = bool(
        (contrasts["aum_minus_oof_auprc"] > 0).all()
        and (contrasts["aum_minus_oof_recall_at_error_count"] >= 0).all()
        and contrasts["aum_minus_oof_recall_at_error_count"].mean() > 0
    )
    decision = "expand_to_six_seed_benchmark" if expand else "do_not_expand_without_method_revision"

    output.mkdir(parents=True)
    atomic_write_csv(metrics, output / "combined_detector_metrics.csv")
    atomic_write_csv(means, output / "method_summary.csv")
    atomic_write_csv(contrasts, output / "aum_vs_oof_seed_contrasts.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    colors = {
        "oof_label_incompatibility": "#2563eb",
        "active_label_cleaning": "#7c3aed",
        "oof_aum": "#dc2626",
    }
    for axis, metric, label in [
        (axes[0], "auprc", "Injected-error AUPRC"),
        (axes[1], "recall_at_error_count", "Recall at 3,600-entry budget"),
    ]:
        for method in EXPECTED_METHODS:
            selected = metrics[metrics["method"] == method].sort_values("seed")
            axis.plot(
                selected["seed"].astype(str),
                selected[metric],
                marker="o",
                linewidth=1.8,
                color=colors[method],
                label=method.replace("_", " "),
            )
        axis.set_xlabel("Training seed")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(output / "oof_aum_screen.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "protocol": "vindr_oof_aum_screen_v1",
        "seeds": list(EXPECTED_SEEDS),
        "screening_only": True,
        "screening_gate": {
            "requirements": [
                "AUM AUPRC exceeds OOF in both seeds",
                "AUM matched-budget recall is non-inferior in both seeds",
                "mean AUM matched-budget recall exceeds OOF",
            ],
            "decision": decision,
        },
        "input_metric_hashes": input_hashes,
        "combined_metrics_sha256": sha256_file(output / "combined_detector_metrics.csv"),
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(output / "aggregate_summary.json", json.dumps(summary, indent=2) + "\n")
    atomic_write_text(output / ".aggregate_complete", "complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    return parser


if __name__ == "__main__":
    aggregate(build_parser().parse_args())
