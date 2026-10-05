#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cleanlab.dataset import overall_label_health_score

from evaluate_mobilenet_data_quality_5seed import (
    LABEL_NAMES,
    binary_from_raw,
    load_train_state,
)


RUN_ROOT = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_mobilenetv3_own_top20_binary_v5_refine_unseen_extension/"
    "20260726_seed42_to_loop15/seed_42/llm_refine"
)
QA_TABLE = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_mobilenetv3_own_top20_binary_v5_llm_refine_5seed/"
    "20260714_065539/evaluation_posthoc_4seed_8loop_excluding_seed7_20260723/"
    "quality_per_seed_stage.csv"
)
DEFAULT_OUTPUT = RUN_ROOT.parent.parent / "analysis" / "seed42_loop01_15_entry_dqs_12_vs_9_labels.png"
DEFAULT_DATA_OUTPUT = (
    RUN_ROOT.parent.parent / "analysis" / "seed42_loop01_15_entry_dqs_12_vs_9_labels.csv"
)
EXCLUDED_LABELS = frozenset({"Atelectasis", "Lung Opacity", "Pneumothorax"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot frozen-evidence entry DQS for seed42 Loops1-15."
    )
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--qa-table", type=Path, default=QA_TABLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-output", type=Path, default=DEFAULT_DATA_OUTPUT)
    return parser.parse_args()


def map_pool_rows(pool_to_index: dict[int, int], values: pd.Series) -> np.ndarray:
    ids = values.astype(int).tolist()
    missing = sorted(set(ids).difference(pool_to_index))
    if missing:
        raise ValueError(f"Action table contains {len(missing)} unknown pool_row_id values")
    return np.asarray([pool_to_index[value] for value in ids], dtype=int)


def apply_cumulative_actions(
    initial_labels: np.ndarray,
    initial_valid: np.ndarray,
    pool_to_index: dict[int, int],
    loop_root: Path,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    relabels = pd.read_csv(loop_root / "applied_relabel_entries.csv")
    masks = pd.read_csv(loop_root / "applied_mask_entries.csv")
    labels = initial_labels.copy()
    valid = initial_valid.copy()

    relabel_rows = map_pool_rows(pool_to_index, relabels["pool_row_id"])
    relabel_columns = relabels["label_index"].to_numpy(dtype=int)
    labels[relabel_rows, relabel_columns] = binary_from_raw(
        relabels["new_raw_label"].to_numpy(dtype=float)
    )
    valid[relabel_rows, relabel_columns] = True

    mask_rows = map_pool_rows(pool_to_index, masks["pool_row_id"])
    mask_columns = masks["label_index"].to_numpy(dtype=int)
    valid[mask_rows, mask_columns] = False
    labels[mask_rows, mask_columns] = np.nan

    return labels, valid, len(relabels), len(masks)


def dqs_metrics(
    labels: np.ndarray,
    valid: np.ndarray,
    probs: np.ndarray,
    included_indices: np.ndarray,
    baseline_valid_entries: int,
) -> dict[str, float | int]:
    subset_labels = labels[:, included_indices]
    subset_valid = valid[:, included_indices]
    subset_probs = probs[:, included_indices]
    flattened_labels = subset_labels[subset_valid].astype(int)
    flattened_probs = subset_probs[subset_valid]
    if np.unique(flattened_labels).size != 2:
        raise ValueError("Flattened DQS requires both binary classes")

    raw_dqs = float(
        overall_label_health_score(
            labels=flattened_labels,
            pred_probs=np.column_stack([1.0 - flattened_probs, flattened_probs]),
            verbose=False,
        )
    )
    current_valid_entries = int(subset_valid.sum())
    coverage = current_valid_entries / baseline_valid_entries
    return {
        "raw_entry_dqs": raw_dqs,
        "valid_entry_coverage": coverage,
        "adjusted_entry_dqs": raw_dqs * coverage,
        "current_valid_entries": current_valid_entries,
        "baseline_valid_entries": baseline_valid_entries,
        "estimated_healthy_entries": raw_dqs * current_valid_entries,
    }


def build_dqs_table(run_root: Path) -> pd.DataFrame:
    initial_path = run_root / "loop_01" / "oof" / "train_cleanlab_sample_details.csv"
    frame, initial_labels, initial_valid, initial_probs = load_train_state(initial_path)
    pool_ids = frame["pool_row_id"].to_numpy(dtype=np.int64)
    if len(np.unique(pool_ids)) != len(pool_ids):
        raise ValueError("Initial OOF table contains duplicate pool_row_id values")
    pool_to_index = {int(pool_id): index for index, pool_id in enumerate(pool_ids)}

    label_sets = {
        "all_12_labels": np.arange(len(LABEL_NAMES), dtype=int),
        "remaining_9_labels": np.asarray(
            [
                index
                for index, label_name in enumerate(LABEL_NAMES)
                if label_name not in EXCLUDED_LABELS
            ],
            dtype=int,
        ),
    }
    baseline_counts = {
        name: int(initial_valid[:, indices].sum()) for name, indices in label_sets.items()
    }

    rows: list[dict[str, float | int | str]] = []
    states: list[tuple[int, np.ndarray, np.ndarray, int, int]] = [
        (0, initial_labels, initial_valid, 0, 0)
    ]
    for loop_id in range(1, 16):
        loop_root = run_root / f"loop_{loop_id:02d}"
        labels, valid, relabel_count, mask_count = apply_cumulative_actions(
            initial_labels,
            initial_valid,
            pool_to_index,
            loop_root,
        )
        states.append((loop_id, labels, valid, relabel_count, mask_count))

    for label_set, indices in label_sets.items():
        included_names = [LABEL_NAMES[index] for index in indices]
        for loop_id, labels, valid, relabel_count, mask_count in states:
            metrics = dqs_metrics(
                labels,
                valid,
                initial_probs,
                indices,
                baseline_counts[label_set],
            )
            rows.append(
                {
                    "seed": 42,
                    "evidence_mode": "frozen_initial_oof_post_action",
                    "label_set": label_set,
                    "included_labels": "|".join(included_names),
                    "excluded_labels": "|".join(sorted(EXCLUDED_LABELS))
                    if label_set == "remaining_9_labels"
                    else "",
                    "loop": loop_id,
                    "cumulative_relabel_entries_all_labels": relabel_count,
                    "cumulative_mask_entries_all_labels": mask_count,
                    **metrics,
                }
            )

    result = pd.DataFrame(rows).sort_values(["label_set", "loop"]).reset_index(drop=True)
    if not np.allclose(
        result["adjusted_entry_dqs"],
        result["raw_entry_dqs"] * result["valid_entry_coverage"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("Adjusted DQS identity failed")
    for _, group in result.groupby("label_set", sort=False):
        if np.any(np.diff(group["valid_entry_coverage"].to_numpy(dtype=float)) > 1e-12):
            raise AssertionError("Valid-entry coverage increased across cumulative states")
    return result


def validate_against_formal_table(frame: pd.DataFrame, qa_path: Path) -> None:
    qa = pd.read_csv(qa_path)
    qa = qa.loc[
        qa["seed"].eq(42)
        & qa["evidence_mode"].eq("frozen_initial_oof_post_action")
        & (
            (qa["method"].eq("baseline") & qa["loop"].eq(0))
            | (qa["method"].eq("refine") & qa["loop"].between(1, 8))
        ),
        ["loop", "dqs_flattened", "valid_entry_coverage", "coverage_adjusted_dqs"],
    ].sort_values("loop").rename(
        columns={
            "dqs_flattened": "expected_raw_entry_dqs",
            "valid_entry_coverage": "expected_valid_entry_coverage",
            "coverage_adjusted_dqs": "expected_adjusted_entry_dqs",
        }
    )
    observed = frame.loc[
        frame["label_set"].eq("all_12_labels") & frame["loop"].between(0, 8),
        ["loop", "raw_entry_dqs", "valid_entry_coverage", "adjusted_entry_dqs"],
    ].sort_values("loop")
    merged = observed.merge(qa, on="loop", validate="one_to_one")
    checks = {
        "raw_entry_dqs": "expected_raw_entry_dqs",
        "valid_entry_coverage": "expected_valid_entry_coverage",
        "adjusted_entry_dqs": "expected_adjusted_entry_dqs",
    }
    for observed_column, expected_column in checks.items():
        if not np.allclose(
            merged[observed_column],
            merged[expected_column],
            rtol=0.0,
            atol=1e-12,
        ):
            delta = np.max(np.abs(merged[observed_column] - merged[expected_column]))
            raise AssertionError(
                f"Formal-table QA failed for {observed_column}; max abs delta={delta}"
            )


def add_panel(
    ax: plt.Axes,
    panel: pd.DataFrame,
    panel_title: str,
    shared_ylim: tuple[float, float],
) -> None:
    ink = "#20242A"
    gray = "#6B7280"
    grid = "#D9DEE5"
    blue = "#2563A6"
    orange = "#D97706"
    loops = panel["loop"].to_numpy(dtype=int)
    raw = panel["raw_entry_dqs"].to_numpy(dtype=float)
    adjusted = panel["adjusted_entry_dqs"].to_numpy(dtype=float)
    baseline = panel.loc[panel["loop"].eq(0), "raw_entry_dqs"].iloc[0]
    plotted = panel.loc[panel["loop"].between(1, 15)]

    ax.plot(
        plotted["loop"],
        plotted["raw_entry_dqs"],
        color=blue,
        linewidth=2.4,
        marker="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.8,
        label="Raw entry DQS",
        zorder=3,
    )
    ax.plot(
        plotted["loop"],
        plotted["adjusted_entry_dqs"],
        color=orange,
        linewidth=2.4,
        marker="s",
        markersize=5.2,
        markerfacecolor="white",
        markeredgewidth=1.8,
        label="Coverage-adjusted entry DQS",
        zorder=3,
    )
    ax.axhline(
        baseline,
        color=gray,
        linewidth=1.6,
        linestyle=(0, (5, 4)),
        label="Frozen-evidence baseline",
        zorder=1,
    )
    ax.axvline(8.5, color="#A7AFBA", linewidth=1.1, linestyle=(0, (2, 3)), zorder=1)
    ax.text(
        8.58,
        shared_ylim[1] - 0.001,
        "selection rule changes",
        color=gray,
        fontsize=8.3,
        va="top",
    )
    ax.text(
        15.25,
        baseline + 0.00035,
        f"baseline {baseline:.4f}",
        ha="right",
        va="bottom",
        color=gray,
        fontsize=8.3,
    )

    for loop_id in (1, 8, 15):
        row = panel.loc[panel["loop"].eq(loop_id)].iloc[0]
        ax.annotate(
            f"{row['raw_entry_dqs']:.4f}",
            (loop_id, row["raw_entry_dqs"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=blue,
            fontsize=8.2,
            fontweight="semibold" if loop_id in {8, 15} else "normal",
        )
        ax.annotate(
            f"{row['adjusted_entry_dqs']:.4f}",
            (loop_id, row["adjusted_entry_dqs"]),
            xytext=(0, -11),
            textcoords="offset points",
            ha="center",
            va="top",
            color=orange,
            fontsize=8.2,
            fontweight="semibold" if loop_id in {8, 15} else "normal",
        )

    final_coverage = float(panel.loc[panel["loop"].eq(15), "valid_entry_coverage"].iloc[0])
    ax.text(
        0.012,
        0.04,
        f"Loop15 valid-entry coverage: {final_coverage:.2%}",
        transform=ax.transAxes,
        color=gray,
        fontsize=8.4,
    )
    ax.set_title(panel_title, loc="left", fontsize=12, fontweight="semibold", color=ink)
    ax.set_ylabel("Entry DQS", fontsize=10, color=ink)
    ax.set_xticks(np.arange(1, 16))
    ax.set_xlim(0.6, 15.4)
    ax.set_ylim(*shared_ylim)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.tick_params(axis="both", colors=ink, labelsize=8.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#A7AFBA")
    ax.spines["bottom"].set_color("#A7AFBA")


def plot_dqs(frame: pd.DataFrame, output: Path) -> None:
    all_values = frame[["raw_entry_dqs", "adjusted_entry_dqs"]].to_numpy(dtype=float)
    lower = np.floor((float(np.nanmin(all_values)) - 0.002) * 200) / 200
    upper = np.ceil((float(np.nanmax(all_values)) + 0.002) * 200) / 200
    shared_ylim = (lower, upper)

    fig, axes = plt.subplots(2, 1, figsize=(13.2, 9.8), dpi=180, sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    all_labels = frame.loc[frame["label_set"].eq("all_12_labels")].sort_values("loop")
    remaining = frame.loc[frame["label_set"].eq("remaining_9_labels")].sort_values("loop")
    add_panel(axes[0], all_labels, "A. All 12 labels", shared_ylim)
    add_panel(
        axes[1],
        remaining,
        "B. Remaining 9 labels after post-hoc driver exclusion",
        shared_ylim,
    )
    axes[1].set_xlabel("Refinement loop", fontsize=10.5, color="#20242A", labelpad=8)

    fig.suptitle(
        "Seed 42 entry DQS across 15 refinement loops",
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
        "Frozen seed42 Loop1 OOF evidence | Excluded in panel B: Atelectasis, "
        "Lung Opacity, Pneumothorax | Focused shared y-axis",
        fontsize=9.4,
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
        "DQS is a confident-learning model-consistency diagnostic, not expert-verified "
        "label accuracy. Panel B is post-hoc and uses its own fixed 9-label baseline denominator.",
        fontsize=8.2,
        color="#6B7280",
    )
    fig.subplots_adjust(left=0.075, right=0.98, top=0.90, bottom=0.13, hspace=0.30)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frame = build_dqs_table(args.run_root)
    validate_against_formal_table(frame, args.qa_table)
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.data_output, index=False)
    plot_dqs(frame, args.output)
    print(f"Validated formal seed42 Loop0-8 DQS and wrote {args.data_output}")
    print(args.output)


if __name__ == "__main__":
    main()
