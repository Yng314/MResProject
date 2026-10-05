#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues

from plot_seed42_loop15_dqs_driver_exclusion_comparison import (
    RUN_ROOT,
    apply_cumulative_actions,
    load_train_state,
)


OUTPUT_ROOT = RUN_ROOT.parent.parent / "analysis"
DEFAULT_OUTPUT = OUTPUT_ROOT / "seed42_loop01_15_sample_dqs_frozen_vs_dynamic.png"
DEFAULT_DATA_OUTPUT = OUTPUT_ROOT / "seed42_loop01_15_sample_dqs_frozen_vs_dynamic.csv"


def sample_dqs_metrics(
    labels: np.ndarray,
    valid: np.ndarray,
    probs: np.ndarray,
    baseline_evaluable_samples: int,
) -> dict[str, float | int]:
    issue_mask = np.zeros_like(valid, dtype=bool)
    for label_index in range(labels.shape[1]):
        label_valid = valid[:, label_index]
        y_true = labels[label_valid, label_index].astype(int)
        y_prob = probs[label_valid, label_index]
        if y_true.size == 0 or np.unique(y_true).size < 2:
            continue
        issue_mask[label_valid, label_index] = find_label_issues(
            labels=y_true,
            pred_probs=np.column_stack([1.0 - y_prob, y_prob]),
            verbose=False,
        ).astype(bool)

    evaluable = valid.any(axis=1)
    n_evaluable = int(evaluable.sum())
    sample_issue_count = int(issue_mask.any(axis=1).sum())
    healthy_samples = n_evaluable - sample_issue_count
    raw_sample_dqs = healthy_samples / n_evaluable
    sample_coverage = n_evaluable / baseline_evaluable_samples
    return {
        "n_evaluable_samples": n_evaluable,
        "sample_issue_count": sample_issue_count,
        "healthy_samples": healthy_samples,
        "sample_coverage": sample_coverage,
        "raw_sample_dqs": raw_sample_dqs,
        "adjusted_sample_dqs": healthy_samples / baseline_evaluable_samples,
    }


def build_sample_dqs_table() -> pd.DataFrame:
    initial_path = RUN_ROOT / "loop_01" / "oof" / "train_cleanlab_sample_details.csv"
    initial_frame, initial_labels, initial_valid, initial_probs = load_train_state(initial_path)
    pool_ids = initial_frame["pool_row_id"].to_numpy(dtype=np.int64)
    if len(np.unique(pool_ids)) != len(pool_ids):
        raise ValueError("Initial OOF table contains duplicate pool_row_id values")
    pool_to_index = {int(pool_id): index for index, pool_id in enumerate(pool_ids)}
    baseline_evaluable_samples = int(initial_valid.any(axis=1).sum())

    states: dict[int, tuple[np.ndarray, np.ndarray]] = {
        0: (initial_labels, initial_valid)
    }
    for loop_id in range(1, 16):
        labels, valid, _, _ = apply_cumulative_actions(
            initial_labels,
            initial_valid,
            pool_to_index,
            RUN_ROOT / f"loop_{loop_id:02d}",
        )
        states[loop_id] = (labels, valid)

    rows: list[dict[str, float | int | str]] = []
    for loop_id, (labels, valid) in states.items():
        rows.append(
            {
                "seed": 42,
                "mode": "frozen_initial_oof_post_action",
                "state_after_loop": loop_id,
                "oof_loop": 1,
                **sample_dqs_metrics(
                    labels,
                    valid,
                    initial_probs,
                    baseline_evaluable_samples,
                ),
            }
        )

    for oof_loop in range(1, 16):
        dynamic_path = (
            RUN_ROOT
            / f"loop_{oof_loop:02d}"
            / "oof"
            / "train_cleanlab_sample_details.csv"
        )
        frame, labels_pre, valid_pre, probs = load_train_state(dynamic_path)
        if not np.array_equal(frame["pool_row_id"].to_numpy(dtype=np.int64), pool_ids):
            raise ValueError(f"Pool-row order differs in Loop {oof_loop} OOF table")

        prior_loop = oof_loop - 1
        expected_labels, expected_valid = states[prior_loop]
        if not np.array_equal(valid_pre, expected_valid):
            raise AssertionError(f"Loop {oof_loop} OOF valid mask is not pre-action state")
        if not np.array_equal(
            np.nan_to_num(labels_pre, nan=-9.0),
            np.nan_to_num(expected_labels, nan=-9.0),
        ):
            raise AssertionError(f"Loop {oof_loop} OOF labels are not pre-action state")

        rows.append(
            {
                "seed": 42,
                "mode": "strict_retrained_pre_action",
                "state_after_loop": prior_loop,
                "oof_loop": oof_loop,
                **sample_dqs_metrics(
                    labels_pre,
                    valid_pre,
                    probs,
                    baseline_evaluable_samples,
                ),
            }
        )

        labels_post, valid_post = states[oof_loop]
        rows.append(
            {
                "seed": 42,
                "mode": "same_oof_immediate_post_action",
                "state_after_loop": oof_loop,
                "oof_loop": oof_loop,
                **sample_dqs_metrics(
                    labels_post,
                    valid_post,
                    probs,
                    baseline_evaluable_samples,
                ),
            }
        )
        print(f"Finished Loop {oof_loop:02d} dynamic sample DQS", flush=True)

    result = pd.DataFrame(rows).sort_values(["mode", "state_after_loop"])
    identity = result["raw_sample_dqs"] * result["sample_coverage"]
    if not np.allclose(identity, result["adjusted_sample_dqs"], atol=1e-12, rtol=0.0):
        raise AssertionError("Adjusted sample DQS identity failed")
    return result.reset_index(drop=True)


def annotate_point(
    ax: plt.Axes,
    x: int,
    y: float,
    text: str,
    color: str,
    offset: tuple[int, int],
) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        ha="center",
        va="bottom" if offset[1] >= 0 else "top",
        color=color,
        fontsize=8.5,
        fontweight="semibold",
    )


def style_axis(ax: plt.Axes, y_limits: tuple[float, float]) -> None:
    ax.set_xlim(0, 15)
    ax.set_ylim(*y_limits)
    ax.set_xticks(range(0, 16))
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6B7280")
    ax.spines["bottom"].set_color("#6B7280")
    ax.tick_params(colors="#374151")
    ax.axvline(8.5, color="#A7AFBA", linewidth=1.1, linestyle=(0, (2, 3)))


def plot_sample_dqs(frame: pd.DataFrame, output: Path) -> None:
    ink = "#20242A"
    gray = "#6B7280"
    light_gray = "#9CA3AF"
    blue = "#2563A6"
    orange = "#D97706"

    frozen = frame.loc[frame["mode"].eq("frozen_initial_oof_post_action")]
    strict = frame.loc[frame["mode"].eq("strict_retrained_pre_action")]
    immediate = frame.loc[frame["mode"].eq("same_oof_immediate_post_action")]
    baseline = float(
        frozen.loc[frozen["state_after_loop"].eq(0), "adjusted_sample_dqs"].iloc[0]
    )

    fig, axes = plt.subplots(2, 1, figsize=(12.2, 9.2), sharex=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Sample-level DQS across iterative refinement loops",
        x=0.07,
        y=0.975,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=ink,
    )
    fig.text(
        0.07,
        0.944,
        "Seed 42, 12 labels. A sample is an issue if any remaining valid entry is flagged; "
        "adjusted DQS = healthy samples / baseline evaluable samples.",
        ha="left",
        fontsize=10.2,
        color=gray,
    )

    ax = axes[0]
    ax.plot(
        frozen["state_after_loop"],
        frozen["raw_sample_dqs"],
        color=blue,
        linewidth=2.4,
        marker="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.6,
        label="Raw sample DQS",
    )
    ax.plot(
        frozen["state_after_loop"],
        frozen["adjusted_sample_dqs"],
        color=orange,
        linewidth=2.4,
        marker="s",
        markersize=5.2,
        markerfacecolor="white",
        markeredgewidth=1.6,
        label="Coverage-adjusted sample DQS",
    )
    ax.axhline(
        baseline,
        color=gray,
        linewidth=1.4,
        linestyle=(0, (5, 4)),
        label=f"Loop 0 baseline ({baseline:.4f})",
    )
    style_axis(ax, (0.725, 0.84))
    ax.set_title(
        "A. Fixed evaluation ruler: Loop 1 OOF evidence",
        loc="left",
        fontsize=12.3,
        fontweight="semibold",
        color=ink,
        pad=10,
    )
    ax.set_ylabel("Sample DQS")
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=9.2)
    for loop_id in (0, 10, 15):
        row = frozen.loc[frozen["state_after_loop"].eq(loop_id)].iloc[0]
        annotate_point(
            ax,
            loop_id,
            float(row["adjusted_sample_dqs"]),
            f"{row['adjusted_sample_dqs']:.4f}",
            orange,
            (8 if loop_id == 0 else 0, -10 if loop_id == 0 else 8),
        )

    ax = axes[1]
    ax.plot(
        frozen["state_after_loop"],
        frozen["adjusted_sample_dqs"],
        color=orange,
        linewidth=2.4,
        marker="s",
        markersize=5.0,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label="Frozen initial OOF, post-action",
    )
    ax.plot(
        strict["state_after_loop"],
        strict["adjusted_sample_dqs"],
        color=blue,
        linewidth=2.2,
        marker="o",
        markersize=5.2,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label="Next-loop retrained OOF (strict)",
    )
    ax.plot(
        immediate["state_after_loop"],
        immediate["adjusted_sample_dqs"],
        color=light_gray,
        linewidth=1.9,
        linestyle=(0, (4, 3)),
        marker="^",
        markersize=4.8,
        markerfacecolor="white",
        markeredgewidth=1.3,
        label="Same-loop OOF, immediate diagnostic",
    )
    ax.axhline(baseline, color=gray, linewidth=1.4, linestyle=(0, (5, 4)))
    style_axis(ax, (0.725, 0.835))
    ax.set_title(
        "B. Coverage-adjusted sample DQS is robust to the OOF evidence choice",
        loc="left",
        fontsize=12.3,
        fontweight="semibold",
        color=ink,
        pad=10,
    )
    ax.set_xlabel("Completed refinement loop")
    ax.set_ylabel("Adjusted sample DQS")
    ax.legend(frameon=False, loc="upper left", fontsize=9.2)
    strict_last = strict.iloc[-1]
    annotate_point(
        ax,
        int(strict_last["state_after_loop"]),
        float(strict_last["adjusted_sample_dqs"]),
        f"{strict_last['adjusted_sample_dqs']:.4f}",
        blue,
        (0, 8),
    )
    immediate_last = immediate.iloc[-1]
    annotate_point(
        ax,
        int(immediate_last["state_after_loop"]),
        float(immediate_last["adjusted_sample_dqs"]),
        f"{immediate_last['adjusted_sample_dqs']:.4f}",
        light_gray,
        (0, -11),
    )
    ax.text(
        14.85,
        0.742,
        "Strict series ends at Loop 14:\nLoop 15 requires Loop 16 OOF.",
        ha="right",
        va="bottom",
        fontsize=8.7,
        color=gray,
    )

    for ax in axes:
        ax.text(
            8.62,
            ax.get_ylim()[0] + 0.003,
            "unseen-entry\nselection starts",
            color=gray,
            fontsize=8.2,
            va="bottom",
        )

    fig.text(
        0.07,
        0.018,
        "Sample coverage changes little because masking one disease-label entry usually leaves "
        "the image usable for other labels. Single-seed diagnostic; no uncertainty bars.",
        ha="left",
        fontsize=9.1,
        color=gray,
    )
    fig.tight_layout(rect=(0.055, 0.055, 0.99, 0.925), h_pad=2.4)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    frame = build_sample_dqs_table()
    DEFAULT_DATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DEFAULT_DATA_OUTPUT, index=False)
    plot_sample_dqs(frame, DEFAULT_OUTPUT)
    print(f"Wrote {DEFAULT_DATA_OUTPUT}")
    print(f"Wrote {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
