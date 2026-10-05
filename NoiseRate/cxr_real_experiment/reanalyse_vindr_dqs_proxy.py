#!/usr/bin/env python3
"""Reanalyse the VinDr controlled-noise benchmark using the thesis DQS definition.

This is a CPU-only post-hoc analysis. It reuses completed OOF predictions and
does not train a model or modify the archived experiment outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cleanlab.dataset import overall_label_health_score
from matplotlib import font_manager
from scipy.stats import spearmanr


SCENARIOS = {
    "clean": 1.0,
    "symmetric_entry_r10": 0.9,
    "symmetric_entry_r20": 0.8,
    "symmetric_entry_r30": 0.7,
}
SEEDS = (13, 42, 97, 123, 211, 307)
KEY_COLUMNS = ["image_id", "label_name"]
ACTION_LABELS = {
    "correction": "Correct known errors",
    "known_error_removal": "Remove known errors",
    "random_removal": "Pseudo-random removal",
    "harmful_relabelling": "Harmful relabelling",
}
ACTION_STYLES = {
    "correction": ("#2A6F97", "o", "-"),
    "known_error_removal": ("#D9822B", "s", "--"),
    "random_removal": ("#6F6F6F", "^", ":"),
    "harmful_relabelling": ("#8E5EA2", "D", "-."),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-pdf", type=Path, required=True)
    parser.add_argument("--figure-png", type=Path, required=True)
    return parser.parse_args()


def stable_order(frame: pd.DataFrame, salt: str) -> np.ndarray:
    tokens = (
        frame["image_id"].astype(str)
        + "|"
        + frame["label_name"].astype(str)
        + "|"
        + salt
    )
    hashes = tokens.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    return np.argsort(hashes.to_numpy(), kind="mergesort")


def load_state(root: Path, scenario: str, seed: int) -> pd.DataFrame:
    scenario_root = root / "scenarios" / scenario / f"seed_{seed}"
    evidence_path = scenario_root / "blind_run" / "entry_evidence.csv"
    reference_path = scenario_root / "prepared" / "private_reference.csv"
    if not evidence_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError(f"Missing evidence for {scenario}, seed {seed}")

    evidence = pd.read_csv(evidence_path)
    reference = pd.read_csv(reference_path)
    expected_evidence = set(KEY_COLUMNS + ["noisy_label", "oof_probability"])
    expected_reference = set(KEY_COLUMNS + ["clean_label", "noisy_label", "injected_error"])
    if not expected_evidence.issubset(evidence.columns):
        raise ValueError(f"Evidence columns missing for {scenario}, seed {seed}")
    if not expected_reference.issubset(reference.columns):
        raise ValueError(f"Reference columns missing for {scenario}, seed {seed}")
    if len(evidence) != 18_000 or len(reference) != 18_000:
        raise ValueError(f"Unexpected row count for {scenario}, seed {seed}")
    if evidence.duplicated(KEY_COLUMNS).any() or reference.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"Duplicate label-entry key for {scenario}, seed {seed}")

    merged = evidence.merge(
        reference,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
        suffixes=("_evidence", "_reference"),
    )
    if len(merged) != 18_000:
        raise ValueError(f"Incomplete evidence/reference join for {scenario}, seed {seed}")
    if merged[["clean_label", "noisy_label_evidence", "noisy_label_reference", "oof_probability"]].isna().any().any():
        raise ValueError(f"Missing required value for {scenario}, seed {seed}")
    if not np.array_equal(
        merged["noisy_label_evidence"].to_numpy(),
        merged["noisy_label_reference"].to_numpy(),
    ):
        raise ValueError(f"Noisy-label mismatch for {scenario}, seed {seed}")
    for column in ("clean_label", "noisy_label_evidence", "injected_error"):
        if not set(merged[column].astype(int).unique()).issubset({0, 1}):
            raise ValueError(f"Non-binary {column} for {scenario}, seed {seed}")
    if not merged["oof_probability"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError(f"Probability outside [0, 1] for {scenario}, seed {seed}")

    return merged.rename(columns={"noisy_label_evidence": "noisy_label"})


def dqs(labels: np.ndarray, probabilities: np.ndarray) -> float:
    pred_probs = np.column_stack([1.0 - probabilities, probabilities])
    return float(
        overall_label_health_score(
            labels=labels.astype(int),
            pred_probs=pred_probs,
            verbose=False,
        )
    )


def state_metrics(
    labels: np.ndarray,
    clean_labels: np.ndarray,
    probabilities: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    n_original = len(labels)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("No valid entries remain")
    current_labels = labels[valid]
    current_clean = clean_labels[valid]
    raw_dqs = dqs(current_labels, probabilities[valid])
    coverage = n_valid / n_original
    known_quality = float((current_labels == current_clean).mean())
    known_correct_mass = float(np.sum(valid & (labels == clean_labels)) / n_original)
    return {
        "n_original": n_original,
        "n_valid": n_valid,
        "coverage": coverage,
        "known_quality": known_quality,
        "known_correct_mass": known_correct_mass,
        "dqs": raw_dqs,
        "adjusted_dqs": raw_dqs * coverage,
    }


def build_anchor_results(root: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for scenario, true_quality in SCENARIOS.items():
        for seed in SEEDS:
            state = load_state(root, scenario, seed)
            labels = state["noisy_label"].to_numpy(dtype=int)
            probabilities = state["oof_probability"].to_numpy(dtype=float)
            score = dqs(labels, probabilities)
            observed_quality = float(
                (labels == state["clean_label"].to_numpy(dtype=int)).mean()
            )
            if not np.isclose(observed_quality, true_quality):
                raise ValueError(f"Known quality mismatch for {scenario}, seed {seed}")
            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "known_quality": observed_quality,
                    "dqs": score,
                    "signed_error": score - observed_quality,
                    "absolute_error": abs(score - observed_quality),
                }
            )
    return pd.DataFrame(rows)


def build_action_results(root: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed in SEEDS:
        state = load_state(root, "symmetric_entry_r20", seed)
        base_labels = state["noisy_label"].to_numpy(dtype=int)
        clean_labels = state["clean_label"].to_numpy(dtype=int)
        probabilities = state["oof_probability"].to_numpy(dtype=float)
        injected = state["injected_error"].to_numpy(dtype=int).astype(bool)
        if int(injected.sum()) != 3_600:
            raise ValueError(f"Expected 3,600 injected errors for seed {seed}")

        error_frame = state.loc[injected, KEY_COLUMNS].reset_index()
        error_indices = error_frame.iloc[stable_order(error_frame, "known_error")]["index"].to_numpy()
        all_frame = state[KEY_COLUMNS].reset_index()
        random_indices = all_frame.iloc[stable_order(all_frame, "random_removal")]["index"].to_numpy()[: len(error_indices)]
        clean_frame = state.loc[~injected, KEY_COLUMNS].reset_index()
        harmful_indices = clean_frame.iloc[stable_order(clean_frame, "harmful_relabelling")]["index"].to_numpy()[: len(error_indices)]

        for step in range(6):
            fraction = step / 5
            affected = int(round(len(error_indices) * fraction))
            for action in ACTION_LABELS:
                labels = base_labels.copy()
                valid = np.ones(len(labels), dtype=bool)
                if action == "correction":
                    selected = error_indices[:affected]
                    labels[selected] = clean_labels[selected]
                elif action == "known_error_removal":
                    selected = error_indices[:affected]
                    valid[selected] = False
                elif action == "random_removal":
                    selected = random_indices[:affected]
                    valid[selected] = False
                elif action == "harmful_relabelling":
                    selected = harmful_indices[:affected]
                    labels[selected] = 1 - clean_labels[selected]
                else:  # pragma: no cover
                    raise AssertionError(action)

                metrics = state_metrics(labels, clean_labels, probabilities, valid)
                rows.append(
                    {
                        "seed": seed,
                        "action": action,
                        "step": step,
                        "affected_fraction_of_initial_errors": fraction,
                        "n_affected": affected,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    metric_columns = [
        "known_quality",
        "known_correct_mass",
        "coverage",
        "dqs",
        "adjusted_dqs",
    ]
    available = [column for column in metric_columns if column in frame.columns]
    summary = frame.groupby(group_columns, as_index=False)[available].agg(["mean", "std"])
    summary.columns = [
        "_".join(value for value in column if value).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary.reset_index(drop=True)


def monotonic(values: np.ndarray, direction: str) -> bool:
    differences = np.diff(values)
    if direction == "increasing":
        return bool(np.all(differences >= -1e-12))
    if direction == "decreasing":
        return bool(np.all(differences <= 1e-12))
    raise ValueError(direction)


def build_qa(anchor: pd.DataFrame, actions: pd.DataFrame) -> dict[str, object]:
    anchor_checks = []
    for seed, frame in anchor.groupby("seed"):
        ordered = frame.sort_values("known_quality")
        anchor_checks.append(
            {
                "seed": int(seed),
                "strictly_increasing_dqs_with_known_quality": bool(
                    np.all(np.diff(ordered["dqs"].to_numpy()) > 0)
                ),
                "spearman_rho": float(
                    spearmanr(ordered["known_quality"], ordered["dqs"]).statistic
                ),
            }
        )

    action_rules = {
        "correction": ("dqs", "increasing"),
        "known_error_removal": ("dqs", "increasing"),
        "random_removal": ("adjusted_dqs", "decreasing"),
        "harmful_relabelling": ("dqs", "decreasing"),
    }
    action_checks = []
    for seed in SEEDS:
        for action, (metric, direction) in action_rules.items():
            values = (
                actions.loc[(actions["seed"] == seed) & (actions["action"] == action)]
                .sort_values("step")[metric]
                .to_numpy()
            )
            action_checks.append(
                {
                    "seed": seed,
                    "action": action,
                    "metric": metric,
                    "expected_direction": direction,
                    "direction_preserved": monotonic(values, direction),
                }
            )

    mean_anchor = anchor.groupby("known_quality", as_index=False)["dqs"].mean()
    return {
        "input_expected_scenarios": list(SCENARIOS),
        "input_expected_seeds": list(SEEDS),
        "action_design": {
            "starting_state": "symmetric_entry_r20",
            "oof_evidence": "held fixed at the starting state",
            "cumulative_steps": 5,
            "entries_added_per_step": 720,
            "ordering": (
                "ascending SHA-256 of image_id|label_name|action within each "
                "eligible pool; independent of OOF scores"
            ),
        },
        "anchor_rows": int(len(anchor)),
        "action_rows": int(len(actions)),
        "mean_anchor_spearman_rho": float(
            spearmanr(mean_anchor["known_quality"], mean_anchor["dqs"]).statistic
        ),
        "anchor_seed_checks": anchor_checks,
        "action_direction_checks": action_checks,
    }


def plot_results(anchor: pd.DataFrame, actions: pd.DataFrame, pdf: Path, png: Path) -> None:
    font_root = Path("/homes/yz3522/.local/share/fonts/msttcorefonts")
    for font_name in (
        "Times_New_Roman.ttf",
        "Times_New_Roman_Bold.ttf",
        "Times_New_Roman_Italic.ttf",
        "Times_New_Roman_Bold_Italic.ttf",
    ):
        font_path = font_root / font_name
        if font_path.is_file():
            font_manager.fontManager.addfont(font_path)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.65))
    identity_colour = "#4D4D4D"
    grid_colour = "#D9D9D9"

    anchor_summary = anchor.groupby("known_quality")["dqs"].agg(["mean", "std"]).reset_index()
    ax = axes[0]
    ax.plot([0.68, 1.01], [0.68, 1.01], linestyle="--", color=identity_colour, linewidth=1.0, label="Exact agreement")
    ax.errorbar(
        anchor_summary["known_quality"],
        anchor_summary["mean"],
        yerr=anchor_summary["std"],
        color="#2A6F97",
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.3,
        linewidth=1.6,
        capsize=3,
        label="VinDr DQS",
    )
    ax.set_title("(a) Known quality levels", loc="left")
    ax.set_xlabel("Known label quality")
    ax.set_ylabel("Raw DQS")
    ax.set_xlim(0.68, 1.01)
    ax.set_ylim(0.68, 1.01)
    ax.legend(frameon=False, loc="lower right")

    action_summary = (
        actions.groupby(["action", "step"], as_index=False)
        .agg(
            known_quality=("known_quality", "mean"),
            known_quality_sd=("known_quality", "std"),
            known_correct_mass=("known_correct_mass", "mean"),
            known_correct_mass_sd=("known_correct_mass", "std"),
            dqs=("dqs", "mean"),
            dqs_sd=("dqs", "std"),
            adjusted_dqs=("adjusted_dqs", "mean"),
            adjusted_dqs_sd=("adjusted_dqs", "std"),
        )
    )

    for panel_index, (x_column, y_column, x_label, y_label, title) in enumerate(
        [
            ("known_quality", "dqs", "Known quality among valid labels", "Raw DQS", "(b) Fixed-evidence actions"),
            ("known_correct_mass", "adjusted_dqs", "Valid and correct fraction of original labels", "Coverage-adjusted DQS", "(c) Coverage-aware comparison"),
        ],
        start=1,
    ):
        ax = axes[panel_index]
        ax.plot([0.58, 1.01], [0.58, 1.01], linestyle="--", color=identity_colour, linewidth=1.0)
        for action in ACTION_LABELS:
            frame = action_summary.loc[action_summary["action"] == action].sort_values("step")
            colour, marker, line_style = ACTION_STYLES[action]
            ax.plot(
                frame[x_column],
                frame[y_column],
                color=colour,
                linestyle=line_style,
                linewidth=1.4,
            )
            ax.errorbar(
                frame[x_column].iloc[1:],
                frame[y_column].iloc[1:],
                yerr=frame[f"{y_column}_sd"].iloc[1:],
                color=colour,
                marker=marker,
                markerfacecolor="white",
                markeredgewidth=1.1,
                linestyle="none",
                linewidth=0.8,
                capsize=2,
                markersize=5,
                label=ACTION_LABELS[action],
            )
        common = action_summary.loc[
            (action_summary["action"] == "correction")
            & (action_summary["step"] == 0)
        ].iloc[0]
        ax.errorbar(
            [common[x_column]],
            [common[y_column]],
            yerr=[common[f"{y_column}_sd"]],
            color="#222222",
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.2,
            linestyle="none",
            linewidth=0.8,
            capsize=2,
            markersize=5,
            zorder=5,
        )
        ax.set_title(title, loc="left")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(0.58, 1.01)
        ax.set_ylim(0.58, 1.01)

    for ax in axes:
        ax.grid(True, color=grid_colour, linewidth=0.6, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_aspect("equal", adjustable="box")

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.58, -0.035))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    anchor = build_anchor_results(args.experiment_root)
    actions = build_action_results(args.experiment_root)
    anchor_summary = anchor.groupby("known_quality", as_index=False).agg(
        dqs_mean=("dqs", "mean"),
        dqs_sd=("dqs", "std"),
        absolute_error_mean=("absolute_error", "mean"),
        absolute_error_sd=("absolute_error", "std"),
    )
    action_summary = (
        actions.groupby(["action", "step", "affected_fraction_of_initial_errors", "n_affected"], as_index=False)
        .agg(
            known_quality_mean=("known_quality", "mean"),
            known_quality_sd=("known_quality", "std"),
            known_correct_mass_mean=("known_correct_mass", "mean"),
            known_correct_mass_sd=("known_correct_mass", "std"),
            coverage_mean=("coverage", "mean"),
            coverage_sd=("coverage", "std"),
            dqs_mean=("dqs", "mean"),
            dqs_sd=("dqs", "std"),
            adjusted_dqs_mean=("adjusted_dqs", "mean"),
            adjusted_dqs_sd=("adjusted_dqs", "std"),
        )
    )
    qa = build_qa(anchor, actions)

    anchor.to_csv(args.output_dir / "vindr_dqs_quality_levels_by_seed.csv", index=False)
    anchor_summary.to_csv(args.output_dir / "vindr_dqs_quality_levels_summary.csv", index=False)
    actions.to_csv(args.output_dir / "vindr_dqs_action_trajectories_by_seed.csv", index=False)
    action_summary.to_csv(args.output_dir / "vindr_dqs_action_trajectories_summary.csv", index=False)
    (args.output_dir / "vindr_dqs_reanalysis_qa.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    plot_results(anchor, actions, args.figure_pdf, args.figure_png)

    print(anchor_summary.to_string(index=False))
    print()
    print(action_summary.loc[action_summary["step"].isin([0, 5])].to_string(index=False))
    print()
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
