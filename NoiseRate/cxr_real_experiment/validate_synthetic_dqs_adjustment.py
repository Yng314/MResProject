#!/usr/bin/env python3
"""Validate raw and coverage-adjusted DQS against known synthetic label truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cleanlab.dataset import overall_label_health_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "cxr_real_experiment"
    / "evaluation_followup_20260713"
    / "synthetic_dqs_validation"
)

SCENARIO_LABELS = {
    "oracle_correction": "Correct known errors",
    "oracle_removal": "Remove known errors",
    "random_removal": "Remove random entries",
    "harmful_correction": "Corrupt known-clean entries",
}
CURRENT_TRUTH_LABEL = "Accuracy among labels that remain"
ORIGINAL_TRUTH_LABEL = "Fraction of original labels that remain and are correct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--noise-rate", type=float, default=0.12)
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--strong-class-sep", type=float, default=1.5)
    parser.add_argument("--weak-class-sep", type=float, default=0.35)
    return parser.parse_args()


def inject_exact_symmetric_noise(
    labels: np.ndarray, noise_rate: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    noisy = labels.copy()
    n_flip = int(round(noise_rate * len(labels)))
    flip_indices = rng.choice(len(labels), size=n_flip, replace=False)
    noisy[flip_indices] = 1 - noisy[flip_indices]
    return noisy, np.sort(flip_indices)


def make_oof_problem(
    *,
    seed: int,
    n_samples: int,
    noise_rate: float,
    class_sep: float,
    cv_folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    features, true_labels = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=2,
        weights=[0.7, 0.3],
        flip_y=0,
        class_sep=class_sep,
        random_state=seed,
    )
    rng = np.random.default_rng(seed + 10_000)
    noisy_labels, error_indices = inject_exact_symmetric_noise(true_labels, noise_rate, rng)
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed + 20_000)
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    probabilities = cross_val_predict(
        model,
        features,
        noisy_labels,
        cv=splitter,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    oof_auroc_vs_truth = float(roc_auc_score(true_labels, probabilities))
    return true_labels, noisy_labels, probabilities, error_indices, oof_auroc_vs_truth


def metric_row(
    *,
    seed: int,
    evidence_quality: str,
    scenario: str,
    loop: int,
    true_labels: np.ndarray,
    current_labels: np.ndarray,
    valid_mask: np.ndarray,
    probabilities: np.ndarray,
    oof_auroc_vs_truth: float,
) -> dict[str, float | int | str]:
    covered_labels = current_labels[valid_mask].astype(int)
    covered_probs = probabilities[valid_mask]
    pred_probs = np.column_stack([1.0 - covered_probs, covered_probs])
    raw_dqs = float(
        overall_label_health_score(
            labels=covered_labels,
            pred_probs=pred_probs,
            verbose=False,
        )
    )
    n_original = len(true_labels)
    n_covered = int(valid_mask.sum())
    coverage = n_covered / n_original
    correct_mask = valid_mask & (current_labels == true_labels)
    true_current_accuracy = float(np.mean(current_labels[valid_mask] == true_labels[valid_mask]))
    true_healthy_mass = float(correct_mask.sum() / n_original)
    adjusted_dqs = raw_dqs * coverage
    estimated_issue_count = int(round((1.0 - raw_dqs) * n_covered))
    return {
        "seed": seed,
        "evidence_quality": evidence_quality,
        "scenario": scenario,
        "loop": loop,
        "n_original": n_original,
        "n_covered": n_covered,
        "coverage": coverage,
        "true_current_accuracy": true_current_accuracy,
        "true_healthy_mass": true_healthy_mass,
        "raw_dqs": raw_dqs,
        "adjusted_dqs": adjusted_dqs,
        "estimated_issue_count": estimated_issue_count,
        "raw_abs_error": abs(raw_dqs - true_current_accuracy),
        "adjusted_abs_error": abs(adjusted_dqs - true_healthy_mass),
        "oof_auroc_vs_truth": oof_auroc_vs_truth,
    }


def build_scenario_state(
    *,
    scenario: str,
    loop: int,
    loops: int,
    true_labels: np.ndarray,
    noisy_labels: np.ndarray,
    error_order: np.ndarray,
    clean_order: np.ndarray,
    random_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current_labels = noisy_labels.copy()
    valid_mask = np.ones(len(noisy_labels), dtype=bool)
    n_action = int(round(loop / loops * len(error_order)))

    if scenario == "oracle_correction":
        indices = error_order[:n_action]
        current_labels[indices] = true_labels[indices]
    elif scenario == "oracle_removal":
        valid_mask[error_order[:n_action]] = False
    elif scenario == "random_removal":
        valid_mask[random_order[:n_action]] = False
    elif scenario == "harmful_correction":
        indices = clean_order[:n_action]
        current_labels[indices] = 1 - current_labels[indices]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return current_labels, valid_mask


def run_trials(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    evidence_settings = {
        "informative_oof": args.strong_class_sep,
        "weak_oof": args.weak_class_sep,
    }
    for evidence_offset, (evidence_quality, class_sep) in enumerate(evidence_settings.items()):
        for seed_index in range(args.n_seeds):
            seed = 13 + seed_index
            problem_seed = seed + evidence_offset * 100_000
            true_labels, noisy_labels, probabilities, error_indices, oof_auc = make_oof_problem(
                seed=problem_seed,
                n_samples=args.n_samples,
                noise_rate=args.noise_rate,
                class_sep=class_sep,
                cv_folds=args.cv_folds,
            )
            rng = np.random.default_rng(problem_seed + 30_000)
            error_order = rng.permutation(error_indices)
            clean_indices = np.flatnonzero(noisy_labels == true_labels)
            clean_order = rng.permutation(clean_indices)
            random_order = rng.permutation(len(true_labels))

            for scenario in SCENARIO_LABELS:
                for loop in range(args.loops + 1):
                    current_labels, valid_mask = build_scenario_state(
                        scenario=scenario,
                        loop=loop,
                        loops=args.loops,
                        true_labels=true_labels,
                        noisy_labels=noisy_labels,
                        error_order=error_order,
                        clean_order=clean_order,
                        random_order=random_order,
                    )
                    rows.append(
                        metric_row(
                            seed=seed,
                            evidence_quality=evidence_quality,
                            scenario=scenario,
                            loop=loop,
                            true_labels=true_labels,
                            current_labels=current_labels,
                            valid_mask=valid_mask,
                            probabilities=probabilities,
                            oof_auroc_vs_truth=oof_auc,
                        )
                    )
    return pd.DataFrame(rows)


def summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "coverage",
        "true_current_accuracy",
        "true_healthy_mass",
        "raw_dqs",
        "adjusted_dqs",
        "estimated_issue_count",
        "raw_abs_error",
        "adjusted_abs_error",
        "oof_auroc_vs_truth",
    ]
    grouped = trials.groupby(["evidence_quality", "scenario", "loop"], as_index=False)
    summary = grouped[metric_columns].agg(["mean", "std"])
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary


def validate_invariants(trials: pd.DataFrame, args: argparse.Namespace) -> dict[str, float | int]:
    tolerance = 1e-12
    if trials.isna().any().any():
        raise AssertionError("Synthetic trial output contains missing values")
    identity_error = np.max(
        np.abs(trials["adjusted_dqs"] - trials["raw_dqs"] * trials["coverage"])
    )
    if identity_error > tolerance:
        raise AssertionError(f"Adjusted DQS identity failed: {identity_error}")

    for (quality, seed), group in trials.groupby(["evidence_quality", "seed"]):
        correction = group[group["scenario"] == "oracle_correction"].sort_values("loop")
        oracle_remove = group[group["scenario"] == "oracle_removal"].sort_values("loop")
        random_remove = group[group["scenario"] == "random_removal"].sort_values("loop")
        harmful = group[group["scenario"] == "harmful_correction"].sort_values("loop")
        if abs(float(correction.iloc[-1]["true_current_accuracy"]) - 1.0) > tolerance:
            raise AssertionError(f"Oracle correction did not reach perfect truth: {quality}/{seed}")
        if abs(float(oracle_remove.iloc[-1]["true_current_accuracy"]) - 1.0) > tolerance:
            raise AssertionError(f"Oracle removal did not remove all errors: {quality}/{seed}")
        if not np.allclose(
            oracle_remove["true_healthy_mass"],
            oracle_remove.iloc[0]["true_healthy_mass"],
            atol=tolerance,
        ):
            raise AssertionError(f"Oracle removal changed true healthy mass: {quality}/{seed}")
        if not np.allclose(
            oracle_remove["coverage"], random_remove["coverage"], atol=tolerance
        ):
            raise AssertionError(f"Removal controls have unmatched coverage: {quality}/{seed}")
        if float(harmful.iloc[-1]["true_current_accuracy"]) >= float(
            harmful.iloc[0]["true_current_accuracy"]
        ):
            raise AssertionError(f"Harmful correction did not reduce accuracy: {quality}/{seed}")

    auc_by_quality = trials.groupby("evidence_quality")["oof_auroc_vs_truth"].mean()
    if auc_by_quality["informative_oof"] <= auc_by_quality["weak_oof"]:
        raise AssertionError("Informative OOF evidence is not stronger than weak OOF evidence")
    return {
        "rows": int(len(trials)),
        "seeds": int(args.n_seeds),
        "loops_including_baseline": int(args.loops + 1),
        "max_adjusted_identity_error": float(identity_error),
        "informative_oof_auroc_mean": float(auc_by_quality["informative_oof"]),
        "weak_oof_auroc_mean": float(auc_by_quality["weak_oof"]),
    }


def plot_trajectories(summary: pd.DataFrame, output_path: Path) -> None:
    data = summary[summary["evidence_quality"] == "informative_oof"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    all_values = data[
        [
            "true_current_accuracy_mean",
            "true_healthy_mass_mean",
            "raw_dqs_mean",
            "adjusted_dqs_mean",
        ]
    ].to_numpy()
    y_min = max(0.0, float(np.nanmin(all_values)) - 0.04)
    y_max = min(1.01, float(np.nanmax(all_values)) + 0.025)

    for axis, scenario in zip(axes_flat, SCENARIO_LABELS):
        subset = data[data["scenario"] == scenario].sort_values("loop")
        x = subset["loop"].to_numpy()
        axis.errorbar(
            x,
            subset["raw_dqs_mean"],
            yerr=subset["raw_dqs_std"].fillna(0),
            color="#2F6FB3",
            marker="o",
            linewidth=2.2,
            capsize=4,
            label="Raw DQS",
        )
        axis.plot(
            x,
            subset["true_current_accuracy_mean"],
            color="#1F2430",
            linestyle="--",
            marker="s",
            linewidth=2.0,
            label=CURRENT_TRUTH_LABEL,
        )
        axis.errorbar(
            x,
            subset["adjusted_dqs_mean"],
            yerr=subset["adjusted_dqs_std"].fillna(0),
            color="#C7332F",
            marker="o",
            linewidth=2.2,
            capsize=4,
            label="Coverage-adjusted DQS",
        )
        axis.plot(
            x,
            subset["true_healthy_mass_mean"],
            color="#657084",
            linestyle=":",
            marker="s",
            linewidth=2.0,
            label=ORIGINAL_TRUTH_LABEL,
        )
        axis.set_title(SCENARIO_LABELS[scenario], fontsize=13, fontweight="bold")
        axis.set_ylim(y_min, y_max)
        axis.set_xticks(x)
        axis.grid(axis="y", color="#E4E8EF", linewidth=0.8)

    for axis in axes[-1, :]:
        axis.set_xlabel("Cleaning loop")
    for axis in axes[:, 0]:
        axis.set_ylabel("Score")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    handles_by_label = dict(zip(labels, handles, strict=True))
    legend_order = [
        CURRENT_TRUTH_LABEL,
        "Raw DQS",
        ORIGINAL_TRUTH_LABEL,
        "Coverage-adjusted DQS",
    ]
    fig.legend(
        [handles_by_label[label] for label in legend_order],
        legend_order,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=2,
        frameon=False,
        fontsize=13,
        handlelength=2.8,
        columnspacing=2.2,
        labelspacing=0.8,
    )
    fig.suptitle(
        "Known-ground-truth validation of raw and coverage-adjusted DQS",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.012,
        "Mean +/- SD over synthetic seeds; all panels use the same y-axis and frozen informative OOF evidence.",
        ha="center",
        fontsize=9.5,
        color="#657084",
    )
    fig.tight_layout(rect=(0, 0.19, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_evidence_sensitivity(trials: pd.DataFrame, output_path: Path) -> None:
    errors = (
        trials.groupby(["evidence_quality", "scenario"], as_index=False)[
            ["raw_abs_error", "adjusted_abs_error"]
        ]
        .mean()
        .melt(
            id_vars=["evidence_quality", "scenario"],
            var_name="metric",
            value_name="mean_absolute_error",
        )
    )
    labels = list(SCENARIO_LABELS)
    x = np.arange(len(labels), dtype=float)
    width = 0.2
    series = [
        ("informative_oof", "raw_abs_error", "Informative OOF: raw", "#2F6FB3"),
        (
            "informative_oof",
            "adjusted_abs_error",
            "Informative OOF: coverage-adjusted",
            "#C7332F",
        ),
        ("weak_oof", "raw_abs_error", "Weak OOF: raw", "#8FB4D8"),
        (
            "weak_oof",
            "adjusted_abs_error",
            "Weak OOF: coverage-adjusted",
            "#E49A96",
        ),
    ]
    fig, axis = plt.subplots(figsize=(13, 5.6))
    for index, (quality, metric, label, color) in enumerate(series):
        values = []
        for scenario in labels:
            row = errors[
                (errors["evidence_quality"] == quality)
                & (errors["metric"] == metric)
                & (errors["scenario"] == scenario)
            ]
            values.append(float(row.iloc[0]["mean_absolute_error"]))
        offset = (index - 1.5) * width
        axis.bar(x + offset, values, width=width, label=label, color=color)
    axis.set_xticks(x, [SCENARIO_LABELS[item] for item in labels])
    axis.set_ylabel("Mean absolute error against the matching ground-truth target")
    overall_mae = errors.groupby("evidence_quality")["mean_absolute_error"].mean()
    axis.set_title(
        "DQS error depends on OOF evidence quality\n"
        f"Average MAE across scenarios: informative {overall_mae['informative_oof']:.3f}; "
        f"weak {overall_mae['weak_oof']:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    axis.grid(axis="y", color="#E4E8EF", linewidth=0.8)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not 0 < args.noise_rate < 0.5:
        raise ValueError("noise-rate must be between 0 and 0.5")
    if args.loops < 1 or args.n_seeds < 2:
        raise ValueError("loops must be >= 1 and n-seeds must be >= 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trials = run_trials(args)
    checks = validate_invariants(trials, args)
    summary = summarize_trials(trials)
    trials.to_csv(args.output_dir / "synthetic_dqs_trials.csv", index=False)
    summary.to_csv(args.output_dir / "synthetic_dqs_summary.csv", index=False)
    plot_trajectories(summary, args.output_dir / "synthetic_dqs_known_truth_trajectories.png")
    plot_evidence_sensitivity(
        trials, args.output_dir / "synthetic_dqs_oof_quality_sensitivity.png"
    )

    error_summary = (
        trials.groupby(["evidence_quality", "scenario"], as_index=False)[
            ["raw_abs_error", "adjusted_abs_error", "oof_auroc_vs_truth"]
        ]
        .mean()
    )
    error_summary.to_csv(args.output_dir / "synthetic_dqs_error_summary.csv", index=False)
    payload = {
        "status": "passed",
        "design": {
            "n_samples": args.n_samples,
            "n_seeds": args.n_seeds,
            "noise_rate": args.noise_rate,
            "loops": args.loops,
            "cv_folds": args.cv_folds,
            "scenarios": list(SCENARIO_LABELS),
            "evidence_quality": ["informative_oof", "weak_oof"],
            "oof_policy": "Logistic-regression cross-validated probabilities trained on baseline noisy labels and frozen across actions",
        },
        "checks": checks,
        "interpretation": {
            "raw_target": CURRENT_TRUTH_LABEL,
            "adjusted_target": ORIGINAL_TRUTH_LABEL,
            "limitation": "Both scores depend on the quality of the OOF evidence and are not direct ground-truth estimators on real data",
        },
    }
    (args.output_dir / "synthetic_dqs_validation.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
