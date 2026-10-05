#!/usr/bin/env python3
"""Compare baseline, simple removal, and fixed LLM refinement across seeds."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


DEFAULT_ROOT = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_mobilenetv3_noes50_clean_3seed/20260707_123320"
)


def prediction_path(root: Path, seed: int, method: str, loop: int = 0) -> Path:
    seed_root = root / f"seed_{seed}"
    if method == "baseline":
        run_root = seed_root / "baseline_no_clean"
    elif method == "remove":
        run_root = (
            seed_root
            / "sample20_remove_loop"
            / "remove_only"
            / f"loop_{loop:02d}"
            / "train_eval"
        )
    elif method == "refine":
        run_root = (
            seed_root
            / "sample20_llm_refine_fixed_xrv_s13"
            / f"loop_{loop:02d}"
            / "train_eval"
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return run_root / "test_study_predictions.csv"


def summary_path(root: Path, seed: int, method: str, loop: int = 0) -> Path:
    return prediction_path(root, seed, method, loop).with_name("test_study_auroc_summary.csv")


def stage_keys(loops: list[int]) -> list[str]:
    return ["baseline"] + [f"remove_L{loop}" for loop in loops] + [
        f"refine_L{loop}" for loop in loops
    ]


def split_stage(stage: str) -> tuple[str, int]:
    if stage == "baseline":
        return "baseline", 0
    method, loop = stage.split("_L", 1)
    return method, int(loop)


def parse_pipe_matrix(values: pd.Series, dtype: type) -> np.ndarray:
    rows = [np.fromstring(str(value), sep="|", dtype=dtype) for value in values]
    return np.vstack(rows)


def load_prediction_arrays(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    labels = parse_pipe_matrix(frame["binary_labels_for_metric"], float)
    valid = parse_pipe_matrix(frame["valid_label_mask"], int).astype(bool)
    probs = parse_pipe_matrix(frame["pred_probs"], float)
    return frame, labels, valid, probs


def weighted_summary_score(path: Path, label_names: set[str] | None = None) -> float:
    frame = pd.read_csv(path)
    if label_names is not None:
        frame = frame.loc[frame["label_name"].isin(label_names)]
    return float(np.average(frame["study_auroc_binary"], weights=frame["study_valid_count"]))


def exact_signflip_p(delta: np.ndarray) -> float:
    observed = abs(float(np.mean(delta)))
    permuted = [
        abs(float(np.mean(np.asarray(signs) * delta)))
        for signs in product([-1, 1], repeat=len(delta))
    ]
    return float(np.mean(np.asarray(permuted) >= observed - 1e-15))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    n_values = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (n_values - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def collect_observed_scores(
    root: Path,
    seeds: list[int],
    loops: list[int],
    scopes: dict[str, set[str] | None],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for stage in stage_keys(loops):
            method, loop = split_stage(stage)
            for scope, labels in scopes.items():
                rows.append(
                    {
                        "seed": seed,
                        "scope": scope,
                        "method": method,
                        "loop": loop,
                        "weighted_auroc": weighted_summary_score(
                            summary_path(root, seed, method, loop), labels
                        ),
                    }
                )
    scores = pd.DataFrame(rows)
    baseline = scores.loc[scores["method"] == "baseline", ["seed", "scope", "weighted_auroc"]]
    baseline = baseline.rename(columns={"weighted_auroc": "baseline_auroc"})
    scores = scores.merge(baseline, on=["seed", "scope"], how="left")
    scores["delta_vs_baseline"] = scores["weighted_auroc"] - scores["baseline_auroc"]
    return scores.sort_values(["scope", "seed", "method", "loop"])


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (scope, method, loop), frame in scores.groupby(["scope", "method", "loop"], sort=True):
        rows.append(
            {
                "scope": scope,
                "method": method,
                "loop": int(loop),
                "n_seeds": int(len(frame)),
                "mean_weighted_auroc": float(frame["weighted_auroc"].mean()),
                "sd_weighted_auroc": float(frame["weighted_auroc"].std(ddof=1)),
                "mean_delta_vs_baseline": float(frame["delta_vs_baseline"].mean()),
                "sd_delta_vs_baseline": float(frame["delta_vs_baseline"].std(ddof=1)),
                "positive_delta_seeds": int((frame["delta_vs_baseline"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["scope", "method", "loop"])


def paired_seed_statistics(scores: pd.DataFrame, loops: list[int]) -> pd.DataFrame:
    comparisons = [
        ("remove_vs_baseline", "remove", "baseline"),
        ("refine_vs_baseline", "refine", "baseline"),
        ("refine_vs_remove", "refine", "remove"),
    ]
    rows: list[dict[str, float | int | str]] = []
    for scope, scope_frame in scores.groupby("scope"):
        pivot = scope_frame.pivot(index="seed", columns=["method", "loop"], values="weighted_auroc")
        for comparison, left, right in comparisons:
            family_rows: list[dict[str, float | int | str]] = []
            for loop in loops:
                left_key = (left, loop if left != "baseline" else 0)
                right_key = (right, loop if right != "baseline" else 0)
                delta = (pivot[left_key] - pivot[right_key]).to_numpy(dtype=float)
                mean = float(delta.mean())
                sd = float(delta.std(ddof=1))
                sem = sd / np.sqrt(len(delta))
                t_critical = float(stats.t.ppf(0.975, len(delta) - 1))
                family_rows.append(
                    {
                        "scope": scope,
                        "comparison": comparison,
                        "loop": loop,
                        "n_seeds": len(delta),
                        "mean_delta": mean,
                        "sd_delta": sd,
                        "t_ci_low_2p5": mean - t_critical * sem,
                        "t_ci_high_97p5": mean + t_critical * sem,
                        "paired_effect_dz": mean / sd if sd else np.nan,
                        "positive_delta_seeds": int((delta > 0).sum()),
                        "exact_signflip_p_two_sided": exact_signflip_p(delta),
                        "paired_t_p_two_sided": float(stats.ttest_1samp(delta, 0).pvalue),
                    }
                )
            exact_adjusted = holm_adjust(
                np.asarray([row["exact_signflip_p_two_sided"] for row in family_rows])
            )
            t_adjusted = holm_adjust(
                np.asarray([row["paired_t_p_two_sided"] for row in family_rows])
            )
            for row, exact_p, t_p in zip(family_rows, exact_adjusted, t_adjusted):
                row["holm5_exact_p"] = float(exact_p)
                row["holm5_paired_t_p"] = float(t_p)
            rows.extend(family_rows)
    return pd.DataFrame(rows).sort_values(["scope", "comparison", "loop"])


def bootstrap_seed(
    root_text: str,
    seed: int,
    loops: list[int],
    scope_indices: dict[str, list[int]],
    n_bootstrap: int,
    bootstrap_seed_value: int,
) -> tuple[int, np.ndarray]:
    root = Path(root_text)
    stages = stage_keys(loops)
    reference_frame, labels, valid, baseline_probs = load_prediction_arrays(
        prediction_path(root, seed, "baseline")
    )
    probabilities = [baseline_probs]
    key_columns = ["study_id", "subject_id", "n_images_in_study", "binary_labels_for_metric", "valid_label_mask"]
    for stage in stages[1:]:
        method, loop = split_stage(stage)
        frame, other_labels, other_valid, probs = load_prediction_arrays(
            prediction_path(root, seed, method, loop)
        )
        if not frame[key_columns].equals(reference_frame[key_columns]):
            raise ValueError(f"Prediction alignment mismatch for seed={seed}, stage={stage}")
        if not np.array_equal(other_labels, labels, equal_nan=True) or not np.array_equal(other_valid, valid):
            raise ValueError(f"Label alignment mismatch for seed={seed}, stage={stage}")
        probabilities.append(probs)

    probs_array = np.stack(probabilities)
    scope_names = list(scope_indices)
    output = np.full((len(scope_names), len(stages), n_bootstrap), np.nan, dtype=float)
    rng = np.random.default_rng(bootstrap_seed_value)
    n_studies = len(reference_frame)

    for boot_index in range(n_bootstrap):
        indices = rng.integers(0, n_studies, size=n_studies)
        label_scores = np.full((len(stages), labels.shape[1]), np.nan, dtype=float)
        label_weights = np.zeros(labels.shape[1], dtype=int)
        for label_index in range(labels.shape[1]):
            mask = valid[indices, label_index]
            y_true = labels[indices, label_index][mask]
            if y_true.size == 0 or np.unique(y_true).size < 2:
                continue
            label_weights[label_index] = int(mask.sum())
            for stage_index in range(len(stages)):
                y_score = probs_array[stage_index, indices, label_index][mask]
                label_scores[stage_index, label_index] = roc_auc_score(y_true, y_score)

        finite_labels = np.isfinite(label_scores[0]) & (label_weights > 0)
        for scope_index, scope in enumerate(scope_names):
            selected = np.asarray(scope_indices[scope], dtype=int)
            selected = selected[finite_labels[selected]]
            if selected.size == 0:
                continue
            output[scope_index, :, boot_index] = np.average(
                label_scores[:, selected], axis=1, weights=label_weights[selected]
            )
    return seed, output


def bootstrap_statistics(
    root: Path,
    seeds: list[int],
    loops: list[int],
    scope_indices: dict[str, list[int]],
    observed_scores: pd.DataFrame,
    n_bootstrap: int,
    bootstrap_seed_value: int,
    workers: int,
) -> pd.DataFrame:
    args = [
        (str(root), seed, loops, scope_indices, n_bootstrap, bootstrap_seed_value)
        for seed in seeds
    ]
    if workers == 1:
        results = [bootstrap_seed(*item) for item in args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(bootstrap_seed, *item) for item in args]
            results = [future.result() for future in futures]
    results.sort(key=lambda item: seeds.index(item[0]))
    boot = np.stack([item[1] for item in results])

    stages = stage_keys(loops)
    stage_index = {stage: index for index, stage in enumerate(stages)}
    comparisons = [
        ("remove_vs_baseline", "remove", "baseline"),
        ("refine_vs_baseline", "refine", "baseline"),
        ("refine_vs_remove", "refine", "remove"),
    ]
    seed_rng = np.random.default_rng(bootstrap_seed_value + 1)
    seed_draws = seed_rng.integers(0, len(seeds), size=(n_bootstrap, len(seeds)))
    rows: list[dict[str, float | int | str]] = []

    for scope_pos, scope in enumerate(scope_indices):
        scope_scores = observed_scores.loc[observed_scores["scope"] == scope]
        pivot = scope_scores.pivot(index="seed", columns=["method", "loop"], values="weighted_auroc")
        for comparison, left, right in comparisons:
            family_rows: list[dict[str, float | int | str]] = []
            for loop in loops:
                left_stage = "baseline" if left == "baseline" else f"{left}_L{loop}"
                right_stage = "baseline" if right == "baseline" else f"{right}_L{loop}"
                delta_by_seed = (
                    boot[:, scope_pos, stage_index[left_stage], :]
                    - boot[:, scope_pos, stage_index[right_stage], :]
                )
                conditional = np.nanmean(delta_by_seed, axis=0)
                hierarchical = np.asarray(
                    [
                        np.nanmean(delta_by_seed[seed_draws[index], index])
                        for index in range(n_bootstrap)
                    ]
                )
                observed_left = pivot[(left, loop if left != "baseline" else 0)]
                observed_right = pivot[(right, loop if right != "baseline" else 0)]
                observed_delta = float((observed_left - observed_right).mean())
                for uncertainty, values in [
                    ("conditional_shared_study", conditional),
                    ("hierarchical_seed_and_study", hierarchical),
                ]:
                    values = values[np.isfinite(values)]
                    probability_le_zero = float((np.sum(values <= 0) + 1) / (len(values) + 1))
                    probability_ge_zero = float((np.sum(values >= 0) + 1) / (len(values) + 1))
                    family_rows.append(
                        {
                            "scope": scope,
                            "comparison": comparison,
                            "loop": loop,
                            "uncertainty": uncertainty,
                            "n_seeds": len(seeds),
                            "n_bootstrap": n_bootstrap,
                            "valid_bootstrap": len(values),
                            "observed_mean_delta": observed_delta,
                            "bootstrap_mean_delta": float(np.mean(values)),
                            "ci_low_2p5": float(np.percentile(values, 2.5)),
                            "ci_high_97p5": float(np.percentile(values, 97.5)),
                            "bootstrap_probability_delta_le_zero": probability_le_zero,
                            "bootstrap_p_two_sided": min(
                                1.0, 2 * min(probability_le_zero, probability_ge_zero)
                            ),
                        }
                    )
            family = pd.DataFrame(family_rows)
            for uncertainty, index in family.groupby("uncertainty").groups.items():
                adjusted = holm_adjust(family.loc[index, "bootstrap_p_two_sided"].to_numpy())
                family.loc[index, "holm5_bootstrap_p"] = adjusted
            rows.extend(family.to_dict("records"))
    return pd.DataFrame(rows).sort_values(["scope", "comparison", "uncertainty", "loop"])


def validate_inputs(root: Path, seeds: list[int], loops: list[int]) -> dict[str, int | float | list]:
    reference, _, _, _ = load_prediction_arrays(prediction_path(root, seeds[0], "baseline"))
    key_columns = ["study_id", "subject_id", "n_images_in_study", "binary_labels_for_metric", "valid_label_mask"]
    failures: list[dict[str, int | str]] = []
    n_files = 0
    for seed in seeds:
        for stage in stage_keys(loops):
            method, loop = split_stage(stage)
            frame = pd.read_csv(prediction_path(root, seed, method, loop), usecols=key_columns)
            n_files += 1
            if not frame.equals(reference[key_columns]):
                failures.append({"seed": seed, "stage": stage, "rows": len(frame)})
    return {
        "prediction_files_checked": n_files,
        "studies_per_file": len(reference),
        "duplicate_study_ids": int(reference["study_id"].duplicated().sum()),
        "alignment_failure_count": len(failures),
        "alignment_failures": failures,
    }


def coverage_summary(root: Path, seeds: list[int], loops: list[int]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    initial_samples = 237717
    for loop in loops:
        removed: list[int] = []
        retained: list[int] = []
        for seed in seeds:
            frame = pd.read_csv(
                root / f"seed_{seed}" / "sample20_remove_loop" / "remove_only" / "loop_metrics.csv"
            )
            row = frame.loc[frame["loop_id"].astype(int) == loop].iloc[0]
            removed.append(int(row["cumulative_removed_samples"]))
            retained.append(int(row["train_samples"]))
        refine = pd.read_csv(
            root
            / f"seed_{seeds[0]}"
            / "sample20_llm_refine_fixed_xrv_s13"
            / "loop_metrics.csv"
        )
        refine_row = refine.loc[refine["loop_id"].astype(int) == loop].iloc[0]
        rows.append(
            {
                "loop": loop,
                "initial_train_samples": initial_samples,
                "remove_train_samples_mean": float(np.mean(retained)),
                "remove_train_samples_sd": float(np.std(retained, ddof=1)),
                "remove_retention_fraction": float(np.mean(retained) / initial_samples),
                "removed_samples_mean": float(np.mean(removed)),
                "removed_samples_sd": float(np.std(removed, ddof=1)),
                "refine_train_samples": int(refine_row["train_samples"]),
                "refine_sample_retention_fraction": float(refine_row["train_samples"] / initial_samples),
                "refine_cumulative_relabel_entries": int(refine_row["applied_relabel_entries"]),
                "refine_cumulative_mask_entries": int(refine_row["applied_mask_entries"]),
            }
        )
    return pd.DataFrame(rows)


def per_label_summary(root: Path, seeds: list[int], loops: list[int]) -> pd.DataFrame:
    support = pd.read_csv(summary_path(root, seeds[0], "baseline")).set_index("label_name")
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        summaries = {"baseline": pd.read_csv(summary_path(root, seed, "baseline")).set_index("label_name")}
        for method in ["remove", "refine"]:
            for loop in loops:
                summaries[f"{method}_L{loop}"] = pd.read_csv(
                    summary_path(root, seed, method, loop)
                ).set_index("label_name")
        for label in support.index:
            baseline = float(summaries["baseline"].loc[label, "study_auroc_binary"])
            for loop in loops:
                remove = float(summaries[f"remove_L{loop}"].loc[label, "study_auroc_binary"])
                refine = float(summaries[f"refine_L{loop}"].loc[label, "study_auroc_binary"])
                rows.extend(
                    [
                        {"seed": seed, "label": label, "comparison": "remove_vs_baseline", "loop": loop, "delta": remove - baseline},
                        {"seed": seed, "label": label, "comparison": "refine_vs_baseline", "loop": loop, "delta": refine - baseline},
                        {"seed": seed, "label": label, "comparison": "refine_vs_remove", "loop": loop, "delta": refine - remove},
                    ]
                )
    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(["label", "comparison", "loop"])
        .agg(
            mean_delta=("delta", "mean"),
            sd_delta=("delta", "std"),
            positive_delta_seeds=("delta", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )
    support_columns = support[
        ["study_valid_count", "study_positive_count", "study_negative_count"]
    ].reset_index().rename(columns={"label_name": "label"})
    support_columns["minority_support"] = support_columns[
        ["study_positive_count", "study_negative_count"]
    ].min(axis=1)
    return grouped.merge(support_columns, on="label", how="left").sort_values(
        ["comparison", "loop", "mean_delta"], ascending=[True, True, False]
    )


def posthoc_comparisons(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for scope, frame in scores.groupby("scope"):
        pivot = frame.pivot(index="seed", columns=["method", "loop"], values="weighted_auroc")
        comparisons = {
            "refine_L5_vs_remove_L1": pivot[("refine", 5)] - pivot[("remove", 1)],
            "refine_L5_vs_best_remove_within_seed": pivot[("refine", 5)]
            - pivot.loc[:, "remove"].max(axis=1),
        }
        for comparison, series in comparisons.items():
            delta = series.to_numpy(dtype=float)
            mean = float(delta.mean())
            sd = float(delta.std(ddof=1))
            half_width = float(stats.t.ppf(0.975, len(delta) - 1) * sd / np.sqrt(len(delta)))
            rows.append(
                {
                    "scope": scope,
                    "comparison": comparison,
                    "mean_delta": mean,
                    "sd_delta": sd,
                    "t_ci_low_2p5": mean - half_width,
                    "t_ci_high_97p5": mean + half_width,
                    "positive_delta_seeds": int((delta > 0).sum()),
                    "exact_signflip_p_two_sided": exact_signflip_p(delta),
                }
            )
    return pd.DataFrame(rows)


def write_plots(
    score_summary: pd.DataFrame,
    observed_scores: pd.DataFrame,
    bootstrap: pd.DataFrame,
    coverage: pd.DataFrame,
    out_dir: Path,
) -> None:
    full = score_summary.loc[score_summary["scope"] == "all_labels"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    baseline = float(full.loc[full["method"] == "baseline", "mean_weighted_auroc"].iloc[0])
    ax.axhline(baseline, color="0.35", linestyle="--", label=f"baseline ({baseline:.3f})")
    for method, color, label in [("remove", "#2b6cb0", "simple remove"), ("refine", "#c53030", "LLM refine")]:
        frame = full.loc[full["method"] == method].sort_values("loop")
        ax.errorbar(
            frame["loop"], frame["mean_weighted_auroc"], yerr=frame["sd_weighted_auroc"],
            marker="o", capsize=4, linewidth=2, color=color, label=label,
        )
    ax.set_xticks(range(1, 6), [f"Loop {loop}" for loop in range(1, 6)])
    ax.set_ylabel("Study-weighted AUROC")
    ax.set_title("Five-seed cleaning trajectories")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "five_seed_method_curve.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), dpi=180, sharey=True)
    for axis, scope, title in zip(axes, ["all_labels", "support_ge_5"], ["All 12 labels", "9 labels with minority support >= 5"]):
        frame = bootstrap.loc[
            (bootstrap["scope"] == scope)
            & (bootstrap["comparison"] == "refine_vs_remove")
            & (bootstrap["uncertainty"] == "hierarchical_seed_and_study")
        ].sort_values("loop")
        values = frame["observed_mean_delta"].to_numpy()
        lower = values - frame["ci_low_2p5"].to_numpy()
        upper = frame["ci_high_97p5"].to_numpy() - values
        axis.axhline(0, color="0.35", linewidth=1)
        axis.errorbar(frame["loop"], values, yerr=[lower, upper], marker="o", capsize=4, linewidth=2)
        axis.set_xticks(range(1, 6))
        axis.set_xlabel("Cleaning loop")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("LLM refine - simple remove AUROC")
    fig.tight_layout()
    fig.savefig(out_dir / "refine_vs_remove_hierarchical_ci.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    frame = observed_scores.loc[
        (observed_scores["scope"] == "all_labels") & (observed_scores["method"] != "baseline")
    ].copy()
    remove = frame.loc[frame["method"] == "remove"].groupby("loop")["weighted_auroc"].mean()
    refine = frame.loc[frame["method"] == "refine"].groupby("loop")["weighted_auroc"].mean()
    ax.plot(100 * coverage["remove_retention_fraction"], remove, marker="o", linewidth=2, label="simple remove")
    ax.plot(100 * coverage["refine_sample_retention_fraction"], refine, marker="o", linewidth=2, label="LLM refine")
    for loop, x, y in zip(coverage["loop"], 100 * coverage["remove_retention_fraction"], remove):
        ax.annotate(f"L{loop}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    for loop, x, y in zip(coverage["loop"], 100 * coverage["refine_sample_retention_fraction"], refine):
        ax.annotate(f"L{loop}", (x, y), xytext=(4, -10), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Training-sample retention (%)")
    ax.set_ylabel("Mean study-weighted AUROC")
    ax.set_title("Performance and sample retention")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "performance_vs_sample_retention.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 42, 97, 123])
    parser.add_argument("--loops", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260711)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = sorted(args.seeds)
    loops = sorted(args.loops)

    qa = validate_inputs(args.root, seeds, loops)
    if qa["alignment_failure_count"]:
        raise SystemExit(f"Input alignment validation failed: {qa}")

    support = pd.read_csv(summary_path(args.root, seeds[0], "baseline"))
    support["minority_support"] = support[
        ["study_positive_count", "study_negative_count"]
    ].min(axis=1)
    stable_labels = set(support.loc[support["minority_support"] >= 5, "label_name"])
    scopes = {"all_labels": None, "support_ge_5": stable_labels}
    scope_indices = {
        "all_labels": support["label_index"].astype(int).tolist(),
        "support_ge_5": support.loc[support["label_name"].isin(stable_labels), "label_index"].astype(int).tolist(),
    }

    observed = collect_observed_scores(args.root, seeds, loops, scopes)
    score_summary = summarize_scores(observed)
    seed_stats = paired_seed_statistics(observed, loops)
    bootstrap = bootstrap_statistics(
        args.root,
        seeds,
        loops,
        scope_indices,
        observed,
        args.bootstrap_iters,
        args.bootstrap_seed,
        min(args.workers, len(seeds)),
    )
    coverage = coverage_summary(args.root, seeds, loops)
    labels = per_label_summary(args.root, seeds, loops)
    posthoc = posthoc_comparisons(observed)

    qa.update(
        {
            "seeds": seeds,
            "loops": loops,
            "bootstrap_iterations": args.bootstrap_iters,
            "bootstrap_seed": args.bootstrap_seed,
            "all_label_count": int(len(support)),
            "support_ge_5_label_count": int(len(stable_labels)),
            "support_ge_5_labels": sorted(stable_labels),
            "bootstrap_study_resamples_shared_across_seeds": True,
        }
    )
    (args.out_dir / "validation_metadata.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    support.to_csv(args.out_dir / "test_label_support.csv", index=False)
    observed.to_csv(args.out_dir / "per_seed_scores.csv", index=False)
    score_summary.to_csv(args.out_dir / "score_summary.csv", index=False)
    seed_stats.to_csv(args.out_dir / "seed_paired_statistics.csv", index=False)
    bootstrap.to_csv(args.out_dir / "paired_bootstrap_statistics.csv", index=False)
    coverage.to_csv(args.out_dir / "coverage_summary.csv", index=False)
    labels.to_csv(args.out_dir / "per_label_deltas.csv", index=False)
    posthoc.to_csv(args.out_dir / "posthoc_method_comparisons.csv", index=False)
    write_plots(score_summary, observed, bootstrap, coverage, args.out_dir)

    print(f"Wrote unified cleaning-method evaluation to {args.out_dir}")
    print(json.dumps(qa, indent=2))
    print("\nAll-label score summary:")
    print(score_summary.loc[score_summary["scope"] == "all_labels"].to_string(index=False))
    print("\nAll-label seed-paired statistics:")
    print(seed_stats.loc[seed_stats["scope"] == "all_labels"].to_string(index=False))
    print("\nBootstrap comparison rows:")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
