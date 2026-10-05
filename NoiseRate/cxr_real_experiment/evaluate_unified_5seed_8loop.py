#!/usr/bin/env python3
"""Run the frozen five-seed, eight-loop removal/refinement evaluation."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from evaluate_mobilenet_data_quality_5seed import (
    annotate_rows,
    load_train_state,
    quality_metrics,
)
from evaluate_own_top20_refinement_endpoints import (
    exact_signflip_p,
    hierarchical_bootstrap,
    holm_adjust,
    paired_t_p,
    summarize_metrics,
    summarize_per_label,
    validate_and_collect,
)
from evaluate_own_top20_refinement_quality import (
    add_sample_health,
    process_seed as process_refinement_quality,
    summarize_quality,
)


DEFAULT_BASE_ROOT = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_mobilenetv3_noes50_clean_3seed/20260707_123320"
)
DEFAULT_REFINE_ROOT = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_mobilenetv3_own_top20_binary_v5_llm_refine_5seed/20260714_065539"
)
DEFAULT_SEEDS = [7, 13, 42, 97, 123]
DEFAULT_LOOPS = list(range(1, 9))


@dataclass(frozen=True)
class Contrast:
    name: str
    left: tuple[str, int]
    right: tuple[str, int]
    tier: str


CONTRASTS = [
    Contrast("refine_L8_vs_remove_L8", ("refine", 8), ("remove", 8), "primary"),
    Contrast("refine_L8_vs_baseline", ("refine", 8), ("baseline", 0), "secondary"),
    Contrast("remove_L8_vs_baseline", ("remove", 8), ("baseline", 0), "secondary"),
    Contrast("refine_L5_vs_remove_L5", ("refine", 5), ("remove", 5), "secondary"),
    Contrast("refine_L5_vs_baseline", ("refine", 5), ("baseline", 0), "secondary"),
    Contrast("remove_L5_vs_baseline", ("remove", 5), ("baseline", 0), "secondary"),
    Contrast("refine_L8_vs_refine_L5", ("refine", 8), ("refine", 5), "secondary"),
    Contrast("remove_L8_vs_remove_L5", ("remove", 8), ("remove", 5), "secondary"),
]

PERFORMANCE_DIRECTIONS = {
    "study_weighted_auroc": 1,
    "study_macro_auroc": 1,
    "macro_average_precision": 1,
    "micro_brier": -1,
    "micro_nll": -1,
}

QUALITY_DIRECTIONS = {
    "dqs_flattened": 1,
    "coverage_adjusted_dqs": 1,
    "valid_entry_coverage": 1,
    "sample_issue_free_rate": 1,
    "coverage_adjusted_sample_health": 1,
    "sample_coverage": 1,
    "entry_issue_rate_valid_entries": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--refine-root", type=Path, default=DEFAULT_REFINE_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--loops", nargs="+", type=int, default=DEFAULT_LOOPS)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    parser.add_argument("--workers", type=int, default=5)
    return parser.parse_args()


def contrast_tuples() -> list[tuple[str, tuple[str, int], tuple[str, int]]]:
    return [(item.name, item.left, item.right) for item in CONTRASTS]


def paired_metric_statistics(
    frame: pd.DataFrame,
    directions: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric, direction in directions.items():
        pivot = frame.pivot(index="seed", columns=["method", "loop"], values=metric)
        family: list[dict[str, Any]] = []
        for contrast in CONTRASTS:
            raw_delta = (pivot[contrast.left] - pivot[contrast.right]).to_numpy(dtype=float)
            improvement = direction * raw_delta
            mean = float(improvement.mean())
            sd = float(improvement.std(ddof=1))
            half_width = float(
                stats.t.ppf(0.975, len(improvement) - 1)
                * sd
                / np.sqrt(len(improvement))
            )
            family.append(
                {
                    "metric": metric,
                    "comparison": contrast.name,
                    "tier": contrast.tier,
                    "left_stage": f"{contrast.left[0]}_L{contrast.left[1]}",
                    "right_stage": f"{contrast.right[0]}_L{contrast.right[1]}",
                    "direction": "higher_is_better" if direction == 1 else "lower_is_better",
                    "n_seeds": len(improvement),
                    "mean_raw_left_minus_right": float(raw_delta.mean()),
                    "mean_improvement": mean,
                    "sd_improvement": sd,
                    "t_ci_low_2p5": mean - half_width,
                    "t_ci_high_97p5": mean + half_width,
                    "paired_effect_dz": mean / sd if sd else np.nan,
                    "positive_improvement_seeds": int((improvement > 0).sum()),
                    "exact_signflip_p_two_sided": exact_signflip_p(improvement),
                    "paired_t_p_two_sided": paired_t_p(improvement),
                }
            )
        exact_adjusted = holm_adjust(
            np.asarray([row["exact_signflip_p_two_sided"] for row in family])
        )
        t_adjusted = holm_adjust(
            np.asarray([row["paired_t_p_two_sided"] for row in family])
        )
        for row, exact_p, t_p in zip(family, exact_adjusted, t_adjusted, strict=True):
            row["holm8_exact_p"] = float(exact_p)
            row["holm8_paired_t_p"] = float(t_p)
        rows.extend(family)
    return pd.DataFrame(rows)


def process_removal_quality(
    base_root: Path,
    seed: int,
    loops: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_root = base_root / f"seed_{seed}"
    remove_root = seed_root / "sample20_remove_loop" / "remove_only"
    initial_path = remove_root / "loop_01" / "oof" / "train_cleanlab_sample_details.csv"
    frame, initial_labels, initial_valid, initial_probs = load_train_state(initial_path)
    baseline_evaluable_samples = int(initial_valid.any(axis=1).sum())
    baseline_valid_entries = int(initial_valid.sum())
    pool_ids = frame["pool_row_id"].to_numpy(dtype=np.int64)

    metric_rows: list[dict[str, Any]] = []
    per_label_rows: list[dict[str, Any]] = []
    for loop in loops:
        removed_path = remove_root / f"loop_{loop:02d}" / "applied_removed_samples.csv"
        if not removed_path.is_file():
            raise FileNotFoundError(removed_path)
        removed = pd.read_csv(removed_path, usecols=["pool_row_id"])
        removed_ids = set(removed["pool_row_id"].astype(int))
        keep = np.fromiter(
            (int(pool_id) not in removed_ids for pool_id in pool_ids),
            dtype=bool,
            count=len(pool_ids),
        )
        metrics, labels, _ = quality_metrics(
            initial_labels[keep],
            initial_valid[keep],
            initial_probs[keep],
            baseline_valid_entries,
            baseline_evaluable_samples,
        )
        metrics = add_sample_health(metrics)
        row, annotated = annotate_rows(
            metrics,
            labels,
            seed,
            "frozen_initial_oof_post_action",
            "remove",
            loop,
            len(removed_ids),
        )
        row["removed_sample_count"] = len(removed_ids)
        row["retained_training_sample_fraction"] = float(keep.mean())
        metric_rows.append(row)
        per_label_rows.extend(annotated)
    return metric_rows, per_label_rows


def process_unified_quality(
    args: tuple[str, str, int, list[int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base_root_text, refine_root_text, seed, loops = args
    refinement_metrics, refinement_labels, actions = process_refinement_quality(
        base_root_text,
        refine_root_text,
        "own",
        seed,
        loops,
        False,
    )
    removal_metrics, removal_labels = process_removal_quality(
        Path(base_root_text),
        seed,
        loops,
    )
    return (
        refinement_metrics + removal_metrics,
        refinement_labels + removal_labels,
        actions,
    )


def build_quality_outputs(
    base_root: Path,
    refine_root: Path,
    seeds: list[int],
    loops: list[int],
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    worker_args = [(str(base_root), str(refine_root), seed, loops) for seed in seeds]
    if workers == 1:
        results = [process_unified_quality(item) for item in worker_args]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(seeds))) as executor:
            results = list(executor.map(process_unified_quality, worker_args))
    metrics = pd.DataFrame([row for result in results for row in result[0]])
    per_label = pd.DataFrame([row for result in results for row in result[1]])
    actions = pd.DataFrame([row for result in results for row in result[2]])
    return metrics, per_label, actions


def add_baseline_line(
    axis: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    label: str = "Baseline",
) -> None:
    baseline = summary.loc[
        summary["method"].eq("baseline") & summary["loop"].eq(0),
        f"{metric}_mean",
    ].iloc[0]
    axis.axhline(
        baseline,
        color="#4F5663",
        linestyle="--",
        linewidth=1.5,
        label=f"{label} ({baseline:.3f})",
    )


def plot_performance(summary: pd.DataFrame, out_dir: Path) -> None:
    specifications = [
        ("study_weighted_auroc", "Study-weighted AUROC", "higher"),
        ("macro_average_precision", "Macro average precision", "higher"),
        ("micro_brier", "Micro Brier score", "lower"),
        ("micro_nll", "Micro negative log-likelihood", "lower"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
    for axis, (metric, title, direction) in zip(axes.flat, specifications, strict=True):
        add_baseline_line(axis, summary, metric)
        for method, color, label in [
            ("remove", "#2B73B6", "Simple removal"),
            ("refine", "#C6372F", "LLM refinement"),
        ]:
            frame = summary.loc[summary["method"].eq(method)].sort_values("loop")
            axis.errorbar(
                frame["loop"],
                frame[f"{metric}_mean"],
                yerr=frame[f"{metric}_sd"],
                marker="o",
                linewidth=2.2,
                elinewidth=1.5,
                capsize=4,
                color=color,
                label=label,
            )
        axis.set_title(f"{title} ({direction} is better)")
        axis.set_xlabel("Cleaning loop")
        axis.set_xticks(DEFAULT_LOOPS)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    axes[0, 0].legend(frameon=False)
    n_seeds = int(summary["n_seeds"].max())
    fig.suptitle(f"{n_seeds}-seed held-out performance trajectories", fontsize=15)
    fig.savefig(out_dir / "performance_trajectories.png", dpi=200)
    plt.close(fig)


def plot_quality(summary: pd.DataFrame, out_dir: Path) -> None:
    frozen = summary.loc[
        summary["evidence_mode"].eq("frozen_initial_oof_post_action")
    ]
    specifications = [
        ("dqs_flattened", "Raw entry DQS"),
        ("coverage_adjusted_dqs", "Coverage-adjusted entry DQS"),
        ("sample_issue_free_rate", "Raw sample DQS"),
        ("coverage_adjusted_sample_health", "Coverage-adjusted sample DQS"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, specifications, strict=True):
        add_baseline_line(axis, frozen, metric)
        for method, color, label in [
            ("remove", "#2B73B6", "Simple removal"),
            ("refine", "#C6372F", "LLM refinement"),
        ]:
            frame = frozen.loc[frozen["method"].eq(method)].sort_values("loop")
            axis.errorbar(
                frame["loop"],
                frame[f"{metric}_mean"],
                yerr=frame[f"{metric}_sd"],
                marker="o",
                linewidth=2.2,
                elinewidth=1.5,
                capsize=4,
                color=color,
                label=label,
            )
        axis.set_title(title)
        axis.set_xticks(DEFAULT_LOOPS)
        axis.grid(axis="y", alpha=0.25)
    fig.supxlabel("Cleaning loop")
    for axis_pair, metric_pair in [
        (axes[0], ["dqs_flattened", "coverage_adjusted_dqs"]),
        (axes[1], ["sample_issue_free_rate", "coverage_adjusted_sample_health"]),
    ]:
        lower_values: list[float] = []
        upper_values: list[float] = []
        for metric in metric_pair:
            means = frozen[f"{metric}_mean"].to_numpy(dtype=float)
            sds = frozen[f"{metric}_sd"].fillna(0).to_numpy(dtype=float)
            lower_values.extend((means - sds).tolist())
            upper_values.extend((means + sds).tolist())
        low = min(lower_values)
        high = max(upper_values)
        margin = max(0.005, 0.05 * (high - low))
        for axis in axis_pair:
            axis.set_ylim(low - margin, high + margin)
    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    axes[0, 0].legend(frameon=False)
    n_seeds = int(summary["n_seeds"].max())
    fig.suptitle(
        f"{n_seeds}-seed frozen-evidence data-quality trajectories",
        fontsize=15,
    )
    fig.savefig(out_dir / "quality_trajectories.png", dpi=200)
    plt.close(fig)


def plot_coverage(summary: pd.DataFrame, out_dir: Path) -> None:
    frozen = summary.loc[
        summary["evidence_mode"].eq("frozen_initial_oof_post_action")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for axis, metric, title in [
        (axes[0], "valid_entry_coverage", "Valid-entry coverage"),
        (axes[1], "sample_coverage", "Sample coverage"),
    ]:
        add_baseline_line(axis, frozen, metric)
        for method, color, label in [
            ("remove", "#2B73B6", "Simple removal"),
            ("refine", "#C6372F", "LLM refinement"),
        ]:
            frame = frozen.loc[frozen["method"].eq(method)].sort_values("loop")
            axis.errorbar(
                frame["loop"],
                frame[f"{metric}_mean"],
                yerr=frame[f"{metric}_sd"],
                marker="o",
                linewidth=2.2,
                capsize=4,
                color=color,
                label=label,
            )
        axis.set_title(title)
        axis.set_xlabel("Cleaning loop")
        axis.set_ylabel("Fraction of baseline denominator")
        axis.set_xticks(DEFAULT_LOOPS)
        axis.set_ylim(0.65, 1.02)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    n_seeds = int(summary["n_seeds"].max())
    fig.suptitle(f"{n_seeds}-seed training-label coverage", fontsize=15)
    fig.savefig(out_dir / "coverage_trajectories.png", dpi=200)
    plt.close(fig)


def plot_bootstrap_forest(bootstrap: pd.DataFrame, out_dir: Path) -> None:
    frame = bootstrap.loc[
        bootstrap["uncertainty"].eq("hierarchical_seed_and_study")
    ].copy()
    order = [item.name for item in CONTRASTS]
    frame["comparison"] = pd.Categorical(frame["comparison"], categories=order, ordered=True)
    frame = frame.sort_values(["comparison", "scope"])
    positions = {name: index for index, name in enumerate(order[::-1])}
    offsets = {"all_labels": -0.12, "support_ge_5": 0.12}
    colors = {"all_labels": "#C6372F", "support_ge_5": "#2B73B6"}
    labels = {"all_labels": "All 12 labels", "support_ge_5": "Minority support >= 5"}

    fig, axis = plt.subplots(figsize=(10.5, 7), constrained_layout=True)
    axis.axvline(0, color="#4F5663", linewidth=1)
    for scope in ["all_labels", "support_ge_5"]:
        scoped = frame.loc[frame["scope"].eq(scope)]
        y = np.asarray([positions[str(name)] + offsets[scope] for name in scoped["comparison"]])
        values = scoped["observed_mean_delta"].to_numpy()
        lower = values - scoped["ci_low_2p5"].to_numpy()
        upper = scoped["ci_high_97p5"].to_numpy() - values
        axis.errorbar(
            values,
            y,
            xerr=[lower, upper],
            fmt="o",
            color=colors[scope],
            capsize=4,
            linewidth=1.8,
            label=labels[scope],
        )
    axis.set_yticks(
        np.arange(len(order)),
        order[::-1],
    )
    axis.set_xlabel("Mean study-weighted AUROC difference")
    n_seeds = int(bootstrap["n_seeds"].max())
    axis.set_title(f"{n_seeds}-seed hierarchical seed + study uncertainty")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(out_dir / "hierarchical_auroc_forest.png", dpi=200)
    plt.close(fig)


def build_checks(
    *,
    performance: pd.DataFrame,
    labels: pd.DataFrame,
    quality: pd.DataFrame,
    quality_labels: pd.DataFrame,
    qa: dict[str, Any],
    bootstrap: pd.DataFrame,
    base_root: Path,
    refine_root: Path,
    seeds: list[int],
    loops: list[int],
) -> pd.DataFrame:
    checks: list[dict[str, str]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(
            {"check": name, "status": "passed" if passed else "failed", "detail": detail}
        )

    expected_stage_rows = len(seeds) * (1 + 2 * len(loops))
    add(
        "performance_stage_rows",
        len(performance) == expected_stage_rows,
        f"expected={expected_stage_rows}, observed={len(performance)}",
    )
    add(
        "prediction_files_and_shared_test",
        qa["prediction_files_checked"] == expected_stage_rows
        and qa["studies_per_file"] == 605
        and qa["duplicate_study_ids"] == 0,
        json.dumps(qa, sort_keys=True),
    )
    add(
        "performance_per_label_rows",
        len(labels) == expected_stage_rows * 12,
        f"expected={expected_stage_rows * 12}, observed={len(labels)}",
    )
    add(
        "quality_stage_rows",
        len(quality) == expected_stage_rows,
        f"expected={expected_stage_rows}, observed={len(quality)}",
    )
    add(
        "quality_per_label_rows",
        len(quality_labels) == expected_stage_rows * 12,
        f"expected={expected_stage_rows * 12}, observed={len(quality_labels)}",
    )
    headline_performance = performance[list(PERFORMANCE_DIRECTIONS)]
    headline_quality = quality[list(QUALITY_DIRECTIONS)]
    add(
        "finite_headline_metrics",
        np.isfinite(headline_performance.to_numpy(dtype=float)).all()
        and np.isfinite(headline_quality.to_numpy(dtype=float)).all(),
        "performance and quality headline metrics",
    )

    marker_count = 0
    for seed in seeds:
        remove_marker = (
            base_root
            / f"seed_{seed}"
            / "sample20_remove_loop"
            / "remove_only"
            / "loop_08"
            / ".train_eval_complete"
        )
        refine_marker = (
            refine_root
            / f"seed_{seed}"
            / "llm_refine"
            / "loop_08"
            / ".train_eval_complete"
        )
        marker_count += int(remove_marker.is_file()) + int(refine_marker.is_file())
    add(
        "loop8_training_transactions",
        marker_count == len(seeds) * 2,
        f"expected={len(seeds) * 2}, observed={marker_count}",
    )

    all_label_points = bootstrap.loc[
        bootstrap["scope"].eq("all_labels")
        & bootstrap["uncertainty"].eq("hierarchical_seed_and_study"),
        ["comparison", "observed_mean_delta"],
    ]
    primary_stats = paired_metric_statistics(
        performance,
        {"study_weighted_auroc": 1},
    )[["comparison", "mean_raw_left_minus_right"]]
    point_check = primary_stats.merge(all_label_points, on="comparison", validate="one_to_one")
    add(
        "bootstrap_point_estimate_alignment",
        np.allclose(
            point_check["mean_raw_left_minus_right"],
            point_check["observed_mean_delta"],
            atol=1e-12,
            rtol=0,
        ),
        "all-label hierarchical observed points match paired seed estimates",
    )
    return pd.DataFrame(checks)


def main() -> int:
    args = parse_args()
    base_root = args.base_root.resolve()
    refine_root = args.refine_root.resolve()
    out_dir = args.out_dir.resolve()
    seeds = sorted(args.seeds)
    loops = sorted(set(args.loops))
    if seeds != DEFAULT_SEEDS:
        raise ValueError(f"Expected frozen seeds {DEFAULT_SEEDS}, got {seeds}")
    if loops != DEFAULT_LOOPS:
        raise ValueError(f"Expected frozen loops {DEFAULT_LOOPS}, got {loops}")
    out_dir.mkdir(parents=True, exist_ok=True)

    performance, label_rows, label_names, qa = validate_and_collect(
        base_root=base_root,
        refine_root=refine_root,
        refine_layout="own",
        seeds=seeds,
        remove_loops=loops,
        refine_loops=loops,
    )
    performance_summary = summarize_metrics(performance)
    performance_stats = paired_metric_statistics(
        performance,
        PERFORMANCE_DIRECTIONS,
    )
    comparisons = contrast_tuples()
    bootstrap = hierarchical_bootstrap(
        base_root=base_root,
        refine_root=refine_root,
        refine_layout="own",
        seeds=seeds,
        comparisons=comparisons,
        label_names=label_names,
        label_rows=label_rows,
        n_bootstrap=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
        workers=args.workers,
    )
    per_label = summarize_per_label(label_rows, comparisons)

    quality, quality_labels, actions = build_quality_outputs(
        base_root,
        refine_root,
        seeds,
        loops,
        args.workers,
    )
    quality_summary = summarize_quality(quality)
    quality_stats = paired_metric_statistics(
        quality,
        QUALITY_DIRECTIONS,
    )

    checks = build_checks(
        performance=performance,
        labels=label_rows,
        quality=quality,
        quality_labels=quality_labels,
        qa=qa,
        bootstrap=bootstrap,
        base_root=base_root,
        refine_root=refine_root,
        seeds=seeds,
        loops=loops,
    )
    if not checks["status"].eq("passed").all():
        raise RuntimeError(f"Unified evaluation checks failed:\n{checks.to_string(index=False)}")

    performance.to_csv(out_dir / "performance_per_seed_stage.csv", index=False)
    performance_summary.to_csv(out_dir / "performance_five_seed_summary.csv", index=False)
    performance_stats.to_csv(out_dir / "performance_paired_statistics.csv", index=False)
    bootstrap.to_csv(out_dir / "performance_hierarchical_bootstrap.csv", index=False)
    per_label.to_csv(out_dir / "performance_per_label_deltas.csv", index=False)
    quality.to_csv(out_dir / "quality_per_seed_stage.csv", index=False)
    quality_summary.to_csv(out_dir / "quality_five_seed_summary.csv", index=False)
    quality_stats.to_csv(out_dir / "quality_paired_statistics.csv", index=False)
    quality_labels.to_csv(out_dir / "quality_per_label.csv", index=False)
    actions.to_csv(out_dir / "refinement_action_oof_support.csv", index=False)
    checks.to_csv(out_dir / "validation_checks.csv", index=False)

    plot_performance(performance_summary, out_dir)
    plot_quality(quality_summary, out_dir)
    plot_coverage(quality_summary, out_dir)
    plot_bootstrap_forest(bootstrap, out_dir)

    metadata = {
        "status": "passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(
            Path(__file__).with_name(
                "unified_5seed_8loop_evaluation_protocol_20260723.md"
            ).resolve()
        ),
        "base_root": str(base_root),
        "refine_root": str(refine_root),
        "seeds": seeds,
        "loops": loops,
        "primary_endpoint": "study_weighted_auroc",
        "primary_comparison": CONTRASTS[0].name,
        "contrasts": [
            {
                "name": item.name,
                "left": list(item.left),
                "right": list(item.right),
                "tier": item.tier,
            }
            for item in CONTRASTS
        ],
        "bootstrap_iterations": args.bootstrap_iters,
        "bootstrap_seed": args.bootstrap_seed,
        "shared_study_resamples_across_seeds": True,
        "quality_evidence": "unchanged seed-specific initial OOF probabilities",
        "multiplicity": "Holm across eight frozen contrasts within each metric",
        "inference_status": (
            "exploratory protocol frozen after model outcomes but before this unified analysis"
        ),
        "checks": checks.to_dict("records"),
    }
    (out_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / ".evaluation_complete").touch()

    print(json.dumps(metadata, indent=2))
    print("\nPrimary performance statistics:")
    print(
        performance_stats.loc[
            performance_stats["comparison"].eq(CONTRASTS[0].name)
        ].to_string(index=False)
    )
    print("\nPrimary hierarchical AUROC statistics:")
    print(
        bootstrap.loc[
            bootstrap["comparison"].astype(str).eq(CONTRASTS[0].name)
            & bootstrap["uncertainty"].eq("hierarchical_seed_and_study")
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
