#!/usr/bin/env python3
"""Evaluate own-top20 refinement quality with seed-specific OOF evidence."""

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

from evaluate_mobilenet_data_quality_5seed import (
    LABEL_NAMES,
    annotate_rows,
    binary_from_raw,
    load_train_state,
    quality_metrics,
)


DEFAULT_BASE_ROOT = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_mobilenetv3_noes50_clean_3seed/20260707_123320"
)
DEFAULT_REFINE_ROOT = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_mobilenetv3_own_top20_llm_refine_5seed/20260713_184706"
)
SEEDS = [7, 13, 42, 97, 123]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--refine-root", type=Path, default=DEFAULT_REFINE_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--loops", nargs="+", type=int, default=list(range(1, 9)))
    parser.add_argument("--checkpoint-loop", type=int, default=5)
    parser.add_argument("--extension-loop", type=int, default=8)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument(
        "--refine-layout",
        choices=["own", "shared"],
        default="own",
        help="own: root/seed_N/llm_refine/loop_NN; shared: root/loop_NN",
    )
    parser.add_argument("--skip-dynamic", action="store_true")
    return parser.parse_args()


def action_loop_path(root: Path, layout: str, seed: int, loop: int) -> Path:
    if layout == "shared":
        return root / f"loop_{loop:02d}"
    return root / f"seed_{seed}" / "llm_refine" / f"loop_{loop:02d}"


def add_sample_health(metrics: dict[str, float | int]) -> dict[str, float | int]:
    result = dict(metrics)
    issue_free = 1.0 - float(result["sample_issue_rate"])
    result["sample_issue_free_rate"] = issue_free
    result["coverage_adjusted_sample_health"] = issue_free * float(result["sample_coverage"])
    return result


def map_pool_rows(pool_to_index: dict[int, int], values: pd.Series) -> np.ndarray:
    missing = sorted(set(values.astype(int)) - set(pool_to_index))
    if missing:
        raise ValueError(f"Action tables contain {len(missing)} unknown pool_row_id values")
    return np.asarray([pool_to_index[int(value)] for value in values], dtype=int)


def process_seed(
    base_root_text: str,
    refine_root_text: str,
    refine_layout: str,
    seed: int,
    loops: list[int],
    include_dynamic: bool,
) -> tuple[list[dict], list[dict], list[dict]]:
    base_root = Path(base_root_text)
    refine_root = Path(refine_root_text)
    initial_path = (
        base_root
        / f"seed_{seed}"
        / "sample20_remove_loop"
        / "remove_only"
        / "loop_01"
        / "oof"
        / "train_cleanlab_sample_details.csv"
    )
    frame, initial_labels, initial_valid, initial_probs = load_train_state(initial_path)
    baseline_evaluable_samples = int(initial_valid.any(axis=1).sum())
    baseline_valid_entries = int(initial_valid.sum())
    pool_ids = frame["pool_row_id"].to_numpy(dtype=np.int64)
    pool_to_index = {int(pool_id): index for index, pool_id in enumerate(pool_ids)}

    metric_rows: list[dict] = []
    per_label_rows: list[dict] = []
    action_rows: list[dict] = []

    baseline_metrics, baseline_labels, baseline_issue_mask = quality_metrics(
        initial_labels,
        initial_valid,
        initial_probs,
        baseline_valid_entries,
        baseline_evaluable_samples,
    )
    baseline_metrics = add_sample_health(baseline_metrics)
    row, label_metrics = annotate_rows(
        baseline_metrics,
        baseline_labels,
        seed,
        "frozen_initial_oof_post_action",
        "baseline",
        0,
        0,
    )
    metric_rows.append(row)
    per_label_rows.extend(label_metrics)

    for loop in loops:
        loop_root = action_loop_path(refine_root, refine_layout, seed, loop)
        relabel_path = loop_root / "applied_relabel_entries.csv"
        mask_path = loop_root / "applied_mask_entries.csv"
        if not relabel_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing action tables under {loop_root}")
        relabels = pd.read_csv(relabel_path)
        masks = pd.read_csv(mask_path)
        refined_labels = initial_labels.copy()
        refined_valid = initial_valid.copy()

        relabel_indices = map_pool_rows(pool_to_index, relabels["pool_row_id"])
        relabel_label_indices = relabels["label_index"].to_numpy(dtype=int)
        old_binary = refined_labels[relabel_indices, relabel_label_indices].copy()
        new_binary = binary_from_raw(relabels["new_raw_label"].to_numpy(dtype=float))
        relabel_probs = initial_probs[relabel_indices, relabel_label_indices]
        old_confidence = np.where(old_binary == 1, relabel_probs, 1.0 - relabel_probs)
        new_confidence = np.where(new_binary == 1, relabel_probs, 1.0 - relabel_probs)
        refined_labels[relabel_indices, relabel_label_indices] = new_binary
        refined_valid[relabel_indices, relabel_label_indices] = True

        mask_indices = map_pool_rows(pool_to_index, masks["pool_row_id"])
        mask_label_indices = masks["label_index"].to_numpy(dtype=int)
        mask_old_labels = refined_labels[mask_indices, mask_label_indices]
        mask_probs = initial_probs[mask_indices, mask_label_indices]
        mask_confidence = np.where(mask_old_labels == 1, mask_probs, 1.0 - mask_probs)
        conflict_mask_count = (
            int(masks["action_resolution"].eq("mask_response_field_conflict").sum())
            if "action_resolution" in masks.columns
            else 0
        )
        refined_valid[mask_indices, mask_label_indices] = False
        refined_labels[mask_indices, mask_label_indices] = np.nan

        metrics, labels, _ = quality_metrics(
            refined_labels,
            refined_valid,
            initial_probs,
            baseline_valid_entries,
            baseline_evaluable_samples,
        )
        metrics = add_sample_health(metrics)
        row, annotated = annotate_rows(
            metrics,
            labels,
            seed,
            "frozen_initial_oof_post_action",
            "refine",
            loop,
            len(relabels) + len(masks),
        )
        row["relabel_count"] = len(relabels)
        row["mask_count"] = len(masks)
        metric_rows.append(row)
        per_label_rows.extend(annotated)

        relabel_was_issue = baseline_issue_mask[relabel_indices, relabel_label_indices]
        mask_was_issue = baseline_issue_mask[mask_indices, mask_label_indices]
        action_rows.append(
            {
                "seed": seed,
                "loop": loop,
                "relabel_count": len(relabels),
                "mask_count": len(masks),
                "response_conflict_mask_count": conflict_mask_count,
                "relabel_mean_self_confidence_change": float(
                    np.mean(new_confidence - old_confidence)
                ),
                "relabel_fraction_improved_self_confidence": float(
                    np.mean(new_confidence > old_confidence)
                ),
                "relabel_fraction_flagged_by_seed_initial_oof": float(np.mean(relabel_was_issue)),
                "mask_mean_original_self_confidence": float(np.mean(mask_confidence)),
                "mask_fraction_flagged_by_seed_initial_oof": float(np.mean(mask_was_issue)),
            }
        )

        if include_dynamic:
            dynamic_path = loop_root / "oof" / "train_cleanlab_sample_details.csv"
            if loop == 1 and not dynamic_path.exists():
                dynamic_path = initial_path
            if not dynamic_path.is_file():
                raise FileNotFoundError(dynamic_path)
            _, dynamic_labels, dynamic_valid, dynamic_probs = load_train_state(dynamic_path)
            dynamic_metrics, dynamic_label_metrics, _ = quality_metrics(
                dynamic_labels,
                dynamic_valid,
                dynamic_probs,
                baseline_valid_entries,
                baseline_evaluable_samples,
            )
            dynamic_metrics = add_sample_health(dynamic_metrics)
            prior_actions = 0
            if loop > 1:
                prior_root = action_loop_path(refine_root, refine_layout, seed, loop - 1)
                prior_actions = len(pd.read_csv(prior_root / "applied_relabel_entries.csv")) + len(
                    pd.read_csv(prior_root / "applied_mask_entries.csv")
                )
            dynamic_row, dynamic_labels_annotated = annotate_rows(
                dynamic_metrics,
                dynamic_label_metrics,
                seed,
                "dynamic_iterative_oof_pre_action",
                "refine",
                loop,
                prior_actions,
            )
            metric_rows.append(dynamic_row)
            per_label_rows.extend(dynamic_labels_annotated)

    return metric_rows, per_label_rows, action_rows


def process_seed_from_args(args: tuple) -> tuple[list[dict], list[dict], list[dict]]:
    return process_seed(*args)


def summarize_quality(frame: pd.DataFrame) -> pd.DataFrame:
    excluded = {"seed", "evidence_mode", "method", "loop"}
    metrics = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in excluded and column not in {"action_count", "relabel_count", "mask_count"}
    ]
    rows: list[dict] = []
    for (evidence_mode, method, loop), group in frame.groupby(
        ["evidence_mode", "method", "loop"], sort=True
    ):
        row: dict[str, float | int | str] = {
            "evidence_mode": evidence_mode,
            "method": method,
            "loop": int(loop),
            "n_seeds": int(len(group)),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["evidence_mode", "method", "loop"])


def exact_signflip_p(delta: np.ndarray) -> float:
    observed = abs(float(np.mean(delta)))
    values = [
        abs(float(np.mean(np.asarray(signs) * delta)))
        for signs in product([-1, 1], repeat=len(delta))
    ]
    return float(np.mean(np.asarray(values) >= observed - 1e-15))


def holm_adjust(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty_like(values, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_t_p(delta: np.ndarray) -> float:
    if np.allclose(delta, delta[0], rtol=0.0, atol=1e-15):
        return 1.0 if float(np.mean(delta)) == 0.0 else 0.0
    return float(stats.ttest_1samp(delta, 0).pvalue)


def quality_statistics(
    frame: pd.DataFrame, checkpoint_loop: int, extension_loop: int
) -> pd.DataFrame:
    source = frame.loc[frame["evidence_mode"].eq("frozen_initial_oof_post_action")]
    comparisons = [
        (f"refine_L{checkpoint_loop}_vs_baseline", ("refine", checkpoint_loop), ("baseline", 0)),
        (f"refine_L{extension_loop}_vs_baseline", ("refine", extension_loop), ("baseline", 0)),
        (
            f"refine_L{extension_loop}_vs_refine_L{checkpoint_loop}",
            ("refine", extension_loop),
            ("refine", checkpoint_loop),
        ),
    ]
    directions = {
        "dqs_flattened": 1,
        "coverage_adjusted_dqs": 1,
        "sample_issue_free_rate": 1,
        "coverage_adjusted_sample_health": 1,
        "valid_entry_coverage": 1,
        "entry_issue_rate_valid_entries": -1,
    }
    rows: list[dict] = []
    for metric, direction in directions.items():
        pivot = source.pivot(index="seed", columns=["method", "loop"], values=metric)
        family: list[dict] = []
        for name, left, right in comparisons:
            delta = direction * (pivot[left] - pivot[right]).to_numpy(dtype=float)
            mean = float(delta.mean())
            sd = float(delta.std(ddof=1))
            half_width = float(stats.t.ppf(0.975, len(delta) - 1) * sd / np.sqrt(len(delta)))
            family.append(
                {
                    "metric": metric,
                    "comparison": name,
                    "direction": "higher_is_better" if direction == 1 else "lower_is_better",
                    "n_seeds": len(delta),
                    "mean_improvement": mean,
                    "sd_improvement": sd,
                    "t_ci_low_2p5": mean - half_width,
                    "t_ci_high_97p5": mean + half_width,
                    "positive_improvement_seeds": int((delta > 0).sum()),
                    "exact_signflip_p_two_sided": exact_signflip_p(delta),
                    "paired_t_p_two_sided": paired_t_p(delta),
                }
            )
        exact_adjusted = holm_adjust(
            np.asarray([row["exact_signflip_p_two_sided"] for row in family])
        )
        t_adjusted = holm_adjust(np.asarray([row["paired_t_p_two_sided"] for row in family]))
        for row, exact_p, t_p in zip(family, exact_adjusted, t_adjusted):
            row["holm3_exact_p"] = float(exact_p)
            row["holm3_paired_t_p"] = float(t_p)
        rows.extend(family)
    return pd.DataFrame(rows)


def add_baseline_to_trajectory(
    summary: pd.DataFrame, metric: str
) -> tuple[pd.Series, pd.DataFrame]:
    frozen = summary.loc[summary["evidence_mode"].eq("frozen_initial_oof_post_action")]
    baseline = frozen.loc[frozen["method"].eq("baseline")].iloc[0]
    refine = frozen.loc[frozen["method"].eq("refine")].sort_values("loop")
    return baseline, refine


def plot_quality(summary: pd.DataFrame, per_seed: pd.DataFrame, out_dir: Path) -> None:
    pairs = [
        (
            "dqs_flattened",
            "coverage_adjusted_dqs",
            "Raw entry DQS",
            "Coverage-adjusted entry DQS",
            "Entry-level quality score",
            (0.79, 0.99),
            "own_top20_entry_dqs_common_scale.png",
        ),
        (
            "sample_issue_free_rate",
            "coverage_adjusted_sample_health",
            "Issue-free rate among retained samples",
            "Issue-free sample mass vs original N",
            "Sample-level issue-free proportion",
            (0.75, 0.93),
            "own_top20_sample_issue_free_rate.png",
        ),
    ]
    frozen_seed = per_seed.loc[
        per_seed["evidence_mode"].eq("frozen_initial_oof_post_action")
    ]
    for raw_metric, adjusted_metric, raw_title, adjusted_title, ylabel, ylim, filename in pairs:
        plotted_values = frozen_seed[[raw_metric, adjusted_metric]].to_numpy(dtype=float)
        shared_ylim = (
            min(ylim[0], float(np.nanmin(plotted_values)) - 0.01),
            max(ylim[1], float(np.nanmax(plotted_values)) + 0.01),
        )
        fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6), sharex=True, sharey=True)
        for axis, metric, title in zip(
            axes, [raw_metric, adjusted_metric], [raw_title, adjusted_title], strict=True
        ):
            baseline, refine = add_baseline_to_trajectory(summary, metric)
            mean_col = f"{metric}_mean"
            sd_col = f"{metric}_sd"
            loops = np.r_[0, refine["loop"].to_numpy()]
            means = np.r_[baseline[mean_col], refine[mean_col].to_numpy()]
            sds = np.r_[baseline[sd_col], refine[sd_col].to_numpy()]
            for seed in sorted(frozen_seed["seed"].unique()):
                seed_baseline = frozen_seed.loc[
                    frozen_seed["seed"].eq(seed) & frozen_seed["method"].eq("baseline"), metric
                ].iloc[0]
                seed_refine = frozen_seed.loc[
                    frozen_seed["seed"].eq(seed) & frozen_seed["method"].eq("refine")
                ].sort_values("loop")
                axis.plot(
                    np.r_[0, seed_refine["loop"].to_numpy()],
                    np.r_[seed_baseline, seed_refine[metric].to_numpy()],
                    color="#C6372F",
                    alpha=0.18,
                    marker="o",
                    markersize=2.5,
                    linewidth=1,
                )
            axis.errorbar(
                loops,
                means,
                yerr=sds,
                color="#C6372F",
                marker="o",
                markersize=6,
                linewidth=2.5,
                elinewidth=2,
                capsize=6,
                label="Own-top20 LLM refinement",
            )
            axis.set_title(title, fontweight="bold")
            axis.set_xlabel("Cleaning loop (0 = baseline)")
            axis.set_ylim(*shared_ylim)
            axis.grid(axis="y", color="#DFE4EA")
        axes[0].set_ylabel(ylabel)
        axes[0].legend(frameon=False)
        fig.suptitle("Five-seed mean +/- SD; faint lines are individual seeds", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    args = parse_args()
    seeds = sorted(args.seeds)
    loops = sorted(set(args.loops))
    if len(seeds) != 5:
        raise ValueError("The pre-locked analysis requires exactly five seeds")
    if args.checkpoint_loop not in loops or args.extension_loop not in loops:
        raise ValueError("Checkpoint and extension loops must be present in --loops")
    if args.refine_layout == "shared" and not args.skip_dynamic:
        raise ValueError("Shared action tables do not provide seed-specific dynamic OOF evidence")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    worker_args = [
        (
            str(args.base_root),
            str(args.refine_root),
            args.refine_layout,
            seed,
            loops,
            not args.skip_dynamic,
        )
        for seed in seeds
    ]
    if args.workers == 1:
        results = [process_seed_from_args(item) for item in worker_args]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(seeds))) as executor:
            results = list(executor.map(process_seed_from_args, worker_args))
    metrics = pd.DataFrame([row for result in results for row in result[0]])
    per_label = pd.DataFrame([row for result in results for row in result[1]])
    actions = pd.DataFrame([row for result in results for row in result[2]])
    summary = summarize_quality(metrics)
    quality_stats = quality_statistics(metrics, args.checkpoint_loop, args.extension_loop)

    metrics.to_csv(args.out_dir / "own_top20_quality_per_seed.csv", index=False)
    summary.to_csv(args.out_dir / "own_top20_quality_summary.csv", index=False)
    quality_stats.to_csv(args.out_dir / "own_top20_quality_prelocked_statistics.csv", index=False)
    per_label.to_csv(args.out_dir / "own_top20_quality_per_label.csv", index=False)
    actions.to_csv(args.out_dir / "own_top20_llm_action_oof_support.csv", index=False)
    metadata = {
        "status": "passed",
        "base_root": str(args.base_root),
        "refine_root": str(args.refine_root),
        "refine_layout": args.refine_layout,
        "seeds": seeds,
        "loops": loops,
        "checkpoint_loop": args.checkpoint_loop,
        "extension_loop": args.extension_loop,
        "frozen_evidence": "Each seed's unchanged Loop1 OOF probabilities evaluated after cumulative actions",
        "dynamic_evidence": (
            "Each seed's iterative OOF probabilities before the action of the named loop"
            if not args.skip_dynamic
            else "not evaluated"
        ),
        "sample_issue_free_rate": "1 - fraction of evaluable samples with any valid entry flagged by confident learning",
        "sample_coverage": "samples retaining at least one valid target label divided by baseline evaluable samples",
        "sample_zero_valid_policy": "samples with zero valid target labels are unevaluable, not issue-free",
        "sample_metric_status": "custom strict diagnostic, not an official Cleanlab DQS",
        "coverage_adjusted_dqs_status": "custom denominator diagnostic, not an official Cleanlab metric",
        "response_conflict_policy": "Inconsistent mismatch_type/recommended_action rows are conservatively masked and counted in the action-support table",
        "multiplicity": "Holm adjustment across three pre-locked checkpoint contrasts within each metric",
    }
    (args.out_dir / "own_top20_quality_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    plot_quality(summary, metrics, args.out_dir)
    print(json.dumps(metadata, indent=2))
    print("\nFrozen quality summary:")
    columns = [
        "method",
        "loop",
        "dqs_flattened_mean",
        "coverage_adjusted_dqs_mean",
        "sample_issue_free_rate_mean",
        "coverage_adjusted_sample_health_mean",
        "valid_entry_coverage_mean",
    ]
    print(
        summary.loc[
            summary["evidence_mode"].eq("frozen_initial_oof_post_action"), columns
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
