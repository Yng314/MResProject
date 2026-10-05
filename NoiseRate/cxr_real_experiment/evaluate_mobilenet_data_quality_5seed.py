#!/usr/bin/env python3
"""Evaluate five-seed label quality, coverage, stability, and test metrics."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations, product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cleanlab.dataset import overall_label_health_score
from cleanlab.filter import find_label_issues
from scipy import stats
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


DEFAULT_ROOT = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_mobilenetv3_noes50_clean_3seed/20260707_123320"
)
DEFAULT_REFINE_SOURCE = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_xrv_iterative_sample20_dual_loop/20260624_150533/llm_refine"
)
DEFAULT_ARCHIVED_DQS = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "dqs_analysis/xrv_iterative_sample20_official_cleanlab_dqs_summary.csv"
)
DEFAULT_ARCHIVED_XRV_ROOT = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_xrv_iterative_sample20_dual_loop/20260624_150533"
)

LABEL_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
]


def parse_pipe_matrix(values: pd.Series, dtype: type) -> np.ndarray:
    return np.vstack([np.fromstring(str(value), sep="|", dtype=dtype) for value in values])


def load_train_state(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    usecols = ["pool_row_id", "binary_labels_for_detection", "valid_label_mask", "pred_probs"]
    frame = pd.read_csv(path, usecols=usecols)
    labels = parse_pipe_matrix(frame["binary_labels_for_detection"], float)
    valid = parse_pipe_matrix(frame["valid_label_mask"], int).astype(bool)
    probs = parse_pipe_matrix(frame["pred_probs"], float)
    return frame, labels, valid, probs


def sample_denominator_metrics(
    valid: np.ndarray,
    issue_mask: np.ndarray,
    baseline_evaluable_samples: int,
) -> dict[str, float | int]:
    if valid.shape != issue_mask.shape:
        raise ValueError("valid and issue_mask must have identical shapes")
    if baseline_evaluable_samples <= 0:
        raise ValueError("baseline_evaluable_samples must be positive")
    if np.any(issue_mask & ~valid):
        raise ValueError("Issue mask contains entries outside the valid-label mask")

    evaluable = valid.any(axis=1)
    n_evaluable = int(evaluable.sum())
    if n_evaluable == 0:
        raise ValueError("No samples retain an evaluable target label")
    sample_issue_count = int(issue_mask.any(axis=1).sum())
    if sample_issue_count > n_evaluable:
        raise AssertionError("Sample issue count exceeds the evaluable-sample denominator")
    return {
        "n_evaluable_samples": n_evaluable,
        "n_zero_valid_samples": int(len(evaluable) - n_evaluable),
        "sample_coverage": n_evaluable / baseline_evaluable_samples,
        "sample_issue_count": sample_issue_count,
        "sample_issue_rate": sample_issue_count / n_evaluable,
        "sample_issue_density_all_rows": sample_issue_count / len(evaluable),
    }


def quality_metrics(
    labels: np.ndarray,
    valid: np.ndarray,
    probs: np.ndarray,
    baseline_valid_entries: int,
    baseline_evaluable_samples: int,
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]], np.ndarray]:
    n_samples, n_labels = labels.shape
    issue_mask = np.zeros_like(valid, dtype=bool)
    label_rows: list[dict[str, float | int | str]] = []
    per_label_dqs: list[float] = []
    per_label_weights: list[int] = []

    for label_index, label_name in enumerate(LABEL_NAMES):
        mask = valid[:, label_index]
        y_true = labels[mask, label_index].astype(int)
        y_prob = probs[mask, label_index]
        valid_count = int(mask.sum())
        positive_count = int(y_true.sum())
        negative_count = int(valid_count - positive_count)
        dqs = float("nan")
        issue_count = 0
        if valid_count and np.unique(y_true).size == 2:
            pred_probs = np.column_stack([1.0 - y_prob, y_prob])
            dqs = float(
                overall_label_health_score(
                    labels=y_true,
                    pred_probs=pred_probs,
                    verbose=False,
                )
            )
            local_issues = find_label_issues(
                labels=y_true,
                pred_probs=pred_probs,
                verbose=False,
            ).astype(bool)
            issue_mask[mask, label_index] = local_issues
            issue_count = int(local_issues.sum())
            per_label_dqs.append(dqs)
            per_label_weights.append(valid_count)
        self_confidence = np.where(y_true == 1, y_prob, 1.0 - y_prob) if valid_count else np.array([])
        label_rows.append(
            {
                "label_index": label_index,
                "label_name": label_name,
                "valid_entries": valid_count,
                "positive_entries": positive_count,
                "negative_entries": negative_count,
                "positive_prevalence": positive_count / valid_count if valid_count else np.nan,
                "dqs": dqs,
                "issue_count": issue_count,
                "valid_entry_issue_rate": issue_count / valid_count if valid_count else np.nan,
                "mean_self_confidence": float(np.mean(self_confidence)) if self_confidence.size else np.nan,
            }
        )

    flattened_labels = labels[valid].astype(int)
    flattened_probs = probs[valid]
    dqs_flattened = float(
        overall_label_health_score(
            labels=flattened_labels,
            pred_probs=np.column_stack([1.0 - flattened_probs, flattened_probs]),
            verbose=False,
        )
    )
    n_valid_entries = int(valid.sum())
    issue_count = int(issue_mask.sum())
    valid_coverage = n_valid_entries / baseline_valid_entries
    sample_metrics = sample_denominator_metrics(
        valid,
        issue_mask,
        baseline_evaluable_samples,
    )
    metrics = {
        "n_samples": n_samples,
        "n_valid_entries": n_valid_entries,
        "valid_entry_coverage": valid_coverage,
        "dqs_flattened": dqs_flattened,
        "dqs_per_label_weighted": float(np.average(per_label_dqs, weights=per_label_weights)),
        "dqs_per_label_mean": float(np.mean(per_label_dqs)),
        **sample_metrics,
        "entry_issue_count": issue_count,
        "entry_issue_density_all_slots": issue_count / (n_samples * n_labels),
        "entry_issue_rate_valid_entries": issue_count / n_valid_entries,
        "estimated_error_entries_dqs": (1.0 - dqs_flattened) * n_valid_entries,
        "estimated_healthy_entries_dqs": dqs_flattened * n_valid_entries,
        "coverage_adjusted_dqs": dqs_flattened * valid_coverage,
        "mean_entry_self_confidence": float(
            np.mean(np.where(flattened_labels == 1, flattened_probs, 1.0 - flattened_probs))
        ),
    }
    return metrics, label_rows, issue_mask


def annotate_rows(
    metrics: dict[str, float | int],
    label_rows: list[dict[str, float | int | str]],
    seed: int,
    evidence_mode: str,
    method: str,
    loop: int,
    action_count: int,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    metric_row = {
        "seed": seed,
        "evidence_mode": evidence_mode,
        "method": method,
        "loop": loop,
        "action_count": action_count,
        **metrics,
    }
    annotated_labels = [
        {
            "seed": seed,
            "evidence_mode": evidence_mode,
            "method": method,
            "loop": loop,
            **row,
        }
        for row in label_rows
    ]
    return metric_row, annotated_labels


def binary_from_raw(values: np.ndarray) -> np.ndarray:
    return np.where(np.isin(values, [1.0, -1.0]), 1.0, 0.0)


def process_seed(
    root_text: str,
    refine_source_text: str,
    seed: int,
    loops: list[int],
) -> tuple[list[dict], list[dict], list[dict]]:
    root = Path(root_text)
    refine_source = Path(refine_source_text)
    seed_root = root / f"seed_{seed}"
    initial_path = (
        seed_root
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
    row, labels = annotate_rows(
        baseline_metrics,
        baseline_labels,
        seed,
        "frozen_initial_oof_post_action",
        "baseline",
        0,
        0,
    )
    metric_rows.append(row)
    per_label_rows.extend(labels)

    for loop in loops:
        remove_root = seed_root / "sample20_remove_loop" / "remove_only" / f"loop_{loop:02d}"
        removed = pd.read_csv(remove_root / "applied_removed_samples.csv", usecols=["pool_row_id"])
        removed_ids = set(removed["pool_row_id"].astype(int))
        keep = np.fromiter((int(pool_id) not in removed_ids for pool_id in pool_ids), dtype=bool)
        metrics, label_metrics, _ = quality_metrics(
            initial_labels[keep],
            initial_valid[keep],
            initial_probs[keep],
            baseline_valid_entries,
            baseline_evaluable_samples,
        )
        row, labels = annotate_rows(
            metrics,
            label_metrics,
            seed,
            "frozen_initial_oof_post_action",
            "remove",
            loop,
            len(removed_ids),
        )
        metric_rows.append(row)
        per_label_rows.extend(labels)

        relabels = pd.read_csv(
            refine_source / f"loop_{loop:02d}" / "applied_relabel_entries.csv"
        )
        masks = pd.read_csv(refine_source / f"loop_{loop:02d}" / "applied_mask_entries.csv")
        refined_labels = initial_labels.copy()
        refined_valid = initial_valid.copy()

        relabel_indices = np.asarray(
            [pool_to_index[int(pool_id)] for pool_id in relabels["pool_row_id"]], dtype=int
        )
        relabel_label_indices = relabels["label_index"].to_numpy(dtype=int)
        old_binary = refined_labels[relabel_indices, relabel_label_indices].copy()
        new_binary = binary_from_raw(relabels["new_raw_label"].to_numpy(dtype=float))
        relabel_probs = initial_probs[relabel_indices, relabel_label_indices]
        old_confidence = np.where(old_binary == 1, relabel_probs, 1.0 - relabel_probs)
        new_confidence = np.where(new_binary == 1, relabel_probs, 1.0 - relabel_probs)
        refined_labels[relabel_indices, relabel_label_indices] = new_binary
        refined_valid[relabel_indices, relabel_label_indices] = True

        mask_indices = np.asarray(
            [pool_to_index[int(pool_id)] for pool_id in masks["pool_row_id"]], dtype=int
        )
        mask_label_indices = masks["label_index"].to_numpy(dtype=int)
        mask_old_labels = refined_labels[mask_indices, mask_label_indices]
        mask_probs = initial_probs[mask_indices, mask_label_indices]
        mask_confidence = np.where(mask_old_labels == 1, mask_probs, 1.0 - mask_probs)
        mask_was_issue = baseline_issue_mask[mask_indices, mask_label_indices]
        relabel_was_issue = baseline_issue_mask[relabel_indices, relabel_label_indices]
        refined_valid[mask_indices, mask_label_indices] = False
        refined_labels[mask_indices, mask_label_indices] = np.nan

        metrics, label_metrics, _ = quality_metrics(
            refined_labels,
            refined_valid,
            initial_probs,
            baseline_valid_entries,
            baseline_evaluable_samples,
        )
        row, labels = annotate_rows(
            metrics,
            label_metrics,
            seed,
            "frozen_initial_oof_post_action",
            "refine",
            loop,
            len(relabels) + len(masks),
        )
        row["relabel_count"] = len(relabels)
        row["mask_count"] = len(masks)
        metric_rows.append(row)
        per_label_rows.extend(labels)
        action_rows.append(
            {
                "seed": seed,
                "loop": loop,
                "relabel_count": len(relabels),
                "mask_count": len(masks),
                "relabel_mean_self_confidence_change": float(np.mean(new_confidence - old_confidence)),
                "relabel_fraction_improved_self_confidence": float(np.mean(new_confidence > old_confidence)),
                "relabel_fraction_flagged_by_seed_oof": float(np.mean(relabel_was_issue)),
                "mask_mean_original_self_confidence": float(np.mean(mask_confidence)),
                "mask_fraction_flagged_by_seed_oof": float(np.mean(mask_was_issue)),
            }
        )

    for loop in loops:
        dynamic_path = (
            seed_root
            / "sample20_remove_loop"
            / "remove_only"
            / f"loop_{loop:02d}"
            / "oof"
            / "train_cleanlab_sample_details.csv"
        )
        if loop == 1:
            metrics = baseline_metrics.copy()
            label_metrics = [row.copy() for row in baseline_labels]
        else:
            _, labels_dynamic, valid_dynamic, probs_dynamic = load_train_state(dynamic_path)
            metrics, label_metrics, _ = quality_metrics(
                labels_dynamic,
                valid_dynamic,
                probs_dynamic,
                baseline_valid_entries,
                baseline_evaluable_samples,
            )
        prior_actions = 0
        if loop > 1:
            prior = pd.read_csv(
                seed_root
                / "sample20_remove_loop"
                / "remove_only"
                / f"loop_{loop - 1:02d}"
                / "applied_removed_samples.csv",
                usecols=["pool_row_id"],
            )
            prior_actions = int(prior["pool_row_id"].nunique())
        row, labels = annotate_rows(
            metrics,
            label_metrics,
            seed,
            "dynamic_iterative_oof_pre_action",
            "remove",
            loop,
            prior_actions,
        )
        metric_rows.append(row)
        per_label_rows.extend(labels)

    return metric_rows, per_label_rows, action_rows


def exact_signflip_p(delta: np.ndarray) -> float:
    observed = abs(float(np.mean(delta)))
    permuted = [
        abs(float(np.mean(np.asarray(signs) * delta)))
        for signs in product([-1, 1], repeat=len(delta))
    ]
    return float(np.mean(np.asarray(permuted) >= observed - 1e-15))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running = 0.0
    n_values = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (n_values - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_statistics(
    frame: pd.DataFrame,
    loops: list[int],
    metric_directions: dict[str, int],
    evidence_mode: str,
) -> pd.DataFrame:
    source = frame.loc[frame["evidence_mode"] == evidence_mode]
    comparisons = [
        ("remove_vs_baseline", "remove", "baseline"),
        ("refine_vs_baseline", "refine", "baseline"),
        ("refine_vs_remove", "refine", "remove"),
    ]
    rows: list[dict] = []
    for metric, direction in metric_directions.items():
        pivot = source.pivot(index="seed", columns=["method", "loop"], values=metric)
        for comparison, left, right in comparisons:
            family: list[dict] = []
            for loop in loops:
                left_key = (left, loop if left != "baseline" else 0)
                right_key = (right, loop if right != "baseline" else 0)
                delta = direction * (pivot[left_key] - pivot[right_key]).to_numpy(dtype=float)
                mean = float(delta.mean())
                sd = float(delta.std(ddof=1))
                half_width = float(stats.t.ppf(0.975, len(delta) - 1) * sd / np.sqrt(len(delta)))
                family.append(
                    {
                        "evidence_mode": evidence_mode,
                        "metric": metric,
                        "comparison": comparison,
                        "loop": loop,
                        "n_seeds": len(delta),
                        "mean_improvement": mean,
                        "sd_improvement": sd,
                        "t_ci_low_2p5": mean - half_width,
                        "t_ci_high_97p5": mean + half_width,
                        "positive_improvement_seeds": int((delta > 0).sum()),
                        "exact_signflip_p_two_sided": exact_signflip_p(delta),
                    }
                )
            adjusted = holm_adjust(
                np.asarray([row["exact_signflip_p_two_sided"] for row in family])
            )
            for row, adjusted_p in zip(family, adjusted):
                row["holm5_exact_p"] = adjusted_p
            rows.extend(family)
    return pd.DataFrame(rows)


def summarize_quality(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "n_samples",
        "n_valid_entries",
        "sample_coverage",
        "valid_entry_coverage",
        "dqs_flattened",
        "dqs_per_label_weighted",
        "dqs_per_label_mean",
        "sample_issue_rate",
        "entry_issue_density_all_slots",
        "entry_issue_rate_valid_entries",
        "estimated_error_entries_dqs",
        "estimated_healthy_entries_dqs",
        "coverage_adjusted_dqs",
        "mean_entry_self_confidence",
    ]
    rows: list[dict] = []
    for keys, group in frame.groupby(["evidence_mode", "method", "loop"]):
        row = {"evidence_mode": keys[0], "method": keys[1], "loop": int(keys[2]), "n_seeds": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["evidence_mode", "method", "loop"])


def removal_stability(root: Path, seeds: list[int], loops: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict] = []
    summary_rows: list[dict] = []
    for loop in loops:
        sets: dict[int, set[int]] = {}
        for seed in seeds:
            path = (
                root
                / f"seed_{seed}"
                / "sample20_remove_loop"
                / "remove_only"
                / f"loop_{loop:02d}"
                / "applied_removed_samples.csv"
            )
            sets[seed] = set(pd.read_csv(path, usecols=["pool_row_id"])["pool_row_id"].astype(int))
        for seed_a, seed_b in combinations(seeds, 2):
            left, right = sets[seed_a], sets[seed_b]
            intersection = len(left & right)
            union = len(left | right)
            pair_rows.append(
                {
                    "loop": loop,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "size_a": len(left),
                    "size_b": len(right),
                    "intersection": intersection,
                    "union": union,
                    "jaccard": intersection / union,
                    "overlap_coefficient": intersection / min(len(left), len(right)),
                }
            )
        counts: dict[int, int] = {}
        for selected in sets.values():
            for pool_id in selected:
                counts[pool_id] = counts.get(pool_id, 0) + 1
        union_count = len(counts)
        summary_rows.append(
            {
                "loop": loop,
                "mean_selected": float(np.mean([len(selected) for selected in sets.values()])),
                "union_selected": union_count,
                "selected_by_at_least_3_seeds": int(sum(count >= 3 for count in counts.values())),
                "selected_by_all_5_seeds": int(sum(count == len(seeds) for count in counts.values())),
                "consensus_3_of_5_fraction_of_union": sum(count >= 3 for count in counts.values()) / union_count,
                "consensus_5_of_5_fraction_of_union": sum(count == len(seeds) for count in counts.values()) / union_count,
            }
        )
    pairs = pd.DataFrame(pair_rows)
    summary = pd.DataFrame(summary_rows)
    aggregate = pairs.groupby("loop").agg(
        mean_pairwise_jaccard=("jaccard", "mean"),
        sd_pairwise_jaccard=("jaccard", "std"),
        min_pairwise_jaccard=("jaccard", "min"),
        max_pairwise_jaccard=("jaccard", "max"),
        mean_overlap_coefficient=("overlap_coefficient", "mean"),
    ).reset_index()
    return pairs, summary.merge(aggregate, on="loop")


def ranking_stability(root: Path, seeds: list[int]) -> pd.DataFrame:
    series: dict[int, pd.Series] = {}
    for seed in seeds:
        path = (
            root
            / f"seed_{seed}"
            / "sample20_remove_loop"
            / "remove_only"
            / "loop_01"
            / "oof"
            / "train_cleanlab_sample_details.csv"
        )
        frame = pd.read_csv(path, usecols=["pool_row_id", "sample_quality_self_confidence"])
        series[seed] = frame.set_index("pool_row_id")["sample_quality_self_confidence"]
    rows: list[dict] = []
    for seed_a, seed_b in combinations(seeds, 2):
        aligned = pd.concat([series[seed_a], series[seed_b]], axis=1, join="inner").dropna()
        correlation = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1]).statistic
        rows.append({"seed_a": seed_a, "seed_b": seed_b, "n_common": len(aligned), "spearman_rho": correlation})
    return pd.DataFrame(rows)


def test_prediction_path(root: Path, seed: int, method: str, loop: int) -> Path:
    seed_root = root / f"seed_{seed}"
    if method == "baseline":
        return seed_root / "baseline_no_clean" / "test_study_predictions.csv"
    if method == "remove":
        return (
            seed_root
            / "sample20_remove_loop"
            / "remove_only"
            / f"loop_{loop:02d}"
            / "train_eval"
            / "test_study_predictions.csv"
        )
    return (
        seed_root
        / "sample20_llm_refine_fixed_xrv_s13"
        / f"loop_{loop:02d}"
        / "train_eval"
        / "test_study_predictions.csv"
    )


def heldout_metrics(root: Path, seeds: list[int], loops: list[int]) -> pd.DataFrame:
    support = pd.read_csv(root / f"seed_{seeds[0]}" / "baseline_no_clean" / "test_study_auroc_summary.csv")
    stable = set(
        support.loc[
            support[["study_positive_count", "study_negative_count"]].min(axis=1) >= 5,
            "label_name",
        ]
    )
    scopes = {"all_labels": set(LABEL_NAMES), "support_ge_5": stable}
    rows: list[dict] = []
    for seed in seeds:
        stages = [("baseline", 0)] + [(method, loop) for method in ["remove", "refine"] for loop in loops]
        for method, loop in stages:
            frame = pd.read_csv(test_prediction_path(root, seed, method, loop))
            labels = parse_pipe_matrix(frame["binary_labels_for_metric"], float)
            valid = parse_pipe_matrix(frame["valid_label_mask"], int).astype(bool)
            probs = parse_pipe_matrix(frame["pred_probs"], float)
            for scope, names in scopes.items():
                label_indices = [index for index, name in enumerate(LABEL_NAMES) if name in names]
                aucs: list[float] = []
                aps: list[float] = []
                briers: list[float] = []
                nlls: list[float] = []
                weights: list[int] = []
                all_y: list[np.ndarray] = []
                all_p: list[np.ndarray] = []
                for label_index in label_indices:
                    mask = valid[:, label_index]
                    y_true = labels[mask, label_index].astype(int)
                    y_prob = probs[mask, label_index]
                    if y_true.size == 0 or np.unique(y_true).size < 2:
                        continue
                    clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
                    aucs.append(float(roc_auc_score(y_true, y_prob)))
                    aps.append(float(average_precision_score(y_true, y_prob)))
                    briers.append(float(np.mean((y_prob - y_true) ** 2)))
                    nlls.append(float(log_loss(y_true, clipped, labels=[0, 1])))
                    weights.append(len(y_true))
                    all_y.append(y_true)
                    all_p.append(y_prob)
                y_flat = np.concatenate(all_y)
                p_flat = np.concatenate(all_p)
                p_clipped = np.clip(p_flat, 1e-7, 1 - 1e-7)
                rows.append(
                    {
                        "seed": seed,
                        "scope": scope,
                        "method": method,
                        "loop": loop,
                        "weighted_auroc": float(np.average(aucs, weights=weights)),
                        "macro_average_precision": float(np.mean(aps)),
                        "valid_weighted_average_precision": float(np.average(aps, weights=weights)),
                        "macro_brier": float(np.mean(briers)),
                        "micro_brier": float(np.mean((p_flat - y_flat) ** 2)),
                        "macro_nll": float(np.mean(nlls)),
                        "micro_nll": float(log_loss(y_flat, p_clipped, labels=[0, 1])),
                        "valid_test_entries": len(y_flat),
                    }
                )
    return pd.DataFrame(rows)


def summarize_heldout(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "weighted_auroc",
        "macro_average_precision",
        "valid_weighted_average_precision",
        "macro_brier",
        "micro_brier",
        "macro_nll",
        "micro_nll",
        "valid_test_entries",
    ]
    rows: list[dict] = []
    for (scope, method, loop), group in frame.groupby(["scope", "method", "loop"]):
        row = {
            "scope": scope,
            "method": method,
            "loop": int(loop),
            "n_seeds": len(group),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["scope", "method", "loop"])


def archived_dqs_with_denominators(path: Path, run_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    baseline_entries = int(frame.loc[frame["loop_id"] == 1, "n_valid_entries"].iloc[0])
    dqs = frame["official_cleanlab_entry_dqs_flattened"]
    frame["valid_entry_coverage"] = frame["n_valid_entries"] / baseline_entries
    frame["coverage_adjusted_dqs"] = dqs * frame["valid_entry_coverage"]
    frame["estimated_healthy_entries_dqs"] = dqs * frame["n_valid_entries"]
    frame["estimated_error_entries_dqs"] = (1 - dqs) * frame["n_valid_entries"]
    frame = frame.rename(columns={"oof_entry_issue_rate": "legacy_entry_issue_density_all_slots"})
    n_samples: list[int] = []
    issue_counts: list[int] = []
    for row in frame.itertuples(index=False):
        detail_path = (
            run_root
            / str(row.branch)
            / f"loop_{int(row.loop_id):02d}"
            / "oof"
            / "train_cleanlab_sample_details.csv"
        )
        details = pd.read_csv(detail_path, usecols=["est_issue_entry_count"])
        n_samples.append(len(details))
        issue_counts.append(int(details["est_issue_entry_count"].sum()))
    frame["n_samples"] = n_samples
    frame["entry_issue_count"] = issue_counts
    frame["entry_issue_rate_valid_entries"] = (
        frame["entry_issue_count"] / frame["n_valid_entries"]
    )
    return frame


def add_label_distribution_deltas(per_label: pd.DataFrame) -> pd.DataFrame:
    frozen = per_label.loc[per_label["evidence_mode"] == "frozen_initial_oof_post_action"].copy()
    baseline = frozen.loc[frozen["method"] == "baseline", [
        "seed", "label_index", "valid_entries", "positive_entries", "negative_entries", "positive_prevalence"
    ]].rename(columns={
        "valid_entries": "baseline_valid_entries",
        "positive_entries": "baseline_positive_entries",
        "negative_entries": "baseline_negative_entries",
        "positive_prevalence": "baseline_positive_prevalence",
    })
    frozen = frozen.merge(baseline, on=["seed", "label_index"], how="left")
    frozen["valid_retention"] = frozen["valid_entries"] / frozen["baseline_valid_entries"]
    frozen["positive_retention"] = frozen["positive_entries"] / frozen["baseline_positive_entries"]
    frozen["negative_retention"] = frozen["negative_entries"] / frozen["baseline_negative_entries"]
    frozen["prevalence_shift"] = frozen["positive_prevalence"] - frozen["baseline_positive_prevalence"]
    return frozen


def write_plots(
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    heldout_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    frozen = summary.loc[summary["evidence_mode"] == "frozen_initial_oof_post_action"]
    baseline = frozen.loc[frozen["method"] == "baseline"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=180)
    for method, color in [("remove", "#2b6cb0"), ("refine", "#c53030")]:
        frame = frozen.loc[frozen["method"] == method].sort_values("loop")
        axes[0].errorbar(frame["loop"], frame["dqs_flattened_mean"], yerr=frame["dqs_flattened_sd"], marker="o", capsize=4, label=method, color=color)
        axes[1].errorbar(frame["loop"], frame["coverage_adjusted_dqs_mean"], yerr=frame["coverage_adjusted_dqs_sd"], marker="o", capsize=4, label=method, color=color)
    axes[0].axhline(baseline["dqs_flattened_mean"], color="0.35", linestyle="--")
    axes[1].axhline(baseline["coverage_adjusted_dqs_mean"], color="0.35", linestyle="--")
    axes[0].set_title("Frozen-OOF DQS")
    axes[1].set_title("Coverage-adjusted DQS")
    for axis in axes:
        axis.set_xlabel("Post-action loop")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "dqs_and_coverage_adjusted_dqs.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=180)
    for method, color in [("remove", "#2b6cb0"), ("refine", "#c53030")]:
        frame = frozen.loc[frozen["method"] == method].sort_values("loop")
        axes[0].plot(frame["loop"], 100 * frame["entry_issue_density_all_slots_mean"], marker="o", label=method, color=color)
        axes[1].plot(frame["loop"], 100 * frame["entry_issue_rate_valid_entries_mean"], marker="o", label=method, color=color)
    axes[0].set_title("Legacy all-slot issue density")
    axes[1].set_title("Issue rate among valid entries")
    for axis in axes:
        axis.set_xlabel("Post-action loop")
        axis.set_ylabel("Percent")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "entry_issue_denominator_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=180)
    ax.errorbar(stability["loop"], stability["mean_pairwise_jaccard"], yerr=stability["sd_pairwise_jaccard"], marker="o", capsize=4)
    ax.set_xlabel("Cumulative removal loop")
    ax.set_ylabel("Pairwise Jaccard")
    ax.set_ylim(0, 1)
    ax.set_title("Simple-remove selection stability")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "remove_set_stability.png")
    plt.close(fig)

    full = heldout_summary.loc[heldout_summary["scope"] == "all_labels"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=180)
    metrics = [
        ("weighted_auroc_mean", "Weighted AUROC", True),
        ("macro_average_precision_mean", "Macro average precision", True),
        ("micro_brier_mean", "Micro Brier", False),
        ("micro_nll_mean", "Micro NLL", False),
    ]
    for axis, (metric, title, _) in zip(axes.flat, metrics):
        baseline_value = float(full.loc[full["method"] == "baseline", metric].iloc[0])
        axis.axhline(baseline_value, color="0.35", linestyle="--")
        for method, color in [("remove", "#2b6cb0"), ("refine", "#c53030")]:
            frame = full.loc[full["method"] == method].sort_values("loop")
            axis.plot(frame["loop"], frame[metric], marker="o", label=method, color=color)
        axis.set_title(title)
        axis.set_xlabel("Loop")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "heldout_supplementary_metrics.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--refine-source", type=Path, default=DEFAULT_REFINE_SOURCE)
    parser.add_argument("--archived-dqs", type=Path, default=DEFAULT_ARCHIVED_DQS)
    parser.add_argument("--archived-xrv-root", type=Path, default=DEFAULT_ARCHIVED_XRV_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 42, 97, 123])
    parser.add_argument("--loops", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = sorted(args.seeds)
    loops = sorted(args.loops)
    worker_args = [(str(args.root), str(args.refine_source), seed, loops) for seed in seeds]
    if args.workers == 1:
        results = [process_seed(*item) for item in worker_args]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(seeds))) as executor:
            futures = [executor.submit(process_seed, *item) for item in worker_args]
            results = [future.result() for future in futures]

    metrics = pd.DataFrame([row for result in results for row in result[0]])
    per_label = pd.DataFrame([row for result in results for row in result[1]])
    actions = pd.DataFrame([row for result in results for row in result[2]])
    quality_summary = summarize_quality(metrics)
    quality_stats = paired_statistics(
        metrics,
        loops,
        {
            "dqs_flattened": 1,
            "dqs_per_label_weighted": 1,
            "coverage_adjusted_dqs": 1,
            "sample_issue_rate": -1,
            "entry_issue_density_all_slots": -1,
            "entry_issue_rate_valid_entries": -1,
        },
        "frozen_initial_oof_post_action",
    )
    stability_pairs, stability_summary = removal_stability(args.root, seeds, loops)
    rank_stability = ranking_stability(args.root, seeds)
    distribution = add_label_distribution_deltas(per_label)
    heldout = heldout_metrics(args.root, seeds, loops)
    heldout_summary = summarize_heldout(heldout)
    heldout_stats = []
    for scope, scope_frame in heldout.groupby("scope"):
        renamed = scope_frame.rename(columns={"scope": "evidence_mode"})
        stats_frame = paired_statistics(
            renamed,
            loops,
            {
                "weighted_auroc": 1,
                "macro_average_precision": 1,
                "valid_weighted_average_precision": 1,
                "micro_brier": -1,
                "micro_nll": -1,
            },
            scope,
        )
        stats_frame = stats_frame.rename(columns={"evidence_mode": "scope"})
        heldout_stats.append(stats_frame)
    heldout_stats_frame = pd.concat(heldout_stats, ignore_index=True)
    archived = archived_dqs_with_denominators(args.archived_dqs, args.archived_xrv_root)

    metrics.to_csv(args.out_dir / "oof_quality_metrics_per_seed.csv", index=False)
    quality_summary.to_csv(args.out_dir / "oof_quality_summary.csv", index=False)
    quality_stats.to_csv(args.out_dir / "oof_quality_paired_statistics.csv", index=False)
    per_label.to_csv(args.out_dir / "oof_quality_per_label.csv", index=False)
    distribution.to_csv(args.out_dir / "label_distribution_shift.csv", index=False)
    actions.to_csv(args.out_dir / "llm_action_oof_support.csv", index=False)
    stability_pairs.to_csv(args.out_dir / "remove_set_stability_pairs.csv", index=False)
    stability_summary.to_csv(args.out_dir / "remove_set_stability_summary.csv", index=False)
    rank_stability.to_csv(args.out_dir / "oof_ranking_stability.csv", index=False)
    heldout.to_csv(args.out_dir / "heldout_supplementary_metrics_per_seed.csv", index=False)
    heldout_summary.to_csv(args.out_dir / "heldout_supplementary_metrics_summary.csv", index=False)
    heldout_stats_frame.to_csv(args.out_dir / "heldout_supplementary_paired_statistics.csv", index=False)
    archived.to_csv(args.out_dir / "archived_xrv_dqs_with_denominators.csv", index=False)

    metadata = {
        "seeds": seeds,
        "loops": loops,
        "cleanlab_dqs_definition": "overall_label_health_score over flattened valid binary label entries",
        "legacy_entry_noise_definition": "issue_count / (n_samples * n_labels)",
        "corrected_entry_noise_definition": "issue_count / n_valid_entries",
        "sample_issue_rate_definition": "samples with any confident-learning issue / samples retaining at least one valid target label",
        "sample_coverage_definition": "samples retaining at least one valid target label / baseline samples with at least one valid target label",
        "sample_zero_valid_policy": "samples with zero valid target labels are unevaluable, not issue-free",
        "coverage_adjusted_dqs_definition": "dqs_flattened * n_valid_entries / baseline_n_valid_entries",
        "coverage_adjusted_dqs_status": "custom denominator diagnostic, not an official Cleanlab metric",
        "dqs_issue_rate_relation": "1 - DQS and find_label_issues count / denominator are different Cleanlab summaries and need not match",
        "frozen_evidence_definition": "post-action labels evaluated with each seed's unchanged loop1 OOF probabilities",
        "dynamic_evidence_definition": "iterative simple-remove OOF probabilities before the action of the named loop",
        "llm_dynamic_five_seed_oof_available": False,
    }
    (args.out_dir / "data_quality_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    write_plots(quality_summary, stability_summary, heldout_summary, args.out_dir)

    print(f"Wrote data-quality evaluation to {args.out_dir}")
    print(json.dumps(metadata, indent=2))
    print("\nFrozen OOF quality summary:")
    print(
        quality_summary.loc[
            quality_summary["evidence_mode"] == "frozen_initial_oof_post_action",
            [
                "method",
                "loop",
                "dqs_flattened_mean",
                "coverage_adjusted_dqs_mean",
                "sample_issue_rate_mean",
                "entry_issue_density_all_slots_mean",
                "entry_issue_rate_valid_entries_mean",
            ],
        ].to_string(index=False)
    )
    print("\nRemoval stability:")
    print(stability_summary.to_string(index=False))
    print("\nLLM action support:")
    print(actions.groupby("loop").mean(numeric_only=True).reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
