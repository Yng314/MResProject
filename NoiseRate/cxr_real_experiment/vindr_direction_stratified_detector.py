#!/usr/bin/env python3
"""Exploratory direction-stratified detector on the locked VinDr benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vindr_detector_alternatives_benchmark import (
    active_label_cleaning_score,
    class_conditional_rank,
    evidence_to_matrix,
    leave_self_out_neighbours,
    load_features,
    simifeat_style_scores,
    validate_no_outcome_columns,
)
from vindr_known_gt_cl_benchmark import (
    atomic_write_csv,
    atomic_write_text,
    exact_sign_flip_p,
    require_columns,
    sha256_file,
)


PROTOCOL_NAME = "vindr_direction_stratified_detector_exploratory_v1"
SEEDS = (13, 42, 97, 123, 211, 307)
FIXED_NEGATIVE_QUOTAS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0)


def atomic_write_json(payload: dict, path: Path) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def stable_top(frame: pd.DataFrame, score: str, budget: int) -> np.ndarray:
    if budget < 0 or budget > len(frame):
        raise ValueError("Budget is outside the candidate queue")
    ranked = frame.sort_values(
        [score, "image_id", "label_name"],
        ascending=[False, True, True],
    )
    return ranked.index.to_numpy()[:budget]


def direction_stratified_selection(
    frame: pd.DataFrame,
    budget: int,
    negative_quota: float,
    positive_score: str = "score_oof_label_incompatibility",
    negative_score: str = "score_simifeat_style_k50",
) -> tuple[np.ndarray, int, int]:
    if not 0.0 <= negative_quota <= 1.0:
        raise ValueError("negative_quota must lie in [0, 1]")
    negative = frame[frame["noisy_label"] == 0]
    positive = frame[frame["noisy_label"] == 1]
    negative_budget = min(int(round(budget * negative_quota)), len(negative))
    positive_budget = min(budget - negative_budget, len(positive))
    if negative_budget + positive_budget < budget:
        shortfall = budget - negative_budget - positive_budget
        extra_negative = min(shortfall, len(negative) - negative_budget)
        negative_budget += extra_negative
        shortfall -= extra_negative
        positive_budget += min(shortfall, len(positive) - positive_budget)
    selected = np.concatenate(
        [
            stable_top(negative, negative_score, negative_budget),
            stable_top(positive, positive_score, positive_budget),
        ]
    )
    if len(selected) != budget or len(np.unique(selected)) != budget:
        raise RuntimeError("Direction-stratified selection did not meet the budget")
    return selected, negative_budget, positive_budget


def load_manifest(parent: Path) -> pd.DataFrame:
    manifest = pd.read_csv(parent / "blind_run_manifest.csv")
    require_columns(
        manifest,
        [
            "array_index",
            "scenario_id",
            "regime",
            "noise_rate",
            "rate_percent",
            "seed",
            "seed_block",
            "prepared_relpath",
            "blind_run_relpath",
        ],
        "blind run manifest",
    )
    manifest = manifest.sort_values("array_index").reset_index(drop=True)
    if len(manifest) != 60 or manifest["seed"].drop_duplicates().tolist() != list(SEEDS):
        raise ValueError("Expected the locked 60-run, six-seed direction benchmark")
    return manifest


def score_run(
    evidence_path: Path,
    output_path: Path,
    feature_image_ids: np.ndarray,
    neighbours: np.ndarray,
    distances: np.ndarray,
) -> None:
    evidence = pd.read_csv(evidence_path)
    ordered, noisy = evidence_to_matrix(evidence, feature_image_ids)
    raw = simifeat_style_scores(noisy, neighbours, distances, [50])[50].reshape(-1)
    output = ordered[
        ["image_id", "fold_id", "label_index", "label_name", "noisy_label"]
    ].copy()
    probability = ordered["oof_probability"].to_numpy(float)
    noisy_flat = ordered["noisy_label"].to_numpy(int)
    output["score_oof_label_incompatibility"] = ordered["cl_first_score"].to_numpy(float)
    output["score_active_label_cleaning"] = active_label_cleaning_score(
        noisy_flat, probability
    )
    output["score_simifeat_raw_k50"] = raw
    output["score_simifeat_style_k50"] = class_conditional_rank(
        output, "score_simifeat_raw_k50"
    )
    output["oof_alternative_probability"] = np.where(
        noisy_flat == 1, 1.0 - probability, probability
    )
    output["simifeat_disagreement_probability"] = 1.0 - np.exp(-raw)
    validate_no_outcome_columns(output, "blind direction-stratified scores")
    if not np.isfinite(output.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("Blind score table contains non-finite values")
    atomic_write_csv(output, output_path)


def score(args: argparse.Namespace) -> None:
    parent = Path(args.parent_root)
    output = Path(args.output_root)
    features_path = Path(args.features_npz)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {output}")
    if not (parent / ".benchmark_verified").is_file():
        raise FileNotFoundError("Parent benchmark verification marker is missing")
    manifest = load_manifest(parent)
    image_ids, features = load_features(features_path)
    neighbours, distances = leave_self_out_neighbours(features, 50, args.n_jobs)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    records = []
    try:
        for row in manifest.itertuples(index=False):
            run = parent / row.blind_run_relpath
            if not (run / ".blind_run_complete").is_file():
                raise FileNotFoundError(f"Blind run is incomplete: {run}")
            relative = Path("blind_scores") / row.scenario_id / f"seed_{int(row.seed)}.csv"
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            score_run(
                run / "entry_evidence.csv",
                destination,
                image_ids,
                neighbours,
                distances,
            )
            records.append(
                {
                    "array_index": int(row.array_index),
                    "scenario_id": row.scenario_id,
                    "regime": row.regime,
                    "noise_rate": float(row.noise_rate),
                    "rate_percent": int(row.rate_percent),
                    "seed": int(row.seed),
                    "seed_block": row.seed_block,
                    "prepared_relpath": row.prepared_relpath,
                    "blind_run_relpath": row.blind_run_relpath,
                    "score_relpath": str(relative),
                    "score_sha256": sha256_file(destination),
                }
            )
            print(f"scored scenario={row.scenario_id} seed={int(row.seed)}", flush=True)
        atomic_write_csv(pd.DataFrame(records), temporary / "score_manifest.csv")
        atomic_write_json(
            {
                "protocol": PROTOCOL_NAME,
                "outcome_blind": True,
                "runs": len(records),
                "features_sha256": sha256_file(features_path),
                "parent_manifest_sha256": sha256_file(parent / "blind_run_manifest.csv"),
                "program_sha256": sha256_file(Path(__file__)),
            },
            temporary / "scoring_summary.json",
        )
        atomic_write_text(temporary / ".blind_scoring_complete", "complete\n")
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def error_metrics(
    merged: pd.DataFrame,
    selected: np.ndarray,
    metadata: dict,
    method: str,
    negative_quota: float,
    reviewed_negative: int,
    reviewed_positive: int,
) -> dict:
    chosen = merged.loc[selected]
    true_errors = int(merged["injected_error"].sum())
    errors_found = int(chosen["injected_error"].sum())
    row = {
        **metadata,
        "method": method,
        "negative_quota": negative_quota,
        "reviewed_entries": len(chosen),
        "reviewed_negative": reviewed_negative,
        "reviewed_positive": reviewed_positive,
        "true_errors": true_errors,
        "errors_found": errors_found,
        "recall": errors_found / true_errors,
        "precision": errors_found / len(chosen),
        "random_expected_recall": len(chosen) / len(merged),
    }
    recalls = []
    for direction in ["0_to_1", "1_to_0"]:
        total = int(merged["flip_direction"].eq(direction).sum())
        found = int(chosen["flip_direction"].eq(direction).sum())
        row[f"true_errors_{direction}"] = total
        row[f"errors_found_{direction}"] = found
        row[f"recall_{direction}"] = found / total if total else np.nan
        if total:
            recalls.append(found / total)
    row["macro_direction_recall"] = float(np.mean(recalls))
    row["min_direction_recall"] = float(np.min(recalls))
    return row


def evaluate_run(scores: pd.DataFrame, private: pd.DataFrame, metadata: dict) -> list[dict]:
    merged = scores.merge(
        private,
        on=["image_id", "label_name"],
        suffixes=("_blind", "_private"),
        validate="one_to_one",
    )
    if len(merged) != 18_000 or not merged["noisy_label_blind"].eq(
        merged["noisy_label_private"]
    ).all():
        raise ValueError("Blind/private alignment failed")
    merged = merged.rename(columns={"noisy_label_blind": "noisy_label"})
    budget = int(merged["injected_error"].sum())
    records = []
    for method, score_column in [
        ("oof_global", "score_oof_label_incompatibility"),
        ("alc_global", "score_active_label_cleaning"),
        ("simifeat_global", "score_simifeat_style_k50"),
    ]:
        selected = stable_top(merged, score_column, budget)
        chosen = merged.loc[selected]
        records.append(
            error_metrics(
                merged,
                selected,
                metadata,
                method,
                float(chosen["noisy_label"].eq(0).mean()),
                int(chosen["noisy_label"].eq(0).sum()),
                int(chosen["noisy_label"].eq(1).sum()),
            )
        )
    for quota in FIXED_NEGATIVE_QUOTAS:
        selected, negative_budget, positive_budget = direction_stratified_selection(
            merged, budget, quota
        )
        records.append(
            error_metrics(
                merged,
                selected,
                metadata,
                f"hybrid_fixed_q{int(round(quota * 100)):03d}",
                quota,
                negative_budget,
                positive_budget,
            )
        )
    for name, mass_column in [
        ("hybrid_oof_mass", "oof_alternative_probability"),
        ("hybrid_simifeat_mass", "simifeat_disagreement_probability"),
    ]:
        mass = merged.groupby("noisy_label")[mass_column].sum()
        quota = float(mass.get(0, 0.0) / mass.sum())
        selected, negative_budget, positive_budget = direction_stratified_selection(
            merged, budget, quota
        )
        records.append(
            error_metrics(
                merged,
                selected,
                metadata,
                name,
                quota,
                negative_budget,
                positive_budget,
            )
        )
    return records


def paired_contrast(metrics: pd.DataFrame, method: str, regime: str, value: str) -> dict:
    frame = metrics[metrics["regime"].eq(regime)]
    pivot = frame[frame["method"].isin(["oof_global", method])].pivot_table(
        index=["seed", "rate_percent"], columns="method", values=value
    )
    seed_differences = (
        pivot[method] - pivot["oof_global"]
    ).groupby(level="seed").mean()
    nonzero = seed_differences[seed_differences.ne(0)].to_numpy(float)
    return {
        "method": method,
        "reference": "oof_global",
        "regime": regime,
        "metric": value,
        "mean_difference": float(seed_differences.mean()),
        "positive_seeds": int(seed_differences.gt(0).sum()),
        "negative_seeds": int(seed_differences.lt(0).sum()),
        "tied_seeds": int(seed_differences.eq(0).sum()),
        "exact_sign_flip_p": float(exact_sign_flip_p(nonzero)) if len(nonzero) else 1.0,
        "post_hoc_exploratory": True,
    }


def make_figure(metrics: pd.DataFrame, path: Path) -> None:
    fixed = metrics[metrics["method"].str.startswith("hybrid_fixed_")]
    summary = fixed.groupby(["regime", "negative_quota"], as_index=False).agg(
        recall=("recall", "mean"),
        recall_sd=("recall", "std"),
        recall_fp=("recall_0_to_1", "mean"),
        recall_fn=("recall_1_to_0", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"balanced": "#5B4B8A", "fp_only": "#C84B31", "fn_only": "#287271"}
    for regime in ["balanced", "fp_only", "fn_only"]:
        frame = summary[summary["regime"].eq(regime)]
        axes[0].plot(
            100 * frame["negative_quota"], frame["recall"], marker="o",
            label=regime, color=colors[regime],
        )
    balanced = summary[summary["regime"].eq("balanced")]
    axes[1].plot(
        100 * balanced["negative_quota"], balanced["recall_fp"],
        marker="o", label="0 to 1 recall", color="#C84B31",
    )
    axes[1].plot(
        100 * balanced["negative_quota"], balanced["recall_fn"],
        marker="o", label="1 to 0 recall", color="#287271",
    )
    axes[1].plot(
        100 * balanced["negative_quota"], balanced["recall"],
        marker="o", label="total recall", color="#5B4B8A",
    )
    axes[0].set_title("Total error recall by corruption regime")
    axes[1].set_title("Balanced corruption: directional trade-off")
    for axis in axes:
        axis.set_xlabel("Review budget assigned to observed-negative entries (%)")
        axis.set_ylabel("Recall at matched total review budget")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> None:
    parent = Path(args.parent_root)
    output = Path(args.output_root)
    if not (output / ".blind_scoring_complete").is_file():
        raise FileNotFoundError("Blind scoring completion marker is missing")
    evaluation = output / "evaluation"
    if evaluation.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {evaluation}")
    evaluation.mkdir()
    manifest = pd.read_csv(output / "score_manifest.csv")
    records = []
    for row in manifest.itertuples(index=False):
        if row.regime == "clean":
            continue
        scores = pd.read_csv(output / row.score_relpath)
        private = pd.read_csv(parent / row.prepared_relpath / "private_reference.csv")
        metadata = {
            "scenario_id": row.scenario_id,
            "regime": row.regime,
            "noise_rate": float(row.noise_rate),
            "rate_percent": int(row.rate_percent),
            "seed": int(row.seed),
            "seed_block": row.seed_block,
        }
        records.extend(evaluate_run(scores, private, metadata))
    metrics = pd.DataFrame(records)
    atomic_write_csv(metrics, evaluation / "matched_budget_metrics.csv")
    aggregate = metrics.groupby(
        ["regime", "rate_percent", "method"], as_index=False
    ).agg(
        recall_mean=("recall", "mean"),
        recall_sd=("recall", "std"),
        precision_mean=("precision", "mean"),
        recall_0_to_1_mean=("recall_0_to_1", "mean"),
        recall_1_to_0_mean=("recall_1_to_0", "mean"),
        macro_direction_recall_mean=("macro_direction_recall", "mean"),
        negative_quota_mean=("negative_quota", "mean"),
    )
    atomic_write_csv(aggregate, evaluation / "aggregate_metrics.csv")
    contrasts = []
    for method in ["hybrid_oof_mass", "hybrid_simifeat_mass", "hybrid_fixed_q050"]:
        for regime in ["balanced", "fp_only", "fn_only"]:
            contrasts.append(paired_contrast(metrics, method, regime, "recall"))
    atomic_write_csv(pd.DataFrame(contrasts), evaluation / "paired_contrasts.csv")
    fixed_balanced = metrics[
        metrics["regime"].eq("balanced")
        & metrics["method"].str.startswith("hybrid_fixed_")
    ]
    diagnostic = fixed_balanced.groupby(
        ["method", "negative_quota"], as_index=False
    ).agg(
        recall=("recall", "mean"),
        macro_direction_recall=("macro_direction_recall", "mean"),
        min_direction_recall=("min_direction_recall", "mean"),
    )
    best_total = diagnostic.loc[diagnostic["recall"].idxmax()].to_dict()
    best_guardrail = diagnostic.loc[diagnostic["min_direction_recall"].idxmax()].to_dict()
    make_figure(metrics, evaluation / "direction_quota_tradeoff.png")
    summary = {
        "protocol": PROTOCOL_NAME,
        "runs_evaluated": int(metrics[["scenario_id", "seed"]].drop_duplicates().shape[0]),
        "methods": sorted(metrics["method"].unique().tolist()),
        "fixed_quota_diagnostic_only": True,
        "best_balanced_total_recall_post_hoc": best_total,
        "best_balanced_min_direction_recall_post_hoc": best_guardrail,
        "program_sha256": sha256_file(Path(__file__)),
        "metrics_sha256": sha256_file(evaluation / "matched_budget_metrics.csv"),
    }
    atomic_write_json(summary, evaluation / "evaluation_summary.json")
    atomic_write_text(evaluation / ".evaluation_complete", "complete\n")


def verify(args: argparse.Namespace) -> None:
    output = Path(args.output_root)
    manifest = pd.read_csv(output / "score_manifest.csv")
    metrics = pd.read_csv(output / "evaluation" / "matched_budget_metrics.csv")
    if len(manifest) != 60 or len(metrics) != 54 * (3 + len(FIXED_NEGATIVE_QUOTAS) + 2):
        raise RuntimeError("Formal row-count verification failed")
    for row in manifest.itertuples(index=False):
        path = output / row.score_relpath
        if not path.is_file() or sha256_file(path) != row.score_sha256:
            raise RuntimeError(f"Blind score hash failed: {path}")
        validate_no_outcome_columns(pd.read_csv(path), "blind score")
    if not np.isfinite(metrics[["recall", "precision", "negative_quota"]].to_numpy()).all():
        raise RuntimeError("Evaluation metrics contain non-finite primary values")
    atomic_write_text(output / ".benchmark_verified", "verified\n")
    print(json.dumps({"verification": "passed", "runs": 60, "evaluation_rows": len(metrics)}, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)
    for command in ["score", "evaluate", "verify"]:
        child = subparsers.add_parser(command)
        child.add_argument("--parent-root", required=True)
        child.add_argument("--output-root", required=True)
        if command == "score":
            child.add_argument("--features-npz", required=True)
            child.add_argument("--n-jobs", type=int, default=4)
    return result


def main() -> None:
    args = parser().parse_args()
    {"score": score, "evaluate": evaluate, "verify": verify}[args.command](args)


if __name__ == "__main__":
    main()
