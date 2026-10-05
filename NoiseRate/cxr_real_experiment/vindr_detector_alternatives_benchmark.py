#!/usr/bin/env python3
"""Outcome-blind detector comparison on the VinDr symmetric-noise benchmark."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from vindr_known_gt_cl_benchmark import (
    LABELS,
    atomic_write_csv,
    atomic_write_text,
    exact_sign_flip_p,
    require_columns,
    sha256_file,
)


PROTOCOL_NAME = "vindr_detector_alternatives_v1"
FORBIDDEN_BLIND_TOKENS = ("clean_label", "reference", "injected", "true_", "error")
DEFAULT_K_VALUES = (10, 20, 50)
PRIMARY_METHODS = (
    "oof_label_incompatibility",
    "active_label_cleaning",
    "simifeat_style_k10",
    "oof_simifeat_rank_fusion",
)


def parse_int_list(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        parsed = [int(token.strip()) for token in value.split(",") if token.strip()]
    else:
        parsed = [int(item) for item in value]
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("Expected a non-empty list of unique integers")
    return parsed


def atomic_write_json(payload: dict, path: Path) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def validate_no_outcome_columns(frame: pd.DataFrame, name: str) -> None:
    forbidden = [
        column
        for column in frame.columns
        if any(token in column.lower() for token in FORBIDDEN_BLIND_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"{name} contains private outcome columns: {forbidden}")


def active_label_cleaning_score(noisy_label: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """Bernhardt et al. score: label cross-entropy minus predictive entropy."""
    y = np.asarray(noisy_label, dtype=np.int64)
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1.0 - 1e-7)
    assigned_nll = -(y * np.log(p) + (1 - y) * np.log(1.0 - p))
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return assigned_nll - entropy


def leave_self_out_neighbours(
    features: np.ndarray,
    max_k: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = NearestNeighbors(
        n_neighbors=max_k + 1,
        metric="cosine",
        algorithm="brute",
        n_jobs=n_jobs,
    )
    model.fit(features)
    distances, indices = model.kneighbors(features, return_distance=True)
    clean_indices = np.empty((len(features), max_k), dtype=np.int64)
    clean_distances = np.empty((len(features), max_k), dtype=np.float32)
    for row in range(len(features)):
        keep = indices[row] != row
        row_indices = indices[row][keep][:max_k]
        row_distances = distances[row][keep][:max_k]
        if len(row_indices) != max_k:
            raise RuntimeError("Could not construct leave-self-out neighbours")
        clean_indices[row] = row_indices
        clean_distances[row] = row_distances
    return clean_indices, clean_distances


def simifeat_style_scores(
    noisy: np.ndarray,
    neighbour_indices: np.ndarray,
    neighbour_distances: np.ndarray,
    k_values: Iterable[int],
) -> dict[int, np.ndarray]:
    """SimiFeat-style feature-neighbour cross-entropy, adapted to binary entries."""
    labels = np.asarray(noisy, dtype=np.int64)
    if labels.ndim != 2 or labels.shape[1] != len(LABELS):
        raise ValueError("Expected an N x 6 noisy-label matrix")
    outputs: dict[int, np.ndarray] = {}
    for k in k_values:
        neighbours = neighbour_indices[:, :k]
        similarities = np.clip(1.0 - neighbour_distances[:, :k], 1e-6, None)
        denominator = similarities.sum(axis=1)
        score = np.empty_like(labels, dtype=np.float64)
        for label_index in range(labels.shape[1]):
            neighbour_labels = labels[neighbours, label_index]
            probability_one = (similarities * neighbour_labels).sum(axis=1) / denominator
            assigned_support = np.where(
                labels[:, label_index] == 1,
                probability_one,
                1.0 - probability_one,
            )
            score[:, label_index] = -np.log(np.clip(assigned_support, 1e-7, 1.0))
        outputs[int(k)] = score
    return outputs


def class_conditional_rank(frame: pd.DataFrame, score_column: str) -> pd.Series:
    return frame.groupby(["label_name", "noisy_label"], sort=False)[score_column].rank(
        method="average", pct=True
    )


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    archive = np.load(path, allow_pickle=False)
    if set(archive.files) != {"image_id", "features"}:
        raise ValueError(f"Unexpected feature archive keys: {archive.files}")
    image_ids = archive["image_id"].astype(str)
    features = archive["features"].astype(np.float32)
    if features.shape != (3_000, 1_024) or len(set(image_ids)) != 3_000:
        raise ValueError("Expected 3,000 unique VinDr images with 1,024-dimensional features")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(features).all() or np.any(norms <= 0):
        raise ValueError("Features are non-finite or contain zero vectors")
    return image_ids, features / norms


def ordered_manifest(args: argparse.Namespace) -> pd.DataFrame:
    manifest_path = Path(args.parent_root) / "scenario_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    require_columns(
        manifest,
        [
            "array_index",
            "scenario_id",
            "noise_rate",
            "rate_percent",
            "seed",
            "prepared_relpath",
            "blind_run_relpath",
        ],
        "scenario manifest",
    )
    manifest = manifest.sort_values("array_index").reset_index(drop=True)
    if args.seeds:
        manifest = manifest[manifest["seed"].isin(parse_int_list(args.seeds))]
    if args.scenarios:
        requested = {token.strip() for token in args.scenarios.split(",") if token.strip()}
        manifest = manifest[manifest["scenario_id"].isin(requested)]
    if manifest.empty:
        raise ValueError("The selected manifest is empty")
    return manifest.reset_index(drop=True)


def evidence_to_matrix(
    evidence: pd.DataFrame,
    feature_image_ids: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    require_columns(
        evidence,
        [
            "image_id",
            "fold_id",
            "label_index",
            "label_name",
            "noisy_label",
            "oof_probability",
            "cl_first_score",
        ],
        "entry evidence",
    )
    validate_no_outcome_columns(evidence, "entry evidence")
    evidence = evidence.copy()
    evidence["image_id"] = evidence["image_id"].astype(str)
    if len(evidence) != 18_000 or evidence[["image_id", "label_name"]].duplicated().any():
        raise ValueError("Expected 18,000 unique image-label evidence rows")
    label_map = {label: index for index, label in enumerate(LABELS)}
    if set(evidence["label_name"]) != set(LABELS):
        raise ValueError("Entry evidence does not contain the six locked labels")
    if not evidence["noisy_label"].isin([0, 1]).all():
        raise ValueError("Noisy labels must be binary")
    order = pd.MultiIndex.from_product(
        [feature_image_ids.tolist(), LABELS], names=["image_id", "label_name"]
    )
    ordered = evidence.set_index(["image_id", "label_name"]).reindex(order).reset_index()
    if ordered.isna().any().any():
        raise ValueError("Features and entry evidence do not align")
    ordered["label_index"] = ordered["label_name"].map(label_map)
    noisy = ordered["noisy_label"].to_numpy(dtype=np.int64).reshape(3_000, len(LABELS))
    return ordered, noisy


def score_one_run(
    evidence_path: Path,
    output_path: Path,
    feature_image_ids: np.ndarray,
    neighbour_indices: np.ndarray,
    neighbour_distances: np.ndarray,
    k_values: list[int],
) -> dict:
    evidence = pd.read_csv(evidence_path)
    ordered, noisy = evidence_to_matrix(evidence, feature_image_ids)
    neighbour_scores = simifeat_style_scores(
        noisy, neighbour_indices, neighbour_distances, k_values
    )
    output = ordered[
        ["image_id", "fold_id", "label_index", "label_name", "noisy_label"]
    ].copy()
    output["score_oof_label_incompatibility"] = ordered["cl_first_score"].to_numpy(float)
    output["score_active_label_cleaning"] = active_label_cleaning_score(
        ordered["noisy_label"].to_numpy(), ordered["oof_probability"].to_numpy()
    )
    for k in k_values:
        raw_column = f"simifeat_raw_k{k}"
        score_column = f"score_simifeat_style_k{k}"
        output[raw_column] = neighbour_scores[k].reshape(-1)
        output[score_column] = class_conditional_rank(output, raw_column)
    output["oof_global_rank"] = output["score_oof_label_incompatibility"].rank(
        method="average", pct=True
    )
    output["score_oof_simifeat_rank_fusion"] = 0.5 * (
        output["oof_global_rank"] + output["score_simifeat_style_k10"]
    )
    output = output.drop(columns=[f"simifeat_raw_k{k}" for k in k_values] + ["oof_global_rank"])
    validate_no_outcome_columns(output, "blind detector scores")
    if not np.isfinite(output.filter(like="score_").to_numpy()).all():
        raise ValueError("One or more detector scores are non-finite")
    atomic_write_csv(output, output_path)
    return {
        "rows": int(len(output)),
        "score_sha256": sha256_file(output_path),
        "entry_evidence_sha256": sha256_file(evidence_path),
    }


def score(args: argparse.Namespace) -> None:
    parent = Path(args.parent_root)
    output = Path(args.output_root)
    features_path = Path(args.features_npz)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {output}")
    if not (parent / ".benchmark_verified").is_file():
        raise FileNotFoundError("Parent VinDr benchmark is not verified")
    if not features_path.parent.joinpath(".features_complete").is_file():
        raise FileNotFoundError("Frozen XRV feature marker is missing")
    manifest = ordered_manifest(args)
    image_ids, features = load_features(features_path)
    k_values = parse_int_list(args.k_values)
    if 10 not in k_values or min(k_values) < 1 or max(k_values) >= len(features):
        raise ValueError("k values must include the locked primary k=10")

    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        neighbours, distances = leave_self_out_neighbours(features, max(k_values), args.n_jobs)
        run_records = []
        for row in manifest.itertuples(index=False):
            run_root = parent / row.blind_run_relpath
            if not (run_root / ".blind_run_complete").is_file():
                raise FileNotFoundError(f"Blind OOF run is incomplete: {run_root}")
            relative = Path("blind_scores") / str(row.scenario_id) / f"seed_{int(row.seed)}.csv"
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            record = score_one_run(
                run_root / "entry_evidence.csv",
                destination,
                image_ids,
                neighbours,
                distances,
                k_values,
            )
            record.update(
                {
                    "array_index": int(row.array_index),
                    "scenario_id": str(row.scenario_id),
                    "noise_rate": float(row.noise_rate),
                    "rate_percent": int(row.rate_percent),
                    "seed": int(row.seed),
                    "prepared_relpath": str(row.prepared_relpath),
                    "score_relpath": str(relative),
                }
            )
            run_records.append(record)
            print(f"scored scenario={row.scenario_id} seed={int(row.seed)}", flush=True)

        score_manifest = pd.DataFrame(run_records).sort_values("array_index")
        atomic_write_csv(score_manifest, temporary / "score_manifest.csv")
        summary = {
            "protocol": PROTOCOL_NAME,
            "outcome_blind": True,
            "runs": int(len(score_manifest)),
            "images": 3_000,
            "entries_per_run": 18_000,
            "k_values": k_values,
            "primary_simifeat_k": 10,
            "neighbour_policy": "cosine similarity, leave-self-out",
            "simifeat_adaptation": "binary per-finding neighbour CE with class-conditional ranking",
            "parent_root": str(parent),
            "parent_manifest_sha256": sha256_file(parent / "scenario_manifest.csv"),
            "features_npz": str(features_path),
            "features_sha256": sha256_file(features_path),
            "program_sha256": sha256_file(Path(__file__)),
            "score_manifest_sha256": sha256_file(temporary / "score_manifest.csv"),
        }
        atomic_write_json(summary, temporary / "scoring_summary.json")
        atomic_write_text(temporary / ".blind_scoring_complete", "complete\n")
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def score_columns(frame: pd.DataFrame) -> dict[str, str]:
    mapping = {
        "oof_label_incompatibility": "score_oof_label_incompatibility",
        "active_label_cleaning": "score_active_label_cleaning",
        "simifeat_style_k10": "score_simifeat_style_k10",
        "simifeat_style_k20": "score_simifeat_style_k20",
        "simifeat_style_k50": "score_simifeat_style_k50",
        "oof_simifeat_rank_fusion": "score_oof_simifeat_rank_fusion",
    }
    missing = sorted(set(mapping.values()) - set(frame.columns))
    if missing:
        raise ValueError(f"Blind scores are missing methods: {missing}")
    return mapping


def merge_private(scores: pd.DataFrame, private_path: Path) -> pd.DataFrame:
    validate_no_outcome_columns(scores, "blind detector scores")
    private = pd.read_csv(private_path)
    require_columns(
        private,
        ["image_id", "label_name", "clean_label", "noisy_label", "injected_error", "flip_direction"],
        "private reference",
    )
    private["image_id"] = private["image_id"].astype(str)
    merged = scores.merge(
        private,
        on=["image_id", "label_name"],
        how="inner",
        validate="one_to_one",
        suffixes=("_blind", "_private"),
    )
    if len(merged) != 18_000:
        raise ValueError("Blind scores and private reference do not align")
    if not np.array_equal(
        merged["noisy_label_blind"].to_numpy(), merged["noisy_label_private"].to_numpy()
    ):
        raise ValueError("Noisy labels differ between blind and private files")
    return merged


def stable_top_indices(frame: pd.DataFrame, score_column: str, budget: int) -> np.ndarray:
    ranked = frame.sort_values(
        [score_column, "image_id", "label_name"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    return ranked.index.to_numpy()[:budget]


def ranking_auc(truth: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    positives = int(truth.sum())
    if positives == 0:
        return float("nan")
    recall = np.concatenate([[0.0], np.cumsum(truth[order]) / positives])
    review_fraction = np.arange(len(truth) + 1, dtype=float) / len(truth)
    return float(np.trapezoid(recall, review_fraction))


def evaluate_run(
    merged: pd.DataFrame,
    method_columns: dict[str, str],
    metadata: dict,
) -> tuple[list[dict], list[dict], list[dict]]:
    truth = merged["injected_error"].to_numpy(dtype=bool)
    prevalence = float(truth.mean())
    true_errors = int(truth.sum())
    if true_errors == 0:
        return [], [], []
    overall_records = []
    budget_records = []
    label_records = []
    fixed_budgets = [(f"fixed_{int(fraction * 100):02d}pct", int(round(fraction * len(merged)))) for fraction in [0.05, 0.10, 0.20, 0.30]]
    fixed_budgets.append(("true_error_count", true_errors))

    for method, column in method_columns.items():
        score = merged[column].to_numpy(dtype=float)
        overall_records.append(
            {
                **metadata,
                "method": method,
                "entries": int(len(merged)),
                "true_errors": true_errors,
                "error_prevalence": prevalence,
                "auprc": float(average_precision_score(truth, score)),
                "auroc": float(roc_auc_score(truth, score)),
                "review_recall_auc": ranking_auc(truth, score),
            }
        )
        for budget_name, budget in fixed_budgets:
            selected = stable_top_indices(merged, column, budget)
            found = int(truth[selected].sum())
            expected = float(budget * prevalence)
            budget_records.append(
                {
                    **metadata,
                    "method": method,
                    "budget_name": budget_name,
                    "reviewed_entries": int(budget),
                    "review_fraction": float(budget / len(merged)),
                    "true_errors": true_errors,
                    "errors_found": found,
                    "precision": float(found / budget),
                    "recall": float(found / true_errors),
                    "random_expected_errors": expected,
                    "enrichment_over_random": float(found / expected) if expected else float("nan"),
                }
            )
        for label in LABELS:
            subset = merged[merged["label_name"] == label]
            subset_truth = subset["injected_error"].to_numpy(dtype=bool)
            subset_errors = int(subset_truth.sum())
            if subset_errors == 0:
                continue
            selected = stable_top_indices(subset, column, subset_errors)
            found = int(subset.loc[selected, "injected_error"].sum())
            label_records.append(
                {
                    **metadata,
                    "method": method,
                    "label_name": label,
                    "true_errors": subset_errors,
                    "auprc": float(average_precision_score(subset_truth, subset[column].to_numpy(float))),
                    "recall_at_label_error_count": float(found / subset_errors),
                }
            )
    return overall_records, budget_records, label_records


def paired_contrasts(overall: pd.DataFrame, budgets: pd.DataFrame) -> pd.DataFrame:
    records = []
    primary_budget = budgets[budgets["budget_name"] == "true_error_count"]
    metric_frames = {
        "auprc": overall,
        "review_recall_auc": overall,
        "recall_at_error_count": primary_budget.rename(columns={"recall": "recall_at_error_count"}),
    }
    for metric, frame in metric_frames.items():
        for scenario_id, scenario in frame.groupby("scenario_id"):
            pivot = scenario.pivot(index="seed", columns="method", values=metric)
            for method in sorted(set(pivot.columns) - {"oof_label_incompatibility"}):
                paired = pivot[[method, "oof_label_incompatibility"]].dropna()
                differences = (paired[method] - paired["oof_label_incompatibility"]).to_numpy()
                records.append(
                    {
                        "scenario_id": scenario_id,
                        "metric": metric,
                        "method": method,
                        "reference_method": "oof_label_incompatibility",
                        "paired_seeds": int(len(differences)),
                        "mean_difference": float(differences.mean()),
                        "positive_seeds": int(np.sum(differences > 0)),
                        "negative_seeds": int(np.sum(differences < 0)),
                        "exact_sign_flip_p": exact_sign_flip_p(differences),
                    }
                )
    return pd.DataFrame(records)


def plot_primary(overall: pd.DataFrame, budgets: pd.DataFrame, output: Path) -> None:
    primary_budget = budgets[budgets["budget_name"] == "true_error_count"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    palette = {
        "oof_label_incompatibility": "#2563eb",
        "active_label_cleaning": "#7c3aed",
        "simifeat_style_k10": "#059669",
        "oof_simifeat_rank_fusion": "#dc2626",
    }
    for method in PRIMARY_METHODS:
        for axis, frame, metric, ylabel in [
            (axes[0], overall, "auprc", "AUPRC for injected errors"),
            (axes[1], primary_budget, "recall", "Error recall at matched review budget"),
        ]:
            selected = frame[frame["method"] == method]
            summary = selected.groupby("rate_percent")[metric].agg(["mean", "std"]).reset_index()
            axis.errorbar(
                summary["rate_percent"],
                summary["mean"],
                yerr=summary["std"].fillna(0),
                marker="o",
                capsize=3,
                linewidth=1.8,
                color=palette[method],
                label=method.replace("_", " "),
            )
            axis.set_xlabel("Injected entry noise (%)")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> None:
    parent = Path(args.parent_root)
    score_root = Path(args.output_root)
    evaluation = score_root / "evaluation"
    if not (score_root / ".blind_scoring_complete").is_file():
        raise FileNotFoundError("Blind scoring is incomplete")
    if evaluation.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {evaluation}")
    score_manifest = pd.read_csv(score_root / "score_manifest.csv")
    overall_records: list[dict] = []
    budget_records: list[dict] = []
    label_records: list[dict] = []
    for row in score_manifest.itertuples(index=False):
        scores_path = score_root / row.score_relpath
        if sha256_file(scores_path) != row.score_sha256:
            raise RuntimeError(f"Blind score hash mismatch: {scores_path}")
        scores = pd.read_csv(scores_path)
        methods = score_columns(scores)
        prepared = parent / row.prepared_relpath
        merged = merge_private(scores, prepared / "private_reference.csv")
        metadata = {
            "array_index": int(row.array_index),
            "scenario_id": str(row.scenario_id),
            "noise_rate": float(row.noise_rate),
            "rate_percent": int(row.rate_percent),
            "seed": int(row.seed),
        }
        overall, budgets, labels = evaluate_run(merged, methods, metadata)
        overall_records.extend(overall)
        budget_records.extend(budgets)
        label_records.extend(labels)
        print(f"evaluated scenario={row.scenario_id} seed={int(row.seed)}", flush=True)

    evaluation.mkdir(parents=True)
    overall_frame = pd.DataFrame(overall_records)
    budget_frame = pd.DataFrame(budget_records)
    label_frame = pd.DataFrame(label_records)
    contrasts = paired_contrasts(overall_frame, budget_frame)
    atomic_write_csv(overall_frame, evaluation / "overall_detection_metrics.csv")
    atomic_write_csv(budget_frame, evaluation / "budget_metrics.csv")
    atomic_write_csv(label_frame, evaluation / "per_label_metrics.csv")
    atomic_write_csv(contrasts, evaluation / "paired_method_contrasts.csv")
    plot_primary(overall_frame, budget_frame, evaluation / "detector_comparison.png")

    endpoint = budget_frame[budget_frame["budget_name"] == "true_error_count"]
    means = endpoint.groupby(["scenario_id", "method"])["recall"].mean().reset_index()
    winners = means.loc[means.groupby("scenario_id")["recall"].idxmax()].to_dict("records")
    summary = {
        "protocol": PROTOCOL_NAME,
        "private_evaluation": True,
        "runs_with_errors": int(overall_frame[["scenario_id", "seed"]].drop_duplicates().shape[0]),
        "methods": sorted(overall_frame["method"].unique().tolist()),
        "primary_budget": "review count matched to the number of injected errors",
        "scenario_winners_by_mean_recall": winners,
        "program_sha256": sha256_file(Path(__file__)),
        "blind_scoring_summary_sha256": sha256_file(score_root / "scoring_summary.json"),
    }
    atomic_write_json(summary, evaluation / "evaluation_summary.json")
    atomic_write_text(evaluation / ".evaluation_complete", "complete\n")


def verify(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    evaluation = root / "evaluation"
    for path in [
        root / ".blind_scoring_complete",
        root / "score_manifest.csv",
        root / "scoring_summary.json",
        evaluation / ".evaluation_complete",
        evaluation / "overall_detection_metrics.csv",
        evaluation / "budget_metrics.csv",
        evaluation / "paired_method_contrasts.csv",
        evaluation / "detector_comparison.png",
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = pd.read_csv(root / "score_manifest.csv")
    overall = pd.read_csv(evaluation / "overall_detection_metrics.csv")
    budgets = pd.read_csv(evaluation / "budget_metrics.csv")
    if manifest[["scenario_id", "seed"]].duplicated().any():
        raise RuntimeError("Duplicate runs in score manifest")
    expected_nonclean = int((manifest["rate_percent"] > 0).sum())
    expected_methods = 6
    if len(overall) != expected_nonclean * expected_methods:
        raise RuntimeError("Overall metric grid is incomplete")
    if len(budgets) != expected_nonclean * expected_methods * 5:
        raise RuntimeError("Budget metric grid is incomplete")
    if not np.isfinite(overall[["auprc", "auroc", "review_recall_auc"]].to_numpy()).all():
        raise RuntimeError("Non-finite overall metrics")
    atomic_write_text(root / ".benchmark_verified", "verified\n")
    print(json.dumps({"verified": True, "runs": int(len(manifest)), "methods": expected_methods}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--parent-root", required=True)
    common.add_argument("--output-root", required=True)

    score_parser = subparsers.add_parser("score", parents=[common])
    score_parser.add_argument("--features-npz", required=True)
    score_parser.add_argument("--k-values", default=",".join(map(str, DEFAULT_K_VALUES)))
    score_parser.add_argument("--n-jobs", type=int, default=8)
    score_parser.add_argument("--seeds", default="")
    score_parser.add_argument("--scenarios", default="")
    score_parser.set_defaults(function=score)

    evaluate_parser = subparsers.add_parser("evaluate", parents=[common])
    evaluate_parser.set_defaults(function=evaluate)

    verify_parser = subparsers.add_parser("verify", parents=[common])
    verify_parser.set_defaults(function=verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
