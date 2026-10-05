#!/usr/bin/env python3
"""Outcome-isolated VinDr noise-direction and DQS sensitivity benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from vindr_known_gt_cl_benchmark import (
    LABELS,
    atomic_write_csv,
    atomic_write_text,
    binary_metrics,
    exact_sign_flip_p,
    fold_support,
    holm_adjust,
    parse_seeds,
    require_columns,
    sha256_file,
    top_budget_mask,
    validate_blind_columns,
)


PROTOCOL_NAME = "vindr_noise_direction_sensitivity_v1"
DEFAULT_SEEDS = [13, 42, 97, 123, 211, 307]
DEFAULT_RATES = [0.10, 0.20, 0.30]
REGIMES = ["balanced", "fp_only", "fn_only"]
METHODS = {
    "cl_first": "cl_first_score",
    "self_confidence": "self_confidence_suspicion",
    "predictive_entropy": "predictive_entropy",
}


def stable_rng(*parts: object) -> np.random.Generator:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def parse_rates(text: str | Iterable[float]) -> list[float]:
    if isinstance(text, str):
        rates = [float(token.strip()) for token in text.split(",") if token.strip()]
    else:
        rates = [float(value) for value in text]
    if not rates or len(rates) != len(set(rates)):
        raise ValueError("Rates must be non-empty and unique")
    if any(rate <= 0 or rate >= 0.5 for rate in rates):
        raise ValueError("Rates must be between 0 and 0.5")
    return rates


def scenario_definitions(rates: Iterable[float]) -> list[dict[str, Any]]:
    scenarios = [
        {
            "scenario_index": 0,
            "scenario_id": "clean",
            "regime": "clean",
            "noise_rate": 0.0,
            "rate_percent": 0,
        }
    ]
    for rate in rates:
        percent = int(round(rate * 100))
        if not np.isclose(rate, percent / 100):
            raise ValueError(f"Rate does not map to an integer percentage: {rate}")
        for regime in REGIMES:
            scenarios.append(
                {
                    "scenario_index": len(scenarios),
                    "scenario_id": f"{regime}_r{percent:02d}",
                    "regime": regime,
                    "noise_rate": float(rate),
                    "rate_percent": percent,
                }
            )
    return scenarios


def make_corruption(
    clean: np.ndarray,
    seed: int,
    scenario: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    noisy = clean.copy()
    injected = np.zeros_like(clean, dtype=bool)
    direction = np.full(clean.shape, "none", dtype=object)
    records = []
    for label_index, label in enumerate(LABELS):
        positive = np.flatnonzero(clean[:, label_index] == 1)
        negative = np.flatnonzero(clean[:, label_index] == 0)
        rate = float(scenario["noise_rate"])
        base_count = int(round(rate * len(positive))) if rate > 0 else 0
        if scenario["regime"] == "clean":
            fp_count = 0
            fn_count = 0
        elif scenario["regime"] == "balanced":
            fp_count = base_count
            fn_count = base_count
        elif scenario["regime"] == "fp_only":
            fp_count = 2 * base_count
            fn_count = 0
        elif scenario["regime"] == "fn_only":
            fp_count = 0
            fn_count = 2 * base_count
        else:
            raise ValueError(f"Unknown corruption regime: {scenario['regime']}")
        if fp_count >= len(negative) or fn_count >= len(positive):
            raise ValueError(
                f"Infeasible corruption for {label}: fp={fp_count}, fn={fn_count}"
            )

        if fp_count:
            rng = stable_rng(PROTOCOL_NAME, seed, label, scenario["scenario_id"], "0_to_1")
            selected = rng.choice(negative, size=fp_count, replace=False)
            noisy[selected, label_index] = 1
            injected[selected, label_index] = True
            direction[selected, label_index] = "0_to_1"
        if fn_count:
            rng = stable_rng(PROTOCOL_NAME, seed, label, scenario["scenario_id"], "1_to_0")
            selected = rng.choice(positive, size=fn_count, replace=False)
            noisy[selected, label_index] = 0
            injected[selected, label_index] = True
            direction[selected, label_index] = "1_to_0"

        records.append(
            {
                "seed": seed,
                "scenario_id": scenario["scenario_id"],
                "regime": scenario["regime"],
                "noise_rate": rate,
                "label_name": label,
                "clean_positives": int(len(positive)),
                "clean_negatives": int(len(negative)),
                "base_count": base_count,
                "false_positive_errors": fp_count,
                "false_negative_errors": fn_count,
                "injected_errors": fp_count + fn_count,
                "noisy_positives": int(noisy[:, label_index].sum()),
            }
        )
    return noisy, injected, direction, pd.DataFrame(records)


def choose_shared_folds(
    scenario_labels: dict[str, np.ndarray],
    seed: int,
    n_splits: int,
) -> tuple[np.ndarray, int]:
    samples = len(next(iter(scenario_labels.values())))
    for attempt in range(2_000):
        split_seed = seed + attempt
        folds = np.full(samples, -1, dtype=np.int64)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
        for fold_id, (_, validation_indices) in enumerate(splitter.split(np.arange(samples))):
            folds[validation_indices] = fold_id
        valid = True
        for labels in scenario_labels.values():
            for fold_id in range(n_splits):
                subset = labels[folds == fold_id]
                positives = subset.sum(axis=0)
                negatives = len(subset) - positives
                if np.any(positives < 5) or np.any(negatives < 5):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return folds, split_seed
    raise ValueError(f"No shared fold assignment satisfies support for seed {seed}")


def private_frame(
    image_ids: pd.Series,
    clean: np.ndarray,
    noisy: np.ndarray,
    injected: np.ndarray,
    direction: np.ndarray,
) -> pd.DataFrame:
    records = []
    for row_index, image_id in enumerate(image_ids.astype(str)):
        for label_index, label in enumerate(LABELS):
            records.append(
                {
                    "image_id": image_id,
                    "label_name": label,
                    "clean_label": int(clean[row_index, label_index]),
                    "noisy_label": int(noisy[row_index, label_index]),
                    "injected_error": int(injected[row_index, label_index]),
                    "flip_direction": str(direction[row_index, label_index]),
                }
            )
    frame = pd.DataFrame(records)
    if len(frame) != 18_000 or frame[["image_id", "label_name"]].duplicated().any():
        raise RuntimeError("Private reference keys are incomplete or duplicated")
    expected = frame["clean_label"].ne(frame["noisy_label"]).astype(int)
    if not frame["injected_error"].eq(expected).all():
        raise RuntimeError("Private error flags disagree with label values")
    return frame


def prepare(args: argparse.Namespace) -> None:
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=False)
    labels = pd.read_csv(args.labels_csv)
    require_columns(labels, ["image_id"] + LABELS, "VinDr consensus labels")
    labels["image_id"] = labels["image_id"].astype(str)
    if len(labels) != 3_000 or labels["image_id"].nunique() != 3_000:
        raise ValueError("Consensus labels must contain 3,000 unique images")
    if not all(labels[label].isin([0, 1]).all() for label in LABELS):
        raise ValueError("Consensus labels must be binary")

    image_index = pd.read_csv(args.parent_image_index)
    require_columns(image_index, ["image_id", "image_path", "output_sha256"], "image index")
    image_index["image_id"] = image_index["image_id"].astype(str)
    cohort = image_index.merge(labels[["image_id"] + LABELS], on="image_id", validate="one_to_one")
    cohort = cohort.sort_values("image_id").reset_index(drop=True)
    if len(cohort) != 3_000:
        raise ValueError("Image index and consensus labels do not align")
    clean = cohort[LABELS].to_numpy(dtype=np.int64)
    atomic_write_csv(cohort[["image_id", "image_path", "output_sha256"]], output / "image_index.csv")

    seeds = parse_seeds(args.seeds)
    scenarios = scenario_definitions(parse_rates(args.rates))
    atomic_write_csv(pd.DataFrame(scenarios), output / "scenarios.csv")
    manifest_records = []
    seed_records = []
    array_index = 0
    for seed in seeds:
        generated = {}
        for scenario in scenarios:
            generated[scenario["scenario_id"]] = make_corruption(clean, seed, scenario)
        folds, fold_seed = choose_shared_folds(
            {key: value[0] for key, value in generated.items()}, seed, args.n_splits
        )
        seed_records.append(
            {
                "seed": seed,
                "seed_block": "parent_overlap" if seed in [13, 42, 97, 123] else "new_replication",
                "fold_assignment_seed": fold_seed,
                "n_splits": args.n_splits,
            }
        )
        for scenario in scenarios:
            scenario_id = scenario["scenario_id"]
            noisy, injected, direction, counts = generated[scenario_id]
            prepared = output / "scenarios" / scenario_id / f"seed_{seed}" / "prepared"
            prepared.mkdir(parents=True, exist_ok=False)
            blind = cohort[["image_id", "image_path"]].copy()
            blind["fold_id"] = folds
            for label_index, label in enumerate(LABELS):
                blind[label] = noisy[:, label_index]
            validate_blind_columns(blind)
            support = fold_support(blind, args.n_splits)
            private = private_frame(
                cohort["image_id"], clean, noisy, injected, direction
            )
            blind_path = prepared / "blind_noisy_cohort.csv"
            private_path = prepared / "private_reference.csv"
            counts_path = prepared / "corruption_counts.csv"
            support_path = prepared / "fold_support.csv"
            atomic_write_csv(blind, blind_path)
            atomic_write_csv(private, private_path)
            atomic_write_csv(counts, counts_path)
            atomic_write_csv(support, support_path)
            summary = {
                "protocol": PROTOCOL_NAME,
                "scenario_id": scenario_id,
                "regime": scenario["regime"],
                "noise_rate": scenario["noise_rate"],
                "seed": seed,
                "seed_block": "parent_overlap" if seed in [13, 42, 97, 123] else "new_replication",
                "samples": 3_000,
                "entries": 18_000,
                "injected_errors": int(injected.sum()),
                "true_quality": float(1.0 - injected.mean()),
                "fold_assignment_seed": fold_seed,
                "blind_cohort_sha256": sha256_file(blind_path),
                "private_reference_sha256": sha256_file(private_path),
                "corruption_counts_sha256": sha256_file(counts_path),
                "fold_support_sha256": sha256_file(support_path),
            }
            atomic_write_text(prepared / "prepare_summary.json", json.dumps(summary, indent=2))
            atomic_write_text(prepared / ".prepare_complete", "complete\n")
            manifest_records.append(
                {
                    "array_index": array_index,
                    "scenario_index": scenario["scenario_index"],
                    "scenario_id": scenario_id,
                    "regime": scenario["regime"],
                    "noise_rate": scenario["noise_rate"],
                    "rate_percent": scenario["rate_percent"],
                    "seed": seed,
                    "seed_block": summary["seed_block"],
                    "injected_errors": summary["injected_errors"],
                    "true_quality": summary["true_quality"],
                    "prepared_relpath": str(prepared.relative_to(output)),
                    "blind_run_relpath": str(
                        (prepared.parent / "blind_run").relative_to(output)
                    ),
                }
            )
            array_index += 1

    manifest = pd.DataFrame(manifest_records)
    expected_rows = len(seeds) * len(scenarios)
    if len(manifest) != expected_rows or manifest["array_index"].nunique() != expected_rows:
        raise RuntimeError("Scenario manifest grid is incomplete")
    for rate, frame in manifest.query("noise_rate > 0").groupby(["seed", "noise_rate"]):
        if frame["injected_errors"].nunique() != 1:
            raise RuntimeError(f"Matched error counts failed for {rate}")
    atomic_write_csv(manifest, output / "scenario_manifest.csv")
    blind_manifest_columns = [
        "array_index",
        "scenario_index",
        "scenario_id",
        "regime",
        "noise_rate",
        "rate_percent",
        "seed",
        "seed_block",
        "prepared_relpath",
        "blind_run_relpath",
    ]
    blind_manifest = manifest[blind_manifest_columns].copy()
    if any(
        token in column.lower()
        for column in blind_manifest.columns
        for token in ["clean", "reference", "injected", "true_", "error"]
    ):
        raise RuntimeError("Blind-run manifest contains a forbidden outcome column")
    atomic_write_csv(blind_manifest, output / "blind_run_manifest.csv")
    atomic_write_csv(pd.DataFrame(seed_records), output / "seed_manifest.csv")
    root_summary = {
        "protocol": PROTOCOL_NAME,
        "seeds": seeds,
        "rates": parse_rates(args.rates),
        "regimes": REGIMES,
        "scenarios_per_seed": len(scenarios),
        "blind_runs": len(manifest),
        "labels": LABELS,
        "consensus_labels_sha256": sha256_file(Path(args.labels_csv)),
        "parent_image_index_sha256": sha256_file(Path(args.parent_image_index)),
        "scenario_manifest_sha256": sha256_file(output / "scenario_manifest.csv"),
        "blind_run_manifest_sha256": sha256_file(output / "blind_run_manifest.csv"),
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(output / "prepare_summary.json", json.dumps(root_summary, indent=2))
    atomic_write_text(output / ".prepare_complete", "complete\n")
    print(json.dumps(root_summary, indent=2), flush=True)


def merge_private(prepared: Path, blind_run: Path) -> pd.DataFrame:
    private = pd.read_csv(prepared / "private_reference.csv")
    evidence = pd.read_csv(blind_run / "entry_evidence.csv")
    require_columns(
        private,
        ["image_id", "label_name", "clean_label", "noisy_label", "injected_error", "flip_direction"],
        "private reference",
    )
    require_columns(
        evidence,
        ["image_id", "label_name", "noisy_label", "oof_probability", "cl_issue"] + list(METHODS.values()),
        "blind evidence",
    )
    if len(private) != 18_000 or len(evidence) != 18_000:
        raise ValueError("Private/evidence row counts must both be 18,000")
    merged = evidence.merge(
        private,
        on=["image_id", "label_name"],
        suffixes=("", "_private"),
        validate="one_to_one",
    )
    if not merged["noisy_label"].eq(merged["noisy_label_private"]).all():
        raise ValueError("Blind and private noisy labels disagree")
    return merged


def budget_names(n: int, true_errors: int) -> list[tuple[str, int]]:
    budgets = [
        (f"{int(fraction * 100)}pct", int(round(n * fraction)))
        for fraction in [0.01, 0.02, 0.05, 0.10]
    ]
    budgets.append(("true_error_count", true_errors))
    return budgets


def scenario_metrics(
    merged: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    truth = merged["injected_error"].to_numpy(dtype=bool)
    issue = merged["cl_issue"].to_numpy(dtype=bool)
    base = dict(metadata)
    overall_records = []
    for method, score_column in METHODS.items():
        prediction = issue if method == "cl_first" else np.zeros(len(merged), dtype=bool)
        metrics = binary_metrics(truth, prediction, merged[score_column].to_numpy())
        overall_records.append({**base, "method": method, **metrics})

    budget_records = []
    true_errors = int(truth.sum())
    if true_errors > 0:
        for budget_name, budget in budget_names(len(merged), true_errors):
            expected_tp = budget * true_errors / len(merged)
            for method, score_column in METHODS.items():
                selected = top_budget_mask(merged, score_column, budget)
                tp = int(np.sum(truth & selected))
                budget_records.append(
                    {
                        **base,
                        "method": method,
                        "budget_name": budget_name,
                        "budget": budget,
                        "tp": tp,
                        "precision": float(tp / budget),
                        "recall": float(tp / true_errors),
                        "enrichment": float((tp / budget) / truth.mean()),
                        "random_expected_tp": float(expected_tp),
                        "random_expected_recall": float(budget / len(merged)),
                        "random_exact_p_ge_observed": float(
                            hypergeom.sf(tp - 1, len(merged), true_errors, budget)
                        ),
                    }
                )

    label_records = []
    for label, frame in merged.groupby("label_name", sort=False):
        for scope in ["all", "0_to_1", "1_to_0"]:
            scoped_truth = (
                frame["injected_error"].to_numpy(dtype=bool)
                if scope == "all"
                else frame["flip_direction"].eq(scope).to_numpy()
            )
            for method, score_column in METHODS.items():
                prediction = (
                    frame["cl_issue"].to_numpy(dtype=bool)
                    if method == "cl_first"
                    else np.zeros(len(frame), dtype=bool)
                )
                metrics = binary_metrics(scoped_truth, prediction, frame[score_column].to_numpy())
                label_records.append(
                    {
                        **base,
                        "label_name": label,
                        "flip_direction": scope,
                        "method": method,
                        **metrics,
                    }
                )

    auroc_records = []
    for label, frame in merged.groupby("label_name", sort=False):
        auroc_records.append(
            {
                **base,
                "label_name": label,
                "noisy_label_auroc": float(
                    roc_auc_score(frame["noisy_label"], frame["oof_probability"])
                ),
                "clean_label_auroc": float(
                    roc_auc_score(frame["clean_label"], frame["oof_probability"])
                ),
            }
        )

    hard = binary_metrics(
        truth,
        issue,
        merged["self_confidence_suspicion"].to_numpy(),
    )
    direction_recalls = {}
    for direction in ["0_to_1", "1_to_0"]:
        directional_truth = merged["flip_direction"].eq(direction).to_numpy()
        direction_recalls[f"hard_recall_{direction}"] = (
            float(np.sum(issue & directional_truth) / directional_truth.sum())
            if directional_truth.sum()
            else float("nan")
        )
    quality = {
        **base,
        "samples": int(merged["image_id"].nunique()),
        "entries": len(merged),
        "true_errors": true_errors,
        "true_quality": float(1.0 - truth.mean()),
        "cl_issue_entries": int(issue.sum()),
        "raw_entry_dqs": float(1.0 - issue.mean()),
        "dqs_signed_error": float(truth.mean() - issue.mean()),
        "dqs_absolute_error": float(abs(truth.mean() - issue.mean())),
        "hard_precision": hard["precision"],
        "hard_recall": hard["recall"],
        "hard_f1": hard["f1"],
        "hard_enrichment": hard["enrichment"],
        **direction_recalls,
    }
    return overall_records, budget_records, label_records, auroc_records, quality


def bootstrap_scenario(
    merged: pd.DataFrame,
    metadata: dict[str, Any],
    iterations: int,
) -> pd.DataFrame:
    grouped = merged.groupby("image_id", sort=True)
    image_ids = np.asarray(list(grouped.groups))
    true_counts = grouped["injected_error"].sum().reindex(image_ids).to_numpy(dtype=float)
    issue_counts = grouped["cl_issue"].sum().reindex(image_ids).to_numpy(dtype=float)
    selected = merged["cl_issue"].astype(bool)
    tp_counts = (
        merged.assign(_tp=selected & merged["injected_error"].astype(bool))
        .groupby("image_id")["_tp"]
        .sum()
        .reindex(image_ids)
        .to_numpy(dtype=float)
    )
    direction_arrays = {}
    for direction in ["0_to_1", "1_to_0"]:
        truth_column = merged["flip_direction"].eq(direction)
        direction_arrays[direction] = (
            merged.assign(_truth=truth_column, _tp=truth_column & selected)
            .groupby("image_id")[["_truth", "_tp"]]
            .sum()
            .reindex(image_ids)
            .to_numpy(dtype=float)
        )

    rng = stable_rng(PROTOCOL_NAME, metadata["seed"], metadata["scenario_id"], "bootstrap")
    weights = rng.multinomial(
        len(image_ids),
        np.full(len(image_ids), 1.0 / len(image_ids)),
        size=iterations,
    )
    true_total = weights @ true_counts
    issue_total = weights @ issue_counts
    tp_total = weights @ tp_counts
    entries = len(image_ids) * len(LABELS)
    with np.errstate(divide="ignore", invalid="ignore"):
        hard_precision = tp_total / issue_total
        hard_recall = tp_total / true_total
    payload = {
        **metadata,
        "iteration": np.arange(iterations),
        "true_quality": 1.0 - true_total / entries,
        "raw_entry_dqs": 1.0 - issue_total / entries,
        "hard_precision": hard_precision,
        "hard_recall": hard_recall,
    }
    for direction, arrays in direction_arrays.items():
        directional_true = weights @ arrays[:, 0]
        directional_tp = weights @ arrays[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            payload[f"hard_recall_{direction}"] = directional_tp / directional_true
    return pd.DataFrame(payload)


def calibration_table(quality: pd.DataFrame) -> pd.DataFrame:
    clean = quality.query("regime == 'clean'")
    records = []
    for seed in sorted(quality["seed"].unique()):
        anchor = clean[clean["seed"] == seed]
        if len(anchor) != 1:
            raise RuntimeError(f"Seed {seed} lacks one clean anchor")
        for regime in REGIMES:
            noisy = quality[(quality["seed"] == seed) & (quality["regime"] == regime)]
            frame = pd.concat([anchor, noisy], ignore_index=True).sort_values("noise_rate")
            if len(frame) != 4:
                raise RuntimeError(f"Calibration grid is incomplete: seed={seed}, regime={regime}")
            x = frame["true_quality"].to_numpy(dtype=float)
            y = frame["raw_entry_dqs"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            rho = float(spearmanr(x, y).statistic)
            errors = y - x
            records.append(
                {
                    "seed": seed,
                    "seed_block": frame["seed_block"].iloc[0],
                    "regime": regime,
                    "points": len(frame),
                    "spearman_rho": rho,
                    "mae": float(np.mean(np.abs(errors))),
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "strictly_decreasing_with_noise": bool(
                        np.all(np.diff(frame["raw_entry_dqs"].to_numpy()) < 0)
                    ),
                }
            )
    return pd.DataFrame(records)


def contrast_tables(
    quality: pd.DataFrame,
    budgets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directional = quality[quality["regime"].isin(["fp_only", "fn_only"])].pivot(
        index=["seed", "seed_block", "noise_rate"],
        columns="regime",
        values="hard_recall",
    ).reset_index()
    directional["fp_minus_fn_recall"] = directional["fp_only"] - directional["fn_only"]
    seed_direction = directional.groupby(["seed", "seed_block"], as_index=False).agg(
        mean_fp_recall=("fp_only", "mean"),
        mean_fn_recall=("fn_only", "mean"),
        mean_direction_gap=("fp_minus_fn_recall", "mean"),
    )
    direction_p = exact_sign_flip_p(seed_direction["mean_direction_gap"].to_numpy())
    seed_direction["six_seed_exact_sign_flip_p"] = direction_p

    primary_budget = budgets[budgets["budget_name"] == "true_error_count"]
    pivot = primary_budget.pivot(
        index=["seed", "seed_block", "scenario_id"],
        columns="method",
        values="recall",
    ).reset_index()
    pivot["cl_minus_self"] = pivot["cl_first"] - pivot["self_confidence"]
    pivot["cl_minus_entropy"] = pivot["cl_first"] - pivot["predictive_entropy"]
    seed_methods = pivot.groupby(["seed", "seed_block"], as_index=False).agg(
        mean_cl_recall=("cl_first", "mean"),
        mean_self_recall=("self_confidence", "mean"),
        mean_entropy_recall=("predictive_entropy", "mean"),
        mean_cl_minus_self=("cl_minus_self", "mean"),
        mean_cl_minus_entropy=("cl_minus_entropy", "mean"),
    )
    contrast_records = []
    for column, name in [
        ("mean_cl_minus_self", "cl_first_minus_self_confidence"),
        ("mean_cl_minus_entropy", "cl_first_minus_predictive_entropy"),
    ]:
        values = seed_methods[column].to_numpy(dtype=float)
        contrast_records.append(
            {
                "contrast": name,
                "mean_difference": float(values.mean()),
                "positive_seeds": int(np.sum(values > 0)),
                "ties": int(np.sum(values == 0)),
                "negative_seeds": int(np.sum(values < 0)),
                "exact_sign_flip_p": exact_sign_flip_p(values),
            }
        )
    adjusted = holm_adjust([record["exact_sign_flip_p"] for record in contrast_records])
    for record, value in zip(contrast_records, adjusted):
        record["holm_p"] = value
    return directional, seed_direction, seed_methods, pd.DataFrame(contrast_records)


def evaluate(args: argparse.Namespace) -> None:
    experiment = Path(args.experiment_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = pd.read_csv(experiment / "scenario_manifest.csv")
    seeds = parse_seeds(args.seeds)
    expected = len(seeds) * 10
    if len(manifest) != expected or sorted(manifest["seed"].unique()) != sorted(seeds):
        raise ValueError("Evaluation manifest does not match the six-seed scenario grid")

    overall_records = []
    budget_records = []
    label_records = []
    auroc_records = []
    quality_records = []
    bootstrap_parts = []
    for row in manifest.sort_values("array_index").itertuples(index=False):
        prepared = experiment / row.prepared_relpath
        blind_run = experiment / row.blind_run_relpath
        if not (prepared / ".prepare_complete").is_file():
            raise FileNotFoundError(f"Prepare marker missing: {prepared}")
        if not (blind_run / ".blind_run_complete").is_file():
            raise FileNotFoundError(f"Blind-run marker missing: {blind_run}")
        merged = merge_private(prepared, blind_run)
        metadata = {
            "scenario_id": row.scenario_id,
            "regime": row.regime,
            "noise_rate": float(row.noise_rate),
            "rate_percent": int(row.rate_percent),
            "seed": int(row.seed),
            "seed_block": row.seed_block,
        }
        overall, budgets, labels, aurocs, quality = scenario_metrics(merged, metadata)
        overall_records.extend(overall)
        budget_records.extend(budgets)
        label_records.extend(labels)
        auroc_records.extend(aurocs)
        quality_records.append(quality)
        bootstrap_parts.append(
            bootstrap_scenario(merged, metadata, args.bootstrap_iterations)
        )
        print(f"[evaluate] {row.array_index + 1}/{len(manifest)} {row.scenario_id} seed={row.seed}", flush=True)

    overall = pd.DataFrame(overall_records)
    budgets = pd.DataFrame(budget_records)
    labels = pd.DataFrame(label_records)
    aurocs = pd.DataFrame(auroc_records)
    quality = pd.DataFrame(quality_records)
    bootstrap = pd.concat(bootstrap_parts, ignore_index=True)
    calibration = calibration_table(quality)
    direction_by_rate, seed_direction, seed_methods, method_contrasts = contrast_tables(
        quality, budgets
    )

    atomic_write_csv(overall, output / "overall_detection_metrics.csv")
    atomic_write_csv(budgets, output / "budget_metrics.csv")
    atomic_write_csv(labels, output / "per_label_direction_metrics.csv")
    atomic_write_csv(aurocs, output / "oof_label_aurocs.csv")
    atomic_write_csv(quality, output / "quality_summary.csv")
    atomic_write_csv(bootstrap, output / "image_cluster_bootstrap.csv")
    atomic_write_csv(calibration, output / "dqs_calibration.csv")
    atomic_write_csv(direction_by_rate, output / "direction_gap_by_seed_rate.csv")
    atomic_write_csv(seed_direction, output / "direction_gap_by_seed.csv")
    atomic_write_csv(seed_methods, output / "method_contrasts_by_seed.csv")
    atomic_write_csv(method_contrasts, output / "method_contrasts.csv")

    noisy_overall = overall[(overall["noise_rate"] > 0) & (overall["method"] == "cl_first")]
    noisy_quality = quality[quality["noise_rate"] > 0]
    cl_budget = budgets[
        (budgets["budget_name"] == "true_error_count") & (budgets["method"] == "cl_first")
    ]
    direction_p = float(seed_direction["six_seed_exact_sign_flip_p"].iloc[0])
    direction_confirmed = bool(
        (seed_direction["mean_direction_gap"] > 0.20).all() and direction_p < 0.05
    )
    direction_rate_mean = direction_by_rate.groupby("noise_rate").agg(
        fp_recall=("fp_only", "mean"), fn_recall=("fn_only", "mean")
    )
    direction_invariant = bool(
        ((direction_rate_mean["fp_recall"] - direction_rate_mean["fn_recall"]).abs() <= 0.10).all()
        and (direction_rate_mean[["fp_recall", "fn_recall"]] >= 0.50).all().all()
    )
    self_contrast = method_contrasts[
        method_contrasts["contrast"] == "cl_first_minus_self_confidence"
    ].iloc[0]
    incremental_supported = bool(
        (seed_methods["mean_cl_minus_self"] > 0.02).all()
        and self_contrast["exact_sign_flip_p"] < 0.05
        and self_contrast["holm_p"] < 0.05
    )
    dqs_supported = bool(
        (calibration["spearman_rho"] >= 0.95).all()
        and (calibration["mae"] <= 0.01).all()
        and calibration["strictly_decreasing_with_noise"].all()
    )
    gates = {
        "detection_auprc_above_prevalence_all_noisy_runs": bool(
            (noisy_overall["auprc"] > noisy_overall["prevalence"]).all()
        ),
        "hard_enrichment_above_one_all_noisy_runs": bool(
            (noisy_quality["hard_enrichment"] > 1).all()
        ),
        "cl_first_above_random_expected_recall_all_noisy_runs": bool(
            (cl_budget["recall"] > cl_budget["random_expected_recall"]).all()
        ),
        "directional_asymmetry_confirmed": direction_confirmed,
        "direction_invariant_performance_supported": direction_invariant,
        "incremental_cl_hard_filter_supported": incremental_supported,
        "dqs_calibration_supported": dqs_supported,
    }
    summary = {
        "protocol": PROTOCOL_NAME,
        "verification_status": "ANALYZED",
        "seeds": seeds,
        "parent_overlap_seeds": [13, 42, 97, 123],
        "new_replication_seeds": [211, 307],
        "scenarios": 10,
        "blind_runs": 60,
        "gates": gates,
        "direction_exact_sign_flip_p": direction_p,
        "mean_direction_gap": float(seed_direction["mean_direction_gap"].mean()),
        "method_contrasts": method_contrasts.to_dict(orient="records"),
        "dqs_calibration_worst_mae": float(calibration["mae"].max()),
        "dqs_calibration_min_spearman": float(calibration["spearman_rho"].min()),
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(output / "evaluation_summary.json", json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    plot = direction_by_rate.groupby("noise_rate").agg(
        fp_mean=("fp_only", "mean"),
        fp_sd=("fp_only", "std"),
        fn_mean=("fn_only", "mean"),
        fn_sd=("fn_only", "std"),
    )
    x = plot.index.to_numpy() * 100
    ax.errorbar(x, plot["fp_mean"], yerr=plot["fp_sd"], marker="o", capsize=4, label="0 to 1 errors")
    ax.errorbar(x, plot["fn_mean"], yerr=plot["fn_sd"], marker="o", capsize=4, label="1 to 0 errors")
    ax.set(xlabel="Noise-rate scale (%)", ylabel="CL hard recall", title="Direction-Specific Error Detection")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "direction_recall_by_rate.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    clean = quality[quality["regime"] == "clean"]
    for regime, frame in quality.groupby("regime", sort=False):
        if regime == "clean":
            continue
        anchored = pd.concat([clean, frame], ignore_index=True)
        mean = anchored.groupby("true_quality")["raw_entry_dqs"].mean().sort_index()
        ax.plot(mean.index, mean.values, marker="o", label=regime.replace("_", " "))
    bounds = [quality["true_quality"].min(), 1.0]
    ax.plot(bounds, bounds, linestyle="--", color="gray", label="ideal calibration")
    ax.set(xlabel="Known true quality", ylabel="Raw entry DQS", title="DQS Calibration Across Known Quality Levels")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "dqs_calibration.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    noisy = overall[(overall["noise_rate"] > 0)]
    plot = noisy.groupby(["regime", "noise_rate", "method"])["auprc"].mean().reset_index()
    for (regime, method), frame in plot.groupby(["regime", "method"]):
        ax.plot(
            frame["noise_rate"] * 100,
            frame["auprc"],
            marker="o",
            label=f"{regime.replace('_', ' ')} / {method.replace('_', ' ')}",
        )
    ax.set(xlabel="Noise-rate scale (%)", ylabel="Injected-error AUPRC", title="Detection Across Noise Regimes")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "auprc_by_regime_rate.png", dpi=180)
    plt.close(fig)

    atomic_write_text(output / ".evaluation_complete", "complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def verify(args: argparse.Namespace) -> None:
    experiment = Path(args.experiment_root)
    evaluation = Path(args.evaluation_dir) if args.evaluation_dir else None
    manifest = pd.read_csv(experiment / "scenario_manifest.csv")
    seeds = parse_seeds(args.seeds)
    if len(manifest) != 60 or sorted(manifest["seed"].unique()) != sorted(seeds):
        raise ValueError("Scenario manifest must contain the exact 60-run grid")
    if manifest[["scenario_id", "seed"]].duplicated().any():
        raise ValueError("Scenario manifest contains duplicate run keys")
    if manifest["scenario_id"].nunique() != 10:
        raise ValueError("Scenario manifest must contain ten scenarios")
    for row in manifest.itertuples(index=False):
        prepared = experiment / row.prepared_relpath
        blind_run = experiment / row.blind_run_relpath
        if not (prepared / ".prepare_complete").is_file():
            raise FileNotFoundError(f"Prepare marker missing: {prepared}")
        if not (blind_run / ".blind_run_complete").is_file():
            raise FileNotFoundError(f"Blind marker missing: {blind_run}")
        blind = pd.read_csv(prepared / "blind_noisy_cohort.csv")
        private = pd.read_csv(prepared / "private_reference.csv")
        evidence = pd.read_csv(blind_run / "entry_evidence.csv")
        validate_blind_columns(blind)
        validate_blind_columns(evidence.drop(columns=["cl_issue"], errors="ignore"))
        if len(blind) != 3_000 or len(private) != 18_000 or len(evidence) != 18_000:
            raise ValueError(f"Row-count verification failed: {row.scenario_id}, {row.seed}")
        if int(private["injected_error"].sum()) != int(row.injected_errors):
            raise ValueError(f"Injected-error count changed: {row.scenario_id}, {row.seed}")
    if evaluation:
        if not (evaluation / ".evaluation_complete").is_file():
            raise FileNotFoundError("Evaluation marker is missing")
        summary = json.loads((evaluation / "evaluation_summary.json").read_text())
        if summary["seeds"] != seeds or summary["blind_runs"] != 60:
            raise ValueError("Evaluation summary grid does not match verification request")
        expected_rows = {
            "overall_detection_metrics.csv": 180,
            "budget_metrics.csv": 810,
            "per_label_direction_metrics.csv": 3_240,
            "oof_label_aurocs.csv": 360,
            "quality_summary.csv": 60,
            "image_cluster_bootstrap.csv": 60_000,
            "dqs_calibration.csv": 18,
            "direction_gap_by_seed_rate.csv": 18,
            "direction_gap_by_seed.csv": 6,
            "method_contrasts_by_seed.csv": 6,
            "method_contrasts.csv": 2,
        }
        for name, expected in expected_rows.items():
            frame = pd.read_csv(evaluation / name)
            if len(frame) != expected:
                raise ValueError(f"{name} has {len(frame)} rows; expected {expected}")
        for name in [
            "direction_recall_by_rate.png",
            "dqs_calibration.png",
            "auprc_by_regime_rate.png",
        ]:
            if (evaluation / name).stat().st_size <= 1_000:
                raise ValueError(f"Figure is missing or empty: {name}")
    atomic_write_text(experiment / ".benchmark_verified", "complete\n")
    print(json.dumps({"protocol": PROTOCOL_NAME, "verification": "passed", "blind_runs": 60}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--labels-csv", type=Path, required=True)
    prepare_parser.add_argument("--parent-image-index", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    prepare_parser.add_argument("--rates", default=",".join(map(str, DEFAULT_RATES)))
    prepare_parser.add_argument("--n-splits", type=int, default=4)
    prepare_parser.set_defaults(function=prepare)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--experiment-root", type=Path, required=True)
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    evaluate_parser.add_argument("--bootstrap-iterations", type=int, default=1_000)
    evaluate_parser.set_defaults(function=evaluate)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--experiment-root", type=Path, required=True)
    verify_parser.add_argument("--evaluation-dir", type=Path)
    verify_parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    verify_parser.set_defaults(function=verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
