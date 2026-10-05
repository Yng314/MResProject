#!/usr/bin/env python3
"""Known-quality VinDr benchmark with exact global symmetric entry noise."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vindr_known_gt_cl_benchmark import (
    LABELS,
    atomic_write_csv,
    atomic_write_text,
    exact_sign_flip_p,
    fold_support,
    parse_seeds,
    require_columns,
    sha256_file,
    validate_blind_columns,
)
from vindr_noise_direction_sensitivity import (
    bootstrap_scenario,
    choose_shared_folds,
    merge_private,
    parse_rates,
    private_frame,
    scenario_metrics,
)


PROTOCOL_NAME = "vindr_global_symmetric_entry_noise_v1"
DEFAULT_SEEDS = [13, 42, 97, 123, 211, 307]
DEFAULT_RATES = [0.10, 0.20, 0.30]
REGIME = "symmetric_entry"


def stable_rng(*parts: object) -> np.random.Generator:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


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
        percent = int(round(float(rate) * 100))
        if not np.isclose(rate, percent / 100):
            raise ValueError(f"Rate does not map to an integer percentage: {rate}")
        scenarios.append(
            {
                "scenario_index": len(scenarios),
                "scenario_id": f"symmetric_entry_r{percent:02d}",
                "regime": REGIME,
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
    rate = float(scenario["noise_rate"])

    for label_index, label in enumerate(LABELS):
        positive = np.flatnonzero(clean[:, label_index] == 1)
        negative = np.flatnonzero(clean[:, label_index] == 0)
        target_total = int(round(rate * len(clean)))
        fn_count = int(round(rate * len(positive)))
        fp_count = target_total - fn_count
        if fp_count > len(negative) or fn_count > len(positive):
            raise ValueError(f"Infeasible corruption for {label}")

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
                "false_positive_errors": fp_count,
                "false_negative_errors": fn_count,
                "injected_errors": fp_count + fn_count,
                "false_positive_rate": float(fp_count / len(negative)),
                "false_negative_rate": float(fn_count / len(positive)),
                "clean_positive_prevalence": float(len(positive) / len(clean)),
                "noisy_positives": int(noisy[:, label_index].sum()),
                "noisy_positive_prevalence": float(noisy[:, label_index].mean()),
            }
        )

    expected = int(round(rate * clean.size))
    if int(injected.sum()) != expected:
        raise RuntimeError(
            f"Global corruption count is {int(injected.sum())}; expected {expected}"
        )
    return noisy, injected, direction, pd.DataFrame(records)


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
    rates = parse_rates(args.rates)
    scenarios = scenario_definitions(rates)
    atomic_write_csv(pd.DataFrame(scenarios), output / "scenarios.csv")
    manifest_records = []
    seed_records = []
    array_index = 0

    for seed in seeds:
        generated = {
            scenario["scenario_id"]: make_corruption(clean, seed, scenario)
            for scenario in scenarios
        }
        folds, fold_seed = choose_shared_folds(
            {key: value[0] for key, value in generated.items()}, seed, args.n_splits
        )
        seed_block = "parent_overlap" if seed in [13, 42, 97, 123] else "new_replication"
        seed_records.append(
            {
                "seed": seed,
                "seed_block": seed_block,
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
            private = private_frame(cohort["image_id"], clean, noisy, injected, direction)

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
                "seed_block": seed_block,
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
                    "seed_block": seed_block,
                    "injected_errors": summary["injected_errors"],
                    "true_quality": summary["true_quality"],
                    "prepared_relpath": str(prepared.relative_to(output)),
                    "blind_run_relpath": str((prepared.parent / "blind_run").relative_to(output)),
                }
            )
            array_index += 1

    manifest = pd.DataFrame(manifest_records)
    expected_rows = len(seeds) * len(scenarios)
    if len(manifest) != expected_rows or manifest["array_index"].tolist() != list(range(expected_rows)):
        raise RuntimeError("Scenario manifest grid is incomplete")
    expected_errors = {0: 0, **{int(round(rate * 100)): int(round(rate * clean.size)) for rate in rates}}
    for row in manifest.itertuples(index=False):
        if int(row.injected_errors) != expected_errors[int(row.rate_percent)]:
            raise RuntimeError(f"Incorrect error count for {row.scenario_id}, seed={row.seed}")
    atomic_write_csv(manifest, output / "scenario_manifest.csv")

    blind_columns = [
        "array_index", "scenario_index", "scenario_id", "regime", "noise_rate",
        "rate_percent", "seed", "seed_block", "prepared_relpath", "blind_run_relpath",
    ]
    blind_manifest = manifest[blind_columns].copy()
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
        "rates": rates,
        "regimes": [REGIME],
        "scenarios_per_seed": len(scenarios),
        "blind_runs": len(manifest),
        "labels": LABELS,
        "target_errors": expected_errors,
        "consensus_labels_sha256": sha256_file(Path(args.labels_csv)),
        "parent_image_index_sha256": sha256_file(Path(args.parent_image_index)),
        "scenario_manifest_sha256": sha256_file(output / "scenario_manifest.csv"),
        "blind_run_manifest_sha256": sha256_file(output / "blind_run_manifest.csv"),
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(output / "prepare_summary.json", json.dumps(root_summary, indent=2))
    atomic_write_text(output / ".prepare_complete", "complete\n")
    print(json.dumps(root_summary, indent=2), flush=True)


def calibration_table(quality: pd.DataFrame) -> pd.DataFrame:
    records = []
    for seed, frame in quality.groupby("seed", sort=True):
        frame = frame.sort_values("noise_rate")
        if len(frame) != 4:
            raise RuntimeError(f"Seed {seed} does not have four quality anchors")
        x = frame["true_quality"].to_numpy(dtype=float)
        y = frame["raw_entry_dqs"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        error = y - x
        records.append(
            {
                "seed": int(seed),
                "seed_block": frame["seed_block"].iloc[0],
                "points": len(frame),
                "spearman_rho": float(spearmanr(x, y).statistic),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "slope": float(slope),
                "intercept": float(intercept),
                "strictly_decreasing_with_noise": bool(np.all(np.diff(y) < 0)),
            }
        )
    return pd.DataFrame(records)


def evaluate(args: argparse.Namespace) -> None:
    experiment = Path(args.experiment_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = pd.read_csv(experiment / "scenario_manifest.csv")
    seeds = parse_seeds(args.seeds)
    expected = len(seeds) * 4
    if len(manifest) != expected or sorted(manifest["seed"].unique()) != sorted(seeds):
        raise ValueError("Evaluation manifest does not match the six-seed four-anchor grid")

    overall_records = []
    budget_records = []
    label_records = []
    auroc_records = []
    quality_records = []
    bootstrap_parts = []
    prevalence_parts = []
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
        bootstrap_parts.append(bootstrap_scenario(merged, metadata, args.bootstrap_iterations))
        counts = pd.read_csv(prepared / "corruption_counts.csv")
        counts["array_index"] = int(row.array_index)
        prevalence_parts.append(counts)
        print(f"[evaluate] {row.array_index + 1}/{len(manifest)} {row.scenario_id} seed={row.seed}", flush=True)

    overall = pd.DataFrame(overall_records)
    budgets = pd.DataFrame(budget_records)
    labels = pd.DataFrame(label_records)
    aurocs = pd.DataFrame(auroc_records)
    quality = pd.DataFrame(quality_records)
    bootstrap = pd.concat(bootstrap_parts, ignore_index=True)
    prevalence = pd.concat(prevalence_parts, ignore_index=True)
    calibration = calibration_table(quality)

    noisy_overall = overall[(overall["noise_rate"] > 0) & (overall["method"] == "cl_first")]
    noisy_quality = quality[quality["noise_rate"] > 0]
    true_budget = budgets[
        (budgets["budget_name"] == "true_error_count") & (budgets["method"] == "cl_first")
    ].copy()
    true_budget["recall_minus_random"] = true_budget["recall"] - true_budget["random_expected_recall"]
    seed_detection = noisy_overall.assign(
        auprc_minus_prevalence=noisy_overall["auprc"] - noisy_overall["prevalence"]
    ).groupby(["seed", "seed_block"], as_index=False).agg(
        mean_auprc_minus_prevalence=("auprc_minus_prevalence", "mean")
    )
    seed_recall = true_budget.groupby(["seed", "seed_block"], as_index=False).agg(
        mean_recall_minus_random=("recall_minus_random", "mean")
    )
    seed_detection = seed_detection.merge(seed_recall, on=["seed", "seed_block"], validate="one_to_one")
    p_auprc = exact_sign_flip_p(seed_detection["mean_auprc_minus_prevalence"].to_numpy())
    p_recall = exact_sign_flip_p(seed_detection["mean_recall_minus_random"].to_numpy())
    seed_detection["six_seed_exact_p_auprc_lift"] = p_auprc
    seed_detection["six_seed_exact_p_recall_lift"] = p_recall

    atomic_write_csv(overall, output / "overall_detection_metrics.csv")
    atomic_write_csv(budgets, output / "budget_metrics.csv")
    atomic_write_csv(labels, output / "per_label_direction_metrics.csv")
    atomic_write_csv(aurocs, output / "oof_label_aurocs.csv")
    atomic_write_csv(quality, output / "quality_summary.csv")
    atomic_write_csv(bootstrap, output / "image_cluster_bootstrap.csv")
    atomic_write_csv(prevalence, output / "corruption_prevalence.csv")
    atomic_write_csv(calibration, output / "dqs_calibration.csv")
    atomic_write_csv(seed_detection, output / "detection_lift_by_seed.csv")

    gates = {
        "auprc_above_error_prevalence_all_noisy_runs": bool((noisy_overall["auprc"] > noisy_overall["prevalence"]).all()),
        "hard_enrichment_above_one_all_noisy_runs": bool((noisy_quality["hard_enrichment"] > 1).all()),
        "true_error_budget_recall_above_random_all_noisy_runs": bool((true_budget["recall"] > true_budget["random_expected_recall"]).all()),
        "dqs_strictly_decreases_all_seeds": bool(calibration["strictly_decreasing_with_noise"].all()),
        "dqs_spearman_at_least_0_95_all_seeds": bool((calibration["spearman_rho"] >= 0.95).all()),
    }
    summary = {
        "protocol": PROTOCOL_NAME,
        "verification_status": "ANALYZED",
        "seeds": seeds,
        "scenarios": 4,
        "blind_runs": len(manifest),
        "known_true_quality_anchors": [1.0, 0.9, 0.8, 0.7],
        "gates": gates,
        "six_seed_exact_p_auprc_lift": p_auprc,
        "six_seed_exact_p_recall_lift": p_recall,
        "dqs_calibration_worst_mae": float(calibration["mae"].max()),
        "dqs_calibration_min_spearman": float(calibration["spearman_rho"].min()),
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(output / "evaluation_summary.json", json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    plot = quality.groupby("true_quality")["raw_entry_dqs"].agg(["mean", "std"]).sort_index()
    ax.errorbar(plot.index, plot["mean"], yerr=plot["std"], marker="o", capsize=4, label="Raw entry DQS")
    bounds = [quality["true_quality"].min(), 1.0]
    ax.plot(bounds, bounds, linestyle="--", color="gray", label="Ideal calibration")
    ax.set(xlabel="Known true entry quality", ylabel="Raw entry DQS", title="DQS Calibration Under Exact Global Noise")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "dqs_calibration.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    plot = noisy_overall.groupby("noise_rate").agg(
        auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"), prevalence=("prevalence", "mean")
    )
    x = plot.index.to_numpy() * 100
    ax.errorbar(x, plot["auprc_mean"], yerr=plot["auprc_sd"], marker="o", capsize=4, label="CL ranking AUPRC")
    ax.plot(x, plot["prevalence"], linestyle="--", marker="o", label="Random error prevalence")
    ax.set(xlabel="Exact global entry noise (%)", ylabel="Injected-error AUPRC", title="CL Detection Under Global Symmetric Noise")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "cl_detection_auprc.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    plot = prevalence.groupby(["noise_rate", "label_name"])["noisy_positive_prevalence"].mean().reset_index()
    for label, frame in plot.groupby("label_name", sort=False):
        ax.plot(frame["noise_rate"] * 100, frame["noisy_positive_prevalence"], marker="o", label=label)
    ax.set(xlabel="Exact global entry noise (%)", ylabel="Observed positive prevalence", title="Prevalence Shift Caused by Symmetric Entry Flips")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "positive_prevalence_shift.png", dpi=180)
    plt.close(fig)

    atomic_write_text(output / ".evaluation_complete", "complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def verify(args: argparse.Namespace) -> None:
    experiment = Path(args.experiment_root)
    evaluation = Path(args.evaluation_dir) if args.evaluation_dir else None
    manifest = pd.read_csv(experiment / "scenario_manifest.csv")
    seeds = parse_seeds(args.seeds)
    if len(manifest) != 24 or sorted(manifest["seed"].unique()) != sorted(seeds):
        raise ValueError("Scenario manifest must contain the exact 24-run grid")
    if manifest[["scenario_id", "seed"]].duplicated().any() or manifest["scenario_id"].nunique() != 4:
        raise ValueError("Scenario manifest keys are incomplete or duplicated")
    expected_errors = {0: 0, 10: 1_800, 20: 3_600, 30: 5_400}
    for row in manifest.itertuples(index=False):
        prepared = experiment / row.prepared_relpath
        blind_run = experiment / row.blind_run_relpath
        if not (prepared / ".prepare_complete").is_file() or not (blind_run / ".blind_run_complete").is_file():
            raise FileNotFoundError(f"Completion marker missing: {row.scenario_id}, seed={row.seed}")
        blind = pd.read_csv(prepared / "blind_noisy_cohort.csv")
        private = pd.read_csv(prepared / "private_reference.csv")
        evidence = pd.read_csv(blind_run / "entry_evidence.csv")
        validate_blind_columns(blind)
        validate_blind_columns(evidence.drop(columns=["cl_issue"], errors="ignore"))
        if len(blind) != 3_000 or len(private) != 18_000 or len(evidence) != 18_000:
            raise ValueError(f"Row-count verification failed: {row.scenario_id}, seed={row.seed}")
        if int(private["injected_error"].sum()) != expected_errors[int(row.rate_percent)]:
            raise ValueError(f"Injected-error count changed: {row.scenario_id}, seed={row.seed}")

    if evaluation:
        if not (evaluation / ".evaluation_complete").is_file():
            raise FileNotFoundError("Evaluation marker is missing")
        summary = json.loads((evaluation / "evaluation_summary.json").read_text())
        if summary["seeds"] != seeds or summary["blind_runs"] != 24:
            raise ValueError("Evaluation summary grid does not match verification request")
        expected_rows = {
            "overall_detection_metrics.csv": 72,
            "budget_metrics.csv": 270,
            "per_label_direction_metrics.csv": 1_296,
            "oof_label_aurocs.csv": 144,
            "quality_summary.csv": 24,
            "image_cluster_bootstrap.csv": 24_000,
            "corruption_prevalence.csv": 144,
            "dqs_calibration.csv": 6,
            "detection_lift_by_seed.csv": 6,
        }
        for name, expected in expected_rows.items():
            actual = len(pd.read_csv(evaluation / name))
            if actual != expected:
                raise ValueError(f"{name} has {actual} rows; expected {expected}")
        for name in ["dqs_calibration.png", "cl_detection_auprc.png", "positive_prevalence_shift.png"]:
            if (evaluation / name).stat().st_size <= 1_000:
                raise ValueError(f"Figure is missing or empty: {name}")
    atomic_write_text(experiment / ".benchmark_verified", "complete\n")
    print(json.dumps({"protocol": PROTOCOL_NAME, "verification": "passed", "blind_runs": 24}))


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
