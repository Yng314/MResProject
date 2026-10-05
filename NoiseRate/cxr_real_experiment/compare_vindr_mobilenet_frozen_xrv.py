#!/usr/bin/env python3
"""Paired MobileNet-versus-frozen-XRV analysis for the VinDr benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vindr_known_gt_cl_benchmark import atomic_write_csv, atomic_write_text, exact_sign_flip_p, sha256_file


SEEDS = [13, 42, 97, 123, 211, 307]
PAIR_KEYS = ["scenario_id", "regime", "noise_rate", "rate_percent", "seed", "seed_block"]


def require_grid(frame: pd.DataFrame, rows: int, name: str, extra_keys: list[str] | None = None) -> None:
    keys = PAIR_KEYS + (extra_keys or [])
    missing = [column for column in keys if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} lacks columns: {missing}")
    if len(frame) != rows or frame[keys].duplicated().any():
        raise ValueError(f"{name} is not the expected unique {rows}-row grid")
    if sorted(frame["seed"].unique().tolist()) != SEEDS:
        raise ValueError(f"{name} seed grid is incomplete")


def primary_decision(fn_by_seed: pd.DataFrame, clean_auroc_by_seed: pd.DataFrame) -> dict[str, object]:
    if len(fn_by_seed) != 6 or len(clean_auroc_by_seed) != 6:
        raise ValueError("Primary decision requires six paired seeds")
    differences = fn_by_seed["mobilenet_minus_frozen_fn_recall"].to_numpy(dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError("Primary FN-recall differences are not finite")
    exact_p = exact_sign_flip_p(differences)
    mean_improvement = float(differences.mean())
    clean_difference = float(clean_auroc_by_seed["mobilenet_minus_frozen_clean_auroc"].mean())
    gates = {
        "positive_in_all_six_seeds": bool((differences > 0).all()),
        "mean_fn_recall_improvement_at_least_0_10": bool(mean_improvement >= 0.10),
        "two_sided_exact_p_below_0_05": bool(exact_p < 0.05),
        "clean_reference_auroc_not_more_than_0_02_lower": bool(clean_difference >= -0.02),
    }
    return {
        "mean_mobilenet_minus_frozen_fn_recall": mean_improvement,
        "two_sided_six_seed_exact_sign_flip_p": exact_p,
        "positive_seeds": int((differences > 0).sum()),
        "mean_mobilenet_minus_frozen_clean_reference_macro_auroc": clean_difference,
        "gates": gates,
        "model_limitation_explanation_supported": bool(all(gates.values())),
    }


def compare(args: argparse.Namespace) -> None:
    mobile = Path(args.mobilenet_evaluation)
    frozen = Path(args.frozen_evaluation)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for root, name in [(mobile, "MobileNet"), (frozen, "frozen-XRV")]:
        if not (root / ".evaluation_complete").is_file():
            raise FileNotFoundError(f"{name} evaluation marker is missing")

    mobile_quality = pd.read_csv(mobile / "quality_summary.csv")
    frozen_quality = pd.read_csv(frozen / "quality_summary.csv")
    require_grid(mobile_quality, 60, "MobileNet quality")
    require_grid(frozen_quality, 60, "frozen-XRV quality")
    quality = mobile_quality.merge(
        frozen_quality,
        on=PAIR_KEYS,
        suffixes=("__mobilenet", "__frozen"),
        validate="one_to_one",
    )
    if len(quality) != 60:
        raise ValueError("Quality grids are not paired one-to-one")
    for column in ["true_errors", "true_quality"]:
        left = quality[f"{column}__mobilenet"].to_numpy(dtype=float)
        right = quality[f"{column}__frozen"].to_numpy(dtype=float)
        if not np.allclose(left, right, atol=0, rtol=0):
            raise ValueError(f"Paired private truth differs between models: {column}")
    quality["mobilenet_minus_frozen_hard_recall"] = (
        quality["hard_recall__mobilenet"] - quality["hard_recall__frozen"]
    )
    quality["mobilenet_minus_frozen_raw_dqs"] = (
        quality["raw_entry_dqs__mobilenet"] - quality["raw_entry_dqs__frozen"]
    )

    fn = quality[(quality["regime"] == "fn_only") & (quality["noise_rate"] > 0)]
    if len(fn) != 18:
        raise ValueError("Paired false-negative scenario grid is incomplete")
    fn_by_seed = fn.groupby(["seed", "seed_block"], as_index=False).agg(
        mobilenet_mean_fn_recall=("hard_recall__mobilenet", "mean"),
        frozen_mean_fn_recall=("hard_recall__frozen", "mean"),
    )
    fn_by_seed["mobilenet_minus_frozen_fn_recall"] = (
        fn_by_seed["mobilenet_mean_fn_recall"] - fn_by_seed["frozen_mean_fn_recall"]
    )

    mobile_auroc = pd.read_csv(mobile / "oof_label_aurocs.csv")
    frozen_auroc = pd.read_csv(frozen / "oof_label_aurocs.csv")
    require_grid(mobile_auroc, 360, "MobileNet AUROC", ["label_name"])
    require_grid(frozen_auroc, 360, "frozen-XRV AUROC", ["label_name"])
    auroc = mobile_auroc.merge(
        frozen_auroc,
        on=PAIR_KEYS + ["label_name"],
        suffixes=("__mobilenet", "__frozen"),
        validate="one_to_one",
    )
    if len(auroc) != 360:
        raise ValueError("AUROC grids are not paired one-to-one")
    auroc["mobilenet_minus_frozen_clean_label_auroc"] = (
        auroc["clean_label_auroc__mobilenet"] - auroc["clean_label_auroc__frozen"]
    )
    clean = auroc[auroc["regime"] == "clean"]
    if len(clean) != 36:
        raise ValueError("Clean-reference label AUROC grid is incomplete")
    clean_by_seed = clean.groupby(["seed", "seed_block"], as_index=False).agg(
        mobilenet_clean_reference_macro_auroc=("clean_label_auroc__mobilenet", "mean"),
        frozen_clean_reference_macro_auroc=("clean_label_auroc__frozen", "mean"),
    )
    clean_by_seed["mobilenet_minus_frozen_clean_auroc"] = (
        clean_by_seed["mobilenet_clean_reference_macro_auroc"]
        - clean_by_seed["frozen_clean_reference_macro_auroc"]
    )

    primary = primary_decision(fn_by_seed, clean_by_seed)
    mobile_direction = pd.read_csv(mobile / "direction_gap_by_seed.csv")
    if len(mobile_direction) != 6 or sorted(mobile_direction["seed"].tolist()) != SEEDS:
        raise ValueError("MobileNet direction-gap grid is incomplete")
    direction_p = exact_sign_flip_p(mobile_direction["mean_direction_gap"].to_numpy(dtype=float))

    mobile_overall = pd.read_csv(mobile / "overall_detection_metrics.csv")
    mobile_budgets = pd.read_csv(mobile / "budget_metrics.csv")
    noisy_overall = mobile_overall[(mobile_overall["noise_rate"] > 0) & (mobile_overall["method"] == "cl_first")]
    noisy_quality = mobile_quality[mobile_quality["noise_rate"] > 0]
    true_budget = mobile_budgets[
        (mobile_budgets["budget_name"] == "true_error_count")
        & (mobile_budgets["method"] == "cl_first")
    ]
    calibration = pd.read_csv(mobile / "dqs_calibration.csv")
    secondary = {
        "mobilenet_direction_gap_mean": float(mobile_direction["mean_direction_gap"].mean()),
        "mobilenet_direction_exact_sign_flip_p": direction_p,
        "mobilenet_directional_asymmetry_confirmed": bool(
            (mobile_direction["mean_direction_gap"] > 0.20).all() and direction_p < 0.05
        ),
        "mobilenet_auprc_above_prevalence_all_noisy_runs": bool(
            (noisy_overall["auprc"] > noisy_overall["prevalence"]).all()
        ),
        "mobilenet_hard_enrichment_above_one_all_noisy_runs": bool(
            (noisy_quality["hard_enrichment"] > 1).all()
        ),
        "mobilenet_true_error_budget_recall_above_random_all_noisy_runs": bool(
            (true_budget["recall"] > true_budget["random_expected_recall"]).all()
        ),
        "mobilenet_dqs_calibration_supported": bool(
            (calibration["spearman_rho"] >= 0.95).all()
            and (calibration["mae"] <= 0.01).all()
            and calibration["strictly_decreasing_with_noise"].all()
        ),
    }

    atomic_write_csv(quality, output / "paired_scenario_quality.csv")
    atomic_write_csv(auroc, output / "paired_label_auroc.csv")
    atomic_write_csv(fn_by_seed, output / "primary_fn_recall_by_seed.csv")
    atomic_write_csv(clean_by_seed, output / "clean_reference_auroc_by_seed.csv")
    summary = {
        "protocol": "vindr_mobilenet_direction_sensitivity_v1",
        "comparison": "mobilenet_v3_small_scratch_minus_frozen_xrv_linear_head",
        "seeds": SEEDS,
        "primary": primary,
        "secondary": secondary,
        "mobilenet_evaluation": str(mobile),
        "frozen_evaluation": str(frozen),
        "mobilenet_quality_sha256": sha256_file(mobile / "quality_summary.csv"),
        "frozen_quality_sha256": sha256_file(frozen / "quality_summary.csv"),
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(output / "paired_comparison_summary.json", json.dumps(summary, indent=2))
    atomic_write_text(output / ".paired_comparison_complete", "complete\n")
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mobilenet-evaluation", type=Path, required=True)
    parser.add_argument("--frozen-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    compare(build_parser().parse_args())


if __name__ == "__main__":
    main()
