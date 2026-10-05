#!/usr/bin/env python3
"""Compare MobileNet OOF/CL, XRV OOF/CL, direct XRV, and fixed rank fusion."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vindr_detector_benchmark_v2 import (
    atomic_write_json,
    detector_metrics,
    merge_private,
    parse_int_list,
    stable_top,
    validate_no_outcomes,
)
from vindr_known_gt_cl_benchmark import (
    atomic_write_csv,
    atomic_write_text,
    exact_sign_flip_p,
    holm_adjust,
    require_columns,
    sha256_file,
)


PROTOCOL_NAME = "vindr_xrv_oof_comparison_v1"
METHOD_COLUMNS = {
    "mobilenet_oof_cl": "score_mobilenet_oof_cl",
    "xrv_oof_cl": "score_xrv_oof_cl",
    "direct_xrv": "score_direct_xrv",
    "xrv_oof_direct_fusion": "score_xrv_oof_direct_fusion",
}


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return deterministic average percentile ranks in [0, 1]."""
    series = pd.Series(np.asarray(values, dtype=float))
    if not np.isfinite(series.to_numpy()).all() or series.empty:
        raise ValueError("Percentile ranking requires finite non-empty values")
    return series.rank(method="average", pct=True).to_numpy(float)


def fixed_rank_fusion(xrv_oof_score: np.ndarray, direct_xrv_score: np.ndarray) -> np.ndarray:
    """Equal-weight global percentile-rank fusion fixed before private evaluation."""
    first = percentile_rank(xrv_oof_score)
    second = percentile_rank(direct_xrv_score)
    return 0.5 * first + 0.5 * second


def selected_manifest(args: argparse.Namespace) -> pd.DataFrame:
    parent = Path(args.parent_root)
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
            "prepared_relpath",
            "blind_run_relpath",
        ],
        "blind run manifest",
    )
    validate_no_outcomes(manifest, "blind run manifest")
    if args.seeds:
        manifest = manifest[manifest["seed"].isin(parse_int_list(args.seeds))]
    if args.scenarios:
        requested = {value.strip() for value in args.scenarios.split(",") if value.strip()}
        manifest = manifest[manifest["scenario_id"].isin(requested)]
    manifest = manifest.sort_values("array_index").reset_index(drop=True)
    if manifest.empty:
        raise ValueError("Selected manifest is empty")
    return manifest


def blind_join(
    mobile_scores: pd.DataFrame,
    xrv_evidence: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        mobile_scores,
        [
            "image_id",
            "fold_id",
            "label_index",
            "label_name",
            "noisy_label",
            "score_oof_cl",
            "score_external_xrv",
        ],
        "MobileNet/direct-XRV scores",
    )
    require_columns(
        xrv_evidence,
        [
            "image_id",
            "fold_id",
            "label_index",
            "label_name",
            "noisy_label",
            "cl_first_score",
        ],
        "XRV OOF evidence",
    )
    validate_no_outcomes(mobile_scores, "MobileNet/direct-XRV scores")
    validate_no_outcomes(xrv_evidence, "XRV OOF evidence")
    keys = ["image_id", "label_name"]
    left = mobile_scores.copy()
    right = xrv_evidence.copy()
    left["image_id"] = left["image_id"].astype(str)
    right["image_id"] = right["image_id"].astype(str)
    merged = left.merge(
        right[
            keys
            + ["fold_id", "label_index", "noisy_label", "cl_first_score"]
        ],
        on=keys,
        validate="one_to_one",
        suffixes=("_mobile", "_xrv"),
    )
    if len(merged) != 18_000:
        raise ValueError("Blind score sources do not align to 18,000 entries")
    for column in ["fold_id", "label_index", "noisy_label"]:
        if not np.array_equal(
            merged[f"{column}_mobile"].to_numpy(), merged[f"{column}_xrv"].to_numpy()
        ):
            raise ValueError(f"Blind score sources disagree on {column}")
    output = pd.DataFrame(
        {
            "image_id": merged["image_id"],
            "fold_id": merged["fold_id_mobile"].astype(int),
            "label_index": merged["label_index_mobile"].astype(int),
            "label_name": merged["label_name"],
            "noisy_label": merged["noisy_label_mobile"].astype(int),
            "score_mobilenet_oof_cl": merged["score_oof_cl"].astype(float),
            "score_xrv_oof_cl": merged["cl_first_score"].astype(float),
            "score_direct_xrv": merged["score_external_xrv"].astype(float),
        }
    )
    output["score_xrv_oof_direct_fusion"] = fixed_rank_fusion(
        output["score_xrv_oof_cl"].to_numpy(),
        output["score_direct_xrv"].to_numpy(),
    )
    validate_no_outcomes(output, "comparison blind scores")
    if not np.isfinite(output[list(METHOD_COLUMNS.values())].to_numpy()).all():
        raise RuntimeError("Comparison scores contain non-finite values")
    return output


def score(args: argparse.Namespace) -> None:
    parent = Path(args.parent_root)
    mobile_root = Path(args.mobile_score_root)
    xrv_root = Path(args.xrv_oof_root)
    output = Path(args.output_root)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison output: {output}")
    if not (parent / ".prepare_complete").is_file():
        raise FileNotFoundError("Parent preparation marker is missing")
    if not (mobile_root / ".benchmark_verified").is_file():
        raise FileNotFoundError("Verified MobileNet/direct-XRV score root is missing")
    manifest = selected_manifest(args)
    mobile_manifest = pd.read_csv(mobile_root / "score_manifest.csv")
    lookup = {
        (str(row.scenario_id), int(row.seed)): row
        for row in mobile_manifest.itertuples(index=False)
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    records = []
    try:
        for row in manifest.itertuples(index=False):
            key = (str(row.scenario_id), int(row.seed))
            if key not in lookup:
                raise KeyError(f"Missing MobileNet/direct-XRV score run: {key}")
            mobile_row = lookup[key]
            mobile_path = mobile_root / mobile_row.score_relpath
            if sha256_file(mobile_path) != mobile_row.score_sha256:
                raise RuntimeError(f"Mobile score hash mismatch: {mobile_path}")
            xrv_path = (
                xrv_root
                / "scenarios"
                / str(row.scenario_id)
                / f"seed_{int(row.seed)}"
                / "blind_run"
                / "entry_evidence.csv"
            )
            xrv_marker = xrv_path.parent / ".blind_run_complete"
            if not xrv_marker.is_file():
                raise FileNotFoundError(f"XRV OOF completion marker is missing: {xrv_marker}")
            frame = blind_join(pd.read_csv(mobile_path), pd.read_csv(xrv_path))
            relative = Path("blind_scores") / str(row.scenario_id) / f"seed_{int(row.seed)}.csv"
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_csv(frame, destination)
            records.append(
                {
                    "array_index": int(row.array_index),
                    "scenario_id": str(row.scenario_id),
                    "regime": str(row.regime),
                    "noise_rate": float(row.noise_rate),
                    "rate_percent": int(row.rate_percent),
                    "seed": int(row.seed),
                    "prepared_relpath": str(row.prepared_relpath),
                    "score_relpath": str(relative),
                    "score_sha256": sha256_file(destination),
                    "mobile_score_sha256": sha256_file(mobile_path),
                    "xrv_evidence_sha256": sha256_file(xrv_path),
                    "rows": int(len(frame)),
                }
            )
            print(f"joined scenario={row.scenario_id} seed={int(row.seed)}", flush=True)
        score_manifest = pd.DataFrame(records).sort_values("array_index")
        atomic_write_csv(score_manifest, temporary / "score_manifest.csv")
        atomic_write_json(
            {
                "protocol": PROTOCOL_NAME,
                "outcome_blind": True,
                "runs": int(len(score_manifest)),
                "methods": list(METHOD_COLUMNS),
                "fusion": "equal-weight global percentile ranks of XRV OOF/CL and direct XRV",
                "program_sha256": sha256_file(Path(__file__)),
            },
            temporary / "scoring_summary.json",
        )
        atomic_write_text(temporary / ".blind_scoring_complete", "complete\n")
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def evaluate_run(frame: pd.DataFrame, metadata: dict) -> list[dict]:
    total_errors = int(frame["injected_error"].sum())
    if total_errors <= 0:
        return []
    records = []
    for method, score_column in METHOD_COLUMNS.items():
        records.append(detector_metrics(frame, method, score_column, total_errors, metadata, "overall"))
        for direction, observed_label in [("0_to_1", 1), ("1_to_0", 0)]:
            queue = frame[frame["noisy_label_blind"] == observed_label].copy()
            queue["injected_error"] = queue["flip_direction"].eq(direction).astype(int)
            direction_errors = int(queue["injected_error"].sum())
            if direction_errors:
                records.append(
                    detector_metrics(
                        queue,
                        method,
                        score_column,
                        direction_errors,
                        metadata,
                        direction,
                    )
                )
    return records


def primary_contrasts(metrics: pd.DataFrame, expected_seeds: int = 6) -> pd.DataFrame:
    primary = metrics[
        metrics["regime"].eq("balanced")
        & metrics["rate_percent"].eq(20)
        & metrics["scope"].eq("overall")
    ]
    pivot = primary.pivot(index="seed", columns="method", values="recall")
    if set(METHOD_COLUMNS) != set(pivot.columns) or len(pivot) != expected_seeds:
        raise RuntimeError(
            "Primary endpoint does not contain the expected paired seeds and four methods"
        )
    records = []
    reference = pivot["mobilenet_oof_cl"]
    for method in ["xrv_oof_cl", "direct_xrv", "xrv_oof_direct_fusion"]:
        differences = (pivot[method] - reference).to_numpy(float)
        records.append(
            {
                "endpoint": "balanced_r20 overall matched-budget recall",
                "method": method,
                "reference_method": "mobilenet_oof_cl",
                "paired_seeds": int(len(differences)),
                "method_mean": float(pivot[method].mean()),
                "reference_mean": float(reference.mean()),
                "mean_difference": float(differences.mean()),
                "positive_seeds": int(np.sum(differences > 0)),
                "negative_seeds": int(np.sum(differences < 0)),
                "exact_sign_flip_p": exact_sign_flip_p(differences),
            }
        )
    result = pd.DataFrame(records)
    result["holm_p_across_three_primary_contrasts"] = holm_adjust(
        result["exact_sign_flip_p"].to_numpy(float)
    )
    return result


def plot_primary(metrics: pd.DataFrame, output: Path) -> None:
    selected = metrics[
        metrics["regime"].eq("balanced")
        & metrics["rate_percent"].eq(20)
        & metrics["scope"].isin(["overall", "1_to_0"])
    ]
    summary = selected.groupby(["scope", "method"])["recall"].agg(["mean", "std"]).reset_index()
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)
    order = list(METHOD_COLUMNS)
    labels = ["MobileNet\nOOF/CL", "XRV\nOOF/CL", "Direct\nXRV", "XRV OOF +\nDirect"]
    colors = ["#2563eb", "#059669", "#dc2626", "#7c3aed"]
    for axis, scope, title in zip(
        axes,
        ["overall", "1_to_0"],
        ["Overall errors", "Missing-positive errors (1 to 0)"],
    ):
        frame = summary[summary["scope"].eq(scope)].set_index("method").loc[order]
        axis.bar(
            range(len(order)),
            frame["mean"],
            yerr=frame["std"].fillna(0),
            color=colors,
            capsize=3,
        )
        axis.set_xticks(range(len(order)), labels)
        axis.set_ylabel("Recall at matched review budget")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("VinDr balanced 20% noise, six new seeds")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def evaluate(args: argparse.Namespace) -> None:
    parent = Path(args.parent_root)
    root = Path(args.output_root)
    evaluation = root / "evaluation"
    if not (root / ".blind_scoring_complete").is_file():
        raise FileNotFoundError("Blind scoring marker is missing")
    if evaluation.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {evaluation}")
    manifest = pd.read_csv(root / "score_manifest.csv")
    records = []
    for row in manifest.itertuples(index=False):
        score_path = root / row.score_relpath
        if sha256_file(score_path) != row.score_sha256:
            raise RuntimeError(f"Comparison score hash mismatch: {score_path}")
        merged = merge_private(
            pd.read_csv(score_path),
            parent / row.prepared_relpath / "private_reference.csv",
        )
        metadata = {
            "array_index": int(row.array_index),
            "scenario_id": str(row.scenario_id),
            "regime": str(row.regime),
            "noise_rate": float(row.noise_rate),
            "rate_percent": int(row.rate_percent),
            "seed": int(row.seed),
        }
        records.extend(evaluate_run(merged, metadata))
        print(f"evaluated scenario={row.scenario_id} seed={int(row.seed)}", flush=True)
    metrics = pd.DataFrame(records)
    if metrics.empty:
        raise RuntimeError("Private evaluation produced no metrics")
    expected_primary_seeds = 1 if args.smoke_mode else 6
    contrasts = primary_contrasts(metrics, expected_primary_seeds)
    evaluation.mkdir(parents=True)
    atomic_write_csv(metrics, evaluation / "detector_metrics.csv")
    atomic_write_csv(contrasts, evaluation / "primary_contrasts.csv")
    plot_primary(metrics, evaluation / "primary_comparison.png")
    atomic_write_json(
        {
            "protocol": PROTOCOL_NAME,
            "private_evaluation": True,
            "smoke_mode": bool(args.smoke_mode),
            "runs": int(len(manifest)),
            "noisy_runs": int(manifest[manifest["regime"] != "clean"].shape[0]),
            "primary_endpoint": "balanced 20% overall recall at injected-error-count review budget",
            "primary_contrasts": contrasts.to_dict("records"),
            "program_sha256": sha256_file(Path(__file__)),
        },
        evaluation / "evaluation_summary.json",
    )
    atomic_write_text(evaluation / ".evaluation_complete", "complete\n")


def verify(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    evaluation = root / "evaluation"
    for path in [
        root / ".blind_scoring_complete",
        root / "score_manifest.csv",
        root / "scoring_summary.json",
        evaluation / ".evaluation_complete",
        evaluation / "detector_metrics.csv",
        evaluation / "primary_contrasts.csv",
        evaluation / "primary_comparison.png",
        evaluation / "evaluation_summary.json",
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = pd.read_csv(root / "score_manifest.csv")
    if len(manifest) != args.expected_runs or manifest[["scenario_id", "seed"]].duplicated().any():
        raise RuntimeError("Comparison manifest count or uniqueness failed")
    for row in manifest.itertuples(index=False):
        path = root / row.score_relpath
        if sha256_file(path) != row.score_sha256:
            raise RuntimeError(f"Comparison score verification failed: {path}")
        frame = pd.read_csv(path)
        validate_no_outcomes(frame, "verified comparison scores")
        if len(frame) != 18_000 or not set(METHOD_COLUMNS.values()).issubset(frame.columns):
            raise RuntimeError(f"Comparison score schema failed: {path}")
    metrics = pd.read_csv(evaluation / "detector_metrics.csv")
    noisy_runs = int((manifest["regime"] != "clean").sum())
    expected_direction_scopes = sum(
        2 if regime == "balanced" else 1
        for regime in manifest.loc[manifest["regime"] != "clean", "regime"]
    )
    expected_rows = noisy_runs * len(METHOD_COLUMNS) + expected_direction_scopes * len(
        METHOD_COLUMNS
    )
    if len(metrics) != expected_rows or not np.isfinite(metrics[["recall", "auprc"]]).all().all():
        raise RuntimeError("Comparison metric count or finiteness failed")
    contrasts = pd.read_csv(evaluation / "primary_contrasts.csv")
    expected_primary_seeds = 1 if args.smoke_mode else 6
    if len(contrasts) != 3 or not (
        contrasts["paired_seeds"] == expected_primary_seeds
    ).all():
        raise RuntimeError("Primary contrast verification failed")
    atomic_write_text(root / ".comparison_verified", "verified\n")
    print(json.dumps({"verified": True, "runs": len(manifest), "metric_rows": len(metrics)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--parent-root", type=Path, required=True)
    score_parser.add_argument("--mobile-score-root", type=Path, required=True)
    score_parser.add_argument("--xrv-oof-root", type=Path, required=True)
    score_parser.add_argument("--output-root", type=Path, required=True)
    score_parser.add_argument("--seeds", default="")
    score_parser.add_argument("--scenarios", default="")
    score_parser.set_defaults(function=score)
    for name, function in [("evaluate", evaluate), ("verify", verify)]:
        command = subparsers.add_parser(name)
        command.add_argument("--parent-root", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        if name == "verify":
            command.add_argument("--expected-runs", type=int, required=True)
        command.add_argument("--smoke-mode", action="store_true")
        command.set_defaults(function=function)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
