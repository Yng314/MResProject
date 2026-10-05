#!/usr/bin/env python3
"""Evaluate frozen CL evidence against two independent expert references."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


ESTIMATORS = ["seed_13", "seed_42", "seed_97", "seed_123", "ensemble"]
EXPECTED_SCORE_ROWS_PER_ESTIMATOR = 8_288
EXPECTED_REFERENCES = {
    "mimic_cxr_2_1": {"entries": 1_796, "issues": 109, "studies": 563, "subjects": 251},
    "medpalm": {"entries": 498, "issues": 127, "studies": 457, "subjects": 179},
}
PROHIBITED_BLIND_COLUMNS = {
    "reader_label",
    "reader_agreement",
    "reference_binary_label",
    "expected_binary_action",
    "true_issue",
}
BOOTSTRAP_METRICS = [
    "prevalence",
    "auprc",
    "auprc_minus_prevalence",
    "auroc",
    "hard_precision",
    "hard_recall",
    "hard_enrichment",
    "policy_precision",
    "policy_recall",
    "policy_enrichment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["select", "evaluate", "verify"])
    parser.add_argument("--score-csv", type=Path, required=True)
    parser.add_argument("--score-manifest", type=Path, required=True)
    parser.add_argument("--mimic-reference-csv", type=Path)
    parser.add_argument("--medpalm-blinded-csv", type=Path)
    parser.add_argument("--medpalm-reference-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--random-replicates", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20_260_803)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def require_file(path: Path | None, argument: str) -> Path:
    if path is None or not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {argument}: {path}")
    return path


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def finite_metric(function, y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(function(y_true, y_score))


def load_scores(score_csv: Path, score_manifest: Path) -> pd.DataFrame:
    manifest = json.loads(require_file(score_manifest, "--score-manifest").read_text())
    if manifest.get("expert_reference_used") is not False:
        raise ValueError("Frozen score manifest is not outcome blind")
    if manifest.get("score_sha256") != sha256_file(require_file(score_csv, "--score-csv")):
        raise ValueError("Frozen score hash differs from its manifest")

    scores = pd.read_csv(score_csv)
    required = {
        "subject_id",
        "study_id",
        "entry_key",
        "label_name",
        "current_binary_label",
        "quality_self_confidence",
        "suspicion_percentile_self",
        "cl_hard_issue",
        "estimator",
    }
    if not required.issubset(scores.columns):
        raise ValueError(f"Frozen score columns missing: {sorted(required - set(scores.columns))}")
    if PROHIBITED_BLIND_COLUMNS.intersection(scores.columns):
        raise ValueError("Frozen score file contains prohibited reference columns")
    if scores["estimator"].value_counts().to_dict() != {
        estimator: EXPECTED_SCORE_ROWS_PER_ESTIMATOR for estimator in ESTIMATORS
    }:
        raise ValueError("Unexpected frozen score row counts")
    if scores.duplicated(["estimator", "entry_key"]).any():
        raise ValueError("Frozen score file contains duplicate estimator-entry rows")
    base_keys: set[str] | None = None
    for estimator in ESTIMATORS:
        keys = set(scores.loc[scores["estimator"].eq(estimator), "entry_key"].astype(str))
        if base_keys is None:
            base_keys = keys
        elif keys != base_keys:
            raise ValueError("Estimators do not share the same candidate entry keys")
    if not scores["cl_hard_issue"].isin([0, 1]).all():
        raise ValueError("CL hard flags are not binary")
    return scores


def select_stage(args: argparse.Namespace) -> None:
    if not (0 < args.top_fraction <= 1):
        raise ValueError("--top-fraction must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / ".evaluation_complete").exists():
        raise RuntimeError("Refusing to modify a completed evaluation")

    scores = load_scores(args.score_csv, args.score_manifest)
    sample_rows: list[pd.DataFrame] = []
    expanded_rows: list[pd.DataFrame] = []
    selection_counts: dict[str, dict[str, int]] = {}

    for estimator in ESTIMATORS:
        frame = scores[scores["estimator"].eq(estimator)].copy()
        studies = (
            frame.groupby(["subject_id", "study_id"], as_index=False)
            .agg(
                valid_entry_count=("entry_key", "size"),
                sample_quality_self_confidence=("quality_self_confidence", "mean"),
                est_issue_entry_count=("cl_hard_issue", "sum"),
            )
        )
        studies["est_issue_sample"] = studies["est_issue_entry_count"].gt(0).astype(int)
        pool = studies[studies["est_issue_sample"].eq(1)].copy()
        pool = pool.sort_values(
            ["sample_quality_self_confidence", "est_issue_entry_count", "study_id"],
            ascending=[True, False, True],
            kind="stable",
        ).reset_index(drop=True)
        pool["sample_issue_rank"] = np.arange(1, len(pool) + 1)
        target = max(1, min(len(pool), int(round(len(pool) * args.top_fraction))))
        pool["selected_top_fraction"] = pool["sample_issue_rank"].le(target)
        pool["estimator"] = estimator

        selected_studies = set(pool.loc[pool["selected_top_fraction"], "study_id"].astype(int))
        expanded = frame[
            frame["study_id"].isin(selected_studies) & frame["cl_hard_issue"].eq(1)
        ].copy()
        expanded = expanded.merge(
            pool[
                [
                    "study_id",
                    "sample_quality_self_confidence",
                    "est_issue_entry_count",
                    "sample_issue_rank",
                ]
            ],
            on="study_id",
            how="left",
            validate="many_to_one",
        )
        if expanded.empty or expanded["entry_key"].duplicated().any():
            raise ValueError(f"Invalid expanded exact-policy selection for {estimator}")

        sample_rows.append(pool)
        expanded_rows.append(expanded)
        selection_counts[estimator] = {
            "suspicious_studies": int(len(pool)),
            "selected_studies": int(target),
            "expanded_hard_entries": int(len(expanded)),
        }

    sample_policy = pd.concat(sample_rows, ignore_index=True)
    expanded_policy = pd.concat(expanded_rows, ignore_index=True)
    if PROHIBITED_BLIND_COLUMNS.intersection(sample_policy.columns) or PROHIBITED_BLIND_COLUMNS.intersection(
        expanded_policy.columns
    ):
        raise ValueError("Blind selection unexpectedly contains reference columns")

    sample_path = args.output_dir / "sample_top20_policy_blind.csv"
    expanded_path = args.output_dir / "expanded_top20_entries_blind.csv"
    sample_policy.to_csv(sample_path, index=False)
    expanded_policy.to_csv(expanded_path, index=False)
    manifest = {
        "protocol": "dual_reference_cl_detection_v1",
        "phase": "outcome_blind_exact_policy_selection",
        "expert_reference_used": False,
        "score_csv_sha256": sha256_file(args.score_csv),
        "score_manifest_sha256": sha256_file(args.score_manifest),
        "top_fraction": args.top_fraction,
        "selection_counts": selection_counts,
        "sample_policy_sha256": sha256_file(sample_path),
        "expanded_policy_sha256": sha256_file(expanded_path),
    }
    write_json(args.output_dir / "selection_manifest.json", manifest)
    (args.output_dir / ".selection_complete").write_text("complete\n", encoding="utf-8")
    print(json.dumps(selection_counts, indent=2))


def verify_blind_selection(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not (args.output_dir / ".selection_complete").exists():
        raise FileNotFoundError("Blind selection completion marker is missing")
    manifest_path = require_file(args.output_dir / "selection_manifest.json", "selection manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("expert_reference_used") is not False:
        raise ValueError("Selection manifest does not certify outcome blindness")
    if manifest.get("score_csv_sha256") != sha256_file(args.score_csv):
        raise ValueError("Selection used a different frozen score file")
    if manifest.get("score_manifest_sha256") != sha256_file(args.score_manifest):
        raise ValueError("Selection used a different score manifest")

    sample_path = require_file(args.output_dir / "sample_top20_policy_blind.csv", "blind sample policy")
    expanded_path = require_file(args.output_dir / "expanded_top20_entries_blind.csv", "blind expanded policy")
    if manifest.get("sample_policy_sha256") != sha256_file(sample_path):
        raise ValueError("Blind sample policy hash mismatch")
    if manifest.get("expanded_policy_sha256") != sha256_file(expanded_path):
        raise ValueError("Blind expanded policy hash mismatch")
    sample_policy = pd.read_csv(sample_path)
    expanded_policy = pd.read_csv(expanded_path)
    if PROHIBITED_BLIND_COLUMNS.intersection(sample_policy.columns) or PROHIBITED_BLIND_COLUMNS.intersection(
        expanded_policy.columns
    ):
        raise ValueError("Blind selection contains prohibited reference columns")
    return sample_policy, expanded_policy, manifest


def build_mimic_reference(scores: pd.DataFrame, reference_csv: Path) -> pd.DataFrame:
    gt = pd.read_csv(require_file(reference_csv, "--mimic-reference-csv"))
    if gt["study_id"].duplicated().any():
        raise ValueError("MIMIC-CXR 2.1 reference contains duplicate studies")
    gt = gt.rename(columns={"Airspace Opacity": "Lung Opacity"})
    label_columns = sorted(set(scores["label_name"]).intersection(gt.columns))
    long_gt = gt.melt(
        id_vars=["study_id"],
        value_vars=label_columns,
        var_name="label_name",
        value_name="reader_label",
    )
    long_gt = long_gt[long_gt["reader_label"].isin([-1.0, 0.0, 1.0])].copy()
    base = scores[scores["estimator"].eq("ensemble")][
        ["entry_key", "subject_id", "study_id", "label_name", "current_binary_label"]
    ]
    reference = base.merge(
        long_gt,
        on=["study_id", "label_name"],
        how="inner",
        validate="many_to_one",
    )
    reference["reference_binary_label"] = reference["reader_label"].map(
        {-1.0: 1, 0.0: 0, 1.0: 1}
    ).astype(int)
    reference["true_issue"] = reference["current_binary_label"].ne(
        reference["reference_binary_label"]
    ).astype(int)
    reference["reference"] = "mimic_cxr_2_1"
    return validate_reference(reference, "mimic_cxr_2_1")


def build_medpalm_reference(blinded_csv: Path, reference_csv: Path) -> pd.DataFrame:
    blinded = pd.read_csv(require_file(blinded_csv, "--medpalm-blinded-csv"))
    private = pd.read_csv(require_file(reference_csv, "--medpalm-reference-csv"))
    reference = blinded[
        ["entry_key", "subject_id", "study_id", "finding", "current_binary_label"]
    ].merge(
        private[
            [
                "entry_key",
                "subject_id",
                "study_id",
                "finding",
                "reader_label",
                "reader_agreement",
                "reference_binary_label",
            ]
        ],
        on=["entry_key", "subject_id", "study_id", "finding"],
        how="inner",
        validate="one_to_one",
    )
    reference = reference.rename(columns={"finding": "label_name"})
    reference["true_issue"] = reference["current_binary_label"].ne(
        reference["reference_binary_label"]
    ).astype(int)
    reference["reference"] = "medpalm"
    return validate_reference(reference, "medpalm")


def validate_reference(reference: pd.DataFrame, name: str) -> pd.DataFrame:
    expected = EXPECTED_REFERENCES[name]
    observed = {
        "entries": int(len(reference)),
        "issues": int(reference["true_issue"].sum()),
        "studies": int(reference["study_id"].nunique()),
        "subjects": int(reference["subject_id"].nunique()),
    }
    if observed != expected:
        raise ValueError(f"Unexpected {name} reference counts: {observed} != {expected}")
    if reference["entry_key"].duplicated().any():
        raise ValueError(f"{name} reference contains duplicate entries")
    return reference.sort_values(["subject_id", "study_id", "label_name"], kind="stable").reset_index(
        drop=True
    )


def core_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["true_issue"].to_numpy(dtype=int)
    score = frame["suspicion_percentile_self"].to_numpy(dtype=float)
    prevalence = float(y.mean())
    hard = frame["cl_hard_issue"].to_numpy(dtype=int) == 1
    selected = frame["selected_by_exact_policy"].to_numpy(dtype=bool)
    issues = int(y.sum())
    hard_tp = int(y[hard].sum())
    policy_tp = int(y[selected].sum())
    hard_precision = safe_div(hard_tp, int(hard.sum()))
    policy_precision = safe_div(policy_tp, int(selected.sum()))
    return {
        "entries": float(len(frame)),
        "issues": float(issues),
        "prevalence": prevalence,
        "auprc": finite_metric(average_precision_score, y, score),
        "auprc_minus_prevalence": finite_metric(average_precision_score, y, score) - prevalence
        if len(np.unique(y)) == 2
        else float("nan"),
        "auroc": finite_metric(roc_auc_score, y, score),
        "hard_flagged_entries": float(hard.sum()),
        "hard_true_issues": float(hard_tp),
        "hard_precision": hard_precision,
        "hard_recall": safe_div(hard_tp, issues),
        "hard_enrichment": safe_div(hard_precision, prevalence),
        "policy_reference_entries": float(selected.sum()),
        "policy_true_issues": float(policy_tp),
        "policy_precision": policy_precision,
        "policy_recall": safe_div(policy_tp, issues),
        "policy_enrichment": safe_div(policy_precision, prevalence),
    }


def bootstrap_subjects(
    frame: pd.DataFrame,
    *,
    reference: str,
    estimator: str,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    groups = [group.index.to_numpy(dtype=int) for _, group in frame.groupby("subject_id", sort=True)]
    rows: list[dict[str, float | int | str]] = []
    for replicate in range(replicates):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in sampled])
        metrics = core_metrics(frame.loc[indices])
        rows.append(
            {
                "reference": reference,
                "estimator": estimator,
                "replicate": replicate,
                **{metric: metrics[metric] for metric in BOOTSTRAP_METRICS},
            }
        )
    return pd.DataFrame(rows)


def summarize_bootstrap(replicates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (reference, estimator), group in replicates.groupby(["reference", "estimator"], sort=True):
        for metric in BOOTSTRAP_METRICS:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "reference": reference,
                    "estimator": estimator,
                    "metric": metric,
                    "bootstrap_valid_replicates": int(len(values)),
                    "ci_lower": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                    "ci_upper": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def random_policy_replicates(
    scores: pd.DataFrame,
    sample_policy: pd.DataFrame,
    references: dict[str, pd.DataFrame],
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for estimator_index, estimator in enumerate(ESTIMATORS):
        estimator_scores = scores[scores["estimator"].eq(estimator)]
        hard_entries = estimator_scores[estimator_scores["cl_hard_issue"].eq(1)][
            ["study_id", "entry_key"]
        ]
        pool = sample_policy[sample_policy["estimator"].eq(estimator)].copy()
        pool_ids = pool["study_id"].to_numpy(dtype=int)
        selected_count = int(pool["selected_top_fraction"].sum())
        for reference_index, (reference_name, reference) in enumerate(references.items()):
            joined = hard_entries.merge(
                reference[["entry_key", "true_issue"]],
                on="entry_key",
                how="inner",
                validate="one_to_one",
            )
            per_study = joined.groupby("study_id").agg(
                reference_entries=("entry_key", "size"),
                true_issues=("true_issue", "sum"),
            )
            covered = per_study["reference_entries"].reindex(pool_ids, fill_value=0).to_numpy(dtype=int)
            issues = per_study["true_issues"].reindex(pool_ids, fill_value=0).to_numpy(dtype=int)
            total_issues = int(reference["true_issue"].sum())
            rng = np.random.default_rng(seed + estimator_index * 101 + reference_index * 10_007)
            for replicate in range(replicates):
                selected_indices = rng.choice(len(pool_ids), size=selected_count, replace=False)
                reference_entries = int(covered[selected_indices].sum())
                true_issues = int(issues[selected_indices].sum())
                rows.append(
                    {
                        "reference": reference_name,
                        "estimator": estimator,
                        "replicate": replicate,
                        "reference_entries": reference_entries,
                        "true_issues": true_issues,
                        "precision": safe_div(true_issues, reference_entries),
                        "recall": safe_div(true_issues, total_issues),
                    }
                )
    return pd.DataFrame(rows)


def summarize_random_policy(
    replicates: pd.DataFrame, metrics: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    observed_lookup = metrics.set_index(["reference", "estimator"])
    for (reference, estimator), group in replicates.groupby(["reference", "estimator"], sort=True):
        observed = int(observed_lookup.loc[(reference, estimator), "policy_true_issues"])
        random_issues = group["true_issues"].to_numpy(dtype=int)
        precision = group["precision"].to_numpy(dtype=float)
        precision = precision[np.isfinite(precision)]
        rows.append(
            {
                "reference": reference,
                "estimator": estimator,
                "observed_policy_true_issues": observed,
                "random_true_issues_mean": float(random_issues.mean()),
                "random_true_issues_ci_lower": float(np.quantile(random_issues, 0.025)),
                "random_true_issues_ci_upper": float(np.quantile(random_issues, 0.975)),
                "random_precision_mean": float(precision.mean()) if len(precision) else np.nan,
                "random_precision_ci_lower": float(np.quantile(precision, 0.025)) if len(precision) else np.nan,
                "random_precision_ci_upper": float(np.quantile(precision, 0.975)) if len(precision) else np.nan,
                "empirical_p_capture_ge_observed": float(
                    (1 + np.count_nonzero(random_issues >= observed)) / (1 + len(random_issues))
                ),
            }
        )
    return pd.DataFrame(rows)


def build_figure(
    private_rows: pd.DataFrame,
    metrics: pd.DataFrame,
    random_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    display_names = {"mimic_cxr_2_1": "MIMIC-CXR 2.1", "medpalm": "Med-PaLM"}
    for row_index, reference in enumerate(["mimic_cxr_2_1", "medpalm"]):
        frame = private_rows[
            private_rows["reference"].eq(reference) & private_rows["estimator"].eq("ensemble")
        ]
        precision, recall, _ = precision_recall_curve(
            frame["true_issue"], frame["suspicion_percentile_self"]
        )
        prevalence = float(frame["true_issue"].mean())
        axes[row_index, 0].plot(recall, precision, color="#d62728", linewidth=2.2, label="CL ensemble")
        axes[row_index, 0].axhline(
            prevalence, color="#666666", linestyle="--", linewidth=1.5, label="Issue prevalence"
        )
        axes[row_index, 0].set_title(f"{display_names[reference]}: continuous CL ranking")
        axes[row_index, 0].set_xlabel("Recall")
        axes[row_index, 0].set_ylabel("Precision")
        axes[row_index, 0].set_xlim(0, 1)
        axes[row_index, 0].set_ylim(0, 1)
        axes[row_index, 0].legend(frameon=False)

        metric = metrics[
            metrics["reference"].eq(reference) & metrics["estimator"].eq("ensemble")
        ].iloc[0]
        random_row = random_summary[
            random_summary["reference"].eq(reference)
            & random_summary["estimator"].eq("ensemble")
        ].iloc[0]
        values = [
            metric["prevalence"],
            metric["hard_precision"],
            metric["policy_precision"],
            random_row["random_precision_mean"],
        ]
        axes[row_index, 1].bar(
            ["Prevalence", "CL hard flags", "Exact top 20%", "Random within\nCL pool"],
            values,
            color=["#999999", "#4c78a8", "#d62728", "#bdbdbd"],
        )
        axes[row_index, 1].set_ylim(0, max(values) * 1.35 if max(values) > 0 else 1)
        axes[row_index, 1].set_ylabel("True-issue fraction")
        axes[row_index, 1].set_title(
            f"Known issues captured: {int(metric['policy_true_issues'])} "
            f"(random mean {random_row['random_true_issues_mean']:.1f})"
        )
        axes[row_index, 1].tick_params(axis="x", labelrotation=10)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def evaluate_stage(args: argparse.Namespace) -> None:
    if (args.output_dir / ".evaluation_complete").exists():
        raise RuntimeError("Refusing to overwrite a completed evaluation")
    sample_policy, expanded_policy, selection_manifest = verify_blind_selection(args)
    scores = load_scores(args.score_csv, args.score_manifest)
    references = {
        "mimic_cxr_2_1": build_mimic_reference(
            scores, require_file(args.mimic_reference_csv, "--mimic-reference-csv")
        ),
        "medpalm": build_medpalm_reference(
            require_file(args.medpalm_blinded_csv, "--medpalm-blinded-csv"),
            require_file(args.medpalm_reference_csv, "--medpalm-reference-csv"),
        ),
    }
    references["mimic_cxr_2_1"].to_csv(args.output_dir / "mimic_reference_private.csv", index=False)
    references["medpalm"].to_csv(args.output_dir / "medpalm_reference_private.csv", index=False)

    private_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    bootstrap_frames: list[pd.DataFrame] = []

    for reference_index, (reference_name, reference) in enumerate(references.items()):
        for estimator_index, estimator in enumerate(ESTIMATORS):
            estimator_scores = scores[scores["estimator"].eq(estimator)].copy()
            frame = estimator_scores.merge(
                reference[
                    [
                        "entry_key",
                        "current_binary_label",
                        "reference_binary_label",
                        "reader_label",
                        "true_issue",
                    ]
                    + (["reader_agreement"] if "reader_agreement" in reference.columns else [])
                ],
                on="entry_key",
                how="inner",
                validate="one_to_one",
                suffixes=("", "_reference"),
            )
            if not frame["current_binary_label"].eq(frame["current_binary_label_reference"]).all():
                raise ValueError(f"Current labels differ for {reference_name}/{estimator}")
            selected_keys = set(
                expanded_policy.loc[
                    expanded_policy["estimator"].eq(estimator), "entry_key"
                ].astype(str)
            )
            frame["selected_by_exact_policy"] = frame["entry_key"].astype(str).isin(selected_keys)
            frame["reference"] = reference_name
            private_frames.append(frame)

            core = core_metrics(frame)
            selection_counts = selection_manifest["selection_counts"][estimator]
            metric_rows.append(
                {
                    "reference": reference_name,
                    "estimator": estimator,
                    **core,
                    **selection_counts,
                }
            )
            for label_name, label_frame in frame.groupby("label_name", sort=True):
                label_rows.append(
                    {
                        "reference": reference_name,
                        "estimator": estimator,
                        "label_name": label_name,
                        **core_metrics(label_frame),
                    }
                )
            bootstrap_frames.append(
                bootstrap_subjects(
                    frame.reset_index(drop=True),
                    reference=reference_name,
                    estimator=estimator,
                    replicates=args.bootstrap_replicates,
                    rng=np.random.default_rng(
                        args.random_seed + reference_index * 10_007 + estimator_index * 101
                    ),
                )
            )

    private_rows = pd.concat(private_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    label_metrics = pd.DataFrame(label_rows)
    bootstrap_replicates = pd.concat(bootstrap_frames, ignore_index=True)
    bootstrap_ci = summarize_bootstrap(bootstrap_replicates)
    random_replicates = random_policy_replicates(
        scores,
        sample_policy,
        references,
        replicates=args.random_replicates,
        seed=args.random_seed,
    )
    random_summary = summarize_random_policy(random_replicates, metrics)
    metrics = metrics.merge(
        random_summary,
        on=["reference", "estimator"],
        how="left",
        validate="one_to_one",
    )

    output_files = {
        "private_rows": args.output_dir / "dual_reference_rows_private.csv",
        "metrics": args.output_dir / "dual_reference_metrics.csv",
        "label_metrics": args.output_dir / "dual_reference_label_metrics.csv",
        "bootstrap_replicates": args.output_dir / "dual_reference_subject_bootstrap_replicates.csv",
        "bootstrap_ci": args.output_dir / "dual_reference_subject_bootstrap_ci.csv",
        "random_replicates": args.output_dir / "dual_reference_random_policy_replicates.csv",
        "random_summary": args.output_dir / "dual_reference_random_policy_summary.csv",
        "figure": args.output_dir / "dual_reference_cl_detection.png",
    }
    private_rows.to_csv(output_files["private_rows"], index=False)
    metrics.to_csv(output_files["metrics"], index=False)
    label_metrics.to_csv(output_files["label_metrics"], index=False)
    bootstrap_replicates.to_csv(output_files["bootstrap_replicates"], index=False)
    bootstrap_ci.to_csv(output_files["bootstrap_ci"], index=False)
    random_replicates.to_csv(output_files["random_replicates"], index=False)
    random_summary.to_csv(output_files["random_summary"], index=False)
    build_figure(private_rows, metrics, random_summary, output_files["figure"])

    ensemble_rows = metrics[metrics["estimator"].eq("ensemble")].set_index("reference")
    summary = {
        "protocol": "dual_reference_cl_detection_v1",
        "primary_estimator": "ensemble",
        "top_fraction": args.top_fraction,
        "bootstrap_replicates": args.bootstrap_replicates,
        "random_policy_replicates": args.random_replicates,
        "mimic_cxr_2_1": ensemble_rows.loc["mimic_cxr_2_1"].to_dict(),
        "medpalm": ensemble_rows.loc["medpalm"].to_dict(),
        "individual_seed_direction": {
            reference_name: {
                "auprc_above_prevalence": int(
                    (
                        metrics[
                            metrics["reference"].eq(reference_name)
                            & metrics["estimator"].ne("ensemble")
                        ]["auprc_minus_prevalence"]
                        > 0
                    ).sum()
                ),
                "hard_enrichment_above_one": int(
                    (
                        metrics[
                            metrics["reference"].eq(reference_name)
                            & metrics["estimator"].ne("ensemble")
                        ]["hard_enrichment"]
                        > 1
                    ).sum()
                ),
                "policy_capture_above_random_mean": int(
                    (
                        metrics[
                            metrics["reference"].eq(reference_name)
                            & metrics["estimator"].ne("ensemble")
                        ]["policy_true_issues"]
                        > metrics[
                            metrics["reference"].eq(reference_name)
                            & metrics["estimator"].ne("ensemble")
                        ]["random_true_issues_mean"]
                    ).sum()
                ),
                "seeds_evaluated": 4,
            }
            for reference_name in references
        },
    }
    write_json(args.output_dir / "dual_reference_summary.json", summary)

    evaluation_manifest = {
        "protocol": "dual_reference_cl_detection_v1",
        "selection_manifest_sha256": sha256_file(args.output_dir / "selection_manifest.json"),
        "mimic_reference_input_sha256": sha256_file(args.mimic_reference_csv),
        "medpalm_blinded_input_sha256": sha256_file(args.medpalm_blinded_csv),
        "medpalm_reference_input_sha256": sha256_file(args.medpalm_reference_csv),
        "output_sha256": {
            name: sha256_file(path) for name, path in output_files.items()
        },
        "summary_sha256": sha256_file(args.output_dir / "dual_reference_summary.json"),
    }
    write_json(args.output_dir / "evaluation_manifest.json", evaluation_manifest)
    (args.output_dir / ".evaluation_complete").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=lambda value: float(value)))


def verify_stage(args: argparse.Namespace) -> None:
    sample_policy, expanded_policy, _ = verify_blind_selection(args)
    if not (args.output_dir / ".evaluation_complete").exists():
        raise FileNotFoundError("Evaluation completion marker is missing")
    manifest = json.loads(require_file(args.output_dir / "evaluation_manifest.json", "evaluation manifest").read_text())
    output_map = {
        "private_rows": args.output_dir / "dual_reference_rows_private.csv",
        "metrics": args.output_dir / "dual_reference_metrics.csv",
        "label_metrics": args.output_dir / "dual_reference_label_metrics.csv",
        "bootstrap_replicates": args.output_dir / "dual_reference_subject_bootstrap_replicates.csv",
        "bootstrap_ci": args.output_dir / "dual_reference_subject_bootstrap_ci.csv",
        "random_replicates": args.output_dir / "dual_reference_random_policy_replicates.csv",
        "random_summary": args.output_dir / "dual_reference_random_policy_summary.csv",
        "figure": args.output_dir / "dual_reference_cl_detection.png",
    }
    for name, path in output_map.items():
        require_file(path, name)
        if manifest["output_sha256"].get(name) != sha256_file(path):
            raise ValueError(f"Output hash mismatch: {name}")
    if manifest.get("summary_sha256") != sha256_file(
        require_file(args.output_dir / "dual_reference_summary.json", "summary")
    ):
        raise ValueError("Summary hash mismatch")

    metrics = pd.read_csv(output_map["metrics"])
    private_rows = pd.read_csv(output_map["private_rows"])
    bootstrap = pd.read_csv(output_map["bootstrap_replicates"])
    random_replicates = pd.read_csv(output_map["random_replicates"])
    if len(metrics) != 10 or metrics.duplicated(["reference", "estimator"]).any():
        raise ValueError("Metrics table does not contain ten unique estimator-reference rows")
    expected_private = sum(item["entries"] for item in EXPECTED_REFERENCES.values()) * len(ESTIMATORS)
    if len(private_rows) != expected_private:
        raise ValueError(f"Unexpected private row count: {len(private_rows)}")
    if len(bootstrap) != args.bootstrap_replicates * 10:
        raise ValueError("Unexpected subject-bootstrap replicate count")
    if len(random_replicates) != args.random_replicates * 10:
        raise ValueError("Unexpected random-policy replicate count")
    if sample_policy.duplicated(["estimator", "study_id"]).any():
        raise ValueError("Blind sample policy contains duplicates")
    if expanded_policy.duplicated(["estimator", "entry_key"]).any():
        raise ValueError("Blind expanded policy contains duplicates")
    print("Dual-reference CL detection benchmark verification passed.")
    print(metrics[metrics["estimator"].eq("ensemble")].to_string(index=False))


def main() -> None:
    args = parse_args()
    if args.mode == "select":
        select_stage(args)
    elif args.mode == "evaluate":
        evaluate_stage(args)
    else:
        verify_stage(args)


if __name__ == "__main__":
    main()
