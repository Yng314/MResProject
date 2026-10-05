#!/usr/bin/env python3
"""Compare CL hard filtering with simple uncertainty rankings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dual_reference_cl_detection_benchmark import (
    ESTIMATORS,
    EXPECTED_REFERENCES,
    PROHIBITED_BLIND_COLUMNS,
    build_medpalm_reference,
    build_mimic_reference,
    load_scores,
    require_file,
    safe_div,
    sha256_file,
    write_json,
)


PROTOCOL = "cl_vs_uncertainty_ablation_v1"
CL_POLICY = "cl_hard_direct"
PRIMARY_BASELINE = "self_confidence_per_label_percentile"
COMMON_POLICIES = [
    CL_POLICY,
    "self_confidence_global",
    PRIMARY_BASELINE,
    "predictive_entropy",
]
ENSEMBLE_ONLY_POLICY = "seed_disagreement"
ALL_POLICIES = COMMON_POLICIES + [ENSEMBLE_ONLY_POLICY]
POLICY_LABELS = {
    CL_POLICY: "CL hard + direct",
    "self_confidence_global": "Global self-confidence",
    PRIMARY_BASELINE: "Per-label percentile",
    "predictive_entropy": "Predictive entropy",
    ENSEMBLE_ONLY_POLICY: "Seed disagreement",
}
ENTRY_BUDGETS = {"seed_13": 326, "seed_42": 290, "seed_97": 289, "seed_123": 332, "ensemble": 186}
PAIR_METRICS = ["true_issues", "precision", "recall", "reference_coverage", "enrichment"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["select", "evaluate", "verify"])
    parser.add_argument("--score-csv", type=Path, required=True)
    parser.add_argument("--score-manifest", type=Path, required=True)
    parser.add_argument("--mimic-reference-csv", type=Path)
    parser.add_argument("--medpalm-blinded-csv", type=Path)
    parser.add_argument("--medpalm-reference-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--random-replicates", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20_260_803)
    return parser.parse_args()


def available_policies(estimator: str) -> list[str]:
    return ALL_POLICIES if estimator == "ensemble" else COMMON_POLICIES


def binary_entropy(probability: pd.Series) -> np.ndarray:
    values = np.clip(probability.to_numpy(float), 1e-12, 1 - 1e-12)
    return -(values * np.log(values) + (1 - values) * np.log(1 - values))


def add_seed_disagreement(scores: pd.DataFrame) -> pd.DataFrame:
    seed_rows = scores[scores["estimator"].ne("ensemble")]
    pivot = seed_rows.pivot(index="entry_key", columns="estimator", values="pred_probability")
    expected = ["seed_13", "seed_42", "seed_97", "seed_123"]
    if sorted(pivot.columns) != sorted(expected) or pivot.isna().any().any():
        raise ValueError("Cannot construct complete four-seed disagreement")
    disagreement = pivot[expected].std(axis=1, ddof=0).rename("seed_probability_std")
    return scores.merge(disagreement, on="entry_key", how="left", validate="many_to_one")


def rank_policy(frame: pd.DataFrame, policy: str, budget: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking = frame.copy()
    if policy == CL_POLICY:
        ranking = ranking[ranking["cl_hard_issue"].eq(1)].copy()
        ranking = ranking.sort_values(
            ["quality_self_confidence", "suspicion_percentile_self", "label_name", "entry_key"],
            ascending=[True, False, True, True], kind="stable"
        )
        ranking["policy_score"] = -ranking["quality_self_confidence"]
        selection_pool = "cl_hard_entries"
    elif policy == "self_confidence_global":
        ranking = ranking.sort_values(
            ["quality_self_confidence", "label_name", "entry_key"],
            ascending=[True, True, True], kind="stable"
        )
        ranking["policy_score"] = -ranking["quality_self_confidence"]
        selection_pool = "all_valid_entries"
    elif policy == PRIMARY_BASELINE:
        ranking = ranking.sort_values(
            ["suspicion_percentile_self", "quality_self_confidence", "label_name", "entry_key"],
            ascending=[False, True, True, True], kind="stable"
        )
        ranking["policy_score"] = ranking["suspicion_percentile_self"]
        selection_pool = "all_valid_entries"
    elif policy == "predictive_entropy":
        ranking["predictive_entropy"] = binary_entropy(ranking["pred_probability"])
        ranking = ranking.sort_values(
            ["predictive_entropy", "quality_self_confidence", "label_name", "entry_key"],
            ascending=[False, True, True, True], kind="stable"
        )
        ranking["policy_score"] = ranking["predictive_entropy"]
        selection_pool = "all_valid_entries"
    elif policy == ENSEMBLE_ONLY_POLICY:
        if ranking["seed_probability_std"].isna().any():
            raise ValueError("Seed disagreement is unavailable")
        ranking = ranking.sort_values(
            ["seed_probability_std", "quality_self_confidence", "label_name", "entry_key"],
            ascending=[False, True, True, True], kind="stable"
        )
        ranking["policy_score"] = ranking["seed_probability_std"]
        selection_pool = "all_valid_entries"
    else:
        raise ValueError(f"Unknown policy: {policy}")
    if len(ranking) < budget:
        raise ValueError(f"Policy pool is smaller than budget: {policy}")
    ranking = ranking.reset_index(drop=True)
    ranking["policy"] = policy
    ranking["selection_pool"] = selection_pool
    ranking["unit_rank"] = np.arange(1, len(ranking) + 1)
    ranking["selected"] = ranking["unit_rank"].le(budget)
    selected = ranking[ranking["selected"]].copy()
    return ranking, selected


def select_stage(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / ".evaluation_complete").exists():
        raise RuntimeError("Refusing to modify a completed evaluation")
    scores = add_seed_disagreement(load_scores(args.score_csv, args.score_manifest))

    margin_diff = np.abs(
        scores["quality_self_confidence"].to_numpy(float)
        - scores["quality_normalized_margin"].to_numpy(float)
    )
    max_margin_diff = float(margin_diff.max())
    if max_margin_diff > 1e-12:
        raise ValueError("Normalized margin is not duplicate self-confidence")

    rankings: list[pd.DataFrame] = []
    selections: list[pd.DataFrame] = []
    counts: dict[str, dict[str, dict[str, int | str]]] = {}
    mismatch_audit: dict[str, dict[str, int | bool]] = {}
    for estimator in ESTIMATORS:
        frame = scores[scores["estimator"].eq(estimator)].copy()
        budget = ENTRY_BUDGETS[estimator]
        low_confidence_count = int(frame["quality_self_confidence"].lt(0.5).sum())
        mismatch_audit[estimator] = {
            "quality_below_0_5_entries": low_confidence_count,
            "entry_budget": budget,
            "fixed_mismatch_prefix_duplicates_global_self_confidence": bool(low_confidence_count >= budget),
        }
        if low_confidence_count < budget:
            raise ValueError("Fixed-mismatch duplicate audit failed")
        counts[estimator] = {}
        for policy in available_policies(estimator):
            ranking, selected = rank_policy(frame, policy, budget)
            rankings.append(ranking)
            selections.append(selected)
            counts[estimator][policy] = {
                "selection_pool": str(ranking["selection_pool"].iloc[0]),
                "pool_entries": int(len(ranking)),
                "selected_entries": int(len(selected)),
                "selected_studies": int(selected["study_id"].nunique()),
            }

    ranking_table = pd.concat(rankings, ignore_index=True)
    selected_table = pd.concat(selections, ignore_index=True)
    if selected_table.duplicated(["estimator", "policy", "entry_key"]).any():
        raise ValueError("Duplicate selected policy-entry keys")
    if not selected_table.groupby(["estimator", "policy"]).size().eq(
        selected_table.groupby(["estimator", "policy"])["estimator"].first().map(ENTRY_BUDGETS)
    ).all():
        raise ValueError("A policy does not match its entry budget")
    for table in [ranking_table, selected_table]:
        if PROHIBITED_BLIND_COLUMNS.intersection(table.columns):
            raise ValueError("Blind artifact contains prohibited reference columns")

    files = {
        "rankings": args.output_dir / "uncertainty_policy_rankings_blind.csv",
        "selections": args.output_dir / "uncertainty_policy_selections_blind.csv",
    }
    ranking_table.to_csv(files["rankings"], index=False)
    selected_table.to_csv(files["selections"], index=False)
    manifest = {
        "protocol": PROTOCOL,
        "phase": "outcome_blind_uncertainty_policy_selection",
        "expert_reference_used": False,
        "score_csv_sha256": sha256_file(args.score_csv),
        "score_manifest_sha256": sha256_file(args.score_manifest),
        "entry_budgets": ENTRY_BUDGETS,
        "policy_counts": counts,
        "duplicate_baseline_audit": {
            "normalized_margin_max_abs_difference": max_margin_diff,
            "fixed_mismatch": mismatch_audit,
        },
        "output_sha256": {name: sha256_file(path) for name, path in files.items()},
    }
    write_json(args.output_dir / "selection_manifest.json", manifest)
    (args.output_dir / ".selection_complete").write_text("complete\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))


def verify_selection(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    require_file(args.output_dir / ".selection_complete", "selection marker")
    manifest = json.loads(require_file(args.output_dir / "selection_manifest.json", "selection manifest").read_text())
    if manifest.get("protocol") != PROTOCOL or manifest.get("expert_reference_used") is not False:
        raise ValueError("Invalid blind selection manifest")
    if manifest.get("score_csv_sha256") != sha256_file(args.score_csv):
        raise ValueError("Score hash mismatch")
    if manifest.get("score_manifest_sha256") != sha256_file(args.score_manifest):
        raise ValueError("Score-manifest hash mismatch")
    paths = {
        "rankings": args.output_dir / "uncertainty_policy_rankings_blind.csv",
        "selections": args.output_dir / "uncertainty_policy_selections_blind.csv",
    }
    tables = {}
    for name, path in paths.items():
        require_file(path, name)
        if manifest["output_sha256"].get(name) != sha256_file(path):
            raise ValueError(f"Blind output hash mismatch: {name}")
        tables[name] = pd.read_csv(path)
        if PROHIBITED_BLIND_COLUMNS.intersection(tables[name].columns):
            raise ValueError(f"Blind outcome leakage: {name}")
    return tables["rankings"], tables["selections"], manifest


def policy_metrics(frame: pd.DataFrame) -> dict[str, float]:
    selected = frame[frame["selected_by_policy"]]
    issues = int(selected["true_issue"].sum())
    covered = int(len(selected))
    total_issues = int(frame["true_issue"].sum())
    prevalence = float(frame["true_issue"].mean())
    precision = safe_div(issues, covered)
    return {
        "reference_entries": int(len(frame)),
        "reference_issues": total_issues,
        "prevalence": prevalence,
        "reference_covered_entries": covered,
        "true_issues": issues,
        "precision": precision,
        "recall": safe_div(issues, total_issues),
        "reference_coverage": safe_div(covered, len(frame)),
        "enrichment": safe_div(precision, prevalence),
    }


def bootstrap_policies(
    frames: dict[str, pd.DataFrame], reference: str, estimator: str, replicates: int, seed: int
) -> pd.DataFrame:
    base = frames[CL_POLICY].reset_index(drop=True)
    groups = [group.index.to_numpy(int) for _, group in base.groupby("subject_id", sort=True)]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for replicate in range(replicates):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in sampled])
        for policy, frame in frames.items():
            values = policy_metrics(frame.iloc[indices])
            rows.append({
                "reference": reference, "estimator": estimator, "policy": policy,
                "replicate": replicate, **{metric: values[metric] for metric in PAIR_METRICS},
            })
    return pd.DataFrame(rows)


def holm_adjust(values: pd.Series) -> pd.Series:
    order = np.argsort(values.to_numpy(float))
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    raw = values.to_numpy(float)
    for rank, position in enumerate(order):
        running = max(running, (total - rank) * raw[position])
        adjusted[position] = min(1.0, running)
    return pd.Series(adjusted, index=values.index)


def summarize_bootstrap(replicates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ci_rows: list[dict[str, object]] = []
    for keys, group in replicates.groupby(["reference", "estimator", "policy"], sort=True):
        for metric in PAIR_METRICS:
            values = group[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            ci_rows.append({
                "reference": keys[0], "estimator": keys[1], "policy": keys[2], "metric": metric,
                "valid_replicates": len(values),
                "ci_lower": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                "ci_upper": float(np.quantile(values, 0.975)) if len(values) else np.nan,
            })

    difference_rows: list[dict[str, object]] = []
    for (reference, estimator), group in replicates.groupby(["reference", "estimator"], sort=True):
        current = group[group["policy"].eq(CL_POLICY)].set_index("replicate")
        for baseline in [p for p in available_policies(estimator) if p != CL_POLICY]:
            other = group[group["policy"].eq(baseline)].set_index("replicate")
            if not current.index.equals(other.index):
                raise ValueError("Paired bootstrap replicate mismatch")
            for metric in PAIR_METRICS:
                delta = current[metric].to_numpy(float) - other[metric].to_numpy(float)
                delta = delta[np.isfinite(delta)]
                p_one_sided = float((1 + np.count_nonzero(delta <= 0)) / (1 + len(delta)))
                difference_rows.append({
                    "reference": reference, "estimator": estimator, "baseline": baseline, "metric": metric,
                    "valid_replicates": len(delta),
                    "mean_cl_minus_baseline": float(delta.mean()) if len(delta) else np.nan,
                    "ci_lower": float(np.quantile(delta, 0.025)) if len(delta) else np.nan,
                    "ci_upper": float(np.quantile(delta, 0.975)) if len(delta) else np.nan,
                    "one_sided_p_cl_better": p_one_sided,
                })
    differences = pd.DataFrame(difference_rows)
    differences["holm_adjusted_p"] = differences.groupby(
        ["reference", "estimator", "metric"], group_keys=False
    )["one_sided_p_cl_better"].apply(holm_adjust)
    return pd.DataFrame(ci_rows), differences


def random_full_pool(
    scores: pd.DataFrame, references: dict[str, pd.DataFrame], replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for estimator_index, estimator in enumerate(ESTIMATORS):
        frame = scores[scores["estimator"].eq(estimator)]
        keys = frame["entry_key"].astype(str).to_numpy()
        budget = ENTRY_BUDGETS[estimator]
        for reference_index, (reference_name, reference) in enumerate(references.items()):
            issue_lookup = reference.set_index("entry_key")["true_issue"]
            known = set(reference["entry_key"].astype(str))
            covered = np.fromiter((key in known for key in keys), dtype=int)
            issues = np.fromiter((int(issue_lookup.get(key, 0)) for key in keys), dtype=int)
            total_issues = int(reference["true_issue"].sum())
            rng = np.random.default_rng(seed + estimator_index * 101 + reference_index * 10_007)
            for replicate in range(replicates):
                chosen = rng.choice(len(keys), size=budget, replace=False)
                n_covered = int(covered[chosen].sum())
                n_issues = int(issues[chosen].sum())
                rows.append({
                    "reference": reference_name, "estimator": estimator, "replicate": replicate,
                    "reference_covered_entries": n_covered, "true_issues": n_issues,
                    "precision": safe_div(n_issues, n_covered), "recall": safe_div(n_issues, total_issues),
                })
    replicates_frame = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for keys, group in replicates_frame.groupby(["reference", "estimator"], sort=True):
        issue_values = group["true_issues"].to_numpy(int)
        precision = group["precision"].to_numpy(float)
        finite = precision[np.isfinite(precision)]
        summaries.append({
            "reference": keys[0], "estimator": keys[1],
            "random_true_issues_mean": float(issue_values.mean()),
            "random_true_issues_ci_lower": float(np.quantile(issue_values, 0.025)),
            "random_true_issues_ci_upper": float(np.quantile(issue_values, 0.975)),
            "random_reference_coverage_mean": float(group["reference_covered_entries"].mean()),
            "random_precision_mean": float(finite.mean()) if len(finite) else np.nan,
        })
    return replicates_frame, pd.DataFrame(summaries)


def add_random_tail(metrics: pd.DataFrame, random_replicates: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for keys, group in random_replicates.groupby(["reference", "estimator"], sort=True):
        issues = group["true_issues"].to_numpy(int)
        for _, row in metrics[
            metrics["reference"].eq(keys[0]) & metrics["estimator"].eq(keys[1])
        ].iterrows():
            summaries.append({
                "reference": keys[0], "estimator": keys[1], "policy": row["policy"],
                "empirical_p_random_capture_ge_observed": float(
                    (1 + np.count_nonzero(issues >= row["true_issues"])) / (1 + len(issues))
                ),
            })
    return metrics.merge(pd.DataFrame(summaries), on=["reference", "estimator", "policy"], validate="one_to_one")


def build_figure(metrics: pd.DataFrame, random_summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    reference_names = {"mimic_cxr_2_1": "MIMIC-CXR 2.1", "medpalm": "Med-PaLM"}
    colors = ["#d62728", "#4c78a8", "#59a14f", "#f28e2b", "#9467bd"]
    for row_index, reference in enumerate(["mimic_cxr_2_1", "medpalm"]):
        frame = metrics[metrics["reference"].eq(reference) & metrics["estimator"].eq("ensemble")]
        frame = frame.set_index("policy").loc[ALL_POLICIES].reset_index()
        random = random_summary[
            random_summary["reference"].eq(reference) & random_summary["estimator"].eq("ensemble")
        ].iloc[0]
        labels = [POLICY_LABELS[p] for p in ALL_POLICIES]
        x = np.arange(len(frame))
        axes[row_index, 0].bar(x, frame["precision"], color=colors)
        axes[row_index, 0].axhline(random["random_precision_mean"], color="black", linestyle="--", label="Full-pool random mean")
        axes[row_index, 0].set_xticks(x, labels, rotation=15)
        axes[row_index, 0].set_ylabel("Known-issue precision")
        axes[row_index, 0].set_title(f"{reference_names[reference]}: equal-budget precision")
        axes[row_index, 0].legend(frameon=False)

        axes[row_index, 1].bar(x, frame["true_issues"], color=colors)
        axes[row_index, 1].axhline(random["random_true_issues_mean"], color="black", linestyle="--", label="Full-pool random mean")
        for index, item in frame.iterrows():
            axes[row_index, 1].text(index, item["true_issues"] + 0.25, f"GT n={int(item['reference_covered_entries'])}", ha="center", fontsize=8)
        axes[row_index, 1].set_xticks(x, labels, rotation=15)
        axes[row_index, 1].set_ylabel("Known issues captured")
        axes[row_index, 1].set_title(f"{reference_names[reference]}: equal-entry-budget capture")
        axes[row_index, 1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def evaluate_stage(args: argparse.Namespace) -> None:
    if (args.output_dir / ".evaluation_complete").exists():
        raise RuntimeError("Refusing to overwrite completed evaluation")
    _, selections, selection_manifest = verify_selection(args)
    scores = add_seed_disagreement(load_scores(args.score_csv, args.score_manifest))
    references = {
        "mimic_cxr_2_1": build_mimic_reference(scores, require_file(args.mimic_reference_csv, "--mimic-reference-csv")),
        "medpalm": build_medpalm_reference(
            require_file(args.medpalm_blinded_csv, "--medpalm-blinded-csv"),
            require_file(args.medpalm_reference_csv, "--medpalm-reference-csv"),
        ),
    }

    private_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    bootstrap_frames: list[pd.DataFrame] = []
    for reference_index, (reference_name, reference) in enumerate(references.items()):
        for estimator_index, estimator in enumerate(ESTIMATORS):
            base = scores[scores["estimator"].eq(estimator)].merge(
                reference[["entry_key", "subject_id", "study_id", "label_name", "true_issue"]],
                on=["entry_key", "subject_id", "study_id", "label_name"], how="inner", validate="one_to_one"
            )
            frames: dict[str, pd.DataFrame] = {}
            for policy in available_policies(estimator):
                selected_keys = set(selections[
                    selections["estimator"].eq(estimator) & selections["policy"].eq(policy)
                ]["entry_key"].astype(str))
                frame = base.copy()
                frame["selected_by_policy"] = frame["entry_key"].astype(str).isin(selected_keys)
                frame["reference"] = reference_name
                frame["policy"] = policy
                frames[policy] = frame
                private_frames.append(frame)
                values = policy_metrics(frame)
                counts = selection_manifest["policy_counts"][estimator][policy]
                metric_rows.append({"reference": reference_name, "estimator": estimator, "policy": policy, **counts, **values})
                for label_name, label_frame in frame.groupby("label_name", sort=True):
                    label_rows.append({"reference": reference_name, "estimator": estimator, "policy": policy,
                                       "label_name": label_name, **policy_metrics(label_frame)})
            bootstrap_frames.append(bootstrap_policies(
                frames, reference_name, estimator, args.bootstrap_replicates,
                args.random_seed + reference_index * 10_007 + estimator_index * 101
            ))

    private_rows = pd.concat(private_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    label_metrics = pd.DataFrame(label_rows)
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    bootstrap_ci, paired = summarize_bootstrap(bootstrap)
    random_replicates, random_summary = random_full_pool(
        scores, references, args.random_replicates, args.random_seed
    )
    metrics = metrics.merge(random_summary, on=["reference", "estimator"], validate="many_to_one")
    metrics = add_random_tail(metrics, random_replicates)

    primary_metrics = metrics[
        metrics["reference"].eq("mimic_cxr_2_1") & metrics["estimator"].eq("ensemble")
    ].set_index("policy")
    primary_pair = paired[
        paired["reference"].eq("mimic_cxr_2_1")
        & paired["estimator"].eq("ensemble")
        & paired["baseline"].eq(PRIMARY_BASELINE)
    ].set_index("metric")
    seed_wins = 0
    for estimator in ["seed_13", "seed_42", "seed_97", "seed_123"]:
        rows = metrics[
            metrics["reference"].eq("mimic_cxr_2_1") & metrics["estimator"].eq(estimator)
        ].set_index("policy")
        seed_wins += int(rows.loc[CL_POLICY, "true_issues"] > rows.loc[PRIMARY_BASELINE, "true_issues"])
    cl_row = primary_metrics.loc[CL_POLICY]
    baseline_row = primary_metrics.loc[PRIMARY_BASELINE]
    gates = {
        "cl_captures_more_issues": bool(cl_row["true_issues"] > baseline_row["true_issues"]),
        "cl_precision_not_lower": bool(cl_row["precision"] >= baseline_row["precision"]),
        "capture_difference_ci_lower_above_zero": bool(primary_pair.loc["true_issues", "ci_lower"] > 0),
        "precision_difference_ci_lower_nonnegative": bool(primary_pair.loc["precision", "ci_lower"] >= 0),
        "paired_capture_p_below_0_05": bool(primary_pair.loc["true_issues", "one_sided_p_cl_better"] < 0.05),
        "seed_capture_wins_at_least_3_of_4": bool(seed_wins >= 3),
        "cl_beats_full_pool_random_p_below_0_05": bool(cl_row["empirical_p_random_capture_ge_observed"] < 0.05),
    }
    summary = {
        "protocol": PROTOCOL,
        "primary_reference": "mimic_cxr_2_1",
        "primary_estimator": "ensemble",
        "primary_comparator": PRIMARY_BASELINE,
        "incremental_cl_value": "supported" if all(gates.values()) else "not_supported",
        "seed_capture_wins": seed_wins,
        "gates": gates,
        "primary_metrics": primary_metrics.reset_index().to_dict(orient="records"),
        "primary_paired_differences": primary_pair.reset_index().to_dict(orient="records"),
    }

    outputs = {
        "private_rows": args.output_dir / "cl_uncertainty_reference_rows_private.csv",
        "metrics": args.output_dir / "cl_uncertainty_metrics.csv",
        "label_metrics": args.output_dir / "cl_uncertainty_label_metrics.csv",
        "bootstrap": args.output_dir / "cl_uncertainty_subject_bootstrap_replicates.csv",
        "bootstrap_ci": args.output_dir / "cl_uncertainty_subject_bootstrap_ci.csv",
        "paired": args.output_dir / "cl_uncertainty_paired_differences.csv",
        "random_replicates": args.output_dir / "cl_uncertainty_full_pool_random_replicates.csv",
        "random_summary": args.output_dir / "cl_uncertainty_full_pool_random_summary.csv",
        "figure": args.output_dir / "cl_vs_uncertainty_equal_budget.png",
    }
    private_rows.to_csv(outputs["private_rows"], index=False)
    metrics.to_csv(outputs["metrics"], index=False)
    label_metrics.to_csv(outputs["label_metrics"], index=False)
    bootstrap.to_csv(outputs["bootstrap"], index=False)
    bootstrap_ci.to_csv(outputs["bootstrap_ci"], index=False)
    paired.to_csv(outputs["paired"], index=False)
    random_replicates.to_csv(outputs["random_replicates"], index=False)
    random_summary.to_csv(outputs["random_summary"], index=False)
    build_figure(metrics, random_summary, outputs["figure"])
    write_json(args.output_dir / "cl_vs_uncertainty_summary.json", summary)
    evaluation_manifest = {
        "protocol": PROTOCOL,
        "selection_manifest_sha256": sha256_file(args.output_dir / "selection_manifest.json"),
        "reference_input_sha256": {
            "mimic": sha256_file(args.mimic_reference_csv),
            "medpalm_blinded": sha256_file(args.medpalm_blinded_csv),
            "medpalm_reference": sha256_file(args.medpalm_reference_csv),
        },
        "output_sha256": {name: sha256_file(path) for name, path in outputs.items()},
        "summary_sha256": sha256_file(args.output_dir / "cl_vs_uncertainty_summary.json"),
    }
    write_json(args.output_dir / "evaluation_manifest.json", evaluation_manifest)
    (args.output_dir / ".evaluation_complete").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def verify_stage(args: argparse.Namespace) -> None:
    rankings, selections, _ = verify_selection(args)
    require_file(args.output_dir / ".evaluation_complete", "evaluation marker")
    manifest = json.loads(require_file(args.output_dir / "evaluation_manifest.json", "evaluation manifest").read_text())
    outputs = {
        "private_rows": args.output_dir / "cl_uncertainty_reference_rows_private.csv",
        "metrics": args.output_dir / "cl_uncertainty_metrics.csv",
        "label_metrics": args.output_dir / "cl_uncertainty_label_metrics.csv",
        "bootstrap": args.output_dir / "cl_uncertainty_subject_bootstrap_replicates.csv",
        "bootstrap_ci": args.output_dir / "cl_uncertainty_subject_bootstrap_ci.csv",
        "paired": args.output_dir / "cl_uncertainty_paired_differences.csv",
        "random_replicates": args.output_dir / "cl_uncertainty_full_pool_random_replicates.csv",
        "random_summary": args.output_dir / "cl_uncertainty_full_pool_random_summary.csv",
        "figure": args.output_dir / "cl_vs_uncertainty_equal_budget.png",
    }
    for name, path in outputs.items():
        require_file(path, name)
        if manifest["output_sha256"].get(name) != sha256_file(path):
            raise ValueError(f"Evaluation output hash mismatch: {name}")
    if manifest.get("summary_sha256") != sha256_file(require_file(
        args.output_dir / "cl_vs_uncertainty_summary.json", "summary"
    )):
        raise ValueError("Summary hash mismatch")

    metrics = pd.read_csv(outputs["metrics"])
    private = pd.read_csv(outputs["private_rows"])
    bootstrap = pd.read_csv(outputs["bootstrap"])
    random = pd.read_csv(outputs["random_replicates"])
    expected_combinations = len(COMMON_POLICIES) * len(ESTIMATORS) + 1
    if len(metrics) != expected_combinations * len(EXPECTED_REFERENCES):
        raise ValueError("Unexpected metric row count")
    expected_private = sum(item["entries"] for item in EXPECTED_REFERENCES.values()) * expected_combinations
    if len(private) != expected_private:
        raise ValueError("Unexpected private row count")
    if len(bootstrap) != args.bootstrap_replicates * expected_combinations * len(EXPECTED_REFERENCES):
        raise ValueError("Unexpected bootstrap replicate count")
    if len(random) != args.random_replicates * len(ESTIMATORS) * len(EXPECTED_REFERENCES):
        raise ValueError("Unexpected random replicate count")
    if rankings.duplicated(["estimator", "policy", "entry_key"]).any():
        raise ValueError("Duplicate ranking keys")
    if selections.duplicated(["estimator", "policy", "entry_key"]).any():
        raise ValueError("Duplicate selection keys")
    observed_budgets = selections.groupby(["estimator", "policy"]).size()
    for (estimator, _), count in observed_budgets.items():
        if count != ENTRY_BUDGETS[estimator]:
            raise ValueError("Selection budget mismatch")
    print("CL versus uncertainty ablation verification passed.")
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
