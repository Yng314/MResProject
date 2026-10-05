#!/usr/bin/env python3
"""Compare locked prioritization policies over frozen CL-hard candidates."""

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


PROTOCOL = "cl_prioritization_followup_v1"
CURRENT = "current_whole_study_mean"
POLICIES = [CURRENT, "worst_entry", "hard_entry_mean", "entry_direct"]
ALTERNATIVES = POLICIES[1:]
STUDY_POLICIES = POLICIES[:3]
POLICY_LABELS = {
    CURRENT: "Whole-study mean",
    "worst_entry": "Worst entry",
    "hard_entry_mean": "Hard-entry mean",
    "entry_direct": "Direct entry",
}
EXPECTED_CURRENT_COUNTS = {
    "seed_13": {"pool_studies": 982, "selected_studies": 196, "selected_entries": 326},
    "seed_42": {"pool_studies": 911, "selected_studies": 182, "selected_entries": 290},
    "seed_97": {"pool_studies": 871, "selected_studies": 174, "selected_entries": 289},
    "seed_123": {"pool_studies": 978, "selected_studies": 196, "selected_entries": 332},
    "ensemble": {"pool_studies": 676, "selected_studies": 135, "selected_entries": 186},
}
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
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--random-replicates", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20_260_803)
    return parser.parse_args()


def study_pool(frame: pd.DataFrame) -> pd.DataFrame:
    studies = (
        frame.groupby(["subject_id", "study_id"], as_index=False)
        .agg(
            valid_entry_count=("entry_key", "size"),
            whole_study_mean=("quality_self_confidence", "mean"),
            worst_entry=("quality_self_confidence", "min"),
            hard_entry_count=("cl_hard_issue", "sum"),
        )
    )
    hard_means = (
        frame[frame["cl_hard_issue"].eq(1)]
        .groupby("study_id", as_index=False)["quality_self_confidence"]
        .mean()
        .rename(columns={"quality_self_confidence": "hard_entry_mean"})
    )
    studies = studies.merge(hard_means, on="study_id", how="inner", validate="one_to_one")
    studies = studies[studies["hard_entry_count"].gt(0)].copy()
    return studies


def select_stage(args: argparse.Namespace) -> None:
    if not 0 < args.top_fraction <= 1:
        raise ValueError("--top-fraction must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / ".evaluation_complete").exists():
        raise RuntimeError("Refusing to modify a completed evaluation")
    scores = load_scores(args.score_csv, args.score_manifest)

    study_rankings: list[pd.DataFrame] = []
    entry_rankings: list[pd.DataFrame] = []
    selected_entries: list[pd.DataFrame] = []
    counts: dict[str, dict[str, dict[str, int | str]]] = {}

    for estimator in ESTIMATORS:
        frame = scores[scores["estimator"].eq(estimator)].copy()
        pool = study_pool(frame)
        target_studies = max(1, min(len(pool), int(round(len(pool) * args.top_fraction))))
        estimator_counts: dict[str, dict[str, int | str]] = {}

        for policy in STUDY_POLICIES:
            score_column = {
                CURRENT: "whole_study_mean",
                "worst_entry": "worst_entry",
                "hard_entry_mean": "hard_entry_mean",
            }[policy]
            ranking = pool.sort_values(
                [score_column, "hard_entry_count", "study_id"],
                ascending=[True, False, True],
                kind="stable",
            ).reset_index(drop=True)
            ranking["policy"] = policy
            ranking["policy_score"] = ranking[score_column]
            ranking["unit_rank"] = np.arange(1, len(ranking) + 1)
            ranking["selected"] = ranking["unit_rank"].le(target_studies)
            ranking["estimator"] = estimator
            study_rankings.append(ranking)

            chosen_studies = set(ranking.loc[ranking["selected"], "study_id"].astype(int))
            expanded = frame[
                frame["study_id"].isin(chosen_studies) & frame["cl_hard_issue"].eq(1)
            ].copy()
            rank_lookup = ranking.set_index("study_id")["unit_rank"]
            expanded["policy"] = policy
            expanded["selection_unit"] = "study"
            expanded["unit_rank"] = expanded["study_id"].map(rank_lookup).astype(int)
            expanded["policy_score"] = expanded["study_id"].map(
                ranking.set_index("study_id")["policy_score"]
            )
            selected_entries.append(expanded)
            estimator_counts[policy] = {
                "selection_unit": "study",
                "pool_units": int(len(pool)),
                "selected_units": int(target_studies),
                "selected_studies": int(len(chosen_studies)),
                "selected_entries": int(len(expanded)),
            }

        current_entry_budget = int(estimator_counts[CURRENT]["selected_entries"])
        hard = frame[frame["cl_hard_issue"].eq(1)].copy()
        hard = hard.sort_values(
            ["quality_self_confidence", "suspicion_percentile_self", "label_name", "entry_key"],
            ascending=[True, False, True, True],
            kind="stable",
        ).reset_index(drop=True)
        hard["policy"] = "entry_direct"
        hard["unit_rank"] = np.arange(1, len(hard) + 1)
        hard["selected"] = hard["unit_rank"].le(current_entry_budget)
        hard["estimator"] = estimator
        entry_rankings.append(hard)
        direct = hard[hard["selected"]].copy()
        direct["selection_unit"] = "entry"
        direct["policy_score"] = direct["quality_self_confidence"]
        selected_entries.append(direct)
        estimator_counts["entry_direct"] = {
            "selection_unit": "entry",
            "pool_units": int(len(hard)),
            "selected_units": current_entry_budget,
            "selected_studies": int(direct["study_id"].nunique()),
            "selected_entries": int(len(direct)),
        }
        counts[estimator] = estimator_counts

        current = estimator_counts[CURRENT]
        expected = EXPECTED_CURRENT_COUNTS[estimator]
        observed = {
            "pool_studies": current["pool_units"],
            "selected_studies": current["selected_studies"],
            "selected_entries": current["selected_entries"],
        }
        if observed != expected:
            raise ValueError(f"Current-policy reproduction failed for {estimator}: {observed}")

    study_table = pd.concat(study_rankings, ignore_index=True)
    entry_table = pd.concat(entry_rankings, ignore_index=True)
    selected_table = pd.concat(selected_entries, ignore_index=True)
    if selected_table.duplicated(["estimator", "policy", "entry_key"]).any():
        raise ValueError("Selected entries contain duplicate policy-entry keys")
    if not selected_table["cl_hard_issue"].eq(1).all():
        raise ValueError("A selected entry is not CL-hard")
    for table in [study_table, entry_table, selected_table]:
        if PROHIBITED_BLIND_COLUMNS.intersection(table.columns):
            raise ValueError("Blind policy artifact contains prohibited outcome columns")

    files = {
        "study_rankings": args.output_dir / "study_policy_rankings_blind.csv",
        "entry_ranking": args.output_dir / "entry_policy_ranking_blind.csv",
        "selected_entries": args.output_dir / "policy_selected_entries_blind.csv",
    }
    study_table.to_csv(files["study_rankings"], index=False)
    entry_table.to_csv(files["entry_ranking"], index=False)
    selected_table.to_csv(files["selected_entries"], index=False)
    manifest = {
        "protocol": PROTOCOL,
        "phase": "outcome_blind_locked_policy_selection",
        "expert_reference_used": False,
        "score_csv_sha256": sha256_file(args.score_csv),
        "score_manifest_sha256": sha256_file(args.score_manifest),
        "top_fraction": args.top_fraction,
        "policies": POLICIES,
        "selection_counts": counts,
        "output_sha256": {name: sha256_file(path) for name, path in files.items()},
    }
    write_json(args.output_dir / "selection_manifest.json", manifest)
    (args.output_dir / ".selection_complete").write_text("complete\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))


def verify_selection(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    require_file(args.output_dir / ".selection_complete", "selection marker")
    manifest_path = require_file(args.output_dir / "selection_manifest.json", "selection manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") != PROTOCOL or manifest.get("expert_reference_used") is not False:
        raise ValueError("Invalid or non-blind selection manifest")
    if manifest.get("score_csv_sha256") != sha256_file(args.score_csv):
        raise ValueError("Selection score hash mismatch")
    if manifest.get("score_manifest_sha256") != sha256_file(args.score_manifest):
        raise ValueError("Selection score-manifest hash mismatch")
    paths = {
        "study_rankings": args.output_dir / "study_policy_rankings_blind.csv",
        "entry_ranking": args.output_dir / "entry_policy_ranking_blind.csv",
        "selected_entries": args.output_dir / "policy_selected_entries_blind.csv",
    }
    tables = {}
    for name, path in paths.items():
        require_file(path, name)
        if manifest["output_sha256"].get(name) != sha256_file(path):
            raise ValueError(f"Blind output hash mismatch: {name}")
        tables[name] = pd.read_csv(path)
        if PROHIBITED_BLIND_COLUMNS.intersection(tables[name].columns):
            raise ValueError(f"Blind outcome leakage in {name}")
    return tables["study_rankings"], tables["entry_ranking"], tables["selected_entries"], manifest


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
    base = frames[CURRENT].reset_index(drop=True)
    groups = [group.index.to_numpy(dtype=int) for _, group in base.groupby("subject_id", sort=True)]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for replicate in range(replicates):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in sampled])
        for policy in POLICIES:
            values = policy_metrics(frames[policy].iloc[indices])
            rows.append(
                {"reference": reference, "estimator": estimator, "policy": policy,
                 "replicate": replicate, **{metric: values[metric] for metric in PAIR_METRICS}}
            )
    return pd.DataFrame(rows)


def summarize_bootstrap(replicates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ci_rows: list[dict[str, object]] = []
    for keys, group in replicates.groupby(["reference", "estimator", "policy"], sort=True):
        for metric in PAIR_METRICS:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            ci_rows.append({
                "reference": keys[0], "estimator": keys[1], "policy": keys[2], "metric": metric,
                "valid_replicates": len(values),
                "ci_lower": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                "ci_upper": float(np.quantile(values, 0.975)) if len(values) else np.nan,
            })

    diff_rows: list[dict[str, object]] = []
    current = replicates[replicates["policy"].eq(CURRENT)].set_index(
        ["reference", "estimator", "replicate"]
    )
    for alternative in ALTERNATIVES:
        alt = replicates[replicates["policy"].eq(alternative)].set_index(
            ["reference", "estimator", "replicate"]
        )
        if not alt.index.equals(current.index):
            raise ValueError("Paired bootstrap indices differ across policies")
        for metric in PAIR_METRICS:
            delta = alt[metric].to_numpy(dtype=float) - current[metric].to_numpy(dtype=float)
            finite = delta[np.isfinite(delta)]
            index_frame = alt.reset_index()[["reference", "estimator"]]
            for (reference, estimator), positions in index_frame.groupby(
                ["reference", "estimator"], sort=True
            ).groups.items():
                values = delta[np.asarray(list(positions), dtype=int)]
                values = values[np.isfinite(values)]
                diff_rows.append({
                    "reference": reference, "estimator": estimator,
                    "alternative": alternative, "metric": metric,
                    "valid_replicates": len(values),
                    "mean_difference": float(values.mean()) if len(values) else np.nan,
                    "ci_lower": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                    "ci_upper": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                    "probability_difference_gt_zero": float(np.mean(values > 0)) if len(values) else np.nan,
                })
    return pd.DataFrame(ci_rows), pd.DataFrame(diff_rows)


def random_policy_replicates(
    scores: pd.DataFrame,
    study_rankings: pd.DataFrame,
    entry_ranking: pd.DataFrame,
    references: dict[str, pd.DataFrame],
    manifest: dict,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for estimator_index, estimator in enumerate(ESTIMATORS):
        estimator_scores = scores[scores["estimator"].eq(estimator)]
        hard = estimator_scores[estimator_scores["cl_hard_issue"].eq(1)][["study_id", "entry_key"]]
        study_ids = (
            study_rankings[
                study_rankings["estimator"].eq(estimator)
                & study_rankings["policy"].eq(CURRENT)
            ]["study_id"].to_numpy(dtype=int)
        )
        hard_keys = entry_ranking[entry_ranking["estimator"].eq(estimator)]["entry_key"].astype(str).to_numpy()
        for reference_index, (reference_name, reference) in enumerate(references.items()):
            issue_lookup = reference.set_index("entry_key")["true_issue"]
            known_lookup = set(reference["entry_key"].astype(str))
            hard_with_ref = hard.copy()
            hard_with_ref["known"] = hard_with_ref["entry_key"].astype(str).isin(known_lookup).astype(int)
            hard_with_ref["issue"] = hard_with_ref["entry_key"].map(issue_lookup).fillna(0).astype(int)
            per_study = hard_with_ref.groupby("study_id").agg(
                reference_entries=("known", "sum"), true_issues=("issue", "sum")
            )
            study_covered = per_study["reference_entries"].reindex(study_ids, fill_value=0).to_numpy(int)
            study_issues = per_study["true_issues"].reindex(study_ids, fill_value=0).to_numpy(int)
            entry_known = np.fromiter((key in known_lookup for key in hard_keys), dtype=int)
            entry_issues = np.fromiter((int(issue_lookup.get(key, 0)) for key in hard_keys), dtype=int)
            total_issues = int(reference["true_issue"].sum())

            for policy_index, policy in enumerate(POLICIES):
                info = manifest["selection_counts"][estimator][policy]
                rng = np.random.default_rng(
                    seed + estimator_index * 101 + reference_index * 10_007 + policy_index * 1_000_003
                )
                for replicate in range(replicates):
                    if info["selection_unit"] == "study":
                        chosen = rng.choice(len(study_ids), size=int(info["selected_units"]), replace=False)
                        covered = int(study_covered[chosen].sum())
                        issues = int(study_issues[chosen].sum())
                    else:
                        chosen = rng.choice(len(hard_keys), size=int(info["selected_units"]), replace=False)
                        covered = int(entry_known[chosen].sum())
                        issues = int(entry_issues[chosen].sum())
                    rows.append({
                        "reference": reference_name, "estimator": estimator, "policy": policy,
                        "replicate": replicate, "reference_covered_entries": covered,
                        "true_issues": issues, "precision": safe_div(issues, covered),
                        "recall": safe_div(issues, total_issues),
                    })
    return pd.DataFrame(rows)


def summarize_random(replicates: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    observed = metrics.set_index(["reference", "estimator", "policy"])
    rows: list[dict[str, object]] = []
    for keys, group in replicates.groupby(["reference", "estimator", "policy"], sort=True):
        obs = observed.loc[keys]
        issues = group["true_issues"].to_numpy(int)
        coverage = group["reference_covered_entries"].to_numpy(int)
        precision = group["precision"].to_numpy(float)
        finite_precision = precision[np.isfinite(precision)]
        rows.append({
            "reference": keys[0], "estimator": keys[1], "policy": keys[2],
            "observed_true_issues": int(obs["true_issues"]),
            "random_true_issues_mean": float(issues.mean()),
            "random_true_issues_ci_lower": float(np.quantile(issues, 0.025)),
            "random_true_issues_ci_upper": float(np.quantile(issues, 0.975)),
            "random_reference_coverage_mean": float(coverage.mean()),
            "random_precision_mean": float(finite_precision.mean()) if len(finite_precision) else np.nan,
            "empirical_p_capture_ge_observed": float((1 + np.count_nonzero(issues >= obs["true_issues"])) / (1 + len(issues))),
            "empirical_p_precision_ge_observed": float((1 + np.count_nonzero(precision >= obs["precision"])) / (1 + np.count_nonzero(np.isfinite(precision)))) if np.isfinite(obs["precision"]) else np.nan,
        })
    return pd.DataFrame(rows)


def build_figure(metrics: pd.DataFrame, random_summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    names = {"mimic_cxr_2_1": "MIMIC-CXR 2.1", "medpalm": "Med-PaLM"}
    colors = ["#777777", "#4c78a8", "#59a14f", "#e15759"]
    for row, reference in enumerate(["mimic_cxr_2_1", "medpalm"]):
        frame = metrics[(metrics["reference"].eq(reference)) & metrics["estimator"].eq("ensemble")]
        frame = frame.set_index("policy").loc[POLICIES].reset_index()
        random = random_summary[(random_summary["reference"].eq(reference)) & random_summary["estimator"].eq("ensemble")]
        random = random.set_index("policy").loc[POLICIES].reset_index()
        labels = [POLICY_LABELS[p] for p in POLICIES]
        x = np.arange(len(POLICIES))
        axes[row, 0].bar(x, frame["precision"], color=colors)
        axes[row, 0].scatter(x, random["random_precision_mean"], marker="D", color="black", label="Matched random mean")
        axes[row, 0].set_xticks(x, labels, rotation=12)
        axes[row, 0].set_ylabel("Known-issue precision")
        axes[row, 0].set_title(f"{names[reference]}: issue yield")
        axes[row, 0].legend(frameon=False)

        axes[row, 1].bar(x, frame["true_issues"], color=colors)
        axes[row, 1].scatter(x, random["random_true_issues_mean"], marker="D", color="black", label="Matched random mean")
        for index, item in frame.iterrows():
            axes[row, 1].text(index, item["true_issues"] + 0.3, f"n={int(item['selected_entries'])}", ha="center", fontsize=8)
        axes[row, 1].set_xticks(x, labels, rotation=12)
        axes[row, 1].set_ylabel("Known issues captured")
        axes[row, 1].set_title(f"{names[reference]}: capture and full-pool entry budget")
        axes[row, 1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def evaluate_stage(args: argparse.Namespace) -> None:
    if (args.output_dir / ".evaluation_complete").exists():
        raise RuntimeError("Refusing to overwrite a completed evaluation")
    study_rankings, entry_ranking, selected_entries, selection_manifest = verify_selection(args)
    scores = load_scores(args.score_csv, args.score_manifest)
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
            policy_frames: dict[str, pd.DataFrame] = {}
            base = scores[scores["estimator"].eq(estimator)].merge(
                reference[["entry_key", "subject_id", "study_id", "label_name", "true_issue"]],
                on=["entry_key", "subject_id", "study_id", "label_name"], how="inner", validate="one_to_one"
            )
            for policy in POLICIES:
                selected_keys = set(selected_entries[
                    selected_entries["estimator"].eq(estimator) & selected_entries["policy"].eq(policy)
                ]["entry_key"].astype(str))
                frame = base.copy()
                frame["selected_by_policy"] = frame["entry_key"].astype(str).isin(selected_keys)
                frame["reference"] = reference_name
                frame["policy"] = policy
                policy_frames[policy] = frame
                private_frames.append(frame)
                values = policy_metrics(frame)
                counts = selection_manifest["selection_counts"][estimator][policy]
                metric_rows.append({"reference": reference_name, "estimator": estimator, "policy": policy, **counts, **values})
                for label_name, label_frame in frame.groupby("label_name", sort=True):
                    label_rows.append({"reference": reference_name, "estimator": estimator, "policy": policy,
                                       "label_name": label_name, **policy_metrics(label_frame)})
            bootstrap_frames.append(bootstrap_policies(
                policy_frames, reference_name, estimator, args.bootstrap_replicates,
                args.random_seed + reference_index * 10_007 + estimator_index * 101
            ))

    private_rows = pd.concat(private_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    label_metrics = pd.DataFrame(label_rows)
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    bootstrap_ci, paired_differences = summarize_bootstrap(bootstrap)
    random_replicates = random_policy_replicates(
        scores, study_rankings, entry_ranking, references, selection_manifest,
        args.random_replicates, args.random_seed
    )
    random_summary = summarize_random(random_replicates, metrics)
    metrics = metrics.merge(random_summary, on=["reference", "estimator", "policy"], validate="one_to_one")

    primary = metrics[metrics["reference"].eq("mimic_cxr_2_1")]
    ensemble = primary[primary["estimator"].eq("ensemble")].set_index("policy")
    seed_frame = primary[primary["estimator"].ne("ensemble")]
    decisions = {}
    for policy in ALTERNATIVES:
        alt = ensemble.loc[policy]
        current = ensemble.loc[CURRENT]
        seed_wins = 0
        for estimator in ["seed_13", "seed_42", "seed_97", "seed_123"]:
            rows = seed_frame[seed_frame["estimator"].eq(estimator)].set_index("policy")
            seed_wins += int(rows.loc[policy, "true_issues"] > rows.loc[CURRENT, "true_issues"])
        gates = {
            "more_issues_than_current": bool(alt["true_issues"] > current["true_issues"]),
            "precision_not_lower_than_current": bool(alt["precision"] >= current["precision"]),
            "matched_random_p_capture_below_0_05": bool(alt["empirical_p_capture_ge_observed"] < 0.05),
            "individual_seed_capture_wins_at_least_3_of_4": bool(seed_wins >= 3),
        }
        decisions[policy] = {"status": "promising" if all(gates.values()) else "not_promising", "seed_capture_wins": seed_wins, "gates": gates}

    outputs = {
        "private_rows": args.output_dir / "policy_reference_rows_private.csv",
        "metrics": args.output_dir / "policy_metrics.csv",
        "label_metrics": args.output_dir / "policy_label_metrics.csv",
        "bootstrap": args.output_dir / "policy_subject_bootstrap_replicates.csv",
        "bootstrap_ci": args.output_dir / "policy_subject_bootstrap_ci.csv",
        "paired_differences": args.output_dir / "policy_paired_bootstrap_differences.csv",
        "random_replicates": args.output_dir / "policy_matched_random_replicates.csv",
        "random_summary": args.output_dir / "policy_matched_random_summary.csv",
        "figure": args.output_dir / "cl_prioritization_policy_comparison.png",
    }
    private_rows.to_csv(outputs["private_rows"], index=False)
    metrics.to_csv(outputs["metrics"], index=False)
    label_metrics.to_csv(outputs["label_metrics"], index=False)
    bootstrap.to_csv(outputs["bootstrap"], index=False)
    bootstrap_ci.to_csv(outputs["bootstrap_ci"], index=False)
    paired_differences.to_csv(outputs["paired_differences"], index=False)
    random_replicates.to_csv(outputs["random_replicates"], index=False)
    random_summary.to_csv(outputs["random_summary"], index=False)
    build_figure(metrics, random_summary, outputs["figure"])
    summary = {"protocol": PROTOCOL, "primary_reference": "mimic_cxr_2_1", "primary_estimator": "ensemble",
               "decision_rule": decisions,
               "primary_metrics": ensemble.reset_index().to_dict(orient="records")}
    write_json(args.output_dir / "cl_prioritization_followup_summary.json", summary)
    evaluation_manifest = {
        "protocol": PROTOCOL,
        "selection_manifest_sha256": sha256_file(args.output_dir / "selection_manifest.json"),
        "reference_input_sha256": {
            "mimic": sha256_file(args.mimic_reference_csv),
            "medpalm_blinded": sha256_file(args.medpalm_blinded_csv),
            "medpalm_reference": sha256_file(args.medpalm_reference_csv),
        },
        "output_sha256": {name: sha256_file(path) for name, path in outputs.items()},
        "summary_sha256": sha256_file(args.output_dir / "cl_prioritization_followup_summary.json"),
    }
    write_json(args.output_dir / "evaluation_manifest.json", evaluation_manifest)
    (args.output_dir / ".evaluation_complete").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def verify_stage(args: argparse.Namespace) -> None:
    study_rankings, entry_ranking, selected_entries, manifest = verify_selection(args)
    require_file(args.output_dir / ".evaluation_complete", "evaluation marker")
    eval_manifest = json.loads(require_file(args.output_dir / "evaluation_manifest.json", "evaluation manifest").read_text())
    outputs = {
        "private_rows": args.output_dir / "policy_reference_rows_private.csv",
        "metrics": args.output_dir / "policy_metrics.csv",
        "label_metrics": args.output_dir / "policy_label_metrics.csv",
        "bootstrap": args.output_dir / "policy_subject_bootstrap_replicates.csv",
        "bootstrap_ci": args.output_dir / "policy_subject_bootstrap_ci.csv",
        "paired_differences": args.output_dir / "policy_paired_bootstrap_differences.csv",
        "random_replicates": args.output_dir / "policy_matched_random_replicates.csv",
        "random_summary": args.output_dir / "policy_matched_random_summary.csv",
        "figure": args.output_dir / "cl_prioritization_policy_comparison.png",
    }
    for name, path in outputs.items():
        require_file(path, name)
        if eval_manifest["output_sha256"].get(name) != sha256_file(path):
            raise ValueError(f"Evaluation output hash mismatch: {name}")
    if eval_manifest.get("summary_sha256") != sha256_file(require_file(
        args.output_dir / "cl_prioritization_followup_summary.json", "summary"
    )):
        raise ValueError("Summary hash mismatch")
    metrics = pd.read_csv(outputs["metrics"])
    private = pd.read_csv(outputs["private_rows"])
    bootstrap = pd.read_csv(outputs["bootstrap"])
    random = pd.read_csv(outputs["random_replicates"])
    if len(metrics) != len(EXPECTED_REFERENCES) * len(ESTIMATORS) * len(POLICIES):
        raise ValueError("Unexpected policy-metric row count")
    expected_private = sum(item["entries"] for item in EXPECTED_REFERENCES.values()) * len(ESTIMATORS) * len(POLICIES)
    if len(private) != expected_private:
        raise ValueError("Unexpected private reference row count")
    if len(bootstrap) != args.bootstrap_replicates * len(EXPECTED_REFERENCES) * len(ESTIMATORS) * len(POLICIES):
        raise ValueError("Unexpected bootstrap replicate count")
    if len(random) != args.random_replicates * len(EXPECTED_REFERENCES) * len(ESTIMATORS) * len(POLICIES):
        raise ValueError("Unexpected random-policy replicate count")
    if study_rankings.duplicated(["estimator", "policy", "study_id"]).any():
        raise ValueError("Duplicate study rankings")
    if entry_ranking.duplicated(["estimator", "entry_key"]).any():
        raise ValueError("Duplicate entry rankings")
    if selected_entries.duplicated(["estimator", "policy", "entry_key"]).any():
        raise ValueError("Duplicate selected entries")
    print("CL prioritization follow-up verification passed.")
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
