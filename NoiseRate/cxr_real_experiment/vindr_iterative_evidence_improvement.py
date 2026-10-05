#!/usr/bin/env python3
"""Iterative VinDr evidence-improvement experiment with fixed probe errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from vindr_iterative_oracle_cleaning import (
    apply_oracle_labels,
    cohort_entries,
    true_state,
    validate_evidence,
    validate_private_reference,
)
from vindr_known_gt_cl_benchmark import (
    LABELS,
    atomic_write_csv,
    atomic_write_text,
    exact_sign_flip_p,
    holm_adjust,
    parse_seeds,
    require_columns,
    sha256_file,
    validate_blind_columns,
)


PROTOCOL_NAME = "vindr_iterative_evidence_improvement_v1"
INITIALIZATIONS = ("scratch", "xrv_pretrained")
DEFAULT_SEEDS = (13, 42, 97, 123, 211, 307)
ENTRIES = 18_000
IMAGES = 3_000
PROBE_IMAGES = 600
PROBE_ENTRIES = 3_600
REVIEW_BUDGET = 360
PROBE_REVIEW_BUDGET = 72
INITIAL_ERRORS = 3_600


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2))


def state_dir(output: Path) -> Path:
    return output / "state"


def loop_dir(output: Path, loop_id: int) -> Path:
    return output / f"loop_{loop_id:02d}"


def evidence_dir(output: Path, loop_id: int) -> Path:
    if loop_id == 0:
        return loop_dir(output, 0) / "oof_evidence"
    return loop_dir(output, loop_id) / "oof_after_action"


def cohort_path(args: argparse.Namespace, loop_id: int) -> Path:
    if loop_id == 0:
        return Path(args.source_prepared) / "blind_noisy_cohort.csv"
    return loop_dir(Path(args.output_dir), loop_id) / "cohort_after_action_blind.csv"


def choose_probe_images(
    image_ids: Iterable[str],
    seed: int,
    probe_images: int = PROBE_IMAGES,
) -> list[str]:
    image_ids = np.array(sorted(map(str, image_ids)))
    if len(image_ids) != IMAGES:
        raise ValueError("Probe split requires exactly 3,000 images")
    rng = np.random.default_rng(seed + 20260813)
    return np.sort(rng.choice(image_ids, size=probe_images, replace=False)).tolist()


def rank_entries(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "image_id", "label_name", "label_index", "cl_first_score",
        "self_confidence_suspicion",
    ]
    require_columns(frame, required, "ranking frame")
    return frame.sort_values(
        ["cl_first_score", "self_confidence_suspicion", "label_index", "image_id"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def select_fixed_budget(
    evidence: pd.DataFrame,
    reviewed_keys: set[str],
    probe_image_ids: set[str],
    budget: int,
) -> tuple[pd.DataFrame, int]:
    candidates = evidence[
        ~evidence["entry_key"].isin(reviewed_keys)
        & ~evidence["image_id"].astype(str).isin(probe_image_ids)
    ].copy()
    ranked = rank_entries(candidates)
    return ranked.head(min(budget, len(ranked))).copy(), int(len(ranked))


def initialize(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    state = state_dir(output)
    state.mkdir()
    prepared = Path(args.source_prepared)
    cohort_path_initial = prepared / "blind_noisy_cohort.csv"
    private_path = Path(args.private_reference)
    cohort = pd.read_csv(cohort_path_initial)
    private = validate_private_reference(pd.read_csv(private_path))
    validate_blind_columns(cohort)
    entries = cohort_entries(cohort)
    initial = true_state(cohort, private)
    if (
        len(cohort) != IMAGES
        or len(entries) != ENTRIES
        or initial["remaining_errors"] != INITIAL_ERRORS
    ):
        raise ValueError("Initial VinDr state does not match the locked 20% corruption")

    probe_ids = choose_probe_images(cohort["image_id"], args.seed)
    probe = pd.DataFrame({"image_id": probe_ids})
    probe_entries = private[private["image_id"].astype(str).isin(set(probe_ids))]
    if len(probe_entries) != PROBE_ENTRIES:
        raise RuntimeError("Probe split does not contain exactly 3,600 entries")
    probe_errors = int(probe_entries["injected_error"].sum())
    if not (600 <= probe_errors <= 840):
        raise RuntimeError("Probe error count is unexpectedly far from the 20% target")
    support = (
        probe_entries[probe_entries["injected_error"].astype(bool)]
        .groupby(["label_name", "flip_direction"], as_index=False)
        .size()
        .rename(columns={"size": "probe_errors"})
    )

    atomic_write_csv(probe, state / "probe_images_blind.csv")
    atomic_write_csv(support, state / "probe_support_private.csv")
    atomic_write_csv(
        pd.DataFrame(columns=["loop", "entry_key", "image_id", "label_name", "current_label"]),
        state / "review_history_blind.csv",
    )
    atomic_write_csv(
        pd.DataFrame(columns=[
            "loop", "entry_key", "image_id", "label_name", "current_label",
            "clean_label", "true_issue", "flip_direction",
        ]),
        state / "review_history_private.csv",
    )
    manifest = {
        "protocol": PROTOCOL_NAME,
        "seed": int(args.seed),
        "initialization": args.initialization,
        "loops": int(args.loops),
        "samples": IMAGES,
        "entries": ENTRIES,
        "initial_errors": INITIAL_ERRORS,
        "initial_quality": float(initial["true_quality"]),
        "probe_images": PROBE_IMAGES,
        "probe_entries": PROBE_ENTRIES,
        "probe_errors": probe_errors,
        "probe_split_method": "seeded random image split without outcome stratification",
        "review_budget_per_loop": int(args.review_budget),
        "probe_review_budget": int(args.probe_review_budget),
        "source_cohort_sha256": sha256_file(cohort_path_initial),
        "private_reference_sha256": sha256_file(private_path),
        "probe_images_sha256": sha256_file(state / "probe_images_blind.csv"),
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(manifest, state / "initialization_manifest.json")
    atomic_write_text(state / ".initialized", "complete\n")
    print(json.dumps(manifest, indent=2), flush=True)


def current_inputs(args: argparse.Namespace, loop_id: int) -> tuple[Path, Path]:
    output = Path(args.output_dir)
    if loop_id == 1:
        return cohort_path(args, 0), evidence_dir(output, 0) / "entry_evidence.csv"
    return cohort_path(args, loop_id - 1), evidence_dir(output, loop_id - 1) / "entry_evidence.csv"


def select(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    if not (state_dir(output) / ".initialized").is_file():
        raise FileNotFoundError("Experiment is not initialized")
    target = loop_dir(output, args.loop_id)
    target.mkdir(parents=True, exist_ok=False)
    current_cohort_path, current_evidence_path = current_inputs(args, args.loop_id)
    if not (current_evidence_path.parent / ".blind_run_complete").is_file():
        raise FileNotFoundError("Current OOF evidence is incomplete")
    cohort = pd.read_csv(current_cohort_path)
    evidence = validate_evidence(cohort, pd.read_csv(current_evidence_path))
    history = pd.read_csv(state_dir(output) / "review_history_blind.csv")
    reviewed = set(history["entry_key"].astype(str)) if not history.empty else set()
    probe_ids = set(pd.read_csv(state_dir(output) / "probe_images_blind.csv")["image_id"].astype(str))
    selected, candidate_count = select_fixed_budget(
        evidence, reviewed, probe_ids, args.review_budget
    )
    if len(selected) != args.review_budget:
        raise RuntimeError("Action pool did not provide the fixed review budget")
    blind_columns = [
        "entry_key", "image_id", "fold_id", "label_index", "label_name",
        "current_label", "cl_issue", "cl_first_score",
        "self_confidence_suspicion", "oof_probability",
    ]
    selected_path = target / "selected_entries_blind.csv"
    atomic_write_csv(selected[blind_columns], selected_path)
    manifest = {
        "protocol": PROTOCOL_NAME,
        "seed": int(args.seed),
        "initialization": args.initialization,
        "loop": int(args.loop_id),
        "action_candidates": candidate_count,
        "review_budget": int(args.review_budget),
        "selected_entries": int(len(selected)),
        "selected_cl_hard_entries": int(selected["cl_issue"].sum()),
        "probe_images_excluded": len(probe_ids),
        "previously_reviewed_excluded": len(reviewed),
        "source_cohort_sha256": sha256_file(current_cohort_path),
        "source_evidence_sha256": sha256_file(current_evidence_path),
        "selected_entries_sha256": sha256_file(selected_path),
    }
    atomic_json(manifest, target / "selection_manifest_blind.json")
    atomic_write_text(target / ".selection_complete", "complete\n")
    print(json.dumps(manifest, indent=2), flush=True)


def oracle_update(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    target = loop_dir(output, args.loop_id)
    if not (target / ".selection_complete").is_file():
        raise FileNotFoundError("Selection is incomplete")
    current_cohort_path, _ = current_inputs(args, args.loop_id)
    cohort = pd.read_csv(current_cohort_path)
    selected = pd.read_csv(target / "selected_entries_blind.csv")
    private = validate_private_reference(pd.read_csv(args.private_reference))
    joined = selected.merge(
        private[["entry_key", "clean_label", "flip_direction"]],
        on="entry_key", how="left", validate="one_to_one",
    )
    if joined["clean_label"].isna().any():
        raise RuntimeError("Selected entries do not align with private reference")
    joined["true_issue"] = joined["current_label"].astype(int).ne(joined["clean_label"].astype(int)).astype(int)
    updated = apply_oracle_labels(cohort, joined)
    validate_blind_columns(updated)
    updated_path = target / "cohort_after_action_blind.csv"
    private_selected_path = target / "selected_entries_private.csv"
    atomic_write_csv(updated, updated_path)
    atomic_write_csv(joined, private_selected_path)

    history_blind_path = state_dir(output) / "review_history_blind.csv"
    history_private_path = state_dir(output) / "review_history_private.csv"
    blind_history = pd.read_csv(history_blind_path)
    private_history = pd.read_csv(history_private_path)
    blind_add = joined[["entry_key", "image_id", "label_name", "current_label"]].copy()
    blind_add.insert(0, "loop", args.loop_id)
    private_add = joined[[
        "entry_key", "image_id", "label_name", "current_label", "clean_label",
        "true_issue", "flip_direction",
    ]].copy()
    private_add.insert(0, "loop", args.loop_id)
    blind_history = pd.concat([blind_history, blind_add], ignore_index=True)
    private_history = pd.concat([private_history, private_add], ignore_index=True)
    if blind_history["entry_key"].duplicated().any() or private_history["entry_key"].duplicated().any():
        raise RuntimeError("An entry was reviewed more than once")
    atomic_write_csv(blind_history, history_blind_path)
    atomic_write_csv(private_history, history_private_path)
    before = true_state(cohort, private)
    after = true_state(updated, private)
    summary = {
        "protocol": PROTOCOL_NAME,
        "seed": int(args.seed),
        "initialization": args.initialization,
        "loop": int(args.loop_id),
        "selected_entries": int(len(joined)),
        "selected_true_errors": int(joined["true_issue"].sum()),
        "selection_precision": float(joined["true_issue"].mean()),
        "quality_before": float(before["true_quality"]),
        "quality_after": float(after["true_quality"]),
        "remaining_errors_after": int(after["remaining_errors"]),
        "cohort_after_action_sha256": sha256_file(updated_path),
        "selected_entries_private_sha256": sha256_file(private_selected_path),
    }
    atomic_json(summary, target / "oracle_summary_private.json")
    atomic_write_text(target / ".oracle_update_complete", "complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def finalize_loop(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    target = loop_dir(output, args.loop_id)
    if not (target / ".oracle_update_complete").is_file():
        raise FileNotFoundError("Oracle update is incomplete")
    evidence_root = evidence_dir(output, args.loop_id)
    if not (evidence_root / ".blind_run_complete").is_file():
        raise FileNotFoundError("Post-action OOF evidence is incomplete")
    cohort = pd.read_csv(target / "cohort_after_action_blind.csv")
    evidence = validate_evidence(cohort, pd.read_csv(evidence_root / "entry_evidence.csv"))
    summary = json.loads((target / "oracle_summary_private.json").read_text())
    summary.update(
        {
            "raw_dqs_after_action": float(1.0 - evidence["cl_issue"].mean()),
            "cl_issue_entries_after_action": int(evidence["cl_issue"].sum()),
            "post_action_evidence_sha256": sha256_file(evidence_root / "entry_evidence.csv"),
        }
    )
    atomic_json(summary, target / "loop_metrics_private.json")
    atomic_write_text(target / ".loop_complete", "complete\n")


def safe_auroc(truth: Iterable[int], score: Iterable[float]) -> float:
    truth_array = np.asarray(list(truth), dtype=int)
    score_array = np.asarray(list(score), dtype=float)
    return float(roc_auc_score(truth_array, score_array)) if len(np.unique(truth_array)) == 2 else float("nan")


def safe_auprc(truth: Iterable[int], score: Iterable[float]) -> float:
    truth_array = np.asarray(list(truth), dtype=int)
    score_array = np.asarray(list(score), dtype=float)
    return float(average_precision_score(truth_array, score_array)) if truth_array.sum() > 0 else float("nan")


def top_key_set(frame: pd.DataFrame, budget: int) -> set[str]:
    return set(rank_entries(frame).head(min(budget, len(frame)))["entry_key"].astype(str))


def audc(frame: pd.DataFrame, value: str) -> float:
    ordered = frame.sort_values("loop")
    return float(np.trapezoid(ordered[value], ordered["review_fraction"]))


def evaluate(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    for loop_id in range(args.loops + 1):
        marker = evidence_dir(output, loop_id) / ".blind_run_complete"
        if not marker.is_file():
            raise FileNotFoundError(f"OOF evidence marker missing for Loop {loop_id}")
        if loop_id > 0 and not (loop_dir(output, loop_id) / ".loop_complete").is_file():
            raise FileNotFoundError(f"Loop completion marker missing for Loop {loop_id}")
    private = validate_private_reference(pd.read_csv(args.private_reference))
    probe_ids = set(pd.read_csv(state_dir(output) / "probe_images_blind.csv")["image_id"].astype(str))
    history = pd.read_csv(state_dir(output) / "review_history_private.csv")
    initial_cohort = pd.read_csv(cohort_path(args, 0))
    initial_evidence = validate_evidence(
        initial_cohort, pd.read_csv(evidence_dir(output, 0) / "entry_evidence.csv")
    ).merge(
        private[["entry_key", "clean_label", "injected_error", "flip_direction"]],
        on="entry_key", validate="one_to_one",
    )
    initial_probe = initial_evidence[initial_evidence["image_id"].astype(str).isin(probe_ids)].copy()
    initial_probe_top = top_key_set(initial_probe, args.probe_review_budget)
    hard_probe_keys = set(
        initial_probe.loc[
            initial_probe["injected_error"].astype(bool)
            & ~initial_probe["entry_key"].isin(initial_probe_top),
            "entry_key",
        ].astype(str)
    )
    hard_probe = private[private["entry_key"].isin(hard_probe_keys)].copy()
    atomic_write_csv(hard_probe, output / "hard_probe_errors_private.csv")

    evidence_rows = []
    discovery_rows = []
    initial_action = initial_evidence[
        ~initial_evidence["image_id"].astype(str).isin(probe_ids)
    ].copy()
    frozen_ranked = rank_entries(initial_action)
    initial_action_errors = int(initial_action["injected_error"].sum())
    cumulative_dynamic_errors = 0
    for loop_id in range(args.loops + 1):
        cohort = pd.read_csv(cohort_path(args, loop_id))
        evidence = validate_evidence(
            cohort, pd.read_csv(evidence_dir(output, loop_id) / "entry_evidence.csv")
        ).merge(
            private[["entry_key", "clean_label", "injected_error", "flip_direction"]],
            on="entry_key", validate="one_to_one",
        )
        current_entries = cohort_entries(cohort).merge(
            private[["entry_key", "clean_label"]], on="entry_key", validate="one_to_one"
        )
        current_errors = current_entries["current_label"].astype(int).ne(current_entries["clean_label"].astype(int))
        label_aurocs = [
            safe_auroc(group["clean_label"], group["oof_probability"])
            for _, group in evidence.groupby("label_name")
        ]
        probe = evidence[evidence["image_id"].astype(str).isin(probe_ids)].copy()
        probe_truth = probe["injected_error"].astype(int)
        probe_top = top_key_set(probe, args.probe_review_budget)
        probe_selected = probe["entry_key"].isin(probe_top)
        probe_found = int((probe_truth.astype(bool) & probe_selected).sum())
        hard_found = len(hard_probe_keys & probe_top)
        hard_ranks = rank_entries(probe).reset_index().set_index("entry_key")["index"] + 1
        hard_percentile = float(
            np.mean([hard_ranks[key] / len(probe) for key in hard_probe_keys])
        ) if hard_probe_keys else float("nan")
        row = {
            "seed": int(args.seed),
            "initialization": args.initialization,
            "loop": loop_id,
            "true_quality": float(1.0 - current_errors.mean()),
            "remaining_errors": int(current_errors.sum()),
            "raw_dqs": float(1.0 - evidence["cl_issue"].mean()),
            "oof_clean_macro_auroc": float(np.nanmean(label_aurocs)),
            "oof_clean_micro_auroc": safe_auroc(evidence["clean_label"], evidence["oof_probability"]),
            "probe_errors": int(probe_truth.sum()),
            "probe_error_auprc": safe_auprc(probe_truth, probe["cl_first_score"]),
            "probe_error_auroc": safe_auroc(probe_truth, probe["cl_first_score"]),
            "probe_errors_found_at_budget": probe_found,
            "probe_error_recall_at_budget": float(probe_found / max(int(probe_truth.sum()), 1)),
            "probe_precision_at_budget": float(probe_found / args.probe_review_budget),
            "hard_probe_errors": len(hard_probe_keys),
            "hard_probe_found_at_budget": hard_found,
            "hard_probe_recall_at_budget": float(hard_found / max(len(hard_probe_keys), 1)),
            "hard_probe_mean_rank_fraction": hard_percentile,
        }
        for direction in ("0_to_1", "1_to_0"):
            truth = probe["flip_direction"].eq(direction).astype(int)
            row[f"probe_{direction}_auprc"] = safe_auprc(truth, probe["cl_first_score"])
            row[f"probe_{direction}_found_at_budget"] = int((truth.astype(bool) & probe_selected).sum())
            row[f"probe_{direction}_errors"] = int(truth.sum())
        evidence_rows.append(row)

        budget = loop_id * args.review_budget
        if loop_id > 0:
            cumulative_dynamic_errors = int(history.loc[history["loop"] <= loop_id, "true_issue"].sum())
        frozen = frozen_ranked.head(budget)
        frozen_errors = int(frozen["injected_error"].sum())
        random_errors = float(budget * initial_action_errors / len(initial_action))
        for method, errors_found in (
            ("dynamic_cl", float(cumulative_dynamic_errors)),
            ("frozen_loop0_cl", float(frozen_errors)),
            ("random_review_expected", random_errors),
        ):
            discovery_rows.append(
                {
                    "seed": int(args.seed),
                    "initialization": args.initialization,
                    "method": method,
                    "loop": loop_id,
                    "cumulative_reviews": budget,
                    "review_fraction": float(budget / len(initial_action)),
                    "cumulative_errors_found": errors_found,
                    "action_error_recall": float(errors_found / initial_action_errors),
                }
            )

    evidence_frame = pd.DataFrame(evidence_rows)
    discovery_frame = pd.DataFrame(discovery_rows)
    atomic_write_csv(evidence_frame, output / "evidence_trajectory_private.csv")
    atomic_write_csv(discovery_frame, output / "discovery_trajectory_private.csv")
    dynamic = discovery_frame[discovery_frame["method"] == "dynamic_cl"]
    frozen = discovery_frame[discovery_frame["method"] == "frozen_loop0_cl"]
    random = discovery_frame[discovery_frame["method"] == "random_review_expected"]
    first = evidence_frame.iloc[0]
    last = evidence_frame.iloc[-1]
    summary = {
        "protocol": PROTOCOL_NAME,
        "seed": int(args.seed),
        "initialization": args.initialization,
        "loops": int(args.loops),
        "initial_quality": float(first["true_quality"]),
        "final_quality": float(last["true_quality"]),
        "oof_clean_macro_auroc_loop0": float(first["oof_clean_macro_auroc"]),
        "oof_clean_macro_auroc_final": float(last["oof_clean_macro_auroc"]),
        "oof_clean_macro_auroc_change": float(last["oof_clean_macro_auroc"] - first["oof_clean_macro_auroc"]),
        "probe_error_auprc_loop0": float(first["probe_error_auprc"]),
        "probe_error_auprc_final": float(last["probe_error_auprc"]),
        "probe_error_auprc_change": float(last["probe_error_auprc"] - first["probe_error_auprc"]),
        "hard_probe_recall_change": float(last["hard_probe_recall_at_budget"] - first["hard_probe_recall_at_budget"]),
        "dynamic_discovery_audc": audc(dynamic, "action_error_recall"),
        "frozen_discovery_audc": audc(frozen, "action_error_recall"),
        "random_discovery_audc": audc(random, "action_error_recall"),
        "dynamic_minus_frozen_audc": audc(dynamic, "action_error_recall") - audc(frozen, "action_error_recall"),
        "final_dynamic_errors_found": int(history["true_issue"].sum()),
        "hard_probe_errors": len(hard_probe_keys),
        "evidence_trajectory_sha256": sha256_file(output / "evidence_trajectory_private.csv"),
        "discovery_trajectory_sha256": sha256_file(output / "discovery_trajectory_private.csv"),
    }
    atomic_json(summary, output / "seed_evaluation_summary.json")
    atomic_write_text(output / ".seed_evaluation_complete", "complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def aggregate(args: argparse.Namespace) -> None:
    experiment = Path(args.experiment_root)
    output = Path(args.aggregate_output)
    output.mkdir(parents=True, exist_ok=False)
    seeds = parse_seeds(args.seeds)
    summaries = []
    evidence_frames = []
    discovery_frames = []
    for initialization in INITIALIZATIONS:
        for seed in seeds:
            root = experiment / initialization / f"seed_{seed}"
            if not (root / ".seed_evaluation_complete").is_file():
                raise FileNotFoundError(f"Seed evaluation is incomplete: {root}")
            summary = json.loads((root / "seed_evaluation_summary.json").read_text())
            summaries.append(summary)
            evidence_frames.append(pd.read_csv(root / "evidence_trajectory_private.csv"))
            discovery_frames.append(pd.read_csv(root / "discovery_trajectory_private.csv"))
    summary_frame = pd.DataFrame(summaries)
    evidence = pd.concat(evidence_frames, ignore_index=True)
    discovery = pd.concat(discovery_frames, ignore_index=True)
    atomic_write_csv(summary_frame, output / "seed_summaries.csv")
    atomic_write_csv(evidence, output / "all_evidence_trajectories.csv")
    atomic_write_csv(discovery, output / "all_discovery_trajectories.csv")

    contrasts = []
    within_metrics = [
        "oof_clean_macro_auroc_change",
        "probe_error_auprc_change",
        "hard_probe_recall_change",
        "dynamic_minus_frozen_audc",
    ]
    for initialization in INITIALIZATIONS:
        branch = summary_frame[summary_frame["initialization"] == initialization].set_index("seed")
        for metric in within_metrics:
            values = branch.loc[seeds, metric].to_numpy(dtype=float)
            contrasts.append(
                {
                    "contrast": f"{initialization}:{metric}",
                    "family": "within_branch_trajectory",
                    "paired_seeds": len(values),
                    "mean_difference": float(values.mean()),
                    "positive_seeds": int((values > 0).sum()),
                    "negative_seeds": int((values < 0).sum()),
                    "exact_sign_flip_p": exact_sign_flip_p(values),
                }
            )
    scratch = summary_frame[summary_frame["initialization"] == "scratch"].set_index("seed")
    pretrained = summary_frame[summary_frame["initialization"] == "xrv_pretrained"].set_index("seed")
    for metric in [
        "oof_clean_macro_auroc_final", "probe_error_auprc_final",
        "dynamic_discovery_audc", "final_quality",
    ]:
        values = pretrained.loc[seeds, metric].to_numpy(dtype=float) - scratch.loc[seeds, metric].to_numpy(dtype=float)
        contrasts.append(
            {
                "contrast": f"xrv_pretrained_minus_scratch:{metric}",
                "family": "between_initializations",
                "paired_seeds": len(values),
                "mean_difference": float(values.mean()),
                "positive_seeds": int((values > 0).sum()),
                "negative_seeds": int((values < 0).sum()),
                "exact_sign_flip_p": exact_sign_flip_p(values),
            }
        )
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame["holm_p_within_family"] = np.nan
    for _, indices in contrast_frame.groupby("family").groups.items():
        p_values = contrast_frame.loc[indices, "exact_sign_flip_p"].tolist()
        contrast_frame.loc[indices, "holm_p_within_family"] = holm_adjust(p_values)
    atomic_write_csv(contrast_frame, output / "paired_tests.csv")

    mean_evidence = evidence.groupby(["initialization", "loop"], as_index=False).agg(
        oof_clean_macro_auroc=("oof_clean_macro_auroc", "mean"),
        probe_error_auprc=("probe_error_auprc", "mean"),
        true_quality=("true_quality", "mean"),
        hard_probe_recall_at_budget=("hard_probe_recall_at_budget", "mean"),
    )
    mean_discovery = discovery.groupby(["initialization", "method", "loop"], as_index=False).agg(
        action_error_recall=("action_error_recall", "mean")
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {"scratch": "#2f6db0", "xrv_pretrained": "#c43c39"}
    for initialization in INITIALIZATIONS:
        branch = mean_evidence[mean_evidence["initialization"] == initialization]
        axes[0, 0].plot(branch["loop"], branch["oof_clean_macro_auroc"], marker="o", label=initialization, color=colors[initialization])
        axes[0, 1].plot(branch["loop"], branch["probe_error_auprc"], marker="o", label=initialization, color=colors[initialization])
        axes[1, 0].plot(branch["loop"], branch["true_quality"], marker="o", label=initialization, color=colors[initialization])
    for method, style in (("dynamic_cl", "-"), ("frozen_loop0_cl", "--"), ("random_review_expected", ":")):
        branch = mean_discovery[
            (mean_discovery["initialization"] == "xrv_pretrained")
            & (mean_discovery["method"] == method)
        ]
        axes[1, 1].plot(branch["loop"], branch["action_error_recall"], linestyle=style, marker="o", label=method)
    axes[0, 0].set_title("Clean-reference OOF macro AUROC")
    axes[0, 1].set_title("Fixed-probe error AUPRC")
    axes[1, 0].set_title("True dataset quality")
    axes[1, 1].set_title("XRV-pretrained discovery efficiency")
    for axis in axes.ravel():
        axis.set_xlabel("Cleaning loop")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "iterative_evidence_summary.png", dpi=200)
    plt.close(figure)
    aggregate_summary = {
        "protocol": PROTOCOL_NAME,
        "seeds": seeds,
        "branches": list(INITIALIZATIONS),
        "seed_runs": int(len(summary_frame)),
        "loops": int(evidence["loop"].max()),
        "paired_tests": int(len(contrast_frame)),
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(aggregate_summary, output / "aggregate_summary.json")
    atomic_write_text(output / ".aggregate_complete", "complete\n")


def common_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", type=Path, required=True)
    common.add_argument("--source-prepared", type=Path, required=True)
    common.add_argument("--private-reference", type=Path, required=True)
    common.add_argument("--seed", type=int, required=True)
    common.add_argument("--initialization", choices=INITIALIZATIONS, required=True)
    common.add_argument("--loops", type=int, default=8)
    common.add_argument("--review-budget", type=int, default=REVIEW_BUDGET)
    common.add_argument("--probe-review-budget", type=int, default=PROBE_REVIEW_BUDGET)
    return common


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = common_parser()
    commands.add_parser("initialize", parents=[common]).set_defaults(function=initialize)
    selection = commands.add_parser("select", parents=[common])
    selection.add_argument("--loop-id", type=int, required=True)
    selection.set_defaults(function=select)
    oracle = commands.add_parser("oracle-update", parents=[common])
    oracle.add_argument("--loop-id", type=int, required=True)
    oracle.set_defaults(function=oracle_update)
    finalizer = commands.add_parser("finalize-loop", parents=[common])
    finalizer.add_argument("--loop-id", type=int, required=True)
    finalizer.set_defaults(function=finalize_loop)
    commands.add_parser("evaluate", parents=[common]).set_defaults(function=evaluate)
    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument("--experiment-root", type=Path, required=True)
    aggregate_parser.add_argument("--aggregate-output", type=Path, required=True)
    aggregate_parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    aggregate_parser.set_defaults(function=aggregate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "loop_id") and not 1 <= args.loop_id <= args.loops:
        raise ValueError("--loop-id must be within 1..--loops")
    if hasattr(args, "review_budget") and args.review_budget <= 0:
        raise ValueError("--review-budget must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
