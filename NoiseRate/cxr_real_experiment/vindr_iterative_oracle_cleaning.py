#!/usr/bin/env python3
"""Iterative VinDr CL selection with selected-only oracle corrections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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


PROTOCOL_NAME = "vindr_iterative_oracle_cleaning_v1"
DEFAULT_SEEDS = [13, 42, 97, 123, 211, 307]
ENTRIES = 18_000
INITIAL_ERRORS = 3_600
INITIAL_QUALITY = 0.80


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2))


def entry_key(image_id: pd.Series, label_name: pd.Series) -> pd.Series:
    return image_id.astype(str) + "::" + label_name.astype(str)


def cohort_entries(cohort: pd.DataFrame) -> pd.DataFrame:
    require_columns(cohort, ["image_id", "image_path", "fold_id"] + LABELS, "blind cohort")
    validate_blind_columns(cohort)
    records = []
    for label_index, label in enumerate(LABELS):
        part = cohort[["image_id", "fold_id", label]].copy()
        part.columns = ["image_id", "fold_id", "current_label"]
        part["label_index"] = label_index
        part["label_name"] = label
        records.append(part)
    frame = pd.concat(records, ignore_index=True)
    frame["entry_key"] = entry_key(frame["image_id"], frame["label_name"])
    if len(frame) != ENTRIES or frame["entry_key"].duplicated().any():
        raise ValueError("Blind cohort does not contain 18,000 unique entries")
    return frame


def validate_private_reference(private: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        private,
        [
            "image_id", "label_name", "clean_label", "noisy_label",
            "injected_error", "flip_direction",
        ],
        "private reference",
    )
    private = private.copy()
    private["entry_key"] = entry_key(private["image_id"], private["label_name"])
    if len(private) != ENTRIES or private["entry_key"].duplicated().any():
        raise ValueError("Private reference does not contain 18,000 unique entries")
    for column in ["clean_label", "noisy_label", "injected_error"]:
        if not private[column].isin([0, 1, False, True]).all():
            raise ValueError(f"Private reference has non-binary values in {column}")
    return private


def validate_evidence(cohort: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    required = [
        "image_id", "fold_id", "label_index", "label_name", "noisy_label",
        "oof_probability", "cl_issue", "self_confidence_suspicion", "cl_first_score",
    ]
    require_columns(evidence, required, "entry evidence")
    validate_blind_columns(evidence.drop(columns=["cl_issue"], errors="ignore"))
    evidence = evidence.copy()
    evidence["entry_key"] = entry_key(evidence["image_id"], evidence["label_name"])
    if len(evidence) != ENTRIES or evidence["entry_key"].duplicated().any():
        raise ValueError("Evidence does not contain 18,000 unique entries")
    current = cohort_entries(cohort)[["entry_key", "current_label"]]
    merged = evidence.merge(current, on="entry_key", validate="one_to_one")
    if not merged["noisy_label"].astype(int).eq(merged["current_label"].astype(int)).all():
        raise ValueError("Evidence labels differ from the current blind cohort")
    if not np.isfinite(merged["oof_probability"]).all():
        raise ValueError("Evidence contains non-finite probabilities")
    return merged


def select_candidates(
    evidence: pd.DataFrame,
    reviewed_keys: set[str],
    top_fraction: float,
) -> tuple[pd.DataFrame, int]:
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    candidate = evidence[
        evidence["cl_issue"].astype(bool)
        & ~evidence["entry_key"].astype(str).isin(reviewed_keys)
    ].copy()
    if candidate.empty:
        return candidate, 0
    selected_count = max(1, int(np.ceil(top_fraction * len(candidate))))
    selected = candidate.sort_values(
        ["cl_first_score", "self_confidence_suspicion", "entry_key"],
        ascending=[False, False, True],
        kind="stable",
    ).head(selected_count)
    if selected["entry_key"].duplicated().any() or selected["entry_key"].isin(reviewed_keys).any():
        raise RuntimeError("Selection contains duplicate or previously reviewed entries")
    if not selected["cl_issue"].astype(bool).all():
        raise RuntimeError("Selection escaped the current CL hard-issue pool")
    return selected, len(candidate)


def apply_oracle_labels(
    cohort: pd.DataFrame,
    selected_private: pd.DataFrame,
) -> pd.DataFrame:
    updated = cohort.copy()
    image_to_row = {str(value): index for index, value in enumerate(updated["image_id"].astype(str))}
    for row in selected_private.itertuples(index=False):
        row_index = image_to_row[str(row.image_id)]
        current = int(updated.at[row_index, str(row.label_name)])
        if current != int(row.current_label):
            raise ValueError(f"Selected label changed before oracle update: {row.entry_key}")
        updated.at[row_index, str(row.label_name)] = int(row.clean_label)
    return updated


def true_state(cohort: pd.DataFrame, private: pd.DataFrame) -> dict[str, Any]:
    current = cohort_entries(cohort)[["entry_key", "current_label"]]
    truth = validate_private_reference(private)
    merged = current.merge(
        truth[["entry_key", "clean_label"]], on="entry_key", validate="one_to_one"
    )
    if len(merged) != ENTRIES:
        raise ValueError("Blind cohort and private reference entry keys differ")
    wrong = merged["current_label"].astype(int).ne(merged["clean_label"].astype(int))
    wrong_frame = merged[wrong]
    fp = int(((wrong_frame["current_label"] == 1) & (wrong_frame["clean_label"] == 0)).sum())
    fn = int(((wrong_frame["current_label"] == 0) & (wrong_frame["clean_label"] == 1)).sum())
    return {
        "remaining_errors": int(wrong.sum()),
        "true_quality": float(1.0 - wrong.mean()),
        "remaining_0_to_1": fp,
        "remaining_1_to_0": fn,
    }


def state_dir(output: Path) -> Path:
    return output / "state"


def loop_dir(output: Path, loop_id: int) -> Path:
    return output / f"loop_{loop_id:02d}"


def history_paths(output: Path) -> dict[str, Path]:
    state = state_dir(output)
    return {
        "blind": state / "review_history_blind.csv",
        "private": state / "review_history_private.csv",
        "marker": state / ".initialized",
    }


def initialize(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    source_prepared = Path(args.source_prepared)
    source_blind_run = Path(args.source_blind_run)
    cohort_path = source_prepared / "blind_noisy_cohort.csv"
    private_path = source_prepared / "private_reference.csv"
    evidence_path = source_blind_run / "entry_evidence.csv"
    for path in [cohort_path, private_path, evidence_path, source_blind_run / ".blind_run_complete"]:
        if not path.exists():
            raise FileNotFoundError(path)
    cohort = pd.read_csv(cohort_path)
    evidence = validate_evidence(cohort, pd.read_csv(evidence_path))
    private = pd.read_csv(private_path)
    state = true_state(cohort, private)
    if state["remaining_errors"] != INITIAL_ERRORS or not np.isclose(state["true_quality"], INITIAL_QUALITY):
        raise ValueError(f"Source is not the locked 20% corruption anchor: {state}")
    if int(evidence["cl_issue"].sum()) <= 0:
        raise ValueError("Source evidence has no CL issues")

    paths = history_paths(output)
    state_dir(output).mkdir(parents=True, exist_ok=False)
    blind_history = pd.DataFrame(columns=["entry_key", "first_review_loop"])
    private_history = pd.DataFrame(
        columns=[
            "entry_key", "image_id", "label_name", "first_review_loop", "current_label",
            "clean_label", "true_issue", "flip_direction", "selected_cl_first_score",
        ]
    )
    atomic_write_csv(blind_history, paths["blind"])
    atomic_write_csv(private_history, paths["private"])
    manifest = {
        "protocol": PROTOCOL_NAME,
        "seed": int(args.seed),
        "loops": int(args.loops),
        "top_fraction": float(args.top_fraction),
        "samples": 3_000,
        "entries": ENTRIES,
        "initial_errors": INITIAL_ERRORS,
        "initial_quality": INITIAL_QUALITY,
        "initial_cl_issues": int(evidence["cl_issue"].sum()),
        "initial_raw_dqs": float(1.0 - evidence["cl_issue"].mean()),
        "source_prepared": str(source_prepared),
        "source_blind_run": str(source_blind_run),
        "source_cohort_sha256": sha256_file(cohort_path),
        "source_private_sha256": sha256_file(private_path),
        "source_evidence_sha256": sha256_file(evidence_path),
        "private_reference_used_for_initialization_check_only": True,
    }
    atomic_json(manifest, state_dir(output) / "initialization_manifest.json")
    paths["marker"].write_text("complete\n")
    print(json.dumps(manifest, indent=2), flush=True)


def current_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.loop_id == 1:
        return (
            Path(args.source_prepared) / "blind_noisy_cohort.csv",
            Path(args.source_blind_run) / "entry_evidence.csv",
        )
    previous = loop_dir(Path(args.output_dir), args.loop_id - 1)
    if not (previous / ".loop_complete").is_file():
        raise FileNotFoundError(f"Previous loop is incomplete: {previous}")
    return previous / "cohort_after_action_blind.csv", previous / "oof_after_action" / "entry_evidence.csv"


def select(args: argparse.Namespace) -> None:
    if args.loop_id is None or not 1 <= args.loop_id <= args.loops:
        raise ValueError("loop_id is outside the configured range")
    output = Path(args.output_dir)
    paths = history_paths(output)
    if not paths["marker"].is_file():
        raise FileNotFoundError("Initialization marker is missing")
    directory = loop_dir(output, args.loop_id)
    directory.mkdir(parents=True, exist_ok=False)
    cohort_path, evidence_path = current_inputs(args)
    cohort = pd.read_csv(cohort_path)
    evidence = validate_evidence(cohort, pd.read_csv(evidence_path))
    reviewed = pd.read_csv(paths["blind"])
    reviewed_keys = set(reviewed["entry_key"].astype(str))
    selected, issue_pool = select_candidates(evidence, reviewed_keys, args.top_fraction)
    blind_columns = [
        "entry_key", "image_id", "fold_id", "label_index", "label_name", "noisy_label",
        "oof_probability", "cl_issue", "self_confidence_suspicion", "cl_first_score",
    ]
    selected_path = directory / "selected_entries_blind.csv"
    atomic_write_csv(selected[blind_columns], selected_path)
    manifest = {
        "protocol": PROTOCOL_NAME,
        "seed": int(args.seed),
        "loop": int(args.loop_id),
        "reviewed_before_loop": len(reviewed_keys),
        "unreviewed_cl_issue_pool": int(issue_pool),
        "selected_entries": int(len(selected)),
        "top_fraction": float(args.top_fraction),
        "raw_dqs_before_action": float(1.0 - evidence["cl_issue"].mean()),
        "current_cohort_sha256": sha256_file(cohort_path),
        "current_evidence_sha256": sha256_file(evidence_path),
        "selected_entries_sha256": sha256_file(selected_path),
        "private_reference_used": False,
    }
    atomic_json(manifest, directory / "selection_manifest_blind.json")
    (directory / ".selection_complete").write_text("complete\n")
    print(json.dumps(manifest, indent=2), flush=True)


def oracle_update(args: argparse.Namespace) -> None:
    if args.loop_id is None or not 1 <= args.loop_id <= args.loops:
        raise ValueError("loop_id is outside the configured range")
    output = Path(args.output_dir)
    directory = loop_dir(output, args.loop_id)
    if not (directory / ".selection_complete").is_file():
        raise FileNotFoundError("Blind selection marker is missing")
    manifest = json.loads((directory / "selection_manifest_blind.json").read_text())
    selected_path = directory / "selected_entries_blind.csv"
    if sha256_file(selected_path) != manifest["selected_entries_sha256"]:
        raise ValueError("Frozen selection hash changed")
    selected = pd.read_csv(selected_path)
    cohort_path, _ = current_inputs(args)
    cohort = pd.read_csv(cohort_path)
    private_path = Path(args.private_reference)
    private = validate_private_reference(pd.read_csv(private_path))
    selected_private = selected.merge(
        private[["entry_key", "clean_label", "flip_direction"]],
        on="entry_key",
        validate="one_to_one",
    ).rename(columns={"noisy_label": "current_label"})
    if len(selected_private) != len(selected):
        raise ValueError("One or more blind selections are missing from the private reference")
    selected_private["true_issue"] = selected_private["current_label"].astype(int).ne(
        selected_private["clean_label"].astype(int)
    ).astype(int)
    selected_private["review_loop"] = int(args.loop_id)
    selected_private_path = directory / "selected_entries_private.csv"
    atomic_write_csv(selected_private, selected_private_path)

    before = true_state(cohort, private)
    updated = apply_oracle_labels(cohort, selected_private)
    after = true_state(updated, private)
    corrected = int(selected_private["true_issue"].sum())
    if before["remaining_errors"] - after["remaining_errors"] != corrected:
        raise RuntimeError("Oracle correction did not remove exactly the selected true issues")
    updated_path = directory / "cohort_after_action_blind.csv"
    atomic_write_csv(updated, updated_path)

    paths = history_paths(output)
    blind_history = pd.read_csv(paths["blind"])
    private_history = pd.read_csv(paths["private"])
    additions_blind = selected_private[["entry_key"]].copy()
    additions_blind["first_review_loop"] = int(args.loop_id)
    new_blind_history = pd.concat([blind_history, additions_blind], ignore_index=True)
    if new_blind_history["entry_key"].duplicated().any():
        raise ValueError("Cumulative blind review history contains duplicates")
    additions_private = pd.DataFrame(
        {
            "entry_key": selected_private["entry_key"],
            "image_id": selected_private["image_id"],
            "label_name": selected_private["label_name"],
            "first_review_loop": int(args.loop_id),
            "current_label": selected_private["current_label"].astype(int),
            "clean_label": selected_private["clean_label"].astype(int),
            "true_issue": selected_private["true_issue"].astype(int),
            "flip_direction": selected_private["flip_direction"],
            "selected_cl_first_score": selected_private["cl_first_score"],
        }
    )
    new_private_history = pd.concat([private_history, additions_private], ignore_index=True)
    if new_private_history["entry_key"].duplicated().any():
        raise ValueError("Cumulative private review history contains duplicates")
    atomic_write_csv(new_blind_history, paths["blind"])
    atomic_write_csv(new_private_history, paths["private"])

    summary = {
        "protocol": PROTOCOL_NAME,
        "seed": int(args.seed),
        "loop": int(args.loop_id),
        "selected_entries": int(len(selected_private)),
        "selected_true_issues": corrected,
        "selected_precision": float(corrected / len(selected_private)) if len(selected_private) else float("nan"),
        "selected_0_to_1_issues": int(
            ((selected_private["true_issue"] == 1) & (selected_private["flip_direction"] == "0_to_1")).sum()
        ),
        "selected_1_to_0_issues": int(
            ((selected_private["true_issue"] == 1) & (selected_private["flip_direction"] == "1_to_0")).sum()
        ),
        "cumulative_reviews": int(len(new_blind_history)),
        "cumulative_true_issues": int(new_private_history["true_issue"].astype(int).sum()),
        "true_quality_before_action": before["true_quality"],
        "true_quality_after_action": after["true_quality"],
        "remaining_errors_before_action": before["remaining_errors"],
        "remaining_errors_after_action": after["remaining_errors"],
        "remaining_0_to_1_after_action": after["remaining_0_to_1"],
        "remaining_1_to_0_after_action": after["remaining_1_to_0"],
        "raw_dqs_before_action": float(manifest["raw_dqs_before_action"]),
        "selected_entries_blind_sha256": sha256_file(selected_path),
        "selected_entries_private_sha256": sha256_file(selected_private_path),
        "private_reference_sha256": sha256_file(private_path),
        "cohort_after_action_sha256": sha256_file(updated_path),
    }
    atomic_json(summary, directory / "oracle_summary_private.json")
    (directory / ".oracle_update_complete").write_text("complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def finalize_loop(args: argparse.Namespace) -> None:
    directory = loop_dir(Path(args.output_dir), args.loop_id)
    if not (directory / ".oracle_update_complete").is_file():
        raise FileNotFoundError("Oracle update marker is missing")
    oof_dir = directory / "oof_after_action"
    if not (oof_dir / ".blind_run_complete").is_file():
        raise FileNotFoundError("Post-action OOF marker is missing")
    cohort_path = directory / "cohort_after_action_blind.csv"
    evidence_path = oof_dir / "entry_evidence.csv"
    evidence = validate_evidence(pd.read_csv(cohort_path), pd.read_csv(evidence_path))
    oof_summary = json.loads((oof_dir / "blind_run_summary.json").read_text())
    summary = json.loads((directory / "oracle_summary_private.json").read_text())
    if oof_summary["seed"] != int(args.seed) or oof_summary["entries"] != ENTRIES:
        raise ValueError("Post-action OOF summary differs from the loop configuration")
    if oof_summary["blind_cohort_sha256"] != sha256_file(cohort_path):
        raise ValueError("Post-action OOF did not use the frozen updated cohort")
    summary["raw_dqs_after_action"] = float(1.0 - evidence["cl_issue"].mean())
    summary["cl_issue_entries_after_action"] = int(evidence["cl_issue"].sum())
    summary["post_action_evidence_sha256"] = sha256_file(evidence_path)
    atomic_json(summary, directory / "loop_metrics_private.json")
    (directory / ".loop_complete").write_text("complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def audc(frame: pd.DataFrame, value: str) -> float:
    ordered = frame.sort_values("review_fraction")
    x = ordered["review_fraction"].to_numpy(dtype=float)
    y = ordered[value].to_numpy(dtype=float)
    return float(np.trapezoid(y, x))


def comparator_trajectory(
    ordered_truth: np.ndarray,
    budgets: list[int],
    method: str,
) -> pd.DataFrame:
    records = []
    cumulative = np.cumsum(ordered_truth.astype(int))
    for loop_id, budget in enumerate(budgets, start=1):
        found = int(cumulative[budget - 1]) if budget else 0
        records.append(
            {
                "method": method,
                "loop": loop_id,
                "cumulative_reviews": budget,
                "review_fraction": budget / ENTRIES,
                "cumulative_true_issues": found,
                "cumulative_recall": found / INITIAL_ERRORS,
                "cumulative_precision": found / budget if budget else float("nan"),
                "true_quality": 1.0 - (INITIAL_ERRORS - found) / ENTRIES,
            }
        )
    return pd.DataFrame(records)


def evaluate(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    initialization = json.loads((state_dir(output) / "initialization_manifest.json").read_text())
    private_path = Path(args.private_reference)
    private = validate_private_reference(pd.read_csv(private_path))
    source_evidence = pd.read_csv(Path(args.source_blind_run) / "entry_evidence.csv")
    source_evidence["entry_key"] = entry_key(source_evidence["image_id"], source_evidence["label_name"])
    source_truth = source_evidence[["entry_key", "cl_issue", "cl_first_score", "self_confidence_suspicion"]].merge(
        private[["entry_key", "injected_error"]], on="entry_key", validate="one_to_one"
    )
    if int(source_truth["injected_error"].sum()) != INITIAL_ERRORS:
        raise ValueError("Private source truth no longer has 3,600 injected errors")

    dynamic_records = [
        {
            "method": "dynamic_cl",
            "loop": 0,
            "cumulative_reviews": 0,
            "review_fraction": 0.0,
            "new_true_issues": 0,
            "cumulative_true_issues": 0,
            "cumulative_recall": 0.0,
            "cumulative_precision": float("nan"),
            "true_quality": INITIAL_QUALITY,
            "raw_dqs": float(initialization["initial_raw_dqs"]),
            "remaining_errors": INITIAL_ERRORS,
            "remaining_0_to_1": int((private["flip_direction"] == "0_to_1").sum()),
            "remaining_1_to_0": int((private["flip_direction"] == "1_to_0").sum()),
        }
    ]
    budgets = []
    for loop_id in range(1, args.loops + 1):
        directory = loop_dir(output, loop_id)
        if not (directory / ".loop_complete").is_file():
            raise FileNotFoundError(f"Loop {loop_id} is incomplete")
        metrics = json.loads((directory / "loop_metrics_private.json").read_text())
        budgets.append(int(metrics["cumulative_reviews"]))
        dynamic_records.append(
            {
                "method": "dynamic_cl",
                "loop": loop_id,
                "cumulative_reviews": int(metrics["cumulative_reviews"]),
                "review_fraction": int(metrics["cumulative_reviews"]) / ENTRIES,
                "new_true_issues": int(metrics["selected_true_issues"]),
                "cumulative_true_issues": int(metrics["cumulative_true_issues"]),
                "cumulative_recall": int(metrics["cumulative_true_issues"]) / INITIAL_ERRORS,
                "cumulative_precision": int(metrics["cumulative_true_issues"]) / int(metrics["cumulative_reviews"]),
                "true_quality": float(metrics["true_quality_after_action"]),
                "raw_dqs": float(metrics["raw_dqs_after_action"]),
                "remaining_errors": int(metrics["remaining_errors_after_action"]),
                "remaining_0_to_1": int(metrics["remaining_0_to_1_after_action"]),
                "remaining_1_to_0": int(metrics["remaining_1_to_0_after_action"]),
            }
        )
    dynamic = pd.DataFrame(dynamic_records)
    if budgets != sorted(budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("Cumulative review budgets are not strictly increasing")

    frozen_order = source_truth.sort_values(
        ["cl_issue", "cl_first_score", "self_confidence_suspicion", "entry_key"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    frozen = comparator_trajectory(
        frozen_order["injected_error"].to_numpy(dtype=int), budgets, "frozen_loop0_cl"
    )
    frozen = pd.concat(
        [
            pd.DataFrame([{
                "method": "frozen_loop0_cl", "loop": 0, "cumulative_reviews": 0,
                "review_fraction": 0.0, "cumulative_true_issues": 0,
                "cumulative_recall": 0.0, "cumulative_precision": float("nan"),
                "true_quality": INITIAL_QUALITY,
            }]),
            frozen,
        ], ignore_index=True,
    )

    rng = np.random.default_rng(int(args.random_seed) + int(args.seed))
    truth = source_truth["injected_error"].to_numpy(dtype=int)
    random_recall = np.empty((args.random_replicates, len(budgets)), dtype=float)
    random_precision = np.empty_like(random_recall)
    for replicate in range(args.random_replicates):
        cumulative = np.cumsum(rng.permutation(truth))
        found = cumulative[np.asarray(budgets) - 1]
        random_recall[replicate] = found / INITIAL_ERRORS
        random_precision[replicate] = found / np.asarray(budgets)
    random_records = [{
        "method": "random_review", "loop": 0, "cumulative_reviews": 0,
        "review_fraction": 0.0, "cumulative_true_issues": 0.0,
        "cumulative_recall": 0.0, "cumulative_precision": float("nan"),
        "true_quality": INITIAL_QUALITY, "recall_ci_low": 0.0, "recall_ci_high": 0.0,
    }]
    for index, budget in enumerate(budgets):
        mean_recall = float(random_recall[:, index].mean())
        random_records.append(
            {
                "method": "random_review",
                "loop": index + 1,
                "cumulative_reviews": budget,
                "review_fraction": budget / ENTRIES,
                "cumulative_true_issues": mean_recall * INITIAL_ERRORS,
                "cumulative_recall": mean_recall,
                "cumulative_precision": float(random_precision[:, index].mean()),
                "true_quality": INITIAL_QUALITY + mean_recall * INITIAL_ERRORS / ENTRIES,
                "recall_ci_low": float(np.quantile(random_recall[:, index], 0.025)),
                "recall_ci_high": float(np.quantile(random_recall[:, index], 0.975)),
            }
        )
    random_frame = pd.DataFrame(random_records)
    trajectory = pd.concat([dynamic, frozen, random_frame], ignore_index=True, sort=False)
    atomic_write_csv(trajectory, output / "iterative_trajectory.csv")

    dqs_frame = dynamic[["true_quality", "raw_dqs"]]
    dqs_rho = float(spearmanr(dqs_frame["true_quality"], dqs_frame["raw_dqs"]).statistic)
    dqs_delta = np.diff(dqs_frame["raw_dqs"].to_numpy(dtype=float))
    quality_delta = np.diff(dqs_frame["true_quality"].to_numpy(dtype=float))
    direction_agreement = int(np.sum(np.sign(dqs_delta) == np.sign(quality_delta)))
    summary = {
        "protocol": PROTOCOL_NAME,
        "seed": int(args.seed),
        "loops": int(args.loops),
        "initial_errors": INITIAL_ERRORS,
        "initial_quality": INITIAL_QUALITY,
        "final_dynamic_reviews": int(dynamic.iloc[-1]["cumulative_reviews"]),
        "final_dynamic_true_issues": int(dynamic.iloc[-1]["cumulative_true_issues"]),
        "final_dynamic_recall": float(dynamic.iloc[-1]["cumulative_recall"]),
        "final_dynamic_quality": float(dynamic.iloc[-1]["true_quality"]),
        "final_dynamic_dqs": float(dynamic.iloc[-1]["raw_dqs"]),
        "final_frozen_recall": float(frozen.iloc[-1]["cumulative_recall"]),
        "final_random_recall": float(random_frame.iloc[-1]["cumulative_recall"]),
        "dynamic_recall_audc": audc(dynamic, "cumulative_recall"),
        "frozen_recall_audc": audc(frozen, "cumulative_recall"),
        "random_recall_audc": audc(random_frame, "cumulative_recall"),
        "dqs_true_quality_spearman": dqs_rho,
        "dqs_quality_delta_direction_agreement": direction_agreement,
        "dqs_quality_delta_transitions": int(args.loops),
        "private_reference_sha256": sha256_file(private_path),
        "trajectory_sha256": sha256_file(output / "iterative_trajectory.csv"),
    }
    atomic_json(summary, output / "seed_evaluation_summary.json")
    (output / ".seed_evaluation_complete").write_text("complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def aggregate(args: argparse.Namespace) -> None:
    root = Path(args.experiment_root)
    output = Path(args.aggregate_output)
    output.mkdir(parents=True, exist_ok=False)
    seeds = parse_seeds(args.seeds)
    trajectories = []
    summaries = []
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        if not (seed_root / ".seed_evaluation_complete").is_file():
            raise FileNotFoundError(f"Seed evaluation is incomplete: {seed}")
        frame = pd.read_csv(seed_root / "iterative_trajectory.csv")
        frame.insert(0, "seed", seed)
        trajectories.append(frame)
        summaries.append(json.loads((seed_root / "seed_evaluation_summary.json").read_text()))
    trajectory = pd.concat(trajectories, ignore_index=True)
    seed_summary = pd.DataFrame(summaries)
    atomic_write_csv(trajectory, output / "all_seed_trajectories.csv")
    atomic_write_csv(seed_summary, output / "seed_summaries.csv")

    dynamic_minus_frozen = seed_summary["dynamic_recall_audc"] - seed_summary["frozen_recall_audc"]
    dynamic_minus_random = seed_summary["dynamic_recall_audc"] - seed_summary["random_recall_audc"]
    raw_p_values = [
        exact_sign_flip_p(dynamic_minus_frozen.to_numpy()),
        exact_sign_flip_p(dynamic_minus_random.to_numpy()),
    ]
    adjusted_p_values = holm_adjust(raw_p_values)
    paired_tests = pd.DataFrame(
        [
            {
                "contrast": "dynamic_cl_minus_frozen_loop0_cl_recall_audc",
                "seed_count": len(seeds),
                "positive_seed_count": int((dynamic_minus_frozen > 0).sum()),
                "mean_difference": float(dynamic_minus_frozen.mean()),
                "exact_p_value": raw_p_values[0],
                "holm_adjusted_p_value": adjusted_p_values[0],
            },
            {
                "contrast": "dynamic_cl_minus_random_review_recall_audc",
                "seed_count": len(seeds),
                "positive_seed_count": int((dynamic_minus_random > 0).sum()),
                "mean_difference": float(dynamic_minus_random.mean()),
                "exact_p_value": raw_p_values[1],
                "holm_adjusted_p_value": adjusted_p_values[1],
            },
        ]
    )
    atomic_write_csv(paired_tests, output / "paired_tests.csv")
    aggregate_summary = {
        "protocol": PROTOCOL_NAME,
        "seeds": seeds,
        "loops": int(seed_summary["loops"].iloc[0]),
        "all_dynamic_audc_above_frozen": bool((dynamic_minus_frozen > 0).all()),
        "all_dynamic_audc_above_random": bool((dynamic_minus_random > 0).all()),
        "dynamic_minus_frozen_audc_mean": float(dynamic_minus_frozen.mean()),
        "dynamic_minus_frozen_exact_p": raw_p_values[0],
        "dynamic_minus_frozen_holm_p": adjusted_p_values[0],
        "dynamic_minus_random_audc_mean": float(dynamic_minus_random.mean()),
        "dynamic_minus_random_exact_p": raw_p_values[1],
        "dynamic_minus_random_holm_p": adjusted_p_values[1],
        "final_dynamic_quality_mean": float(seed_summary["final_dynamic_quality"].mean()),
        "final_dynamic_quality_sd": float(seed_summary["final_dynamic_quality"].std()),
        "final_dynamic_recall_mean": float(seed_summary["final_dynamic_recall"].mean()),
        "final_dynamic_recall_sd": float(seed_summary["final_dynamic_recall"].std()),
        "dqs_min_spearman": float(seed_summary["dqs_true_quality_spearman"].min()),
        "dqs_all_transition_direction_agreement": bool(
            (seed_summary["dqs_quality_delta_direction_agreement"] == seed_summary["dqs_quality_delta_transitions"]).all()
        ),
    }
    atomic_json(aggregate_summary, output / "aggregate_summary.json")

    dynamic = trajectory[trajectory["method"] == "dynamic_cl"]
    summary = dynamic.groupby("loop").agg(
        true_quality_mean=("true_quality", "mean"),
        true_quality_sd=("true_quality", "std"),
        raw_dqs_mean=("raw_dqs", "mean"),
        raw_dqs_sd=("raw_dqs", "std"),
        recall_mean=("cumulative_recall", "mean"),
        recall_sd=("cumulative_recall", "std"),
        reviews_mean=("cumulative_reviews", "mean"),
    ).reset_index()
    atomic_write_csv(summary, output / "dynamic_loop_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes[0].errorbar(summary["loop"], summary["true_quality_mean"], yerr=summary["true_quality_sd"], marker="o", capsize=3, label="Known true quality")
    axes[0].errorbar(summary["loop"], summary["raw_dqs_mean"], yerr=summary["raw_dqs_sd"], marker="o", capsize=3, label="Raw entry DQS")
    axes[0].set(xlabel="Cleaning loop", ylabel="Score", title="True Quality and DQS")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    for method, frame in trajectory.groupby("method", sort=False):
        plot = frame.groupby("loop")["cumulative_recall"].agg(["mean", "std"]).reset_index()
        axes[1].errorbar(plot["loop"], plot["mean"], yerr=plot["std"], marker="o", capsize=3, label=method.replace("_", " "))
    axes[1].set(xlabel="Cleaning loop", ylabel="Recall of 3,600 injected errors", title="Error Discovery by Review Strategy")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "iterative_oracle_summary.png", dpi=180)
    plt.close(fig)
    (output / ".aggregate_complete").write_text("complete\n")
    print(json.dumps(aggregate_summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", type=Path, required=True)
    common.add_argument("--source-prepared", type=Path, required=True)
    common.add_argument("--source-blind-run", type=Path, required=True)
    common.add_argument("--seed", type=int, required=True)
    common.add_argument("--loops", type=int, default=8)
    common.add_argument("--top-fraction", type=float, default=0.20)

    subparsers.add_parser("initialize", parents=[common]).set_defaults(function=initialize)
    selection = subparsers.add_parser("select", parents=[common])
    selection.add_argument("--loop-id", type=int, required=True)
    selection.set_defaults(function=select)
    oracle = subparsers.add_parser("oracle-update", parents=[common])
    oracle.add_argument("--loop-id", type=int, required=True)
    oracle.add_argument("--private-reference", type=Path, required=True)
    oracle.set_defaults(function=oracle_update)
    finalizer = subparsers.add_parser("finalize-loop", parents=[common])
    finalizer.add_argument("--loop-id", type=int, required=True)
    finalizer.set_defaults(function=finalize_loop)
    evaluator = subparsers.add_parser("evaluate", parents=[common])
    evaluator.add_argument("--private-reference", type=Path, required=True)
    evaluator.add_argument("--random-replicates", type=int, default=10_000)
    evaluator.add_argument("--random-seed", type=int, default=20260805)
    evaluator.set_defaults(function=evaluate)

    aggregator = subparsers.add_parser("aggregate")
    aggregator.add_argument("--experiment-root", type=Path, required=True)
    aggregator.add_argument("--aggregate-output", type=Path, required=True)
    aggregator.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    aggregator.set_defaults(function=aggregate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
