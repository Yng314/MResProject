#!/usr/bin/env python3
"""One-shot full issue-pool cleaning followed by iterative new-issue cleaning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vindr_iterative_evidence_improvement import rank_entries
from vindr_known_gt_cl_benchmark import (
    atomic_write_csv,
    atomic_write_text,
    exact_sign_flip_p,
    parse_seeds,
    sha256_file,
)
from vindr_self_optimization_stress import (
    ACTION_ENTRIES,
    ACTION_IMAGES,
    SENTINEL_ENTRIES,
    SENTINEL_IMAGES,
    cohort_entries,
    macro_auroc,
    safe_auprc,
    safe_auroc,
    true_state,
    validate_evidence,
    validate_private,
)


PROTOCOL_NAME = "vindr_full_issue_pool_iteration_v1"
DEFAULT_SEEDS = (11003,)
LOOPS = 5


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2))


def state_dir(output: Path) -> Path:
    return output / "state"


def loop_dir(output: Path, loop_id: int) -> Path:
    return output / f"loop_{loop_id:02d}"


def evidence_dir(args: argparse.Namespace, loop_id: int) -> Path:
    if loop_id == 0:
        return Path(args.initial_evidence_dir)
    return loop_dir(Path(args.output_dir), loop_id) / "oof_after_action"


def cohort_path(args: argparse.Namespace, loop_id: int) -> Path:
    if loop_id == 0:
        return Path(args.source_prepared) / "blind_noisy_cohort.csv"
    return loop_dir(Path(args.output_dir), loop_id) / "cohort_after_action_blind.csv"


def initialize(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    state = state_dir(output)
    state.mkdir()
    prepared = Path(args.source_prepared)
    initial_evidence = Path(args.initial_evidence_dir)
    if not (prepared / ".prepare_complete").is_file():
        raise FileNotFoundError("Prepared condition is incomplete")
    if not (initial_evidence / ".blind_run_complete").is_file():
        raise FileNotFoundError("Locked Loop-0 evidence is incomplete")
    action = pd.read_csv(prepared / "blind_noisy_cohort.csv")
    sentinel = pd.read_csv(prepared / "sentinel_blind_cohort.csv")
    private = validate_private(pd.read_csv(args.private_reference), ACTION_ENTRIES)
    if len(action) != ACTION_IMAGES or len(sentinel) != SENTINEL_IMAGES:
        raise ValueError("Action/sentinel sizes do not match the locked source")
    evidence = validate_evidence(
        action, pd.read_csv(initial_evidence / "entry_evidence.csv"), ACTION_ENTRIES
    )
    if set(action["image_id"].astype(str)) & set(sentinel["image_id"].astype(str)):
        raise ValueError("Action and sentinel image sets overlap")
    initial = true_state(action, private)
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
        "seed": args.seed,
        "scenario_id": "hard_r30",
        "initialization": "scratch",
        "loops": args.loops,
        "loop_1_role": "one_shot_frozen_stop_after_full_initial_issue_pool",
        "loops_2_to_final_role": "iterative_full_new_issue_pools",
        "previously_reviewed_entries_excluded": True,
        "action_entries": ACTION_ENTRIES,
        "sentinel_entries": SENTINEL_ENTRIES,
        "initial_errors": initial["remaining_errors"],
        "initial_issue_pool": int(evidence["cl_issue"].sum()),
        "source_prepared_sha256": sha256_file(prepared / "blind_noisy_cohort.csv"),
        "initial_evidence_sha256": sha256_file(initial_evidence / "entry_evidence.csv"),
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(manifest, state / "initialization_manifest.json")
    atomic_write_text(state / ".initialized", "complete\n")
    print(json.dumps(manifest, indent=2), flush=True)


def select_all_new_issues(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    target = loop_dir(output, args.loop_id)
    target.mkdir(parents=True, exist_ok=False)
    previous_loop = args.loop_id - 1
    cohort = pd.read_csv(cohort_path(args, previous_loop))
    source_evidence = evidence_dir(args, previous_loop)
    if not (source_evidence / ".blind_run_complete").is_file():
        raise FileNotFoundError("Source evidence is incomplete")
    evidence = validate_evidence(
        cohort, pd.read_csv(source_evidence / "entry_evidence.csv"), ACTION_ENTRIES
    )
    history = pd.read_csv(state_dir(output) / "review_history_blind.csv")
    reviewed = set(history["entry_key"].astype(str)) if not history.empty else set()
    all_issues = evidence[evidence["cl_issue"].astype(bool)].copy()
    selected = rank_entries(all_issues[~all_issues["entry_key"].isin(reviewed)].copy())
    if selected.empty:
        raise RuntimeError("No new unreviewed issues remain; fixed-loop run should stop here")
    if selected["entry_key"].duplicated().any() or selected["entry_key"].isin(reviewed).any():
        raise RuntimeError("Selection contains duplicate or previously reviewed entries")
    columns = [
        "entry_key", "image_id", "fold_id", "label_index", "label_name",
        "current_label", "cl_issue", "cl_first_score",
        "self_confidence_suspicion", "oof_probability",
    ]
    selected_path = target / "selected_entries_blind.csv"
    atomic_write_csv(selected[columns], selected_path)
    manifest = {
        "protocol": PROTOCOL_NAME,
        "loop": args.loop_id,
        "all_current_issue_entries": len(all_issues),
        "previously_reviewed_current_issues": int(all_issues["entry_key"].isin(reviewed).sum()),
        "new_unreviewed_issue_entries": len(selected),
        "cumulative_previously_reviewed_entries": len(reviewed),
        "source_evidence_sha256": sha256_file(source_evidence / "entry_evidence.csv"),
        "selected_entries_sha256": sha256_file(selected_path),
    }
    atomic_json(manifest, target / "selection_manifest_blind.json")
    atomic_write_text(target / ".selection_complete", "complete\n")
    print(json.dumps(manifest, indent=2), flush=True)


def inspect_next_issue_pool(args: argparse.Namespace) -> None:
    """Inspect whether another full-pool loop has any unseen issues to review."""
    output = Path(args.output_dir)
    previous_loop = args.loop_id - 1
    cohort = pd.read_csv(cohort_path(args, previous_loop))
    source_evidence = evidence_dir(args, previous_loop)
    if not (source_evidence / ".blind_run_complete").is_file():
        raise FileNotFoundError("Source evidence is incomplete")
    evidence = validate_evidence(
        cohort, pd.read_csv(source_evidence / "entry_evidence.csv"), ACTION_ENTRIES
    )
    history = pd.read_csv(state_dir(output) / "review_history_blind.csv")
    reviewed = set(history["entry_key"].astype(str)) if not history.empty else set()
    all_issues = evidence[evidence["cl_issue"].astype(bool)].copy()
    unseen = all_issues[~all_issues["entry_key"].isin(reviewed)].copy()
    payload = {
        "protocol": PROTOCOL_NAME,
        "requested_loop": int(args.loop_id),
        "completed_loops": int(previous_loop),
        "all_current_issue_entries": int(len(all_issues)),
        "previously_reviewed_current_issues": int(
            all_issues["entry_key"].isin(reviewed).sum()
        ),
        "new_unreviewed_issue_entries": int(len(unseen)),
        "continue": bool(len(unseen)),
        "source_evidence_sha256": sha256_file(source_evidence / "entry_evidence.csv"),
    }
    atomic_json(
        payload,
        state_dir(output) / f"next_loop_{args.loop_id:02d}_inspection.json",
    )
    if unseen.empty:
        atomic_json(payload, state_dir(output) / "natural_stop.json")
    print(int(len(unseen)), flush=True)


def apply_oracle(cohort: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    updated = cohort.copy()
    row_index = {str(value): index for index, value in enumerate(updated["image_id"].astype(str))}
    for row in selected.itertuples(index=False):
        index = row_index[str(row.image_id)]
        if int(updated.at[index, str(row.label_name)]) != int(row.current_label):
            raise ValueError(f"Selected entry changed before correction: {row.entry_key}")
        updated.at[index, str(row.label_name)] = int(row.clean_label)
    return updated


def oracle_update(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    target = loop_dir(output, args.loop_id)
    if not (target / ".selection_complete").is_file():
        raise FileNotFoundError("Selection is incomplete")
    before = pd.read_csv(cohort_path(args, args.loop_id - 1))
    selected = pd.read_csv(target / "selected_entries_blind.csv")
    private = validate_private(pd.read_csv(args.private_reference), ACTION_ENTRIES)
    joined = selected.merge(
        private[["entry_key", "clean_label", "flip_direction"]],
        on="entry_key", how="left", validate="one_to_one",
    )
    if joined["clean_label"].isna().any():
        raise RuntimeError("Selected entries do not align with private reference")
    joined["true_issue"] = joined["current_label"].astype(int).ne(
        joined["clean_label"].astype(int)
    ).astype(int)
    after = apply_oracle(before, joined)
    after_path = target / "cohort_after_action_blind.csv"
    atomic_write_csv(after, after_path)
    atomic_write_csv(joined, target / "selected_entries_private.csv")

    blind_history_path = state_dir(output) / "review_history_blind.csv"
    private_history_path = state_dir(output) / "review_history_private.csv"
    blind_history = pd.read_csv(blind_history_path)
    private_history = pd.read_csv(private_history_path)
    blind_add = joined[["entry_key", "image_id", "label_name", "current_label"]].copy()
    blind_add.insert(0, "loop", args.loop_id)
    private_add = joined[[
        "entry_key", "image_id", "label_name", "current_label", "clean_label",
        "true_issue", "flip_direction",
    ]].copy()
    private_add.insert(0, "loop", args.loop_id)
    blind_history = pd.concat([blind_history, blind_add], ignore_index=True)
    private_history = pd.concat([private_history, private_add], ignore_index=True)
    if blind_history["entry_key"].duplicated().any():
        raise RuntimeError("An entry was reviewed more than once")
    atomic_write_csv(blind_history, blind_history_path)
    atomic_write_csv(private_history, private_history_path)
    before_state = true_state(before, private)
    after_state = true_state(after, private)
    summary = {
        "protocol": PROTOCOL_NAME,
        "loop": args.loop_id,
        "selected_new_issue_entries": len(joined),
        "selected_true_errors": int(joined["true_issue"].sum()),
        "selection_precision": float(joined["true_issue"].mean()),
        "quality_before": before_state["true_quality"],
        "quality_after": after_state["true_quality"],
        "remaining_errors_after": after_state["remaining_errors"],
        "cohort_after_sha256": sha256_file(after_path),
    }
    atomic_json(summary, target / "oracle_summary_private.json")
    atomic_write_text(target / ".oracle_update_complete", "complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def finalize_loop(args: argparse.Namespace) -> None:
    target = loop_dir(Path(args.output_dir), args.loop_id)
    evidence = evidence_dir(args, args.loop_id)
    if not (target / ".oracle_update_complete").is_file():
        raise FileNotFoundError("Oracle update is incomplete")
    if not (evidence / ".blind_run_complete").is_file():
        raise FileNotFoundError("Post-action OOF is incomplete")
    action = validate_evidence(
        pd.read_csv(target / "cohort_after_action_blind.csv"),
        pd.read_csv(evidence / "entry_evidence.csv"), ACTION_ENTRIES,
    )
    summary = json.loads((target / "oracle_summary_private.json").read_text())
    summary.update(
        {
            "post_action_issue_pool": int(action["cl_issue"].sum()),
            "raw_dqs_after_action": float(1.0 - action["cl_issue"].mean()),
            "post_action_evidence_sha256": sha256_file(evidence / "entry_evidence.csv"),
        }
    )
    atomic_json(summary, target / "loop_metrics_private.json")
    atomic_write_text(target / ".loop_complete", "complete\n")


def evaluate(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    for loop_id in range(args.loops + 1):
        if not (evidence_dir(args, loop_id) / ".blind_run_complete").is_file():
            raise FileNotFoundError(f"Evidence missing at Loop {loop_id}")
        if loop_id and not (loop_dir(output, loop_id) / ".loop_complete").is_file():
            raise FileNotFoundError(f"Loop marker missing at Loop {loop_id}")
    private = validate_private(pd.read_csv(args.private_reference), ACTION_ENTRIES)
    sentinel_private = validate_private(
        pd.read_csv(args.sentinel_private_reference), SENTINEL_ENTRIES
    )
    sentinel_cohort = pd.read_csv(Path(args.source_prepared) / "sentinel_blind_cohort.csv")
    history = pd.read_csv(state_dir(output) / "review_history_private.csv")
    rows = []
    for loop_id in range(args.loops + 1):
        action_cohort = pd.read_csv(cohort_path(args, loop_id))
        action = validate_evidence(
            action_cohort,
            pd.read_csv(evidence_dir(args, loop_id) / "entry_evidence.csv"),
            ACTION_ENTRIES,
        ).merge(
            private[["entry_key", "clean_label", "injected_error", "flip_direction"]],
            on="entry_key", validate="one_to_one",
        )
        sentinel = validate_evidence(
            sentinel_cohort,
            pd.read_csv(evidence_dir(args, loop_id) / "sentinel_entry_evidence.csv"),
            SENTINEL_ENTRIES,
        ).merge(
            sentinel_private[["entry_key", "clean_label", "injected_error", "flip_direction"]],
            on="entry_key", validate="one_to_one",
        )
        state = true_state(action_cohort, private)
        if loop_id:
            selected = pd.read_csv(loop_dir(output, loop_id) / "selected_entries_private.csv")
            new_reviews = len(selected)
            new_errors = int(selected["true_issue"].sum())
        else:
            new_reviews = 0
            new_errors = 0
        cumulative = history[history["loop"] <= loop_id] if loop_id else history.iloc[0:0]
        rows.append(
            {
                "seed": args.seed,
                "loop": loop_id,
                "phase": "baseline" if loop_id == 0 else (
                    "one_shot_frozen_stop" if loop_id == 1 else "iterative_extension"
                ),
                "current_issue_pool": int(action["cl_issue"].sum()),
                "new_reviews": new_reviews,
                "new_true_errors": new_errors,
                "new_issue_precision": float(new_errors / new_reviews) if new_reviews else float("nan"),
                "cumulative_reviews": len(cumulative),
                "cumulative_errors_corrected": int(cumulative["true_issue"].sum()),
                "true_quality": state["true_quality"],
                "remaining_errors": state["remaining_errors"],
                "raw_dqs": float(1.0 - action["cl_issue"].mean()),
                "action_clean_macro_auroc": macro_auroc(action),
                "sentinel_clean_macro_auroc": macro_auroc(sentinel),
                "sentinel_error_auprc": safe_auprc(
                    sentinel["injected_error"], sentinel["cl_first_score"]
                ),
                "sentinel_error_auroc": safe_auroc(
                    sentinel["injected_error"], sentinel["cl_first_score"]
                ),
            }
        )
    trajectory = pd.DataFrame(rows)
    trajectory_path = output / "full_issue_iteration_trajectory_private.csv"
    atomic_write_csv(trajectory, trajectory_path)
    one_shot = trajectory.loc[trajectory["loop"].eq(1)].iloc[0]
    final = trajectory.iloc[-1]
    summary = {
        "protocol": PROTOCOL_NAME,
        "seed": args.seed,
        "loops": args.loops,
        "initial_issue_pool": int(trajectory.iloc[0]["current_issue_pool"]),
        "one_shot_reviews": int(one_shot["cumulative_reviews"]),
        "one_shot_errors_corrected": int(one_shot["cumulative_errors_corrected"]),
        "one_shot_quality": float(one_shot["true_quality"]),
        "one_shot_action_auroc": float(one_shot["action_clean_macro_auroc"]),
        "one_shot_sentinel_auroc": float(one_shot["sentinel_clean_macro_auroc"]),
        "iterative_extra_reviews": int(final["cumulative_reviews"] - one_shot["cumulative_reviews"]),
        "iterative_extra_errors_corrected": int(
            final["cumulative_errors_corrected"] - one_shot["cumulative_errors_corrected"]
        ),
        "iterative_extra_quality": float(final["true_quality"] - one_shot["true_quality"]),
        "final_quality": float(final["true_quality"]),
        "final_action_auroc": float(final["action_clean_macro_auroc"]),
        "final_sentinel_auroc": float(final["sentinel_clean_macro_auroc"]),
        "post_one_shot_action_auroc_change": float(
            final["action_clean_macro_auroc"] - one_shot["action_clean_macro_auroc"]
        ),
        "post_one_shot_sentinel_auroc_change": float(
            final["sentinel_clean_macro_auroc"] - one_shot["sentinel_clean_macro_auroc"]
        ),
        "trajectory_sha256": sha256_file(trajectory_path),
    }
    atomic_json(summary, output / "seed_evaluation_summary.json")
    atomic_write_text(output / ".seed_evaluation_complete", "complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def aggregate(args: argparse.Namespace) -> None:
    root = Path(args.experiment_root)
    output = Path(args.aggregate_output)
    output.mkdir(parents=True, exist_ok=False)
    seeds = parse_seeds(args.seeds)
    summaries = []
    trajectories = []
    for seed in seeds:
        run = root / f"seed_{seed}"
        if not (run / ".seed_evaluation_complete").is_file():
            raise FileNotFoundError(f"Incomplete seed evaluation: {run}")
        summaries.append(json.loads((run / "seed_evaluation_summary.json").read_text()))
        trajectories.append(pd.read_csv(run / "full_issue_iteration_trajectory_private.csv"))
    summary = pd.DataFrame(summaries)
    trajectory = pd.concat(trajectories, ignore_index=True)
    atomic_write_csv(summary, output / "seed_summaries.csv")
    atomic_write_csv(trajectory, output / "all_trajectories.csv")
    tests = []
    for metric in (
        "iterative_extra_quality",
        "post_one_shot_action_auroc_change",
        "post_one_shot_sentinel_auroc_change",
    ):
        values = summary[metric].to_numpy(dtype=float)
        tests.append(
            {
                "contrast": metric,
                "paired_seeds": len(values),
                "mean_difference": float(values.mean()),
                "positive_seeds": int((values > 0).sum()),
                "negative_seeds": int((values < 0).sum()),
                "exact_sign_flip_p": exact_sign_flip_p(values) if len(values) > 1 else float("nan"),
                "analysis_role": "single_seed_descriptive_pilot" if len(values) == 1 else "paired_seed_analysis",
            }
        )
    atomic_write_csv(pd.DataFrame(tests), output / "paired_tests.csv")

    means = trajectory.groupby("loop", as_index=False).agg(
        true_quality=("true_quality", "mean"),
        raw_dqs=("raw_dqs", "mean"),
        action_auroc=("action_clean_macro_auroc", "mean"),
        sentinel_auroc=("sentinel_clean_macro_auroc", "mean"),
        current_issue_pool=("current_issue_pool", "mean"),
        new_reviews=("new_reviews", "mean"),
        new_true_errors=("new_true_errors", "mean"),
    )
    atomic_write_csv(means, output / "mean_trajectory.csv")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].plot(means["loop"], means["true_quality"], marker="o", label="True quality")
    axes[0].plot(means["loop"], means["raw_dqs"], marker="o", label="Raw DQS")
    axes[1].plot(means["loop"], means["action_auroc"], marker="o", label="Action OOF AUROC")
    axes[1].plot(means["loop"], means["sentinel_auroc"], marker="o", label="Sentinel AUROC")
    axes[2].plot(means["loop"], means["new_reviews"], marker="o", label="New issues reviewed")
    axes[2].plot(means["loop"], means["new_true_errors"], marker="o", label="True errors corrected")
    for axis, title in zip(
        axes,
        ["Dataset quality", "Evidence-model AUROC", "Per-loop full issue pool"],
    ):
        axis.set_title(title)
        axis.set_xlabel("Cleaning loop")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "full_issue_pool_iteration_summary.png", dpi=200)
    plt.close(figure)
    atomic_json(
        {
            "protocol": PROTOCOL_NAME,
            "seeds": seeds,
            "runs": len(summary),
            "loop_1_interpretation": "one-shot full initial issue-pool cleaning",
            "loops_2_to_final_interpretation": "incremental iterative full new-issue cleaning",
            "program_sha256": sha256_file(Path(__file__)),
        },
        output / "aggregate_summary.json",
    )
    atomic_write_text(output / ".aggregate_complete", "complete\n")


def common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-prepared", type=Path, required=True)
    parser.add_argument("--private-reference", type=Path, required=True)
    parser.add_argument("--sentinel-private-reference", type=Path, required=True)
    parser.add_argument("--initial-evidence-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--loops", type=int, default=LOOPS)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = common_parser()
    commands.add_parser("initialize", parents=[common]).set_defaults(function=initialize)
    selection = commands.add_parser("select", parents=[common])
    selection.add_argument("--loop-id", type=int, required=True)
    selection.set_defaults(function=select_all_new_issues)
    inspection = commands.add_parser("inspect-next", parents=[common])
    inspection.add_argument("--loop-id", type=int, required=True)
    inspection.set_defaults(function=inspect_next_issue_pool)
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
    args.function(args)


if __name__ == "__main__":
    main()
