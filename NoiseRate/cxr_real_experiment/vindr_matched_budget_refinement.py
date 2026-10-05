#!/usr/bin/env python3
"""Matched-budget VinDr comparison of dynamic, frozen, and one-shot review."""

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
    macro_auroc,
    safe_auprc,
    safe_auroc,
    true_state,
    validate_evidence,
    validate_private,
)


PROTOCOL_NAME = "vindr_matched_budget_refinement_v1"
DEFAULT_SEEDS = (11003, 13007, 17011, 19001, 23003, 27011, 31013, 37003)
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


def cumulative_review_endpoints(total: int, loops: int = LOOPS) -> list[int]:
    if total < loops or loops <= 0:
        raise ValueError("The total review budget must support every loop")
    endpoints = [0] + [(loop_id * total) // loops for loop_id in range(1, loops + 1)]
    if endpoints[-1] != total or any(
        right <= left for left, right in zip(endpoints, endpoints[1:])
    ):
        raise RuntimeError("Review endpoints do not form a positive exact partition")
    return endpoints


def loop_budget(manifest: dict[str, Any], loop_id: int) -> int:
    endpoints = list(map(int, manifest["cumulative_review_endpoints"]))
    return endpoints[loop_id] - endpoints[loop_id - 1]


def apply_oracle(cohort: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    updated = cohort.copy()
    row_index = {str(value): index for index, value in enumerate(updated["image_id"].astype(str))}
    for row in selected.itertuples(index=False):
        index = row_index[str(row.image_id)]
        label = str(row.label_name)
        if int(updated.at[index, label]) != int(row.current_label):
            raise ValueError(f"Selected entry changed before correction: {row.entry_key}")
        updated.at[index, label] = int(row.clean_label)
    return updated


def initialize(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    state = state_dir(output)
    state.mkdir()
    prepared = Path(args.source_prepared)
    initial_evidence = Path(args.initial_evidence_dir)
    full_seed = Path(args.full_seed_dir)
    if not (prepared / ".prepare_complete").is_file():
        raise FileNotFoundError("Prepared condition is incomplete")
    if not (initial_evidence / ".blind_run_complete").is_file():
        raise FileNotFoundError("Locked Loop-0 evidence is incomplete")
    if not (full_seed / ".worker_complete").is_file():
        raise FileNotFoundError("Matched full-pool seed is incomplete")

    cohort = pd.read_csv(prepared / "blind_noisy_cohort.csv")
    sentinel = pd.read_csv(prepared / "sentinel_blind_cohort.csv")
    private = validate_private(pd.read_csv(args.private_reference), ACTION_ENTRIES)
    if len(cohort) != ACTION_IMAGES or len(sentinel) != SENTINEL_IMAGES:
        raise ValueError("Action/sentinel sizes do not match the locked source")
    evidence = validate_evidence(
        cohort, pd.read_csv(initial_evidence / "entry_evidence.csv"), ACTION_ENTRIES
    )
    initial_issue_pool = int(evidence["cl_issue"].sum())
    endpoints = cumulative_review_endpoints(initial_issue_pool, args.loops)
    ranked = rank_entries(evidence).copy()
    ranked.insert(0, "initial_rank", np.arange(1, len(ranked) + 1))
    ranking_path = state / "frozen_loop0_ranking_blind.csv"
    atomic_write_csv(ranked, ranking_path)
    top_keys = set(ranked.head(initial_issue_pool)["entry_key"].astype(str))
    issue_keys = set(evidence.loc[evidence["cl_issue"].astype(bool), "entry_key"].astype(str))
    if top_keys != issue_keys:
        raise RuntimeError("The frozen endpoint is not exactly the initial issue pool")

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
    initial = true_state(cohort, private)
    full_manifest = json.loads((full_seed / "state/initialization_manifest.json").read_text())
    if full_manifest["initial_evidence_sha256"] != sha256_file(initial_evidence / "entry_evidence.csv"):
        raise RuntimeError("Full-pool and matched-budget runs do not share Loop-0 evidence")
    manifest = {
        "protocol": PROTOCOL_NAME,
        "seed": int(args.seed),
        "scenario_id": "hard_r30",
        "loops": int(args.loops),
        "selection": "rerank all unreviewed entries with current CL-first evidence",
        "issue_first_scoring": True,
        "initial_issue_pool": initial_issue_pool,
        "total_review_budget": initial_issue_pool,
        "cumulative_review_endpoints": endpoints,
        "per_loop_review_budgets": [
            endpoints[index] - endpoints[index - 1] for index in range(1, len(endpoints))
        ],
        "initial_errors": int(initial["remaining_errors"]),
        "action_entries": ACTION_ENTRIES,
        "sentinel_entries": SENTINEL_ENTRIES,
        "source_prepared_sha256": sha256_file(prepared / "blind_noisy_cohort.csv"),
        "initial_evidence_sha256": sha256_file(initial_evidence / "entry_evidence.csv"),
        "frozen_ranking_sha256": sha256_file(ranking_path),
        "matched_full_seed_dir": str(full_seed),
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(manifest, state / "initialization_manifest.json")
    atomic_write_text(state / ".initialized", "complete\n")
    print(json.dumps(manifest, indent=2), flush=True)


def select_dynamic(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    state = state_dir(output)
    if not (state / ".initialized").is_file():
        raise FileNotFoundError("Experiment is not initialized")
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
    history = pd.read_csv(state / "review_history_blind.csv")
    reviewed = set(history["entry_key"].astype(str)) if not history.empty else set()
    candidates = evidence[~evidence["entry_key"].isin(reviewed)].copy()
    ranked = rank_entries(candidates)
    manifest = json.loads((state / "initialization_manifest.json").read_text())
    budget = loop_budget(manifest, args.loop_id)
    selected = ranked.head(budget).copy()
    if len(selected) != budget:
        raise RuntimeError("Unreviewed entries do not satisfy the fixed loop budget")
    if selected["entry_key"].duplicated().any() or selected["entry_key"].isin(reviewed).any():
        raise RuntimeError("Selection contains duplicate or previously reviewed entries")
    columns = [
        "entry_key", "image_id", "fold_id", "label_index", "label_name",
        "current_label", "cl_issue", "cl_first_score",
        "self_confidence_suspicion", "oof_probability",
    ]
    selected_path = target / "selected_entries_blind.csv"
    atomic_write_csv(selected[columns], selected_path)
    selection_manifest = {
        "protocol": PROTOCOL_NAME,
        "loop": int(args.loop_id),
        "review_budget": budget,
        "cumulative_target": int(manifest["cumulative_review_endpoints"][args.loop_id]),
        "available_unreviewed_entries": int(len(ranked)),
        "current_unreviewed_issue_entries": int(ranked["cl_issue"].sum()),
        "selected_entries": int(len(selected)),
        "selected_current_issue_entries": int(selected["cl_issue"].sum()),
        "selected_nonissue_fallback_entries": int((~selected["cl_issue"].astype(bool)).sum()),
        "source_evidence_sha256": sha256_file(source_evidence / "entry_evidence.csv"),
        "selected_entries_sha256": sha256_file(selected_path),
    }
    atomic_json(selection_manifest, target / "selection_manifest_blind.json")
    atomic_write_text(target / ".selection_complete", "complete\n")
    print(json.dumps(selection_manifest, indent=2), flush=True)


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
        raise RuntimeError("Selected entries do not align with the private reference")
    joined["true_issue"] = joined["current_label"].astype(int).ne(
        joined["clean_label"].astype(int)
    ).astype(int)
    after = apply_oracle(before, joined)
    after_path = target / "cohort_after_action_blind.csv"
    private_path = target / "selected_entries_private.csv"
    atomic_write_csv(after, after_path)
    atomic_write_csv(joined, private_path)

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
        "loop": int(args.loop_id),
        "selected_entries": int(len(joined)),
        "selected_true_errors": int(joined["true_issue"].sum()),
        "selection_precision": float(joined["true_issue"].mean()),
        "quality_before": float(before_state["true_quality"]),
        "quality_after": float(after_state["true_quality"]),
        "remaining_errors_after": int(after_state["remaining_errors"]),
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


def area_under_discovery_curve(frame: pd.DataFrame) -> float:
    ordered = frame.sort_values("loop")
    return float(np.trapezoid(
        ordered["initial_error_recall"].to_numpy(dtype=float),
        ordered["review_fraction_of_initial_pool"].to_numpy(dtype=float),
    ))


def evaluate(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    for loop_id in range(args.loops + 1):
        if not (evidence_dir(args, loop_id) / ".blind_run_complete").is_file():
            raise FileNotFoundError(f"Evidence missing at Loop {loop_id}")
        if loop_id and not (loop_dir(output, loop_id) / ".loop_complete").is_file():
            raise FileNotFoundError(f"Loop marker missing at Loop {loop_id}")
    manifest = json.loads((state_dir(output) / "initialization_manifest.json").read_text())
    endpoints = list(map(int, manifest["cumulative_review_endpoints"]))
    private = validate_private(pd.read_csv(args.private_reference), ACTION_ENTRIES)
    sentinel_private = validate_private(
        pd.read_csv(args.sentinel_private_reference), SENTINEL_ENTRIES
    )
    sentinel_cohort = pd.read_csv(Path(args.source_prepared) / "sentinel_blind_cohort.csv")
    history = pd.read_csv(state_dir(output) / "review_history_private.csv")
    initial_cohort = pd.read_csv(cohort_path(args, 0))
    frozen = pd.read_csv(state_dir(output) / "frozen_loop0_ranking_blind.csv").merge(
        private[["entry_key", "clean_label", "injected_error", "flip_direction"]],
        on="entry_key", validate="one_to_one",
    )
    initial_errors = int(manifest["initial_errors"])
    initial_issue_pool = int(manifest["initial_issue_pool"])

    evidence_rows: list[dict[str, Any]] = []
    discovery_rows: list[dict[str, Any]] = []
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
        dynamic_state = true_state(action_cohort, private)
        cumulative = history[history["loop"] <= loop_id] if loop_id else history.iloc[0:0]
        if len(cumulative) != endpoints[loop_id]:
            raise RuntimeError(f"Dynamic review accounting failed at Loop {loop_id}")
        dynamic_errors = int(cumulative["true_issue"].sum())
        frozen_errors = int(frozen.head(endpoints[loop_id])["injected_error"].sum())
        random_errors = float(endpoints[loop_id] * initial_errors / ACTION_ENTRIES)
        evidence_rows.append(
            {
                "seed": int(args.seed),
                "loop": loop_id,
                "cumulative_reviews": endpoints[loop_id],
                "true_quality": float(dynamic_state["true_quality"]),
                "remaining_errors": int(dynamic_state["remaining_errors"]),
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
        for method, errors_found in (
            ("dynamic_reranking", float(dynamic_errors)),
            ("frozen_loop0_ranking", float(frozen_errors)),
            ("random_review_expected", random_errors),
        ):
            discovery_rows.append(
                {
                    "seed": int(args.seed),
                    "method": method,
                    "loop": loop_id,
                    "cumulative_reviews": endpoints[loop_id],
                    "review_fraction_of_initial_pool": float(
                        endpoints[loop_id] / initial_issue_pool
                    ),
                    "cumulative_errors_found": errors_found,
                    "initial_error_recall": float(errors_found / initial_errors),
                    "implied_true_quality": float(
                        1.0 - (initial_errors - errors_found) / ACTION_ENTRIES
                    ),
                }
            )

    evidence_frame = pd.DataFrame(evidence_rows)
    discovery_frame = pd.DataFrame(discovery_rows)
    evidence_path = output / "dynamic_evidence_trajectory_private.csv"
    discovery_path = output / "matched_discovery_trajectory_private.csv"
    atomic_write_csv(evidence_frame, evidence_path)
    atomic_write_csv(discovery_frame, discovery_path)

    full_seed = Path(args.full_seed_dir)
    full_selection = pd.read_csv(full_seed / "loop_01/selected_entries_blind.csv")
    frozen_endpoint = frozen.head(initial_issue_pool)
    if set(full_selection["entry_key"].astype(str)) != set(
        frozen_endpoint["entry_key"].astype(str)
    ):
        raise RuntimeError("One-shot and frozen endpoints do not review the same Loop-0 pool")
    full_trajectory = pd.read_csv(full_seed / "full_issue_iteration_trajectory_private.csv")
    one_shot = full_trajectory.loc[full_trajectory["loop"].eq(1)]
    if len(one_shot) != 1:
        raise RuntimeError("Full-pool one-shot endpoint is missing")
    one_shot = one_shot.iloc[0]
    dynamic = discovery_frame[discovery_frame["method"].eq("dynamic_reranking")]
    frozen_curve = discovery_frame[discovery_frame["method"].eq("frozen_loop0_ranking")]
    final_evidence = evidence_frame.iloc[-1]
    dynamic_final = dynamic.iloc[-1]
    frozen_final = frozen_curve.iloc[-1]
    if int(one_shot["cumulative_reviews"]) != initial_issue_pool:
        raise RuntimeError("One-shot review count is not the initial issue-pool size")
    if not np.isclose(float(one_shot["true_quality"]), float(frozen_final["implied_true_quality"])):
        raise RuntimeError("Frozen and one-shot oracle endpoints disagree")
    summary = {
        "protocol": PROTOCOL_NAME,
        "seed": int(args.seed),
        "loops": int(args.loops),
        "initial_issue_pool": initial_issue_pool,
        "total_review_budget": initial_issue_pool,
        "dynamic_final_errors_found": int(dynamic_final["cumulative_errors_found"]),
        "frozen_final_errors_found": int(frozen_final["cumulative_errors_found"]),
        "one_shot_errors_found": int(one_shot["cumulative_errors_corrected"]),
        "dynamic_minus_frozen_endpoint_errors": float(
            dynamic_final["cumulative_errors_found"] - frozen_final["cumulative_errors_found"]
        ),
        "dynamic_discovery_audc": area_under_discovery_curve(dynamic),
        "frozen_discovery_audc": area_under_discovery_curve(frozen_curve),
        "dynamic_minus_frozen_audc": float(
            area_under_discovery_curve(dynamic) - area_under_discovery_curve(frozen_curve)
        ),
        "dynamic_final_quality": float(final_evidence["true_quality"]),
        "frozen_final_quality": float(frozen_final["implied_true_quality"]),
        "one_shot_quality": float(one_shot["true_quality"]),
        "dynamic_final_action_auroc": float(final_evidence["action_clean_macro_auroc"]),
        "one_shot_action_auroc": float(one_shot["action_clean_macro_auroc"]),
        "dynamic_minus_one_shot_action_auroc": float(
            final_evidence["action_clean_macro_auroc"] - one_shot["action_clean_macro_auroc"]
        ),
        "dynamic_final_sentinel_auroc": float(final_evidence["sentinel_clean_macro_auroc"]),
        "one_shot_sentinel_auroc": float(one_shot["sentinel_clean_macro_auroc"]),
        "dynamic_minus_one_shot_sentinel_auroc": float(
            final_evidence["sentinel_clean_macro_auroc"] - one_shot["sentinel_clean_macro_auroc"]
        ),
        "frozen_endpoint_matches_one_shot_keys": True,
        "dynamic_evidence_sha256": sha256_file(evidence_path),
        "matched_discovery_sha256": sha256_file(discovery_path),
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
    evidence_frames = []
    discovery_frames = []
    for seed in seeds:
        run = root / f"seed_{seed}"
        if not (run / ".seed_evaluation_complete").is_file():
            raise FileNotFoundError(f"Incomplete seed evaluation: {run}")
        summaries.append(json.loads((run / "seed_evaluation_summary.json").read_text()))
        evidence_frames.append(pd.read_csv(run / "dynamic_evidence_trajectory_private.csv"))
        discovery_frames.append(pd.read_csv(run / "matched_discovery_trajectory_private.csv"))
    summary = pd.DataFrame(summaries)
    evidence = pd.concat(evidence_frames, ignore_index=True)
    discovery = pd.concat(discovery_frames, ignore_index=True)
    atomic_write_csv(summary, output / "seed_summaries.csv")
    atomic_write_csv(evidence, output / "all_dynamic_evidence_trajectories.csv")
    atomic_write_csv(discovery, output / "all_matched_discovery_trajectories.csv")

    tests = []
    for metric in (
        "dynamic_minus_frozen_endpoint_errors",
        "dynamic_minus_frozen_audc",
        "dynamic_minus_one_shot_action_auroc",
        "dynamic_minus_one_shot_sentinel_auroc",
    ):
        values = summary[metric].to_numpy(dtype=float)
        tests.append(
            {
                "contrast": metric,
                "paired_seeds": len(values),
                "mean_difference": float(values.mean()),
                "positive_seeds": int((values > 0).sum()),
                "negative_seeds": int((values < 0).sum()),
                "exact_sign_flip_p": exact_sign_flip_p(values),
            }
        )
    atomic_write_csv(pd.DataFrame(tests), output / "paired_tests.csv")

    means = discovery.groupby(["method", "loop"], as_index=False).agg(
        initial_error_recall=("initial_error_recall", "mean"),
        review_fraction_of_initial_pool=("review_fraction_of_initial_pool", "mean"),
    )
    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    styles = {
        "dynamic_reranking": ("-", "o"),
        "frozen_loop0_ranking": ("--", "s"),
        "random_review_expected": (":", "^"),
    }
    for method, (line, marker) in styles.items():
        branch = means[means["method"].eq(method)].sort_values("loop")
        axis.plot(
            branch["review_fraction_of_initial_pool"],
            branch["initial_error_recall"],
            linestyle=line, marker=marker, label=method,
        )
    axis.set_xlabel("Fraction of initial issue-pool budget reviewed")
    axis.set_ylabel("Recall of known initial errors")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "matched_budget_discovery_summary.png", dpi=200)
    plt.close(figure)
    atomic_json(
        {
            "protocol": PROTOCOL_NAME,
            "seeds": seeds,
            "runs": len(summary),
            "loops": LOOPS,
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
    parser.add_argument("--full-seed-dir", type=Path, required=True)
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
    selection.set_defaults(function=select_dynamic)
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
    if hasattr(args, "loops") and args.loops != LOOPS:
        raise ValueError(f"This protocol is locked to {LOOPS} loops")
    if hasattr(args, "loop_id") and not 1 <= args.loop_id <= args.loops:
        raise ValueError("--loop-id must be within 1..--loops")
    args.function(args)


if __name__ == "__main__":
    main()
