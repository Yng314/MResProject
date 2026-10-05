#!/usr/bin/env python3
"""VinDr stress test for iterative CL self-optimization under controlled noise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from vindr_iterative_oracle_cleaning import entry_key
from vindr_iterative_evidence_improvement import rank_entries
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
from vindr_noise_direction_sensitivity import choose_shared_folds, stable_rng


PROTOCOL_NAME = "vindr_self_optimization_stress_v1"
DEFAULT_SEEDS = (11003, 13007, 17011, 19001, 23003, 27011, 31013, 37003)
STRUCTURES = ("uniform", "hard")
RATES = (0.20, 0.30, 0.40)
INITIALIZATIONS = ("scratch", "xrv_pretrained")
ACTION_IMAGES = 2_400
ACTION_ENTRIES = ACTION_IMAGES * len(LABELS)
SENTINEL_IMAGES = 600
SENTINEL_ENTRIES = SENTINEL_IMAGES * len(LABELS)
REVIEW_BUDGET = 288
LOOPS = 5
SENTINEL_NOISE_RATE = 0.20
SENTINEL_REVIEW_BUDGET = 72


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2))


def scenario_id(structure: str, rate: float) -> str:
    return f"{structure}_r{int(round(rate * 100)):02d}"


def parse_rates(text: str) -> list[float]:
    rates = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not rates or any(rate <= 0 or rate >= 0.5 for rate in rates):
        raise ValueError("Noise rates must be non-empty and within (0, 0.5)")
    return rates


def cohort_entries(cohort: pd.DataFrame) -> pd.DataFrame:
    require_columns(cohort, ["image_id", "image_path", "fold_id"] + LABELS, "cohort")
    validate_blind_columns(cohort)
    parts = []
    for label_index, label in enumerate(LABELS):
        part = cohort[["image_id", "fold_id", label]].copy()
        part.columns = ["image_id", "fold_id", "current_label"]
        part["label_index"] = label_index
        part["label_name"] = label
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    frame["entry_key"] = entry_key(frame["image_id"], frame["label_name"])
    expected = len(cohort) * len(LABELS)
    if len(frame) != expected or frame["entry_key"].duplicated().any():
        raise ValueError("Cohort entry keys are incomplete or duplicated")
    return frame


def private_frame(
    image_ids: pd.Series,
    clean: np.ndarray,
    noisy: np.ndarray,
    injected: np.ndarray,
    direction: np.ndarray,
) -> pd.DataFrame:
    records = []
    for row_index, image_id in enumerate(image_ids.astype(str)):
        for label_index, label in enumerate(LABELS):
            records.append(
                {
                    "image_id": image_id,
                    "label_name": label,
                    "clean_label": int(clean[row_index, label_index]),
                    "noisy_label": int(noisy[row_index, label_index]),
                    "injected_error": int(injected[row_index, label_index]),
                    "flip_direction": str(direction[row_index, label_index]),
                }
            )
    frame = pd.DataFrame(records)
    frame["entry_key"] = entry_key(frame["image_id"], frame["label_name"])
    if len(frame) != len(image_ids) * len(LABELS) or frame["entry_key"].duplicated().any():
        raise RuntimeError("Private reference keys are incomplete or duplicated")
    expected = frame["clean_label"].ne(frame["noisy_label"]).astype(int)
    if not frame["injected_error"].eq(expected).all():
        raise RuntimeError("Private error flags disagree with clean/noisy labels")
    return frame


def validate_private(private: pd.DataFrame, expected_entries: int) -> pd.DataFrame:
    require_columns(
        private,
        [
            "image_id", "label_name", "clean_label", "noisy_label",
            "injected_error", "flip_direction",
        ],
        "private reference",
    )
    result = private.copy()
    result["entry_key"] = entry_key(result["image_id"], result["label_name"])
    if len(result) != expected_entries or result["entry_key"].duplicated().any():
        raise ValueError("Private reference size or keys do not match the locked design")
    return result


def validate_evidence(
    cohort: pd.DataFrame,
    evidence: pd.DataFrame,
    expected_entries: int,
) -> pd.DataFrame:
    require_columns(
        evidence,
        [
            "image_id", "fold_id", "label_index", "label_name", "noisy_label",
            "oof_probability", "cl_issue", "self_confidence_suspicion",
            "cl_first_score",
        ],
        "entry evidence",
    )
    result = evidence.copy()
    result["entry_key"] = entry_key(result["image_id"], result["label_name"])
    if len(result) != expected_entries or result["entry_key"].duplicated().any():
        raise ValueError("Evidence size or keys do not match the locked design")
    current = cohort_entries(cohort)[["entry_key", "current_label"]]
    merged = result.merge(current, on="entry_key", validate="one_to_one")
    if not merged["noisy_label"].astype(int).eq(merged["current_label"].astype(int)).all():
        raise ValueError("Evidence labels differ from the current cohort")
    if not np.isfinite(merged["oof_probability"]).all():
        raise ValueError("Evidence contains non-finite probabilities")
    return merged


def ordered_candidates(
    clean: np.ndarray,
    image_ids: pd.Series,
    hardness: pd.DataFrame,
    seed: int,
    structure: str,
) -> dict[tuple[int, str], np.ndarray]:
    hard = hardness.copy()
    hard["entry_key"] = entry_key(hard["image_id"], hard["label_name"])
    hard_scores = hard.set_index("entry_key")["label_quality_self_confidence"]
    orders: dict[tuple[int, str], np.ndarray] = {}
    for label_index, label in enumerate(LABELS):
        for clean_value, direction in ((0, "0_to_1"), (1, "1_to_0")):
            indices = np.flatnonzero(clean[:, label_index] == clean_value)
            if structure == "uniform":
                rng = stable_rng(PROTOCOL_NAME, seed, structure, label, direction)
                orders[(label_index, direction)] = rng.permutation(indices)
                continue
            keys = pd.Series(image_ids.iloc[indices].astype(str)).reset_index(drop=True) + "::" + label
            scores = hard_scores.reindex(keys).to_numpy(dtype=float)
            if not np.isfinite(scores).all():
                raise ValueError("Independent hardness evidence does not cover all entries")
            rng = stable_rng(PROTOCOL_NAME, seed, structure, label, direction, "tie")
            tie_break = rng.random(len(indices))
            order = np.lexsort((tie_break, scores))
            orders[(label_index, direction)] = indices[order]
    return orders


def corrupt_from_orders(
    clean: np.ndarray,
    orders: dict[tuple[int, str], np.ndarray],
    rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    noisy = clean.copy()
    injected = np.zeros_like(clean, dtype=bool)
    direction = np.full(clean.shape, "none", dtype=object)
    counts = []
    for label_index, label in enumerate(LABELS):
        positives = np.flatnonzero(clean[:, label_index] == 1)
        negatives = np.flatnonzero(clean[:, label_index] == 0)
        target_total = int(round(rate * len(clean)))
        fn_count = int(round(rate * len(positives)))
        fp_count = target_total - fn_count
        selected_fp = orders[(label_index, "0_to_1")][:fp_count]
        selected_fn = orders[(label_index, "1_to_0")][:fn_count]
        if len(selected_fp) != fp_count or len(selected_fn) != fn_count:
            raise ValueError(f"Infeasible corruption for {label}")
        noisy[selected_fp, label_index] = 1
        noisy[selected_fn, label_index] = 0
        injected[selected_fp, label_index] = True
        injected[selected_fn, label_index] = True
        direction[selected_fp, label_index] = "0_to_1"
        direction[selected_fn, label_index] = "1_to_0"
        counts.append(
            {
                "label_name": label,
                "clean_positives": int(len(positives)),
                "clean_negatives": int(len(negatives)),
                "false_positive_errors": fp_count,
                "false_negative_errors": fn_count,
                "injected_errors": fp_count + fn_count,
            }
        )
    expected = int(round(rate * clean.size))
    if int(injected.sum()) != expected:
        raise RuntimeError(f"Injected {int(injected.sum())} errors; expected {expected}")
    return noisy, injected, direction, pd.DataFrame(counts)


def prepare(args: argparse.Namespace) -> None:
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=False)
    labels = pd.read_csv(args.labels_csv)
    image_index = pd.read_csv(args.image_index)
    hardness = pd.read_csv(args.hardness_evidence)
    require_columns(labels, ["image_id"] + LABELS, "VinDr labels")
    require_columns(image_index, ["image_id", "image_path", "output_sha256"], "image index")
    require_columns(
        hardness,
        ["image_id", "label_name", "noisy_label", "label_quality_self_confidence"],
        "hardness evidence",
    )
    labels["image_id"] = labels["image_id"].astype(str)
    image_index["image_id"] = image_index["image_id"].astype(str)
    hardness["image_id"] = hardness["image_id"].astype(str)
    cohort = image_index.merge(labels[["image_id"] + LABELS], on="image_id", validate="one_to_one")
    cohort = cohort.sort_values("image_id").reset_index(drop=True)
    if len(cohort) != ACTION_IMAGES + SENTINEL_IMAGES:
        raise ValueError("VinDr cohort must contain exactly 3,000 images")
    hard_check = hardness.merge(
        labels.melt(id_vars="image_id", value_vars=LABELS, var_name="label_name", value_name="clean_label"),
        on=["image_id", "label_name"], validate="one_to_one",
    )
    if len(hard_check) != 18_000 or not hard_check["noisy_label"].astype(int).eq(hard_check["clean_label"]).all():
        raise ValueError("Hardness evidence is not a complete clean-label reference")

    seeds = parse_seeds(args.seeds)
    rates = parse_rates(args.rates)
    structures = [value.strip() for value in args.structures.split(",") if value.strip()]
    if any(value not in STRUCTURES for value in structures):
        raise ValueError(f"Structures must be drawn from {STRUCTURES}")
    atomic_write_csv(
        cohort[["image_id", "image_path", "output_sha256"]], output / "image_index.csv"
    )
    scenarios = [
        {"scenario_id": scenario_id(structure, rate), "structure": structure, "noise_rate": rate}
        for structure in structures for rate in rates
    ]
    atomic_write_csv(pd.DataFrame(scenarios), output / "scenarios.csv")
    manifest_rows = []

    for seed in seeds:
        rng = stable_rng(PROTOCOL_NAME, seed, "sentinel_split")
        sentinel_indices = np.sort(rng.choice(len(cohort), size=SENTINEL_IMAGES, replace=False))
        action_indices = np.setdiff1d(np.arange(len(cohort)), sentinel_indices)
        action = cohort.iloc[action_indices].reset_index(drop=True)
        sentinel = cohort.iloc[sentinel_indices].reset_index(drop=True)
        action_clean = action[LABELS].to_numpy(dtype=np.int64)
        sentinel_clean = sentinel[LABELS].to_numpy(dtype=np.int64)
        action_hardness = hardness[hardness["image_id"].isin(set(action["image_id"]))].copy()
        sentinel_hardness = hardness[hardness["image_id"].isin(set(sentinel["image_id"]))].copy()

        generated: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]] = {}
        for structure in structures:
            orders = ordered_candidates(action_clean, action["image_id"], action_hardness, seed, structure)
            for rate in rates:
                generated[scenario_id(structure, rate)] = corrupt_from_orders(action_clean, orders, rate)
        folds, fold_seed = choose_shared_folds(
            {key: value[0] for key, value in generated.items()}, seed, args.n_splits
        )

        sentinel_orders = ordered_candidates(
            sentinel_clean, sentinel["image_id"], sentinel_hardness, seed, "hard"
        )
        sentinel_noisy, sentinel_injected, sentinel_direction, sentinel_counts = corrupt_from_orders(
            sentinel_clean, sentinel_orders, args.sentinel_noise_rate
        )
        sentinel_blind = sentinel[["image_id", "image_path"]].copy()
        sentinel_blind["fold_id"] = -1
        for label_index, label in enumerate(LABELS):
            sentinel_blind[label] = sentinel_noisy[:, label_index]
        sentinel_private = private_frame(
            sentinel["image_id"], sentinel_clean, sentinel_noisy,
            sentinel_injected, sentinel_direction,
        )

        for scenario in scenarios:
            sid = scenario["scenario_id"]
            noisy, injected, direction, counts = generated[sid]
            prepared = output / "scenarios" / sid / f"seed_{seed}" / "prepared"
            prepared.mkdir(parents=True, exist_ok=False)
            blind = action[["image_id", "image_path"]].copy()
            blind["fold_id"] = folds
            for label_index, label in enumerate(LABELS):
                blind[label] = noisy[:, label_index]
            validate_blind_columns(blind)
            private = private_frame(action["image_id"], action_clean, noisy, injected, direction)
            paths = {
                "blind": prepared / "blind_noisy_cohort.csv",
                "private": prepared / "private_reference.csv",
                "sentinel_blind": prepared / "sentinel_blind_cohort.csv",
                "sentinel_private": prepared / "sentinel_private_reference.csv",
                "counts": prepared / "corruption_counts.csv",
                "sentinel_counts": prepared / "sentinel_corruption_counts.csv",
            }
            atomic_write_csv(blind, paths["blind"])
            atomic_write_csv(private.drop(columns="entry_key"), paths["private"])
            atomic_write_csv(sentinel_blind, paths["sentinel_blind"])
            atomic_write_csv(sentinel_private.drop(columns="entry_key"), paths["sentinel_private"])
            atomic_write_csv(counts, paths["counts"])
            atomic_write_csv(sentinel_counts, paths["sentinel_counts"])
            summary = {
                "protocol": PROTOCOL_NAME,
                "scenario_id": sid,
                "structure": scenario["structure"],
                "noise_rate": scenario["noise_rate"],
                "seed": seed,
                "action_images": ACTION_IMAGES,
                "action_entries": ACTION_ENTRIES,
                "action_errors": int(injected.sum()),
                "sentinel_images": SENTINEL_IMAGES,
                "sentinel_entries": SENTINEL_ENTRIES,
                "sentinel_noise_rate": args.sentinel_noise_rate,
                "sentinel_errors": int(sentinel_injected.sum()),
                "fold_assignment_seed": fold_seed,
                "hardness_evidence_sha256": sha256_file(Path(args.hardness_evidence)),
                **{f"{key}_sha256": sha256_file(path) for key, path in paths.items()},
            }
            atomic_json(summary, prepared / "prepare_summary.json")
            atomic_write_text(prepared / ".prepare_complete", "complete\n")
            manifest_rows.append(
                {
                    "scenario_id": sid,
                    "structure": scenario["structure"],
                    "noise_rate": scenario["noise_rate"],
                    "seed": seed,
                    "prepared_path": str(prepared),
                    "action_errors": int(injected.sum()),
                    "sentinel_errors": int(sentinel_injected.sum()),
                    "blind_sha256": sha256_file(paths["blind"]),
                    "private_sha256": sha256_file(paths["private"]),
                    "sentinel_blind_sha256": sha256_file(paths["sentinel_blind"]),
                    "sentinel_private_sha256": sha256_file(paths["sentinel_private"]),
                }
            )
    manifest = pd.DataFrame(manifest_rows).sort_values(["scenario_id", "seed"]).reset_index(drop=True)
    manifest.insert(0, "manifest_index", np.arange(len(manifest)))
    atomic_write_csv(manifest, output / "prepared_manifest.csv")
    atomic_json(
        {
            "protocol": PROTOCOL_NAME,
            "seeds": seeds,
            "structures": structures,
            "rates": rates,
            "prepared_runs": len(manifest),
            "labels_sha256": sha256_file(Path(args.labels_csv)),
            "source_image_index_sha256": sha256_file(Path(args.image_index)),
            "hardness_evidence_sha256": sha256_file(Path(args.hardness_evidence)),
            "program_sha256": sha256_file(Path(__file__)),
        },
        output / "prepare_manifest.json",
    )
    atomic_write_text(output / ".prepare_complete", "complete\n")


def state_dir(output: Path) -> Path:
    return output / "state"


def loop_dir(output: Path, loop_id: int) -> Path:
    return output / f"loop_{loop_id:02d}"


def evidence_dir(output: Path, loop_id: int) -> Path:
    suffix = "oof_evidence" if loop_id == 0 else "oof_after_action"
    return loop_dir(output, loop_id) / suffix


def current_cohort(args: argparse.Namespace, loop_id: int) -> Path:
    if loop_id == 0:
        return Path(args.source_prepared) / "blind_noisy_cohort.csv"
    return loop_dir(Path(args.output_dir), loop_id) / "cohort_after_action_blind.csv"


def true_state(cohort: pd.DataFrame, private: pd.DataFrame) -> dict[str, Any]:
    entries = cohort_entries(cohort)[["entry_key", "current_label"]]
    merged = entries.merge(private[["entry_key", "clean_label"]], on="entry_key", validate="one_to_one")
    wrong = merged["current_label"].astype(int).ne(merged["clean_label"].astype(int))
    return {"remaining_errors": int(wrong.sum()), "true_quality": float(1.0 - wrong.mean())}


def initialize(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    state = state_dir(output)
    state.mkdir()
    prepared = Path(args.source_prepared)
    action = pd.read_csv(prepared / "blind_noisy_cohort.csv")
    sentinel = pd.read_csv(prepared / "sentinel_blind_cohort.csv")
    action_private = validate_private(pd.read_csv(args.private_reference), ACTION_ENTRIES)
    sentinel_private = validate_private(pd.read_csv(args.sentinel_private_reference), SENTINEL_ENTRIES)
    if len(action) != ACTION_IMAGES or len(sentinel) != SENTINEL_IMAGES:
        raise ValueError("Action/sentinel sizes do not match the locked design")
    if set(action["image_id"].astype(str)) & set(sentinel["image_id"].astype(str)):
        raise ValueError("Action and sentinel image sets overlap")
    action_state = true_state(action, action_private)
    if action_state["remaining_errors"] != int(round(args.noise_rate * ACTION_ENTRIES)):
        raise ValueError("Action corruption does not match --noise-rate")
    if int(sentinel_private["injected_error"].sum()) != int(round(SENTINEL_NOISE_RATE * SENTINEL_ENTRIES)):
        raise ValueError("Sentinel corruption does not match the locked rate")
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
    atomic_json(
        {
            "protocol": PROTOCOL_NAME,
            "scenario_id": args.scenario_id,
            "structure": args.structure,
            "noise_rate": args.noise_rate,
            "seed": args.seed,
            "initialization": args.initialization,
            "loops": args.loops,
            "review_budget": args.review_budget,
            "action_images": ACTION_IMAGES,
            "sentinel_images": SENTINEL_IMAGES,
            "action_initial_errors": action_state["remaining_errors"],
            "sentinel_errors": int(sentinel_private["injected_error"].sum()),
            "action_sha256": sha256_file(prepared / "blind_noisy_cohort.csv"),
            "sentinel_sha256": sha256_file(prepared / "sentinel_blind_cohort.csv"),
        },
        state / "initialization_manifest.json",
    )
    atomic_write_text(state / ".initialized", "complete\n")


def select(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    target = loop_dir(output, args.loop_id)
    target.mkdir(parents=True, exist_ok=False)
    previous_loop = args.loop_id - 1
    cohort = pd.read_csv(current_cohort(args, previous_loop))
    evidence_root = evidence_dir(output, previous_loop)
    if not (evidence_root / ".blind_run_complete").is_file():
        raise FileNotFoundError("Previous evidence is incomplete")
    evidence = validate_evidence(cohort, pd.read_csv(evidence_root / "entry_evidence.csv"), ACTION_ENTRIES)
    history = pd.read_csv(state_dir(output) / "review_history_blind.csv")
    reviewed = set(history["entry_key"].astype(str)) if not history.empty else set()
    candidates = evidence[~evidence["entry_key"].isin(reviewed)].copy()
    selected = rank_entries(candidates).head(args.review_budget).copy()
    if len(selected) != args.review_budget or selected["entry_key"].duplicated().any():
        raise RuntimeError("Selection did not produce the fixed no-repeat budget")
    columns = [
        "entry_key", "image_id", "fold_id", "label_index", "label_name",
        "current_label", "cl_issue", "cl_first_score",
        "self_confidence_suspicion", "oof_probability",
    ]
    atomic_write_csv(selected[columns], target / "selected_entries_blind.csv")
    atomic_json(
        {
            "protocol": PROTOCOL_NAME,
            "loop": args.loop_id,
            "eligible_unreviewed_entries": len(candidates),
            "review_budget": args.review_budget,
            "selected_cl_issues": int(selected["cl_issue"].sum()),
            "source_evidence_sha256": sha256_file(evidence_root / "entry_evidence.csv"),
        },
        target / "selection_manifest_blind.json",
    )
    atomic_write_text(target / ".selection_complete", "complete\n")


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
    before_path = current_cohort(args, args.loop_id - 1)
    before = pd.read_csv(before_path)
    selected = pd.read_csv(target / "selected_entries_blind.csv")
    private = validate_private(pd.read_csv(args.private_reference), ACTION_ENTRIES)
    joined = selected.merge(
        private[["entry_key", "clean_label", "flip_direction"]],
        on="entry_key", how="left", validate="one_to_one",
    )
    joined["true_issue"] = joined["current_label"].astype(int).ne(joined["clean_label"].astype(int)).astype(int)
    after = apply_oracle(before, joined)
    after_path = target / "cohort_after_action_blind.csv"
    atomic_write_csv(after, after_path)
    atomic_write_csv(joined, target / "selected_entries_private.csv")

    blind_path = state_dir(output) / "review_history_blind.csv"
    private_path = state_dir(output) / "review_history_private.csv"
    blind_history = pd.read_csv(blind_path)
    private_history = pd.read_csv(private_path)
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
        raise RuntimeError("A reviewed entry re-entered a later loop")
    atomic_write_csv(blind_history, blind_path)
    atomic_write_csv(private_history, private_path)
    before_state = true_state(before, private)
    after_state = true_state(after, private)
    atomic_json(
        {
            "protocol": PROTOCOL_NAME,
            "loop": args.loop_id,
            "selected_entries": len(joined),
            "selected_true_errors": int(joined["true_issue"].sum()),
            "selection_precision": float(joined["true_issue"].mean()),
            "quality_before": before_state["true_quality"],
            "quality_after": after_state["true_quality"],
            "remaining_errors_after": after_state["remaining_errors"],
            "cohort_after_sha256": sha256_file(after_path),
        },
        target / "oracle_summary_private.json",
    )
    atomic_write_text(target / ".oracle_update_complete", "complete\n")


def finalize_loop(args: argparse.Namespace) -> None:
    target = loop_dir(Path(args.output_dir), args.loop_id)
    evidence = evidence_dir(Path(args.output_dir), args.loop_id)
    if not (target / ".oracle_update_complete").is_file() or not (evidence / ".blind_run_complete").is_file():
        raise FileNotFoundError("Oracle update or post-action evidence is incomplete")
    action = validate_evidence(
        pd.read_csv(target / "cohort_after_action_blind.csv"),
        pd.read_csv(evidence / "entry_evidence.csv"),
        ACTION_ENTRIES,
    )
    summary = json.loads((target / "oracle_summary_private.json").read_text())
    summary.update(
        {
            "raw_dqs_after_action": float(1.0 - action["cl_issue"].mean()),
            "cl_issue_entries_after_action": int(action["cl_issue"].sum()),
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
    return float(average_precision_score(truth_array, score_array)) if truth_array.sum() else float("nan")


def macro_auroc(frame: pd.DataFrame) -> float:
    values = [safe_auroc(group["clean_label"], group["oof_probability"]) for _, group in frame.groupby("label_name")]
    return float(np.nanmean(values))


def audc(frame: pd.DataFrame, value: str) -> float:
    ordered = frame.sort_values("review_fraction")
    return float(np.trapezoid(ordered[value], ordered["review_fraction"]))


def evaluate(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    for loop_id in range(args.loops + 1):
        if not (evidence_dir(output, loop_id) / ".blind_run_complete").is_file():
            raise FileNotFoundError(f"Evidence missing at Loop {loop_id}")
        if loop_id and not (loop_dir(output, loop_id) / ".loop_complete").is_file():
            raise FileNotFoundError(f"Loop marker missing at Loop {loop_id}")
    action_private = validate_private(pd.read_csv(args.private_reference), ACTION_ENTRIES)
    sentinel_private = validate_private(pd.read_csv(args.sentinel_private_reference), SENTINEL_ENTRIES)
    sentinel_cohort = pd.read_csv(Path(args.source_prepared) / "sentinel_blind_cohort.csv")
    history = pd.read_csv(state_dir(output) / "review_history_private.csv")
    initial_action = validate_evidence(
        pd.read_csv(current_cohort(args, 0)),
        pd.read_csv(evidence_dir(output, 0) / "entry_evidence.csv"),
        ACTION_ENTRIES,
    ).merge(
        action_private[["entry_key", "clean_label", "injected_error", "flip_direction"]],
        on="entry_key", validate="one_to_one",
    )
    frozen_rank = rank_entries(initial_action)
    initial_errors = int(initial_action["injected_error"].sum())
    evidence_rows = []
    discovery_rows = []
    previous_dynamic_errors = 0

    for loop_id in range(args.loops + 1):
        action_cohort = pd.read_csv(current_cohort(args, loop_id))
        action = validate_evidence(
            action_cohort,
            pd.read_csv(evidence_dir(output, loop_id) / "entry_evidence.csv"),
            ACTION_ENTRIES,
        ).merge(
            action_private[["entry_key", "clean_label", "injected_error", "flip_direction"]],
            on="entry_key", validate="one_to_one",
        )
        sentinel = validate_evidence(
            sentinel_cohort,
            pd.read_csv(evidence_dir(output, loop_id) / "sentinel_entry_evidence.csv"),
            SENTINEL_ENTRIES,
        ).merge(
            sentinel_private[["entry_key", "clean_label", "injected_error", "flip_direction"]],
            on="entry_key", validate="one_to_one",
        )
        current = true_state(action_cohort, action_private)
        sentinel_rank = rank_entries(sentinel)
        sentinel_top = sentinel_rank.head(args.sentinel_review_budget)
        sentinel_found = int(sentinel_top["injected_error"].sum())
        dynamic_errors = int(history.loc[history["loop"] <= loop_id, "true_issue"].sum()) if loop_id else 0
        marginal_errors = dynamic_errors - previous_dynamic_errors
        previous_dynamic_errors = dynamic_errors
        evidence_rows.append(
            {
                "seed": args.seed,
                "scenario_id": args.scenario_id,
                "structure": args.structure,
                "noise_rate": args.noise_rate,
                "initialization": args.initialization,
                "loop": loop_id,
                "true_quality": current["true_quality"],
                "remaining_errors": current["remaining_errors"],
                "raw_dqs": float(1.0 - action["cl_issue"].mean()),
                "action_clean_macro_auroc": macro_auroc(action),
                "sentinel_clean_macro_auroc": macro_auroc(sentinel),
                "sentinel_error_auprc": safe_auprc(sentinel["injected_error"], sentinel["cl_first_score"]),
                "sentinel_error_auroc": safe_auroc(sentinel["injected_error"], sentinel["cl_first_score"]),
                "sentinel_error_recall_at_budget": float(sentinel_found / sentinel["injected_error"].sum()),
                "sentinel_error_precision_at_budget": float(sentinel_found / args.sentinel_review_budget),
                "marginal_errors_corrected": marginal_errors,
                "marginal_action_yield": float(marginal_errors / args.review_budget) if loop_id else float("nan"),
            }
        )
        budget = loop_id * args.review_budget
        frozen_errors = int(frozen_rank.head(budget)["injected_error"].sum())
        random_errors = float(budget * initial_errors / ACTION_ENTRIES)
        for method, found in (
            ("dynamic_cl", float(dynamic_errors)),
            ("frozen_loop0_cl", float(frozen_errors)),
            ("random_review_expected", random_errors),
        ):
            discovery_rows.append(
                {
                    "seed": args.seed,
                    "scenario_id": args.scenario_id,
                    "structure": args.structure,
                    "noise_rate": args.noise_rate,
                    "initialization": args.initialization,
                    "method": method,
                    "loop": loop_id,
                    "cumulative_reviews": budget,
                    "review_fraction": float(budget / ACTION_ENTRIES),
                    "cumulative_errors_found": found,
                    "action_error_recall": float(found / initial_errors),
                }
            )

    evidence_frame = pd.DataFrame(evidence_rows)
    discovery_frame = pd.DataFrame(discovery_rows)
    atomic_write_csv(evidence_frame, output / "evidence_trajectory_private.csv")
    atomic_write_csv(discovery_frame, output / "discovery_trajectory_private.csv")
    branches = {method: discovery_frame[discovery_frame["method"] == method] for method in discovery_frame["method"].unique()}
    first, last = evidence_frame.iloc[0], evidence_frame.iloc[-1]
    dqs_rho = spearmanr(evidence_frame["true_quality"], evidence_frame["raw_dqs"]).statistic
    summary = {
        "protocol": PROTOCOL_NAME,
        "seed": args.seed,
        "scenario_id": args.scenario_id,
        "structure": args.structure,
        "noise_rate": args.noise_rate,
        "initialization": args.initialization,
        "loops": args.loops,
        "initial_quality": float(first["true_quality"]),
        "final_quality": float(last["true_quality"]),
        "quality_change": float(last["true_quality"] - first["true_quality"]),
        "raw_dqs_change": float(last["raw_dqs"] - first["raw_dqs"]),
        "dqs_true_quality_spearman": float(dqs_rho),
        "action_clean_macro_auroc_change": float(last["action_clean_macro_auroc"] - first["action_clean_macro_auroc"]),
        "sentinel_clean_macro_auroc_change": float(last["sentinel_clean_macro_auroc"] - first["sentinel_clean_macro_auroc"]),
        "sentinel_error_auprc_change": float(last["sentinel_error_auprc"] - first["sentinel_error_auprc"]),
        "dynamic_discovery_audc": audc(branches["dynamic_cl"], "action_error_recall"),
        "frozen_discovery_audc": audc(branches["frozen_loop0_cl"], "action_error_recall"),
        "random_discovery_audc": audc(branches["random_review_expected"], "action_error_recall"),
        "dynamic_minus_frozen_audc": audc(branches["dynamic_cl"], "action_error_recall") - audc(branches["frozen_loop0_cl"], "action_error_recall"),
        "final_dynamic_error_recall": float(branches["dynamic_cl"].sort_values("loop").iloc[-1]["action_error_recall"]),
        "final_frozen_error_recall": float(branches["frozen_loop0_cl"].sort_values("loop").iloc[-1]["action_error_recall"]),
        "evidence_trajectory_sha256": sha256_file(output / "evidence_trajectory_private.csv"),
        "discovery_trajectory_sha256": sha256_file(output / "discovery_trajectory_private.csv"),
    }
    atomic_json(summary, output / "seed_evaluation_summary.json")
    atomic_write_text(output / ".seed_evaluation_complete", "complete\n")


def aggregate(args: argparse.Namespace) -> None:
    root = Path(args.experiment_root)
    output = Path(args.aggregate_output)
    output.mkdir(parents=True, exist_ok=False)
    run_manifest = pd.read_csv(args.run_manifest)
    summaries = []
    evidence = []
    discovery = []
    for row in run_manifest.itertuples(index=False):
        run = root / str(row.initialization) / str(row.scenario_id) / f"seed_{int(row.seed)}"
        if not (run / ".seed_evaluation_complete").is_file():
            raise FileNotFoundError(f"Incomplete formal run: {run}")
        summaries.append(json.loads((run / "seed_evaluation_summary.json").read_text()))
        evidence.append(pd.read_csv(run / "evidence_trajectory_private.csv"))
        discovery.append(pd.read_csv(run / "discovery_trajectory_private.csv"))
    summary = pd.DataFrame(summaries)
    evidence_frame = pd.concat(evidence, ignore_index=True)
    discovery_frame = pd.concat(discovery, ignore_index=True)
    atomic_write_csv(summary, output / "seed_summaries.csv")
    atomic_write_csv(evidence_frame, output / "all_evidence_trajectories.csv")
    atomic_write_csv(discovery_frame, output / "all_discovery_trajectories.csv")

    tests = []
    for initialization, scenario in summary.groupby(["initialization", "scenario_id"]):
        values = scenario["dynamic_minus_frozen_audc"].to_numpy(dtype=float)
        tests.append(
            {
                "initialization": initialization[0] if isinstance(initialization, tuple) else initialization,
                "scenario_id": initialization[1] if isinstance(initialization, tuple) else "",
                "contrast": "dynamic_minus_frozen_discovery_audc",
                "paired_seeds": len(values),
                "mean_difference": float(values.mean()),
                "positive_seeds": int((values > 0).sum()),
                "exact_sign_flip_p": exact_sign_flip_p(values),
            }
        )
    tests_frame = pd.DataFrame(tests)
    tests_frame["analysis_role"] = np.where(
        (tests_frame["initialization"] == "scratch") & (tests_frame["scenario_id"] == "hard_r30"),
        "primary", "secondary",
    )
    tests_frame["holm_p_secondary"] = np.nan
    secondary = tests_frame["analysis_role"].eq("secondary")
    tests_frame.loc[secondary, "holm_p_secondary"] = holm_adjust(
        tests_frame.loc[secondary, "exact_sign_flip_p"].tolist()
    )
    atomic_write_csv(tests_frame, output / "paired_tests.csv")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    scratch = evidence_frame[evidence_frame["initialization"] == "scratch"]
    for sid, branch in scratch.groupby("scenario_id"):
        means = branch.groupby("loop", as_index=False).agg(
            true_quality=("true_quality", "mean"),
            raw_dqs=("raw_dqs", "mean"),
            sentinel_error_auprc=("sentinel_error_auprc", "mean"),
        )
        axes[0].plot(means["loop"], means["true_quality"], marker="o", label=sid)
        axes[1].plot(means["loop"], means["raw_dqs"], marker="o", label=sid)
        axes[2].plot(means["loop"], means["sentinel_error_auprc"], marker="o", label=sid)
    for axis, title in zip(
        axes,
        ["Known action-set quality", "Raw entry DQS", "Held-out sentinel error AUPRC"],
    ):
        axis.set_title(title)
        axis.set_xlabel("Cleaning loop")
        axis.grid(alpha=0.25)
    axes[2].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output / "self_optimization_stress_summary.png", dpi=200)
    plt.close(figure)
    atomic_json(
        {
            "protocol": PROTOCOL_NAME,
            "formal_runs": len(summary),
            "scratch_runs": int((summary["initialization"] == "scratch").sum()),
            "xrv_corner_runs": int((summary["initialization"] == "xrv_pretrained").sum()),
            "primary_contrast": "scratch hard_r30 dynamic-minus-frozen discovery AUDC",
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
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--structure", choices=STRUCTURES, required=True)
    parser.add_argument("--noise-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--initialization", choices=INITIALIZATIONS, required=True)
    parser.add_argument("--loops", type=int, default=LOOPS)
    parser.add_argument("--review-budget", type=int, default=REVIEW_BUDGET)
    parser.add_argument("--sentinel-review-budget", type=int, default=SENTINEL_REVIEW_BUDGET)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--labels-csv", type=Path, required=True)
    prepare_parser.add_argument("--image-index", type=Path, required=True)
    prepare_parser.add_argument("--hardness-evidence", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    prepare_parser.add_argument("--rates", default=",".join(map(str, RATES)))
    prepare_parser.add_argument("--structures", default=",".join(STRUCTURES))
    prepare_parser.add_argument("--sentinel-noise-rate", type=float, default=SENTINEL_NOISE_RATE)
    prepare_parser.add_argument("--n-splits", type=int, default=4)
    prepare_parser.set_defaults(function=prepare)
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
    aggregate_parser.add_argument("--run-manifest", type=Path, required=True)
    aggregate_parser.set_defaults(function=aggregate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "loop_id") and not 1 <= args.loop_id <= args.loops:
        raise ValueError("--loop-id must be within 1..--loops")
    args.function(args)


if __name__ == "__main__":
    main()
