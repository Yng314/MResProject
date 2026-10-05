#!/usr/bin/env python3
"""Prepare and freeze the VinDr symmetric-r20 calibration-transfer inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROTOCOL_NAME = "vindr_symmetric_r20_top20_calibrated_v1"
EVIDENCE_PROTOCOL_NAME = "vindr_symmetric_r20_top20_calibrated_evidence_v1"
SOURCE_PROTOCOL_NAME = "vindr_global_symmetric_entry_noise_v1"
SCORE_PROTOCOL_NAME = "vindr_dqs_calibration_transfer_v1"
LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Lung Opacity",
    "Pleural effusion",
    "Pneumonia",
]
PROBABILITY_COLUMNS = [f"probability__{label}" for label in LABELS]
LOCKED_SEEDS = (13, 42, 97, 123, 211, 307, 509, 701)
N_SPLITS = 4
ACTION_IMAGES = 2_400
SENTINEL_IMAGES = 600
ACTION_ENTRIES = ACTION_IMAGES * len(LABELS)
SENTINEL_ENTRIES = SENTINEL_IMAGES * len(LABELS)
NOISE_RATE = 0.20
ACTION_ERRORS = int(ACTION_ENTRIES * NOISE_RATE)
SENTINEL_ERRORS = int(SENTINEL_ENTRIES * NOISE_RATE)
SCENARIO = {
    "scenario_id": "symmetric_entry_r20",
    "regime": "symmetric_entry",
    "noise_rate": NOISE_RATE,
}
SPLIT_RULE = "lowest_sha256(vindr-symmetric-r20-calibration-v1::image_id)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def validate_blind_columns(frame: pd.DataFrame) -> None:
    forbidden_tokens = ("clean", "reference", "injected", "true_", "error")
    forbidden = [
        column for column in frame.columns
        if any(token in column.lower() for token in forbidden_tokens)
    ]
    if forbidden:
        raise ValueError(f"Blind cohort contains forbidden outcome columns: {forbidden}")


def fold_support(blind: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for fold_id in range(n_splits):
        fold = blind.loc[blind["fold_id"].astype(int).eq(fold_id)]
        if fold.empty:
            raise ValueError(f"Fold {fold_id} is empty")
        for label in LABELS:
            positives = int(fold[label].sum())
            negatives = int(len(fold) - positives)
            records.append(
                {
                    "fold_id": fold_id,
                    "label_name": label,
                    "samples": int(len(fold)),
                    "positive_entries": positives,
                    "negative_entries": negatives,
                }
            )
    support = pd.DataFrame(records)
    if (support[["positive_entries", "negative_entries"]] < 5).any().any():
        raise ValueError("At least one action fold lacks binary-label support")
    return support


def _stable_rng(*parts: object) -> np.random.Generator:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def make_corruption(
    clean: np.ndarray,
    seed: int,
    scenario: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Exact copy of the archived symmetric-entry corruption rule."""
    noisy = clean.copy()
    injected = np.zeros_like(clean, dtype=bool)
    direction = np.full(clean.shape, "none", dtype=object)
    records: list[dict[str, Any]] = []
    rate = float(scenario["noise_rate"])
    for label_index, label in enumerate(LABELS):
        positive = np.flatnonzero(clean[:, label_index] == 1)
        negative = np.flatnonzero(clean[:, label_index] == 0)
        target_total = int(round(rate * len(clean)))
        fn_count = int(round(rate * len(positive)))
        fp_count = target_total - fn_count
        if fp_count > len(negative) or fn_count > len(positive):
            raise ValueError(f"Infeasible corruption for {label}")
        if fp_count:
            rng = _stable_rng(
                SOURCE_PROTOCOL_NAME, seed, label, scenario["scenario_id"], "0_to_1"
            )
            selected = rng.choice(negative, size=fp_count, replace=False)
            noisy[selected, label_index] = 1
            injected[selected, label_index] = True
            direction[selected, label_index] = "0_to_1"
        if fn_count:
            rng = _stable_rng(
                SOURCE_PROTOCOL_NAME, seed, label, scenario["scenario_id"], "1_to_0"
            )
            selected = rng.choice(positive, size=fn_count, replace=False)
            noisy[selected, label_index] = 0
            injected[selected, label_index] = True
            direction[selected, label_index] = "1_to_0"
        records.append(
            {
                "seed": seed,
                "scenario_id": scenario["scenario_id"],
                "regime": scenario["regime"],
                "noise_rate": rate,
                "label_name": label,
                "clean_positives": int(len(positive)),
                "clean_negatives": int(len(negative)),
                "false_positive_errors": fp_count,
                "false_negative_errors": fn_count,
                "injected_errors": fp_count + fn_count,
                "false_positive_rate": float(fp_count / len(negative)),
                "false_negative_rate": float(fn_count / len(positive)),
                "clean_positive_prevalence": float(len(positive) / len(clean)),
                "noisy_positives": int(noisy[:, label_index].sum()),
                "noisy_positive_prevalence": float(noisy[:, label_index].mean()),
            }
        )
    if int(injected.sum()) != int(round(rate * clean.size)):
        raise RuntimeError("Symmetric corruption did not produce the exact error count")
    return noisy, injected, direction, pd.DataFrame(records)


def self_confidence(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels and probabilities must be aligned binary arrays")
    if not np.isfinite(probabilities).all() or not (
        (probabilities >= 0.0) & (probabilities <= 1.0)
    ).all():
        raise ValueError("Probabilities must be finite and lie in [0, 1]")
    return np.where(labels == 1, probabilities, 1.0 - probabilities)


def select_prevalence_matching_threshold(
    scores: np.ndarray, correct: np.ndarray
) -> tuple[float, float, int]:
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=np.int64)
    if scores.ndim != 1 or correct.ndim != 1 or len(scores) != len(correct) or not len(scores):
        raise ValueError("Scores and correctness targets must be aligned non-empty vectors")
    if not np.isfinite(scores).all() or not np.isin(correct, [0, 1]).all():
        raise ValueError("Scores must be finite and correctness targets binary")
    unique = np.unique(scores)
    candidates = np.concatenate(
        ([np.nextafter(unique[-1], np.inf)], unique, [np.nextafter(unique[0], -np.inf)])
    )
    target = int(correct.sum())
    best: tuple[int, float, int] | None = None
    for threshold in candidates:
        count = int((scores >= threshold).sum())
        key = (abs(count - target), -float(threshold), count)
        if best is None or key < best:
            best = key
            best_threshold = float(threshold)
            best_count = count
    return best_threshold, best_count / len(scores), best_count


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_locked_seeds(text: str | Iterable[int]) -> list[int]:
    if isinstance(text, str):
        seeds = [int(value.strip()) for value in text.split(",") if value.strip()]
    else:
        seeds = [int(value) for value in text]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique")
    if set(seeds) != set(LOCKED_SEEDS):
        raise ValueError(f"Seeds must be exactly {list(LOCKED_SEEDS)}")
    return sorted(seeds)


def deterministic_split(image_ids: Iterable[str]) -> tuple[list[str], list[str]]:
    ids = [str(value) for value in image_ids]
    if len(ids) != ACTION_IMAGES + SENTINEL_IMAGES or len(set(ids)) != len(ids):
        raise ValueError("The source must contain exactly 3,000 unique image ids")

    def split_digest(image_id: str) -> str:
        value = f"vindr-symmetric-r20-calibration-v1::{image_id}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    ranked = sorted(ids, key=lambda image_id: (split_digest(image_id), image_id))
    sentinel = sorted(ranked[:SENTINEL_IMAGES])
    action = sorted(ranked[SENTINEL_IMAGES:])
    return action, sentinel


def _source_prepared(source_roots: list[Path], seed: int) -> tuple[Path, Path]:
    matches: list[tuple[Path, Path]] = []
    for root in source_roots:
        prepared = root / "scenarios" / SCENARIO["scenario_id"] / f"seed_{seed}" / "prepared"
        if (prepared / ".prepare_complete").is_file():
            matches.append((root, prepared))
    if len(matches) != 1:
        raise ValueError(f"Expected one archived source for seed {seed}, found {len(matches)}")
    return matches[0]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_hash(summary: dict[str, Any], key: str, path: Path) -> None:
    expected = summary.get(key)
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise RuntimeError(f"Hash verification failed for {path}")


def _load_archived_source(prepared: Path, seed: int) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    paths = {
        "blind": prepared / "blind_noisy_cohort.csv",
        "private": prepared / "private_reference.csv",
        "summary": prepared / "prepare_summary.json",
    }
    if not (prepared / ".prepare_complete").is_file():
        raise FileNotFoundError(f"Archived prepare marker missing: {prepared}")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"Archived source is incomplete: {prepared}")
    summary = _load_json(paths["summary"])
    if summary.get("protocol") != SOURCE_PROTOCOL_NAME or int(summary.get("seed", -1)) != seed:
        raise RuntimeError(f"Archived source provenance failed for seed {seed}")
    _verify_hash(summary, "blind_cohort_sha256", paths["blind"])
    _verify_hash(summary, "private_reference_sha256", paths["private"])

    blind = pd.read_csv(paths["blind"])
    private = pd.read_csv(paths["private"])
    require_columns(blind, ["image_id", "image_path", "fold_id"] + LABELS, "archived blind cohort")
    require_columns(
        private,
        ["image_id", "label_name", "clean_label", "noisy_label", "injected_error", "flip_direction"],
        "archived private reference",
    )
    validate_blind_columns(blind)
    blind["image_id"] = blind["image_id"].astype(str)
    private["image_id"] = private["image_id"].astype(str)
    if len(blind) != 3_000 or blind["image_id"].nunique() != 3_000:
        raise ValueError(f"Archived cohort for seed {seed} is not the full VinDr cohort")
    if len(private) != 18_000 or private[["image_id", "label_name"]].duplicated().any():
        raise ValueError(f"Archived private reference for seed {seed} is incomplete")
    if sorted(blind["fold_id"].unique().tolist()) != list(range(N_SPLITS)):
        raise ValueError(f"Archived fold ids for seed {seed} are not four-fold")

    clean_wide = private.pivot(index="image_id", columns="label_name", values="clean_label").reindex(
        index=blind["image_id"], columns=LABELS
    )
    noisy_wide = private.pivot(index="image_id", columns="label_name", values="noisy_label").reindex(
        index=blind["image_id"], columns=LABELS
    )
    if clean_wide.isna().any().any() or noisy_wide.isna().any().any():
        raise ValueError(f"Archived labels do not align for seed {seed}")
    clean = clean_wide.to_numpy(dtype=np.int64)
    archived_noisy = noisy_wide.to_numpy(dtype=np.int64)
    blind_noisy = blind[LABELS].to_numpy(dtype=np.int64)
    if not np.isin(clean, [0, 1]).all() or not np.array_equal(archived_noisy, blind_noisy):
        raise ValueError(f"Archived labels fail binary/noisy-cohort validation for seed {seed}")
    return blind.sort_values("image_id").reset_index(drop=True), clean_wide.sort_index().to_numpy(dtype=np.int64), summary


def private_frame(
    image_ids: pd.Series,
    clean: np.ndarray,
    noisy: np.ndarray,
    injected: np.ndarray,
    direction: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
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
    expected = clean.size
    if len(frame) != expected or frame[["image_id", "label_name"]].duplicated().any():
        raise RuntimeError("Private reference keys are incomplete or duplicated")
    if not frame["injected_error"].eq(frame["clean_label"].ne(frame["noisy_label"]).astype(int)).all():
        raise RuntimeError("Private error flags disagree with clean/noisy labels")
    return frame


def _subset_and_corrupt(
    source: pd.DataFrame,
    clean_by_id: pd.DataFrame,
    ids: list[str],
    seed: int,
    sentinel: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = source.set_index("image_id").loc[ids].reset_index()
    clean = clean_by_id.loc[ids, LABELS].to_numpy(dtype=np.int64)
    noisy, injected, direction, counts = make_corruption(clean, seed, SCENARIO)
    blind = selected[["image_id", "image_path"]].copy()
    blind["fold_id"] = -1 if sentinel else selected["fold_id"].astype(int)
    for label_index, label in enumerate(LABELS):
        blind[label] = noisy[:, label_index]
    validate_blind_columns(blind)
    private = private_frame(blind["image_id"], clean, noisy, injected, direction)
    expected_images = SENTINEL_IMAGES if sentinel else ACTION_IMAGES
    expected_errors = SENTINEL_ERRORS if sentinel else ACTION_ERRORS
    if len(blind) != expected_images or int(injected.sum()) != expected_errors:
        raise RuntimeError("Prepared cohort has an incorrect size or corruption count")
    if not sentinel:
        fold_support(blind, N_SPLITS)
    elif not blind["fold_id"].eq(-1).all():
        raise RuntimeError("Sentinel fold ids must be -1")
    return blind, private, counts


def prepare(args: argparse.Namespace) -> None:
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=False)
    seeds = parse_locked_seeds(args.seeds)
    source_roots = [Path(path) for path in args.source_root]
    if len(set(source_roots)) != len(source_roots):
        raise ValueError("--source-root values must be unique")

    first_root, first_prepared = _source_prepared(source_roots, seeds[0])
    first_source, first_clean, _ = _load_archived_source(first_prepared, seeds[0])
    first_clean_by_id = pd.DataFrame(first_clean, index=first_source["image_id"], columns=LABELS)
    action_ids, sentinel_ids = deterministic_split(first_source["image_id"])
    split = pd.DataFrame({
        "image_id": action_ids + sentinel_ids,
        "split": ["action"] * ACTION_IMAGES + ["sentinel"] * SENTINEL_IMAGES,
    }).sort_values("image_id").reset_index(drop=True)
    split_path = output / "split_manifest.csv"
    atomic_write_csv(split, split_path)

    image_index_path = first_root / "image_index.csv"
    image_index = pd.read_csv(image_index_path)
    require_columns(image_index, ["image_id", "image_path", "output_sha256"], "archived image index")
    image_index["image_id"] = image_index["image_id"].astype(str)
    image_index = image_index.set_index("image_id").loc[sorted(first_source["image_id"])].reset_index()
    if len(image_index) != 3_000 or not image_index["image_path"].eq(first_source["image_path"]).all():
        raise ValueError("Archived image index does not align with the source cohort")
    output_image_index = output / "image_index.csv"
    atomic_write_csv(image_index, output_image_index)

    manifest_records: list[dict[str, Any]] = []
    clean_reference_hash: str | None = None
    for seed in seeds:
        source_root, source_prepared = _source_prepared(source_roots, seed)
        source, clean, source_summary = _load_archived_source(source_prepared, seed)
        if source["image_id"].tolist() != first_source["image_id"].tolist():
            raise ValueError(f"Seed {seed} does not share the canonical 3,000-image cohort")
        clean_by_id = pd.DataFrame(clean, index=source["image_id"], columns=LABELS)
        clean_hash = hashlib.sha256(clean_by_id.to_numpy(dtype=np.int8).tobytes()).hexdigest()
        if clean_reference_hash is None:
            clean_reference_hash = clean_hash
        elif clean_hash != clean_reference_hash:
            raise ValueError(f"Seed {seed} clean labels differ from the shared reference")

        action, action_private, action_counts = _subset_and_corrupt(
            source, clean_by_id, action_ids, seed, sentinel=False
        )
        sentinel, sentinel_private, sentinel_counts = _subset_and_corrupt(
            source, clean_by_id, sentinel_ids, seed, sentinel=True
        )
        if set(action["image_id"]) & set(sentinel["image_id"]):
            raise RuntimeError("Action and sentinel cohorts overlap")
        if set(action["image_id"]) | set(sentinel["image_id"]) != set(source["image_id"]):
            raise RuntimeError("Action and sentinel cohorts do not cover the source")

        prepared = output / f"seed_{seed}" / "prepared"
        prepared.mkdir(parents=True, exist_ok=False)
        paths = {
            "blind": prepared / "blind_noisy_cohort.csv",
            "private": prepared / "private_reference.csv",
            "sentinel_blind": prepared / "sentinel_blind_cohort.csv",
            "sentinel_private": prepared / "sentinel_private_reference.csv",
            "counts": prepared / "corruption_counts.csv",
            "sentinel_counts": prepared / "sentinel_corruption_counts.csv",
            "support": prepared / "fold_support.csv",
        }
        atomic_write_csv(action, paths["blind"])
        atomic_write_csv(action_private, paths["private"])
        atomic_write_csv(sentinel, paths["sentinel_blind"])
        atomic_write_csv(sentinel_private, paths["sentinel_private"])
        atomic_write_csv(action_counts, paths["counts"])
        atomic_write_csv(sentinel_counts, paths["sentinel_counts"])
        atomic_write_csv(fold_support(action, N_SPLITS), paths["support"])
        summary = {
            "protocol": PROTOCOL_NAME,
            "seed": seed,
            "scenario_id": SCENARIO["scenario_id"],
            "noise_rate": NOISE_RATE,
            "action_images": ACTION_IMAGES,
            "action_entries": ACTION_ENTRIES,
            "action_errors": ACTION_ERRORS,
            "action_true_quality": 1.0 - NOISE_RATE,
            "sentinel_images": SENTINEL_IMAGES,
            "sentinel_entries": SENTINEL_ENTRIES,
            "sentinel_errors": SENTINEL_ERRORS,
            "sentinel_true_quality": 1.0 - NOISE_RATE,
            "n_splits": N_SPLITS,
            "split_rule": SPLIT_RULE,
            "split_manifest_sha256": sha256_file(split_path),
            "source_root": str(source_root),
            "source_prepared": str(source_prepared),
            "source_prepare_summary_sha256": sha256_file(source_prepared / "prepare_summary.json"),
            "source_blind_sha256": source_summary["blind_cohort_sha256"],
            "source_private_sha256": source_summary["private_reference_sha256"],
            **{f"{key}_sha256": sha256_file(path) for key, path in paths.items()},
        }
        atomic_json(summary, prepared / "prepare_summary.json")
        atomic_write_text(prepared / ".prepare_complete", "complete\n")
        manifest_records.append(
            {
                "seed": seed,
                "prepared_path": str(prepared),
                "blind_sha256": summary["blind_sha256"],
                "private_sha256": summary["private_sha256"],
                "sentinel_blind_sha256": summary["sentinel_blind_sha256"],
                "sentinel_private_sha256": summary["sentinel_private_sha256"],
            }
        )

    manifest_path = output / "prepared_manifest.csv"
    atomic_write_csv(pd.DataFrame(manifest_records).sort_values("seed"), manifest_path)
    root_summary = {
        "protocol": PROTOCOL_NAME,
        "seeds": seeds,
        "action_images": ACTION_IMAGES,
        "sentinel_images": SENTINEL_IMAGES,
        "split_shared_across_seeds": True,
        "split_rule": SPLIT_RULE,
        "split_manifest_sha256": sha256_file(split_path),
        "image_index_sha256": sha256_file(output_image_index),
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "shared_clean_label_matrix_sha256": clean_reference_hash,
        "source_roots": [str(path) for path in source_roots],
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(root_summary, output / "prepare_manifest.json")
    atomic_write_text(output / ".prepare_complete", "complete\n")
    print(json.dumps(root_summary, indent=2), flush=True)


def _validate_score(score_dir: Path, blind_path: Path, sentinel_path: Path, seed: int) -> dict[str, Any]:
    marker = score_dir / ".score_complete"
    summary_path = score_dir / "score_summary.json"
    action_oof_path = score_dir / "action_oof_predictions.csv"
    sentinel_folds_path = score_dir / "sentinel_fold_predictions.csv"
    if not marker.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Completed score output missing: {score_dir}")
    summary = _load_json(summary_path)
    if summary.get("protocol") != SCORE_PROTOCOL_NAME or int(summary.get("seed", -1)) != seed:
        raise RuntimeError(f"Score provenance failed for seed {seed}")
    expected = {
        "action_samples": ACTION_IMAGES,
        "action_entries": ACTION_ENTRIES,
        "sentinel_samples": SENTINEL_IMAGES,
        "sentinel_entries_per_fold_model": SENTINEL_ENTRIES,
        "n_splits": N_SPLITS,
    }
    if any(int(summary.get(key, -1)) != value for key, value in expected.items()):
        raise RuntimeError(f"Score dimensions are not locked for seed {seed}")
    _verify_hash(summary, "action_oof_sha256", action_oof_path)
    _verify_hash(summary, "sentinel_folds_sha256", sentinel_folds_path)
    if summary.get("blind_cohort_sha256") != sha256_file(blind_path):
        raise RuntimeError(f"Score action cohort hash mismatch for seed {seed}")
    if summary.get("sentinel_cohort_sha256") != sha256_file(sentinel_path):
        raise RuntimeError(f"Score sentinel cohort hash mismatch for seed {seed}")
    return summary


def materialize_evidence(args: argparse.Namespace) -> None:
    # The model helper imports torch; keep it lazy so preparation/calibration
    # remain CPU-only and can be validated without loading the training stack.
    from vindr_mobilenet_oof import build_evidence

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    blind_path = Path(args.blind_cohort)
    score_dir = Path(args.score_dir)
    sentinel_path = Path(args.sentinel_cohort)
    blind = pd.read_csv(blind_path)
    require_columns(blind, ["image_id", "fold_id"] + LABELS, "action blind cohort")
    validate_blind_columns(blind)
    blind["image_id"] = blind["image_id"].astype(str)
    if len(blind) != ACTION_IMAGES or blind["image_id"].nunique() != ACTION_IMAGES:
        raise ValueError("Action blind cohort must contain 2,400 unique images")
    if sorted(blind["fold_id"].unique().tolist()) != list(range(N_SPLITS)):
        raise ValueError("Action blind cohort must retain four archived folds")
    summary = _validate_score(score_dir, blind_path, sentinel_path, args.seed)
    action_oof_path = score_dir / "action_oof_predictions.csv"
    score = pd.read_csv(action_oof_path)
    require_columns(score, ["image_id", "fold_id"] + PROBABILITY_COLUMNS, "action OOF predictions")
    score["image_id"] = score["image_id"].astype(str)
    aligned = blind[["image_id", "fold_id"]].merge(
        score[["image_id", "fold_id"]],
        on="image_id",
        suffixes=("_blind", "_score"),
        validate="one_to_one",
    )
    if len(aligned) != ACTION_IMAGES or not aligned["fold_id_blind"].eq(aligned["fold_id_score"]).all():
        raise ValueError("Action OOF predictions do not align with the blind cohort")
    score = score.set_index("image_id").loc[blind["image_id"]].reset_index()
    probabilities = score[PROBABILITY_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Action OOF probabilities are invalid")
    labels = blind[LABELS].to_numpy(dtype=np.int64)
    oof, entries, issue = build_evidence(blind, labels, probabilities)
    oof_path = output / "oof_predictions.csv"
    entries_path = output / "entry_evidence.csv"
    atomic_write_csv(oof, oof_path)
    atomic_write_csv(entries, entries_path)
    run_summary = {
        "protocol": EVIDENCE_PROTOCOL_NAME,
        "outcome_blind": True,
        "seed": int(args.seed),
        "samples": ACTION_IMAGES,
        "entries": ACTION_ENTRIES,
        "n_splits": N_SPLITS,
        "cl_issue_entries": int(issue.sum()),
        "raw_entry_dqs": float(1.0 - issue.mean()),
        "blind_cohort_sha256": sha256_file(blind_path),
        "source_score_summary_sha256": sha256_file(score_dir / "score_summary.json"),
        "source_score_program_sha256": summary.get("program_sha256"),
        "source_score_helper_sha256": summary.get("source_helper_sha256"),
        "source_action_oof_sha256": summary["action_oof_sha256"],
        "source_sentinel_folds_sha256": summary["sentinel_folds_sha256"],
        "oof_predictions_sha256": sha256_file(oof_path),
        "entry_evidence_sha256": sha256_file(entries_path),
        "oof_sha256": sha256_file(oof_path),
        "entries_sha256": sha256_file(entries_path),
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(run_summary, output / "blind_run_summary.json")
    atomic_write_text(output / ".blind_run_complete", "complete\n")
    print(json.dumps(run_summary, indent=2), flush=True)


def _wide_sentinel_entries(frame: pd.DataFrame, fold_id: int) -> pd.DataFrame:
    require_columns(frame, ["image_id", "model_fold_id"] + PROBABILITY_COLUMNS, "sentinel scores")
    selected = frame.loc[frame["model_fold_id"].astype(int).eq(fold_id)].copy()
    selected["image_id"] = selected["image_id"].astype(str)
    if len(selected) != SENTINEL_IMAGES or selected["image_id"].duplicated().any():
        raise ValueError(f"Sentinel predictions for fold {fold_id} are incomplete")
    long = selected.melt(
        id_vars=["image_id", "model_fold_id"],
        value_vars=PROBABILITY_COLUMNS,
        var_name="probability_name",
        value_name="probability",
    )
    long["label_name"] = long["probability_name"].str.removeprefix("probability__")
    return long.drop(columns="probability_name")


def calibrate(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    score_root = Path(args.score_root)
    prepared_root = Path(args.prepared_root)
    seeds = parse_locked_seeds(args.seeds)
    records: list[dict[str, Any]] = []
    score_provenance: list[dict[str, Any]] = []

    for seed in seeds:
        prepared = prepared_root / f"seed_{seed}" / "prepared"
        if not (prepared / ".prepare_complete").is_file():
            raise FileNotFoundError(f"Prepared source missing for seed {seed}")
        prepared_summary_path = prepared / "prepare_summary.json"
        prepared_summary = _load_json(prepared_summary_path)
        if prepared_summary.get("protocol") != PROTOCOL_NAME or int(
            prepared_summary.get("seed", -1)
        ) != seed:
            raise RuntimeError(f"Prepared provenance failed for seed {seed}")
        blind_path = prepared / "blind_noisy_cohort.csv"
        sentinel_path = prepared / "sentinel_blind_cohort.csv"
        sentinel_private_path = prepared / "sentinel_private_reference.csv"
        _verify_hash(prepared_summary, "blind_sha256", blind_path)
        _verify_hash(prepared_summary, "sentinel_blind_sha256", sentinel_path)
        _verify_hash(prepared_summary, "sentinel_private_sha256", sentinel_private_path)
        score_dir = score_root / f"seed_{seed}"
        score_summary = _validate_score(score_dir, blind_path, sentinel_path, seed)
        sentinel_private = pd.read_csv(sentinel_private_path)
        require_columns(
            sentinel_private,
            ["image_id", "label_name", "noisy_label", "injected_error"],
            "sentinel private reference",
        )
        sentinel_private["image_id"] = sentinel_private["image_id"].astype(str)
        if len(sentinel_private) != SENTINEL_ENTRIES or sentinel_private[["image_id", "label_name"]].duplicated().any():
            raise ValueError(f"Sentinel private reference is incomplete for seed {seed}")
        if not sentinel_private["injected_error"].isin([0, 1]).all():
            raise ValueError(f"Sentinel error flags are not binary for seed {seed}")
        if int(sentinel_private["injected_error"].sum()) != SENTINEL_ERRORS:
            raise ValueError(f"Sentinel does not contain exactly 720 errors for seed {seed}")
        sentinel_predictions = pd.read_csv(score_dir / "sentinel_fold_predictions.csv")
        for fold_id in range(N_SPLITS):
            entries = _wide_sentinel_entries(sentinel_predictions, fold_id).merge(
                sentinel_private[["image_id", "label_name", "noisy_label", "injected_error"]],
                on=["image_id", "label_name"],
                how="inner",
                validate="one_to_one",
            )
            if len(entries) != SENTINEL_ENTRIES:
                raise RuntimeError(f"Sentinel calibration join is incomplete for seed {seed}, fold {fold_id}")
            scores = self_confidence(
                entries["noisy_label"].to_numpy(dtype=np.int64),
                entries["probability"].to_numpy(dtype=float),
            )
            correct = 1 - entries["injected_error"].to_numpy(dtype=np.int64)
            threshold, estimated_quality, estimated_count = select_prevalence_matching_threshold(
                scores, correct
            )
            records.append(
                {
                    "seed": seed,
                    "fold_id": fold_id,
                    "threshold": threshold,
                    "calibration_entries": SENTINEL_ENTRIES,
                    "calibration_correct_entries": int(correct.sum()),
                    "calibration_estimated_correct_entries": estimated_count,
                    "calibration_true_quality": float(correct.mean()),
                    "calibration_estimated_quality": estimated_quality,
                    "calibration_absolute_error": abs(estimated_quality - float(correct.mean())),
                    "sentinel_fold_predictions_sha256": score_summary["sentinel_folds_sha256"],
                    "sentinel_private_reference_sha256": sha256_file(sentinel_private_path),
                }
            )
        score_provenance.append(
            {
                "seed": seed,
                "score_summary_sha256": sha256_file(score_dir / "score_summary.json"),
                "action_oof_sha256": score_summary["action_oof_sha256"],
                "sentinel_folds_sha256": score_summary["sentinel_folds_sha256"],
            }
        )

    thresholds = pd.DataFrame(records).sort_values(["seed", "fold_id"]).reset_index(drop=True)
    if len(thresholds) != len(LOCKED_SEEDS) * N_SPLITS or thresholds[["seed", "fold_id"]].duplicated().any():
        raise RuntimeError("Frozen threshold table must contain exactly 32 unique seed/fold rows")
    if not thresholds["calibration_true_quality"].eq(0.80).all():
        raise RuntimeError("Sentinel calibration quality must be exactly 0.80")
    threshold_path = output / "calibration_thresholds_private.csv"
    provenance_path = output / "score_provenance.csv"
    atomic_write_csv(thresholds, threshold_path)
    atomic_write_csv(pd.DataFrame(score_provenance).sort_values("seed"), provenance_path)
    threshold_hash = sha256_file(threshold_path)
    manifest = {
        "protocol": PROTOCOL_NAME,
        "seeds": seeds,
        "threshold_rows": len(thresholds),
        "calibration_design": "one fold-specific global self-confidence threshold per seed and fold model",
        "threshold_target": "known correct-label mass on the independent 600-image sentinel cohort",
        "tie_break": "higher threshold",
        "action_private_reference_accessed": False,
        "thresholds_sha256": threshold_hash,
        "score_provenance_sha256": sha256_file(provenance_path),
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(manifest, output / "calibration_manifest_private.json")
    atomic_write_text(output / ".calibration_frozen", threshold_hash + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-root", action="append", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--seeds", default=",".join(map(str, LOCKED_SEEDS)))
    prepare_parser.set_defaults(func=prepare)

    evidence_parser = subparsers.add_parser("materialize-evidence")
    evidence_parser.add_argument("--blind-cohort", type=Path, required=True)
    evidence_parser.add_argument("--sentinel-cohort", type=Path, required=True)
    evidence_parser.add_argument("--score-dir", type=Path, required=True)
    evidence_parser.add_argument("--output-dir", type=Path, required=True)
    evidence_parser.add_argument("--seed", type=int, required=True)
    evidence_parser.set_defaults(func=materialize_evidence)

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--score-root", type=Path, required=True)
    calibrate_parser.add_argument("--prepared-root", type=Path, required=True)
    calibrate_parser.add_argument("--output-dir", type=Path, required=True)
    calibrate_parser.add_argument("--seeds", default=",".join(map(str, LOCKED_SEEDS)))
    calibrate_parser.set_defaults(func=calibrate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
