#!/usr/bin/env python3
"""Calibrate VinDr DQS on held-out labels and transfer it to action OOF states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from vindr_known_gt_cl_benchmark import (
    LABELS,
    atomic_write_csv,
    atomic_write_text,
    build_inner_split,
    fold_support,
    require_columns,
    sha256_file,
    validate_blind_columns,
)
from vindr_mobilenet_sentinel_oof import ACTION_IMAGES, SENTINEL_IMAGES, train_fold
from vindr_self_optimization_stress import corrupt_from_orders, ordered_candidates


PROTOCOL_NAME = "vindr_dqs_calibration_transfer_v1"
ARCHIVED_EVIDENCE_PROTOCOL = "vindr_full_issue_pool_mobilenet_v1"
N_SPLITS = 4
ACTION_ENTRIES = ACTION_IMAGES * len(LABELS)
SENTINEL_ENTRIES = SENTINEL_IMAGES * len(LABELS)
MATCHED_SENTINEL_NOISE_RATE = 0.30
ARCHIVED_SENTINEL_NOISE_RATE = 0.20
COMPLETE_CASE_LOOPS = (0, 1, 2, 3)
PROBABILITY_COLUMNS = [f"probability__{label}" for label in LABELS]


def self_confidence(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape:
        raise ValueError("Labels and probabilities must have identical shapes")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary")
    if not np.isfinite(probabilities).all() or not (
        (probabilities >= 0.0) & (probabilities <= 1.0)
    ).all():
        raise ValueError("Probabilities must be finite and lie in [0, 1]")
    return np.where(labels == 1, probabilities, 1.0 - probabilities)


def select_prevalence_matching_threshold(
    scores: np.ndarray, correct: np.ndarray
) -> tuple[float, float, int]:
    """Match the estimated correct mass; this is not a correctness classifier."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=np.int64)
    if scores.ndim != 1 or correct.ndim != 1 or len(scores) != len(correct):
        raise ValueError("Scores and correctness targets must be aligned vectors")
    if len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Scores must be non-empty and finite")
    if not np.isin(correct, [0, 1]).all():
        raise ValueError("Correctness targets must be binary")

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


def _parse_seeds(text: str) -> list[int]:
    seeds = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique integer seeds")
    return seeds


def _validate_score_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, ...]:
    blind = pd.read_csv(args.blind_cohort)
    sentinel = pd.read_csv(args.sentinel_cohort)
    split_reference = pd.read_csv(args.split_reference_cohort)
    image_index = pd.read_csv(args.image_index)
    for frame, name in (
        (blind, "action cohort"),
        (sentinel, "sentinel cohort"),
        (split_reference, "split reference"),
    ):
        require_columns(frame, ["image_id", "image_path", "fold_id"] + LABELS, name)
        validate_blind_columns(frame)
        frame["image_id"] = frame["image_id"].astype(str)
    require_columns(image_index, ["image_id", "image_path", "output_sha256"], "image index")
    image_index["image_id"] = image_index["image_id"].astype(str)
    if len(blind) != ACTION_IMAGES or blind["image_id"].nunique() != ACTION_IMAGES:
        raise ValueError("Action cohort must contain 2,400 unique images")
    if len(sentinel) != SENTINEL_IMAGES or sentinel["image_id"].nunique() != SENTINEL_IMAGES:
        raise ValueError("Sentinel cohort must contain 600 unique images")
    action_ids = set(blind["image_id"])
    sentinel_ids = set(sentinel["image_id"])
    if action_ids & sentinel_ids:
        raise ValueError("Sentinel images overlap the action cohort")
    if action_ids | sentinel_ids != set(image_index["image_id"]):
        raise ValueError("Action and sentinel cohorts do not cover the locked image index")
    if not sentinel["fold_id"].eq(-1).all():
        raise ValueError("Sentinel fold ids must all be -1")
    if sorted(blind["fold_id"].unique()) != list(range(args.n_splits)):
        raise ValueError("Action fold ids do not match --n-splits")
    aligned = blind[["image_id", "fold_id"]].merge(
        split_reference[["image_id", "fold_id"]],
        on="image_id",
        suffixes=("_current", "_reference"),
        validate="one_to_one",
    )
    if len(aligned) != ACTION_IMAGES or not aligned["fold_id_current"].eq(
        aligned["fold_id_reference"]
    ).all():
        raise ValueError("Current action cohort does not align with locked folds")
    image_root = Path(args.image_root)
    for frame in (blind, sentinel):
        if not frame["image_path"].map(lambda value: (image_root / value).is_file()).all():
            raise FileNotFoundError("At least one locked VinDr PNG is missing")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal fold-matched scoring requires CUDA")
    return blind, sentinel, split_reference, image_index


def _source_helper_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "vindr_mobilenet_sentinel_oof.py",
        "vindr_mobilenet_oof.py",
        "vindr_known_gt_cl_benchmark.py",
        "cxr_real_noise_validation_smoke.py",
        "cxr_real_full_train_eval_cleanlab_xrv12.py",
    )
    return {name: sha256_file(root / name) for name in names}


def score(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    blind, sentinel, split_reference, image_index = _validate_score_inputs(args)
    labels = blind[LABELS].to_numpy(dtype=np.int64)
    sentinel_labels = sentinel[LABELS].to_numpy(dtype=np.int64)
    split_reference = split_reference.set_index("image_id").loc[blind["image_id"]].reset_index()
    split_labels = split_reference[LABELS].to_numpy(dtype=np.int64)
    folds = blind["fold_id"].to_numpy(dtype=np.int64)
    oof = np.full(labels.shape, np.nan, dtype=np.float32)
    sentinel_fold_arrays: list[np.ndarray] = []
    sentinel_fold_frames: list[pd.DataFrame] = []
    training_records: list[dict[str, Any]] = []
    best_epochs: list[int] = []

    for fold_id in range(args.n_splits):
        outer_train = np.flatnonzero(folds != fold_id)
        outer_validation = np.flatnonzero(folds == fold_id)
        inner_train, inner_validation, inner_seed = build_inner_split(
            outer_train, split_labels, seed=args.seed * 100 + fold_id
        )
        outer_probability, sentinel_probability, records, best_epoch, best_loss = train_fold(
            blind,
            labels,
            sentinel,
            sentinel_labels,
            inner_train,
            inner_validation,
            outer_validation,
            Path(args.image_root),
            seed=args.seed * 100 + fold_id,
            args=args,
        )
        oof[outer_validation] = outer_probability
        sentinel_fold_arrays.append(sentinel_probability)
        sentinel_frame = sentinel[["image_id"]].copy()
        sentinel_frame.insert(1, "model_fold_id", fold_id)
        for label_index, column in enumerate(PROBABILITY_COLUMNS):
            sentinel_frame[column] = sentinel_probability[:, label_index]
        sentinel_fold_frames.append(sentinel_frame)
        best_epochs.append(int(best_epoch))
        for record in records:
            training_records.append(
                {
                    **record,
                    "fold_id": fold_id,
                    "inner_train_samples": int(len(inner_train)),
                    "inner_validation_samples": int(len(inner_validation)),
                    "outer_validation_samples": int(len(outer_validation)),
                    "sentinel_samples": SENTINEL_IMAGES,
                    "inner_split_seed": int(inner_seed),
                    "best_epoch": int(best_epoch),
                    "best_inner_validation_loss": float(best_loss),
                }
            )
        print(f"[score] fold {fold_id + 1}/{args.n_splits} complete", flush=True)

    sentinel_mean = np.mean(np.stack(sentinel_fold_arrays), axis=0).astype(np.float32)
    if not np.isfinite(oof).all() or not np.isfinite(sentinel_mean).all():
        raise RuntimeError("Action or sentinel probability matrix is incomplete")
    action_oof = blind[["image_id", "fold_id"]].copy()
    sentinel_mean_frame = sentinel[["image_id", "fold_id"]].copy()
    for label_index, column in enumerate(PROBABILITY_COLUMNS):
        action_oof[column] = oof[:, label_index]
        sentinel_mean_frame[column] = sentinel_mean[:, label_index]
    sentinel_folds = pd.concat(sentinel_fold_frames, ignore_index=True)
    if len(sentinel_folds) != SENTINEL_IMAGES * args.n_splits:
        raise RuntimeError("Fold-specific sentinel prediction count is incomplete")

    paths = {
        "action_oof": output / "action_oof_predictions.csv",
        "sentinel_folds": output / "sentinel_fold_predictions.csv",
        "sentinel_mean": output / "sentinel_mean_predictions.csv",
        "history": output / "training_history.csv",
        "support": output / "fold_support.csv",
    }
    atomic_write_csv(action_oof, paths["action_oof"])
    atomic_write_csv(sentinel_folds, paths["sentinel_folds"])
    atomic_write_csv(sentinel_mean_frame, paths["sentinel_mean"])
    atomic_write_csv(pd.DataFrame(training_records), paths["history"])
    atomic_write_csv(fold_support(blind, args.n_splits), paths["support"])
    summary = {
        "protocol": PROTOCOL_NAME,
        "outcome_blind": True,
        "seed": int(args.seed),
        "action_samples": ACTION_IMAGES,
        "action_entries": ACTION_ENTRIES,
        "sentinel_samples": SENTINEL_IMAGES,
        "sentinel_entries_per_fold_model": SENTINEL_ENTRIES,
        "n_splits": int(args.n_splits),
        "sentinel_prediction_aggregation": "fold-specific plus archived-compatible mean",
        "best_epochs": best_epochs,
        "epochs_max": int(args.epochs),
        "early_stopping_patience": int(args.early_stopping_patience),
        "learning_rate": float(args.learning_rate),
        "batch_size": int(args.batch_size),
        "blind_cohort_sha256": sha256_file(Path(args.blind_cohort)),
        "sentinel_cohort_sha256": sha256_file(Path(args.sentinel_cohort)),
        "split_reference_sha256": sha256_file(Path(args.split_reference_cohort)),
        "image_index_sha256": sha256_file(Path(args.image_index)),
        **{f"{name}_sha256": sha256_file(path) for name, path in paths.items()},
        "source_helper_sha256": _source_helper_hashes(),
        "program_sha256": sha256_file(Path(__file__)),
        "cuda_device": torch.cuda.get_device_name(0),
    }
    atomic_write_text(output / "score_summary.json", json.dumps(summary, indent=2))
    atomic_write_text(output / ".score_complete", "complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def _wide_probabilities_to_entries(frame: pd.DataFrame, fold_id: int) -> pd.DataFrame:
    require_columns(frame, ["image_id", "model_fold_id"] + PROBABILITY_COLUMNS, "sentinel scores")
    selected = frame.loc[frame["model_fold_id"].eq(fold_id)].copy()
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


def _matrix_to_private_entries(
    image_ids: pd.Series,
    clean: np.ndarray,
    noisy: np.ndarray,
    injected: np.ndarray,
    direction: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row_index, image_id in enumerate(image_ids.astype(str)):
        for label_index, label_name in enumerate(LABELS):
            records.append(
                {
                    "seed": int(seed),
                    "image_id": image_id,
                    "label_name": label_name,
                    "clean_label": int(clean[row_index, label_index]),
                    "noisy_label": int(noisy[row_index, label_index]),
                    "injected_error": int(injected[row_index, label_index]),
                    "flip_direction": str(direction[row_index, label_index]),
                }
            )
    frame = pd.DataFrame(records)
    if len(frame) != SENTINEL_ENTRIES or frame[["image_id", "label_name"]].duplicated().any():
        raise RuntimeError("Reconstructed sentinel entries are incomplete")
    expected_error = frame["clean_label"].ne(frame["noisy_label"]).astype(int)
    if not frame["injected_error"].eq(expected_error).all():
        raise RuntimeError("Reconstructed sentinel error flags are inconsistent")
    return frame


def reconstruct_matched_sentinel(
    archived_private: pd.DataFrame,
    sentinel_image_ids: pd.Series,
    hardness: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_columns(
        archived_private,
        ["image_id", "label_name", "clean_label", "noisy_label", "injected_error", "flip_direction"],
        "archived sentinel private reference",
    )
    archived = archived_private.copy()
    archived["image_id"] = archived["image_id"].astype(str)
    sentinel_ids = sentinel_image_ids.astype(str).reset_index(drop=True)
    if len(archived) != SENTINEL_ENTRIES or archived[["image_id", "label_name"]].duplicated().any():
        raise ValueError("Archived sentinel private reference is incomplete")
    archived_error = archived["clean_label"].ne(archived["noisy_label"]).astype(int)
    if not archived["injected_error"].astype(int).eq(archived_error).all():
        raise ValueError("Archived sentinel error flags disagree with clean/noisy labels")
    clean_wide = (
        archived.pivot(index="image_id", columns="label_name", values="clean_label")
        .reindex(index=sentinel_ids, columns=LABELS)
    )
    if clean_wide.isna().any().any():
        raise ValueError("Sentinel clean labels do not align with the locked image order")
    clean = clean_wide.to_numpy(dtype=np.int64)
    hard = hardness.copy()
    require_columns(hard, ["image_id", "label_name", "label_quality_self_confidence"], "hardness")
    hard["image_id"] = hard["image_id"].astype(str)
    hard = hard.loc[hard["image_id"].isin(set(sentinel_ids))]
    if len(hard) != SENTINEL_ENTRIES or hard[["image_id", "label_name"]].duplicated().any():
        raise ValueError("Hardness evidence does not cover the sentinel entries exactly")
    orders = ordered_candidates(clean, sentinel_ids, hard, seed, "hard")
    noisy20, injected20, direction20, _ = corrupt_from_orders(
        clean, orders, ARCHIVED_SENTINEL_NOISE_RATE
    )
    recreated20 = _matrix_to_private_entries(
        sentinel_ids, clean, noisy20, injected20, direction20, seed
    ).drop(columns="seed")
    archived_sorted = archived.sort_values(["image_id", "label_name"]).reset_index(drop=True)
    recreated_sorted = recreated20.sort_values(["image_id", "label_name"]).reset_index(drop=True)
    compare_columns = ["clean_label", "noisy_label", "injected_error", "flip_direction"]
    if not archived_sorted[["image_id", "label_name"]].equals(
        recreated_sorted[["image_id", "label_name"]]
    ) or not archived_sorted[compare_columns].astype(str).equals(
        recreated_sorted[compare_columns].astype(str)
    ):
        raise RuntimeError("The reconstructed hard-20% sentinel state does not match the archive")

    noisy30, injected30, direction30, counts30 = corrupt_from_orders(
        clean, orders, MATCHED_SENTINEL_NOISE_RATE
    )
    matched = _matrix_to_private_entries(
        sentinel_ids, clean, noisy30, injected30, direction30, seed
    )
    expected_errors = int(round(MATCHED_SENTINEL_NOISE_RATE * SENTINEL_ENTRIES))
    if int(matched["injected_error"].sum()) != expected_errors:
        raise RuntimeError("Matched sentinel corruption has the wrong error count")
    summary = {
        "seed": int(seed),
        "archived_r20_exact_reproduction": True,
        "matched_noise_rate": MATCHED_SENTINEL_NOISE_RATE,
        "matched_entries": SENTINEL_ENTRIES,
        "matched_errors": expected_errors,
        "matched_true_quality": 1.0 - MATCHED_SENTINEL_NOISE_RATE,
        "per_label_errors": {
            str(row.label_name): int(row.injected_errors) for row in counts30.itertuples(index=False)
        },
    }
    return matched, summary


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_internal_sha(summary: dict[str, Any], key: str, path: Path) -> None:
    expected = summary.get(key)
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise RuntimeError(f"Hash verification failed for {path}")


def _verify_exact_replay(score_dir: Path, archived_dir: Path, seed: int) -> dict[str, Any]:
    if not (score_dir / ".score_complete").is_file():
        raise FileNotFoundError(f"Completed score output missing for seed {seed}")
    if not (archived_dir / ".blind_run_complete").is_file():
        raise FileNotFoundError(f"Archived Round-0 marker missing for seed {seed}")
    score_summary = _load_json(score_dir / "score_summary.json")
    archived_summary = _load_json(archived_dir / "blind_run_summary.json")
    if score_summary.get("protocol") != PROTOCOL_NAME or int(score_summary.get("seed", -1)) != seed:
        raise RuntimeError(f"Score provenance failed for seed {seed}")
    if archived_summary.get("protocol") != ARCHIVED_EVIDENCE_PROTOCOL or int(
        archived_summary.get("seed", -1)
    ) != seed:
        raise RuntimeError(f"Archived evidence provenance failed for seed {seed}")
    if score_summary.get("best_epochs") != archived_summary.get("best_epochs"):
        raise RuntimeError(f"Best epochs did not reproduce for seed {seed}")
    expected_score_settings = {
        "n_splits": N_SPLITS,
        "epochs_max": 50,
        "early_stopping_patience": 8,
        "batch_size": 32,
    }
    if any(score_summary.get(key) != value for key, value in expected_score_settings.items()):
        raise RuntimeError(f"Score settings are not locked for seed {seed}")
    if abs(float(score_summary.get("learning_rate", np.nan)) - 0.001) > 1e-15:
        raise RuntimeError(f"Score learning rate is not locked for seed {seed}")
    if "A16" not in str(score_summary.get("cuda_device", "")):
        raise RuntimeError(f"Score was not produced on an A16 for seed {seed}")
    if score_summary.get("program_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError(f"Score program hash is stale for seed {seed}")
    if score_summary.get("source_helper_sha256") != _source_helper_hashes():
        raise RuntimeError(f"Score helper hashes are stale for seed {seed}")
    input_pairs = (
        ("blind_cohort_sha256", "blind_cohort_sha256"),
        ("sentinel_cohort_sha256", "sentinel_cohort_sha256"),
        ("split_reference_sha256", "split_reference_sha256"),
        ("image_index_sha256", "image_index_sha256"),
    )
    if any(score_summary.get(new) != archived_summary.get(old) for new, old in input_pairs):
        raise RuntimeError(f"Score inputs do not match the archive for seed {seed}")
    score_paths = {
        "action_oof_sha256": score_dir / "action_oof_predictions.csv",
        "sentinel_folds_sha256": score_dir / "sentinel_fold_predictions.csv",
        "sentinel_mean_sha256": score_dir / "sentinel_mean_predictions.csv",
        "history_sha256": score_dir / "training_history.csv",
        "support_sha256": score_dir / "fold_support.csv",
    }
    for key, path in score_paths.items():
        _require_internal_sha(score_summary, key, path)
    sentinel_folds = pd.read_csv(score_paths["sentinel_folds_sha256"])
    require_columns(
        sentinel_folds,
        ["image_id", "model_fold_id"] + PROBABILITY_COLUMNS,
        "fold-specific sentinel predictions",
    )
    if len(sentinel_folds) != SENTINEL_IMAGES * N_SPLITS:
        raise RuntimeError(f"Fold-specific sentinel predictions are incomplete for seed {seed}")
    if sorted(sentinel_folds["model_fold_id"].unique()) != list(range(N_SPLITS)):
        raise RuntimeError(f"Fold-specific sentinel model ids are incomplete for seed {seed}")
    if sentinel_folds[["model_fold_id", "image_id"]].duplicated().any():
        raise RuntimeError(f"Fold-specific sentinel identities are duplicated for seed {seed}")
    sentinel_values = sentinel_folds[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(sentinel_values).all() or not (
        (sentinel_values >= 0.0) & (sentinel_values <= 1.0)
    ).all():
        raise RuntimeError(f"Fold-specific sentinel probabilities are invalid for seed {seed}")

    checks = (
        (score_dir / "action_oof_predictions.csv", archived_dir / "oof_predictions.csv", "oof_sha256"),
        (
            score_dir / "sentinel_mean_predictions.csv",
            archived_dir / "sentinel_predictions.csv",
            "sentinel_oof_sha256",
        ),
        (score_dir / "training_history.csv", archived_dir / "training_history.csv", "history_sha256"),
        (score_dir / "fold_support.csv", archived_dir / "fold_support.csv", "support_sha256"),
    )
    record: dict[str, Any] = {"seed": seed, "best_epochs_exact": True}
    for new_path, archived_path, archived_key in checks:
        _require_internal_sha(archived_summary, archived_key, archived_path)
        new_hash = sha256_file(new_path)
        archived_hash = sha256_file(archived_path)
        if new_hash != archived_hash:
            detail = ""
            if new_path.name in {"action_oof_predictions.csv", "sentinel_mean_predictions.csv"}:
                new = pd.read_csv(new_path)
                old = pd.read_csv(archived_path)
                if list(new.columns) == list(old.columns) and len(new) == len(old):
                    detail = f"; max_abs_diff={np.max(np.abs(new[PROBABILITY_COLUMNS].to_numpy(float) - old[PROBABILITY_COLUMNS].to_numpy(float))):.9g}"
            raise RuntimeError(f"Exact replay failed for seed {seed}: {new_path.name}{detail}")
        record[f"{new_path.stem}_sha256"] = new_hash
        record[f"{new_path.stem}_exact"] = True
    return record


def calibrate(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    score_root = Path(args.score_root)
    prepared_root = Path(args.prepared_root)
    trajectory_root = Path(args.trajectory_root)
    hardness_path = Path(args.hardness_evidence)
    seeds = _parse_seeds(args.seeds)
    hardness = pd.read_csv(
        hardness_path,
        usecols=["image_id", "label_name", "label_quality_self_confidence"],
    )
    threshold_records: list[dict[str, Any]] = []
    reproduction_records: list[dict[str, Any]] = []
    matched_records: list[pd.DataFrame] = []
    reconstruction_summaries: list[dict[str, Any]] = []

    for seed in seeds:
        score_dir = score_root / f"seed_{seed}"
        archived_dir = trajectory_root / "loop0_evidence" / f"seed_{seed}"
        prepared = prepared_root / f"seed_{seed}" / "prepared"
        reproduction_records.append(_verify_exact_replay(score_dir, archived_dir, seed))
        if not (prepared / ".prepare_complete").is_file():
            raise FileNotFoundError(f"Prepared sentinel marker missing for seed {seed}")
        sentinel_cohort = pd.read_csv(prepared / "sentinel_blind_cohort.csv")
        sentinel_private = pd.read_csv(prepared / "sentinel_private_reference.csv")
        sentinel_cohort["image_id"] = sentinel_cohort["image_id"].astype(str)
        matched, reconstruction = reconstruct_matched_sentinel(
            sentinel_private, sentinel_cohort["image_id"], hardness, seed
        )
        matched_records.append(matched)
        reconstruction_summaries.append(reconstruction)
        sentinel_predictions = pd.read_csv(score_dir / "sentinel_fold_predictions.csv")
        for fold_id in range(N_SPLITS):
            entries = _wide_probabilities_to_entries(sentinel_predictions, fold_id).merge(
                matched[["image_id", "label_name", "noisy_label", "injected_error"]],
                on=["image_id", "label_name"],
                how="inner",
                validate="one_to_one",
            )
            if len(entries) != SENTINEL_ENTRIES:
                raise RuntimeError("Sentinel calibration join is incomplete")
            scores = self_confidence(
                entries["noisy_label"].to_numpy(dtype=int),
                entries["probability"].to_numpy(dtype=float),
            )
            correct = 1 - entries["injected_error"].to_numpy(dtype=int)
            threshold, estimated_quality, estimated_count = select_prevalence_matching_threshold(
                scores, correct
            )
            threshold_records.append(
                {
                    "seed": seed,
                    "fold_id": fold_id,
                    "threshold": threshold,
                    "calibration_entries": len(entries),
                    "calibration_correct_entries": int(correct.sum()),
                    "calibration_estimated_correct_entries": estimated_count,
                    "calibration_true_quality": float(correct.mean()),
                    "calibration_estimated_quality": estimated_quality,
                    "calibration_absolute_error": abs(estimated_quality - float(correct.mean())),
                }
            )

    thresholds = pd.DataFrame(threshold_records).sort_values(["seed", "fold_id"])
    if len(thresholds) != len(seeds) * N_SPLITS or thresholds[["seed", "fold_id"]].duplicated().any():
        raise RuntimeError("The frozen threshold table is incomplete")
    if not thresholds["calibration_true_quality"].eq(0.70).all():
        raise RuntimeError("Matched sentinel calibration quality is not 0.70")
    reproduction = pd.DataFrame(reproduction_records).sort_values("seed")
    matched_frame = pd.concat(matched_records, ignore_index=True).sort_values(
        ["seed", "image_id", "label_name"]
    )
    threshold_path = output / "calibration_thresholds_private.csv"
    reproduction_path = output / "round0_exact_reproduction.csv"
    matched_path = output / "matched_sentinel_reference_private.csv"
    atomic_write_csv(thresholds, threshold_path)
    atomic_write_csv(reproduction, reproduction_path)
    atomic_write_csv(matched_frame, matched_path)
    threshold_hash = sha256_file(threshold_path)
    manifest = {
        "protocol": PROTOCOL_NAME,
        "seeds": seeds,
        "calibration_design": "fold-specific global thresholds on hard-r30 matched sentinel labels",
        "threshold_target": "total known correct-label mass, not entry-level correctness",
        "reconstruction": reconstruction_summaries,
        "hardness_evidence_sha256": sha256_file(hardness_path),
        "thresholds_sha256": threshold_hash,
        "round0_exact_reproduction_sha256": sha256_file(reproduction_path),
        "matched_sentinel_reference_sha256": sha256_file(matched_path),
        "action_reference_accessed": False,
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(output / "calibration_manifest_private.json", json.dumps(manifest, indent=2))
    atomic_write_text(output / ".calibration_frozen", f"{threshold_hash}\n")
    print(json.dumps(manifest, indent=2), flush=True)


def _validate_action_evidence(path: Path, seed: int, loop_id: int) -> pd.DataFrame:
    if not (path / ".blind_run_complete").is_file():
        raise FileNotFoundError(f"Evidence marker missing for seed {seed} Loop {loop_id}")
    summary = _load_json(path / "blind_run_summary.json")
    if summary.get("protocol") != ARCHIVED_EVIDENCE_PROTOCOL or int(summary.get("seed", -1)) != seed:
        raise RuntimeError(f"Evidence provenance failed for seed {seed} Loop {loop_id}")
    evidence_path = path / "entry_evidence.csv"
    _require_internal_sha(summary, "entries_sha256", evidence_path)
    evidence = pd.read_csv(evidence_path)
    require_columns(
        evidence,
        [
            "image_id",
            "fold_id",
            "label_name",
            "noisy_label",
            "oof_probability",
            "label_quality_self_confidence",
        ],
        "action entry evidence",
    )
    if len(evidence) != ACTION_ENTRIES or evidence[["image_id", "label_name"]].duplicated().any():
        raise RuntimeError(f"Seed {seed} Loop {loop_id} action evidence is incomplete")
    if sorted(evidence["fold_id"].unique()) != list(range(N_SPLITS)):
        raise RuntimeError(f"Seed {seed} Loop {loop_id} action folds are incomplete")
    if not np.isin(evidence["noisy_label"].to_numpy(), [0, 1]).all():
        raise RuntimeError(f"Seed {seed} Loop {loop_id} labels are not binary")
    probability = evidence["oof_probability"].to_numpy(dtype=float)
    saved_confidence = evidence["label_quality_self_confidence"].to_numpy(dtype=float)
    if not np.isfinite(saved_confidence).all() or not (
        (saved_confidence >= 0.0) & (saved_confidence <= 1.0)
    ).all():
        raise RuntimeError(f"Seed {seed} Loop {loop_id} self-confidence is invalid")
    probability32 = probability.astype(np.float32)
    saved_confidence32 = saved_confidence.astype(np.float32)
    recalculated32 = np.where(
        evidence["noisy_label"].to_numpy(dtype=int) == 1,
        probability32,
        np.float32(1.0) - probability32,
    ).astype(np.float32)
    if not np.array_equal(recalculated32, saved_confidence32):
        raise RuntimeError(f"Seed {seed} Loop {loop_id} float32 self-confidence mismatch")
    return evidence


def _correlation(frame: pd.DataFrame, method: str) -> float | None:
    value = float(frame[["true_quality", "calibrated_threshold_dqs"]].corr(method=method).iloc[0, 1])
    return value if np.isfinite(value) else None


def evaluate(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    thresholds_path = Path(args.thresholds)
    threshold_hash = sha256_file(thresholds_path)
    if threshold_hash != args.expected_thresholds_sha256:
        raise RuntimeError("Frozen threshold hash does not match --expected-thresholds-sha256")
    thresholds = pd.read_csv(thresholds_path)
    require_columns(thresholds, ["seed", "fold_id", "threshold"], "frozen thresholds")
    seeds = _parse_seeds(args.seeds)
    if len(thresholds) != len(seeds) * N_SPLITS or thresholds[["seed", "fold_id"]].duplicated().any():
        raise RuntimeError("Frozen threshold table is incomplete")
    if set(thresholds["seed"].astype(int)) != set(seeds):
        raise RuntimeError("Frozen threshold seeds do not match --seeds")
    if sorted(thresholds["fold_id"].unique()) != list(range(N_SPLITS)):
        raise RuntimeError("Frozen thresholds do not cover all folds")
    if not thresholds["threshold"].between(0.0, 1.0, inclusive="neither").all():
        raise RuntimeError("Frozen thresholds are outside (0, 1)")

    trajectory_root = Path(args.trajectory_root)
    trajectory_records: list[dict[str, Any]] = []
    for seed in seeds:
        seed_thresholds = thresholds.loc[
            thresholds["seed"].eq(seed), ["fold_id", "threshold"]
        ]
        seed_dir = trajectory_root / f"seed_{seed}"
        if not (seed_dir / ".seed_evaluation_complete").is_file() or not (
            seed_dir / ".worker_complete"
        ).is_file():
            raise FileNotFoundError(f"Completed trajectory markers are missing for seed {seed}")
        trajectory_summary = _load_json(seed_dir / "seed_evaluation_summary.json")
        if trajectory_summary.get("protocol") != "vindr_full_issue_pool_iteration_v1" or int(
            trajectory_summary.get("seed", -1)
        ) != seed:
            raise RuntimeError(f"Trajectory summary provenance failed for seed {seed}")
        known_path = seed_dir / "full_issue_iteration_trajectory_private.csv"
        _require_internal_sha(trajectory_summary, "trajectory_sha256", known_path)
        known = pd.read_csv(known_path)
        require_columns(known, ["seed", "loop", "true_quality", "remaining_errors", "raw_dqs"], "trajectory")
        if known[["seed", "loop"]].duplicated().any() or not known["seed"].eq(seed).all():
            raise RuntimeError(f"Trajectory provenance failed for seed {seed}")
        terminal_loop = int(trajectory_summary.get("loops", -1))
        if terminal_loop < 3 or terminal_loop > 5 or known["loop"].astype(int).tolist() != list(
            range(terminal_loop + 1)
        ):
            raise RuntimeError(f"Trajectory loops are not contiguous or locked for seed {seed}")
        if not set(COMPLETE_CASE_LOOPS).issubset(set(known["loop"].astype(int))):
            raise RuntimeError(f"Seed {seed} lacks the locked Loop 0-3 complete-case trajectory")
        for row in known.sort_values("loop").itertuples(index=False):
            loop_id = int(row.loop)
            evidence_dir = (
                trajectory_root / "loop0_evidence" / f"seed_{seed}"
                if loop_id == 0
                else trajectory_root / f"seed_{seed}" / f"loop_{loop_id:02d}" / "oof_after_action"
            )
            evidence = _validate_action_evidence(evidence_dir, seed, loop_id).merge(
                seed_thresholds, on="fold_id", how="left", validate="many_to_one"
            )
            if evidence["threshold"].isna().any():
                raise RuntimeError("At least one action entry lacks a fold threshold")
            expected_quality = 1.0 - float(row.remaining_errors) / ACTION_ENTRIES
            if abs(expected_quality - float(row.true_quality)) > 1e-12:
                raise RuntimeError(f"True-quality accounting failed for seed {seed} Loop {loop_id}")
            estimate = float(
                (
                    evidence["label_quality_self_confidence"].to_numpy(dtype=float)
                    >= evidence["threshold"].to_numpy(dtype=float)
                ).mean()
            )
            trajectory_records.append(
                {
                    "seed": seed,
                    "loop": loop_id,
                    "true_quality": float(row.true_quality),
                    "calibrated_threshold_dqs": estimate,
                    "absolute_error": abs(estimate - float(row.true_quality)),
                    "archived_cl_nonissue_rate": float(row.raw_dqs),
                }
            )

    trajectory = pd.DataFrame(trajectory_records).sort_values(["seed", "loop"])
    trajectory_path = output / "action_transfer_trajectory_private.csv"
    atomic_write_csv(trajectory, trajectory_path)
    loop_summary = (
        trajectory.groupby("loop", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            true_quality_mean=("true_quality", "mean"),
            true_quality_sd=("true_quality", "std"),
            calibrated_dqs_mean=("calibrated_threshold_dqs", "mean"),
            calibrated_dqs_sd=("calibrated_threshold_dqs", "std"),
            absolute_error_mean=("absolute_error", "mean"),
        )
        .sort_values("loop")
    )
    loop_summary_path = output / "action_transfer_loop_summary_private.csv"
    atomic_write_csv(loop_summary, loop_summary_path)

    per_seed_records: list[dict[str, Any]] = []
    for seed, frame in trajectory.groupby("seed"):
        frame = frame.sort_values("loop")
        true_direction = np.sign(np.diff(frame["true_quality"].to_numpy(dtype=float)))
        estimated_direction = np.sign(
            np.diff(frame["calibrated_threshold_dqs"].to_numpy(dtype=float))
        )
        agreement = int((true_direction == estimated_direction).sum())
        per_seed_records.append(
            {
                "seed": int(seed),
                "states": len(frame),
                "terminal_loop": int(frame.iloc[-1]["loop"]),
                "mean_absolute_error": float(frame["absolute_error"].mean()),
                "loop0_absolute_error": float(frame.loc[frame["loop"].eq(0), "absolute_error"].iloc[0]),
                "terminal_absolute_error": float(frame.iloc[-1]["absolute_error"]),
                "spearman": _correlation(frame, "spearman"),
                "adjacent_direction_agreement": agreement,
                "adjacent_direction_total": len(true_direction),
                "adjacent_direction_agreement_rate": (
                    float(agreement / len(true_direction)) if len(true_direction) else None
                ),
            }
        )
    per_seed = pd.DataFrame(per_seed_records).sort_values("seed")
    per_seed_path = output / "action_transfer_seed_summary_private.csv"
    atomic_write_csv(per_seed, per_seed_path)
    complete = trajectory.loc[trajectory["loop"].isin(COMPLETE_CASE_LOOPS)].copy()
    complete_counts = complete.groupby("loop")["seed"].nunique().to_dict()
    if complete_counts != {loop: len(seeds) for loop in COMPLETE_CASE_LOOPS}:
        raise RuntimeError("Loop 0-3 complete-case counts are not balanced")

    summary = {
        "protocol": PROTOCOL_NAME,
        "seeds": seeds,
        "thresholds_sha256": threshold_hash,
        "primary_seed_weighted_mae": float(per_seed["mean_absolute_error"].mean()),
        "complete_case_loops": list(COMPLETE_CASE_LOOPS),
        "complete_case_seed_weighted_mae": float(
            complete.groupby("seed")["absolute_error"].mean().mean()
        ),
        "loop0_mean_absolute_error": float(
            trajectory.loc[trajectory["loop"].eq(0), "absolute_error"].mean()
        ),
        "mean_seed_terminal_absolute_error": float(per_seed["terminal_absolute_error"].mean()),
        "pooled_state_mae_secondary": float(trajectory["absolute_error"].mean()),
        "pearson_all_states_secondary": _correlation(trajectory, "pearson"),
        "spearman_all_states_secondary": _correlation(trajectory, "spearman"),
        "mean_seed_spearman": float(per_seed["spearman"].dropna().mean()),
        "adjacent_direction_agreement": int(per_seed["adjacent_direction_agreement"].sum()),
        "adjacent_direction_total": int(per_seed["adjacent_direction_total"].sum()),
        "adjacent_direction_agreement_rate": float(
            per_seed["adjacent_direction_agreement"].sum()
            / per_seed["adjacent_direction_total"].sum()
        ),
        "available_seed_counts_by_loop": {
            str(int(row.loop)): int(row.seeds) for row in loop_summary.itertuples(index=False)
        },
        "threshold_min": float(thresholds["threshold"].min()),
        "threshold_median": float(thresholds["threshold"].median()),
        "threshold_max": float(thresholds["threshold"].max()),
        "trajectory_sha256": sha256_file(trajectory_path),
        "loop_summary_sha256": sha256_file(loop_summary_path),
        "seed_summary_sha256": sha256_file(per_seed_path),
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(output / "action_transfer_summary_private.json", json.dumps(summary, indent=2))

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]})
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(
        loop_summary["loop"],
        loop_summary["true_quality_mean"],
        marker="o",
        linewidth=2.0,
        label="Known label quality",
        color="#2F5597",
    )
    ax.plot(
        loop_summary["loop"],
        loop_summary["calibrated_dqs_mean"],
        marker="s",
        linewidth=2.0,
        label="Calibrated DQS",
        color="#C55A11",
    )
    for row in loop_summary.itertuples(index=False):
        ax.annotate(
            f"n={int(row.seeds)}",
            (float(row.loop), max(float(row.true_quality_mean), float(row.calibrated_dqs_mean))),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("Refinement loop")
    ax.set_ylabel("Quality")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "action_transfer_trajectory.png", dpi=300)
    fig.savefig(output / "action_transfer_trajectory.pdf")
    plt.close(fig)
    atomic_write_text(output / ".analysis_complete", "complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--blind-cohort", type=Path, required=True)
    score_parser.add_argument("--sentinel-cohort", type=Path, required=True)
    score_parser.add_argument("--split-reference-cohort", type=Path, required=True)
    score_parser.add_argument("--image-index", type=Path, required=True)
    score_parser.add_argument("--image-root", type=Path, required=True)
    score_parser.add_argument("--output-dir", type=Path, required=True)
    score_parser.add_argument("--seed", type=int, required=True)
    score_parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    score_parser.add_argument("--epochs", type=int, default=50)
    score_parser.add_argument("--early-stopping-patience", type=int, default=8)
    score_parser.add_argument("--learning-rate", type=float, default=1e-3)
    score_parser.add_argument("--batch-size", type=int, default=32)
    score_parser.add_argument("--num-workers", type=int, default=4)
    score_parser.add_argument("--device", choices=["cuda"], default="cuda")
    score_parser.set_defaults(func=score)

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--score-root", type=Path, required=True)
    calibrate_parser.add_argument("--prepared-root", type=Path, required=True)
    calibrate_parser.add_argument("--trajectory-root", type=Path, required=True)
    calibrate_parser.add_argument("--hardness-evidence", type=Path, required=True)
    calibrate_parser.add_argument("--output-dir", type=Path, required=True)
    calibrate_parser.add_argument("--seeds", required=True)
    calibrate_parser.set_defaults(func=calibrate)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--thresholds", type=Path, required=True)
    evaluate_parser.add_argument("--expected-thresholds-sha256", required=True)
    evaluate_parser.add_argument("--trajectory-root", type=Path, required=True)
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--seeds", required=True)
    evaluate_parser.set_defaults(func=evaluate)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.func(arguments)
