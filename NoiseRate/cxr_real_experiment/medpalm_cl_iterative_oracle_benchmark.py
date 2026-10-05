#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from cxr_real_full_train_eval_cleanlab_xrv12 import (
    LABEL_NAMES,
    RealCXRXRV12Dataset,
    aggregate_test_predictions_by_study,
    create_model,
)
from cxr_real_oof_cleanlab_smoke import train_one_fold
from medpalm_cl_detection_benchmark import (
    EXPECTED_BENCHMARK_ENTRIES,
    EXPECTED_BENCHMARK_ISSUES,
    EXPECTED_IMAGE_ROWS,
    EXPECTED_STUDIES,
    build_official_test_pool,
    score_one_estimator,
    sha256_file,
    write_json,
)


PROTOCOL_NAME = "medpalm_dynamic_cl_oracle_v1"
EXPECTED_BENCHMARK_SUBJECTS = 179
DEFAULT_LOOPS = 5
DEFAULT_REVIEW_BUDGET = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dynamic-OOF iterative CL benchmark with selected-only oracle corrections."
    )
    parser.add_argument(
        "mode", choices=["preflight", "initialize", "run-loop", "oracle-update", "evaluate"]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--loop-id", type=int)
    parser.add_argument("--loops", type=int, default=DEFAULT_LOOPS)
    parser.add_argument("--review-budget", type=int, default=DEFAULT_REVIEW_BUDGET)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--chexpert-csv", type=Path)
    parser.add_argument("--split-csv", type=Path)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--benchmark-blinded", type=Path)
    parser.add_argument("--benchmark-reference", type=Path)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--random-replicates", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260803)
    return parser.parse_args()


def require_path(path: Path | None, name: str) -> Path:
    if path is None:
        raise ValueError(f"{name} is required")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def benchmark_blinded(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "entry_key",
        "subject_id",
        "study_id",
        "finding",
        "current_binary_label",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Blinded benchmark is missing columns: {sorted(missing)}")
    if len(frame) != EXPECTED_BENCHMARK_ENTRIES:
        raise ValueError(f"Expected 498 benchmark entries, found {len(frame)}")
    if frame["entry_key"].duplicated().any():
        raise ValueError("Blinded benchmark contains duplicate entry keys")
    if frame["subject_id"].nunique() != EXPECTED_BENCHMARK_SUBJECTS:
        raise ValueError("Unexpected benchmark subject count")
    return frame


def benchmark_reference(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "entry_key",
        "subject_id",
        "study_id",
        "finding",
        "reference_binary_label",
        "expected_binary_action",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Private reference is missing columns: {sorted(missing)}")
    if len(frame) != EXPECTED_BENCHMARK_ENTRIES or frame["entry_key"].duplicated().any():
        raise ValueError("Private reference must contain 498 unique entries")
    if int(frame["expected_binary_action"].eq("relabel").sum()) != EXPECTED_BENCHMARK_ISSUES:
        raise ValueError("Unexpected expert issue count")
    return frame


def state_paths(output_dir: Path) -> dict[str, Path]:
    state = output_dir / "state_private"
    return {
        "dir": state,
        "history": state / "review_history_private.csv",
        "overrides": state / "oracle_overrides_private.csv",
        "marker": state / ".initialized",
    }


def loop_dir(output_dir: Path, loop_id: int) -> Path:
    return output_dir / f"loop_{loop_id:02d}"


def initialize(args: argparse.Namespace) -> None:
    blind_path = require_path(args.benchmark_blinded, "--benchmark-blinded")
    blind = benchmark_blinded(blind_path)
    paths = state_paths(args.output_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": PROTOCOL_NAME,
        "benchmark_entries": len(blind),
        "benchmark_subjects": int(blind["subject_id"].nunique()),
        "benchmark_blinded_sha256": sha256_file(blind_path),
        "loops": args.loops,
        "review_budget": args.review_budget,
        "seed": args.seed,
        "n_splits": args.n_splits,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "early_stopping_patience": args.early_stopping_patience,
        "reference_used": False,
    }
    history_columns = [
        "entry_key",
        "first_review_loop",
        "true_issue",
        "subject_id",
        "study_id",
        "finding",
        "original_binary_label",
        "reference_binary_label",
        "selected_hard_flag",
        "selected_suspicion_percentile",
    ]
    override_columns = [
        "entry_key",
        "subject_id",
        "study_id",
        "finding",
        "reference_binary_label",
        "first_review_loop",
    ]
    if paths["marker"].exists():
        existing_manifest = json.loads(
            (paths["dir"] / "initialization_manifest.json").read_text(encoding="utf-8")
        )
        if existing_manifest != manifest:
            raise ValueError("Existing state was initialized with a different configuration")
        history = pd.read_csv(paths["history"])
        overrides = pd.read_csv(paths["overrides"])
        if list(history.columns) != history_columns or list(overrides.columns) != override_columns:
            raise ValueError("Existing state schema differs from the locked protocol")
        print("[initialize] Existing state is valid; reusing it.")
        return
    atomic_csv(pd.DataFrame(columns=history_columns), paths["history"])
    atomic_csv(pd.DataFrame(columns=override_columns), paths["overrides"])
    atomic_json(manifest, paths["dir"] / "initialization_manifest.json")
    paths["marker"].write_text("complete\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def data_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        image_root=require_path(args.image_root, "--image-root"),
        chexpert_csv=require_path(args.chexpert_csv, "--chexpert-csv"),
        split_csv=require_path(args.split_csv, "--split-csv"),
        metadata_csv=require_path(args.metadata_csv, "--metadata-csv"),
    )


def subject_grouped_splits(rows: pd.DataFrame, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    subjects = np.asarray(sorted(rows["subject_id"].unique()), dtype=np.int64)
    if len(subjects) < n_splits:
        raise ValueError("Fewer subjects than OOF folds")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    output: list[tuple[np.ndarray, np.ndarray]] = []
    subject_values = rows["subject_id"].to_numpy(dtype=np.int64)
    for train_subject_idx, validation_subject_idx in splitter.split(subjects):
        train_subjects = set(subjects[train_subject_idx].tolist())
        validation_subjects = set(subjects[validation_subject_idx].tolist())
        if train_subjects & validation_subjects:
            raise ValueError("Subject overlap across OOF split")
        train_idx = np.flatnonzero(np.isin(subject_values, list(train_subjects)))
        validation_idx = np.flatnonzero(np.isin(subject_values, list(validation_subjects)))
        output.append((train_idx, validation_idx))
    validation_all = np.concatenate([validation for _, validation in output])
    if len(validation_all) != len(rows) or len(np.unique(validation_all)) != len(rows):
        raise ValueError("OOF validation rows are not an exact partition")
    return output


def validate_benchmark_pool(blind: pd.DataFrame, rows: pd.DataFrame) -> None:
    available = set(
        zip(
            rows["subject_id"].astype(int),
            rows["study_id"].astype(int),
        )
    )
    missing = [
        (int(row.subject_id), int(row.study_id), row.finding)
        for row in blind.itertuples(index=False)
        if (int(row.subject_id), int(row.study_id)) not in available
    ]
    if missing:
        raise ValueError(f"Benchmark studies missing from official AP/PA pool: {missing[:10]}")
    unknown = sorted(set(blind["finding"]) - set(LABEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown benchmark findings: {unknown}")


def preflight(args: argparse.Namespace) -> None:
    blind = benchmark_blinded(require_path(args.benchmark_blinded, "--benchmark-blinded"))
    rows, _, _, _ = build_official_test_pool(data_args(args))
    validate_benchmark_pool(blind, rows)
    splits = subject_grouped_splits(rows, args.n_splits, args.seed)
    fold_rows = [int(len(validation)) for _, validation in splits]
    fold_subjects = [int(rows.iloc[validation]["subject_id"].nunique()) for _, validation in splits]
    payload = {
        "protocol": PROTOCOL_NAME,
        "image_rows": len(rows),
        "studies": int(rows["study_id"].nunique()),
        "subjects": int(rows["subject_id"].nunique()),
        "benchmark_entries": len(blind),
        "fold_validation_rows": fold_rows,
        "fold_validation_subjects": fold_subjects,
        "subject_leakage": False,
    }
    if payload["image_rows"] != EXPECTED_IMAGE_ROWS or payload["studies"] != EXPECTED_STUDIES:
        raise ValueError("Official test pool count changed")
    print(json.dumps(payload, indent=2))


def load_state(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = state_paths(output_dir)
    if not paths["marker"].exists():
        raise FileNotFoundError("State initialization marker is missing")
    return pd.read_csv(paths["history"]), pd.read_csv(paths["overrides"])


def apply_oracle_overrides(
    rows: pd.DataFrame,
    raw: np.ndarray,
    binary: np.ndarray,
    valid: np.ndarray,
    overrides: pd.DataFrame,
) -> None:
    for override in overrides.itertuples(index=False):
        label_index = LABEL_NAMES.index(str(override.finding))
        mask = (
            rows["subject_id"].eq(int(override.subject_id))
            & rows["study_id"].eq(int(override.study_id))
        ).to_numpy()
        if not mask.any():
            raise ValueError(f"Override study not found: {override.entry_key}")
        if not valid[mask, label_index].all():
            raise ValueError(f"Override targets an invalid entry: {override.entry_key}")
        value = int(override.reference_binary_label)
        binary[mask, label_index] = value
        raw[mask, label_index] = float(value)


def current_state_digest(binary: np.ndarray, valid: np.ndarray, history_path: Path, override_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(binary.astype(np.float32).tobytes())
    digest.update(valid.astype(np.uint8).tobytes())
    digest.update(history_path.read_bytes())
    digest.update(override_path.read_bytes())
    return digest.hexdigest()


def fold_assignment_frame(rows: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    assignments = np.full(len(rows), -1, dtype=int)
    for fold_number, (_, validation) in enumerate(splits, start=1):
        assignments[validation] = fold_number
    if (assignments < 1).any():
        raise ValueError("Missing OOF fold assignment")
    return pd.DataFrame(
        {
            "pool_row_id": rows["pool_row_id"].to_numpy(dtype=np.int64),
            "subject_id": rows["subject_id"].to_numpy(dtype=np.int64),
            "study_id": rows["study_id"].to_numpy(dtype=np.int64),
            "dicom_id": rows["dicom_id"].astype(str).to_numpy(),
            "fold": assignments,
        }
    )


def run_fold(
    args: argparse.Namespace,
    rows: pd.DataFrame,
    binary: np.ndarray,
    valid: np.ndarray,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    fold_number: int,
) -> np.ndarray:
    fold_seed = int(args.seed * 1000 + fold_number)
    set_seed(fold_seed)
    train_dataset = RealCXRXRV12Dataset(
        rows=rows.iloc[train_idx].reset_index(drop=True),
        image_root=require_path(args.image_root, "--image-root"),
        image_size=224,
        binary_labels=binary[train_idx],
        valid_mask=valid[train_idx],
    )
    validation_dataset = RealCXRXRV12Dataset(
        rows=rows.iloc[validation_idx].reset_index(drop=True),
        image_root=require_path(args.image_root, "--image-root"),
        image_size=224,
        binary_labels=binary[validation_idx],
        valid_mask=valid[validation_idx],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = create_model(
        model_backbone="mobilenet_v3_small_scratch",
        xrv_weights="densenet121-res224-all",
    )
    device = torch.device(args.device)
    probabilities = train_one_fold(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        recover_best_weights=True,
        fold_idx=fold_number,
        n_folds=args.n_splits,
    )
    del model, train_loader, validation_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if probabilities.shape != (len(validation_idx), len(LABEL_NAMES)):
        raise ValueError(f"Unexpected fold probability shape: {probabilities.shape}")
    return probabilities.astype(np.float32)


def run_loop(args: argparse.Namespace) -> None:
    if args.loop_id is None or not 1 <= args.loop_id <= args.loops:
        raise ValueError("--loop-id must be within the configured loop range")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    blind_path = require_path(args.benchmark_blinded, "--benchmark-blinded")
    blind = benchmark_blinded(blind_path)
    directory = loop_dir(args.output_dir, args.loop_id)
    directory.mkdir(parents=True, exist_ok=True)
    completion_marker = directory / ".blind_selection_complete"
    if completion_marker.exists():
        manifest = json.loads((directory / "blind_selection_manifest.json").read_text())
        for name in ["benchmark_scores_blind.csv", "selected_entries_blind.csv"]:
            if sha256_file(directory / name) != manifest[f"{name}_sha256"]:
                raise ValueError(f"Completed blind output hash mismatch: {name}")
        print(f"[loop {args.loop_id}] Blind selection already complete; reusing it.")
        return

    history, overrides = load_state(args.output_dir)
    expected_reviewed = min((args.loop_id - 1) * args.review_budget, EXPECTED_BENCHMARK_ENTRIES)
    if len(history) != expected_reviewed:
        raise ValueError(
            f"Loop {args.loop_id} expected {expected_reviewed} prior reviews, found {len(history)}"
        )

    rows, raw, binary, valid = build_official_test_pool(data_args(args))
    validate_benchmark_pool(blind, rows)
    apply_oracle_overrides(rows, raw, binary, valid, overrides)
    paths = state_paths(args.output_dir)
    state_digest = current_state_digest(binary, valid, paths["history"], paths["overrides"])
    pre_manifest_path = directory / "blind_pre_manifest.json"
    pre_manifest = {
        "protocol": PROTOCOL_NAME,
        "loop": args.loop_id,
        "seed": args.seed,
        "n_splits": args.n_splits,
        "epochs": args.epochs,
        "reviewed_before": len(history),
        "oracle_overrides_before": len(overrides),
        "current_state_sha256": state_digest,
        "benchmark_blinded_sha256": sha256_file(blind_path),
        "private_reference_used": False,
    }
    if pre_manifest_path.exists():
        existing = json.loads(pre_manifest_path.read_text())
        if existing != pre_manifest:
            raise ValueError("Loop state changed after fold computation began")
    else:
        atomic_json(pre_manifest, pre_manifest_path)

    splits = subject_grouped_splits(rows, args.n_splits, args.seed)
    assignments = fold_assignment_frame(rows, splits)
    assignment_path = directory / "subject_grouped_fold_assignments.csv"
    if assignment_path.exists():
        existing = pd.read_csv(assignment_path)
        if not existing.equals(assignments):
            raise ValueError("Fold assignments changed during resume")
    else:
        atomic_csv(assignments, assignment_path)

    oof_probabilities = np.full((len(rows), len(LABEL_NAMES)), np.nan, dtype=np.float32)
    for fold_number, (train_idx, validation_idx) in enumerate(splits, start=1):
        fold_path = directory / f"fold_{fold_number}_probabilities.npz"
        fold_marker = directory / f".fold_{fold_number}_complete"
        if fold_marker.exists() and fold_path.exists():
            payload = np.load(fold_path)
            saved_idx = payload["validation_indices"]
            probabilities = payload["probabilities"]
            if not np.array_equal(saved_idx, validation_idx):
                raise ValueError(f"Fold {fold_number} validation indices changed")
            if probabilities.shape != (len(validation_idx), len(LABEL_NAMES)):
                raise ValueError(f"Fold {fold_number} saved probabilities have wrong shape")
            print(f"[loop {args.loop_id}] reusing completed fold {fold_number}")
        else:
            probabilities = run_fold(
                args,
                rows,
                binary,
                valid,
                train_idx,
                validation_idx,
                fold_number,
            )
            temporary = directory / f"fold_{fold_number}_probabilities.tmp.npz"
            np.savez_compressed(
                temporary,
                validation_indices=validation_idx,
                probabilities=probabilities,
            )
            temporary.replace(fold_path)
            fold_marker.write_text("complete\n", encoding="utf-8")
        oof_probabilities[validation_idx] = probabilities
    if not np.isfinite(oof_probabilities).all():
        raise ValueError("OOF probability matrix is incomplete")

    image_prediction_path = directory / "oof_image_probabilities.csv"
    image_predictions = pd.DataFrame(oof_probabilities, columns=LABEL_NAMES)
    image_predictions.insert(0, "dicom_id", rows["dicom_id"].astype(str).to_numpy())
    image_predictions.insert(0, "study_id", rows["study_id"].to_numpy(dtype=np.int64))
    image_predictions.insert(0, "subject_id", rows["subject_id"].to_numpy(dtype=np.int64))
    atomic_csv(image_predictions, image_prediction_path)

    study_df, study_raw, study_binary, study_valid, study_probability = (
        aggregate_test_predictions_by_study(
            rows=rows,
            raw_labels=raw,
            y_binary=binary,
            valid_mask=valid,
            y_prob=oof_probabilities,
            agg_method="max",
        )
    )
    study_frame = study_df[["study_id", "subject_id", "n_images_in_study"]].copy()
    for label_index, label in enumerate(LABEL_NAMES):
        study_frame[f"raw::{label}"] = study_raw[:, label_index]
        study_frame[f"binary::{label}"] = study_binary[:, label_index]
        study_frame[f"valid::{label}"] = study_valid[:, label_index].astype(int)
    scored = score_one_estimator(
        study_frame,
        estimator=f"dynamic_seed_{args.seed}_loop_{args.loop_id}",
        probabilities=study_probability,
    )
    full_score_path = directory / "full_pool_cl_scores_blind.csv"
    atomic_csv(scored, full_score_path)

    benchmark_scores = scored.merge(
        blind[
            ["entry_key", "subject_id", "study_id", "finding", "current_binary_label"]
        ],
        on=["entry_key", "subject_id", "study_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_original"),
    )
    if len(benchmark_scores) != EXPECTED_BENCHMARK_ENTRIES:
        raise ValueError(f"Expected 498 scored benchmark entries, found {len(benchmark_scores)}")
    if not benchmark_scores["label_name"].eq(benchmark_scores["finding"]).all():
        raise ValueError("Finding mismatch after benchmark join")
    history_keys = set(history["entry_key"].astype(str))
    benchmark_scores["reviewed_before_loop"] = benchmark_scores["entry_key"].isin(history_keys)
    candidate = benchmark_scores[~benchmark_scores["reviewed_before_loop"]].copy()
    selected_count = min(args.review_budget, len(candidate))
    if selected_count <= 0:
        raise ValueError("No unreviewed benchmark entries remain")
    selected = candidate.sort_values(
        ["suspicion_percentile_self", "entry_key"],
        ascending=[False, True],
        kind="stable",
    ).head(selected_count)
    if selected["entry_key"].isin(history_keys).any() or selected["entry_key"].duplicated().any():
        raise ValueError("Blind selection is not globally unseen and unique")
    benchmark_score_path = directory / "benchmark_scores_blind.csv"
    selected_path = directory / "selected_entries_blind.csv"
    atomic_csv(benchmark_scores, benchmark_score_path)
    atomic_csv(selected, selected_path)
    manifest = {
        **pre_manifest,
        "full_pool_scored_entries": len(scored),
        "benchmark_scored_entries": len(benchmark_scores),
        "unreviewed_candidates": len(candidate),
        "selected_entries": len(selected),
        "selected_hard_flags": int(selected["cl_hard_issue"].sum()),
        "full_pool_cl_scores_blind.csv_sha256": sha256_file(full_score_path),
        "benchmark_scores_blind.csv_sha256": sha256_file(benchmark_score_path),
        "selected_entries_blind.csv_sha256": sha256_file(selected_path),
        "oof_image_probabilities.csv_sha256": sha256_file(image_prediction_path),
        "private_reference_used": False,
    }
    atomic_json(manifest, directory / "blind_selection_manifest.json")
    completion_marker.write_text("complete\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def oracle_update(args: argparse.Namespace) -> None:
    if args.loop_id is None or not 1 <= args.loop_id <= args.loops:
        raise ValueError("--loop-id must be within the configured loop range")
    directory = loop_dir(args.output_dir, args.loop_id)
    if not (directory / ".blind_selection_complete").exists():
        raise FileNotFoundError("Blind selection completion marker is missing")
    manifest = json.loads((directory / "blind_selection_manifest.json").read_text())
    selected_path = directory / "selected_entries_blind.csv"
    if sha256_file(selected_path) != manifest["selected_entries_blind.csv_sha256"]:
        raise ValueError("Frozen blind selection hash mismatch")
    selected = pd.read_csv(selected_path)
    history, overrides = load_state(args.output_dir)
    selected_keys = set(selected["entry_key"].astype(str))
    overlap = selected_keys & set(history["entry_key"].astype(str))
    completion_marker = directory / ".oracle_update_complete"
    if completion_marker.exists():
        if overlap != selected_keys:
            raise ValueError("Completed oracle update is absent from cumulative history")
        print(f"[loop {args.loop_id}] Oracle update already complete; reusing it.")
        return
    if overlap:
        raise ValueError("Partial oracle state transaction detected")

    reference_path = require_path(args.benchmark_reference, "--benchmark-reference")
    reference = benchmark_reference(reference_path)
    private = selected.merge(
        reference[
            [
                "entry_key",
                "subject_id",
                "study_id",
                "finding",
                "reference_binary_label",
                "expected_binary_action",
            ]
        ],
        on=["entry_key", "subject_id", "study_id", "finding"],
        how="inner",
        validate="one_to_one",
    )
    if len(private) != len(selected):
        raise ValueError("One or more selected entries lacks private reference")
    if not private["current_binary_label"].eq(private["current_binary_label_original"]).all():
        raise ValueError("An unseen selected entry was already modified")
    private["true_issue"] = private["current_binary_label_original"].ne(
        private["reference_binary_label"]
    ).astype(int)
    if not private["true_issue"].eq(
        private["expected_binary_action"].eq("relabel").astype(int)
    ).all():
        raise ValueError("Expert action and binary reference disagree")
    private["review_loop"] = args.loop_id
    private_path = directory / "selected_entries_private_evaluation.csv"
    atomic_csv(private, private_path)

    additions = pd.DataFrame(
        {
            "entry_key": private["entry_key"],
            "first_review_loop": args.loop_id,
            "true_issue": private["true_issue"].astype(int),
            "subject_id": private["subject_id"].astype(int),
            "study_id": private["study_id"].astype(int),
            "finding": private["finding"],
            "original_binary_label": private["current_binary_label_original"].astype(int),
            "reference_binary_label": private["reference_binary_label"].astype(int),
            "selected_hard_flag": private["cl_hard_issue"].astype(int),
            "selected_suspicion_percentile": private["suspicion_percentile_self"],
        }
    )
    new_history = pd.concat([history, additions], ignore_index=True)
    if new_history["entry_key"].duplicated().any():
        raise ValueError("Cumulative review history contains duplicates")
    issue_additions = additions[additions["true_issue"].eq(1)][
        [
            "entry_key",
            "subject_id",
            "study_id",
            "finding",
            "reference_binary_label",
            "first_review_loop",
        ]
    ]
    new_overrides = pd.concat([overrides, issue_additions], ignore_index=True)
    if new_overrides["entry_key"].duplicated().any():
        raise ValueError("Cumulative oracle override table contains duplicates")
    selected_issues = int(private["true_issue"].sum())
    cumulative_issues = int(new_history["true_issue"].sum())
    summary = {
        "protocol": PROTOCOL_NAME,
        "loop": args.loop_id,
        "selected_entries": len(private),
        "selected_true_issues": selected_issues,
        "selected_precision": selected_issues / len(private),
        "selected_hard_flags": int(private["cl_hard_issue"].sum()),
        "selected_hard_flag_true_issues": int(
            private.loc[private["cl_hard_issue"].eq(1), "true_issue"].sum()
        ),
        "cumulative_reviews": len(new_history),
        "cumulative_true_issues": cumulative_issues,
        "cumulative_recall": cumulative_issues / EXPECTED_BENCHMARK_ISSUES,
        "cumulative_precision": cumulative_issues / len(new_history),
        "remaining_true_issues": EXPECTED_BENCHMARK_ISSUES - cumulative_issues,
        "cumulative_oracle_overrides": len(new_overrides),
        "selected_entries_blind_sha256": sha256_file(selected_path),
        "private_reference_sha256": sha256_file(reference_path),
    }
    atomic_json(summary, directory / "oracle_loop_summary.json")
    paths = state_paths(args.output_dir)
    atomic_csv(new_overrides, paths["overrides"])
    atomic_csv(new_history, paths["history"])
    completion_marker.write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def trajectory_from_order(order: pd.DataFrame, budgets: list[int], method: str) -> pd.DataFrame:
    rows = []
    true_issue = order["true_issue"].astype(int).to_numpy()
    for budget in budgets:
        found = int(true_issue[:budget].sum())
        rows.append(
            {
                "method": method,
                "cumulative_reviews": budget,
                "review_fraction": budget / EXPECTED_BENCHMARK_ENTRIES,
                "cumulative_true_issues": found,
                "cumulative_recall": found / EXPECTED_BENCHMARK_ISSUES,
                "cumulative_precision": found / budget,
                "remaining_true_issues": EXPECTED_BENCHMARK_ISSUES - found,
            }
        )
    return pd.DataFrame(rows)


def discovery_audc(frame: pd.DataFrame) -> float:
    ordered = frame.sort_values("review_fraction")
    x = np.concatenate([[0.0], ordered["review_fraction"].to_numpy(dtype=float)])
    y = np.concatenate([[0.0], ordered["cumulative_recall"].to_numpy(dtype=float)])
    return float(np.trapezoid(y, x))


def evaluate(args: argparse.Namespace) -> None:
    reference_path = require_path(args.benchmark_reference, "--benchmark-reference")
    reference = benchmark_reference(reference_path)
    summaries = []
    selected_dynamic = []
    for loop_id in range(1, args.loops + 1):
        directory = loop_dir(args.output_dir, loop_id)
        if not (directory / ".oracle_update_complete").exists():
            raise FileNotFoundError(f"Loop {loop_id} oracle update is incomplete")
        summaries.append(json.loads((directory / "oracle_loop_summary.json").read_text()))
        selected_dynamic.append(pd.read_csv(directory / "selected_entries_private_evaluation.csv"))
    dynamic = pd.DataFrame(summaries).rename(
        columns={"loop": "checkpoint", "selected_true_issues": "new_true_issues"}
    )
    dynamic.insert(0, "method", "dynamic_cl")
    dynamic["review_fraction"] = dynamic["cumulative_reviews"] / EXPECTED_BENCHMARK_ENTRIES
    dynamic = dynamic[
        [
            "method",
            "checkpoint",
            "cumulative_reviews",
            "review_fraction",
            "new_true_issues",
            "cumulative_true_issues",
            "cumulative_recall",
            "cumulative_precision",
            "remaining_true_issues",
            "selected_hard_flags",
            "selected_hard_flag_true_issues",
        ]
    ]
    if int(dynamic.iloc[-1]["cumulative_reviews"]) != EXPECTED_BENCHMARK_ENTRIES:
        raise ValueError("Dynamic trajectory did not review all 498 entries")
    if int(dynamic.iloc[-1]["cumulative_true_issues"]) != EXPECTED_BENCHMARK_ISSUES:
        raise ValueError("Dynamic trajectory did not recover all 127 issues at full review")

    budgets = dynamic["cumulative_reviews"].astype(int).tolist()
    loop_one_scores = pd.read_csv(loop_dir(args.output_dir, 1) / "benchmark_scores_blind.csv")
    frozen = loop_one_scores.merge(
        reference[["entry_key", "expected_binary_action"]],
        on="entry_key",
        how="inner",
        validate="one_to_one",
    )
    frozen["true_issue"] = frozen["expected_binary_action"].eq("relabel").astype(int)
    frozen = frozen.sort_values(
        ["suspicion_percentile_self", "entry_key"],
        ascending=[False, True],
        kind="stable",
    )
    frozen_trajectory = trajectory_from_order(frozen, budgets, "frozen_cl")
    frozen_trajectory.insert(1, "checkpoint", range(1, len(frozen_trajectory) + 1))
    frozen_trajectory["new_true_issues"] = frozen_trajectory["cumulative_true_issues"].diff().fillna(
        frozen_trajectory["cumulative_true_issues"]
    ).astype(int)
    frozen_trajectory["selected_hard_flags"] = np.nan
    frozen_trajectory["selected_hard_flag_true_issues"] = np.nan
    frozen_trajectory = frozen_trajectory[dynamic.columns]
    if int(dynamic.iloc[0]["cumulative_true_issues"]) != int(
        frozen_trajectory.iloc[0]["cumulative_true_issues"]
    ):
        raise ValueError("Dynamic and frozen Loop-1 selections differ")

    rng = np.random.default_rng(args.random_seed)
    truth = frozen["true_issue"].to_numpy(dtype=int)
    random_rows = []
    random_audc = []
    fractions = np.asarray(budgets, dtype=float) / EXPECTED_BENCHMARK_ENTRIES
    for replicate in range(args.random_replicates):
        permutation = rng.permutation(truth)
        recalls = np.asarray([permutation[:budget].sum() for budget in budgets]) / EXPECTED_BENCHMARK_ISSUES
        random_audc.append(float(np.trapezoid(np.concatenate([[0.0], recalls]), np.concatenate([[0.0], fractions]))))
        for checkpoint, (budget, recall) in enumerate(zip(budgets, recalls), start=1):
            random_rows.append(
                {
                    "replicate": replicate,
                    "checkpoint": checkpoint,
                    "cumulative_reviews": budget,
                    "review_fraction": budget / EXPECTED_BENCHMARK_ENTRIES,
                    "cumulative_recall": float(recall),
                }
            )
    random_replicates = pd.DataFrame(random_rows)
    random_summary = (
        random_replicates.groupby(
            ["checkpoint", "cumulative_reviews", "review_fraction"], as_index=False
        )["cumulative_recall"]
        .agg(
            mean="mean",
            ci_lower=lambda x: np.quantile(x, 0.025),
            ci_upper=lambda x: np.quantile(x, 0.975),
        )
    )

    trajectory = pd.concat([dynamic, frozen_trajectory], ignore_index=True)
    dynamic_audc = discovery_audc(dynamic)
    frozen_audc = discovery_audc(frozen_trajectory)
    random_audc_array = np.asarray(random_audc, dtype=float)
    checkpoint_comparison = dynamic[
        ["checkpoint", "cumulative_reviews", "cumulative_recall"]
    ].merge(
        frozen_trajectory[["checkpoint", "cumulative_recall"]],
        on="checkpoint",
        suffixes=("_dynamic", "_frozen"),
    )
    checkpoint_comparison["dynamic_minus_frozen_recall"] = (
        checkpoint_comparison["cumulative_recall_dynamic"]
        - checkpoint_comparison["cumulative_recall_frozen"]
    )
    summary = {
        "protocol": PROTOCOL_NAME,
        "seed": args.seed,
        "loops": args.loops,
        "review_budget": args.review_budget,
        "total_entries": EXPECTED_BENCHMARK_ENTRIES,
        "total_true_issues": EXPECTED_BENCHMARK_ISSUES,
        "dynamic_audc": dynamic_audc,
        "frozen_audc": frozen_audc,
        "dynamic_minus_frozen_audc": dynamic_audc - frozen_audc,
        "random_audc_mean": float(random_audc_array.mean()),
        "random_audc_ci_lower": float(np.quantile(random_audc_array, 0.025)),
        "random_audc_ci_upper": float(np.quantile(random_audc_array, 0.975)),
        "dynamic_better_than_frozen_audc": bool(dynamic_audc > frozen_audc),
        "interpretation": "single_seed_descriptive_oracle_cl_detection_efficiency",
        "private_reference_sha256": sha256_file(reference_path),
    }
    atomic_csv(trajectory, args.output_dir / "iterative_detection_trajectory.csv")
    atomic_csv(checkpoint_comparison, args.output_dir / "dynamic_vs_frozen_checkpoints.csv")
    atomic_csv(random_summary, args.output_dir / "random_review_summary.csv")
    atomic_csv(random_replicates, args.output_dir / "random_review_replicates.csv")
    atomic_json(summary, args.output_dir / "iterative_detection_summary.json")

    figure, axis = plt.subplots(figsize=(7.6, 5.2))
    for method, color in [("dynamic_cl", "#D62728"), ("frozen_cl", "#4C78A8")]:
        frame = trajectory[trajectory["method"].eq(method)].sort_values("review_fraction")
        axis.plot(
            np.concatenate([[0.0], frame["review_fraction"]]),
            np.concatenate([[0.0], frame["cumulative_recall"]]),
            marker="o",
            linewidth=2.2,
            label=method.replace("_", " "),
            color=color,
        )
    axis.plot(
        np.concatenate([[0.0], random_summary["review_fraction"]]),
        np.concatenate([[0.0], random_summary["mean"]]),
        linestyle="--",
        color="#777777",
        label="random review",
    )
    axis.fill_between(
        random_summary["review_fraction"],
        random_summary["ci_lower"],
        random_summary["ci_upper"],
        color="#BBBBBB",
        alpha=0.3,
        linewidth=0,
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.03)
    axis.set_xlabel("Cumulative fraction of 498 entries reviewed")
    axis.set_ylabel("Cumulative recall of 127 expert issues")
    axis.set_title("Does dynamic OOF find remaining issues earlier?")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.output_dir / "iterative_detection_trajectory.png", dpi=180)
    plt.close(figure)
    (args.output_dir / ".evaluation_complete").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    if args.mode == "preflight":
        preflight(args)
    elif args.mode == "initialize":
        initialize(args)
    elif args.mode == "run-loop":
        run_loop(args)
    elif args.mode == "oracle-update":
        oracle_update(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
