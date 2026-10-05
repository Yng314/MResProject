#!/usr/bin/env python3
"""Replay MIMIC-CXR Round-0 OOF models and save outcome-blind test predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from cleanlab.dataset import overall_label_health_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from cxr_real_oof_cleanlab_smoke import (
    build_dataset_and_model_factories,
    set_seed,
    train_one_fold,
)
from cxr_real_noise_validation_smoke import (
    filter_existing_rows,
    load_real_pool,
    project_raw_labels_to_binary,
)
from cxr_real_full_train_eval_cleanlab_xrv12 import LABEL_NAMES


EXPECTED_TRAIN_IMAGES = 237_717
EXPECTED_TEST_IMAGES = 3_414
EXPECTED_TEST_STUDIES = 3_050


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--chexpert-csv", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--archived-detail-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_pipe_matrix(values: pd.Series, dtype: type) -> np.ndarray:
    return np.vstack([np.fromstring(str(value), sep="|", dtype=dtype) for value in values])


def load_pool(args: argparse.Namespace, split: str) -> pd.DataFrame:
    rows = load_real_pool(
        chexpert_csv=args.chexpert_csv,
        split_csv=args.split_csv,
        split_name=split,
        metadata_csv=args.metadata_csv,
        allowed_views=["AP", "PA"],
    )
    return filter_existing_rows(rows, image_root=args.image_root).reset_index(drop=True)


def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device, non_blocking=True)
            batches.append(torch.sigmoid(model(images)).cpu().numpy())
    return np.vstack(batches)


def flattened_dqs(labels: np.ndarray, valid: np.ndarray, probs: np.ndarray) -> float:
    y = labels[valid].astype(int)
    p = probs[valid]
    return float(
        overall_label_health_score(
            labels=y,
            pred_probs=np.column_stack([1.0 - p, p]),
            verbose=False,
        )
    )


def reproduction_summary(
    archived_path: Path,
    train_rows: pd.DataFrame,
    labels: np.ndarray,
    valid: np.ndarray,
    new_probs: np.ndarray,
) -> dict[str, Any]:
    archived = pd.read_csv(
        archived_path,
        usecols=[
            "pool_row_id",
            "subject_id",
            "study_id",
            "dicom_id",
            "binary_labels_for_detection",
            "valid_label_mask",
            "pred_probs",
        ],
    )
    if len(archived) != len(train_rows):
        raise ValueError("Archived and replayed training rows differ in length")
    expected_ids = train_rows["pool_row_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(archived["pool_row_id"].to_numpy(dtype=np.int64), expected_ids):
        raise ValueError("Archived and replayed pool_row_id ordering differs")
    for column in ["subject_id", "study_id", "dicom_id"]:
        if not archived[column].astype(str).eq(train_rows[column].astype(str)).all():
            raise ValueError(f"Archived and replayed {column} ordering differs")

    archived_labels = parse_pipe_matrix(archived["binary_labels_for_detection"], float)
    archived_valid = parse_pipe_matrix(archived["valid_label_mask"], int).astype(bool)
    archived_probs = parse_pipe_matrix(archived["pred_probs"], float)
    if not np.array_equal(archived_valid, valid):
        raise ValueError("Archived and replayed valid-label masks differ")
    if not np.allclose(archived_labels[valid], labels[valid], atol=0, rtol=0):
        raise ValueError("Archived and replayed binary labels differ")

    correlations: dict[str, float] = {}
    maes: dict[str, float] = {}
    for index, label in enumerate(LABEL_NAMES):
        left = archived_probs[:, index]
        right = new_probs[:, index]
        correlations[label] = float(np.corrcoef(left, right)[0, 1])
        maes[label] = float(np.mean(np.abs(left - right)))
    return {
        "row_identity_exact": True,
        "archived_probability_sha256": sha256_file(archived_path),
        "flattened_probability_mae": float(np.mean(np.abs(archived_probs - new_probs))),
        "flattened_probability_max_abs_error": float(np.max(np.abs(archived_probs - new_probs))),
        "per_label_pearson": correlations,
        "per_label_probability_mae": maes,
        "archived_dqs": flattened_dqs(labels, valid, archived_probs),
        "replay_dqs": flattened_dqs(labels, valid, new_probs),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    set_seed(args.seed)

    train_rows = load_pool(args, "train")
    test_rows = load_pool(args, "test")
    if len(train_rows) != EXPECTED_TRAIN_IMAGES:
        raise ValueError(f"Expected {EXPECTED_TRAIN_IMAGES} train images, found {len(train_rows)}")
    if len(test_rows) != EXPECTED_TEST_IMAGES:
        raise ValueError(f"Expected {EXPECTED_TEST_IMAGES} test images, found {len(test_rows)}")
    if test_rows["study_id"].nunique() != EXPECTED_TEST_STUDIES:
        raise ValueError("Unexpected official-test study count")

    _, train_labels, train_valid = project_raw_labels_to_binary(train_rows, LABEL_NAMES)
    _, test_labels, test_valid = project_raw_labels_to_binary(test_rows, LABEL_NAMES)
    dataset_cls, make_model = build_dataset_and_model_factories(
        model_backbone="mobilenet_v3_small_scratch",
        xrv_weights="densenet121-res224-all",
        xrv_cache_dir=None,
    )
    test_dataset = dataset_cls(
        rows=test_rows,
        image_root=args.image_root,
        image_size=args.image_size,
        binary_labels=test_labels,
        valid_mask=test_valid,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    splits = list(
        KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed).split(
            np.arange(len(train_rows))
        )
    )
    action_probs = np.zeros_like(train_labels, dtype=np.float32)
    action_fold = np.zeros(len(train_rows), dtype=np.int8)
    test_fold_probs = np.zeros(
        (args.n_splits, len(test_rows), len(LABEL_NAMES)), dtype=np.float32
    )
    checkpoint_hashes: dict[str, str] = {}

    for fold_no, (train_idx, val_idx) in enumerate(splits, start=1):
        train_dataset = dataset_cls(
            rows=train_rows.iloc[train_idx].reset_index(drop=True),
            image_root=args.image_root,
            image_size=args.image_size,
            binary_labels=train_labels[train_idx],
            valid_mask=train_valid[train_idx],
        )
        val_dataset = dataset_cls(
            rows=train_rows.iloc[val_idx].reset_index(drop=True),
            image_root=args.image_root,
            image_size=args.image_size,
            binary_labels=train_labels[val_idx],
            valid_mask=train_valid[val_idx],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        model = make_model(n_labels=len(LABEL_NAMES))
        action_probs[val_idx] = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            early_stopping_patience=args.early_stopping_patience,
            recover_best_weights=True,
            fold_idx=fold_no,
            n_folds=args.n_splits,
        )
        action_fold[val_idx] = fold_no
        test_fold_probs[fold_no - 1] = predict(model, test_loader, device)
        checkpoint_path = args.output_dir / f"fold_{fold_no}_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "seed": args.seed,
                "fold": fold_no,
                "label_names": LABEL_NAMES,
                "model_backbone": "mobilenet_v3_small_scratch",
            },
            checkpoint_path,
        )
        checkpoint_hashes[str(fold_no)] = sha256_file(checkpoint_path)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not np.all(action_fold > 0) or not np.isfinite(action_probs).all():
        raise RuntimeError("Incomplete action OOF predictions")
    if not np.isfinite(test_fold_probs).all():
        raise RuntimeError("Incomplete test-fold predictions")

    np.savez_compressed(
        args.output_dir / "action_oof_predictions.npz",
        probabilities=action_probs,
        fold=action_fold,
        pool_row_id=train_rows["pool_row_id"].to_numpy(dtype=np.int64),
    )
    test_index = test_rows[
        ["pool_row_id", "subject_id", "study_id", "dicom_id", "image_path"]
    ].copy()
    test_index.to_csv(args.output_dir / "test_image_index.csv", index=False)
    np.savez_compressed(
        args.output_dir / "test_fold_predictions.npz",
        probabilities=test_fold_probs,
    )
    replay = reproduction_summary(
        args.archived_detail_csv,
        train_rows,
        train_labels,
        train_valid,
        action_probs,
    )
    write_json(args.output_dir / "reproduction_summary.json", replay)

    artifacts = [
        "action_oof_predictions.npz",
        "test_image_index.csv",
        "test_fold_predictions.npz",
        "reproduction_summary.json",
    ]
    manifest = {
        "protocol": "mimic_dqs_calibration_transfer_v1",
        "phase": "outcome_blind_fold_scoring",
        "expert_reference_used": False,
        "seed": args.seed,
        "n_splits": args.n_splits,
        "train_images": len(train_rows),
        "test_images": len(test_rows),
        "test_studies": int(test_rows["study_id"].nunique()),
        "label_names": LABEL_NAMES,
        "checkpoint_sha256": checkpoint_hashes,
        "artifact_sha256": {
            name: sha256_file(args.output_dir / name) for name in artifacts
        },
        "source_inputs": {
            "chexpert_sha256": sha256_file(args.chexpert_csv),
            "split_sha256": sha256_file(args.split_csv),
            "metadata_sha256": sha256_file(args.metadata_csv),
            "archived_detail_sha256": sha256_file(args.archived_detail_csv),
        },
    }
    write_json(args.output_dir / "score_manifest.json", manifest)
    (args.output_dir / ".score_complete").write_text("complete\n", encoding="utf-8")
    print(json.dumps({"seed": args.seed, "reproduction": replay}, indent=2))


if __name__ == "__main__":
    main()
