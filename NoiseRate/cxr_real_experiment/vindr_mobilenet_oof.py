#!/usr/bin/env python3
"""Outcome-blind MobileNetV3 OOF evidence for the VinDr known-error benchmark."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models as tv_models

from cxr_real_full_train_eval_cleanlab_xrv12 import RealCXRXRV12Dataset
from cxr_real_noise_validation_smoke import MaskedBCEWithLogitsLoss, estimate_issues
from vindr_known_gt_cl_benchmark import (
    LABELS,
    atomic_write_csv,
    atomic_write_text,
    build_inner_split,
    fold_support,
    require_columns,
    set_seed,
    sha256_file,
    validate_blind_columns,
)


PROTOCOL_NAME = "vindr_mobilenet_direction_sensitivity_v1"


def create_mobilenet(num_labels: int = len(LABELS)) -> nn.Module:
    model = tv_models.mobilenet_v3_small(weights=None)
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        1,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=False,
    )
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_labels)
    return model


def make_loader(
    rows: pd.DataFrame,
    labels: np.ndarray,
    image_root: Path,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    valid = np.ones_like(labels, dtype=bool)
    dataset = RealCXRXRV12Dataset(
        rows=rows.reset_index(drop=True),
        image_root=image_root,
        image_size=224,
        binary_labels=labels,
        valid_mask=valid,
    )
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=True,
    )


def train_fold(
    blind: pd.DataFrame,
    labels: np.ndarray,
    inner_train: np.ndarray,
    inner_validation: np.ndarray,
    outer_validation: np.ndarray,
    image_root: Path,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, Any]], int, float]:
    set_seed(seed)
    device = torch.device(args.device)
    train_loader = make_loader(
        blind.iloc[inner_train],
        labels[inner_train],
        image_root,
        args.batch_size,
        args.num_workers,
        True,
        seed,
    )
    inner_loader = make_loader(
        blind.iloc[inner_validation],
        labels[inner_validation],
        image_root,
        args.batch_size,
        args.num_workers,
        False,
        seed,
    )
    outer_loader = make_loader(
        blind.iloc[outer_validation],
        labels[outer_validation],
        image_root,
        args.batch_size,
        args.num_workers,
        False,
        seed,
    )

    model = create_mobilenet().to(device)
    criterion = MaskedBCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    records: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_rows = 0
        for images, targets, valid in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(images), targets, valid)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(images)
            train_rows += len(images)

        model.eval()
        inner_loss_sum = 0.0
        inner_rows = 0
        with torch.no_grad():
            for images, targets, valid in inner_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                valid = valid.to(device, non_blocking=True)
                loss = criterion(model(images), targets, valid)
                inner_loss_sum += float(loss.item()) * len(images)
                inner_rows += len(images)
        inner_loss = inner_loss_sum / max(inner_rows, 1)
        records.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / max(train_rows, 1),
                "inner_validation_loss": inner_loss,
            }
        )
        print(
            f"[mobilenet] epoch={epoch}/{args.epochs} "
            f"train={records[-1]['train_loss']:.6f} inner={inner_loss:.6f}",
            flush=True,
        )
        if inner_loss < best_loss - 1e-7:
            best_loss = inner_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("MobileNet did not produce a valid best state")
    model.load_state_dict(best_state)
    model.eval()
    probabilities = []
    with torch.no_grad():
        for images, _, _ in outer_loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    result = np.vstack(probabilities).astype(np.float32)
    del model, optimizer, train_loader, inner_loader, outer_loader
    torch.cuda.empty_cache()
    return result, records, best_epoch, best_loss


def build_evidence(
    blind: pd.DataFrame,
    labels: np.ndarray,
    oof: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    valid = np.ones_like(labels, dtype=bool)
    issue, quality_self, quality_margin, quality_entropy, _, _, _ = estimate_issues(
        labels,
        valid,
        oof,
    )
    epsilon = np.finfo(np.float32).eps
    clipped = np.clip(oof, epsilon, 1.0 - epsilon)
    entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
    suspicion = 1.0 - quality_self
    cl_first_score = issue.astype(np.float32) * 2.0 + suspicion

    oof_frame = blind[["image_id", "fold_id"]].copy()
    for label_index, label in enumerate(LABELS):
        oof_frame[f"probability__{label}"] = oof[:, label_index]

    entry_records = []
    for row_index, row in blind.iterrows():
        for label_index, label in enumerate(LABELS):
            entry_records.append(
                {
                    "image_id": str(row["image_id"]),
                    "fold_id": int(row["fold_id"]),
                    "label_index": label_index,
                    "label_name": label,
                    "noisy_label": int(labels[row_index, label_index]),
                    "oof_probability": float(oof[row_index, label_index]),
                    "cl_issue": int(issue[row_index, label_index]),
                    "label_quality_self_confidence": float(quality_self[row_index, label_index]),
                    "label_quality_normalized_margin": float(quality_margin[row_index, label_index]),
                    "label_quality_confidence_weighted_entropy": float(quality_entropy[row_index, label_index]),
                    "self_confidence_suspicion": float(suspicion[row_index, label_index]),
                    "predictive_entropy": float(entropy[row_index, label_index]),
                    "cl_first_score": float(cl_first_score[row_index, label_index]),
                }
            )
    entries = pd.DataFrame(entry_records)
    validate_blind_columns(entries.drop(columns=["cl_issue"], errors="ignore"))
    return oof_frame, entries, issue


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    blind_path = Path(args.blind_cohort)
    image_root = Path(args.image_root)
    image_index_path = Path(args.image_index)
    blind = pd.read_csv(blind_path)
    index = pd.read_csv(image_index_path)
    require_columns(blind, ["image_id", "image_path", "fold_id"] + LABELS, "blind cohort")
    require_columns(index, ["image_id", "image_path", "output_sha256"], "image index")
    validate_blind_columns(blind)
    if len(blind) != 3_000 or blind["image_id"].nunique() != 3_000:
        raise ValueError("Blind cohort must contain exactly 3,000 unique images")
    if len(index) != 3_000 or index["image_id"].nunique() != 3_000:
        raise ValueError("Image index must contain exactly 3,000 unique images")
    aligned = blind[["image_id", "image_path"]].merge(
        index[["image_id", "image_path"]],
        on="image_id",
        suffixes=("_blind", "_index"),
        validate="one_to_one",
    )
    if len(aligned) != 3_000 or not aligned["image_path_blind"].eq(aligned["image_path_index"]).all():
        raise ValueError("Blind cohort does not align with the locked image index")
    if not blind["image_path"].map(lambda value: (image_root / value).is_file()).all():
        raise FileNotFoundError("At least one locked VinDr PNG is missing")
    if sorted(blind["fold_id"].unique().tolist()) != list(range(args.n_splits)):
        raise ValueError("Blind cohort fold ids do not match --n-splits")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal MobileNet OOF requires CUDA")

    support = fold_support(blind, args.n_splits)
    labels = blind[LABELS].to_numpy(dtype=np.int64)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Blind labels are not binary")
    folds = blind["fold_id"].to_numpy(dtype=np.int64)
    oof = np.full(labels.shape, np.nan, dtype=np.float32)
    training_records = []
    best_epochs = []
    for fold_id in range(args.n_splits):
        outer_train = np.flatnonzero(folds != fold_id)
        outer_validation = np.flatnonzero(folds == fold_id)
        inner_train, inner_validation, inner_split_seed = build_inner_split(
            outer_train,
            labels,
            seed=args.seed * 100 + fold_id,
        )
        probabilities, records, best_epoch, best_loss = train_fold(
            blind,
            labels,
            inner_train,
            inner_validation,
            outer_validation,
            image_root,
            seed=args.seed * 100 + fold_id,
            args=args,
        )
        oof[outer_validation] = probabilities
        best_epochs.append(best_epoch)
        for record in records:
            record.update(
                {
                    "fold_id": fold_id,
                    "inner_train_samples": int(len(inner_train)),
                    "inner_validation_samples": int(len(inner_validation)),
                    "outer_validation_samples": int(len(outer_validation)),
                    "inner_split_seed": int(inner_split_seed),
                    "best_epoch": int(best_epoch),
                    "best_inner_validation_loss": float(best_loss),
                }
            )
            training_records.append(record)
        print(
            f"[oof] fold {fold_id + 1}/{args.n_splits} complete "
            f"epochs={len(records)} best_epoch={best_epoch} best_inner={best_loss:.6f}",
            flush=True,
        )
    if not np.isfinite(oof).all():
        raise RuntimeError("MobileNet OOF probability matrix is incomplete")

    oof_frame, entries, issue = build_evidence(blind, labels, oof)
    oof_path = output / "oof_predictions.csv"
    entry_path = output / "entry_evidence.csv"
    atomic_write_csv(oof_frame, oof_path)
    atomic_write_csv(entries, entry_path)
    atomic_write_csv(pd.DataFrame(training_records), output / "training_history.csv")
    atomic_write_csv(support, output / "fold_support.csv")
    model = create_mobilenet()
    total_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    del model
    summary = {
        "protocol": PROTOCOL_NAME,
        "outcome_blind": True,
        "seed": int(args.seed),
        "samples": int(len(blind)),
        "entries": int(len(entries)),
        "n_splits": int(args.n_splits),
        "model": "torchvision mobilenet_v3_small scratch, one input channel, six outputs",
        "model_parameters": total_parameters,
        "image_preprocessing": "XRayCenterCrop + XRayResizer(224) + XRV normalize(maxval=255)",
        "loss": "unweighted masked BCE-with-logits",
        "optimizer": "Adam",
        "nested_early_stopping": True,
        "epochs_max": int(args.epochs),
        "early_stopping_patience": int(args.early_stopping_patience),
        "best_epochs": best_epochs,
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "cl_issue_entries": int(issue.sum()),
        "raw_entry_dqs": float(1.0 - issue.mean()),
        "cuda_device": torch.cuda.get_device_name(0),
        "blind_cohort_sha256": sha256_file(blind_path),
        "image_index_sha256": sha256_file(image_index_path),
        "oof_predictions_sha256": sha256_file(oof_path),
        "entry_evidence_sha256": sha256_file(entry_path),
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(output / "blind_run_summary.json", json.dumps(summary, indent=2))
    atomic_write_text(output / ".blind_run_complete", "complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-cohort", type=Path, required=True)
    parser.add_argument("--image-index", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
