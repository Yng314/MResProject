#!/usr/bin/env python3
"""DenseNet OOF evidence with a completely training-excluded VinDr sentinel set."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from cxr_real_noise_validation_smoke import MaskedBCEWithLogitsLoss
from vindr_densenet_oof import build_evidence, create_densenet, make_loader
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


PROTOCOL_NAME = "vindr_self_optimization_stress_v1"
ACTION_IMAGES = 2_400
SENTINEL_IMAGES = 600


def train_fold(
    blind: pd.DataFrame,
    labels: np.ndarray,
    sentinel: pd.DataFrame,
    sentinel_labels: np.ndarray,
    inner_train: np.ndarray,
    inner_validation: np.ndarray,
    outer_validation: np.ndarray,
    image_root: Path,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], int, float]:
    set_seed(seed)
    device = torch.device(args.device)
    train_loader = make_loader(
        blind.iloc[inner_train], labels[inner_train], image_root,
        args.batch_size, args.num_workers, True, seed,
    )
    inner_loader = make_loader(
        blind.iloc[inner_validation], labels[inner_validation], image_root,
        args.batch_size, args.num_workers, False, seed,
    )
    outer_loader = make_loader(
        blind.iloc[outer_validation], labels[outer_validation], image_root,
        args.batch_size, args.num_workers, False, seed,
    )
    sentinel_loader = make_loader(
        sentinel, sentinel_labels, image_root,
        args.batch_size, args.num_workers, False, seed,
    )
    model = create_densenet(args.initialization, args.xrv_cache_dir).to(device)
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
        train_loss = train_loss_sum / max(train_rows, 1)
        inner_loss = inner_loss_sum / max(inner_rows, 1)
        records.append(
            {"epoch": epoch, "train_loss": train_loss, "inner_validation_loss": inner_loss}
        )
        print(
            f"[{args.initialization}] epoch={epoch}/{args.epochs} "
            f"train={train_loss:.6f} inner={inner_loss:.6f}",
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
        raise RuntimeError("DenseNet did not produce a valid best state")
    model.load_state_dict(best_state)
    model.eval()

    def predict(loader: torch.utils.data.DataLoader) -> np.ndarray:
        probabilities = []
        with torch.no_grad():
            for images, _, _ in loader:
                logits = model(images.to(device, non_blocking=True))
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
        return np.vstack(probabilities).astype(np.float32)

    outer_probability = predict(outer_loader)
    sentinel_probability = predict(sentinel_loader)
    del model, optimizer, train_loader, inner_loader, outer_loader, sentinel_loader
    torch.cuda.empty_cache()
    return outer_probability, sentinel_probability, records, best_epoch, best_loss


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    blind_path = Path(args.blind_cohort)
    sentinel_path = Path(args.sentinel_cohort)
    split_reference_path = Path(args.split_reference_cohort)
    image_index_path = Path(args.image_index)
    image_root = Path(args.image_root)
    blind = pd.read_csv(blind_path)
    sentinel = pd.read_csv(sentinel_path)
    split_reference = pd.read_csv(split_reference_path)
    image_index = pd.read_csv(image_index_path)
    for frame, name in ((blind, "action cohort"), (sentinel, "sentinel cohort"), (split_reference, "split reference")):
        require_columns(frame, ["image_id", "image_path", "fold_id"] + LABELS, name)
        validate_blind_columns(frame)
    require_columns(image_index, ["image_id", "image_path", "output_sha256"], "image index")
    if len(blind) != ACTION_IMAGES or blind["image_id"].nunique() != ACTION_IMAGES:
        raise ValueError("Action cohort must contain 2,400 unique images")
    if len(sentinel) != SENTINEL_IMAGES or sentinel["image_id"].nunique() != SENTINEL_IMAGES:
        raise ValueError("Sentinel cohort must contain 600 unique images")
    action_ids = set(blind["image_id"].astype(str))
    sentinel_ids = set(sentinel["image_id"].astype(str))
    if action_ids & sentinel_ids:
        raise ValueError("Sentinel images overlap the training action set")
    if action_ids | sentinel_ids != set(image_index["image_id"].astype(str)):
        raise ValueError("Action plus sentinel images do not cover the locked image index")
    if not sentinel["fold_id"].eq(-1).all():
        raise ValueError("Sentinel fold ids must be -1")
    aligned = blind[["image_id", "fold_id"]].merge(
        split_reference[["image_id", "fold_id"]],
        on="image_id", suffixes=("_current", "_reference"), validate="one_to_one",
    )
    if len(aligned) != ACTION_IMAGES or not aligned["fold_id_current"].eq(aligned["fold_id_reference"]).all():
        raise ValueError("Current action cohort does not align with locked folds")
    for frame in (blind, sentinel):
        if not frame["image_path"].map(lambda value: (image_root / value).is_file()).all():
            raise FileNotFoundError("At least one locked VinDr PNG is missing")
    if sorted(blind["fold_id"].unique()) != list(range(args.n_splits)):
        raise ValueError("Action fold ids do not match --n-splits")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal DenseNet OOF requires CUDA")

    labels = blind[LABELS].to_numpy(dtype=np.int64)
    sentinel_labels = sentinel[LABELS].to_numpy(dtype=np.int64)
    split_reference = split_reference.set_index("image_id").loc[blind["image_id"]].reset_index()
    split_labels = split_reference[LABELS].to_numpy(dtype=np.int64)
    folds = blind["fold_id"].to_numpy(dtype=np.int64)
    oof = np.full(labels.shape, np.nan, dtype=np.float32)
    sentinel_fold_predictions = []
    training_records = []
    best_epochs = []
    for fold_id in range(args.n_splits):
        outer_train = np.flatnonzero(folds != fold_id)
        outer_validation = np.flatnonzero(folds == fold_id)
        inner_train, inner_validation, inner_seed = build_inner_split(
            outer_train, split_labels, seed=args.seed * 100 + fold_id
        )
        outer_probability, sentinel_probability, records, best_epoch, best_loss = train_fold(
            blind, labels, sentinel, sentinel_labels,
            inner_train, inner_validation, outer_validation,
            image_root, seed=args.seed * 100 + fold_id, args=args,
        )
        oof[outer_validation] = outer_probability
        sentinel_fold_predictions.append(sentinel_probability)
        best_epochs.append(best_epoch)
        for record in records:
            record.update(
                {
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
            training_records.append(record)
        print(f"[oof] fold {fold_id + 1}/{args.n_splits} complete", flush=True)
    sentinel_probability = np.mean(np.stack(sentinel_fold_predictions), axis=0).astype(np.float32)
    if not np.isfinite(oof).all() or not np.isfinite(sentinel_probability).all():
        raise RuntimeError("Action or sentinel probability matrix is incomplete")

    oof_frame, entries, action_issues = build_evidence(blind, labels, oof)
    sentinel_oof_frame, sentinel_entries, sentinel_issues = build_evidence(
        sentinel, sentinel_labels, sentinel_probability
    )
    paths = {
        "oof": output / "oof_predictions.csv",
        "entries": output / "entry_evidence.csv",
        "sentinel_oof": output / "sentinel_predictions.csv",
        "sentinel_entries": output / "sentinel_entry_evidence.csv",
        "history": output / "training_history.csv",
        "support": output / "fold_support.csv",
    }
    atomic_write_csv(oof_frame, paths["oof"])
    atomic_write_csv(entries, paths["entries"])
    atomic_write_csv(sentinel_oof_frame, paths["sentinel_oof"])
    atomic_write_csv(sentinel_entries, paths["sentinel_entries"])
    atomic_write_csv(pd.DataFrame(training_records), paths["history"])
    atomic_write_csv(fold_support(blind, args.n_splits), paths["support"])
    model = create_densenet(args.initialization, args.xrv_cache_dir)
    total_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    del model
    summary = {
        "protocol": PROTOCOL_NAME,
        "outcome_blind": True,
        "seed": args.seed,
        "initialization": args.initialization,
        "action_samples": ACTION_IMAGES,
        "action_entries": len(entries),
        "sentinel_samples": SENTINEL_IMAGES,
        "sentinel_entries": len(sentinel_entries),
        "sentinel_excluded_from_all_training_and_inner_validation": True,
        "sentinel_prediction_aggregation": "mean probability across four outer-fold models",
        "architecture": "TorchXRayVision one-channel DenseNet121 with six-output head",
        "model_parameters": total_parameters,
        "epochs_max": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "best_epochs": best_epochs,
        "action_cl_issue_entries": int(action_issues.sum()),
        "sentinel_cl_issue_entries": int(sentinel_issues.sum()),
        "blind_cohort_sha256": sha256_file(blind_path),
        "sentinel_cohort_sha256": sha256_file(sentinel_path),
        "split_reference_sha256": sha256_file(split_reference_path),
        "image_index_sha256": sha256_file(image_index_path),
        **{f"{key}_sha256": sha256_file(path) for key, path in paths.items()},
        "program_sha256": sha256_file(Path(__file__)),
        "cuda_device": torch.cuda.get_device_name(0),
    }
    atomic_write_text(output / "blind_run_summary.json", json.dumps(summary, indent=2))
    atomic_write_text(output / ".blind_run_complete", "complete\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-cohort", type=Path, required=True)
    parser.add_argument("--sentinel-cohort", type=Path, required=True)
    parser.add_argument("--split-reference-cohort", type=Path, required=True)
    parser.add_argument("--image-index", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--initialization", choices=["scratch", "xrv_pretrained"], required=True)
    parser.add_argument("--xrv-cache-dir")
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
