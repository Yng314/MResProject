#!/usr/bin/env python3
"""Outcome-isolated VinDr-CXR known-GT confident-learning benchmark."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchxrayvision as xrv
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

from cxr_real_noise_validation_smoke import estimate_issues


PROTOCOL_NAME = "vindr_known_gt_cl_detection_v1"
LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Lung Opacity",
    "Pleural effusion",
    "Pneumonia",
]
DEFAULT_SEEDS = [13, 42, 97, 123]
FORBIDDEN_BLIND_TOKENS = ("clean", "reference", "injected", "true_", "error")


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


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def parse_seeds(text: str | Iterable[int]) -> list[int]:
    if isinstance(text, str):
        seeds = [int(token.strip()) for token in text.split(",") if token.strip()]
    else:
        seeds = [int(value) for value in text]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be non-empty and unique")
    return seeds


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def validate_blind_columns(frame: pd.DataFrame) -> None:
    forbidden = [
        column
        for column in frame.columns
        if any(token in column.lower() for token in FORBIDDEN_BLIND_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"Blind cohort contains forbidden outcome columns: {forbidden}")


def fold_support(blind: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    records = []
    for fold_id in range(n_splits):
        fold = blind[blind["fold_id"] == fold_id]
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
        raise ValueError(f"Insufficient fold support:\n{support.to_string(index=False)}")
    return support


def prepare(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    labels = pd.read_csv(args.labels_csv)
    require_columns(labels, ["image_id"] + LABELS, "VinDr consensus labels")
    if len(labels) != 3000 or labels["image_id"].nunique() != 3000:
        raise ValueError("VinDr consensus labels must contain exactly 3,000 unique images")
    labels["image_id"] = labels["image_id"].astype(str)
    for label in LABELS:
        if not labels[label].isin([0, 1]).all():
            raise ValueError(f"Consensus label is not binary: {label}")

    conversion = pd.read_csv(args.conversion_manifest)
    require_columns(conversion, ["image_id", "output_path", "output_sha256"], "conversion manifest")
    conversion["image_id"] = conversion["image_id"].astype(str)
    if len(conversion) != 3000 or conversion["image_id"].nunique() != 3000:
        raise ValueError("Conversion manifest must contain exactly 3,000 unique images")
    cohort = labels.merge(
        conversion[["image_id", "output_path", "output_sha256"]],
        on="image_id",
        how="inner",
        validate="one_to_one",
    ).sort_values("image_id").reset_index(drop=True)
    if len(cohort) != 3000:
        raise ValueError("Consensus labels and PNG conversion do not align")
    cohort["image_path"] = cohort["output_path"].map(lambda value: Path(value).name)
    if not cohort["output_path"].map(lambda value: Path(value).is_file()).all():
        raise FileNotFoundError("At least one converted PNG is missing")

    index_path = output_root / "image_index.csv"
    atomic_write_csv(cohort[["image_id", "image_path", "output_sha256"]], index_path)
    seed_summaries = []
    for seed in parse_seeds(args.seeds):
        seed_root = output_root / f"seed_{seed}" / "prepared"
        seed_root.mkdir(parents=True, exist_ok=False)
        folds = np.full(len(cohort), -1, dtype=np.int64)
        splitter = KFold(n_splits=args.n_splits, shuffle=True, random_state=seed)
        for fold_id, (_, validation_indices) in enumerate(splitter.split(cohort)):
            folds[validation_indices] = fold_id
        if np.any(folds < 0):
            raise RuntimeError("Incomplete OOF fold assignment")

        clean = cohort[LABELS].to_numpy(dtype=np.int64)
        noisy = clean.copy()
        injected = np.zeros_like(clean, dtype=bool)
        direction = np.full(clean.shape, "none", dtype=object)
        corruption_records = []
        for label_index, label in enumerate(LABELS):
            rng = np.random.default_rng(seed * 1000 + label_index)
            positive = np.flatnonzero(clean[:, label_index] == 1)
            negative = np.flatnonzero(clean[:, label_index] == 0)
            count = int(round(args.noise_fraction * len(positive)))
            if count < 1 or count >= len(positive) or count >= len(negative):
                raise ValueError(f"Invalid corruption count for {label}: {count}")
            positive_to_negative = rng.choice(positive, size=count, replace=False)
            negative_to_positive = rng.choice(negative, size=count, replace=False)
            noisy[positive_to_negative, label_index] = 0
            noisy[negative_to_positive, label_index] = 1
            injected[positive_to_negative, label_index] = True
            injected[negative_to_positive, label_index] = True
            direction[positive_to_negative, label_index] = "1_to_0"
            direction[negative_to_positive, label_index] = "0_to_1"
            corruption_records.append(
                {
                    "seed": seed,
                    "label_name": label,
                    "clean_positives": int(len(positive)),
                    "clean_negatives": int(len(negative)),
                    "flips_each_direction": count,
                    "injected_errors": int(2 * count),
                    "observed_positives": int(noisy[:, label_index].sum()),
                }
            )
            if int(noisy[:, label_index].sum()) != len(positive):
                raise RuntimeError(f"Prevalence changed after balanced corruption: {label}")

        blind = cohort[["image_id", "image_path"]].copy()
        blind["fold_id"] = folds
        for label_index, label in enumerate(LABELS):
            blind[label] = noisy[:, label_index]
        validate_blind_columns(blind)
        support = fold_support(blind, args.n_splits)

        private_records = []
        for row_index, image_id in enumerate(cohort["image_id"]):
            for label_index, label in enumerate(LABELS):
                private_records.append(
                    {
                        "image_id": image_id,
                        "label_name": label,
                        "clean_label": int(clean[row_index, label_index]),
                        "noisy_label": int(noisy[row_index, label_index]),
                        "injected_error": int(injected[row_index, label_index]),
                        "flip_direction": str(direction[row_index, label_index]),
                    }
                )
        private = pd.DataFrame(private_records)
        if len(private) != len(cohort) * len(LABELS):
            raise RuntimeError("Private reference row count is wrong")
        if not (
            private["injected_error"]
            == (private["clean_label"] != private["noisy_label"]).astype(int)
        ).all():
            raise RuntimeError("Injected-error flags disagree with label values")

        blind_path = seed_root / "blind_noisy_cohort.csv"
        private_path = seed_root / "private_reference.csv"
        support_path = seed_root / "fold_support.csv"
        corruption_path = seed_root / "corruption_counts.csv"
        atomic_write_csv(blind, blind_path)
        atomic_write_csv(private, private_path)
        atomic_write_csv(support, support_path)
        atomic_write_csv(pd.DataFrame(corruption_records), corruption_path)
        summary = {
            "protocol": PROTOCOL_NAME,
            "seed": seed,
            "samples": int(len(blind)),
            "entries": int(len(private)),
            "labels": LABELS,
            "noise_fraction_of_positive_support_per_direction": float(args.noise_fraction),
            "injected_errors": int(private["injected_error"].sum()),
            "n_splits": int(args.n_splits),
            "grouping": "image-level; no patient identifier available in released DICOM headers",
            "blind_cohort_sha256": sha256_file(blind_path),
            "private_reference_sha256": sha256_file(private_path),
            "fold_support_sha256": sha256_file(support_path),
            "corruption_counts_sha256": sha256_file(corruption_path),
        }
        atomic_write_text(seed_root / "prepare_summary.json", json.dumps(summary, indent=2))
        atomic_write_text(seed_root / ".prepare_complete", "complete\n")
        seed_summaries.append(summary)

    root_summary = {
        "protocol": PROTOCOL_NAME,
        "seeds": parse_seeds(args.seeds),
        "labels": LABELS,
        "samples": int(len(cohort)),
        "image_index_sha256": sha256_file(index_path),
        "consensus_labels_sha256": sha256_file(Path(args.labels_csv)),
        "conversion_manifest_sha256": sha256_file(Path(args.conversion_manifest)),
        "seed_injected_errors": {
            str(summary["seed"]): summary["injected_errors"] for summary in seed_summaries
        },
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(output_root / "prepare_summary.json", json.dumps(root_summary, indent=2))
    atomic_write_text(output_root / ".prepare_complete", "complete\n")
    print(json.dumps(root_summary, indent=2), flush=True)


class PNGIndexDataset(Dataset):
    def __init__(self, index: pd.DataFrame, image_root: Path):
        self.index = index.reset_index(drop=True)
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.image_root / str(self.index.iloc[index]["image_path"])
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32)
        if array.shape != (224, 224):
            raise ValueError(f"Unexpected PNG shape: {path} -> {array.shape}")
        array = xrv.datasets.normalize(array[np.newaxis, :, :], maxval=255)
        return torch.from_numpy(array).float(), index


class FrozenXRVEncoder(nn.Module):
    def __init__(self, cache_dir: str):
        super().__init__()
        self.encoder = xrv.models.DenseNet(
            weights="densenet121-res224-all",
            cache_dir=cache_dir,
        )
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()

    def train(self, mode: bool = True) -> "FrozenXRVEncoder":
        super().train(False)
        self.encoder.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder.features(images)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return features.flatten(1)


def extract_features(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    index = pd.read_csv(args.image_index)
    require_columns(index, ["image_id", "image_path", "output_sha256"], "image index")
    if len(index) != 3000 or index["image_id"].nunique() != 3000:
        raise ValueError("Feature extraction requires exactly 3,000 unique images")
    image_root = Path(args.image_root)
    if not index["image_path"].map(lambda value: (image_root / value).is_file()).all():
        raise FileNotFoundError("At least one indexed PNG is missing")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    set_seed(args.seed)
    dataset = PNGIndexDataset(index, image_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    model = FrozenXRVEncoder(args.xrv_cache_dir).to(device)
    features = np.full((len(index), 1024), np.nan, dtype=np.float32)
    with torch.no_grad():
        for batch_number, (images, indices) in enumerate(loader, start=1):
            encoded = model(images.to(device)).cpu().numpy().astype(np.float32)
            features[indices.numpy()] = encoded
            if batch_number % 10 == 0 or batch_number == len(loader):
                print(f"[features] {int(np.isfinite(features[:, 0]).sum())}/{len(index)}", flush=True)
    if not np.isfinite(features).all() or features.shape != (3000, 1024):
        raise RuntimeError("Feature extraction is incomplete or non-finite")
    variable_features = int(np.sum(np.ptp(features, axis=0) > 0))
    if variable_features < int(0.95 * features.shape[1]):
        raise RuntimeError(
            f"Too many constant extracted features: {variable_features}/{features.shape[1]} vary"
        )

    features_path = output / "xrv_features.npz"
    atomic_save_npz(
        features_path,
        image_id=index["image_id"].astype(str).to_numpy(dtype="U32"),
        features=features,
    )
    summary = {
        "protocol": PROTOCOL_NAME,
        "outcome_blind": True,
        "samples": int(len(index)),
        "feature_dimension": int(features.shape[1]),
        "encoder": "torchxrayvision densenet121-res224-all frozen",
        "image_index_sha256": sha256_file(Path(args.image_index)),
        "features_sha256": sha256_file(features_path),
        "feature_mean": float(features.mean()),
        "feature_std": float(features.std()),
        "variable_feature_dimensions": variable_features,
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(output / "feature_summary.json", json.dumps(summary, indent=2))
    atomic_write_text(output / ".features_complete", "complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    image_ids = payload["image_id"].astype(str)
    features = payload["features"].astype(np.float32)
    if len(image_ids) != len(features) or features.ndim != 2:
        raise ValueError("Feature payload shape is invalid")
    if len(set(image_ids)) != len(image_ids) or not np.isfinite(features).all():
        raise ValueError("Feature payload ids or values are invalid")
    return image_ids, features


def build_inner_split(
    indices: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    for attempt in range(1_000):
        split_seed = seed + attempt
        inner_train, inner_validation = train_test_split(
            indices,
            test_size=0.12,
            random_state=split_seed,
            shuffle=True,
        )
        supports_are_valid = True
        for subset in [inner_train, inner_validation]:
            positive = labels[subset].sum(axis=0)
            negative = len(subset) - positive
            if np.any(positive < 3) or np.any(negative < 3):
                supports_are_valid = False
                break
        if supports_are_valid:
            return np.asarray(inner_train), np.asarray(inner_validation), split_seed
    raise ValueError("Unable to construct an inner split with minimum class support")


def train_linear_head(
    features: np.ndarray,
    labels: np.ndarray,
    inner_train: np.ndarray,
    inner_validation: np.ndarray,
    outer_validation: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    set_seed(seed)
    device = torch.device(args.device)
    model = nn.Linear(features.shape[1], labels.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    train_dataset = TensorDataset(
        torch.from_numpy(features[inner_train]),
        torch.from_numpy(labels[inner_train].astype(np.float32)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_x = torch.from_numpy(features[inner_validation]).to(device)
    validation_y = torch.from_numpy(labels[inner_validation].astype(np.float32)).to(device)
    best_loss = float("inf")
    best_state = None
    no_improve = 0
    records = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        training_loss = 0.0
        training_rows = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            training_loss += float(loss.item()) * len(batch_x)
            training_rows += len(batch_x)
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(validation_x), validation_y).item())
        records.append(
            {
                "epoch": epoch,
                "train_loss": training_loss / max(training_rows, 1),
                "inner_validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("Linear head did not produce a model state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        outer_x = torch.from_numpy(features[outer_validation]).to(device)
        probabilities = torch.sigmoid(model(outer_x)).cpu().numpy()
    return probabilities.astype(np.float32), records


def run_blind(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    blind_path = Path(args.blind_cohort)
    blind = pd.read_csv(blind_path)
    require_columns(blind, ["image_id", "image_path", "fold_id"] + LABELS, "blind cohort")
    validate_blind_columns(blind)
    if len(blind) != 3000 or blind["image_id"].nunique() != 3000:
        raise ValueError("Blind cohort must contain exactly 3,000 unique images")
    if sorted(blind["fold_id"].unique().tolist()) != list(range(args.n_splits)):
        raise ValueError("Blind cohort fold ids do not match --n-splits")
    support = fold_support(blind, args.n_splits)
    labels = blind[LABELS].to_numpy(dtype=np.int64)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Blind labels are not binary")

    feature_ids, feature_values = load_features(Path(args.features))
    feature_lookup = {image_id: index for index, image_id in enumerate(feature_ids)}
    if set(feature_ids) != set(blind["image_id"].astype(str)):
        raise ValueError("Feature ids do not match blind cohort ids")
    features = feature_values[
        [feature_lookup[image_id] for image_id in blind["image_id"].astype(str)]
    ]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    folds = blind["fold_id"].to_numpy(dtype=np.int64)
    oof = np.full(labels.shape, np.nan, dtype=np.float32)
    training_records = []
    for fold_id in range(args.n_splits):
        outer_train = np.flatnonzero(folds != fold_id)
        outer_validation = np.flatnonzero(folds == fold_id)
        inner_train, inner_validation, inner_split_seed = build_inner_split(
            outer_train,
            labels,
            seed=args.seed * 100 + fold_id,
        )
        probabilities, records = train_linear_head(
            features,
            labels,
            inner_train,
            inner_validation,
            outer_validation,
            seed=args.seed * 100 + fold_id,
            args=args,
        )
        oof[outer_validation] = probabilities
        for record in records:
            record.update(
                {
                    "fold_id": fold_id,
                    "inner_train_samples": int(len(inner_train)),
                    "inner_validation_samples": int(len(inner_validation)),
                    "outer_validation_samples": int(len(outer_validation)),
                    "inner_split_seed": int(inner_split_seed),
                }
            )
            training_records.append(record)
        print(
            f"[oof] fold {fold_id + 1}/{args.n_splits} complete "
            f"epochs={len(records)} inner_best={min(r['inner_validation_loss'] for r in records):.6f}",
            flush=True,
        )
    if not np.isfinite(oof).all():
        raise RuntimeError("OOF probability matrix is incomplete")

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
    atomic_write_csv(oof_frame, output / "oof_predictions.csv")

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
                    "label_quality_self_confidence": float(
                        quality_self[row_index, label_index]
                    ),
                    "label_quality_normalized_margin": float(
                        quality_margin[row_index, label_index]
                    ),
                    "label_quality_confidence_weighted_entropy": float(
                        quality_entropy[row_index, label_index]
                    ),
                    "self_confidence_suspicion": float(suspicion[row_index, label_index]),
                    "predictive_entropy": float(entropy[row_index, label_index]),
                    "cl_first_score": float(cl_first_score[row_index, label_index]),
                }
            )
    entries = pd.DataFrame(entry_records)
    validate_blind_columns(entries.drop(columns=["cl_issue"], errors="ignore"))
    entry_path = output / "entry_evidence.csv"
    atomic_write_csv(entries, entry_path)
    atomic_write_csv(pd.DataFrame(training_records), output / "training_history.csv")
    atomic_write_csv(support, output / "fold_support.csv")
    summary = {
        "protocol": PROTOCOL_NAME,
        "outcome_blind": True,
        "seed": int(args.seed),
        "samples": int(len(blind)),
        "entries": int(len(entries)),
        "n_splits": int(args.n_splits),
        "model": "frozen XRV DenseNet-121 features + independently trained linear OOF heads",
        "nested_early_stopping": True,
        "epochs_max": int(args.epochs),
        "early_stopping_patience": int(args.early_stopping_patience),
        "learning_rate": float(args.learning_rate),
        "cl_issue_entries": int(issue.sum()),
        "raw_entry_dqs": float(1.0 - issue.mean()),
        "blind_cohort_sha256": sha256_file(blind_path),
        "features_sha256": sha256_file(Path(args.features)),
        "entry_evidence_sha256": sha256_file(entry_path),
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(output / "blind_run_summary.json", json.dumps(summary, indent=2))
    atomic_write_text(output / ".blind_run_complete", "complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def binary_metrics(truth: np.ndarray, prediction: np.ndarray, score: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    score = np.asarray(score, dtype=float)
    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    tn = int(np.sum(~truth & ~prediction))
    prevalence = float(truth.mean())
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = safe_ratio(2 * precision * recall, precision + recall)
    if len(np.unique(truth)) == 2:
        auprc = float(average_precision_score(truth, score))
        auroc = float(roc_auc_score(truth, score))
    else:
        auprc = float("nan")
        auroc = float("nan")
    return {
        "n": int(len(truth)),
        "true_errors": int(truth.sum()),
        "selected": int(prediction.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "prevalence": prevalence,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "enrichment": safe_ratio(precision, prevalence),
        "auprc": auprc,
        "auroc": auroc,
    }


def top_budget_mask(frame: pd.DataFrame, score_column: str, budget: int) -> np.ndarray:
    ranked = frame.assign(_row=np.arange(len(frame))).sort_values(
        [score_column, "image_id", "label_name"],
        ascending=[False, True, True],
        kind="stable",
    )
    selected_rows = ranked.head(budget)["_row"].to_numpy(dtype=int)
    mask = np.zeros(len(frame), dtype=bool)
    mask[selected_rows] = True
    return mask


def budget_definitions(n: int, true_errors: int) -> list[tuple[str, int]]:
    definitions = [(f"{int(fraction * 100)}pct", max(1, int(round(n * fraction)))) for fraction in [0.01, 0.02, 0.05, 0.10]]
    definitions.append(("true_error_count", int(true_errors)))
    return definitions


def evaluate_seed(
    seed: int,
    prepared_root: Path,
    evidence_root: Path,
    random_iterations: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    private_path = prepared_root / "private_reference.csv"
    evidence_path = evidence_root / "entry_evidence.csv"
    private = pd.read_csv(private_path)
    evidence = pd.read_csv(evidence_path)
    require_columns(
        private,
        ["image_id", "label_name", "clean_label", "noisy_label", "injected_error", "flip_direction"],
        "private reference",
    )
    require_columns(
        evidence,
        ["image_id", "label_name", "noisy_label", "oof_probability", "cl_issue", "self_confidence_suspicion", "predictive_entropy", "cl_first_score"],
        "blind evidence",
    )
    if len(private) != 18000 or len(evidence) != 18000:
        raise ValueError(f"Seed {seed} does not contain exactly 18,000 entries")
    keys = ["image_id", "label_name"]
    if private.duplicated(keys).any() or evidence.duplicated(keys).any():
        raise ValueError(f"Seed {seed} contains duplicate entry keys")
    merged = evidence.merge(private, on=keys, how="left", suffixes=("", "_private"), validate="one_to_one")
    if merged["injected_error"].isna().any():
        raise ValueError(f"Seed {seed} private join is incomplete")
    if not merged["noisy_label"].eq(merged["noisy_label_private"]).all():
        raise ValueError(f"Seed {seed} noisy labels changed between blind and private stages")
    truth = merged["injected_error"].to_numpy(dtype=bool)
    methods = {
        "cl_first": "cl_first_score",
        "self_confidence": "self_confidence_suspicion",
        "predictive_entropy": "predictive_entropy",
    }

    overall_records = []
    for method, score_column in methods.items():
        prediction = merged["cl_issue"].to_numpy(dtype=bool) if method == "cl_first" else np.zeros(len(merged), dtype=bool)
        metrics = binary_metrics(truth, prediction, merged[score_column].to_numpy())
        metrics.update({"seed": seed, "method": method})
        overall_records.append(metrics)
    hard_metrics = binary_metrics(
        truth,
        merged["cl_issue"].to_numpy(dtype=bool),
        merged["self_confidence_suspicion"].to_numpy(),
    )

    budget_records = []
    random_records = []
    rng = np.random.default_rng(seed + 100_000)
    true_errors = int(truth.sum())
    for budget_name, budget in budget_definitions(len(merged), true_errors):
        random_tp = rng.hypergeometric(
            ngood=true_errors,
            nbad=len(merged) - true_errors,
            nsample=budget,
            size=random_iterations,
        )
        for iteration, tp in enumerate(random_tp):
            random_records.append(
                {
                    "seed": seed,
                    "budget_name": budget_name,
                    "budget": budget,
                    "iteration": iteration,
                    "tp": int(tp),
                }
            )
        for method, score_column in methods.items():
            selected = top_budget_mask(merged, score_column, budget)
            tp = int(np.sum(truth & selected))
            precision = safe_ratio(tp, budget)
            recall = safe_ratio(tp, true_errors)
            budget_records.append(
                {
                    "seed": seed,
                    "method": method,
                    "budget_name": budget_name,
                    "budget": budget,
                    "tp": tp,
                    "precision": precision,
                    "recall": recall,
                    "enrichment": safe_ratio(precision, float(truth.mean())),
                    "random_mean_tp": float(random_tp.mean()),
                    "random_p_ge_observed": float((1 + np.sum(random_tp >= tp)) / (random_iterations + 1)),
                }
            )

    per_label_records = []
    for label, frame in merged.groupby("label_name", sort=False):
        for direction in ["0_to_1", "1_to_0"]:
            scoped_truth = frame["flip_direction"].eq(direction).to_numpy()
            for method, score_column in methods.items():
                metrics = binary_metrics(
                    scoped_truth,
                    frame["cl_issue"].to_numpy(dtype=bool) if method == "cl_first" else np.zeros(len(frame), dtype=bool),
                    frame[score_column].to_numpy(),
                )
                metrics.update({"seed": seed, "label_name": label, "flip_direction": direction, "method": method})
                per_label_records.append(metrics)
        scoped_truth = frame["injected_error"].to_numpy(dtype=bool)
        for method, score_column in methods.items():
            metrics = binary_metrics(
                scoped_truth,
                frame["cl_issue"].to_numpy(dtype=bool) if method == "cl_first" else np.zeros(len(frame), dtype=bool),
                frame[score_column].to_numpy(),
            )
            metrics.update({"seed": seed, "label_name": label, "flip_direction": "all", "method": method})
            per_label_records.append(metrics)

    clean_aurocs = []
    for label, frame in merged.groupby("label_name", sort=False):
        clean_aurocs.append(
            {
                "seed": seed,
                "label_name": label,
                "noisy_label_auroc": float(roc_auc_score(frame["noisy_label"], frame["oof_probability"])),
                "clean_label_auroc": float(roc_auc_score(frame["clean_label"], frame["oof_probability"])),
            }
        )

    true_quality = float(1.0 - truth.mean())
    raw_dqs = float(1.0 - merged["cl_issue"].mean())
    summary = {
        "seed": seed,
        "samples": int(merged["image_id"].nunique()),
        "entries": int(len(merged)),
        "true_errors": true_errors,
        "true_quality": true_quality,
        "raw_entry_dqs": raw_dqs,
        "dqs_signed_error": raw_dqs - true_quality,
        "dqs_absolute_error": abs(raw_dqs - true_quality),
        "cl_hard": hard_metrics,
        "private_reference_sha256": sha256_file(private_path),
        "entry_evidence_sha256": sha256_file(evidence_path),
    }
    return (
        pd.DataFrame(overall_records),
        pd.DataFrame(budget_records),
        pd.DataFrame(per_label_records),
        pd.DataFrame(random_records),
        pd.DataFrame(clean_aurocs),
        summary,
    )


def bootstrap_seed_entries(
    merged: pd.DataFrame,
    seed: int,
    iterations: int,
) -> pd.DataFrame:
    image_ids = merged["image_id"].drop_duplicates().to_numpy()
    groups = {image_id: frame.copy() for image_id, frame in merged.groupby("image_id")}
    rng = np.random.default_rng(seed + 200_000)
    records = []
    for iteration in range(iterations):
        sampled = rng.choice(image_ids, size=len(image_ids), replace=True)
        parts = []
        for draw, image_id in enumerate(sampled):
            part = groups[image_id].copy()
            part["bootstrap_image"] = draw
            parts.append(part)
        frame = pd.concat(parts, ignore_index=True)
        truth = frame["injected_error"].to_numpy(dtype=bool)
        true_errors = int(truth.sum())
        cl_metrics = binary_metrics(
            truth,
            frame["cl_issue"].to_numpy(dtype=bool),
            frame["self_confidence_suspicion"].to_numpy(),
        )
        budget = true_errors
        cl_selected = top_budget_mask(frame, "cl_first_score", budget)
        self_selected = top_budget_mask(frame, "self_confidence_suspicion", budget)
        records.append(
            {
                "seed": seed,
                "iteration": iteration,
                "prevalence": float(truth.mean()),
                "self_auprc": float(average_precision_score(truth, frame["self_confidence_suspicion"])),
                "self_auprc_minus_prevalence": float(average_precision_score(truth, frame["self_confidence_suspicion"]) - truth.mean()),
                "cl_hard_enrichment": cl_metrics["enrichment"],
                "cl_first_recall_at_error_count": safe_ratio(int(np.sum(truth & cl_selected)), true_errors),
                "self_recall_at_error_count": safe_ratio(int(np.sum(truth & self_selected)), true_errors),
                "cl_minus_self_recall_at_error_count": safe_ratio(int(np.sum(truth & cl_selected)), true_errors) - safe_ratio(int(np.sum(truth & self_selected)), true_errors),
            }
        )
    return pd.DataFrame(records)


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    permutations = [abs(float(np.mean(values * np.asarray(signs)))) for signs in itertools.product([-1.0, 1.0], repeat=len(values))]
    return float(np.mean(np.asarray(permutations) >= observed - 1e-15))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.zeros(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def evaluate(args: argparse.Namespace) -> None:
    experiment_root = Path(args.experiment_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    seeds = parse_seeds(args.seeds)
    overall_parts = []
    budget_parts = []
    label_parts = []
    random_parts = []
    auroc_parts = []
    summaries = []
    bootstrap_parts = []
    merged_by_seed: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        prepared_root = experiment_root / f"seed_{seed}" / "prepared"
        evidence_root = experiment_root / f"seed_{seed}" / "blind_run"
        if not (prepared_root / ".prepare_complete").is_file():
            raise FileNotFoundError(f"Seed {seed} prepare marker is missing")
        if not (evidence_root / ".blind_run_complete").is_file():
            raise FileNotFoundError(f"Seed {seed} blind-run marker is missing")
        result = evaluate_seed(seed, prepared_root, evidence_root, args.random_iterations)
        overall, budgets, labels, random_frame, aurocs, summary = result
        overall_parts.append(overall)
        budget_parts.append(budgets)
        label_parts.append(labels)
        random_parts.append(random_frame)
        auroc_parts.append(aurocs)
        summaries.append(summary)

        evidence = pd.read_csv(evidence_root / "entry_evidence.csv")
        private = pd.read_csv(prepared_root / "private_reference.csv")
        merged = evidence.merge(private, on=["image_id", "label_name"], suffixes=("", "_private"), validate="one_to_one")
        merged_by_seed[seed] = merged
        if args.bootstrap_iterations > 0:
            bootstrap_parts.append(
                bootstrap_seed_entries(merged, seed, args.bootstrap_iterations)
            )

    overall_frame = pd.concat(overall_parts, ignore_index=True)
    budget_frame = pd.concat(budget_parts, ignore_index=True)
    label_frame = pd.concat(label_parts, ignore_index=True)
    random_frame = pd.concat(random_parts, ignore_index=True)
    auroc_frame = pd.concat(auroc_parts, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    atomic_write_csv(overall_frame, output / "overall_detection_metrics.csv")
    atomic_write_csv(budget_frame, output / "budget_metrics.csv")
    atomic_write_csv(label_frame, output / "per_label_direction_metrics.csv")
    atomic_write_csv(random_frame, output / "matched_random_replicates.csv")
    atomic_write_csv(auroc_frame, output / "oof_label_aurocs.csv")
    atomic_write_csv(summary_frame, output / "seed_quality_summary.csv")

    bootstrap_ci = []
    if bootstrap_parts:
        bootstrap = pd.concat(bootstrap_parts, ignore_index=True)
        atomic_write_csv(bootstrap, output / "image_cluster_bootstrap.csv")
        rng = np.random.default_rng(args.bootstrap_seed)
        hierarchical_records = []
        by_seed = {seed: frame.reset_index(drop=True) for seed, frame in bootstrap.groupby("seed")}
        fields = [
            "self_auprc_minus_prevalence",
            "cl_hard_enrichment",
            "cl_first_recall_at_error_count",
            "self_recall_at_error_count",
            "cl_minus_self_recall_at_error_count",
        ]
        for iteration in range(args.bootstrap_iterations):
            sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            record = {"iteration": iteration}
            for field in fields:
                values = []
                for sampled_seed in sampled_seeds:
                    frame = by_seed[int(sampled_seed)]
                    row = frame.iloc[int(rng.integers(0, len(frame)))]
                    values.append(float(row[field]))
                record[field] = float(np.mean(values))
            hierarchical_records.append(record)
        hierarchical = pd.DataFrame(hierarchical_records)
        atomic_write_csv(hierarchical, output / "hierarchical_seed_image_bootstrap.csv")
        for field in fields:
            bootstrap_ci.append(
                {
                    "metric": field,
                    "point_estimate": float(
                        bootstrap.groupby("seed")[field].mean().mean()
                    ),
                    "ci_lower": float(hierarchical[field].quantile(0.025)),
                    "ci_upper": float(hierarchical[field].quantile(0.975)),
                }
            )
        atomic_write_csv(pd.DataFrame(bootstrap_ci), output / "hierarchical_bootstrap_ci.csv")

    error_budget = budget_frame[budget_frame["budget_name"] == "true_error_count"]
    contrast_records = []
    for comparator in ["self_confidence", "predictive_entropy"]:
        pivot = error_budget[error_budget["method"].isin(["cl_first", comparator])].pivot(
            index="seed", columns="method", values="recall"
        )
        values = (pivot["cl_first"] - pivot[comparator]).to_numpy(dtype=float)
        contrast_records.append(
            {
                "contrast": f"cl_first_minus_{comparator}_recall_at_error_count",
                "mean_difference": float(values.mean()),
                "positive_seeds": int(np.sum(values > 0)),
                "ties": int(np.sum(values == 0)),
                "negative_seeds": int(np.sum(values < 0)),
                "exact_sign_flip_p": exact_sign_flip_p(values),
            }
        )
    adjusted = holm_adjust([record["exact_sign_flip_p"] for record in contrast_records])
    for record, value in zip(contrast_records, adjusted):
        record["holm_p"] = value
    atomic_write_csv(pd.DataFrame(contrast_records), output / "paired_seed_contrasts.csv")

    self_rows = overall_frame[overall_frame["method"] == "self_confidence"].set_index("seed")
    cl_rows = overall_frame[overall_frame["method"] == "cl_first"].set_index("seed")
    primary_budget = error_budget[error_budget["method"] == "cl_first"].set_index("seed")
    supported_labels = label_frame[
        (label_frame["method"] == "cl_first")
        & (label_frame["flip_direction"] == "all")
        & (label_frame["label_name"].isin(LABELS))
    ]
    gate_conditions = {
        "auprc_above_prevalence_all_seeds": bool(
            (self_rows["auprc"] > self_rows["prevalence"]).all()
        ),
        "cl_hard_enrichment_above_one_all_seeds": bool(
            (cl_rows["enrichment"] > 1.0).all()
        ),
        "cl_first_above_random_mean_all_seeds": bool(
            (primary_budget["tp"] > primary_budget["random_mean_tp"]).all()
        ),
        "no_supported_label_repeated_failure": bool(
            not (
                supported_labels.assign(failed=supported_labels["enrichment"] < 1.0)
                .groupby("label_name")["failed"]
                .sum()
                .ge(3)
                .any()
            )
        ),
    }
    final_summary = {
        "protocol": PROTOCOL_NAME,
        "verification_status": "ANALYZED",
        "seeds": seeds,
        "labels": LABELS,
        "primary_gate_pass": bool(all(gate_conditions.values())),
        "primary_gate_conditions": gate_conditions,
        "incremental_cl_filter_claim_separate": True,
        "seed_quality": summaries,
        "paired_seed_contrasts": contrast_records,
        "bootstrap_ci": bootstrap_ci,
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(output / "evaluation_summary.json", json.dumps(final_summary, indent=2))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    plot = overall_frame.pivot(index="seed", columns="method", values="auprc")
    for method in ["cl_first", "self_confidence", "predictive_entropy"]:
        ax.plot(plot.index.astype(str), plot[method], marker="o", label=method.replace("_", " "))
    prevalence = overall_frame.groupby("seed")["prevalence"].first()
    ax.plot(plot.index.astype(str), prevalence.loc[plot.index], linestyle="--", color="gray", label="error prevalence")
    ax.set_xlabel("Experiment seed")
    ax.set_ylabel("Injected-error AUPRC")
    ax.set_title("VinDr Known-GT Label-Error Detection")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "injected_error_auprc_by_seed.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    budget_plot = budget_frame.groupby(["budget_name", "method"], sort=False)["recall"].mean().reset_index()
    names = ["1pct", "2pct", "5pct", "10pct", "true_error_count"]
    x_values = np.arange(len(names))
    for method in ["cl_first", "self_confidence", "predictive_entropy"]:
        values = budget_plot[budget_plot["method"] == method].set_index("budget_name").loc[names, "recall"]
        ax.plot(x_values, values, marker="o", label=method.replace("_", " "))
    ax.set_xticks(x_values, ["1%", "2%", "5%", "10%", "# true errors"])
    ax.set_xlabel("Review budget")
    ax.set_ylabel("Injected-error recall")
    ax.set_title("Equal-Budget Error Capture")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "error_recall_by_budget.png", dpi=180)
    plt.close(fig)

    atomic_write_text(output / ".evaluation_complete", "complete\n")
    print(json.dumps(final_summary, indent=2), flush=True)


def verify(args: argparse.Namespace) -> None:
    experiment_root = Path(args.experiment_root)
    seeds = parse_seeds(args.seeds)
    if not (experiment_root / ".prepare_complete").is_file():
        raise FileNotFoundError("Experiment prepare marker is missing")
    if not (experiment_root / "features" / ".features_complete").is_file():
        raise FileNotFoundError("Feature marker is missing")
    for seed in seeds:
        prepared = experiment_root / f"seed_{seed}" / "prepared"
        blind_run = experiment_root / f"seed_{seed}" / "blind_run"
        if not (prepared / ".prepare_complete").is_file():
            raise FileNotFoundError(f"Seed {seed} prepare marker is missing")
        if not (blind_run / ".blind_run_complete").is_file():
            raise FileNotFoundError(f"Seed {seed} blind-run marker is missing")
        blind = pd.read_csv(prepared / "blind_noisy_cohort.csv")
        evidence = pd.read_csv(blind_run / "entry_evidence.csv")
        private = pd.read_csv(prepared / "private_reference.csv")
        validate_blind_columns(blind)
        if len(blind) != 3000 or len(evidence) != 18000 or len(private) != 18000:
            raise ValueError(f"Seed {seed} row-count verification failed")
    if args.evaluation_dir:
        evaluation = Path(args.evaluation_dir)
        if not (evaluation / ".evaluation_complete").is_file():
            raise FileNotFoundError("Evaluation marker is missing")
        summary = json.loads((evaluation / "evaluation_summary.json").read_text())
        if summary["seeds"] != seeds:
            raise ValueError("Evaluation seed set differs from requested seed set")
    marker = experiment_root / ".benchmark_verified"
    atomic_write_text(marker, "complete\n")
    print(json.dumps({"protocol": PROTOCOL_NAME, "seeds": seeds, "verification": "passed"}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--labels-csv", type=Path, required=True)
    prepare_parser.add_argument("--conversion-manifest", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    prepare_parser.add_argument("--n-splits", type=int, default=4)
    prepare_parser.add_argument("--noise-fraction", type=float, default=0.20)
    prepare_parser.set_defaults(func=prepare)

    features_parser = subparsers.add_parser("extract-features")
    features_parser.add_argument("--image-index", type=Path, required=True)
    features_parser.add_argument("--image-root", type=Path, required=True)
    features_parser.add_argument("--output-dir", type=Path, required=True)
    features_parser.add_argument("--xrv-cache-dir", required=True)
    features_parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    features_parser.add_argument("--batch-size", type=int, default=32)
    features_parser.add_argument("--num-workers", type=int, default=4)
    features_parser.add_argument("--seed", type=int, default=13)
    features_parser.set_defaults(func=extract_features)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--blind-cohort", type=Path, required=True)
    run_parser.add_argument("--features", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--n-splits", type=int, default=4)
    run_parser.add_argument("--epochs", type=int, default=50)
    run_parser.add_argument("--early-stopping-patience", type=int, default=8)
    run_parser.add_argument("--learning-rate", type=float, default=1e-3)
    run_parser.add_argument("--batch-size", type=int, default=128)
    run_parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    run_parser.set_defaults(func=run_blind)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--experiment-root", type=Path, required=True)
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    evaluate_parser.add_argument("--random-iterations", type=int, default=10_000)
    evaluate_parser.add_argument("--bootstrap-iterations", type=int, default=1_000)
    evaluate_parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    evaluate_parser.set_defaults(func=evaluate)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--experiment-root", type=Path, required=True)
    verify_parser.add_argument("--evaluation-dir", type=Path)
    verify_parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    verify_parser.set_defaults(func=verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
