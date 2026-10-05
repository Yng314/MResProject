#!/usr/bin/env python3
"""Outcome-blind REFLACX Phase-3 confident-learning pilot.

The command is intentionally split into three transactions:

1. ``prepare`` writes a blind training cohort and a separate private reference.
2. ``run`` receives only the blind cohort and generates OOF/CL evidence.
3. ``evaluate`` joins the completed evidence to the private reference.

This prevents REFLACX certainty labels from entering model fitting, OOF
prediction, confident-learning issue estimation, or candidate selection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tv_models
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from cxr_real_full_train_eval_cleanlab_xrv12 import RealCXRXRV12Dataset
from cxr_real_noise_validation_smoke import (
    MaskedBCEWithLogitsLoss,
    estimate_issues,
    rank_from_quality,
    safe_nanmean,
)


TARGET_TO_REFLACX = {
    "Atelectasis": "Atelectasis",
    "Consolidation": "Consolidation",
    "Cardiomegaly": "Enlarged cardiac silhouette",
    "Edema": "Pulmonary edema",
    "Lung Lesion": "Lung nodule or mass",
    "Pneumothorax": "Pneumothorax",
}
TARGET_LABELS = list(TARGET_TO_REFLACX)
ID_COLUMNS = ["pool_row_id", "subject_id", "study_id", "dicom_id"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def binary_projection(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw, dtype=np.float32)
    binary = np.full_like(raw, np.nan, dtype=np.float32)
    valid = np.isin(raw, [-1.0, 0.0, 1.0])
    binary[(raw == -1.0) | (raw == 1.0)] = 1.0
    binary[raw == 0.0] = 0.0
    return binary, valid


def build_group_folds(
    rows: pd.DataFrame,
    binary: np.ndarray,
    valid: np.ndarray,
    n_splits: int,
    seed: int,
    trials: int = 2000,
) -> np.ndarray:
    subjects = rows["subject_id"].drop_duplicates().to_numpy(dtype=np.int64)
    subject_to_indices = {
        int(subject): np.flatnonzero(rows["subject_id"].to_numpy(dtype=np.int64) == int(subject))
        for subject in subjects
    }
    vectors = {}
    for subject, indices in subject_to_indices.items():
        parts = [np.array([len(indices)], dtype=float)]
        for label_index in range(binary.shape[1]):
            mask = valid[indices, label_index]
            values = binary[indices, label_index][mask]
            parts.append(
                np.array(
                    [
                        float(mask.sum()),
                        float((values == 0).sum()),
                        float((values == 1).sum()),
                    ]
                )
            )
        vectors[subject] = np.concatenate(parts)

    total = np.sum(np.stack(list(vectors.values())), axis=0)
    target = total / float(n_splits)
    scale = np.maximum(target, 1.0)
    rng = np.random.default_rng(seed)
    best_assignment: dict[int, int] | None = None
    best_score = float("inf")

    for _ in range(trials):
        permutation = rng.permutation(subjects)
        assignment = {int(subject): int(index % n_splits) for index, subject in enumerate(permutation)}
        fold_vectors = np.zeros((n_splits, len(total)), dtype=float)
        for subject, fold_id in assignment.items():
            fold_vectors[fold_id] += vectors[subject]
        relative = (fold_vectors - target[None, :]) / scale[None, :]
        score = float(np.mean(relative**2) + np.max(np.abs(relative)) * 0.05)
        if score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError("Unable to construct subject-grouped folds")
    folds = rows["subject_id"].map(best_assignment).to_numpy(dtype=np.int64)
    if np.any(folds < 0):
        raise RuntimeError("Fold assignment contains unmapped subjects")
    return folds


def fold_support_table(
    rows: pd.DataFrame,
    binary: np.ndarray,
    valid: np.ndarray,
    folds: np.ndarray,
) -> pd.DataFrame:
    records = []
    for fold_id in sorted(np.unique(folds)):
        indices = np.flatnonzero(folds == fold_id)
        for label_index, label_name in enumerate(TARGET_LABELS):
            mask = valid[indices, label_index]
            values = binary[indices, label_index][mask]
            records.append(
                {
                    "fold_id": int(fold_id),
                    "label_index": int(label_index),
                    "label_name": label_name,
                    "samples": int(len(indices)),
                    "subjects": int(rows.iloc[indices]["subject_id"].nunique()),
                    "valid_entries": int(mask.sum()),
                    "negative_entries": int((values == 0).sum()),
                    "positive_entries": int((values == 1).sum()),
                }
            )
    support = pd.DataFrame(records)
    invalid = support[
        (support["valid_entries"] < 10)
        | (support["negative_entries"] == 0)
        | (support["positive_entries"] == 0)
    ]
    if not invalid.empty:
        raise ValueError(f"Fold support is insufficient:\n{invalid.to_string(index=False)}")
    return support


def prepare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase3_path = Path(args.phase3_csv)
    chexpert_path = Path(args.chexpert_csv)
    image_root = Path(args.image_root)

    phase3 = pd.read_csv(phase3_path)
    required_reference = set(TARGET_TO_REFLACX.values())
    missing_reference = sorted(required_reference - set(phase3.columns))
    if missing_reference:
        raise ValueError(f"REFLACX Phase 3 is missing columns: {missing_reference}")
    if phase3["dicom_id"].duplicated().any():
        raise ValueError("Phase 3 must contain one reader observation per dicom_id")

    cohort = phase3[["subject_id", "dicom_id", "image"]].copy()
    cohort["study_id"] = (
        cohort["image"].astype(str).str.extract(r"/s(\d+)/", expand=False).astype(np.int64)
    )
    cohort["image_path"] = (
        cohort["image"]
        .astype(str)
        .str.extract(r"/files/(p\d{2}/p\d+/s\d+/[^/]+)\.dcm$", expand=False)
        + ".jpg"
    )
    if cohort["study_id"].isna().any() or cohort["image_path"].isna().any():
        raise ValueError("Unable to parse MIMIC study/image paths from REFLACX metadata")

    chexpert = pd.read_csv(
        chexpert_path,
        usecols=["subject_id", "study_id"] + TARGET_LABELS,
    )
    if chexpert.duplicated(["subject_id", "study_id"]).any():
        raise ValueError("CheXpert table must be unique by subject_id/study_id")
    blind = cohort.merge(
        chexpert,
        on=["subject_id", "study_id"],
        how="left",
        validate="one_to_one",
    )
    raw = blind[TARGET_LABELS].to_numpy(dtype=np.float32)
    binary, valid = binary_projection(raw)
    eligible = valid.any(axis=1)
    blind = blind.loc[eligible].reset_index(drop=True)
    raw = raw[eligible]
    binary = binary[eligible]
    valid = valid[eligible]

    image_exists = blind["image_path"].map(lambda relative: (image_root / relative).is_file())
    if not image_exists.all():
        missing = blind.loc[~image_exists, "image_path"].head(10).tolist()
        raise FileNotFoundError(f"Missing REFLACX-linked JPG files: {missing}")

    blind.insert(0, "pool_row_id", np.arange(len(blind), dtype=np.int64))
    folds = build_group_folds(
        rows=blind,
        binary=binary,
        valid=valid,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    blind["fold_id"] = folds
    support = fold_support_table(blind, binary, valid, folds)

    reference_source = phase3.set_index("dicom_id")
    reference = blind[ID_COLUMNS].copy()
    for target, source in TARGET_TO_REFLACX.items():
        reference[f"certainty__{target}"] = (
            reference["dicom_id"].map(reference_source[source]).astype(np.int64)
        )

    blind_path = output_dir / "blind_cohort.csv"
    reference_path = output_dir / "private_reference.csv"
    support_path = output_dir / "fold_support.csv"
    blind.to_csv(blind_path, index=False)
    reference.to_csv(reference_path, index=False)
    support.to_csv(support_path, index=False)

    manifest = {
        "protocol": "reflacx_phase3_cl_only_pilot_v1",
        "phase": 3,
        "seed": int(args.seed),
        "n_splits": int(args.n_splits),
        "target_to_reflacx": TARGET_TO_REFLACX,
        "binary_projection": {"-1": 1, "0": 0, "1": 1, "missing": "invalid"},
        "reference_policy_primary": "certainty >= 3",
        "reference_policy_sensitivity": "certainty >= 4 with certainty 3 excluded",
        "phase3_images": int(len(phase3)),
        "eligible_images": int(len(blind)),
        "eligible_subjects": int(blind["subject_id"].nunique()),
        "valid_entries": int(valid.sum()),
        "blind_columns": list(blind.columns),
        "blind_sha256": sha256_file(blind_path),
        "private_reference_sha256": sha256_file(reference_path),
        "fold_support_sha256": sha256_file(support_path),
        "protocol_code_sha256": sha256_file(Path(__file__)),
        "sources": {
            "phase3_csv": str(phase3_path.resolve()),
            "chexpert_csv": str(chexpert_path.resolve()),
            "image_root": str(image_root.resolve()),
        },
    }
    (output_dir / "preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


def create_mobilenet(n_labels: int) -> nn.Module:
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
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_labels)
    return model


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    patience: int,
    fold_id: int,
    n_splits: int,
) -> np.ndarray:
    criterion = MaskedBCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for images, labels, mask in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels, mask)
            loss.backward()
            optimizer.step()

        model.eval()
        loss_sum = 0.0
        entry_count = 0
        with torch.no_grad():
            for images, labels, mask in validation_loader:
                images = images.to(device)
                labels = labels.to(device)
                mask = mask.to(device)
                logits = model(images)
                elementwise = nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    reduction="none",
                )
                loss_sum += float((elementwise * mask.float()).sum().item())
                entry_count += int(mask.sum().item())
        validation_loss = loss_sum / max(entry_count, 1)
        print(
            f"[oof] fold {fold_id + 1}/{n_splits} "
            f"epoch {epoch + 1}/{epochs} val_loss={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[oof] fold {fold_id + 1} early stopping", flush=True)
                break

    if best_state is None:
        raise RuntimeError(f"Fold {fold_id} did not produce a model state")
    model.load_state_dict(best_state)
    probabilities = []
    model.eval()
    with torch.no_grad():
        for images, _, _ in validation_loader:
            probabilities.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
    return np.vstack(probabilities)


def validate_blind_cohort(blind: pd.DataFrame, n_splits: int) -> tuple[np.ndarray, np.ndarray]:
    forbidden = [
        column
        for column in blind.columns
        if "reference" in column.lower() or "certainty" in column.lower()
    ]
    if forbidden:
        raise ValueError(f"Blind cohort contains forbidden outcome columns: {forbidden}")
    required = set(ID_COLUMNS + ["image_path", "fold_id"] + TARGET_LABELS)
    missing = sorted(required - set(blind.columns))
    if missing:
        raise ValueError(f"Blind cohort is missing columns: {missing}")
    if blind["dicom_id"].duplicated().any():
        raise ValueError("Blind cohort contains duplicate images")
    if sorted(blind["fold_id"].unique().tolist()) != list(range(n_splits)):
        raise ValueError("Blind cohort fold ids do not match requested n_splits")
    if blind.groupby("subject_id")["fold_id"].nunique().max() != 1:
        raise ValueError("A subject appears in more than one OOF fold")
    raw = blind[TARGET_LABELS].to_numpy(dtype=np.float32)
    binary, valid = binary_projection(raw)
    if not valid.any(axis=1).all():
        raise ValueError("Blind cohort contains samples with no valid target entries")
    fold_support_table(blind, binary, valid, blind["fold_id"].to_numpy(dtype=int))
    return binary, valid


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blind_path = Path(args.blind_cohort)
    image_root = Path(args.image_root)
    blind = pd.read_csv(blind_path)
    binary, valid = validate_blind_cohort(blind, args.n_splits)
    if not blind["image_path"].map(lambda relative: (image_root / relative).is_file()).all():
        raise FileNotFoundError("At least one blind-cohort image is missing")

    if args.preflight_only:
        preflight_count = min(4, len(blind))
        dataset = RealCXRXRV12Dataset(
            rows=blind.iloc[:preflight_count].reset_index(drop=True),
            image_root=image_root,
            image_size=args.image_size,
            binary_labels=binary[:preflight_count],
            valid_mask=valid[:preflight_count],
        )
        images, labels, mask = next(
            iter(DataLoader(dataset, batch_size=min(2, preflight_count), shuffle=False))
        )
        model = create_mobilenet(len(TARGET_LABELS))
        with torch.no_grad():
            output = model(images)
            loss = MaskedBCEWithLogitsLoss()(output, labels, mask)
        if tuple(output.shape) != (len(images), len(TARGET_LABELS)):
            raise RuntimeError(f"Unexpected model output shape: {tuple(output.shape)}")
        if not torch.isfinite(loss):
            raise RuntimeError("Real-image preflight produced a non-finite loss")
        print(
            "Blind-cohort, real-image preprocessing, and six-label MobileNet "
            f"preflight passed (loss={float(loss):.6f})."
        )
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    set_seed(args.seed)
    raw = blind[TARGET_LABELS].to_numpy(dtype=np.float32)
    folds = blind["fold_id"].to_numpy(dtype=np.int64)
    oof = np.full_like(binary, np.nan, dtype=np.float32)

    for fold_id in range(args.n_splits):
        set_seed(args.seed + fold_id)
        train_indices = np.flatnonzero(folds != fold_id)
        validation_indices = np.flatnonzero(folds == fold_id)
        train_dataset = RealCXRXRV12Dataset(
            rows=blind.iloc[train_indices].reset_index(drop=True),
            image_root=image_root,
            image_size=args.image_size,
            binary_labels=binary[train_indices],
            valid_mask=valid[train_indices],
        )
        validation_dataset = RealCXRXRV12Dataset(
            rows=blind.iloc[validation_indices].reset_index(drop=True),
            image_root=image_root,
            image_size=args.image_size,
            binary_labels=binary[validation_indices],
            valid_mask=valid[validation_indices],
        )
        generator = torch.Generator()
        generator.manual_seed(args.seed + fold_id)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        model = create_mobilenet(len(TARGET_LABELS))
        oof[validation_indices] = train_fold(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            patience=args.early_stopping_patience,
            fold_id=fold_id,
            n_splits=args.n_splits,
        )

    if not np.isfinite(oof).all():
        raise RuntimeError("OOF predictions are incomplete")
    issue, quality_self, quality_margin, quality_entropy, _, _, _ = estimate_issues(
        y_binary=binary,
        valid_mask=valid,
        y_prob=oof,
    )
    sample_quality = np.array([safe_nanmean(row) for row in quality_self], dtype=float)
    sample_issue = issue.any(axis=1)
    sample_rank = rank_from_quality(sample_quality)
    issue_indices = np.flatnonzero(sample_issue)
    if len(issue_indices) == 0:
        raise RuntimeError("Confident learning did not identify any issue samples")
    ranked_issue_indices = issue_indices[
        np.argsort(sample_quality[issue_indices], kind="stable")
    ]
    selected_count = max(1, int(round(len(ranked_issue_indices) * args.top_fraction)))
    selected_indices = ranked_issue_indices[:selected_count]
    selected_sample = np.zeros(len(blind), dtype=bool)
    selected_sample[selected_indices] = True
    selected_candidate = issue & selected_sample[:, None]

    oof_frame = blind[ID_COLUMNS + ["fold_id"]].copy()
    for label_index, label_name in enumerate(TARGET_LABELS):
        oof_frame[f"probability__{label_name}"] = oof[:, label_index]
    oof_frame.to_csv(output_dir / "oof_predictions.csv", index=False)

    sample_frame = blind[ID_COLUMNS + ["image_path", "fold_id"]].copy()
    sample_frame["est_issue_sample"] = sample_issue.astype(int)
    sample_frame["sample_quality_self_confidence"] = sample_quality
    sample_frame["issue_rank_self_confidence"] = sample_rank
    sample_frame["est_issue_entry_count"] = issue.sum(axis=1)
    sample_frame["selected_top20_of_issue_samples"] = selected_sample.astype(int)
    sample_frame.to_csv(output_dir / "sample_evidence.csv", index=False)

    entry_records = []
    for row_index, row in blind.iterrows():
        for label_index, label_name in enumerate(TARGET_LABELS):
            entry_records.append(
                {
                    **{column: row[column] for column in ID_COLUMNS},
                    "fold_id": int(row["fold_id"]),
                    "label_index": int(label_index),
                    "label_name": label_name,
                    "raw_label": raw[row_index, label_index],
                    "binary_label": binary[row_index, label_index],
                    "valid_label": int(valid[row_index, label_index]),
                    "pred_probability": float(oof[row_index, label_index]),
                    "entry_quality_self_confidence": quality_self[row_index, label_index],
                    "entry_quality_normalized_margin": quality_margin[row_index, label_index],
                    "entry_quality_confidence_weighted_entropy": quality_entropy[row_index, label_index],
                    "est_issue_entry": int(issue[row_index, label_index]),
                    "selected_top20_candidate": int(selected_candidate[row_index, label_index]),
                }
            )
    entries = pd.DataFrame(entry_records)
    entries.to_csv(output_dir / "entry_evidence.csv", index=False)

    noisy_aurocs = []
    for label_index, label_name in enumerate(TARGET_LABELS):
        mask = valid[:, label_index]
        noisy_aurocs.append(
            {
                "label_name": label_name,
                "valid_entries": int(mask.sum()),
                "noisy_label_auroc": float(
                    roc_auc_score(binary[mask, label_index], oof[mask, label_index])
                ),
            }
        )
    pd.DataFrame(noisy_aurocs).to_csv(output_dir / "noisy_label_oof_auroc.csv", index=False)

    valid_entries = int(valid.sum())
    summary = {
        "protocol": "reflacx_phase3_cl_only_pilot_v1",
        "outcome_blind": True,
        "blind_cohort_sha256": sha256_file(blind_path),
        "seed": int(args.seed),
        "n_splits": int(args.n_splits),
        "epochs_max": int(args.epochs),
        "model_backbone": "mobilenet_v3_small_scratch",
        "samples": int(len(blind)),
        "subjects": int(blind["subject_id"].nunique()),
        "valid_entries": valid_entries,
        "cl_issue_entries": int(issue.sum()),
        "cl_issue_samples": int(sample_issue.sum()),
        "selected_issue_samples": int(selected_sample.sum()),
        "selected_candidate_entries": int(selected_candidate.sum()),
        "raw_entry_dqs": float(1.0 - issue.sum() / valid_entries),
        "raw_sample_dqs": float(1.0 - sample_issue.mean()),
        "entry_evidence_sha256": sha256_file(output_dir / "entry_evidence.csv"),
        "protocol_code_sha256": sha256_file(Path(__file__)),
    }
    (output_dir / "blind_run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / ".blind_run_complete").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def binary_detection_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray,
) -> dict[str, float]:
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
    finite = np.isfinite(score)
    average_precision = float("nan")
    roc_auc = float("nan")
    if finite.sum() > 1 and len(np.unique(truth[finite])) == 2:
        average_precision = float(average_precision_score(truth[finite], score[finite]))
        roc_auc = float(roc_auc_score(truth[finite], score[finite]))
    return {
        "n": int(len(truth)),
        "true_issues": int(truth.sum()),
        "predicted_issues": int(prediction.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "prevalence": prevalence,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "enrichment": safe_ratio(precision, prevalence),
        "average_precision": average_precision,
        "roc_auc": roc_auc,
    }


def evaluate_scope(entries: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, dict[str, float]]:
    if scope == "certainty_ge3":
        scoped = entries.copy()
        scoped["reference_binary"] = (scoped["certainty"] >= 3).astype(int)
    elif scope == "certainty_ge4_excluding3":
        scoped = entries[entries["certainty"] != 3].copy()
        scoped["reference_binary"] = (scoped["certainty"] >= 4).astype(int)
    else:
        raise ValueError(f"Unknown reference scope: {scope}")

    scoped["true_issue"] = (
        scoped["binary_label"].astype(int) != scoped["reference_binary"].astype(int)
    )
    scoped["suspicion_score"] = 1.0 - scoped["entry_quality_self_confidence"]
    issue_metrics = binary_detection_metrics(
        truth=scoped["true_issue"].to_numpy(),
        prediction=scoped["est_issue_entry"].to_numpy(),
        score=scoped["suspicion_score"].to_numpy(),
    )
    selected_metrics = binary_detection_metrics(
        truth=scoped["true_issue"].to_numpy(),
        prediction=scoped["selected_top20_candidate"].to_numpy(),
        score=scoped["suspicion_score"].to_numpy(),
    )
    valid_count = len(scoped)
    result = {
        "scope": scope,
        "valid_entries": int(valid_count),
        "true_quality": float(1.0 - scoped["true_issue"].mean()),
        "raw_entry_dqs": float(1.0 - scoped["est_issue_entry"].sum() / valid_count),
        "dqs_minus_true_quality": float(
            scoped["true_issue"].mean() - scoped["est_issue_entry"].sum() / valid_count
        ),
    }
    result.update({f"issue_{key}": value for key, value in issue_metrics.items()})
    result.update({f"selected_{key}": value for key, value in selected_metrics.items()})
    return scoped, result


def bootstrap_subjects(
    scoped: pd.DataFrame,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    subjects = scoped["subject_id"].drop_duplicates().to_numpy()
    groups = {subject: frame.copy() for subject, frame in scoped.groupby("subject_id")}
    rng = np.random.default_rng(seed)
    records = []
    for iteration in range(iterations):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        parts = []
        for draw_index, subject in enumerate(sampled):
            part = groups[subject].copy()
            part["bootstrap_subject"] = draw_index
            parts.append(part)
        replicate = pd.concat(parts, ignore_index=True)
        _, metrics = evaluate_scope(
            replicate.rename(columns={"reference_binary": "_unused"}),
            scope="certainty_ge3",
        )
        metrics["iteration"] = iteration
        records.append(metrics)
    return pd.DataFrame(records)


def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blind = pd.read_csv(args.blind_cohort)
    reference = pd.read_csv(args.private_reference)
    evidence = pd.read_csv(args.entry_evidence)
    validate_blind_cohort(blind, int(blind["fold_id"].nunique()))

    if evidence["valid_label"].eq(1).sum() == 0:
        raise ValueError("Entry evidence contains no valid labels")
    evidence = evidence[evidence["valid_label"] == 1].copy()
    if evidence.duplicated(ID_COLUMNS + ["label_name"]).any():
        raise ValueError("Entry evidence contains duplicate keys")
    if reference.duplicated(ID_COLUMNS).any():
        raise ValueError("Private reference contains duplicate keys")

    reference_long = reference.melt(
        id_vars=ID_COLUMNS,
        value_vars=[f"certainty__{label}" for label in TARGET_LABELS],
        var_name="certainty_label",
        value_name="certainty",
    )
    reference_long["label_name"] = reference_long["certainty_label"].str.removeprefix(
        "certainty__"
    )
    merged = evidence.merge(
        reference_long[ID_COLUMNS + ["label_name", "certainty"]],
        on=ID_COLUMNS + ["label_name"],
        how="left",
        validate="one_to_one",
    )
    if merged["certainty"].isna().any():
        raise ValueError("Reference join left missing certainty values")

    overall_records = []
    per_label_records = []
    primary_scoped = None
    for scope in ("certainty_ge3", "certainty_ge4_excluding3"):
        scoped, overall = evaluate_scope(merged, scope)
        overall_records.append(overall)
        if scope == "certainty_ge3":
            primary_scoped = scoped
        for label_name, label_frame in scoped.groupby("label_name", sort=False):
            _, label_metrics = evaluate_scope(label_frame, scope)
            label_metrics["label_name"] = label_name
            per_label_records.append(label_metrics)

    overall_frame = pd.DataFrame(overall_records)
    per_label_frame = pd.DataFrame(per_label_records)
    overall_frame.to_csv(output_dir / "overall_metrics.csv", index=False)
    per_label_frame.to_csv(output_dir / "per_label_metrics.csv", index=False)
    merged.to_csv(output_dir / "entry_evidence_with_private_reference.csv", index=False)

    if primary_scoped is None:
        raise RuntimeError("Primary reference scope was not evaluated")
    bootstrap = bootstrap_subjects(
        primary_scoped,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    bootstrap.to_csv(output_dir / "subject_bootstrap_replicates.csv", index=False)
    ci_fields = [
        "true_quality",
        "raw_entry_dqs",
        "dqs_minus_true_quality",
        "issue_precision",
        "issue_recall",
        "issue_enrichment",
        "issue_average_precision",
        "issue_roc_auc",
        "selected_precision",
        "selected_recall",
        "selected_enrichment",
    ]
    ci_records = []
    for field in ci_fields:
        values = pd.to_numeric(bootstrap[field], errors="coerce").dropna()
        ci_records.append(
            {
                "metric": field,
                "bootstrap_valid_replicates": int(len(values)),
                "ci_lower": float(values.quantile(0.025)),
                "ci_upper": float(values.quantile(0.975)),
            }
        )
    pd.DataFrame(ci_records).to_csv(output_dir / "subject_bootstrap_ci.csv", index=False)

    primary = overall_frame.set_index("scope").loc["certainty_ge3"]
    gate_pass = bool(
        primary["issue_enrichment"] > 1.0
        and primary["selected_enrichment"] > 1.0
        and primary["issue_average_precision"] > primary["issue_prevalence"]
    )
    summary = {
        "protocol": "reflacx_phase3_cl_only_pilot_v1",
        "reference_used_only_after_blind_run": True,
        "primary_reference_scope": "certainty >= 3",
        "gate_pass": gate_pass,
        "gate_definition": (
            "issue enrichment > 1, selected-candidate enrichment > 1, "
            "and AUPRC > disagreement prevalence"
        ),
        "primary_metrics": {
            key: (
                int(value)
                if isinstance(value, (np.integer,))
                else float(value)
                if isinstance(value, (np.floating,))
                else value
            )
            for key, value in primary.to_dict().items()
        },
        "blind_cohort_sha256": sha256_file(Path(args.blind_cohort)),
        "private_reference_sha256": sha256_file(Path(args.private_reference)),
        "entry_evidence_sha256": sha256_file(Path(args.entry_evidence)),
        "protocol_code_sha256": sha256_file(Path(__file__)),
    }
    (output_dir / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    primary_labels = per_label_frame[
        per_label_frame["scope"] == "certainty_ge3"
    ].copy()
    x = np.arange(len(primary_labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width, primary_labels["issue_prevalence"], width, label="Background disagreement")
    ax.bar(x, primary_labels["issue_precision"], width, label="CL issue precision")
    ax.bar(
        x + width,
        primary_labels["selected_precision"],
        width,
        label="Pipeline candidate precision",
    )
    ax.set_xticks(x, primary_labels["label_name"], rotation=25, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.set_title("REFLACX Phase 3: CL Detection by Label")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "cl_detection_by_label.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    values = [primary["true_quality"], primary["raw_entry_dqs"]]
    ax.bar(["Reference agreement", "Raw entry DQS"], values, color=["#4c78a8", "#e45756"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Quality")
    ax.set_title("State 0: Reference Quality vs DQS")
    for index, value in enumerate(values):
        ax.text(index, value + 0.02, f"{value:.3f}", ha="center")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "state0_reference_quality_vs_dqs.png", dpi=180)
    plt.close(fig)

    (output_dir / ".evaluation_complete").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--phase3-csv", required=True)
    prepare_parser.add_argument("--chexpert-csv", required=True)
    prepare_parser.add_argument("--image-root", required=True)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--seed", type=int, default=13)
    prepare_parser.add_argument("--n-splits", type=int, default=4)
    prepare_parser.set_defaults(func=prepare)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--blind-cohort", required=True)
    run_parser.add_argument("--image-root", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--seed", type=int, default=13)
    run_parser.add_argument("--n-splits", type=int, default=4)
    run_parser.add_argument("--epochs", type=int, default=50)
    run_parser.add_argument("--batch-size", type=int, default=32)
    run_parser.add_argument("--image-size", type=int, default=224)
    run_parser.add_argument("--learning-rate", type=float, default=1e-3)
    run_parser.add_argument("--early-stopping-patience", type=int, default=10)
    run_parser.add_argument("--num-workers", type=int, default=4)
    run_parser.add_argument("--top-fraction", type=float, default=0.20)
    run_parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    run_parser.add_argument("--preflight-only", action="store_true")
    run_parser.set_defaults(func=run)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--blind-cohort", required=True)
    evaluate_parser.add_argument("--private-reference", required=True)
    evaluate_parser.add_argument("--entry-evidence", required=True)
    evaluate_parser.add_argument("--output-dir", required=True)
    evaluate_parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    evaluate_parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    evaluate_parser.set_defaults(func=evaluate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
