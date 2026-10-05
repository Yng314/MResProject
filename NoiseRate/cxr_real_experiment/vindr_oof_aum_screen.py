#!/usr/bin/env python3
"""Screen an OOF AUM-style training-dynamics score on VinDr known errors."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from cxr_real_noise_validation_smoke import MaskedBCEWithLogitsLoss
from vindr_detector_alternatives_benchmark import active_label_cleaning_score
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
from vindr_mobilenet_oof import build_evidence, create_mobilenet, make_loader


PROTOCOL_NAME = "vindr_oof_aum_screen_v1"


def train_fold(
    blind: pd.DataFrame,
    labels: np.ndarray,
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
    outer_sign = 2.0 * labels[outer_validation].astype(np.float32) - 1.0

    model = create_mobilenet().to(device)
    criterion = MaskedBCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    margin_sum = np.zeros_like(outer_sign, dtype=np.float64)
    margin_epochs = 0
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

        outer_logits = []
        with torch.no_grad():
            for images, _, _ in outer_loader:
                outer_logits.append(model(images.to(device, non_blocking=True)).cpu().numpy())
        epoch_logits = np.vstack(outer_logits).astype(np.float32)
        if epoch_logits.shape != outer_sign.shape or not np.isfinite(epoch_logits).all():
            raise RuntimeError("Outer-fold epoch logits are incomplete")
        margin_sum += outer_sign * epoch_logits
        margin_epochs += 1

        records.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / max(train_rows, 1),
                "inner_validation_loss": inner_loss,
                "outer_assigned_margin_mean": float(np.mean(outer_sign * epoch_logits)),
            }
        )
        print(
            f"[oof-aum] epoch={epoch}/{args.epochs} "
            f"train={records[-1]['train_loss']:.6f} inner={inner_loss:.6f} "
            f"margin={records[-1]['outer_assigned_margin_mean']:.6f}",
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

    if best_state is None or margin_epochs == 0:
        raise RuntimeError("Training did not produce a valid state or trajectory")
    model.load_state_dict(best_state)
    model.eval()
    probabilities = []
    with torch.no_grad():
        for images, _, _ in outer_loader:
            probabilities.append(torch.sigmoid(model(images.to(device, non_blocking=True))).cpu().numpy())
    result = np.vstack(probabilities).astype(np.float32)
    mean_margin = (margin_sum / margin_epochs).astype(np.float32)
    del model, optimizer, train_loader, inner_loader, outer_loader
    torch.cuda.empty_cache()
    return result, mean_margin, records, best_epoch, best_loss


def run_blind(args: argparse.Namespace) -> None:
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
        raise ValueError("Blind cohort must contain 3,000 unique images")
    aligned = blind[["image_id", "image_path"]].merge(
        index[["image_id", "image_path"]], on="image_id", suffixes=("_blind", "_index"), validate="one_to_one"
    )
    if len(aligned) != 3_000 or not aligned["image_path_blind"].eq(aligned["image_path_index"]).all():
        raise ValueError("Blind cohort and image index do not align")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("OOF AUM screen requires CUDA")

    labels = blind[LABELS].to_numpy(dtype=np.int64)
    folds = blind["fold_id"].to_numpy(dtype=np.int64)
    oof = np.full(labels.shape, np.nan, dtype=np.float32)
    mean_margin = np.full(labels.shape, np.nan, dtype=np.float32)
    history_records = []
    best_epochs = []
    for fold_id in range(args.n_splits):
        outer_train = np.flatnonzero(folds != fold_id)
        outer_validation = np.flatnonzero(folds == fold_id)
        inner_train, inner_validation, inner_split_seed = build_inner_split(
            outer_train, labels, seed=args.seed * 100 + fold_id
        )
        probabilities, margins, records, best_epoch, best_loss = train_fold(
            blind, labels, inner_train, inner_validation, outer_validation,
            image_root, args.seed * 100 + fold_id, args,
        )
        oof[outer_validation] = probabilities
        mean_margin[outer_validation] = margins
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
            history_records.append(record)
        print(f"[oof-aum] fold={fold_id} complete best_epoch={best_epoch}", flush=True)
    if not np.isfinite(oof).all() or not np.isfinite(mean_margin).all():
        raise RuntimeError("OOF probability or AUM matrix is incomplete")

    oof_frame, entries, issue = build_evidence(blind, labels, oof)
    aum_records = []
    for row_index, row in blind.iterrows():
        for label_index, label in enumerate(LABELS):
            aum_records.append(
                {
                    "image_id": str(row["image_id"]),
                    "fold_id": int(row["fold_id"]),
                    "label_index": label_index,
                    "label_name": label,
                    "noisy_label": int(labels[row_index, label_index]),
                    "mean_assigned_margin": float(mean_margin[row_index, label_index]),
                    "score_oof_aum": float(-mean_margin[row_index, label_index]),
                }
            )
    aum_frame = pd.DataFrame(aum_records)
    validate_blind_columns(aum_frame)
    oof_path = output / "oof_predictions.csv"
    evidence_path = output / "entry_evidence.csv"
    aum_path = output / "oof_aum_scores.csv"
    atomic_write_csv(oof_frame, oof_path)
    atomic_write_csv(entries, evidence_path)
    atomic_write_csv(aum_frame, aum_path)
    atomic_write_csv(pd.DataFrame(history_records), output / "training_history.csv")
    atomic_write_csv(fold_support(blind, args.n_splits), output / "fold_support.csv")
    summary = {
        "protocol": PROTOCOL_NAME,
        "outcome_blind": True,
        "seed": int(args.seed),
        "samples": 3_000,
        "entries": 18_000,
        "n_splits": int(args.n_splits),
        "score_definition": "negative mean assigned-label binary logit margin over outer-fold epoch checkpoints",
        "best_epochs": best_epochs,
        "epochs_max": int(args.epochs),
        "early_stopping_patience": int(args.early_stopping_patience),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "cl_issue_entries": int(issue.sum()),
        "blind_cohort_sha256": sha256_file(blind_path),
        "image_index_sha256": sha256_file(image_index_path),
        "oof_predictions_sha256": sha256_file(oof_path),
        "entry_evidence_sha256": sha256_file(evidence_path),
        "oof_aum_scores_sha256": sha256_file(aum_path),
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(output / "blind_run_summary.json", json.dumps(summary, indent=2))
    atomic_write_text(output / ".blind_run_complete", "complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def top_recall(frame: pd.DataFrame, score_column: str, budget: int) -> tuple[int, float]:
    selected = frame.sort_values(
        [score_column, "image_id", "label_name"],
        ascending=[False, True, True],
        kind="mergesort",
    ).head(budget)
    found = int(selected["injected_error"].sum())
    return found, float(found / frame["injected_error"].sum())


def evaluate(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    if not (output / ".blind_run_complete").is_file():
        raise FileNotFoundError("Blind AUM run is incomplete")
    evaluation = output / "private_evaluation"
    evaluation.mkdir(parents=False, exist_ok=False)
    evidence = pd.read_csv(output / "entry_evidence.csv")
    aum = pd.read_csv(output / "oof_aum_scores.csv")
    private = pd.read_csv(args.private_reference)
    require_columns(private, ["image_id", "label_name", "injected_error"], "private reference")
    blind = evidence.merge(
        aum[["image_id", "label_name", "score_oof_aum"]],
        on=["image_id", "label_name"], validate="one_to_one",
    )
    merged = blind.merge(
        private[["image_id", "label_name", "injected_error"]],
        on=["image_id", "label_name"], validate="one_to_one",
    )
    truth = merged["injected_error"].to_numpy(dtype=bool)
    true_errors = int(truth.sum())
    if len(merged) != 18_000 or true_errors <= 0:
        raise RuntimeError("Private AUM evaluation grid is invalid")
    merged["score_active_label_cleaning"] = active_label_cleaning_score(
        merged["noisy_label"].to_numpy(), merged["oof_probability"].to_numpy()
    )
    methods = {
        "oof_label_incompatibility": "cl_first_score",
        "active_label_cleaning": "score_active_label_cleaning",
        "oof_aum": "score_oof_aum",
    }
    records = []
    for method, column in methods.items():
        found, recall = top_recall(merged, column, true_errors)
        records.append(
            {
                "seed": int(args.seed),
                "method": method,
                "entries": int(len(merged)),
                "true_errors": true_errors,
                "auprc": float(average_precision_score(truth, merged[column])),
                "reviewed_entries": true_errors,
                "errors_found": found,
                "recall_at_error_count": recall,
            }
        )
    metrics = pd.DataFrame(records)
    atomic_write_csv(metrics, evaluation / "detector_metrics.csv")
    atomic_write_text(
        evaluation / "evaluation_summary.json",
        json.dumps(
            {
                "protocol": PROTOCOL_NAME,
                "seed": int(args.seed),
                "private_reference_sha256": sha256_file(Path(args.private_reference)),
                "metrics": records,
            },
            indent=2,
        ),
    )
    atomic_write_text(evaluation / ".evaluation_complete", "complete\n")
    atomic_write_text(output / ".benchmark_verified", "verified\n")
    print(metrics.to_string(index=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    blind = subparsers.add_parser("run-blind")
    blind.add_argument("--blind-cohort", type=Path, required=True)
    blind.add_argument("--image-index", type=Path, required=True)
    blind.add_argument("--image-root", type=Path, required=True)
    blind.add_argument("--output-dir", type=Path, required=True)
    blind.add_argument("--seed", type=int, required=True)
    blind.add_argument("--n-splits", type=int, default=4)
    blind.add_argument("--epochs", type=int, default=100)
    blind.add_argument("--early-stopping-patience", type=int, default=10)
    blind.add_argument("--learning-rate", type=float, default=1e-3)
    blind.add_argument("--batch-size", type=int, default=32)
    blind.add_argument("--num-workers", type=int, default=4)
    blind.add_argument("--device", choices=["cuda"], default="cuda")
    blind.set_defaults(function=run_blind)
    private = subparsers.add_parser("evaluate")
    private.add_argument("--output-dir", type=Path, required=True)
    private.add_argument("--private-reference", type=Path, required=True)
    private.add_argument("--seed", type=int, required=True)
    private.set_defaults(function=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
