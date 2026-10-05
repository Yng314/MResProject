#!/usr/bin/env python3
"""
Run K-fold OOF predictions on real MIMIC-CXR-JPG train data and export
cleanlab-compatible issue CSVs using the same schema expected by the existing
top-k sample/entry selection scripts.
"""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cxr_real_noise_validation_smoke import (  # noqa: E402
    LABEL_NAMES as DENSENET_LABEL_NAMES,
    MaskedBCEWithLogitsLoss,
    RealCXRSmokeDataset,
    create_model as create_standard_model,
    estimate_issues,
    filter_existing_rows,
    load_real_pool,
    project_raw_labels_to_binary,
    rank_from_quality,
    safe_nanmean,
    sample_rows,
    stringify_float_array,
)
from cxr_real_full_train_eval_cleanlab_xrv12 import (  # noqa: E402
    LABEL_NAMES as XRV12_LABEL_NAMES,
    RealCXRXRV12Dataset,
    apply_entry_exclusions,
    apply_entry_relabels,
    apply_sample_exclusions,
    create_model as create_xrv_model,
)


XRV12_BACKBONES = {
    "xrv_densenet121_direct",
    "xrv_densenet121_linearhead",
    "resnet18_scratch",
    "mobilenet_v3_small_scratch",
}


@dataclass
class OOFSmokeSummary:
    n_samples: int
    n_labels: int
    estimated_noise_rate_sample: float
    estimated_noise_rate_entry: float
    mean_sample_quality_self_confidence: float
    mean_sample_quality_normalized_margin: float
    mean_sample_quality_confidence_weighted_entropy: float
    mean_per_label_issue_rate: float
    seed: int
    n_splits: int
    epochs: int
    batch_size: int
    model_backbone: str
    allowed_views: str
    excluded_sample_count: int
    excluded_entry_count: int
    relabeled_entry_count: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_label_names(model_backbone: str) -> list[str]:
    if model_backbone in XRV12_BACKBONES:
        return list(XRV12_LABEL_NAMES)
    return list(DENSENET_LABEL_NAMES)


def build_dataset_and_model_factories(
    model_backbone: str,
    xrv_weights: str,
    xrv_cache_dir: str | None,
):
    if model_backbone in XRV12_BACKBONES:
        dataset_cls = RealCXRXRV12Dataset

        def make_model(n_labels: int):
            del n_labels
            return create_xrv_model(
                model_backbone=model_backbone,
                xrv_weights=xrv_weights,
                xrv_cache_dir=xrv_cache_dir,
            )

        return dataset_cls, make_model

    dataset_cls = RealCXRSmokeDataset

    def make_model(n_labels: int):
        return create_standard_model(model_backbone=model_backbone, n_labels=n_labels)

    return dataset_cls, make_model


def train_one_fold(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    early_stopping_patience: int,
    recover_best_weights: bool,
    fold_idx: int,
    n_folds: int,
) -> np.ndarray:
    criterion = MaskedBCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    best_state = None
    best_val_loss = float("inf")
    no_improve = 0

    for epoch_idx in range(epochs):
        print(f"[oof] fold {fold_idx}/{n_folds} epoch {epoch_idx + 1}/{epochs} training...")
        model.train()
        for xb, yb, mb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb, mb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                mb = mb.to(device)
                logits = model(xb)
                batch_loss = criterion(logits, yb, mb)
                batch_size = int(xb.size(0))
                val_loss_sum += float(batch_loss.item()) * batch_size
                val_count += batch_size

        val_loss = val_loss_sum / max(1, val_count)
        print(f"[oof] fold {fold_idx}/{n_folds} epoch {epoch_idx + 1}/{epochs} val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                print(f"[oof] fold {fold_idx}/{n_folds} early stopping")
                break

    if recover_best_weights and best_state is not None:
        model.load_state_dict(best_state)

    probs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, _, _ in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(probs)


def get_oof_probabilities(
    *,
    rows: pd.DataFrame,
    image_root: Path,
    y_binary: np.ndarray,
    valid_mask: np.ndarray,
    image_size: int,
    n_splits: int,
    seed: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    early_stopping_patience: int,
    recover_best_weights: bool,
    device: torch.device,
    model_backbone: str,
    xrv_weights: str,
    xrv_cache_dir: str | None,
    num_workers: int,
) -> np.ndarray:
    dataset_cls, make_model = build_dataset_and_model_factories(
        model_backbone=model_backbone,
        xrv_weights=xrv_weights,
        xrv_cache_dir=xrv_cache_dir,
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_probs = np.zeros_like(y_binary, dtype=np.float32)
    split_iter = list(kf.split(np.arange(len(rows))))
    total_folds = len(split_iter)

    for fold_no, (train_idx, val_idx) in enumerate(split_iter, start=1):
        train_rows = rows.iloc[train_idx].reset_index(drop=True)
        val_rows = rows.iloc[val_idx].reset_index(drop=True)
        train_ds = dataset_cls(
            rows=train_rows,
            image_root=image_root,
            image_size=image_size,
            binary_labels=y_binary[train_idx],
            valid_mask=valid_mask[train_idx],
        )
        val_ds = dataset_cls(
            rows=val_rows,
            image_root=image_root,
            image_size=image_size,
            binary_labels=y_binary[val_idx],
            valid_mask=valid_mask[val_idx],
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        model = make_model(n_labels=y_binary.shape[1])
        fold_probs = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
            recover_best_weights=recover_best_weights,
            fold_idx=fold_no,
            n_folds=total_folds,
        )
        oof_probs[val_idx] = fold_probs

    return oof_probs


def generic_per_label_summary_df(
    *,
    label_names: Sequence[str],
    y_binary: np.ndarray,
    valid_mask: np.ndarray,
    y_prob: np.ndarray,
    issue_mask: np.ndarray,
    quality_self: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for j, label_name in enumerate(label_names):
        mask = valid_mask[:, j]
        valid_count = int(mask.sum())
        pos_count = int(np.nansum(y_binary[mask, j])) if valid_count > 0 else 0
        neg_count = int(valid_count - pos_count)
        issue_rate = float(issue_mask[mask, j].mean()) if valid_count > 0 else float("nan")
        mean_quality = safe_nanmean(quality_self[mask, j]) if valid_count > 0 else float("nan")
        rows.append(
            {
                "label_index": int(j),
                "label_name": label_name,
                "valid_count": valid_count,
                "positive_count": pos_count,
                "negative_count": neg_count,
                "estimated_entry_issue_rate": issue_rate,
                "mean_entry_quality_self_confidence": mean_quality,
            }
        )
    return pd.DataFrame(rows)


def build_cleanlab_outputs(
    *,
    rows: pd.DataFrame,
    label_names: Sequence[str],
    raw_labels: np.ndarray,
    y_binary: np.ndarray,
    valid_mask: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    issue_mask, quality_self, quality_margin, quality_entropy, rank_self, rank_margin, rank_entropy = estimate_issues(
        y_binary=y_binary,
        valid_mask=valid_mask,
        y_prob=y_prob,
    )

    sample_quality_self = np.array([safe_nanmean(x) for x in quality_self], dtype=np.float32)
    sample_quality_margin = np.array([safe_nanmean(x) for x in quality_margin], dtype=np.float32)
    sample_quality_entropy = np.array([safe_nanmean(x) for x in quality_entropy], dtype=np.float32)
    sample_issue_mask = issue_mask.any(axis=1)
    sample_rank_self = rank_from_quality(sample_quality_self)
    sample_rank_margin = rank_from_quality(sample_quality_margin)
    sample_rank_entropy = rank_from_quality(sample_quality_entropy)

    sample_detail_df = pd.DataFrame(
        {
            "sample_local_index": np.arange(len(rows), dtype=np.int64),
            "pool_row_id": rows["pool_row_id"].to_numpy(dtype=np.int64),
            "subject_id": rows["subject_id"].to_numpy(dtype=np.int64),
            "study_id": rows["study_id"].to_numpy(dtype=np.int64),
            "dicom_id": rows["dicom_id"].astype(str).to_numpy(),
            "image_path": rows["image_path"].astype(str).to_numpy(),
            "est_issue_sample": sample_issue_mask.astype(int),
            "est_issue_entry_count": issue_mask.sum(axis=1).astype(int),
            "issue_rank_self_confidence": sample_rank_self,
            "issue_rank_normalized_margin": sample_rank_margin,
            "issue_rank_confidence_weighted_entropy": sample_rank_entropy,
            "sample_quality_self_confidence": sample_quality_self,
            "sample_quality_normalized_margin": sample_quality_margin,
            "sample_quality_confidence_weighted_entropy": sample_quality_entropy,
            "raw_labels_4class": [stringify_float_array(x) for x in raw_labels],
            "binary_labels_for_detection": [stringify_float_array(x) for x in y_binary],
            "valid_label_mask": ["|".join(map(str, m.astype(int).tolist())) for m in valid_mask],
            "pred_probs": ["|".join(f"{v:.6f}" for v in row.tolist()) for row in y_prob],
            "issue_entry_mask": ["|".join(map(str, m.astype(int).tolist())) for m in issue_mask],
        }
    )

    entry_rows: List[Dict[str, object]] = []
    for i in range(len(rows)):
        for j, label_name in enumerate(label_names):
            raw_val = raw_labels[i, j]
            binary_val = y_binary[i, j]
            entry_rows.append(
                {
                    "sample_local_index": int(i),
                    "pool_row_id": int(rows.iloc[i]["pool_row_id"]),
                    "subject_id": int(rows.iloc[i]["subject_id"]),
                    "study_id": int(rows.iloc[i]["study_id"]),
                    "dicom_id": str(rows.iloc[i]["dicom_id"]),
                    "image_path": str(rows.iloc[i]["image_path"]),
                    "label_index": int(j),
                    "label_name": label_name,
                    "raw_label": None if np.isnan(raw_val) else float(raw_val),
                    "binary_label": None if np.isnan(binary_val) else float(binary_val),
                    "valid_label": int(valid_mask[i, j]),
                    "pred_prob": float(y_prob[i, j]),
                    "est_issue_entry": int(issue_mask[i, j]),
                    "entry_quality_self_confidence": float(quality_self[i, j]) if np.isfinite(quality_self[i, j]) else np.nan,
                    "entry_quality_normalized_margin": float(quality_margin[i, j]) if np.isfinite(quality_margin[i, j]) else np.nan,
                    "entry_quality_confidence_weighted_entropy": float(quality_entropy[i, j]) if np.isfinite(quality_entropy[i, j]) else np.nan,
                    "entry_issue_rank_self_confidence": float(rank_self[i, j]) if np.isfinite(rank_self[i, j]) else np.nan,
                    "entry_issue_rank_normalized_margin": float(rank_margin[i, j]) if np.isfinite(rank_margin[i, j]) else np.nan,
                    "entry_issue_rank_confidence_weighted_entropy": float(rank_entropy[i, j]) if np.isfinite(rank_entropy[i, j]) else np.nan,
                }
            )
    entry_detail_df = pd.DataFrame(entry_rows)
    per_label_df = generic_per_label_summary_df(
        label_names=label_names,
        y_binary=y_binary,
        valid_mask=valid_mask,
        y_prob=y_prob,
        issue_mask=issue_mask,
        quality_self=quality_self,
    )
    summary_df = pd.DataFrame(
        [
            {
                "n_samples": int(len(rows)),
                "n_labels": int(len(label_names)),
                "estimated_noise_rate_sample": float(sample_issue_mask.mean()),
                "estimated_noise_rate_entry": float(issue_mask.mean()),
                "mean_sample_quality_self_confidence": safe_nanmean(sample_quality_self),
                "mean_sample_quality_normalized_margin": safe_nanmean(sample_quality_margin),
                "mean_sample_quality_confidence_weighted_entropy": safe_nanmean(sample_quality_entropy),
                "mean_per_label_issue_rate": float(np.nanmean(per_label_df["estimated_entry_issue_rate"])),
            }
        ]
    )

    sample_detail_df.to_csv(output_dir / "train_cleanlab_sample_details.csv", index=False)
    entry_detail_df.to_csv(output_dir / "train_cleanlab_entry_details.csv", index=False)
    sample_detail_df[sample_detail_df["est_issue_sample"] == 1].to_csv(
        output_dir / "train_cleanlab_sample_issues_only.csv",
        index=False,
    )
    entry_detail_df[entry_detail_df["est_issue_entry"] == 1].to_csv(
        output_dir / "train_cleanlab_entry_issues_only.csv",
        index=False,
    )
    per_label_df.to_csv(output_dir / "train_cleanlab_per_label_summary.csv", index=False)
    summary_df.to_csv(output_dir / "train_cleanlab_summary.csv", index=False)
    return sample_detail_df, entry_detail_df, per_label_df, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-fold OOF cleanlab smoke on real MIMIC-CXR-JPG train data.")
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--chexpert-csv", type=str, required=True)
    parser.add_argument("--split-csv", type=str, required=True)
    parser.add_argument("--metadata-csv", type=str, default=None)
    parser.add_argument("--allowed-views", nargs="*", default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--num-samples", type=int, default=1000, help="Use -1 to run on the full split.")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument(
        "--model-backbone",
        type=str,
        default="xrv_densenet121_linearhead",
        choices=[
            "lightcnn",
            "densenet121_pretrained",
            "xrv_densenet121_direct",
            "xrv_densenet121_linearhead",
            "resnet18_scratch",
            "mobilenet_v3_small_scratch",
        ],
    )
    parser.add_argument("--xrv-weights", type=str, default="densenet121-res224-all")
    parser.add_argument("--xrv-cache-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--exclude-sample-csv", type=str, default=None)
    parser.add_argument("--exclude-issue-col", type=str, default="est_issue_sample")
    parser.add_argument("--exclude-entry-csv", type=str, default=None)
    parser.add_argument("--exclude-entry-issue-col", type=str, default="est_issue_entry")
    parser.add_argument("--override-entry-csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    image_root = Path(args.image_root)
    chexpert_csv = Path(args.chexpert_csv)
    split_csv = Path(args.split_csv)
    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else None
    exclude_sample_csv = Path(args.exclude_sample_csv) if args.exclude_sample_csv else None
    exclude_entry_csv = Path(args.exclude_entry_csv) if args.exclude_entry_csv else None
    override_entry_csv = Path(args.override_entry_csv) if args.override_entry_csv else None
    label_names = get_label_names(args.model_backbone)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "results_oof_smoke" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Image root: {image_root}")
    print(f"CheXpert CSV: {chexpert_csv}")
    print(f"Split CSV: {split_csv}")
    print(f"Metadata CSV: {metadata_csv}")
    print(f"Allowed Views: {args.allowed_views}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"Model backbone: {args.model_backbone}")
    print(f"Label names ({len(label_names)}): {label_names}")

    pool_df = load_real_pool(
        chexpert_csv=chexpert_csv,
        split_csv=split_csv,
        split_name=args.train_split,
        metadata_csv=metadata_csv,
        allowed_views=args.allowed_views,
    )
    pool_df = filter_existing_rows(pool_df, image_root=image_root)
    before_sample_exclusions = len(pool_df)
    pool_df = apply_sample_exclusions(
        train_rows=pool_df,
        exclude_sample_csv=exclude_sample_csv,
        issue_col=args.exclude_issue_col,
    )
    excluded_sample_count = before_sample_exclusions - len(pool_df)
    if args.num_samples > 0:
        rows = sample_rows(pool_df, n_samples=args.num_samples, seed=args.seed, image_root=image_root)
    else:
        rows = pool_df.reset_index(drop=True)
    raw_labels, y_binary, valid_mask = project_raw_labels_to_binary(rows, label_names)
    raw_labels, y_binary, valid_mask, relabeled_entry_count = apply_entry_relabels(
        train_rows=rows,
        raw_labels=raw_labels,
        y_binary=y_binary,
        valid_mask=valid_mask,
        override_entry_csv=override_entry_csv,
    )
    raw_labels, y_binary, valid_mask, excluded_entry_count = apply_entry_exclusions(
        train_rows=rows,
        raw_labels=raw_labels,
        y_binary=y_binary,
        valid_mask=valid_mask,
        exclude_entry_csv=exclude_entry_csv,
        issue_col=args.exclude_entry_issue_col,
    )

    oof_probs = get_oof_probabilities(
        rows=rows,
        image_root=image_root,
        y_binary=y_binary,
        valid_mask=valid_mask,
        image_size=args.image_size,
        n_splits=args.n_splits,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        recover_best_weights=True,
        device=device,
        model_backbone=args.model_backbone,
        xrv_weights=args.xrv_weights,
        xrv_cache_dir=args.xrv_cache_dir,
        num_workers=args.num_workers,
    )

    oof_pred_df = pd.DataFrame(oof_probs, columns=label_names)
    oof_pred_df.insert(0, "dicom_id", rows["dicom_id"].astype(str).to_numpy())
    oof_pred_df.insert(0, "study_id", rows["study_id"].to_numpy(dtype=np.int64))
    oof_pred_df.insert(0, "subject_id", rows["subject_id"].to_numpy(dtype=np.int64))
    oof_pred_df.to_csv(output_dir / "train_oof_pred_probs.csv", index=False)

    sample_detail_df, entry_detail_df, per_label_df, cleanlab_summary_df = build_cleanlab_outputs(
        rows=rows,
        label_names=label_names,
        raw_labels=raw_labels,
        y_binary=y_binary,
        valid_mask=valid_mask,
        y_prob=oof_probs,
        output_dir=output_dir,
    )

    summary = OOFSmokeSummary(
        n_samples=int(len(rows)),
        n_labels=int(len(label_names)),
        estimated_noise_rate_sample=float(sample_detail_df["est_issue_sample"].mean()),
        estimated_noise_rate_entry=float(entry_detail_df["est_issue_entry"].mean()),
        mean_sample_quality_self_confidence=safe_nanmean(sample_detail_df["sample_quality_self_confidence"].to_numpy()),
        mean_sample_quality_normalized_margin=safe_nanmean(sample_detail_df["sample_quality_normalized_margin"].to_numpy()),
        mean_sample_quality_confidence_weighted_entropy=safe_nanmean(sample_detail_df["sample_quality_confidence_weighted_entropy"].to_numpy()),
        mean_per_label_issue_rate=float(np.nanmean(per_label_df["estimated_entry_issue_rate"])),
        seed=int(args.seed),
        n_splits=int(args.n_splits),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        model_backbone=str(args.model_backbone),
        allowed_views="" if not args.allowed_views else "|".join(args.allowed_views),
        excluded_sample_count=int(excluded_sample_count),
        excluded_entry_count=int(excluded_entry_count),
        relabeled_entry_count=int(relabeled_entry_count),
    )
    pd.DataFrame([summary.__dict__]).to_csv(output_dir / "oof_cleanlab_smoke_summary.csv", index=False)

    with open(output_dir / "oof_cleanlab_smoke_report.txt", "w", encoding="utf-8") as f:
        f.write("Real CXR OOF Cleanlab Smoke Report\n")
        f.write("=================================\n\n")
        for key, value in vars(args).items():
            f.write(f"{key}={value}\n")
        f.write("\n[OOF Summary]\n")
        for key, value in summary.__dict__.items():
            f.write(f"{key}={value}\n")
        f.write("\n[Cleanlab Summary]\n")
        f.write(cleanlab_summary_df.to_string(index=False))
        f.write("\n")

    print("\nSaved:")
    print(f"- OOF preds:       {output_dir / 'train_oof_pred_probs.csv'}")
    print(f"- Sample details:  {output_dir / 'train_cleanlab_sample_details.csv'}")
    print(f"- Entry details:   {output_dir / 'train_cleanlab_entry_details.csv'}")
    print(f"- Sample issues:   {output_dir / 'train_cleanlab_sample_issues_only.csv'}")
    print(f"- Entry issues:    {output_dir / 'train_cleanlab_entry_issues_only.csv'}")
    print(f"- Per-label:       {output_dir / 'train_cleanlab_per_label_summary.csv'}")
    print(f"- Summary:         {output_dir / 'oof_cleanlab_smoke_summary.csv'}")
    print(f"- Report:          {output_dir / 'oof_cleanlab_smoke_report.txt'}")


if __name__ == "__main__":
    main()
