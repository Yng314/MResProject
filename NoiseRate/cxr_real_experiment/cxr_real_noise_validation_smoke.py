#!/usr/bin/env python3
"""
Smoke-test cleanlab issue detection on real MIMIC-CXR-JPG image-level data.

This script intentionally does not modify the existing toy experiments. It uses
real image-level MIMIC-CXR-JPG samples, preserves raw 4-state CheXpert labels
in outputs, but projects them to binary labels for training and cleanlab:

- 1.0  -> 1
- -1.0 -> 1
- 0.0  -> 0
- NaN  -> ignored
"""

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

import sys


NOISERATE_ROOT = Path(__file__).resolve().parents[1]
if str(NOISERATE_ROOT) not in sys.path:
    sys.path.insert(0, str(NOISERATE_ROOT))

from models.lightweight_cnn import LightweightCNN  # noqa: E402


LABEL_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

SHORT_LABEL_NAMES = {
    "Atelectasis": "Ate",
    "Cardiomegaly": "Card",
    "Consolidation": "Cons",
    "Edema": "Edem",
    "Enlarged Cardiomediastinum": "ECM",
    "Fracture": "Frac",
    "Lung Lesion": "Les",
    "Lung Opacity": "Opac",
    "No Finding": "NF",
    "Pleural Effusion": "Eff",
    "Pleural Other": "PO",
    "Pneumonia": "PNA",
    "Pneumothorax": "PTX",
    "Support Devices": "Supp",
}


@dataclass
class SmokeSummary:
    n_samples: int
    n_labels: int
    estimated_noise_rate_sample: float
    estimated_noise_rate_entry: float
    mean_sample_quality_self_confidence: float
    mean_sample_quality_normalized_margin: float
    mean_sample_quality_confidence_weighted_entropy: float
    macro_auroc_binary: float
    mean_per_label_issue_rate: float
    seed: int
    n_splits: int
    epochs: int
    batch_size: int
    model_backbone: str


def construct_image_relpath(subject_id: int, study_id: int, dicom_id: str) -> str:
    subject_prefix = str(subject_id)[:2]
    return f"p{subject_prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg"


def merge_view_metadata(
    rows: pd.DataFrame,
    metadata_csv: Path | None,
    allowed_views: Sequence[str] | None,
) -> pd.DataFrame:
    if metadata_csv is None:
        return rows.reset_index(drop=True)

    allowed_views = [str(v).strip().upper() for v in (allowed_views or []) if str(v).strip()]
    print(f"Loading metadata file: {metadata_csv}")
    metadata_df = pd.read_csv(metadata_csv, usecols=["dicom_id", "subject_id", "study_id", "ViewPosition"])
    metadata_df["ViewPosition"] = metadata_df["ViewPosition"].astype(str).str.upper()
    merged = rows.merge(
        metadata_df,
        on=["dicom_id", "subject_id", "study_id"],
        how="left",
    )
    missing_view = int(merged["ViewPosition"].isna().sum())
    if missing_view:
        print(f"Metadata merge left {missing_view} rows without ViewPosition.")
    if allowed_views:
        before = len(merged)
        merged = merged[merged["ViewPosition"].isin(allowed_views)].copy()
        print(f"Applied ViewPosition filter {allowed_views}: kept {len(merged)} / {before} rows")
    return merged.reset_index(drop=True)


def stringify_float_array(arr: np.ndarray) -> str:
    values = []
    for v in arr.tolist():
        if isinstance(v, float) and np.isnan(v):
            values.append("nan")
        else:
            values.append(str(v))
    return "|".join(values)


def parse_pipe_floats(text: str) -> List[float]:
    if pd.isna(text):
        return []
    out: List[float] = []
    for token in str(text).split("|"):
        token = token.strip().lower()
        if token in {"", "nan", "none"}:
            out.append(float("nan"))
        else:
            out.append(float(token))
    return out


def parse_pipe_ints(text: str) -> List[int]:
    if pd.isna(text):
        return []
    out: List[int] = []
    for token in str(text).split("|"):
        token = token.strip().lower()
        if token in {"", "nan", "none"}:
            out.append(0)
        else:
            out.append(int(float(token)))
    return out


def format_raw_label(raw_value: float) -> str:
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return "NA"
    if raw_value == 1.0:
        return "+"
    if raw_value == 0.0:
        return "-"
    if raw_value == -1.0:
        return "U"
    return str(raw_value)


def format_pred_label(prob: float) -> str:
    if prob is None or (isinstance(prob, float) and np.isnan(prob)):
        return "NA"
    return "+" if float(prob) >= 0.5 else "-"


def build_label_token_rows(row: pd.Series, labels_per_line: int = 3):
    raw_vals = parse_pipe_floats(row.get("raw_labels_4class", ""))
    probs = parse_pipe_floats(row.get("pred_probs", ""))
    issue_flags = parse_pipe_ints(row.get("issue_entry_mask", ""))
    token_rows = []
    current_row = []

    for idx, label_name in enumerate(LABEL_NAMES):
        raw_val = raw_vals[idx] if idx < len(raw_vals) else float("nan")
        pred_prob = probs[idx] if idx < len(probs) else float("nan")
        issue_flag = issue_flags[idx] if idx < len(issue_flags) else 0
        short_name = SHORT_LABEL_NAMES.get(label_name, label_name[:4])
        prob_txt = "NA" if np.isnan(pred_prob) else f"{pred_prob:.2f}"
        name_area = TextArea(
            short_name,
            textprops=dict(
                color=("#d62728" if issue_flag == 1 else "#111111"),
                fontsize=7.5,
                fontweight=("bold" if issue_flag == 1 else "normal"),
                family="monospace",
            ),
        )
        rest_area = TextArea(
            f":{format_raw_label(raw_val)}->{format_pred_label(pred_prob)}({prob_txt})",
            textprops=dict(color="#111111", fontsize=7.5, family="monospace"),
        )
        token = HPacker(children=[name_area, rest_area], align="baseline", pad=0, sep=0)
        current_row.append(token)
        if len(current_row) == labels_per_line:
            token_rows.append(current_row)
            current_row = []

    if current_row:
        token_rows.append(current_row)
    return token_rows


def build_rank_map_from_ranked(ranked: np.ndarray, n_items: int) -> np.ndarray:
    rank_map = np.full(n_items, np.nan, dtype=float)
    ranked = np.asarray(ranked)
    if ranked.dtype == bool:
        ranked_idx = np.where(ranked)[0]
    else:
        ranked_idx = ranked.astype(int)
    for rank_no, idx in enumerate(ranked_idx, start=1):
        rank_map[int(idx)] = float(rank_no)
    return rank_map


def rank_from_quality(quality: np.ndarray) -> np.ndarray:
    rank = np.full(len(quality), np.nan, dtype=float)
    valid = np.isfinite(quality)
    if not np.any(valid):
        return rank
    order = np.argsort(quality[valid], kind="stable")
    valid_idx = np.where(valid)[0][order]
    for rank_no, idx in enumerate(valid_idx, start=1):
        rank[idx] = float(rank_no)
    return rank


def safe_nanmean(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def load_real_pool(
    chexpert_csv: Path,
    split_csv: Path,
    split_name: str,
    metadata_csv: Path | None = None,
    allowed_views: Sequence[str] | None = None,
) -> pd.DataFrame:
    print(f"Loading split file: {split_csv}")
    split_df = pd.read_csv(split_csv, usecols=["dicom_id", "study_id", "subject_id", "split"])
    split_df = split_df[split_df["split"] == split_name].copy().reset_index(drop=True)
    print(f"Split rows retained ({split_name}): {len(split_df)}")

    print(f"Loading CheXpert labels: {chexpert_csv}")
    labels_df = pd.read_csv(chexpert_csv, usecols=["subject_id", "study_id"] + LABEL_NAMES)
    missing_label_cols = [c for c in LABEL_NAMES if c not in labels_df.columns]
    if missing_label_cols:
        raise ValueError(f"Missing CheXpert label columns: {missing_label_cols}")
    print(f"Label rows loaded: {len(labels_df)}")

    df = split_df.merge(
        labels_df[["subject_id", "study_id"] + LABEL_NAMES],
        on=["subject_id", "study_id"],
        how="left",
    )
    print(f"Merged image-level rows: {len(df)}")
    df["image_path"] = [
        construct_image_relpath(int(sid), int(stid), str(did))
        for sid, stid, did in zip(df["subject_id"], df["study_id"], df["dicom_id"])
    ]
    df = merge_view_metadata(df, metadata_csv=metadata_csv, allowed_views=allowed_views)

    df["pool_row_id"] = np.arange(len(df), dtype=np.int64)
    print("Skipping global image existence scan for smoke speed; sampled rows will be checked lazily.")
    return df


def filter_existing_rows(rows: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    exists_mask = rows["image_path"].map(lambda rel: (image_root / rel).exists())
    missing_count = int((~exists_mask).sum())
    if missing_count:
        print(f"Filtered out {missing_count} rows with missing image files.")
    return rows.loc[exists_mask].reset_index(drop=True)


def sample_rows(rows: pd.DataFrame, n_samples: int, seed: int, image_root: Path) -> pd.DataFrame:
    if n_samples <= 0:
        print(f"Using full pool without sampling: n={len(rows)}")
        return filter_existing_rows(rows.reset_index(drop=True), image_root=image_root)

    if len(rows) < n_samples:
        raise ValueError(f"Not enough rows in pool: {len(rows)} < requested {n_samples}")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(rows))
    chunk_size = max(256, n_samples * 2)
    selected_parts = []
    selected_count = 0

    for start in range(0, len(perm), chunk_size):
        idx = perm[start:start + chunk_size]
        candidate = rows.iloc[idx].copy()
        exists_mask = candidate["image_path"].map(lambda rel: (image_root / rel).exists())
        candidate = candidate[exists_mask]
        if len(candidate) == 0:
            continue
        need = n_samples - selected_count
        selected_parts.append(candidate.iloc[:need].copy())
        selected_count += min(len(candidate), need)
        if selected_count >= n_samples:
            break

    if selected_count < n_samples:
        raise ValueError(f"Could only find {selected_count} existing rows, need {n_samples}")

    selected_rows = pd.concat(selected_parts, ignore_index=True).iloc[:n_samples].reset_index(drop=True)
    return filter_existing_rows(selected_rows, image_root=image_root)


def project_raw_labels_to_binary(rows: pd.DataFrame, label_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = rows[list(label_names)].to_numpy(dtype=np.float32)
    binary = np.full_like(raw, np.nan, dtype=np.float32)
    valid = ~np.isnan(raw)
    binary[(raw == 1.0) | (raw == -1.0)] = 1.0
    binary[raw == 0.0] = 0.0
    return raw, binary, valid


class RealCXRSmokeDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        image_root: Path,
        image_size: int,
        binary_labels: np.ndarray,
        valid_mask: np.ndarray,
    ):
        self.rows = rows.reset_index(drop=True)
        self.image_root = image_root
        self.image_size = int(image_size)
        self.binary_labels = binary_labels.astype(np.float32)
        self.valid_mask = valid_mask.astype(bool)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        rel_path = self.rows.iloc[index]["image_path"]
        image = Image.open(self.image_root / rel_path).convert("RGB").resize((self.image_size, self.image_size))
        x = np.asarray(image, dtype=np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x)

        labels = torch.from_numpy(np.nan_to_num(self.binary_labels[index], nan=0.0))
        label_mask = torch.from_numpy(self.valid_mask[index])
        return x, labels, label_mask


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        masked = bce * label_mask.float()
        denom = label_mask.float().sum().clamp_min(1.0)
        return masked.sum() / denom


def create_model(model_backbone: str, n_labels: int) -> nn.Module:
    if model_backbone == "lightcnn":
        return LightweightCNN(num_classes=n_labels, input_channels=3, dropout=0.3)

    if model_backbone == "densenet121_pretrained":
        from torchvision.models import DenseNet121_Weights, densenet121

        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, n_labels)
        return model

    raise ValueError(f"Unknown model_backbone: {model_backbone}")


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    early_stopping_patience: int,
    recover_best_weights: bool,
    trial_tag: str,
    fold_idx: int,
    n_folds: int,
) -> np.ndarray:
    criterion = MaskedBCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch_idx in range(epochs):
        print(f"[{trial_tag}] fold {fold_idx}/{n_folds} epoch {epoch_idx + 1}/{epochs} training...")
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
        print(f"[{trial_tag}] fold {fold_idx}/{n_folds} epoch {epoch_idx + 1}/{epochs} val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                print(f"[{trial_tag}] fold {fold_idx}/{n_folds} early stopping")
                break

    if recover_best_weights and best_state is not None:
        model.load_state_dict(best_state)

    probs = []
    model.eval()
    with torch.no_grad():
        for xb, _, _ in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(probs)


def get_oof_probabilities(
    rows: pd.DataFrame,
    image_root: Path,
    y_binary: np.ndarray,
    valid_mask: np.ndarray,
    image_size: int,
    n_splits: int,
    seed: int,
    batch_size: int,
    epochs: int,
    lr: float,
    early_stopping_patience: int,
    recover_best_weights: bool,
    device: torch.device,
    model_backbone: str,
) -> np.ndarray:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_probs = np.zeros_like(y_binary, dtype=np.float32)
    split_iter = list(kf.split(np.arange(len(rows))))
    total_folds = len(split_iter)

    for fold_no, (train_idx, val_idx) in enumerate(split_iter, start=1):
        train_rows = rows.iloc[train_idx].reset_index(drop=True)
        val_rows = rows.iloc[val_idx].reset_index(drop=True)
        train_ds = RealCXRSmokeDataset(
            rows=train_rows,
            image_root=image_root,
            image_size=image_size,
            binary_labels=y_binary[train_idx],
            valid_mask=valid_mask[train_idx],
        )
        val_ds = RealCXRSmokeDataset(
            rows=val_rows,
            image_root=image_root,
            image_size=image_size,
            binary_labels=y_binary[val_idx],
            valid_mask=valid_mask[val_idx],
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        model = create_model(model_backbone=model_backbone, n_labels=y_binary.shape[1])
        fold_probs = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            early_stopping_patience=early_stopping_patience,
            recover_best_weights=recover_best_weights,
            trial_tag="real-smoke",
            fold_idx=fold_no,
            n_folds=total_folds,
        )
        oof_probs[val_idx] = fold_probs

    return oof_probs


def compute_macro_auroc(y_binary: np.ndarray, valid_mask: np.ndarray, y_prob: np.ndarray) -> float:
    scores = []
    for j in range(y_binary.shape[1]):
        mask = valid_mask[:, j]
        if mask.sum() < 2:
            continue
        yj = y_binary[mask, j]
        pj = y_prob[mask, j]
        if len(np.unique(yj)) < 2:
            continue
        try:
            scores.append(float(roc_auc_score(yj, pj)))
        except ValueError:
            pass
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def estimate_issues_per_label(
    y_binary: np.ndarray,
    valid_mask: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples, n_labels = y_binary.shape
    issue_mask = np.zeros((n_samples, n_labels), dtype=bool)
    quality_self = np.full((n_samples, n_labels), np.nan, dtype=np.float32)
    quality_margin = np.full((n_samples, n_labels), np.nan, dtype=np.float32)
    quality_entropy = np.full((n_samples, n_labels), np.nan, dtype=np.float32)
    rank_self = np.full((n_samples, n_labels), np.nan, dtype=np.float32)
    rank_margin = np.full((n_samples, n_labels), np.nan, dtype=np.float32)
    rank_entropy = np.full((n_samples, n_labels), np.nan, dtype=np.float32)

    for j in range(n_labels):
        mask = valid_mask[:, j]
        if mask.sum() < 10:
            continue
        labels_j = y_binary[mask, j].astype(int)
        if len(np.unique(labels_j)) < 2:
            continue
        pred_prob_j = y_prob[mask, j]
        pred_probs_2c = np.column_stack([1.0 - pred_prob_j, pred_prob_j])

        issue_local = find_label_issues(labels=labels_j, pred_probs=pred_probs_2c).astype(bool)
        issue_mask[mask, j] = issue_local

        qual_self_local = get_label_quality_scores(labels=labels_j, pred_probs=pred_probs_2c, method="self_confidence")
        qual_margin_local = get_label_quality_scores(labels=labels_j, pred_probs=pred_probs_2c, method="normalized_margin")
        qual_entropy_local = get_label_quality_scores(
            labels=labels_j,
            pred_probs=pred_probs_2c,
            method="confidence_weighted_entropy",
        )
        quality_self[mask, j] = qual_self_local
        quality_margin[mask, j] = qual_margin_local
        quality_entropy[mask, j] = qual_entropy_local

        rank_self[mask, j] = build_rank_map_from_ranked(
            ranked=find_label_issues(labels=labels_j, pred_probs=pred_probs_2c, return_indices_ranked_by="self_confidence"),
            n_items=int(mask.sum()),
        )
        rank_margin[mask, j] = build_rank_map_from_ranked(
            ranked=find_label_issues(labels=labels_j, pred_probs=pred_probs_2c, return_indices_ranked_by="normalized_margin"),
            n_items=int(mask.sum()),
        )
        rank_entropy[mask, j] = build_rank_map_from_ranked(
            ranked=find_label_issues(
                labels=labels_j,
                pred_probs=pred_probs_2c,
                return_indices_ranked_by="confidence_weighted_entropy",
            ),
            n_items=int(mask.sum()),
        )

    return issue_mask, quality_self, quality_margin, quality_entropy, rank_self, rank_margin, rank_entropy


def scatter_back_local_rank(full_rank: np.ndarray, mask: np.ndarray, local_rank: np.ndarray) -> None:
    full_rank[mask] = local_rank


def estimate_issues(
    y_binary: np.ndarray,
    valid_mask: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples, n_labels = y_binary.shape
    issue_mask = np.zeros((n_samples, n_labels), dtype=bool)
    quality_self = np.full((n_samples, n_labels), np.nan, dtype=np.float32)
    quality_margin = np.full((n_samples, n_labels), np.nan, dtype=np.float32)
    quality_entropy = np.full((n_samples, n_labels), np.nan, dtype=np.float32)
    rank_self = np.full((n_samples, n_labels), np.nan, dtype=np.float32)
    rank_margin = np.full((n_samples, n_labels), np.nan, dtype=np.float32)
    rank_entropy = np.full((n_samples, n_labels), np.nan, dtype=np.float32)

    for j in range(n_labels):
        mask = valid_mask[:, j]
        if mask.sum() < 10:
            continue
        labels_j = y_binary[mask, j].astype(int)
        if len(np.unique(labels_j)) < 2:
            continue
        pred_prob_j = y_prob[mask, j]
        pred_probs_2c = np.column_stack([1.0 - pred_prob_j, pred_prob_j])

        issue_local = find_label_issues(labels=labels_j, pred_probs=pred_probs_2c).astype(bool)
        issue_mask[mask, j] = issue_local

        quality_self[mask, j] = get_label_quality_scores(labels=labels_j, pred_probs=pred_probs_2c, method="self_confidence")
        quality_margin[mask, j] = get_label_quality_scores(labels=labels_j, pred_probs=pred_probs_2c, method="normalized_margin")
        quality_entropy[mask, j] = get_label_quality_scores(
            labels=labels_j,
            pred_probs=pred_probs_2c,
            method="confidence_weighted_entropy",
        )

        scatter_back_local_rank(
            rank_self[:, j],
            mask,
            build_rank_map_from_ranked(
                ranked=find_label_issues(labels=labels_j, pred_probs=pred_probs_2c, return_indices_ranked_by="self_confidence"),
                n_items=int(mask.sum()),
            ),
        )
        scatter_back_local_rank(
            rank_margin[:, j],
            mask,
            build_rank_map_from_ranked(
                ranked=find_label_issues(labels=labels_j, pred_probs=pred_probs_2c, return_indices_ranked_by="normalized_margin"),
                n_items=int(mask.sum()),
            ),
        )
        scatter_back_local_rank(
            rank_entropy[:, j],
            mask,
            build_rank_map_from_ranked(
                ranked=find_label_issues(
                    labels=labels_j,
                    pred_probs=pred_probs_2c,
                    return_indices_ranked_by="confidence_weighted_entropy",
                ),
                n_items=int(mask.sum()),
            ),
        )

    return issue_mask, quality_self, quality_margin, quality_entropy, rank_self, rank_margin, rank_entropy


def per_label_summary_df(
    y_binary: np.ndarray,
    valid_mask: np.ndarray,
    y_prob: np.ndarray,
    issue_mask: np.ndarray,
    quality_self: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for j, label_name in enumerate(LABEL_NAMES):
        mask = valid_mask[:, j]
        valid_count = int(mask.sum())
        pos_count = int(np.nansum(y_binary[mask, j])) if valid_count > 0 else 0
        neg_count = int(valid_count - pos_count)
        issue_rate = float(issue_mask[mask, j].mean()) if valid_count > 0 else float("nan")
        mean_quality = safe_nanmean(quality_self[mask, j]) if valid_count > 0 else float("nan")
        score = float("nan")
        if valid_count > 1 and len(np.unique(y_binary[mask, j])) >= 2:
            try:
                score = float(roc_auc_score(y_binary[mask, j], y_prob[mask, j]))
            except ValueError:
                pass
        rows.append(
            {
                "label_index": int(j),
                "label_name": label_name,
                "valid_count": valid_count,
                "positive_count": pos_count,
                "negative_count": neg_count,
                "estimated_entry_issue_rate": issue_rate,
                "mean_entry_quality_self_confidence": mean_quality,
                "oof_auroc_binary": score,
            }
        )
    return pd.DataFrame(rows)


def save_quality_plots(sample_df: pd.DataFrame, per_label_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = pd.to_numeric(sample_df["sample_quality_self_confidence"], errors="coerce").dropna()
    ax.hist(vals, bins=30, color="tab:blue", alpha=0.85)
    ax.set_title("Sample Quality (Self Confidence)")
    ax.set_xlabel("Quality")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "sample_quality_hist.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    data = per_label_df.sort_values("estimated_entry_issue_rate", ascending=False)
    ax.bar(data["label_name"], data["estimated_entry_issue_rate"], color="tab:orange")
    ax.set_ylabel("Estimated Entry Issue Rate")
    ax.set_xlabel("Label")
    ax.set_title("Per-label Estimated Issue Rate")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(output_dir / "per_label_issue_rate.png", dpi=180)
    plt.close(fig)


def save_issue_grid(sample_df: pd.DataFrame, image_root: Path, output_png: Path, n_show: int = 8, image_size: int = 224) -> None:
    candidates = sample_df.sort_values(
        ["est_issue_sample", "sample_quality_self_confidence", "est_issue_entry_count"],
        ascending=[False, True, False],
    ).head(n_show)
    if len(candidates) == 0:
        return

    n_cols = 2
    n_rows = int(np.ceil(len(candidates) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.4 * n_cols, 6.6 * n_rows))
    axes = np.asarray(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for idx, (_, row) in enumerate(candidates.iterrows()):
        ax = axes[idx]
        image = Image.open(image_root / row["image_path"]).convert("RGB").resize((image_size, image_size))
        ax.imshow(np.asarray(image))
        ax.axis("off")
        title = (
            f"study={int(row['study_id'])}\n"
            f"dicom={str(row['dicom_id'])[:12]}...\n"
            f"sample_issue={int(row['est_issue_sample'])} "
            f"entry_cnt={int(row['est_issue_entry_count'])}\n"
            f"q={float(row['sample_quality_self_confidence']):.3f}"
        )
        ax.set_title(title, fontsize=8, loc="left", pad=10)

        row_boxes = []
        for token_row in build_label_token_rows(row, labels_per_line=3):
            row_boxes.append(HPacker(children=token_row, align="baseline", pad=0, sep=8))
        label_box = VPacker(children=row_boxes, align="left", pad=0, sep=2)
        anchored = AnchoredOffsetbox(
            loc="lower left",
            child=label_box,
            pad=0.25,
            borderpad=0.35,
            frameon=True,
            bbox_to_anchor=(0.0, 0.0),
            bbox_transform=ax.transAxes,
        )
        anchored.patch.set_facecolor((1.0, 1.0, 1.0, 0.82))
        anchored.patch.set_edgecolor((0.7, 0.7, 0.7, 0.9))
        ax.add_artist(anchored)

    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Smoke-test cleanlab issue detection on real MIMIC-CXR-JPG data.")
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--chexpert-csv", type=str, required=True)
    parser.add_argument("--split-csv", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num-samples", type=int, default=1000, help="Use -1 to run on the full split.")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--model-backbone", type=str, default="lightcnn", choices=["lightcnn", "densenet121_pretrained"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    image_root = Path(args.image_root)
    chexpert_csv = Path(args.chexpert_csv)
    split_csv = Path(args.split_csv)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "results_smoke" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Image root: {image_root}")
    print(f"CheXpert CSV: {chexpert_csv}")
    print(f"Split CSV: {split_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")

    pool_df = load_real_pool(
        chexpert_csv=chexpert_csv,
        split_csv=split_csv,
        split_name=args.split,
    )
    rows = sample_rows(pool_df, n_samples=args.num_samples, seed=args.seed, image_root=image_root)
    raw_labels, y_binary, valid_mask = project_raw_labels_to_binary(rows, LABEL_NAMES)

    y_prob = get_oof_probabilities(
        rows=rows,
        image_root=image_root,
        y_binary=y_binary,
        valid_mask=valid_mask,
        image_size=args.image_size,
        n_splits=args.n_splits,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        recover_best_weights=True,
        device=device,
        model_backbone=args.model_backbone,
    )
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
    macro_auroc = compute_macro_auroc(y_binary=y_binary, valid_mask=valid_mask, y_prob=y_prob)

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

    entry_rows = []
    for i in range(len(rows)):
        for j, label_name in enumerate(LABEL_NAMES):
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
    per_label_df = per_label_summary_df(
        y_binary=y_binary,
        valid_mask=valid_mask,
        y_prob=y_prob,
        issue_mask=issue_mask,
        quality_self=quality_self,
    )

    summary = SmokeSummary(
        n_samples=int(len(rows)),
        n_labels=int(len(LABEL_NAMES)),
        estimated_noise_rate_sample=float(sample_issue_mask.mean()),
        estimated_noise_rate_entry=float(issue_mask.mean()),
        mean_sample_quality_self_confidence=safe_nanmean(sample_quality_self),
        mean_sample_quality_normalized_margin=safe_nanmean(sample_quality_margin),
        mean_sample_quality_confidence_weighted_entropy=safe_nanmean(sample_quality_entropy),
        macro_auroc_binary=float(macro_auroc),
        mean_per_label_issue_rate=float(np.nanmean(per_label_df["estimated_entry_issue_rate"])),
        seed=int(args.seed),
        n_splits=int(args.n_splits),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        model_backbone=str(args.model_backbone),
    )
    summary_df = pd.DataFrame([summary.__dict__])

    sample_detail_path = output_dir / "real_noise_sample_details.csv"
    entry_detail_path = output_dir / "real_noise_entry_details.csv"
    sample_issue_path = output_dir / "real_noise_sample_issues_only.csv"
    entry_issue_path = output_dir / "real_noise_entry_issues_only.csv"
    per_label_path = output_dir / "real_noise_per_label_summary.csv"
    summary_path = output_dir / "real_noise_summary.csv"

    sample_detail_df.to_csv(sample_detail_path, index=False)
    entry_detail_df.to_csv(entry_detail_path, index=False)
    sample_detail_df[sample_detail_df["est_issue_sample"] == 1].to_csv(sample_issue_path, index=False)
    entry_detail_df[entry_detail_df["est_issue_entry"] == 1].to_csv(entry_issue_path, index=False)
    per_label_df.to_csv(per_label_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    save_quality_plots(sample_detail_df, per_label_df, output_dir)
    save_issue_grid(sample_detail_df, image_root=image_root, output_png=output_dir / "real_noise_issue_grid.png")

    with open(output_dir / "real_noise_report.txt", "w", encoding="utf-8") as f:
        f.write("Real CXR Smoke Noise Validation Report\n")
        f.write("=====================================\n\n")
        f.write(f"image_root={image_root}\n")
        f.write(f"chexpert_csv={chexpert_csv}\n")
        f.write(f"split_csv={split_csv}\n")
        f.write(f"split={args.split}\n")
        f.write(f"device={device}\n")
        f.write(f"num_samples={args.num_samples}\n")
        f.write(f"n_splits={args.n_splits}\n")
        f.write(f"epochs={args.epochs}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"model_backbone={args.model_backbone}\n")
        f.write("\nSummary\n")
        for key, value in summary.__dict__.items():
            f.write(f"{key}={value}\n")

    print("\nSaved:")
    print(f"- Sample details: {sample_detail_path}")
    print(f"- Entry details:  {entry_detail_path}")
    print(f"- Sample issues:  {sample_issue_path}")
    print(f"- Entry issues:   {entry_issue_path}")
    print(f"- Per-label:      {per_label_path}")
    print(f"- Summary:        {summary_path}")
    print(f"- Report:         {output_dir / 'real_noise_report.txt'}")


if __name__ == "__main__":
    main()
