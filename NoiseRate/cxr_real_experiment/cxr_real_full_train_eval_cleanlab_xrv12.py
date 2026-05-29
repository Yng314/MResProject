#!/usr/bin/env python3
"""
Train/evaluate a direct TorchXRayVision DenseNet on real MIMIC-CXR-JPG data.

This variant intentionally uses the XRV pretrained encoder + decoder directly,
and only supervises the 12 CheXpert labels that can be mapped onto the XRV
output space:

- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Enlarged Cardiomediastinum
- Fracture
- Lung Lesion
- Lung Opacity
- Pleural Effusion
- Pleural Other
- Pneumonia
- Pneumothorax

It supports:
- AP/PA filtering via metadata CSV
- full-train baseline
- sample-level cleaned retraining
- entry-level cleaned retraining
- GT test evaluation
- fresh train-set cleanlab outputs
"""

import argparse
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

import sys


SCRIPT_DIR = Path(__file__).resolve().parent
NOISERATE_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(NOISERATE_ROOT) not in sys.path:
    sys.path.insert(0, str(NOISERATE_ROOT))

from cxr_real_noise_validation_smoke import (  # noqa: E402
    MaskedBCEWithLogitsLoss,
    construct_image_relpath,
    estimate_issues,
    filter_existing_rows,
    load_real_pool,
    merge_view_metadata,
    project_raw_labels_to_binary,
    rank_from_quality,
    safe_nanmean,
    save_quality_plots,
    stringify_float_array,
)
from utils.xrv_utils import get_ordered_pathology_list  # noqa: E402

import torchxrayvision as xrv  # noqa: E402


LABEL_NAMES = get_ordered_pathology_list()
SHORT_LABEL_NAMES = {
    "Atelectasis": "Ate",
    "Cardiomegaly": "Card",
    "Consolidation": "Cons",
    "Edema": "Edem",
    "Enlarged Cardiomediastinum": "ECM",
    "Fracture": "Frac",
    "Lung Lesion": "Les",
    "Lung Opacity": "Opac",
    "Pleural Effusion": "Eff",
    "Pleural Other": "PO",
    "Pneumonia": "PNA",
    "Pneumothorax": "PTX",
}
CHEXPERT_TO_XRV_NAME = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
    "Fracture": "Fracture",
    "Lung Lesion": "Lung Lesion",
    "Lung Opacity": "Lung Opacity",
    "Pleural Effusion": "Effusion",
    "Pleural Other": "Pleural_Thickening",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
}


@dataclass
class XRV12RunSummary:
    train_samples: int
    train_studies: int
    optimization_train_samples: int
    optimization_val_samples: int
    test_images: int
    test_studies: int
    excluded_entry_count: int
    train_cleanlab_sample_issue_rate: float
    train_cleanlab_entry_issue_rate: float
    test_image_macro_auroc_binary: float
    test_study_macro_auroc_binary: float
    best_epoch: int
    best_val_loss: float
    seed: int
    epochs: int
    val_fraction: float
    early_stopping_patience: int
    recover_best_weights: int
    batch_size: int
    learning_rate: float
    model_backbone: str
    xrv_weights: str
    study_aggregation: str
    metadata_csv: str
    allowed_views: str
    cleaned_by_exclusion: int
    cleaned_by_entry_exclusion: int
    n_labels: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RealCXRXRV12Dataset(Dataset):
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
        self.center_crop = xrv.datasets.XRayCenterCrop()
        self.resizer = xrv.datasets.XRayResizer(self.image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        rel_path = self.rows.iloc[index]["image_path"]
        image = Image.open(self.image_root / rel_path).convert("L")
        x = np.asarray(image)
        if x.max() <= 1.0:
            x = (x * 255).astype(np.uint8)
        if x.ndim == 2:
            x = x[np.newaxis, :, :]
        x = self.center_crop(x)
        x = self.resizer(x)
        x = xrv.datasets.normalize(x, maxval=255)
        if x.ndim == 2:
            x = x[np.newaxis, :, :]
        x = torch.from_numpy(x).float()

        labels = torch.from_numpy(np.nan_to_num(self.binary_labels[index], nan=0.0))
        label_mask = torch.from_numpy(self.valid_mask[index])
        return x, labels, label_mask


class XRVDirectCheXpert12Model(nn.Module):
    def __init__(self, weights: str = "densenet121-res224-all", cache_dir: str | None = None):
        super().__init__()
        self.base_model = xrv.models.DenseNet(weights=weights, cache_dir=cache_dir)
        self.model_pathologies = list(self.base_model.pathologies)
        target_indices = []
        for label_name in LABEL_NAMES:
            xrv_name = CHEXPERT_TO_XRV_NAME[label_name]
            if xrv_name not in self.model_pathologies:
                raise ValueError(f"XRV pathology not found for {label_name}: {xrv_name}")
            target_indices.append(self.model_pathologies.index(xrv_name))
        self.register_buffer("target_indices", torch.tensor(target_indices, dtype=torch.long), persistent=False)
        self.selected_pathologies = [self.model_pathologies[i] for i in target_indices]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 3:
            x = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        logits = self.base_model(x)
        return logits.index_select(1, self.target_indices.to(logits.device))


class XRVEncoderCheXpert12HeadModel(nn.Module):
    def __init__(self, weights: str = "densenet121-res224-all", cache_dir: str | None = None):
        super().__init__()
        self.base_model = xrv.models.DenseNet(weights=weights, cache_dir=cache_dir)
        self.feature_dim = 1024
        self.classifier = nn.Linear(self.feature_dim, len(LABEL_NAMES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 3:
            x = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        features = self.base_model.features(x)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.flatten(1)
        return self.classifier(features)


def create_model(model_backbone: str, xrv_weights: str, xrv_cache_dir: str | None = None) -> nn.Module:
    if model_backbone == "xrv_densenet121_direct":
        return XRVDirectCheXpert12Model(weights=xrv_weights, cache_dir=xrv_cache_dir)
    if model_backbone == "xrv_densenet121_linearhead":
        return XRVEncoderCheXpert12HeadModel(weights=xrv_weights, cache_dir=xrv_cache_dir)
    raise ValueError(f"Unknown model_backbone: {model_backbone}")


def load_test_rows(
    test_csv: Path,
    split_csv: Path,
    image_root: Path,
    limit: int,
    metadata_csv: Path | None = None,
    allowed_views: Sequence[str] | None = None,
) -> pd.DataFrame:
    print(f"Loading test GT CSV: {test_csv}")
    test_df = pd.read_csv(test_csv)
    if "Airspace Opacity" in test_df.columns and "Lung Opacity" not in test_df.columns:
        test_df["Lung Opacity"] = test_df["Airspace Opacity"]

    for label_name in LABEL_NAMES:
        if label_name not in test_df.columns:
            test_df[label_name] = np.nan

    split_df = pd.read_csv(split_csv, usecols=["dicom_id", "study_id", "subject_id", "split"])
    split_df = split_df[split_df["split"] == "test"].copy().reset_index(drop=True)
    split_df = merge_view_metadata(split_df, metadata_csv=metadata_csv, allowed_views=allowed_views)

    df = split_df.merge(test_df[["study_id"] + LABEL_NAMES], on="study_id", how="inner")
    print(f"Merged labeled test rows: {len(df)} images from {df['study_id'].nunique()} studies")
    df["image_path"] = [
        construct_image_relpath(int(sid), int(stid), str(did))
        for sid, stid, did in zip(df["subject_id"], df["study_id"], df["dicom_id"])
    ]
    df["pool_row_id"] = np.arange(len(df), dtype=np.int64)
    df = filter_existing_rows(df, image_root=image_root)

    if limit > 0 and len(df) > limit:
        df = df.iloc[:limit].reset_index(drop=True)
        print(f"Applied test limit: {len(df)} rows")

    return df


def apply_sample_exclusions(
    train_rows: pd.DataFrame,
    exclude_sample_csv: Path | None,
    issue_col: str,
) -> pd.DataFrame:
    if exclude_sample_csv is None:
        return train_rows.reset_index(drop=True)

    print(f"Loading exclusion CSV: {exclude_sample_csv}")
    exclude_df = pd.read_csv(exclude_sample_csv)
    if issue_col in exclude_df.columns:
        exclude_df = exclude_df[exclude_df[issue_col] == 1].copy()
    if "pool_row_id" not in exclude_df.columns:
        raise ValueError(f"Exclusion CSV must contain pool_row_id column: {exclude_sample_csv}")

    exclude_ids = set(exclude_df["pool_row_id"].astype(np.int64).tolist())
    before = len(train_rows)
    filtered = train_rows[~train_rows["pool_row_id"].isin(exclude_ids)].reset_index(drop=True)
    removed = before - len(filtered)
    print(f"Applied sample exclusions: removed {removed} rows using {len(exclude_ids)} pool_row_id values")
    return filtered


def apply_entry_exclusions(
    train_rows: pd.DataFrame,
    raw_labels: np.ndarray,
    y_binary: np.ndarray,
    valid_mask: np.ndarray,
    exclude_entry_csv: Path | None,
    issue_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if exclude_entry_csv is None:
        return raw_labels, y_binary, valid_mask, 0

    print(f"Loading entry exclusion CSV: {exclude_entry_csv}")
    exclude_df = pd.read_csv(exclude_entry_csv)
    if issue_col in exclude_df.columns:
        exclude_df = exclude_df[exclude_df[issue_col] == 1].copy()
    required_cols = {"pool_row_id", "label_index"}
    missing_cols = required_cols - set(exclude_df.columns)
    if missing_cols:
        raise ValueError(f"Entry exclusion CSV missing columns {sorted(missing_cols)}: {exclude_entry_csv}")

    pool_row_to_local = {
        int(pool_row_id): int(local_idx)
        for local_idx, pool_row_id in enumerate(train_rows["pool_row_id"].to_numpy(dtype=np.int64))
    }

    excluded_count = 0
    y_binary = y_binary.copy()
    valid_mask = valid_mask.copy()
    for pool_row_id, label_idx in exclude_df[["pool_row_id", "label_index"]].itertuples(index=False):
        local_idx = pool_row_to_local.get(int(pool_row_id))
        if local_idx is None:
            continue
        label_idx = int(label_idx)
        if 0 <= label_idx < y_binary.shape[1] and valid_mask[local_idx, label_idx]:
            valid_mask[local_idx, label_idx] = False
            y_binary[local_idx, label_idx] = np.nan
            excluded_count += 1

    print(f"Applied entry exclusions: masked {excluded_count} label entries")
    return raw_labels, y_binary, valid_mask, excluded_count


def split_train_val_rows(
    train_rows: pd.DataFrame,
    raw_labels: np.ndarray,
    y_binary: np.ndarray,
    valid_mask: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if val_fraction <= 0.0:
        empty_rows = train_rows.iloc[0:0].copy().reset_index(drop=True)
        empty_raw = raw_labels[:0].copy()
        empty_binary = y_binary[:0].copy()
        empty_mask = valid_mask[:0].copy()
        return (
            train_rows.reset_index(drop=True),
            raw_labels,
            y_binary,
            valid_mask,
            empty_rows,
            empty_raw,
            empty_binary,
            empty_mask,
        )

    n_samples = len(train_rows)
    n_val = max(1, int(round(n_samples * val_fraction)))
    n_val = min(n_val, n_samples - 1)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return (
        train_rows.iloc[train_idx].reset_index(drop=True),
        raw_labels[train_idx],
        y_binary[train_idx],
        valid_mask[train_idx],
        train_rows.iloc[val_idx].reset_index(drop=True),
        raw_labels[val_idx],
        y_binary[val_idx],
        valid_mask[val_idx],
    )


def build_loader(
    rows: pd.DataFrame,
    image_root: Path,
    y_binary: np.ndarray,
    valid_mask: np.ndarray,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = RealCXRXRV12Dataset(
        rows=rows,
        image_root=image_root,
        image_size=image_size,
        binary_labels=y_binary,
        valid_mask=valid_mask,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_full_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    early_stopping_patience: int,
    recover_best_weights: bool,
) -> Tuple[torch.nn.Module, int, float]:
    criterion = MaskedBCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    best_state = None
    best_epoch = 0
    best_val_loss = float("inf")
    no_improve = 0

    for epoch_idx in range(epochs):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        print(f"[xrv12-train] epoch {epoch_idx + 1}/{epochs} training...")
        for xb, yb, mb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb, mb)
            loss.backward()
            optimizer.step()

            batch_size = int(xb.size(0))
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size

        train_loss = loss_sum / max(1, sample_count)
        print(f"[xrv12-train] epoch {epoch_idx + 1}/{epochs} train_loss={train_loss:.6f}")

        if val_loader is None:
            continue

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                mb = mb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb, mb)
                batch_size = int(xb.size(0))
                val_loss_sum += float(loss.item()) * batch_size
                val_count += batch_size

        val_loss = val_loss_sum / max(1, val_count)
        print(f"[xrv12-train] epoch {epoch_idx + 1}/{epochs} val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch_idx + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                print(f"[xrv12-train] early stopping at epoch {epoch_idx + 1} (best_epoch={best_epoch})")
                break

    if val_loader is not None and recover_best_weights and best_state is not None:
        model.load_state_dict(best_state)
        print(f"[xrv12-train] restored best weights from epoch {best_epoch}")
    elif val_loader is None:
        best_epoch = epochs
        best_val_loss = float("nan")
    elif best_epoch == 0:
        best_epoch = epochs

    return model, int(best_epoch), float(best_val_loss)


def predict_probabilities(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    tag: str,
) -> np.ndarray:
    model.eval()
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for batch_idx, (xb, _, _) in enumerate(data_loader, start=1):
            xb = xb.to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            if batch_idx % 200 == 0:
                print(f"[{tag}] processed {batch_idx} batches")
    return np.vstack(probs)


def compute_per_label_aurocs(
    y_binary: np.ndarray,
    valid_mask: np.ndarray,
    y_prob: np.ndarray,
    prefix: str,
) -> Tuple[pd.DataFrame, float]:
    rows: List[Dict[str, float]] = []
    aucs: List[float] = []

    for label_idx, label_name in enumerate(LABEL_NAMES):
        mask = valid_mask[:, label_idx]
        valid_count = int(mask.sum())
        pos_count = int(np.nansum(y_binary[mask, label_idx])) if valid_count > 0 else 0
        neg_count = int(valid_count - pos_count)
        auroc = float("nan")
        if valid_count >= 2:
            labels_j = y_binary[mask, label_idx]
            probs_j = y_prob[mask, label_idx]
            if len(np.unique(labels_j)) >= 2:
                try:
                    auroc = float(roc_auc_score(labels_j, probs_j))
                    aucs.append(auroc)
                except ValueError:
                    pass
        rows.append(
            {
                "label_index": int(label_idx),
                "label_name": label_name,
                f"{prefix}_valid_count": valid_count,
                f"{prefix}_positive_count": pos_count,
                f"{prefix}_negative_count": neg_count,
                f"{prefix}_auroc_binary": auroc,
            }
        )

    macro = float(np.mean(aucs)) if aucs else float("nan")
    return pd.DataFrame(rows), macro


def aggregate_test_predictions_by_study(
    rows: pd.DataFrame,
    raw_labels: np.ndarray,
    y_binary: np.ndarray,
    valid_mask: np.ndarray,
    y_prob: np.ndarray,
    agg_method: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    meta_df = pd.DataFrame(
        {
            "study_id": rows["study_id"].to_numpy(dtype=np.int64),
            "subject_id": rows["subject_id"].to_numpy(dtype=np.int64),
            "dicom_id": rows["dicom_id"].astype(str).to_numpy(),
        }
    )
    pred_cols = [f"pred::{name}" for name in LABEL_NAMES]
    raw_cols = [f"raw::{name}" for name in LABEL_NAMES]
    bin_cols = [f"bin::{name}" for name in LABEL_NAMES]
    mask_cols = [f"mask::{name}" for name in LABEL_NAMES]

    work_df = pd.concat(
        [
            meta_df,
            pd.DataFrame(y_prob, columns=pred_cols),
            pd.DataFrame(raw_labels, columns=raw_cols),
            pd.DataFrame(y_binary, columns=bin_cols),
            pd.DataFrame(valid_mask.astype(int), columns=mask_cols),
        ],
        axis=1,
    )

    grouped_rows: List[Dict[str, object]] = []
    for study_id, g in work_df.groupby("study_id", sort=True):
        row: Dict[str, object] = {
            "study_id": int(study_id),
            "subject_id": int(g["subject_id"].iloc[0]),
            "n_images_in_study": int(len(g)),
        }
        for label_name in LABEL_NAMES:
            pred_col = f"pred::{label_name}"
            if agg_method == "mean":
                row[pred_col] = float(g[pred_col].mean())
            else:
                row[pred_col] = float(g[pred_col].max())
            row[f"raw::{label_name}"] = g[f"raw::{label_name}"].iloc[0]
            row[f"bin::{label_name}"] = g[f"bin::{label_name}"].iloc[0]
            row[f"mask::{label_name}"] = g[f"mask::{label_name}"].iloc[0]
        grouped_rows.append(row)

    study_df = pd.DataFrame(grouped_rows)
    study_prob = study_df[pred_cols].to_numpy(dtype=np.float32)
    study_raw = study_df[raw_cols].to_numpy(dtype=np.float32)
    study_binary = study_df[bin_cols].to_numpy(dtype=np.float32)
    study_valid = study_df[mask_cols].to_numpy(dtype=np.int64).astype(bool)
    return study_df, study_raw, study_binary, study_valid, study_prob


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


def save_issue_grid(sample_df: pd.DataFrame, image_root: Path, output_png: Path, n_show: int = 8, image_size: int = 224) -> None:
    import matplotlib.pyplot as plt

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


def build_train_cleanlab_outputs(
    rows: pd.DataFrame,
    raw_labels: np.ndarray,
    y_binary: np.ndarray,
    valid_mask: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
    image_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    cleanlab_summary_df = pd.DataFrame(
        [
            {
                "n_samples": int(len(rows)),
                "n_labels": int(len(LABEL_NAMES)),
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
    cleanlab_summary_df.to_csv(output_dir / "train_cleanlab_summary.csv", index=False)

    save_quality_plots(sample_detail_df, per_label_df, output_dir)
    save_issue_grid(
        sample_detail_df,
        image_root=image_root,
        output_png=output_dir / "train_cleanlab_issue_grid.png",
    )
    return sample_detail_df, entry_detail_df, per_label_df, cleanlab_summary_df


def build_test_prediction_details(
    rows: pd.DataFrame,
    raw_labels: np.ndarray,
    y_binary: np.ndarray,
    valid_mask: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": rows["subject_id"].to_numpy(dtype=np.int64),
            "study_id": rows["study_id"].to_numpy(dtype=np.int64),
            "dicom_id": rows["dicom_id"].astype(str).to_numpy(),
            "image_path": rows["image_path"].astype(str).to_numpy(),
            "raw_labels_4class": [stringify_float_array(x) for x in raw_labels],
            "binary_labels_for_metric": [stringify_float_array(x) for x in y_binary],
            "valid_label_mask": ["|".join(map(str, m.astype(int).tolist())) for m in valid_mask],
            "pred_probs": ["|".join(f"{v:.6f}" for v in row.tolist()) for row in y_prob],
        }
    )


def write_report(
    output_dir: Path,
    summary_df: pd.DataFrame,
    image_metrics_df: pd.DataFrame,
    study_metrics_df: pd.DataFrame,
    train_cleanlab_summary_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    with open(output_dir / "baseline_run_report.txt", "w", encoding="utf-8") as f:
        f.write("Real CXR XRV12 Full-Train + Test Eval + Train Cleanlab\n")
        f.write("====================================================\n\n")
        for key, value in vars(args).items():
            f.write(f"{key}={value}\n")
        f.write("\n[Run Summary]\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n[Test Image-Level AUROC]\n")
        f.write(image_metrics_df.to_string(index=False))
        f.write("\n\n[Test Study-Level AUROC]\n")
        f.write(study_metrics_df.to_string(index=False))
        f.write("\n\n[Train Cleanlab Summary]\n")
        f.write(train_cleanlab_summary_df.to_string(index=False))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-train XRV12 baseline on real MIMIC-CXR-JPG with GT test evaluation and train cleanlab.")
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--chexpert-csv", type=str, required=True)
    parser.add_argument("--split-csv", type=str, required=True)
    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument("--metadata-csv", type=str, default=None)
    parser.add_argument("--allowed-views", nargs="*", default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--train-limit", type=int, default=-1)
    parser.add_argument("--test-limit", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--recover-best-weights", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--model-backbone",
        type=str,
        default="xrv_densenet121_direct",
        choices=["xrv_densenet121_direct", "xrv_densenet121_linearhead"],
    )
    parser.add_argument("--xrv-weights", type=str, default="densenet121-res224-all")
    parser.add_argument("--xrv-cache-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study-aggregation", type=str, default="max", choices=["max", "mean"])
    parser.add_argument("--exclude-sample-csv", type=str, default=None)
    parser.add_argument("--exclude-issue-col", type=str, default="est_issue_sample")
    parser.add_argument("--exclude-entry-csv", type=str, default=None)
    parser.add_argument("--exclude-entry-issue-col", type=str, default="est_issue_entry")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    image_root = Path(args.image_root)
    chexpert_csv = Path(args.chexpert_csv)
    split_csv = Path(args.split_csv)
    test_csv = Path(args.test_csv)
    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else None
    exclude_sample_csv = Path(args.exclude_sample_csv) if args.exclude_sample_csv else None
    exclude_entry_csv = Path(args.exclude_entry_csv) if args.exclude_entry_csv else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "results_xrv12" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Image root: {image_root}")
    print(f"CheXpert CSV: {chexpert_csv}")
    print(f"Split CSV: {split_csv}")
    print(f"Test CSV: {test_csv}")
    print(f"Metadata CSV: {metadata_csv}")
    print(f"Allowed Views: {args.allowed_views}")
    print(f"XRV weights: {args.xrv_weights}")
    print(f"XRV cache dir: {args.xrv_cache_dir}")
    print(f"Label names ({len(LABEL_NAMES)}): {LABEL_NAMES}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")

    train_rows = load_real_pool(
        chexpert_csv=chexpert_csv,
        split_csv=split_csv,
        split_name=args.train_split,
        metadata_csv=metadata_csv,
        allowed_views=args.allowed_views,
    )
    train_rows = filter_existing_rows(train_rows, image_root=image_root)
    train_rows = apply_sample_exclusions(
        train_rows=train_rows,
        exclude_sample_csv=exclude_sample_csv,
        issue_col=args.exclude_issue_col,
    )
    if args.train_limit > 0 and len(train_rows) > args.train_limit:
        train_rows = train_rows.iloc[:args.train_limit].reset_index(drop=True)
        print(f"Applied train limit: {len(train_rows)} rows")
    train_raw, train_binary, train_valid = project_raw_labels_to_binary(train_rows, LABEL_NAMES)
    train_raw, train_binary, train_valid, excluded_entry_count = apply_entry_exclusions(
        train_rows=train_rows,
        raw_labels=train_raw,
        y_binary=train_binary,
        valid_mask=train_valid,
        exclude_entry_csv=exclude_entry_csv,
        issue_col=args.exclude_entry_issue_col,
    )
    (
        opt_train_rows,
        _opt_train_raw,
        opt_train_binary,
        opt_train_valid,
        opt_val_rows,
        _opt_val_raw,
        opt_val_binary,
        opt_val_valid,
    ) = split_train_val_rows(
        train_rows=train_rows,
        raw_labels=train_raw,
        y_binary=train_binary,
        valid_mask=train_valid,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(
        f"Optimization split: train={len(opt_train_rows)} samples, "
        f"val={len(opt_val_rows)} samples, val_fraction={args.val_fraction}"
    )

    test_rows = load_test_rows(
        test_csv=test_csv,
        split_csv=split_csv,
        image_root=image_root,
        limit=args.test_limit,
        metadata_csv=metadata_csv,
        allowed_views=args.allowed_views,
    )
    test_raw, test_binary, test_valid = project_raw_labels_to_binary(test_rows, LABEL_NAMES)

    train_loader = build_loader(
        rows=opt_train_rows,
        image_root=image_root,
        y_binary=opt_train_binary,
        valid_mask=opt_train_valid,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = None
    if len(opt_val_rows) > 0:
        val_loader = build_loader(
            rows=opt_val_rows,
            image_root=image_root,
            y_binary=opt_val_binary,
            valid_mask=opt_val_valid,
            image_size=args.image_size,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    train_eval_loader = build_loader(
        rows=train_rows,
        image_root=image_root,
        y_binary=train_binary,
        valid_mask=train_valid,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = build_loader(
        rows=test_rows,
        image_root=image_root,
        y_binary=test_binary,
        valid_mask=test_valid,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = create_model(
        model_backbone=args.model_backbone,
        xrv_weights=args.xrv_weights,
        xrv_cache_dir=args.xrv_cache_dir,
    )
    model, best_epoch, best_val_loss = train_full_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        recover_best_weights=args.recover_best_weights,
    )

    checkpoint_path = output_dir / "baseline_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_names": LABEL_NAMES,
            "xrv_weights": args.xrv_weights,
            "selected_pathologies": CHEXPERT_TO_XRV_NAME,
            "args": vars(args),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}")

    train_prob = predict_probabilities(model, train_eval_loader, device=device, tag="train-predict")
    test_prob = predict_probabilities(model, test_loader, device=device, tag="test-predict")

    train_prob_df = pd.DataFrame(train_prob, columns=LABEL_NAMES)
    train_prob_df.insert(0, "dicom_id", train_rows["dicom_id"].astype(str).to_numpy())
    train_prob_df.insert(0, "study_id", train_rows["study_id"].to_numpy(dtype=np.int64))
    train_prob_df.insert(0, "subject_id", train_rows["subject_id"].to_numpy(dtype=np.int64))
    train_prob_df.to_csv(output_dir / "train_pred_probs.csv", index=False)

    test_pred_df = build_test_prediction_details(
        rows=test_rows,
        raw_labels=test_raw,
        y_binary=test_binary,
        valid_mask=test_valid,
        y_prob=test_prob,
    )
    test_pred_df.to_csv(output_dir / "test_image_predictions.csv", index=False)

    image_metrics_df, image_macro = compute_per_label_aurocs(
        y_binary=test_binary,
        valid_mask=test_valid,
        y_prob=test_prob,
        prefix="image",
    )
    study_df, study_raw, study_binary, study_valid, study_prob = aggregate_test_predictions_by_study(
        rows=test_rows,
        raw_labels=test_raw,
        y_binary=test_binary,
        valid_mask=test_valid,
        y_prob=test_prob,
        agg_method=args.study_aggregation,
    )
    study_metrics_df, study_macro = compute_per_label_aurocs(
        y_binary=study_binary,
        valid_mask=study_valid,
        y_prob=study_prob,
        prefix="study",
    )

    study_pred_df = study_df[["study_id", "subject_id", "n_images_in_study"]].copy()
    study_pred_df["raw_labels_4class"] = [stringify_float_array(x) for x in study_raw]
    study_pred_df["binary_labels_for_metric"] = [stringify_float_array(x) for x in study_binary]
    study_pred_df["valid_label_mask"] = ["|".join(map(str, m.astype(int).tolist())) for m in study_valid]
    study_pred_df["pred_probs"] = ["|".join(f"{v:.6f}" for v in row.tolist()) for row in study_prob]
    study_pred_df.to_csv(output_dir / "test_study_predictions.csv", index=False)

    train_sample_detail_df, train_entry_detail_df, _train_cleanlab_per_label_df, train_cleanlab_summary_df = build_train_cleanlab_outputs(
        rows=train_rows,
        raw_labels=train_raw,
        y_binary=train_binary,
        valid_mask=train_valid,
        y_prob=train_prob,
        output_dir=output_dir,
        image_root=image_root,
    )

    image_metrics_df.to_csv(output_dir / "test_image_auroc_summary.csv", index=False)
    study_metrics_df.to_csv(output_dir / "test_study_auroc_summary.csv", index=False)

    summary_df = pd.DataFrame(
        [
            XRV12RunSummary(
                train_samples=int(len(train_rows)),
                train_studies=int(train_rows["study_id"].nunique()),
                optimization_train_samples=int(len(opt_train_rows)),
                optimization_val_samples=int(len(opt_val_rows)),
                test_images=int(len(test_rows)),
                test_studies=int(study_df["study_id"].nunique()),
                excluded_entry_count=int(excluded_entry_count),
                train_cleanlab_sample_issue_rate=float(train_sample_detail_df["est_issue_sample"].mean()),
                train_cleanlab_entry_issue_rate=float(train_entry_detail_df["est_issue_entry"].mean()),
                test_image_macro_auroc_binary=float(image_macro),
                test_study_macro_auroc_binary=float(study_macro),
                best_epoch=int(best_epoch),
                best_val_loss=float(best_val_loss),
                seed=int(args.seed),
                epochs=int(args.epochs),
                val_fraction=float(args.val_fraction),
                early_stopping_patience=int(args.early_stopping_patience),
                recover_best_weights=int(args.recover_best_weights),
                batch_size=int(args.batch_size),
                learning_rate=float(args.learning_rate),
                model_backbone=str(args.model_backbone),
                xrv_weights=str(args.xrv_weights),
                study_aggregation=str(args.study_aggregation),
                metadata_csv="" if metadata_csv is None else str(metadata_csv),
                allowed_views="" if not args.allowed_views else "|".join(args.allowed_views),
                cleaned_by_exclusion=int(exclude_sample_csv is not None),
                cleaned_by_entry_exclusion=int(exclude_entry_csv is not None),
                n_labels=int(len(LABEL_NAMES)),
            ).__dict__
        ]
    )
    summary_df.to_csv(output_dir / "baseline_run_summary.csv", index=False)

    write_report(
        output_dir=output_dir,
        summary_df=summary_df,
        image_metrics_df=image_metrics_df,
        study_metrics_df=study_metrics_df,
        train_cleanlab_summary_df=train_cleanlab_summary_df,
        args=args,
    )

    print("\nSaved:")
    print(f"- Checkpoint:      {checkpoint_path}")
    print(f"- Run summary:     {output_dir / 'baseline_run_summary.csv'}")
    print(f"- Test image AUC:  {output_dir / 'test_image_auroc_summary.csv'}")
    print(f"- Test study AUC:  {output_dir / 'test_study_auroc_summary.csv'}")
    print(f"- Test image preds:{output_dir / 'test_image_predictions.csv'}")
    print(f"- Test study preds:{output_dir / 'test_study_predictions.csv'}")
    print(f"- Train cleanlab:  {output_dir / 'train_cleanlab_sample_details.csv'}")


if __name__ == "__main__":
    main()
