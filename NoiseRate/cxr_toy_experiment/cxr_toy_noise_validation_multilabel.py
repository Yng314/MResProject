"""
Multi-label toy experiment in image domain:
Use MIMIC-CXR backgrounds + synthetic shapes to validate whether cleanlab-estimated
noise rate tracks injected label noise as task complexity increases (K configurable).
"""

import argparse
import copy
import math
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from cleanlab import filter as binary_filter
from cleanlab import rank as binary_rank
from cleanlab.multilabel_classification import filter as multilabel_filter
from cleanlab.multilabel_classification import rank as multilabel_rank
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from NoiseRate.models.lightweight_cnn import LightweightCNN

OVERLAY_COLOR_MAP = {
    "white": (255, 255, 255),
    "blue": (0, 120, 255),
}


DIGIT_LABELS = [f"digit_{i}" for i in range(10)]
SHAPE_LABELS = ["triangle", "square", "star", "heart"]
LABEL_NAMES = DIGIT_LABELS + SHAPE_LABELS
POSITION_LAYOUT_VERSION = 2

DIGIT_SEGMENTS = {
    0: ("a", "b", "c", "d", "e", "f"),
    1: ("b", "c"),
    2: ("a", "b", "g", "e", "d"),
    3: ("a", "b", "g", "c", "d"),
    4: ("f", "g", "b", "c"),
    5: ("a", "f", "g", "c", "d"),
    6: ("a", "f", "g", "e", "c", "d"),
    7: ("a", "b", "c"),
    8: ("a", "b", "c", "d", "e", "f", "g"),
    9: ("a", "b", "c", "d", "f", "g"),
}


def _draw_star(draw: ImageDraw.ImageDraw, x: int, y: int, r: int, color: Tuple[int, int, int]) -> None:
    pts = []
    outer = r
    inner = max(3, int(r * 0.45))
    for i in range(10):
        ang = -np.pi / 2 + i * np.pi / 5
        rr = outer if (i % 2 == 0) else inner
        px = int(x + rr * np.cos(ang))
        py = int(y + rr * np.sin(ang))
        pts.append((px, py))
    draw.polygon(pts, fill=color)


def _draw_heart(draw: ImageDraw.ImageDraw, x: int, y: int, r: int, color: Tuple[int, int, int]) -> None:
    top_r = max(3, int(r * 0.55))
    offset = int(r * 0.45)
    draw.ellipse(
        (x - offset - top_r, y - top_r, x - offset + top_r, y + top_r),
        fill=color,
    )
    draw.ellipse(
        (x + offset - top_r, y - top_r, x + offset + top_r, y + top_r),
        fill=color,
    )
    tip_y = y + int(r * 1.20)
    pts = [
        (x - int(r * 1.35), y + int(r * 0.15)),
        (x + int(r * 1.35), y + int(r * 0.15)),
        (x, tip_y),
    ]
    draw.polygon(pts, fill=color)


def _draw_digit(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    r: int,
    digit: int,
    color: Tuple[int, int, int],
) -> None:
    segments = DIGIT_SEGMENTS.get(int(digit), ())
    if not segments:
        return
    half_w = max(3, int(r * 0.85))
    half_h = max(5, int(r * 1.35))
    thick = max(2, int(r * 0.30))
    t2 = thick // 2
    boxes = {
        "a": (x - half_w, y - half_h - t2, x + half_w, y - half_h + t2),
        "g": (x - half_w, y - t2, x + half_w, y + t2),
        "d": (x - half_w, y + half_h - t2, x + half_w, y + half_h + t2),
        "f": (x - half_w - t2, y - half_h, x - half_w + t2, y),
        "b": (x + half_w - t2, y - half_h, x + half_w + t2, y),
        "e": (x - half_w - t2, y, x - half_w + t2, y + half_h),
        "c": (x + half_w - t2, y, x + half_w + t2, y + half_h),
    }
    for seg in segments:
        draw.rectangle(boxes[seg], fill=color)


def draw_overlay_label(
    draw: ImageDraw.ImageDraw,
    row: pd.Series,
    label_name: str,
    w: int,
    h: int,
    overlay_rgb: Tuple[int, int, int],
) -> None:
    active = int(row[f"{label_name}_label"])
    if active == 0:
        return

    x = int(float(row[f"{label_name}_x_frac"]) * w)
    y = int(float(row[f"{label_name}_y_frac"]) * h)
    r = int(float(row[f"{label_name}_r_frac"]) * min(w, h))
    r = max(5, min(r, min(w, h) // 6))
    pad = int(r * 1.5)
    x = max(pad, min(x, w - pad))
    y = max(pad, min(y, h - pad))

    if label_name.startswith("digit_"):
        digit = int(label_name.split("_")[1])
        _draw_digit(draw, x=x, y=y, r=r, digit=digit, color=overlay_rgb)
        return
    if label_name == "triangle":
        pts = [(x, y - r), (x - r, y + r), (x + r, y + r)]
        draw.polygon(pts, fill=overlay_rgb)
        return
    if label_name == "square":
        draw.rectangle((x - r, y - r, x + r, y + r), fill=overlay_rgb)
        return
    if label_name == "star":
        _draw_star(draw, x=x, y=y, r=r, color=overlay_rgb)
        return
    if label_name == "heart":
        _draw_heart(draw, x=x, y=y, r=r, color=overlay_rgb)
        return
    raise ValueError(f"Unknown label name: {label_name}")


@dataclass
class TrialResult:
    pos_rate: float
    n_labels: int
    seed: int
    injected_noise_rate: float
    true_noise_rate_sample: float
    true_noise_rate_entry: float
    estimated_noise_rate_sample: float
    estimated_noise_rate_entry: float
    issue_precision_sample: float
    issue_recall_sample: float
    oof_auroc_macro_vs_clean: float
    oof_auroc_macro_vs_noisy: float


class CXRMultiLabelToyDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        dataset_root: Path,
        image_size: int,
        y_train: np.ndarray,
        shape_names: Sequence[str],
        input_channels: int,
        overlay_rgb: Tuple[int, int, int],
    ):
        self.rows = rows.reset_index(drop=True)
        self.dataset_root = dataset_root
        self.image_size = int(image_size)
        self.y_train = y_train.astype(np.float32)
        self.shape_names = list(shape_names)
        self.input_channels = int(input_channels)
        self.overlay_rgb = tuple(overlay_rgb)

    def __len__(self) -> int:
        return len(self.rows)

    def _draw_label(self, draw: ImageDraw.ImageDraw, row: pd.Series, label_name: str, w: int, h: int) -> None:
        draw_overlay_label(
            draw=draw,
            row=row,
            label_name=label_name,
            w=w,
            h=h,
            overlay_rgb=self.overlay_rgb,
        )

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        rel_path = row["image_path"]
        img_path = self.dataset_root / rel_path
        img = Image.open(img_path).convert("RGB").resize((self.image_size, self.image_size))
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for label_name in self.shape_names:
            self._draw_label(draw, row, label_name, w=w, h=h)

        x = np.asarray(img, dtype=np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x)

        y = torch.from_numpy(self.y_train[index].copy())
        return x, y


def create_model(model_backbone: str, n_labels: int) -> Tuple[nn.Module, int]:
    if model_backbone == "lightcnn":
        return LightweightCNN(num_classes=n_labels, input_channels=3, dropout=0.3), 3

    if model_backbone == "densenet121_pretrained":
        try:
            from torchvision.models import DenseNet121_Weights, densenet121
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for densenet121_pretrained. "
                "Please install torchvision in your conda env."
            ) from exc
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, n_labels)
        return model, 3

    raise ValueError(f"Unknown model_backbone: {model_backbone}")


def train_multilabel_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    early_stopping_patience: int,
    recover_best_weights: bool,
    trial_tag: str = "",
    fold_idx: int = 0,
    n_folds: int = 0,
) -> np.ndarray:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve_count = 0

    for epoch_idx in range(epochs):
        if trial_tag:
            print(
                f"[{trial_tag}] fold {fold_idx}/{n_folds} epoch {epoch_idx + 1}/{epochs} training..."
            )
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                batch_loss = criterion(logits, yb)
                batch_size = int(yb.size(0))
                val_loss_sum += float(batch_loss.item()) * batch_size
                val_count += batch_size

        val_loss = val_loss_sum / max(1, val_count)
        if trial_tag:
            print(
                f"[{trial_tag}] fold {fold_idx}/{n_folds} epoch {epoch_idx + 1}/{epochs} "
                f"val_loss={val_loss:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch_idx + 1
            best_state = copy.deepcopy(model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= early_stopping_patience:
                if trial_tag:
                    print(
                        f"[{trial_tag}] fold {fold_idx}/{n_folds} early stopping at "
                        f"epoch {epoch_idx + 1} (best epoch {best_epoch}, "
                        f"best_val_loss={best_val_loss:.6f})"
                    )
                break

    if recover_best_weights and best_state is not None:
        model.load_state_dict(best_state)
        if trial_tag:
            print(
                f"[{trial_tag}] fold {fold_idx}/{n_folds} restored best weights "
                f"from epoch {best_epoch} (val_loss={best_val_loss:.6f})"
            )
    elif (not recover_best_weights) and trial_tag:
        print(f"[{trial_tag}] fold {fold_idx}/{n_folds} skip restoring best weights.")

    if trial_tag:
        print(f"[{trial_tag}] fold {fold_idx}/{n_folds} validating...")
    probs = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            fold_prob = torch.sigmoid(logits).cpu().numpy()
            probs.append(fold_prob)
    return np.vstack(probs)


def get_oof_probabilities(
    rows: pd.DataFrame,
    dataset_root: Path,
    y_train: np.ndarray,
    shape_names: Sequence[str],
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
    overlay_rgb: Tuple[int, int, int],
    trial_tag: str,
    split_mode: str = "kfold",
) -> np.ndarray:
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if len(rows) < n_splits:
        raise ValueError(f"n_splits={n_splits} cannot be greater than n_samples={len(rows)}")

    # Configure CV split strategy.
    n_labels = y_train.shape[1]
    weights = (2 ** np.arange(n_labels)).astype(np.int64)
    strat_code = (y_train.astype(np.int64) * weights[None, :]).sum(axis=1)
    label_count = y_train.astype(np.int64).sum(axis=1)
    indices = np.arange(len(rows))

    def _can_stratify(target: np.ndarray) -> bool:
        _, counts = np.unique(target, return_counts=True)
        return (len(counts) >= 2) and (int(counts.min()) >= int(n_splits))

    split_iter = None
    split_mode_used = ""

    if split_mode == "kfold":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = list(kf.split(indices))
        split_mode_used = "kfold"
    elif split_mode == "stratified_label_count":
        if not _can_stratify(label_count):
            raise ValueError(
                "split_mode=stratified_label_count is invalid for this trial: "
                f"min class count in label_count is < n_splits ({n_splits})."
            )
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = list(skf.split(indices, label_count))
        split_mode_used = "stratified_label_count"
    elif split_mode == "stratified_multilabel_code":
        if not _can_stratify(strat_code):
            raise ValueError(
                "split_mode=stratified_multilabel_code is invalid for this trial: "
                f"min class count in multilabel-code is < n_splits ({n_splits})."
            )
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = list(skf.split(indices, strat_code))
        split_mode_used = "stratified_multilabel_code"
    elif split_mode == "auto":
        if _can_stratify(strat_code):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split_iter = list(skf.split(indices, strat_code))
            split_mode_used = "stratified_multilabel_code"
        elif _can_stratify(label_count):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split_iter = list(skf.split(indices, label_count))
            split_mode_used = "stratified_label_count"
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split_iter = list(kf.split(indices))
            split_mode_used = "kfold"
    else:
        raise ValueError(
            f"Unknown split_mode={split_mode}. "
            "Use one of: kfold, stratified_label_count, stratified_multilabel_code, auto."
        )

    if trial_tag:
        print(f"[{trial_tag}] split_mode={split_mode_used}, n_splits={n_splits}")

    total_folds = len(split_iter)
    oof_probs = np.zeros_like(y_train, dtype=np.float32)

    for fold_no, (train_idx, val_idx) in enumerate(split_iter, start=1):
        train_rows = rows.iloc[train_idx].reset_index(drop=True)
        val_rows = rows.iloc[val_idx].reset_index(drop=True)
        y_train_fold = y_train[train_idx]
        y_val_fold = y_train[val_idx]

        model, input_channels = create_model(model_backbone=model_backbone, n_labels=n_labels)
        train_ds = CXRMultiLabelToyDataset(
            rows=train_rows,
            dataset_root=dataset_root,
            image_size=image_size,
            y_train=y_train_fold,
            shape_names=shape_names,
            input_channels=input_channels,
            overlay_rgb=overlay_rgb,
        )
        val_ds = CXRMultiLabelToyDataset(
            rows=val_rows,
            dataset_root=dataset_root,
            image_size=image_size,
            y_train=y_val_fold,
            shape_names=shape_names,
            input_channels=input_channels,
            overlay_rgb=overlay_rgb,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        fold_probs = train_multilabel_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            early_stopping_patience=early_stopping_patience,
            recover_best_weights=recover_best_weights,
            trial_tag=trial_tag,
            fold_idx=fold_no,
            n_folds=total_folds,
        )
        oof_probs[val_idx] = fold_probs

    return oof_probs


def build_or_load_pool(
    metadata_csv: Path,
    dataset_root: Path,
    cache_csv: Path,
    view_positions: Sequence[str],
    pool_seed: int,
) -> pd.DataFrame:
    def _filter_existing(df_in: pd.DataFrame) -> pd.DataFrame:
        exists_mask = df_in["image_path"].astype(str).map(lambda p: (dataset_root / p).exists())
        df_out = df_in[exists_mask].copy().reset_index(drop=True)
        dropped = int((~exists_mask).sum())
        if dropped > 0:
            print(f"Dropped missing-image rows: {dropped}")
        return df_out

    def _assign_labels_and_params(df_in: pd.DataFrame) -> pd.DataFrame:
        n = len(df_in)
        if n < 2:
            raise ValueError("Not enough samples to build toy pool.")
        rng = np.random.RandomState(pool_seed)
        df = df_in.copy()

        # Exact 50% balance per label.
        for shape_name in LABEL_NAMES:
            lbl = np.zeros(n, dtype=np.int64)
            lbl[: n // 2] = 1
            rng.shuffle(lbl)
            df[f"{shape_name}_label"] = lbl

        # Fully random layout (per sample) with non-overlap:
        # randomly assign each label to a unique grid slot, then add small jitter.
        grid_side = 6
        axis = np.linspace(0.10, 0.90, grid_side, dtype=np.float32)
        slots = np.array([(x, y) for y in axis for x in axis], dtype=np.float32)  # (36, 2)
        n_slots = int(slots.shape[0])
        n_labels = len(LABEL_NAMES)
        if n_slots < n_labels:
            raise ValueError(f"Need >= {n_labels} slots, got {n_slots}.")

        # Random unique slot per label for each sample.
        slot_order = np.argsort(rng.random_sample((n, n_slots)), axis=1)
        chosen_slots = slot_order[:, :n_labels]

        jitter = 0.01
        r_min, r_max = 0.024, 0.034
        for j, shape_name in enumerate(LABEL_NAMES):
            slot_idx = chosen_slots[:, j]
            base_x = slots[slot_idx, 0]
            base_y = slots[slot_idx, 1]
            x = base_x + rng.uniform(-jitter, jitter, size=n)
            y = base_y + rng.uniform(-jitter, jitter, size=n)
            df[f"{shape_name}_x_frac"] = np.clip(x, 0.06, 0.94)
            df[f"{shape_name}_y_frac"] = np.clip(y, 0.06, 0.94)
            df[f"{shape_name}_r_frac"] = rng.uniform(r_min, r_max, size=n)
            df[f"{shape_name}_intensity"] = rng.randint(210, 255, size=n)

        df["position_layout_version"] = POSITION_LAYOUT_VERSION

        return df

    if cache_csv.exists():
        df = pd.read_csv(cache_csv)
        df = _filter_existing(df)
        if "pool_row_id" not in df.columns:
            df["pool_row_id"] = np.arange(len(df), dtype=np.int64)
        required = []
        for shape_name in LABEL_NAMES:
            required.extend(
                [
                    f"{shape_name}_label",
                    f"{shape_name}_x_frac",
                    f"{shape_name}_y_frac",
                    f"{shape_name}_r_frac",
                    f"{shape_name}_intensity",
                ]
            )
        has_required = set(required).issubset(set(df.columns))
        has_layout_version = "position_layout_version" in df.columns
        layout_ok = (
            has_layout_version
            and len(df) > 0
            and int(pd.to_numeric(df["position_layout_version"], errors="coerce").fillna(-1).iloc[0])
            == POSITION_LAYOUT_VERSION
        )
        rebuilt = False
        if (not has_required) or (not layout_ok):
            df = _assign_labels_and_params(df)
            rebuilt = True
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_csv, index=False)
        if rebuilt:
            print(
                f"Rebuilt toy pool with random non-overlap layout (v{POSITION_LAYOUT_VERSION}): "
                f"{cache_csv} (n={len(df)})"
            )
        else:
            print(f"Using existing toy pool: {cache_csv} (n={len(df)})")
        return df

    df = pd.read_csv(metadata_csv)
    df = df[df["split"] == "train"].copy()
    df = df[df["ViewPosition"].isin(view_positions)].copy()
    df = df[df["image_path"].notna()].copy()
    df = df.reset_index(drop=True)
    df = _filter_existing(df)
    df["pool_row_id"] = np.arange(len(df), dtype=np.int64)
    df = _assign_labels_and_params(df)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, index=False)
    print(
        f"Built toy pool with random non-overlap layout (v{POSITION_LAYOUT_VERSION}): "
        f"{cache_csv} (n={len(df)})"
    )
    return df


def sample_rows(rows: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    if len(rows) < n_samples:
        raise ValueError(f"Not enough rows in pool: {len(rows)} < {n_samples}")
    return rows.sample(n=n_samples, random_state=seed).reset_index(drop=True)


def build_clean_labels(
    n_samples: int, n_labels: int, pos_rate: float, seed: int
) -> np.ndarray:
    if not (0.0 <= pos_rate <= 1.0):
        raise ValueError(f"pos_rate must be in [0,1], got {pos_rate}")
    rng = np.random.RandomState(seed)
    y = np.zeros((n_samples, n_labels), dtype=np.int64)
    n_pos = int(round(pos_rate * n_samples))
    n_pos = max(0, min(n_pos, n_samples))
    for j in range(n_labels):
        col = np.zeros(n_samples, dtype=np.int64)
        col[:n_pos] = 1
        rng.shuffle(col)
        y[:, j] = col
    return y


def inject_entry_noise_per_class(
    y_clean: np.ndarray, noise_rate: float, rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Entry-level symmetric noise per class:
    - For each label class j, flip exactly round(noise_rate * N) entries in that column.
    - Same sample may be flipped in multiple classes (randomly).
    """
    n, k = y_clean.shape
    y_noisy = y_clean.copy()
    entry_flip = np.zeros((n, k), dtype=bool)
    n_flip_per_class = int(round(noise_rate * n))
    if n_flip_per_class <= 0:
        return y_noisy, entry_flip

    n_flip_per_class = min(n_flip_per_class, n)
    for j in range(k):
        flip_idx = rng.choice(n, size=n_flip_per_class, replace=False)
        y_noisy[flip_idx, j] = 1 - y_noisy[flip_idx, j]
        entry_flip[flip_idx, j] = True
    return y_noisy, entry_flip


def calc_macro_auroc(y_clean: np.ndarray, y_prob: np.ndarray) -> float:
    scores = []
    for j in range(y_clean.shape[1]):
        yj = y_clean[:, j]
        pj = y_prob[:, j]
        if len(np.unique(yj)) < 2:
            continue
        try:
            scores.append(float(roc_auc_score(yj, pj)))
        except ValueError:
            pass
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def calc_per_label_auroc(
    y_clean: np.ndarray,
    y_prob: np.ndarray,
    label_names: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for j, label_name in enumerate(label_names):
        yj = y_clean[:, j]
        pj = y_prob[:, j]
        has_both_classes = int(len(np.unique(yj)) >= 2)
        if has_both_classes:
            try:
                score = float(roc_auc_score(yj, pj))
            except ValueError:
                score = float("nan")
        else:
            score = float("nan")
        rows.append(
            {
                "label_index": int(j),
                "label_name": str(label_name),
                "oof_auroc_vs_clean": score,
                "has_both_classes": has_both_classes,
            }
        )
    return pd.DataFrame(rows)


def to_multilabel_list(y_binary: np.ndarray) -> List[List[int]]:
    labels_list: List[List[int]] = []
    for i in range(y_binary.shape[0]):
        labels_i = np.where(y_binary[i] == 1)[0].tolist()
        labels_list.append(labels_i)
    return labels_list


def estimate_issue_masks_official(y_noisy: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Official cleanlab multi-label APIs:
    - sample-level issues: multilabel_filter.find_label_issues(...)
    - per-class/entry issues: multilabel_filter.find_multilabel_issues_per_class(...)
    """
    labels_list = to_multilabel_list(y_noisy.astype(int))

    issue_sample_mask = multilabel_filter.find_label_issues(
        labels=labels_list,
        pred_probs=y_prob,
    )
    issue_sample_mask = np.asarray(issue_sample_mask).astype(bool)

    per_class = multilabel_filter.find_multilabel_issues_per_class(
        labels=labels_list,
        pred_probs=y_prob,
    )
    # cleanlab versions may return either:
    # 1) np.ndarray bool mask of shape (N, K), or
    # 2) list of length K, each bool mask of shape (N,).
    # Also supports index-list tuple when return_indices_ranked_by is used.
    n, k = y_noisy.shape
    issue_entry_mask = np.zeros((n, k), dtype=bool)
    if isinstance(per_class, np.ndarray) and per_class.dtype == bool and per_class.shape == (n, k):
        issue_entry_mask = per_class
    elif isinstance(per_class, list) and len(per_class) == k and all(
        isinstance(x, np.ndarray) and x.dtype == bool for x in per_class
    ):
        issue_entry_mask = np.stack(per_class, axis=1)
    elif isinstance(per_class, tuple) and len(per_class) >= 1 and isinstance(per_class[0], list):
        idx_list = per_class[0]
        for j, idx in enumerate(idx_list):
            issue_entry_mask[np.asarray(idx, dtype=int), j] = True
    else:
        raise RuntimeError(
            "Unexpected return format from cleanlab.find_multilabel_issues_per_class: "
            f"type={type(per_class)}"
        )

    return issue_sample_mask, issue_entry_mask


def build_rank_map_from_ranked(ranked: np.ndarray, n_samples: int) -> np.ndarray:
    rank_map = np.full(n_samples, np.nan, dtype=float)
    ranked = np.asarray(ranked)
    if ranked.dtype == bool:
        ranked_idx = np.where(ranked)[0]
    else:
        ranked_idx = ranked.astype(int)
    for r, idx in enumerate(ranked_idx, start=1):
        rank_map[int(idx)] = float(r)
    return rank_map


def get_issue_rank_map(labels_list: List[List[int]], y_prob: np.ndarray, method: str) -> np.ndarray:
    ranked = multilabel_filter.find_label_issues(
        labels=labels_list,
        pred_probs=y_prob,
        return_indices_ranked_by=method,
    )
    return build_rank_map_from_ranked(ranked=ranked, n_samples=len(labels_list))


def run_single_trial(
    pool_df: pd.DataFrame,
    dataset_root: Path,
    pos_rate: float,
    n_labels: int,
    seed: int,
    noise_rate: float,
    n_samples: int,
    image_size: int,
    n_splits: int,
    batch_size: int,
    epochs: int,
    lr: float,
    early_stopping_patience: int,
    recover_best_weights: bool,
    device: torch.device,
    model_backbone: str,
    overlay_rgb: Tuple[int, int, int],
    split_mode: str,
) -> Tuple[TrialResult, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shape_names = LABEL_NAMES[:n_labels]
    label_cols = [f"{s}_label" for s in shape_names]
    rows = sample_rows(pool_df, n_samples=n_samples, seed=seed)
    y_clean = build_clean_labels(
        n_samples=n_samples,
        n_labels=n_labels,
        pos_rate=pos_rate,
        seed=seed + 12345,
    )
    for j, c in enumerate(label_cols):
        rows[c] = y_clean[:, j]
    y_noisy, flip_entry_mask = inject_entry_noise_per_class(
        y_clean=y_clean, noise_rate=noise_rate, rng=np.random.RandomState(seed + 9999)
    )
    flip_sample_mask = flip_entry_mask.any(axis=1)

    trial_tag = f"K={n_labels},seed={seed},noise={noise_rate:.2f}"
    y_prob = get_oof_probabilities(
        rows=rows,
        dataset_root=dataset_root,
        y_train=y_noisy.astype(np.float32),
        shape_names=shape_names,
        image_size=image_size,
        n_splits=n_splits,
        seed=seed,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        early_stopping_patience=early_stopping_patience,
        recover_best_weights=recover_best_weights,
        device=device,
        model_backbone=model_backbone,
        overlay_rgb=overlay_rgb,
        trial_tag=trial_tag,
        split_mode=split_mode,
    )
    auroc_macro = calc_macro_auroc(y_clean=y_clean, y_prob=y_prob)
    auroc_macro_noisy = calc_macro_auroc(y_clean=y_noisy, y_prob=y_prob)
    per_label_auroc_df = calc_per_label_auroc(
        y_clean=y_clean,
        y_prob=y_prob,
        label_names=shape_names,
    )
    if n_labels == 1:
        y_noisy_col = y_noisy[:, 0].astype(int)
        y_prob_col = y_prob[:, 0]
        pred_probs_2c = np.column_stack([1.0 - y_prob_col, y_prob_col])

        issue_sample_mask = binary_filter.find_label_issues(
            labels=y_noisy_col,
            pred_probs=pred_probs_2c,
        ).astype(bool)
        issue_entry_mask = issue_sample_mask[:, None]

        rank_self = build_rank_map_from_ranked(
            ranked=binary_filter.find_label_issues(
                labels=y_noisy_col,
                pred_probs=pred_probs_2c,
                return_indices_ranked_by="self_confidence",
            ),
            n_samples=len(y_noisy_col),
        )
        rank_margin = build_rank_map_from_ranked(
            ranked=binary_filter.find_label_issues(
                labels=y_noisy_col,
                pred_probs=pred_probs_2c,
                return_indices_ranked_by="normalized_margin",
            ),
            n_samples=len(y_noisy_col),
        )
        rank_entropy = build_rank_map_from_ranked(
            ranked=binary_filter.find_label_issues(
                labels=y_noisy_col,
                pred_probs=pred_probs_2c,
                return_indices_ranked_by="confidence_weighted_entropy",
            ),
            n_samples=len(y_noisy_col),
        )

        sample_quality_self = binary_rank.get_label_quality_scores(
            labels=y_noisy_col,
            pred_probs=pred_probs_2c,
            method="self_confidence",
        )
        sample_quality_margin = binary_rank.get_label_quality_scores(
            labels=y_noisy_col,
            pred_probs=pred_probs_2c,
            method="normalized_margin",
        )
        sample_quality_entropy = binary_rank.get_label_quality_scores(
            labels=y_noisy_col,
            pred_probs=pred_probs_2c,
            method="confidence_weighted_entropy",
        )
        entry_quality_self = sample_quality_self[:, None]
        entry_quality_margin = sample_quality_margin[:, None]
        entry_quality_entropy = sample_quality_entropy[:, None]
    else:
        issue_sample_mask, issue_entry_mask = estimate_issue_masks_official(
            y_noisy=y_noisy,
            y_prob=y_prob,
        )
        labels_list = to_multilabel_list(y_noisy.astype(int))

        rank_self = get_issue_rank_map(labels_list, y_prob, method="self_confidence")
        rank_margin = get_issue_rank_map(labels_list, y_prob, method="normalized_margin")
        rank_entropy = get_issue_rank_map(labels_list, y_prob, method="confidence_weighted_entropy")

        sample_quality_self = multilabel_rank.get_label_quality_scores(
            labels=labels_list,
            pred_probs=y_prob,
            method="self_confidence",
        )
        sample_quality_margin = multilabel_rank.get_label_quality_scores(
            labels=labels_list,
            pred_probs=y_prob,
            method="normalized_margin",
        )
        sample_quality_entropy = multilabel_rank.get_label_quality_scores(
            labels=labels_list,
            pred_probs=y_prob,
            method="confidence_weighted_entropy",
        )

        entry_quality_self = multilabel_rank.get_label_quality_scores_per_class(
            labels=labels_list,
            pred_probs=y_prob,
            method="self_confidence",
        )
        entry_quality_margin = multilabel_rank.get_label_quality_scores_per_class(
            labels=labels_list,
            pred_probs=y_prob,
            method="normalized_margin",
        )
        entry_quality_entropy = multilabel_rank.get_label_quality_scores_per_class(
            labels=labels_list,
            pred_probs=y_prob,
            method="confidence_weighted_entropy",
        )

    tp = int((issue_sample_mask & flip_sample_mask).sum())
    fp = int((issue_sample_mask & ~flip_sample_mask).sum())
    fn = int((~issue_sample_mask & flip_sample_mask).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    trial_result = TrialResult(
        pos_rate=float(pos_rate),
        n_labels=n_labels,
        seed=seed,
        injected_noise_rate=float(noise_rate),
        true_noise_rate_sample=float(flip_sample_mask.mean()),
        true_noise_rate_entry=float(flip_entry_mask.mean()),
        estimated_noise_rate_sample=float(issue_sample_mask.mean()),
        estimated_noise_rate_entry=float(issue_entry_mask.mean()),
        issue_precision_sample=float(precision),
        issue_recall_sample=float(recall),
        oof_auroc_macro_vs_clean=float(auroc_macro),
        oof_auroc_macro_vs_noisy=float(auroc_macro_noisy),
    )

    pool_row_id = (
        rows["pool_row_id"].to_numpy(dtype=np.int64)
        if "pool_row_id" in rows.columns
        else np.arange(len(rows), dtype=np.int64)
    )
    image_paths = rows["image_path"].astype(str).to_numpy()
    sample_detail_df = pd.DataFrame(
        {
            "pos_rate": float(pos_rate),
            "n_labels": int(n_labels),
            "seed": int(seed),
            "injected_noise_rate": float(noise_rate),
            "sample_local_index": np.arange(len(rows), dtype=np.int64),
            "pool_row_id": pool_row_id,
            "image_path": image_paths,
            "true_flip_sample": flip_sample_mask.astype(int),
            "true_flip_entry_count": flip_entry_mask.sum(axis=1).astype(int),
            "est_issue_sample": issue_sample_mask.astype(int),
            "est_issue_entry_count": issue_entry_mask.sum(axis=1).astype(int),
            "issue_rank_self_confidence": rank_self,
            "issue_rank_normalized_margin": rank_margin,
            "issue_rank_confidence_weighted_entropy": rank_entropy,
            "sample_quality_self_confidence": sample_quality_self,
            "sample_quality_normalized_margin": sample_quality_margin,
            "sample_quality_confidence_weighted_entropy": sample_quality_entropy,
            "clean_labels": ["|".join(map(str, r.tolist())) for r in y_clean],
            "noisy_labels": ["|".join(map(str, r.tolist())) for r in y_noisy],
            "pred_probs": ["|".join(f"{x:.6f}" for x in r.tolist()) for r in y_prob],
            "flip_entry_mask": ["|".join(map(str, r.astype(int).tolist())) for r in flip_entry_mask],
            "issue_entry_mask": ["|".join(map(str, r.astype(int).tolist())) for r in issue_entry_mask],
        }
    )

    entry_rows = []
    for i in range(len(rows)):
        for j in range(n_labels):
            entry_rows.append(
                {
                    "pos_rate": float(pos_rate),
                    "n_labels": int(n_labels),
                    "seed": int(seed),
                    "injected_noise_rate": float(noise_rate),
                    "sample_local_index": int(i),
                    "pool_row_id": int(pool_row_id[i]),
                    "image_path": str(image_paths[i]),
                    "label_index": int(j),
                    "label_name": LABEL_NAMES[j],
                    "clean_label": int(y_clean[i, j]),
                    "noisy_label": int(y_noisy[i, j]),
                    "pred_prob": float(y_prob[i, j]),
                    "true_flip_entry": int(flip_entry_mask[i, j]),
                    "est_issue_entry": int(issue_entry_mask[i, j]),
                    "entry_quality_self_confidence": float(entry_quality_self[i, j]),
                    "entry_quality_normalized_margin": float(entry_quality_margin[i, j]),
                    "entry_quality_confidence_weighted_entropy": float(entry_quality_entropy[i, j]),
                }
            )
    entry_detail_df = pd.DataFrame(entry_rows)

    per_label_auroc_df.insert(0, "injected_noise_rate", float(noise_rate))
    per_label_auroc_df.insert(0, "seed", int(seed))
    per_label_auroc_df.insert(0, "n_labels", int(n_labels))
    per_label_auroc_df.insert(0, "pos_rate", float(pos_rate))

    return trial_result, sample_detail_df, entry_detail_df, per_label_auroc_df


def plot_k_comparison(
    summary_df: pd.DataFrame,
    output_png: Path,
    pos_rate: float,
    n_label_values: Sequence[int],
) -> None:
    k_values = list(n_label_values)
    n_panels = len(k_values)
    n_cols = min(4, max(1, n_panels))
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    for idx, k in enumerate(k_values):
        ax1 = axes[idx]
        ax2 = ax1.twinx()
        data = summary_df[summary_df["n_labels"] == k].sort_values("injected_noise_rate")
        if len(data) == 0:
            ax1.axis("off")
            continue
        ax1.errorbar(
            data["injected_noise_rate"],
            data["estimated_noise_rate_sample_mean"],
            yerr=data["estimated_noise_rate_sample_std"],
            fmt="o-",
            capsize=3,
            color="tab:blue",
            label="Estimated(sample)",
        )
        ax1.errorbar(
            data["injected_noise_rate"],
            data["estimated_noise_rate_entry_mean"],
            yerr=data["estimated_noise_rate_entry_std"],
            fmt="d-",
            capsize=3,
            color="tab:green",
            label="Estimated(entry)",
        )
        ax1.plot(
            data["injected_noise_rate"],
            data["true_noise_rate_sample_mean"],
            "k:",
            linewidth=1.5,
            label="True(sample)",
        )
        ax1.plot(
            data["injected_noise_rate"],
            data["true_noise_rate_entry_mean"],
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label="True(entry)",
        )
        auroc_yerr = (
            data["oof_auroc_macro_vs_clean_std"].fillna(0.0)
            if "oof_auroc_macro_vs_clean_std" in data.columns
            else None
        )
        ax2.errorbar(
            data["injected_noise_rate"],
            data["oof_auroc_macro_vs_clean_mean"],
            yerr=auroc_yerr,
            fmt="s--",
            capsize=3,
            color="tab:orange",
            label="AUROC macro (vs clean GT)",
        )
        if "oof_auroc_macro_vs_noisy_mean" in data.columns:
            auroc_noisy_yerr = (
                data["oof_auroc_macro_vs_noisy_std"].fillna(0.0)
                if "oof_auroc_macro_vs_noisy_std" in data.columns
                else None
            )
            ax2.errorbar(
                data["injected_noise_rate"],
                data["oof_auroc_macro_vs_noisy_mean"],
                yerr=auroc_noisy_yerr,
                fmt="^-.",
                capsize=3,
                color="tab:red",
                label="AUROC macro (vs pseudo/noisy)",
            )
        ax1.set_title(f"K={k} labels (pos_rate={pos_rate:.1f})")
        ax1.set_xlabel("Injected Noise Rate (per-class entry-level)")
        ax1.set_ylabel("Estimated Noise Rate (sample-level)")
        ax2.set_ylabel("OOF AUROC")
        x_max = float(max(0.5, data["injected_noise_rate"].max()))
        ax1.set_xlim(0.0, x_max)
        ax1.set_ylim(0.0, 1.0)
        ax2.set_ylim(0.0, 1.02)
        ax1.grid(alpha=0.3)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    for idx in range(n_panels, len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def save_multilabel_examples_grid(
    pool_df: pd.DataFrame,
    dataset_root: Path,
    image_size: int,
    output_png: Path,
    overlay_rgb: Tuple[int, int, int],
    label_names: Sequence[str],
    seed: int = 123,
) -> None:
    label_cols = [f"{s}_label" for s in label_names]
    any_mask = (pool_df[label_cols].sum(axis=1) > 0)
    pos_df = pool_df[any_mask]
    if len(pos_df) < 5:
        return

    pos_rows = pos_df.sample(n=5, random_state=seed).reset_index(drop=True)
    neg_rows = pos_rows.copy()

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle("Multilabel Toy Examples: Overlay On (Top) vs Raw Background (Bottom)", fontsize=12)

    for col in range(5):
        for row_idx, rows_sel in enumerate([pos_rows, neg_rows]):
            row = rows_sel.iloc[col]
            rel_path = row["image_path"]
            img_path = dataset_root / rel_path
            img = Image.open(img_path).convert("RGB").resize((image_size, image_size))
            draw = ImageDraw.Draw(img)
            w, h = img.size

            if row_idx == 0:
                for label_name in label_names:
                    draw_overlay_label(
                        draw=draw,
                        row=row,
                        label_name=label_name,
                        w=w,
                        h=h,
                        overlay_rgb=overlay_rgb,
                    )

            axes[row_idx, col].imshow(np.asarray(img))
            axes[row_idx, col].axis("off")

    axes[0, 0].set_ylabel("Overlay on", fontsize=10)
    axes[1, 0].set_ylabel("Raw image", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def parse_noise_levels(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    for v in values:
        if v < 0 or v > 1:
            raise ValueError(f"noise rate must be in [0,1], got {v}")
    return values


def parse_pos_rates(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    for v in values:
        if v < 0 or v > 1:
            raise ValueError(f"pos_rate must be in [0,1], got {v}")
    return values


def parse_n_labels_list(text: str, max_n_labels: int) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("n-labels-list cannot be empty.")
    for v in values:
        if v < 1 or v > max_n_labels:
            raise ValueError(f"n_labels must be in [1,{max_n_labels}], got {v}")
    return values


def main():
    parser = argparse.ArgumentParser(
        description="CXR multi-label toy noise-rate validation with synthetic symbols."
    )
    parser.add_argument("--metadata-csv", type=str, default="datasets/mimic-cxr-clean/train/metadata.csv")
    parser.add_argument("--dataset-root", type=str, default="datasets/mimic-cxr-clean/train")
    parser.add_argument(
        "--output-root",
        type=str,
        default="NoiseRate/cxr_toy_experiment/results_multilabel",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--cache-csv",
        type=str,
        default="NoiseRate/cxr_toy_experiment/cache/toy_pool_train_multilabel.csv",
    )
    parser.add_argument("--pool-seed", type=int, default=20260211)
    parser.add_argument("--n-samples", type=int, default=1200)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--split-mode",
        type=str,
        default="kfold",
        choices=["kfold", "stratified_label_count", "stratified_multilabel_code", "auto"],
        help="Cross-validation split strategy. Use kfold for consistent comparability across trials.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--no-recover-best-weights", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-levels", type=str, default="0,0.1,0.2,0.3,0.5")
    parser.add_argument(
        "--pos-rates",
        type=str,
        default="0.5",
        help="Comma-separated positive rates for each binary label, e.g. '0.1,0.2,0.3,0.4,0.5'.",
    )
    parser.add_argument(
        "--n-labels-list",
        type=str,
        default="1,2,3,4",
        help=f"Comma-separated K values (number of labels). Max is {len(LABEL_NAMES)}.",
    )
    parser.add_argument("--view-positions", type=str, default="PA,AP")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--overlay-color",
        type=str,
        default="white",
        choices=["white", "blue"],
        help="Overlay shape color.",
    )
    parser.add_argument(
        "--model-backbone",
        type=str,
        default="densenet121_pretrained",
        choices=["lightcnn", "densenet121_pretrained"],
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.output_root) / f"{args.model_backbone}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = Path(args.metadata_csv)
    dataset_root = Path(args.dataset_root)
    cache_csv = Path(args.cache_csv)
    n_samples = int(args.n_samples)
    noise_levels = parse_noise_levels(args.noise_levels)
    pos_rates = parse_pos_rates(args.pos_rates)
    n_labels_list = parse_n_labels_list(args.n_labels_list, max_n_labels=len(LABEL_NAMES))
    view_positions = [x.strip() for x in args.view_positions.split(",") if x.strip()]
    recover_best = not args.no_recover_best_weights
    overlay_rgb = OVERLAY_COLOR_MAP[args.overlay_color]

    pool_df = build_or_load_pool(
        metadata_csv=metadata_csv,
        dataset_root=dataset_root,
        cache_csv=cache_csv,
        view_positions=view_positions,
        pool_seed=args.pool_seed,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Model backbone: {args.model_backbone}")
    print(f"Split mode: {args.split_mode}")
    print(f"Output dir: {output_dir}")

    seeds = list(range(args.n_seeds))
    all_results: List[TrialResult] = []
    sample_detail_csv = output_dir / "cxr_toy_multilabel_sample_details.csv"
    entry_detail_csv = output_dir / "cxr_toy_multilabel_entry_details.csv"
    sample_issues_csv = output_dir / "cxr_toy_multilabel_sample_issues_only.csv"
    entry_issues_csv = output_dir / "cxr_toy_multilabel_entry_issues_only.csv"
    per_label_trials_csv = output_dir / "cxr_toy_multilabel_per_label_auroc_trials.csv"
    per_label_summary_csv = output_dir / "cxr_toy_multilabel_per_label_auroc_summary.csv"
    for p in [
        sample_detail_csv,
        entry_detail_csv,
        sample_issues_csv,
        entry_issues_csv,
        per_label_trials_csv,
        per_label_summary_csv,
    ]:
        if p.exists():
            p.unlink()
    sample_written = False
    entry_written = False
    sample_issues_written = False
    entry_issues_written = False
    per_label_written = False
    total_trials = len(pos_rates) * len(n_labels_list) * len(seeds) * len(noise_levels)
    trial_counter = 0

    for pos_rate in pos_rates:
        for n_labels in n_labels_list:
            for seed in seeds:
                for noise_rate in noise_levels:
                    trial_counter += 1
                    print(
                        f"\n[trial {trial_counter}/{total_trials}] "
                        f"pos_rate={pos_rate:.2f}, K={n_labels}, seed={seed}, "
                        f"injected_noise={noise_rate:.2f}, n_samples={n_samples}"
                    )
                    result, sample_detail_df, entry_detail_df, per_label_auroc_df = run_single_trial(
                        pool_df=pool_df,
                        dataset_root=dataset_root,
                        pos_rate=pos_rate,
                        n_labels=n_labels,
                        seed=seed,
                        noise_rate=noise_rate,
                        n_samples=n_samples,
                        image_size=args.image_size,
                        n_splits=args.n_splits,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        lr=args.lr,
                        early_stopping_patience=args.early_stopping_patience,
                        recover_best_weights=recover_best,
                        device=device,
                        model_backbone=args.model_backbone,
                        overlay_rgb=overlay_rgb,
                        split_mode=args.split_mode,
                    )
                    all_results.append(result)
                    sample_detail_df.to_csv(
                        sample_detail_csv,
                        mode="a",
                        header=not sample_written,
                        index=False,
                    )
                    sample_written = True
                    entry_detail_df.to_csv(
                        entry_detail_csv,
                        mode="a",
                        header=not entry_written,
                        index=False,
                    )
                    entry_written = True
                    per_label_auroc_df.to_csv(
                        per_label_trials_csv,
                        mode="a",
                        header=not per_label_written,
                        index=False,
                    )
                    per_label_written = True
                    sample_issues_df = sample_detail_df[sample_detail_df["est_issue_sample"] == 1]
                    if len(sample_issues_df) > 0:
                        sample_issues_df.to_csv(
                            sample_issues_csv,
                            mode="a",
                            header=not sample_issues_written,
                            index=False,
                        )
                        sample_issues_written = True
                    entry_issues_df = entry_detail_df[entry_detail_df["est_issue_entry"] == 1]
                    if len(entry_issues_df) > 0:
                        entry_issues_df.to_csv(
                            entry_issues_csv,
                            mode="a",
                            header=not entry_issues_written,
                            index=False,
                        )
                        entry_issues_written = True
                    print(
                        f"pos_rate={pos_rate:.2f} K={n_labels} seed={seed} noise={noise_rate:.2f} "
                        f"true_sample={result.true_noise_rate_sample:.3f} "
                        f"est_sample={result.estimated_noise_rate_sample:.3f} "
                        f"auroc_clean={result.oof_auroc_macro_vs_clean:.3f} "
                        f"auroc_pseudo={result.oof_auroc_macro_vs_noisy:.3f}"
                    )

    trial_df = pd.DataFrame([r.__dict__ for r in all_results])
    trial_csv = output_dir / "cxr_toy_multilabel_trials.csv"
    trial_df.to_csv(trial_csv, index=False)

    summary_df = (
        trial_df.groupby(["pos_rate", "n_labels", "injected_noise_rate"], as_index=False)
        .agg(
            true_noise_rate_sample_mean=("true_noise_rate_sample", "mean"),
            true_noise_rate_sample_std=("true_noise_rate_sample", "std"),
            true_noise_rate_entry_mean=("true_noise_rate_entry", "mean"),
            true_noise_rate_entry_std=("true_noise_rate_entry", "std"),
            estimated_noise_rate_sample_mean=("estimated_noise_rate_sample", "mean"),
            estimated_noise_rate_sample_std=("estimated_noise_rate_sample", "std"),
            estimated_noise_rate_entry_mean=("estimated_noise_rate_entry", "mean"),
            estimated_noise_rate_entry_std=("estimated_noise_rate_entry", "std"),
            issue_precision_sample_mean=("issue_precision_sample", "mean"),
            issue_recall_sample_mean=("issue_recall_sample", "mean"),
            oof_auroc_macro_vs_clean_mean=("oof_auroc_macro_vs_clean", "mean"),
            oof_auroc_macro_vs_clean_std=("oof_auroc_macro_vs_clean", "std"),
            oof_auroc_macro_vs_noisy_mean=("oof_auroc_macro_vs_noisy", "mean"),
            oof_auroc_macro_vs_noisy_std=("oof_auroc_macro_vs_noisy", "std"),
        )
        .sort_values(["pos_rate", "n_labels", "injected_noise_rate"])
    )
    summary_csv = output_dir / "cxr_toy_multilabel_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    if per_label_written:
        per_label_trials_df = pd.read_csv(per_label_trials_csv)
        per_label_summary_df = (
            per_label_trials_df.groupby(
                ["pos_rate", "n_labels", "injected_noise_rate", "label_index", "label_name"],
                as_index=False,
            )
            .agg(
                oof_auroc_vs_clean_mean=("oof_auroc_vs_clean", "mean"),
                oof_auroc_vs_clean_std=("oof_auroc_vs_clean", "std"),
                n_trials=("label_index", "size"),
                has_both_classes_all=("has_both_classes", "min"),
            )
            .sort_values(["pos_rate", "n_labels", "injected_noise_rate", "label_index"])
        )
        per_label_summary_df.to_csv(per_label_summary_csv, index=False)

    plot_paths: List[Path] = []
    for pos_rate in pos_rates:
        plot_path = output_dir / f"cxr_toy_multilabel_k_sweep_pos{int(round(pos_rate * 100)):02d}.png"
        d = summary_df[summary_df["pos_rate"] == pos_rate].copy()
        plot_k_comparison(
            summary_df=d,
            output_png=plot_path,
            pos_rate=pos_rate,
            n_label_values=n_labels_list,
        )
        plot_paths.append(plot_path)
    examples_path = output_dir / "cxr_toy_multilabel_examples_5x2.png"
    save_multilabel_examples_grid(
        pool_df=pool_df,
        dataset_root=dataset_root,
        image_size=args.image_size,
        output_png=examples_path,
        overlay_rgb=overlay_rgb,
        label_names=LABEL_NAMES[: max(n_labels_list)],
        seed=args.pool_seed,
    )

    report_path = output_dir / "cxr_toy_multilabel_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CXR Multi-label Toy Noise Validation Report\n")
        f.write("==========================================\n\n")
        f.write("noise_injection_mode=per-class entry-level symmetric flips\n")
        f.write(f"metadata_csv={metadata_csv}\n")
        f.write(f"dataset_root={dataset_root}\n")
        f.write(f"cache_csv={cache_csv}\n")
        f.write(f"position_layout_version={POSITION_LAYOUT_VERSION}\n")
        f.write(f"pool_seed={args.pool_seed}\n")
        f.write(f"model_backbone={args.model_backbone}\n")
        f.write(f"n_samples={n_samples}\n")
        f.write(f"image_size={args.image_size}\n")
        f.write(f"n_seeds={args.n_seeds}\n")
        f.write(f"n_splits={args.n_splits}\n")
        f.write(f"split_mode={args.split_mode}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"epochs={args.epochs}\n")
        f.write(f"early_stopping_patience={args.early_stopping_patience}\n")
        f.write(f"recover_best_weights={recover_best}\n")
        f.write(f"lr={args.lr}\n")
        f.write(f"noise_levels={noise_levels}\n")
        f.write(f"pos_rates={pos_rates}\n")
        f.write(f"n_labels_list={n_labels_list}\n")
        f.write(f"view_positions={view_positions}\n")
        f.write(f"overlay_color={args.overlay_color}\n")
        f.write(f"device={device}\n\n")
        f.write(f"sample_detail_csv={sample_detail_csv}\n")
        f.write(f"entry_detail_csv={entry_detail_csv}\n")
        f.write(f"sample_issues_csv={sample_issues_csv}\n")
        f.write(f"entry_issues_csv={entry_issues_csv}\n\n")
        f.write(f"per_label_auroc_trials_csv={per_label_trials_csv}\n")
        f.write(f"per_label_auroc_summary_csv={per_label_summary_csv}\n\n")
        for pos_rate in pos_rates:
            for k in n_labels_list:
                d = summary_df[
                    (summary_df["pos_rate"] == pos_rate) & (summary_df["n_labels"] == k)
                ].sort_values("injected_noise_rate")
                x = d["injected_noise_rate"].to_numpy()
                y = d["estimated_noise_rate_sample_mean"].to_numpy()
                if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
                    corr = float(np.corrcoef(x, y)[0, 1])
                else:
                    corr = float("nan")
                f.write(
                    f"pos_rate={pos_rate:.2f}, K={k} Pearson(injected, estimated_sample_mean)={corr:.4f}\n"
                )

    print("\nSaved:")
    print(f"- Trials:  {trial_csv}")
    print(f"- Summary: {summary_csv}")
    print(f"- Sample details: {sample_detail_csv}")
    print(f"- Entry details:  {entry_detail_csv}")
    print(f"- Sample issues:  {sample_issues_csv}")
    print(f"- Entry issues:   {entry_issues_csv}")
    print(f"- Per-label AUROC trials:  {per_label_trials_csv}")
    print(f"- Per-label AUROC summary: {per_label_summary_csv}")
    for p in plot_paths:
        print(f"- Plot:    {p}")
    print(f"- Examples:{examples_path}")
    print(f"- Report:  {report_path}")


if __name__ == "__main__":
    main()
