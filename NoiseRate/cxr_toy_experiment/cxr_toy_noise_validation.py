"""
Toy experiment in image domain:
Use MIMIC-CXR backgrounds + synthetic circle overlays to validate whether
cleanlab-estimated noise rate tracks injected label noise.
"""

import argparse
import copy
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from cleanlab.filter import find_label_issues
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from NoiseRate.models.lightweight_cnn import LightweightCNN

OVERLAY_COLOR_MAP = {
    "white": (255, 255, 255),
    "blue": (0, 120, 255),
}


@dataclass
class TrialResult:
    seed: int
    injected_noise_rate: float
    true_noise_rate: float
    estimated_noise_rate: float
    issue_precision: float
    issue_recall: float
    oof_auroc_vs_clean: float


class CXRBinaryToyDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        dataset_root: Path,
        image_size: int,
        train_label: np.ndarray,
        input_channels: int,
        overlay_rgb: Tuple[int, int, int],
    ):
        self.rows = rows.reset_index(drop=True)
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.train_label = train_label.astype(np.int64)
        self.input_channels = int(input_channels)
        self.overlay_rgb = tuple(overlay_rgb)

    def __len__(self) -> int:
        return len(self.rows)

    def _add_overlay(self, img: Image.Image, index: int) -> Image.Image:
        row = self.rows.iloc[index]
        signal_label = int(row["signal_label"])
        if signal_label == 0:
            return img

        draw = ImageDraw.Draw(img)
        w, h = img.size

        # Configurable-color circular signal on CXR background.
        x = int(float(row["circle_x_frac"]) * w)
        y = int(float(row["circle_y_frac"]) * h)
        size = int(float(row["circle_r_frac"]) * min(w, h))
        size = max(6, min(size, min(w, h) // 3))
        x = max(size, min(x, w - size))
        y = max(size, min(y, h - size))
        draw.ellipse((x - size, y - size, x + size, y + size), fill=self.overlay_rgb)

        return img

    def __getitem__(self, index: int):
        rel_path = self.rows.iloc[index]["image_path"]
        img_path = self.dataset_root / rel_path
        img = Image.open(img_path).convert("RGB").resize((self.image_size, self.image_size))
        img = self._add_overlay(img, index)

        x = np.asarray(img, dtype=np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x)
        y = int(self.train_label[index])
        return x, y


def inject_symmetric_noise(
    y_clean: np.ndarray, noise_rate: float, rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    y_noisy = y_clean.copy()
    flip_mask = rng.rand(len(y_clean)) < noise_rate
    y_noisy[flip_mask] = 1 - y_noisy[flip_mask]
    return y_noisy, flip_mask


def train_binary_cnn(
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
    criterion = nn.CrossEntropyLoss()
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
            fold_prob = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(fold_prob)

    return np.vstack(probs)


def create_model(model_backbone: str) -> Tuple[nn.Module, int]:
    if model_backbone == "lightcnn":
        return LightweightCNN(num_classes=2, input_channels=3, dropout=0.3), 3

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
        model.classifier = nn.Linear(in_features, 2)
        return model, 3

    raise ValueError(f"Unknown model_backbone: {model_backbone}")


def get_oof_probabilities(
    dataset: CXRBinaryToyDataset,
    y_train_labels: np.ndarray,
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
    trial_tag: str = "",
) -> np.ndarray:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_probs = np.zeros((len(dataset), 2), dtype=np.float32)
    split_iter = list(skf.split(np.arange(len(dataset)), y_train_labels))
    total_folds = len(split_iter)

    for fold_no, (train_idx, val_idx) in enumerate(split_iter, start=1):
        train_rows = dataset.rows.iloc[train_idx].reset_index(drop=True)
        val_rows = dataset.rows.iloc[val_idx].reset_index(drop=True)

        train_labels = y_train_labels[train_idx]
        val_labels = y_train_labels[val_idx]

        model, input_channels = create_model(model_backbone)
        train_ds = CXRBinaryToyDataset(
            rows=train_rows,
            dataset_root=dataset.dataset_root,
            image_size=dataset.image_size,
            train_label=train_labels,
            input_channels=input_channels,
            overlay_rgb=overlay_rgb,
        )
        val_ds = CXRBinaryToyDataset(
            rows=val_rows,
            dataset_root=dataset.dataset_root,
            image_size=dataset.image_size,
            train_label=val_labels,
            input_channels=input_channels,
            overlay_rgb=overlay_rgb,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        fold_probs = train_binary_cnn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            early_stopping_patience=early_stopping_patience,
            recover_best_weights=recover_best_weights,
            trial_tag=trial_tag or f"seed={seed}",
            fold_idx=fold_no,
            n_folds=total_folds,
        )
        oof_probs[val_idx] = fold_probs

    return oof_probs


def calc_auroc_vs_clean(y_clean: np.ndarray, oof_probs: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_clean, oof_probs[:, 1]))
    except ValueError:
        return float("nan")


def build_or_load_toy_pool(
    metadata_csv: Path,
    dataset_root: Path,
    cache_csv: Path,
    view_positions: Sequence[str],
    pool_seed: int,
) -> pd.DataFrame:
    def _filter_existing_images(df_in: pd.DataFrame) -> pd.DataFrame:
        exists_mask = df_in["image_path"].astype(str).map(lambda p: (dataset_root / p).exists())
        df_out = df_in[exists_mask].copy().reset_index(drop=True)
        dropped = int((~exists_mask).sum())
        if dropped > 0:
            print(f"Dropped missing-image rows: {dropped}")
        return df_out

    def _assign_synthetic_columns(df_in: pd.DataFrame) -> pd.DataFrame:
        n_local = len(df_in)
        if n_local < 2:
            raise ValueError("Not enough valid train samples after filtering missing images.")

        rng_local = np.random.RandomState(pool_seed)
        signal_local = np.zeros(n_local, dtype=np.int64)
        signal_local[: n_local // 2] = 1
        rng_local.shuffle(signal_local)

        df_out = df_in.copy()
        df_out["signal_label"] = signal_local
        df_out["circle_x_frac"] = rng_local.uniform(0.2, 0.8, size=n_local)
        df_out["circle_y_frac"] = rng_local.uniform(0.2, 0.8, size=n_local)
        df_out["circle_r_frac"] = rng_local.uniform(0.06, 0.16, size=n_local)
        df_out["circle_intensity"] = rng_local.randint(210, 255, size=n_local)
        return df_out

    if cache_csv.exists():
        df = pd.read_csv(cache_csv)
        df = _filter_existing_images(df)
        required_cols = {
            "signal_label",
            "circle_x_frac",
            "circle_y_frac",
            "circle_r_frac",
            "circle_intensity",
        }
        if not required_cols.issubset(set(df.columns)):
            df = _assign_synthetic_columns(df)
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_csv, index=False)
        print(f"Using existing toy pool: {cache_csv} (n={len(df)})")
        return df

    df = pd.read_csv(metadata_csv)
    df = df[df["split"] == "train"].copy()
    df = df[df["ViewPosition"].isin(view_positions)].copy()
    df = df[df["image_path"].notna()].copy()
    df = df.reset_index(drop=True)
    df = _filter_existing_images(df)
    df = _assign_synthetic_columns(df)

    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, index=False)
    print(f"Built toy pool: {cache_csv} (n={len(df)})")
    return df


def sample_from_toy_pool(pool_df: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    if len(pool_df) < n_samples:
        raise ValueError(f"Not enough samples in toy pool: {len(pool_df)} < {n_samples}")
    if n_samples % 2 != 0:
        raise ValueError(
            f"n_samples must be even for strict 1:1 class balance, got {n_samples}"
        )

    n_per_class = n_samples // 2
    pos_df = pool_df[pool_df["signal_label"] == 1]
    neg_df = pool_df[pool_df["signal_label"] == 0]
    if len(pos_df) < n_per_class or len(neg_df) < n_per_class:
        raise ValueError(
            "Not enough samples per class in toy pool for strict 1:1 sampling: "
            f"need {n_per_class} each, got pos={len(pos_df)}, neg={len(neg_df)}"
        )

    pos_sample = pos_df.sample(n=n_per_class, random_state=seed)
    neg_sample = neg_df.sample(n=n_per_class, random_state=seed + 1)
    out = pd.concat([pos_sample, neg_sample], axis=0)
    out = out.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    return out


def run_single_trial(
    pool_df: pd.DataFrame,
    dataset_root: Path,
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
) -> TrialResult:
    rows = sample_from_toy_pool(pool_df=pool_df, n_samples=n_samples, seed=seed)
    y_clean = rows["signal_label"].to_numpy(dtype=np.int64)

    y_noisy, flip_mask = inject_symmetric_noise(y_clean, noise_rate, rng=np.random.RandomState(seed + 10000))

    full_ds = CXRBinaryToyDataset(
        rows=rows,
        dataset_root=dataset_root,
        image_size=image_size,
        train_label=y_noisy,
        input_channels=1,
        overlay_rgb=overlay_rgb,
    )

    oof_probs = get_oof_probabilities(
        dataset=full_ds,
        y_train_labels=y_noisy,
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
        trial_tag=f"seed={seed},noise={noise_rate:.2f}",
    )
    auroc_vs_clean = calc_auroc_vs_clean(y_clean, oof_probs)

    issue_indices = find_label_issues(
        labels=y_noisy,
        pred_probs=oof_probs,
        return_indices_ranked_by="self_confidence",
    )
    issue_mask = np.zeros(len(y_noisy), dtype=bool)
    issue_mask[issue_indices] = True

    tp = int((issue_mask & flip_mask).sum())
    fp = int((issue_mask & ~flip_mask).sum())
    fn = int((~issue_mask & flip_mask).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    return TrialResult(
        seed=seed,
        injected_noise_rate=float(noise_rate),
        true_noise_rate=float(flip_mask.mean()),
        estimated_noise_rate=float(issue_mask.mean()),
        issue_precision=float(precision),
        issue_recall=float(recall),
        oof_auroc_vs_clean=float(auroc_vs_clean),
    )


def plot_main_curves(summary_df: pd.DataFrame, output_png: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()

    data = summary_df.sort_values("injected_noise_rate")
    ax1.errorbar(
        data["injected_noise_rate"],
        data["estimated_noise_rate_mean"],
        yerr=data["estimated_noise_rate_std"],
        fmt="o-",
        capsize=3,
        color="tab:blue",
        label="Estimated noise",
    )
    ax1.plot([0, 0.5], [0, 0.5], linestyle=":", color="gray", alpha=0.7, label="Ideal y=x")

    ax2.plot(
        data["injected_noise_rate"],
        data["oof_auroc_vs_clean_mean"],
        "s--",
        color="tab:orange",
        label="AUROC vs clean labels",
    )

    ax1.set_xlabel("Injected Noise Rate")
    ax1.set_ylabel("Estimated Noise Rate")
    ax2.set_ylabel("OOF AUROC")
    ax1.set_xlim(0.0, 0.5)
    ax1.set_ylim(0.0, 0.55)
    ax2.set_ylim(0.0, 1.02)
    ax1.grid(alpha=0.3)
    ax1.set_title("CXR Toy: Injected vs Estimated Noise + AUROC")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def save_toy_examples_grid(
    pool_df: pd.DataFrame,
    dataset_root: Path,
    image_size: int,
    output_png: Path,
    overlay_rgb: Tuple[int, int, int],
    seed: int = 123,
) -> None:
    rng = np.random.RandomState(seed)
    pos_df = pool_df[pool_df["signal_label"] == 1]
    neg_df = pool_df[pool_df["signal_label"] == 0]
    if len(pos_df) < 5 or len(neg_df) < 5:
        return

    pos_rows = pos_df.sample(n=5, random_state=seed).reset_index(drop=True)
    neg_rows = neg_df.sample(n=5, random_state=seed + 1).reset_index(drop=True)

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle("Toy Examples: Circle Overlay (Top) vs No Overlay (Bottom)", fontsize=12)

    for col in range(5):
        for row_idx, rows_sel in enumerate([pos_rows, neg_rows]):
            row = rows_sel.iloc[col]
            rel_path = row["image_path"]
            img_path = dataset_root / rel_path
            img = Image.open(img_path).convert("RGB").resize((image_size, image_size))
            if int(row["signal_label"]) == 1:
                draw = ImageDraw.Draw(img)
                w, h = img.size
                x = int(float(row["circle_x_frac"]) * w)
                y = int(float(row["circle_y_frac"]) * h)
                size = int(float(row["circle_r_frac"]) * min(w, h))
                size = max(6, min(size, min(w, h) // 3))
                x = max(size, min(x, w - size))
                y = max(size, min(y, h - size))
                draw.ellipse((x - size, y - size, x + size, y + size), fill=overlay_rgb)

            axes[row_idx, col].imshow(np.asarray(img))
            axes[row_idx, col].axis("off")

    axes[0, 0].set_ylabel("Circle overlay", fontsize=10)
    axes[1, 0].set_ylabel("No overlay", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def parse_noise_levels(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    for v in values:
        if v < 0 or v > 1:
            raise ValueError(f"noise rate must be in [0,1], got {v}")
    return values


def main():
    parser = argparse.ArgumentParser(description="CXR+circle toy noise-rate validation.")
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="datasets/mimic-cxr-clean/train/metadata.csv",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="datasets/mimic-cxr-clean/train",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="NoiseRate/cxr_toy_experiment/results",
        help="Root folder where run-specific result folders will be created.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional explicit output dir. If omitted, uses output-root/<backbone>_<timestamp>.",
    )
    parser.add_argument(
        "--cache-csv",
        type=str,
        default="NoiseRate/cxr_toy_experiment/cache/toy_pool_train_pa_ap.csv",
    )
    parser.add_argument("--pool-seed", type=int, default=20260210)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1200,
        help="Number of samples per trial. Must be even for strict 1:1 class balance.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Stop training if val loss does not improve for this many epochs.",
    )
    parser.add_argument(
        "--no-recover-best-weights",
        action="store_true",
        help="Disable restoring the best-validation-loss weights at the end of each fold.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-levels", type=str, default="0,0.1,0.2,0.3,0.5")
    parser.add_argument("--view-positions", type=str, default="PA,AP")
    parser.add_argument("--device", type=str, default="cpu")
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
        default="lightcnn",
        choices=["lightcnn", "densenet121_pretrained"],
        help="Run a single backbone per execution.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.output_root) / f"{args.model_backbone}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = Path(args.metadata_csv)
    dataset_root = Path(args.dataset_root)
    cache_csv = Path(args.cache_csv)
    noise_levels = parse_noise_levels(args.noise_levels)
    n_samples = int(args.n_samples)
    view_positions = [x.strip() for x in args.view_positions.split(",") if x.strip()]
    overlay_rgb = OVERLAY_COLOR_MAP[args.overlay_color]
    pool_df = build_or_load_toy_pool(
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

    torch.manual_seed(0)
    np.random.seed(0)

    seeds = list(range(args.n_seeds))
    all_results: List[TrialResult] = []
    total_trials = len(seeds) * len(noise_levels)
    trial_counter = 0
    print(f"Model backbone: {args.model_backbone}")
    print(f"Output dir: {output_dir}")

    for seed in seeds:
        for noise_rate in noise_levels:
            trial_counter += 1
            print(
                f"\n[trial {trial_counter}/{total_trials}] "
                f"seed={seed}, injected_noise={noise_rate:.2f}, n_samples={n_samples}"
            )
            result = run_single_trial(
                pool_df=pool_df,
                dataset_root=dataset_root,
                seed=seed,
                noise_rate=noise_rate,
                n_samples=n_samples,
                image_size=args.image_size,
                n_splits=args.n_splits,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                early_stopping_patience=args.early_stopping_patience,
                recover_best_weights=not args.no_recover_best_weights,
                device=device,
                model_backbone=args.model_backbone,
                overlay_rgb=overlay_rgb,
            )
            all_results.append(result)
            print(
                f"seed={seed} noise={noise_rate:.2f} true={result.true_noise_rate:.3f} "
                f"est={result.estimated_noise_rate:.3f} auroc={result.oof_auroc_vs_clean:.3f}"
            )

    trial_df = pd.DataFrame([r.__dict__ for r in all_results])
    trial_csv = output_dir / "cxr_toy_trials.csv"
    trial_df.to_csv(trial_csv, index=False)

    summary_df = (
        trial_df.groupby(["injected_noise_rate"], as_index=False)
        .agg(
            true_noise_rate_mean=("true_noise_rate", "mean"),
            true_noise_rate_std=("true_noise_rate", "std"),
            estimated_noise_rate_mean=("estimated_noise_rate", "mean"),
            estimated_noise_rate_std=("estimated_noise_rate", "std"),
            issue_precision_mean=("issue_precision", "mean"),
            issue_recall_mean=("issue_recall", "mean"),
            oof_auroc_vs_clean_mean=("oof_auroc_vs_clean", "mean"),
        )
        .sort_values(["injected_noise_rate"])
    )
    summary_csv = output_dir / "cxr_toy_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    plot_path = output_dir / "cxr_toy_injected_vs_estimated.png"
    plot_main_curves(summary_df, plot_path)
    examples_path = output_dir / "cxr_toy_examples_5x2.png"
    save_toy_examples_grid(
        pool_df=pool_df,
        dataset_root=dataset_root,
        image_size=args.image_size,
        output_png=examples_path,
        overlay_rgb=overlay_rgb,
        seed=args.pool_seed,
    )

    report_path = output_dir / "cxr_toy_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CXR Toy Noise Validation Report\n")
        f.write("===============================\n\n")
        f.write(f"metadata_csv={metadata_csv}\n")
        f.write(f"dataset_root={dataset_root}\n")
        f.write(f"cache_csv={cache_csv}\n")
        f.write(f"pool_seed={args.pool_seed}\n")
        f.write(f"model_backbone={args.model_backbone}\n")
        f.write(f"n_samples={n_samples}\n")
        f.write(f"image_size={args.image_size}\n")
        f.write(f"n_seeds={args.n_seeds}\n")
        f.write(f"n_splits={args.n_splits}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"epochs={args.epochs}\n")
        f.write(f"early_stopping_patience={args.early_stopping_patience}\n")
        f.write(f"recover_best_weights={not args.no_recover_best_weights}\n")
        f.write(f"lr={args.lr}\n")
        f.write(f"noise_levels={noise_levels}\n")
        f.write(f"view_positions={view_positions}\n")
        f.write(f"overlay_color={args.overlay_color}\n")
        f.write(f"device={device}\n\n")

        x = summary_df["injected_noise_rate"].to_numpy()
        y = summary_df["estimated_noise_rate_mean"].to_numpy()
        if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
            corr = float(np.corrcoef(x, y)[0, 1])
            f.write(f"Pearson(injected, estimated_mean)={corr:.4f}\n")
        else:
            f.write("Pearson(injected, estimated_mean)=nan\n")

    print("\nSaved:")
    print(f"- Trials:  {trial_csv}")
    print(f"- Summary: {summary_csv}")
    print(f"- Plot:    {plot_path}")
    print(f"- Examples:{examples_path}")
    print(f"- Report:  {report_path}")


if __name__ == "__main__":
    main()
