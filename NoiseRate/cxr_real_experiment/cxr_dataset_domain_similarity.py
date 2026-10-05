#!/usr/bin/env python3
"""Compare VinDr-CXR and MIMIC-CXR image domains in frozen XRV feature spaces."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchxrayvision as xrv
from PIL import Image
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


WEIGHTS = (
    "densenet121-res224-all",
    "densenet121-res224-nih",
    "densenet121-res224-pc",
)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mimic_relpath(subject_id: int, study_id: int, dicom_id: str) -> str:
    return (
        f"p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
    )


def build_locked_index(args: argparse.Namespace) -> pd.DataFrame:
    vindr = pd.read_csv(args.vindr_index)
    required_vindr = {"image_id", "image_path"}
    if not required_vindr.issubset(vindr.columns):
        raise ValueError(f"VinDr index is missing {sorted(required_vindr - set(vindr.columns))}")
    vindr = vindr[["image_id", "image_path"]].copy()
    vindr["source"] = "VinDr-CXR"
    vindr["subject_id"] = ""
    vindr["study_id"] = ""
    vindr["view_position"] = "unknown"
    vindr["root"] = str(Path(args.vindr_image_root))
    if len(vindr) != args.samples or vindr["image_id"].nunique() != args.samples:
        raise ValueError(f"Expected {args.samples} unique VinDr images, found {len(vindr)}")

    split = pd.read_csv(
        args.mimic_split_csv,
        usecols=["dicom_id", "study_id", "subject_id", "split"],
    )
    split = split[split["split"].eq(args.mimic_split)].copy()
    metadata = pd.read_csv(
        args.mimic_metadata_csv,
        usecols=["dicom_id", "study_id", "subject_id", "ViewPosition"],
    )
    metadata["ViewPosition"] = metadata["ViewPosition"].astype(str).str.upper()
    mimic = split.merge(
        metadata,
        on=["dicom_id", "study_id", "subject_id"],
        how="inner",
        validate="one_to_one",
    )
    mimic = mimic[mimic["ViewPosition"].isin(["AP", "PA"])].copy()
    mimic["image_path"] = [
        mimic_relpath(int(subject), int(study), str(dicom))
        for subject, study, dicom in zip(
            mimic["subject_id"], mimic["study_id"], mimic["dicom_id"]
        )
    ]
    mimic_root = Path(args.mimic_image_root)
    mimic = mimic[mimic["image_path"].map(lambda value: (mimic_root / value).is_file())]

    # One image per subject avoids repeated-patient information in the domain classifier.
    rng = np.random.default_rng(args.seed)
    mimic["_order"] = rng.permutation(len(mimic))
    mimic = mimic.sort_values("_order").drop_duplicates("subject_id", keep="first")
    if len(mimic) < args.samples:
        raise ValueError(f"Only {len(mimic)} unique MIMIC subjects are available")
    selected = rng.choice(len(mimic), size=args.samples, replace=False)
    mimic = mimic.iloc[selected].copy().sort_values("image_path").reset_index(drop=True)
    mimic = mimic.rename(columns={"dicom_id": "image_id", "ViewPosition": "view_position"})
    mimic["source"] = "MIMIC-CXR"
    mimic["root"] = str(mimic_root)
    mimic = mimic[
        ["image_id", "image_path", "source", "subject_id", "study_id", "view_position", "root"]
    ]

    combined = pd.concat([vindr[mimic.columns], mimic], ignore_index=True)
    if combined.groupby("source").size().to_dict() != {
        "MIMIC-CXR": args.samples,
        "VinDr-CXR": args.samples,
    }:
        raise RuntimeError("Locked source counts are incorrect")
    if mimic["subject_id"].nunique() != args.samples:
        raise RuntimeError("MIMIC sampling is not subject-unique")
    missing = [
        str(Path(row.root) / row.image_path)
        for row in combined.itertuples(index=False)
        if not (Path(row.root) / row.image_path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Locked index contains {len(missing)} missing images")
    return combined


class LockedImageDataset(Dataset):
    def __init__(self, index: pd.DataFrame):
        self.index = index.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        row = self.index.iloc[item]
        with Image.open(Path(row["root"]) / row["image_path"]) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32)
        if array.shape != (224, 224):
            raise ValueError(f"Expected a 224x224 image, got {array.shape}")
        array = xrv.datasets.normalize(array[np.newaxis, :, :], maxval=255)
        return torch.from_numpy(array).float(), item


class FrozenXRV(nn.Module):
    def __init__(self, weights: str, cache_dir: str):
        super().__init__()
        self.model = xrv.models.DenseNet(weights=weights, cache_dir=cache_dir)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.features(images)
        return nn.functional.adaptive_avg_pool2d(features, (1, 1)).flatten(1)


def extract_features(
    index: pd.DataFrame,
    weights: str,
    cache_dir: str,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    loader = DataLoader(
        LockedImageDataset(index),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    model = FrozenXRV(weights, cache_dir).to(device)
    features = np.full((len(index), 1024), np.nan, dtype=np.float32)
    with torch.no_grad():
        for batch, (images, indices) in enumerate(loader, start=1):
            values = model(images.to(device, non_blocking=True)).cpu().numpy()
            features[indices.numpy()] = values.astype(np.float32)
            if batch % 20 == 0 or batch == len(loader):
                print(f"[{weights}] {int(np.isfinite(features[:, 0]).sum())}/{len(index)}", flush=True)
    if features.shape != (len(index), 1024) or not np.isfinite(features).all():
        raise RuntimeError(f"Incomplete features for {weights}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features


def bootstrap_auc(
    labels: np.ndarray, scores: np.ndarray, rng: np.random.Generator, draws: int
) -> tuple[float, float]:
    values = []
    by_class = [np.flatnonzero(labels == value) for value in [0, 1]]
    for _ in range(draws):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in by_class]
        )
        values.append(roc_auc_score(labels[indices], scores[indices]))
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def c2st(
    values: np.ndarray,
    labels: np.ndarray,
    seed: int,
    bootstrap_draws: int,
) -> dict[str, float]:
    probabilities = np.full(len(labels), np.nan, dtype=float)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_aucs = []
    for train, test in folds.split(values, labels):
        classifier = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        classifier.fit(values[train], labels[train])
        probabilities[test] = classifier.predict_proba(values[test])[:, 1]
        fold_aucs.append(roc_auc_score(labels[test], probabilities[test]))
    auc = float(roc_auc_score(labels, probabilities))
    low, high = bootstrap_auc(
        labels, probabilities, np.random.default_rng(seed + 101), bootstrap_draws
    )
    return {
        "auc": auc,
        "ci_low": float(low),
        "ci_high": float(high),
        "fold_mean": float(np.mean(fold_aucs)),
        "fold_sd": float(np.std(fold_aucs, ddof=1)),
    }


def internal_c2st_controls(values: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    output = {}
    for source_value, source_name in [(0, "vindr"), (1, "mimic")]:
        source = values[labels == source_value]
        random_labels = np.zeros(len(source), dtype=int)
        random_labels[rng.choice(len(source), size=len(source) // 2, replace=False)] = 1
        output[source_name] = c2st(source, random_labels, seed + source_value, 200)["auc"]
    return output


def rbf_mmd_test(
    values: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    seed: int,
    permutations: int,
    internal_repeats: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    pair_a = rng.integers(0, len(values), size=20_000)
    pair_b = rng.integers(0, len(values), size=20_000)
    squared = np.sum((values[pair_a] - values[pair_b]) ** 2, axis=1)
    squared = squared[squared > 0]
    median_squared = float(np.median(squared))
    gamma = 1.0 / (2.0 * median_squared)

    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    with torch.no_grad():
        distances = torch.cdist(tensor, tensor).square_()
        kernel = torch.exp(distances.mul_(-gamma))
        sign = torch.as_tensor(
            np.where(labels == 0, 1.0, -1.0).astype(np.float32),
            dtype=torch.float32,
            device=device,
        )
        n_group = int(np.sum(labels == 0))
        observed = float((sign @ kernel @ sign).item() / (n_group**2))

        exceed = 0
        batch_size = 50
        for start in range(0, permutations, batch_size):
            current = min(batch_size, permutations - start)
            signs = -np.ones((current, len(labels)), dtype=np.float32)
            for row in range(current):
                positive = rng.choice(len(labels), size=n_group, replace=False)
                signs[row, positive] = 1.0
            signs_t = torch.as_tensor(signs, device=device)
            null = torch.sum((signs_t @ kernel) * signs_t, dim=1) / (n_group**2)
            exceed += int(torch.sum(null >= observed).item())

        controls = {}
        for source_value, source_name in [(0, "vindr"), (1, "mimic")]:
            source_idx = np.flatnonzero(labels == source_value)
            source_kernel = kernel[source_idx][:, source_idx]
            half = len(source_idx) // 2
            values_internal = []
            for _ in range(internal_repeats):
                control_sign = -np.ones(len(source_idx), dtype=np.float32)
                control_sign[rng.choice(len(source_idx), size=half, replace=False)] = 1.0
                control_t = torch.as_tensor(control_sign, device=device)
                values_internal.append(
                    float((control_t @ source_kernel @ control_t).item() / (half**2))
                )
            controls[source_name] = float(np.mean(values_internal))
    internal_mean = float(np.mean(list(controls.values())))
    return {
        "mmd2": observed,
        "permutation_p": float((exceed + 1) / (permutations + 1)),
        "bandwidth_squared": median_squared,
        "within_vindr_mean": controls["vindr"],
        "within_mimic_mean": controls["mimic"],
        "cross_within_ratio": float(observed / internal_mean) if internal_mean > 0 else float("inf"),
    }


def covariance_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = eigh((matrix + matrix.T) / 2.0)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def frechet_distance(first: np.ndarray, second: np.ndarray) -> float:
    mean_first = first.mean(axis=0)
    mean_second = second.mean(axis=0)
    cov_first = np.cov(first, rowvar=False) + np.eye(first.shape[1]) * 1e-6
    cov_second = np.cov(second, rowvar=False) + np.eye(second.shape[1]) * 1e-6
    root_first = covariance_sqrt(cov_first)
    middle = root_first @ cov_second @ root_first
    root_middle = covariance_sqrt(middle)
    distance = (
        np.sum((mean_first - mean_second) ** 2)
        + np.trace(cov_first)
        + np.trace(cov_second)
        - 2.0 * np.trace(root_middle)
    )
    return float(max(distance, 0.0))


def frechet_reference(
    values: np.ndarray,
    labels: np.ndarray,
    seed: int,
    repeats: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    groups = [values[labels == value] for value in [0, 1]]
    half = min(len(group) for group in groups) // 2
    cross, within_vindr, within_mimic = [], [], []
    for _ in range(repeats):
        orders = [rng.permutation(len(group)) for group in groups]
        cross.append(
            frechet_distance(groups[0][orders[0][:half]], groups[1][orders[1][:half]])
        )
        within_vindr.append(
            frechet_distance(groups[0][orders[0][:half]], groups[0][orders[0][half : 2 * half]])
        )
        within_mimic.append(
            frechet_distance(groups[1][orders[1][:half]], groups[1][orders[1][half : 2 * half]])
        )
    within = float(np.mean(within_vindr + within_mimic))
    return {
        "full_cross": frechet_distance(groups[0], groups[1]),
        "matched_cross_mean": float(np.mean(cross)),
        "matched_cross_sd": float(np.std(cross, ddof=1)),
        "within_vindr_mean": float(np.mean(within_vindr)),
        "within_mimic_mean": float(np.mean(within_mimic)),
        "cross_within_ratio": float(np.mean(cross) / within),
    }


def analyze_features(
    features: np.ndarray,
    labels: np.ndarray,
    weights: str,
    output: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    standardized = StandardScaler().fit_transform(features)
    dimensions = min(args.pca_dimensions, standardized.shape[1], len(standardized) - 1)
    pca = PCA(n_components=dimensions, random_state=args.seed)
    values = pca.fit_transform(standardized).astype(np.float32)

    c2st_result = c2st(values, labels, args.seed, args.bootstrap_draws)
    c2st_result["internal_controls"] = internal_c2st_controls(values, labels, args.seed + 200)
    mmd_result = rbf_mmd_test(
        values,
        labels,
        device,
        args.seed + 300,
        args.mmd_permutations,
        args.internal_repeats,
    )
    frechet_result = frechet_reference(
        values, labels, args.seed + 400, args.internal_repeats
    )

    rng = np.random.default_rng(args.seed)
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    for value, name, color in [(0, "VinDr-CXR", "#0072B2"), (1, "MIMIC-CXR", "#D55E00")]:
        indices = np.flatnonzero(labels == value)
        indices = rng.choice(indices, size=min(1000, len(indices)), replace=False)
        axis.scatter(values[indices, 0], values[indices, 1], s=10, alpha=0.35, label=name, color=color)
    axis.set_xlabel("PCA component 1")
    axis.set_ylabel("PCA component 2")
    axis.set_title(f"Frozen XRV feature space: {weights}")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / f"{weights}_pca.png", dpi=200)
    plt.close(fig)

    return {
        "weights": weights,
        "samples_per_dataset": int(np.sum(labels == 0)),
        "raw_feature_dimensions": int(features.shape[1]),
        "pca_dimensions": int(dimensions),
        "pca_variance_explained": float(np.sum(pca.explained_variance_ratio_)),
        "c2st": c2st_result,
        "mmd": mmd_result,
        "frechet": frechet_result,
    }


def plot_summary(summaries: list[dict[str, object]], output: Path) -> None:
    short = [str(item["weights"]).replace("densenet121-res224-", "") for item in summaries]
    auc = [float(item["c2st"]["auc"]) for item in summaries]
    auc_low = [float(item["c2st"]["ci_low"]) for item in summaries]
    auc_high = [float(item["c2st"]["ci_high"]) for item in summaries]
    mmd_ratio = [float(item["mmd"]["cross_within_ratio"]) for item in summaries]
    frechet_ratio = [float(item["frechet"]["cross_within_ratio"]) for item in summaries]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    positions = np.arange(len(short))
    axes[0].bar(positions, auc, color="#0072B2")
    axes[0].errorbar(
        positions,
        auc,
        yerr=[np.asarray(auc) - np.asarray(auc_low), np.asarray(auc_high) - np.asarray(auc)],
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    axes[0].axhline(0.5, color="0.4", linestyle="--", linewidth=1)
    axes[0].set_ylim(0.45, 1.0)
    axes[0].set_title("Dataset classifier AUROC")
    axes[0].set_ylabel("Higher = more separable")

    axes[1].bar(positions, mmd_ratio, color="#009E73")
    axes[1].axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    axes[1].set_title("MMD cross / within")
    axes[1].set_ylabel("Higher = larger domain gap")

    axes[2].bar(positions, frechet_ratio, color="#D55E00")
    axes[2].axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    axes[2].set_title("Frechet cross / within")
    axes[2].set_ylabel("Higher = larger domain gap")
    for axis in axes:
        axis.set_xticks(positions, short)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "xrv_domain_similarity_summary.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vindr-index", type=Path, required=True)
    parser.add_argument("--vindr-image-root", type=Path, required=True)
    parser.add_argument("--mimic-split-csv", type=Path, required=True)
    parser.add_argument("--mimic-metadata-csv", type=Path, required=True)
    parser.add_argument("--mimic-image-root", type=Path, required=True)
    parser.add_argument("--mimic-split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xrv-cache-dir", required=True)
    parser.add_argument("--weights", nargs="+", default=list(WEIGHTS))
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pca-dimensions", type=int, default=128)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--mmd-permutations", type=int, default=1000)
    parser.add_argument("--internal-repeats", type=int, default=50)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.output_dir.exists():
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
        if (args.output_dir / ".complete").exists():
            raise FileExistsError(f"Completed output cannot be resumed: {args.output_dir}")
    else:
        args.output_dir.mkdir(parents=True)
    set_seed(args.seed)

    index = build_locked_index(args)
    locked_index_path = args.output_dir / "locked_image_index.csv"
    if locked_index_path.exists():
        previous_index = pd.read_csv(locked_index_path, dtype=str).fillna("")
        current_index = index.astype(str).fillna("")
        pd.testing.assert_frame_equal(previous_index, current_index, check_dtype=False)
        print("Reusing verified locked image index", flush=True)
    else:
        atomic_csv(index, locked_index_path)
    labels = index["source"].map({"VinDr-CXR": 0, "MIMIC-CXR": 1}).to_numpy(dtype=int)
    device = torch.device(args.device)

    summaries = []
    for weights in args.weights:
        if weights not in WEIGHTS:
            raise ValueError(f"Unapproved weights: {weights}")
        feature_path = args.output_dir / f"{weights}_features.npz"
        if feature_path.exists():
            with np.load(feature_path, allow_pickle=True) as payload:
                features = payload["features"]
                saved_source = payload["source"]
                needs_string_migration = bool(payload["image_id"].dtype.hasobject)
                saved_image_ids = payload["image_id"].astype(str)
            if features.shape != (len(index), 1024) or not np.isfinite(features).all():
                raise RuntimeError(f"Existing feature payload is invalid: {feature_path}")
            if not np.array_equal(saved_source, labels):
                raise RuntimeError(f"Existing feature source labels differ: {feature_path}")
            if not np.array_equal(
                saved_image_ids, index["image_id"].astype(str).to_numpy()
            ):
                raise RuntimeError(f"Existing feature image ids differ: {feature_path}")
            # Migrate early object-string payloads to pickle-free fixed Unicode storage.
            if needs_string_migration:
                atomic_npz(
                    feature_path,
                    features=features,
                    source=labels,
                    image_id=saved_image_ids.astype("U64"),
                )
            print(f"Reusing verified features: {weights}", flush=True)
        else:
            features = extract_features(
                index,
                weights,
                args.xrv_cache_dir,
                device,
                args.batch_size,
                args.workers,
            )
            atomic_npz(
                feature_path,
                features=features,
                source=labels,
                image_id=index["image_id"].astype(str).to_numpy(dtype="U64"),
            )
        summary = analyze_features(features, labels, weights, args.output_dir, args, device)
        summaries.append(summary)
        atomic_text(json.dumps(summary, indent=2), args.output_dir / f"{weights}_metrics.json")

    rows = []
    for item in summaries:
        rows.append(
            {
                "weights": item["weights"],
                "c2st_auc": item["c2st"]["auc"],
                "c2st_ci_low": item["c2st"]["ci_low"],
                "c2st_ci_high": item["c2st"]["ci_high"],
                "c2st_vindr_internal": item["c2st"]["internal_controls"]["vindr"],
                "c2st_mimic_internal": item["c2st"]["internal_controls"]["mimic"],
                "mmd2": item["mmd"]["mmd2"],
                "mmd_permutation_p": item["mmd"]["permutation_p"],
                "mmd_cross_within_ratio": item["mmd"]["cross_within_ratio"],
                "frechet_full_cross": item["frechet"]["full_cross"],
                "frechet_cross_within_ratio": item["frechet"]["cross_within_ratio"],
                "pca_variance_explained": item["pca_variance_explained"],
            }
        )
    atomic_csv(pd.DataFrame(rows), args.output_dir / "domain_similarity_summary.csv")
    plot_summary(summaries, args.output_dir)
    atomic_text(json.dumps(summaries, indent=2), args.output_dir / "all_metrics.json")
    atomic_text("complete\n", args.output_dir / ".complete")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
