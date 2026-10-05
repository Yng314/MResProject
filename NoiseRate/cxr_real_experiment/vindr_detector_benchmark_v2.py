#!/usr/bin/env python3
"""Blind multi-evidence detector benchmark for VinDr label corruption."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchxrayvision as xrv
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.utils.extmath import randomized_svd
from torch.utils.data import DataLoader, Dataset

from vindr_known_gt_cl_benchmark import (
    LABELS,
    atomic_save_npz,
    atomic_write_csv,
    atomic_write_text,
    exact_sign_flip_p,
    holm_adjust,
    require_columns,
    set_seed,
    sha256_file,
)


PROTOCOL_NAME = "vindr_detector_benchmark_v2"
XRV_WEIGHTS = "densenet121-res224-all"
XRV_LABEL_MAP = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Lung Opacity": "Lung Opacity",
    "Pleural effusion": "Effusion",
    "Pneumonia": "Pneumonia",
}
FORBIDDEN_BLIND_TOKENS = ("clean_label", "reference", "injected", "true_", "error")
METHOD_COLUMNS = {
    "oof_cl": "score_oof_cl",
    "active_label_cleaning": "score_active_label_cleaning",
    "fine_gmm": "score_fine_gmm",
    "external_xrv": "score_external_xrv",
}


def atomic_write_json(payload: dict, path: Path) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_int_list(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        result = [int(token.strip()) for token in value.split(",") if token.strip()]
    else:
        result = [int(item) for item in value]
    if not result or len(result) != len(set(result)):
        raise ValueError("Expected a non-empty list of unique integers")
    return result


def validate_no_outcomes(frame: pd.DataFrame, name: str) -> None:
    forbidden = [
        column
        for column in frame.columns
        if any(token in column.lower() for token in FORBIDDEN_BLIND_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"{name} contains private outcome columns: {forbidden}")


def stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


class IndexedPNGDataset(Dataset):
    def __init__(self, index: pd.DataFrame, image_root: Path):
        self.index = index.reset_index(drop=True)
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        path = self.image_root / str(self.index.iloc[item]["image_path"])
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32)
        if array.shape != (224, 224):
            raise ValueError(f"Expected 224x224 PNG, got {array.shape}: {path}")
        array = xrv.datasets.normalize(array[np.newaxis, :, :], maxval=255)
        return torch.from_numpy(array).float(), item


class FrozenXRVWithEvidence(nn.Module):
    def __init__(self, cache_dir: str):
        super().__init__()
        self.model = xrv.models.DenseNet(weights=XRV_WEIGHTS, cache_dir=cache_dir)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        missing = sorted(set(XRV_LABEL_MAP.values()) - set(self.model.pathologies))
        if missing:
            raise ValueError(f"XRV model is missing required findings: {missing}")
        self.output_indices = [self.model.pathologies.index(XRV_LABEL_MAP[label]) for label in LABELS]

    def train(self, mode: bool = True) -> "FrozenXRVWithEvidence":
        super().train(False)
        self.model.eval()
        return self

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model.features2(images)
        logits = self.model.classifier(features)
        probabilities = torch.sigmoid(logits)
        thresholds = self.model.op_threshs.to(probabilities.device).expand_as(probabilities)
        normalized = torch.full_like(probabilities, 0.5)
        finite = ~torch.isnan(thresholds)
        lower = (probabilities < thresholds) & finite
        upper = (~lower) & finite
        normalized[lower] = probabilities[lower] / (2.0 * thresholds[lower])
        normalized[upper] = 1.0 - (
            (1.0 - probabilities[upper]) / (2.0 * (1.0 - thresholds[upper]))
        )
        return features, normalized[:, self.output_indices]


def extract_external(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite external evidence: {output}")
    index_path = Path(args.image_index)
    index = pd.read_csv(index_path)
    require_columns(index, ["image_id", "image_path", "output_sha256"], "image index")
    index["image_id"] = index["image_id"].astype(str)
    if len(index) != 3_000 or index["image_id"].nunique() != 3_000:
        raise ValueError("External evidence requires 3,000 unique VinDr images")
    image_root = Path(args.image_root)
    if not index["image_path"].map(lambda value: (image_root / value).is_file()).all():
        raise FileNotFoundError("One or more indexed VinDr PNGs are missing")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        set_seed(args.seed)
        device = torch.device(args.device)
        loader = DataLoader(
            IndexedPNGDataset(index, image_root),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        model = FrozenXRVWithEvidence(args.xrv_cache_dir).to(device)
        features = np.full((len(index), 1_024), np.nan, dtype=np.float32)
        probabilities = np.full((len(index), len(LABELS)), np.nan, dtype=np.float32)
        with torch.no_grad():
            for batch, (images, indices) in enumerate(loader, start=1):
                latent, predicted = model(images.to(device, non_blocking=True))
                features[indices.numpy()] = latent.cpu().numpy().astype(np.float32)
                probabilities[indices.numpy()] = predicted.cpu().numpy().astype(np.float32)
                if batch % 10 == 0 or batch == len(loader):
                    complete = int(np.isfinite(features[:, 0]).sum())
                    print(f"[external-xrv] {complete}/{len(index)}", flush=True)
        if not np.isfinite(features).all() or not np.isfinite(probabilities).all():
            raise RuntimeError("External XRV evidence is incomplete or non-finite")
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise RuntimeError("External XRV probabilities are outside [0, 1]")

        payload = temporary / "xrv_external_evidence.npz"
        atomic_save_npz(
            payload,
            image_id=index["image_id"].to_numpy(dtype="U64"),
            label_name=np.asarray(LABELS, dtype="U32"),
            features=features,
            probabilities=probabilities,
        )
        summary = {
            "protocol": PROTOCOL_NAME,
            "outcome_blind": True,
            "images": int(len(index)),
            "features": list(features.shape),
            "probabilities": list(probabilities.shape),
            "encoder": XRV_WEIGHTS,
            "probability_normalization": "TorchXRayVision operating-point normalization",
            "label_map": XRV_LABEL_MAP,
            "image_index_sha256": sha256_file(index_path),
            "evidence_sha256": sha256_file(payload),
            "program_sha256": sha256_file(Path(__file__)),
        }
        atomic_write_json(summary, temporary / "external_evidence_summary.json")
        atomic_write_text(temporary / ".external_evidence_complete", "complete\n")
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_external(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    required = {"image_id", "label_name", "features", "probabilities"}
    if set(payload.files) != required:
        raise ValueError(f"Unexpected external evidence keys: {payload.files}")
    image_ids = payload["image_id"].astype(str)
    label_names = payload["label_name"].astype(str).tolist()
    features = payload["features"].astype(np.float32)
    probabilities = payload["probabilities"].astype(np.float32)
    if label_names != list(LABELS):
        raise ValueError("External evidence label order differs from the locked labels")
    if features.shape != (3_000, 1_024) or probabilities.shape != (3_000, len(LABELS)):
        raise ValueError("External evidence arrays have unexpected shapes")
    if len(set(image_ids)) != 3_000 or not np.isfinite(features).all():
        raise ValueError("External feature ids or values are invalid")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("External probabilities are invalid")
    return image_ids, features, probabilities


def fine_gmm_suspicion(
    features: np.ndarray,
    noisy_labels: np.ndarray,
    random_state: int = 0,
) -> np.ndarray:
    """Core FINE score: top singular-vector alignment followed by a two-component GMM."""
    values = np.asarray(features, dtype=np.float64)
    labels = np.asarray(noisy_labels, dtype=np.int64)
    if values.ndim != 2 or labels.ndim != 2 or len(values) != len(labels):
        raise ValueError("FINE expects N x D features and N x L binary labels")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("FINE labels must be binary")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(values).all():
        raise ValueError("FINE features contain a zero vector or non-finite value")
    normalized = values / norms
    suspicion = np.full(labels.shape, np.nan, dtype=np.float64)
    for label_index in range(labels.shape[1]):
        for observed_label in (0, 1):
            indices = np.flatnonzero(labels[:, label_index] == observed_label)
            if len(indices) < 20:
                raise ValueError(
                    f"FINE stratum is too small: label={label_index}, observed={observed_label}"
                )
            _, _, right = randomized_svd(
                normalized[indices],
                n_components=1,
                n_oversamples=8,
                n_iter=4,
                random_state=stable_seed(PROTOCOL_NAME, random_state, label_index, observed_label),
                flip_sign=True,
            )
            alignment = np.abs(normalized[indices] @ right[0])
            mixture = GaussianMixture(
                n_components=2,
                covariance_type="full",
                reg_covar=1e-6,
                n_init=5,
                max_iter=250,
                random_state=stable_seed("fine-gmm", random_state, label_index, observed_label),
            )
            posterior = mixture.fit(alignment.reshape(-1, 1)).predict_proba(
                alignment.reshape(-1, 1)
            )
            if not mixture.converged_:
                raise RuntimeError("FINE GMM did not converge")
            clean_component = int(np.argmax(mixture.means_.reshape(-1)))
            suspicion[indices, label_index] = 1.0 - posterior[:, clean_component]
    if not np.isfinite(suspicion).all() or np.any((suspicion < 0) | (suspicion > 1)):
        raise RuntimeError("FINE produced invalid suspicion scores")
    return suspicion.astype(np.float32)


def active_label_cleaning_score(noisy: np.ndarray, probability: np.ndarray) -> np.ndarray:
    y = np.asarray(noisy, dtype=np.int64)
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1.0 - 1e-7)
    assigned_nll = -(y * np.log(p) + (1 - y) * np.log(1.0 - p))
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return assigned_nll - entropy


def ordered_manifest(args: argparse.Namespace) -> pd.DataFrame:
    parent = Path(args.parent_root)
    manifest_path = parent / "blind_run_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    require_columns(
        manifest,
        [
            "array_index",
            "scenario_id",
            "regime",
            "noise_rate",
            "rate_percent",
            "seed",
            "prepared_relpath",
            "blind_run_relpath",
        ],
        "blind run manifest",
    )
    validate_no_outcomes(manifest, "blind run manifest")
    manifest = manifest.sort_values("array_index").reset_index(drop=True)
    if args.seeds:
        manifest = manifest[manifest["seed"].isin(parse_int_list(args.seeds))]
    if args.scenarios:
        requested = {item.strip() for item in args.scenarios.split(",") if item.strip()}
        manifest = manifest[manifest["scenario_id"].isin(requested)]
    if manifest.empty:
        raise ValueError("Selected blind manifest is empty")
    return manifest.reset_index(drop=True)


def align_evidence(
    evidence: pd.DataFrame,
    image_ids: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    require_columns(
        evidence,
        [
            "image_id",
            "fold_id",
            "label_index",
            "label_name",
            "noisy_label",
            "oof_probability",
            "cl_first_score",
        ],
        "entry evidence",
    )
    validate_no_outcomes(evidence, "entry evidence")
    evidence = evidence.copy()
    evidence["image_id"] = evidence["image_id"].astype(str)
    if len(evidence) != 18_000 or evidence[["image_id", "label_name"]].duplicated().any():
        raise ValueError("Expected 18,000 unique image-label evidence rows")
    image_order = {image_id: index for index, image_id in enumerate(image_ids)}
    label_order = {label: index for index, label in enumerate(LABELS)}
    if set(evidence["image_id"]) != set(image_ids) or set(evidence["label_name"]) != set(LABELS):
        raise ValueError("Entry evidence does not align with external evidence ids or labels")
    evidence["_image_order"] = evidence["image_id"].map(image_order)
    evidence["_label_order"] = evidence["label_name"].map(label_order)
    evidence = evidence.sort_values(["_image_order", "_label_order"]).reset_index(drop=True)
    expected_labels = np.tile(np.asarray(LABELS), len(image_ids))
    expected_images = np.repeat(image_ids, len(LABELS))
    if not np.array_equal(evidence["label_name"].to_numpy(), expected_labels):
        raise RuntimeError("Label ordering failed")
    if not np.array_equal(evidence["image_id"].to_numpy(), expected_images):
        raise RuntimeError("Image ordering failed")
    noisy = evidence["noisy_label"].to_numpy(np.int64).reshape(len(image_ids), len(LABELS))
    oof = evidence["oof_probability"].to_numpy(float).reshape(len(image_ids), len(LABELS))
    if not np.isin(noisy, [0, 1]).all() or not np.isfinite(oof).all():
        raise ValueError("Noisy labels or OOF probabilities are invalid")
    return evidence.drop(columns=["_image_order", "_label_order"]), noisy, oof


def score_one_run(
    evidence_path: Path,
    output_path: Path,
    image_ids: np.ndarray,
    features: np.ndarray,
    external_probabilities: np.ndarray,
) -> dict:
    evidence = pd.read_csv(evidence_path)
    ordered, noisy, oof = align_evidence(evidence, image_ids)
    fine = fine_gmm_suspicion(features, noisy)
    external = np.where(noisy == 1, 1.0 - external_probabilities, external_probabilities)
    output = ordered[["image_id", "fold_id", "label_index", "label_name", "noisy_label"]].copy()
    output["score_oof_cl"] = ordered["cl_first_score"].to_numpy(float)
    output["score_active_label_cleaning"] = active_label_cleaning_score(noisy, oof).reshape(-1)
    output["score_fine_gmm"] = fine.reshape(-1)
    output["score_external_xrv"] = external.reshape(-1)
    validate_no_outcomes(output, "blind detector scores")
    if not np.isfinite(output.filter(like="score_").to_numpy()).all():
        raise RuntimeError("Blind detector scores contain non-finite values")
    atomic_write_csv(output, output_path)
    return {
        "rows": int(len(output)),
        "entry_evidence_sha256": sha256_file(evidence_path),
        "score_sha256": sha256_file(output_path),
    }


def score(args: argparse.Namespace) -> None:
    parent = Path(args.parent_root)
    output = Path(args.output_root)
    external_path = Path(args.external_evidence)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite blind scoring root: {output}")
    if not (parent / ".prepare_complete").is_file():
        raise FileNotFoundError("Parent prepare marker is missing")
    if not external_path.parent.joinpath(".external_evidence_complete").is_file():
        raise FileNotFoundError("External evidence completion marker is missing")
    manifest = ordered_manifest(args)
    image_ids, features, external_probabilities = load_external(external_path)
    parent_index = pd.read_csv(parent / "image_index.csv")
    if parent_index["image_id"].astype(str).tolist() != image_ids.tolist():
        raise ValueError("Parent image index and external evidence are not identically ordered")

    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    records = []
    try:
        for row in manifest.itertuples(index=False):
            blind_run = parent / row.blind_run_relpath
            if not (blind_run / ".blind_run_complete").is_file():
                raise FileNotFoundError(f"Blind OOF run is incomplete: {blind_run}")
            relative = Path("blind_scores") / str(row.scenario_id) / f"seed_{int(row.seed)}.csv"
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            record = score_one_run(
                blind_run / "entry_evidence.csv",
                destination,
                image_ids,
                features,
                external_probabilities,
            )
            record.update(
                {
                    "array_index": int(row.array_index),
                    "scenario_id": str(row.scenario_id),
                    "regime": str(row.regime),
                    "noise_rate": float(row.noise_rate),
                    "rate_percent": int(row.rate_percent),
                    "seed": int(row.seed),
                    "prepared_relpath": str(row.prepared_relpath),
                    "blind_run_relpath": str(row.blind_run_relpath),
                    "score_relpath": str(relative),
                }
            )
            records.append(record)
            print(f"scored scenario={row.scenario_id} seed={int(row.seed)}", flush=True)
        score_manifest = pd.DataFrame(records).sort_values("array_index")
        atomic_write_csv(score_manifest, temporary / "score_manifest.csv")
        atomic_write_json(
            {
                "protocol": PROTOCOL_NAME,
                "outcome_blind": True,
                "runs": int(len(score_manifest)),
                "methods": list(METHOD_COLUMNS),
                "fine": "per-finding/per-observed-label top singular-vector alignment + 2-GMM",
                "external_model": XRV_WEIGHTS,
                "parent_manifest_sha256": sha256_file(parent / "blind_run_manifest.csv"),
                "external_evidence_sha256": sha256_file(external_path),
                "program_sha256": sha256_file(Path(__file__)),
            },
            temporary / "scoring_summary.json",
        )
        atomic_write_text(temporary / ".blind_scoring_complete", "complete\n")
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def merge_private(scores: pd.DataFrame, private_path: Path) -> pd.DataFrame:
    validate_no_outcomes(scores, "blind scores")
    private = pd.read_csv(private_path)
    require_columns(
        private,
        ["image_id", "label_name", "clean_label", "noisy_label", "injected_error", "flip_direction"],
        "private reference",
    )
    private["image_id"] = private["image_id"].astype(str)
    merged = scores.merge(
        private,
        on=["image_id", "label_name"],
        validate="one_to_one",
        suffixes=("_blind", "_private"),
    )
    if len(merged) != 18_000:
        raise ValueError("Blind scores and private reference do not align")
    if not np.array_equal(
        merged["noisy_label_blind"].to_numpy(), merged["noisy_label_private"].to_numpy()
    ):
        raise ValueError("Noisy labels differ between blind and private files")
    return merged


def stable_top(frame: pd.DataFrame, score_column: str, budget: int) -> np.ndarray:
    if budget <= 0 or budget > len(frame):
        raise ValueError(f"Invalid review budget {budget} for {len(frame)} candidates")
    ranked = frame.sort_values(
        [score_column, "image_id", "label_name"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    return ranked.index.to_numpy()[:budget]


def ranking_recall_auc(truth: np.ndarray, score: np.ndarray) -> float:
    positives = int(np.sum(truth))
    if positives <= 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    recall = np.concatenate([[0.0], np.cumsum(truth[order]) / positives])
    fraction = np.arange(len(truth) + 1, dtype=float) / len(truth)
    return float(np.trapezoid(recall, fraction))


def detector_metrics(
    frame: pd.DataFrame,
    method: str,
    score_column: str,
    budget: int,
    metadata: dict,
    scope: str,
) -> dict:
    truth = frame["injected_error"].to_numpy(dtype=bool)
    score = frame[score_column].to_numpy(float)
    true_errors = int(truth.sum())
    if true_errors <= 0:
        raise ValueError("Detector metrics require at least one true error")
    selected = stable_top(frame, score_column, budget)
    found = int(frame.loc[selected, "injected_error"].sum())
    prevalence = float(truth.mean())
    expected = float(budget * prevalence)
    return {
        **metadata,
        "scope": scope,
        "method": method,
        "candidate_entries": int(len(frame)),
        "true_errors": true_errors,
        "error_prevalence": prevalence,
        "reviewed_entries": int(budget),
        "review_fraction": float(budget / len(frame)),
        "errors_found": found,
        "precision": float(found / budget),
        "recall": float(found / true_errors),
        "random_expected_recall": float(budget / len(frame)),
        "enrichment_over_random": float(found / expected) if expected else float("nan"),
        "auprc": float(average_precision_score(truth, score)),
        "auroc": float(roc_auc_score(truth, score)),
        "review_recall_auc": ranking_recall_auc(truth, score),
    }


def evaluate_run(merged: pd.DataFrame, metadata: dict) -> list[dict]:
    records: list[dict] = []
    total_errors = int(merged["injected_error"].sum())
    if total_errors <= 0:
        return records
    for method, score_column in METHOD_COLUMNS.items():
        records.append(
            detector_metrics(
                merged,
                method,
                score_column,
                total_errors,
                metadata,
                "overall",
            )
        )
        for direction, observed_label in [("0_to_1", 1), ("1_to_0", 0)]:
            queue = merged[merged["noisy_label_blind"] == observed_label]
            direction_errors = int(queue["flip_direction"].eq(direction).sum())
            if direction_errors:
                direction_frame = queue.copy()
                direction_frame["injected_error"] = direction_frame["flip_direction"].eq(direction).astype(int)
                records.append(
                    detector_metrics(
                        direction_frame,
                        method,
                        score_column,
                        direction_errors,
                        metadata,
                        direction,
                    )
                )
    return records


def selected_overlaps(merged: pd.DataFrame, metadata: dict) -> list[dict]:
    budget = int(merged["injected_error"].sum())
    if budget <= 0:
        return []
    selected = {
        method: set(stable_top(merged, column, budget).tolist())
        for method, column in METHOD_COLUMNS.items()
    }
    records = []
    reference = selected["oof_cl"]
    for method, indices in selected.items():
        if method == "oof_cl":
            continue
        intersection = len(reference & indices)
        union = len(reference | indices)
        records.append(
            {
                **metadata,
                "method": method,
                "reference_method": "oof_cl",
                "budget": budget,
                "intersection": intersection,
                "overlap_fraction_of_budget": intersection / budget,
                "jaccard": intersection / union,
            }
        )
    return records


def paired_contrasts(metrics: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (regime, rate_percent, scope, metric), frame in (
        metrics.melt(
            id_vars=["regime", "rate_percent", "seed", "scope", "method"],
            value_vars=["recall", "auprc", "review_recall_auc"],
            var_name="metric",
            value_name="value",
        ).groupby(["regime", "rate_percent", "scope", "metric"], sort=True)
    ):
        pivot = frame.pivot(index="seed", columns="method", values="value")
        if "oof_cl" not in pivot:
            continue
        for method in sorted(set(pivot.columns) - {"oof_cl"}):
            paired = pivot[[method, "oof_cl"]].dropna()
            differences = (paired[method] - paired["oof_cl"]).to_numpy(float)
            if not len(differences):
                continue
            records.append(
                {
                    "regime": regime,
                    "rate_percent": int(rate_percent),
                    "scope": scope,
                    "metric": metric,
                    "method": method,
                    "reference_method": "oof_cl",
                    "paired_seeds": int(len(differences)),
                    "mean_difference": float(differences.mean()),
                    "positive_seeds": int(np.sum(differences > 0)),
                    "negative_seeds": int(np.sum(differences < 0)),
                    "exact_sign_flip_p": exact_sign_flip_p(differences),
                }
            )
    result = pd.DataFrame(records)
    if result.empty:
        return result
    result["holm_p_within_scope_metric"] = np.nan
    for _, indices in result.groupby(["scope", "metric"]).groups.items():
        result.loc[indices, "holm_p_within_scope_metric"] = holm_adjust(
            result.loc[indices, "exact_sign_flip_p"].to_numpy(float)
        )
    return result


def plot_results(metrics: pd.DataFrame, output: Path) -> None:
    scopes = ["overall", "0_to_1", "1_to_0"]
    titles = ["Overall error recall", "Observed-positive queue: 0 to 1", "Observed-negative queue: 1 to 0"]
    colors = {
        "oof_cl": "#2563eb",
        "active_label_cleaning": "#7c3aed",
        "fine_gmm": "#059669",
        "external_xrv": "#dc2626",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for axis, scope, title in zip(axes, scopes, titles):
        selected = metrics[metrics["scope"] == scope]
        for method in METHOD_COLUMNS:
            frame = selected[selected["method"] == method]
            summary = frame.groupby(["regime", "rate_percent"])["recall"].agg(["mean", "std"]).reset_index()
            for regime, regime_frame in summary.groupby("regime"):
                axis.errorbar(
                    regime_frame["rate_percent"],
                    regime_frame["mean"],
                    yerr=regime_frame["std"].fillna(0),
                    marker={"balanced": "o", "fp_only": "s", "fn_only": "^"}.get(regime, "o"),
                    linestyle={"balanced": "-", "fp_only": "--", "fn_only": ":"}.get(regime, "-"),
                    color=colors[method],
                    linewidth=1.5,
                    capsize=2,
                    label=f"{method} / {regime}",
                )
        axis.set_title(title)
        axis.set_xlabel("Injected noise rate (%)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Recall at matched review budget")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.19, 1, 1))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> None:
    parent = Path(args.parent_root)
    score_root = Path(args.output_root)
    evaluation = score_root / "evaluation"
    if not (score_root / ".blind_scoring_complete").is_file():
        raise FileNotFoundError("Blind scoring marker is missing")
    if evaluation.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {evaluation}")
    manifest = pd.read_csv(score_root / "score_manifest.csv")
    metrics_records = []
    overlap_records = []
    for row in manifest.itertuples(index=False):
        scores_path = score_root / row.score_relpath
        if sha256_file(scores_path) != row.score_sha256:
            raise RuntimeError(f"Blind score hash mismatch: {scores_path}")
        scores = pd.read_csv(scores_path)
        merged = merge_private(scores, parent / row.prepared_relpath / "private_reference.csv")
        metadata = {
            "array_index": int(row.array_index),
            "scenario_id": str(row.scenario_id),
            "regime": str(row.regime),
            "noise_rate": float(row.noise_rate),
            "rate_percent": int(row.rate_percent),
            "seed": int(row.seed),
        }
        metrics_records.extend(evaluate_run(merged, metadata))
        overlap_records.extend(selected_overlaps(merged, metadata))
        print(f"evaluated scenario={row.scenario_id} seed={int(row.seed)}", flush=True)
    metrics = pd.DataFrame(metrics_records)
    overlaps = pd.DataFrame(overlap_records)
    if metrics.empty:
        raise RuntimeError("Private evaluation produced no noisy-run metrics")
    contrasts = paired_contrasts(metrics)
    evaluation.mkdir(parents=True)
    atomic_write_csv(metrics, evaluation / "detector_metrics.csv")
    atomic_write_csv(overlaps, evaluation / "selected_set_overlap.csv")
    atomic_write_csv(contrasts, evaluation / "paired_method_contrasts.csv")
    plot_results(metrics, evaluation / "detector_v2_recall.png")
    endpoint = metrics[(metrics["scope"] == "overall") & (metrics["rate_percent"] == 20)]
    summary = (
        endpoint.groupby(["regime", "method"])["recall"]
        .agg(["mean", "std"])
        .reset_index()
        .to_dict("records")
    )
    atomic_write_json(
        {
            "protocol": PROTOCOL_NAME,
            "private_evaluation": True,
            "scored_runs": int(len(manifest)),
            "noisy_runs": int(metrics[["scenario_id", "seed"]].drop_duplicates().shape[0]),
            "methods": list(METHOD_COLUMNS),
            "primary_endpoint": "overall recall at review budget equal to injected-error count",
            "direction_guardrail": "1_to_0 recall within the observed-negative queue at matched direction-error budget",
            "rate20_overall_summary": summary,
            "program_sha256": sha256_file(Path(__file__)),
            "scoring_summary_sha256": sha256_file(score_root / "scoring_summary.json"),
        },
        evaluation / "evaluation_summary.json",
    )
    atomic_write_text(evaluation / ".evaluation_complete", "complete\n")


def verify(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    evaluation = root / "evaluation"
    required = [
        root / ".blind_scoring_complete",
        root / "score_manifest.csv",
        root / "scoring_summary.json",
        evaluation / ".evaluation_complete",
        evaluation / "detector_metrics.csv",
        evaluation / "selected_set_overlap.csv",
        evaluation / "paired_method_contrasts.csv",
        evaluation / "detector_v2_recall.png",
        evaluation / "evaluation_summary.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = pd.read_csv(root / "score_manifest.csv")
    expected_runs = args.expected_runs if args.expected_runs is not None else len(manifest)
    if len(manifest) != expected_runs or manifest[["scenario_id", "seed"]].duplicated().any():
        raise RuntimeError("Score manifest run count or uniqueness failed")
    for row in manifest.itertuples(index=False):
        path = root / row.score_relpath
        if not path.is_file() or sha256_file(path) != row.score_sha256:
            raise RuntimeError(f"Blind score artifact failed verification: {path}")
        scores = pd.read_csv(path)
        validate_no_outcomes(scores, "verified blind score")
        if len(scores) != 18_000 or not set(METHOD_COLUMNS.values()).issubset(scores.columns):
            raise RuntimeError(f"Blind score schema or row count failed: {path}")
    metrics = pd.read_csv(evaluation / "detector_metrics.csv")
    noisy_runs = manifest[manifest["regime"] != "clean"]
    expected_metric_rows = len(noisy_runs) * len(METHOD_COLUMNS)
    expected_metric_rows += int(
        sum(
            1
            for row in noisy_runs.itertuples(index=False)
            for _method in METHOD_COLUMNS
            for direction in ("0_to_1", "1_to_0")
            if direction == "0_to_1" and row.regime in {"balanced", "fp_only"}
            or direction == "1_to_0" and row.regime in {"balanced", "fn_only"}
        )
    )
    if len(metrics) != expected_metric_rows:
        raise RuntimeError(
            f"Expected {expected_metric_rows} metric rows, found {len(metrics)}"
        )
    if not np.isfinite(metrics[["recall", "precision", "auprc", "review_recall_auc"]].to_numpy()).all():
        raise RuntimeError("Evaluation metrics contain non-finite primary values")
    atomic_write_text(root / ".benchmark_verified", "verified\n")
    print(
        json.dumps(
            {
                "verified": True,
                "scored_runs": int(len(manifest)),
                "noisy_runs": int(len(noisy_runs)),
                "metric_rows": int(len(metrics)),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract-external")
    extract_parser.add_argument("--image-index", type=Path, required=True)
    extract_parser.add_argument("--image-root", type=Path, required=True)
    extract_parser.add_argument("--output-dir", type=Path, required=True)
    extract_parser.add_argument("--xrv-cache-dir", required=True)
    extract_parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    extract_parser.add_argument("--batch-size", type=int, default=32)
    extract_parser.add_argument("--num-workers", type=int, default=4)
    extract_parser.add_argument("--seed", type=int, default=13)
    extract_parser.set_defaults(function=extract_external)

    for command, function in [("score", score), ("evaluate", evaluate), ("verify", verify)]:
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--parent-root", type=Path, required=True)
        command_parser.add_argument("--output-root", type=Path, required=True)
        if command == "score":
            command_parser.add_argument("--external-evidence", type=Path, required=True)
            command_parser.add_argument("--seeds", default="")
            command_parser.add_argument("--scenarios", default="")
        if command == "verify":
            command_parser.add_argument("--expected-runs", type=int)
        command_parser.set_defaults(function=function)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
