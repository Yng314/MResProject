#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from cxr_real_full_train_eval_cleanlab_xrv12 import (
    LABEL_NAMES,
    RealCXRXRV12Dataset,
    aggregate_test_predictions_by_study,
    create_model,
)
from cxr_real_noise_validation_smoke import (
    filter_existing_rows,
    load_real_pool,
    project_raw_labels_to_binary,
)


PROTOCOL_NAME = "medpalm_cl_detection_v1"
EXPECTED_IMAGE_ROWS = 3414
EXPECTED_STUDIES = 3050
EXPECTED_BENCHMARK_ENTRIES = 498
EXPECTED_BENCHMARK_ISSUES = 127
EXPECTED_BENCHMARK_CORRECT = 371
DEFAULT_SEEDS = [13, 42, 97, 123]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Held-out Med-PaLM confident-learning detection benchmark."
    )
    parser.add_argument("mode", choices=["infer", "score", "evaluate"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--chexpert-csv", type=Path)
    parser.add_argument("--split-csv", type=Path)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Frozen checkpoint as SEED=PATH. Repeat once per seed.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--benchmark-blinded", type=Path)
    parser.add_argument("--benchmark-reference", type=Path)
    parser.add_argument("--llm-evaluation", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--synthetic-replicates", type=int, default=100)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_path(path: Path | None, name: str) -> Path:
    if path is None:
        raise ValueError(f"{name} is required")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_checkpoints(items: list[str]) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Checkpoint must be SEED=PATH: {item}")
        seed_text, path_text = item.split("=", 1)
        seed = int(seed_text)
        path = Path(path_text)
        if seed in checkpoints:
            raise ValueError(f"Duplicate checkpoint seed: {seed}")
        if not path.exists():
            raise FileNotFoundError(path)
        checkpoints[seed] = path
    if sorted(checkpoints) != DEFAULT_SEEDS:
        raise ValueError(f"Expected checkpoints for {DEFAULT_SEEDS}, found {sorted(checkpoints)}")
    return checkpoints


def build_official_test_pool(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    image_root = require_path(args.image_root, "--image-root")
    rows = load_real_pool(
        chexpert_csv=require_path(args.chexpert_csv, "--chexpert-csv"),
        split_csv=require_path(args.split_csv, "--split-csv"),
        split_name="test",
        metadata_csv=require_path(args.metadata_csv, "--metadata-csv"),
        allowed_views=["AP", "PA"],
    )
    rows = filter_existing_rows(rows, image_root=image_root)
    if len(rows) != EXPECTED_IMAGE_ROWS:
        raise ValueError(f"Expected {EXPECTED_IMAGE_ROWS} AP/PA images, found {len(rows)}")
    if rows["study_id"].nunique() != EXPECTED_STUDIES:
        raise ValueError(
            f"Expected {EXPECTED_STUDIES} studies, found {rows['study_id'].nunique()}"
        )
    raw, binary, valid = project_raw_labels_to_binary(rows, LABEL_NAMES)
    return rows, raw, binary, valid


def infer(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "full_test_study_predictions.csv"
    marker = args.output_dir / ".inference_complete"
    if marker.exists():
        raise ValueError(f"Refusing to overwrite completed inference: {args.output_dir}")

    checkpoints = parse_checkpoints(args.checkpoint)
    set_seed(20260803)
    rows, raw, binary, valid = build_official_test_pool(args)
    dataset = RealCXRXRV12Dataset(
        rows=rows,
        image_root=require_path(args.image_root, "--image-root"),
        image_size=224,
        binary_labels=binary,
        valid_mask=valid,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    models: dict[int, torch.nn.Module] = {}
    checkpoint_metadata: dict[str, Any] = {}
    for seed, path in sorted(checkpoints.items()):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("label_names") != LABEL_NAMES:
            raise ValueError(f"Checkpoint label order differs for seed {seed}")
        checkpoint_args = payload.get("args", {})
        if checkpoint_args.get("model_backbone") != "mobilenet_v3_small_scratch":
            raise ValueError(f"Seed {seed} is not a MobileNetV3-small scratch checkpoint")
        if checkpoint_args.get("exclude_sample_csv") or checkpoint_args.get("exclude_entry_csv"):
            raise ValueError(f"Seed {seed} checkpoint is not an uncleaned baseline")
        model = create_model(
            model_backbone="mobilenet_v3_small_scratch",
            xrv_weights="densenet121-res224-all",
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(device).eval()
        models[seed] = model
        checkpoint_metadata[str(seed)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "training_seed": int(checkpoint_args.get("seed", seed)),
            "epochs": int(checkpoint_args.get("epochs", -1)),
            "model_backbone": checkpoint_args.get("model_backbone"),
            "study_aggregation": checkpoint_args.get("study_aggregation"),
        }

    predictions: dict[int, list[np.ndarray]] = {seed: [] for seed in models}
    with torch.no_grad():
        for batch_index, (images, _, _) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            for seed, model in models.items():
                predictions[seed].append(torch.sigmoid(model(images)).cpu().numpy())
            if batch_index % 20 == 0 or batch_index == len(loader):
                print(f"[inference] batch {batch_index}/{len(loader)}")

    study_output: pd.DataFrame | None = None
    for seed in sorted(predictions):
        image_probs = np.vstack(predictions[seed])
        if image_probs.shape != (len(rows), len(LABEL_NAMES)):
            raise ValueError(f"Unexpected prediction shape for seed {seed}: {image_probs.shape}")
        study_df, study_raw, study_binary, study_valid, study_prob = (
            aggregate_test_predictions_by_study(
                rows=rows,
                raw_labels=raw,
                y_binary=binary,
                valid_mask=valid,
                y_prob=image_probs,
                agg_method="max",
            )
        )
        if study_output is None:
            study_output = study_df[["study_id", "subject_id", "n_images_in_study"]].copy()
            for label_index, label in enumerate(LABEL_NAMES):
                study_output[f"raw::{label}"] = study_raw[:, label_index]
                study_output[f"binary::{label}"] = study_binary[:, label_index]
                study_output[f"valid::{label}"] = study_valid[:, label_index].astype(int)
        else:
            if not study_output["study_id"].eq(study_df["study_id"]).all():
                raise ValueError("Study ordering differs between seed predictions")
        for label_index, label in enumerate(LABEL_NAMES):
            study_output[f"prob::seed_{seed}::{label}"] = study_prob[:, label_index]

    assert study_output is not None
    if len(study_output) != EXPECTED_STUDIES:
        raise ValueError("Study aggregation produced an unexpected row count")
    study_output.to_csv(output_path, index=False)
    manifest = {
        "protocol": PROTOCOL_NAME,
        "phase": "inference",
        "outcome_blind": True,
        "image_rows": int(len(rows)),
        "studies": int(len(study_output)),
        "subjects": int(study_output["subject_id"].nunique()),
        "seeds": sorted(checkpoints),
        "aggregation": "max",
        "checkpoints": checkpoint_metadata,
        "prediction_sha256": sha256_file(output_path),
        "expert_reference_used": False,
    }
    write_json(args.output_dir / "inference_manifest.json", manifest)
    marker.write_text("complete\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def percentile_high(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True, ascending=True).to_numpy()


def score_one_estimator(
    frame: pd.DataFrame,
    estimator: str,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for label_index, label in enumerate(LABEL_NAMES):
        valid = frame[f"valid::{label}"].eq(1).to_numpy()
        labels = frame.loc[valid, f"binary::{label}"].to_numpy(dtype=int)
        probs = probabilities[valid, label_index]
        if len(labels) < 10 or len(np.unique(labels)) < 2:
            print(
                f"[score:{estimator}] skipping {label}: "
                f"n={len(labels)}, classes={np.unique(labels).tolist()}"
            )
            continue
        probs_2c = np.column_stack([1.0 - probs, probs])
        quality_self = get_label_quality_scores(
            labels=labels, pred_probs=probs_2c, method="self_confidence"
        )
        quality_margin = get_label_quality_scores(
            labels=labels, pred_probs=probs_2c, method="normalized_margin"
        )
        hard_issue = find_label_issues(labels=labels, pred_probs=probs_2c).astype(int)
        suspicion_self = 1.0 - quality_self
        suspicion_margin = 1.0 - quality_margin
        selected = frame.loc[valid, ["subject_id", "study_id", f"raw::{label}"]].copy()
        selected = selected.rename(columns={f"raw::{label}": "current_raw_label"})
        selected["entry_key"] = (
            selected["subject_id"].astype(str)
            + "::"
            + selected["study_id"].astype(str)
            + "::"
            + label.replace(" ", "_")
        )
        selected["label_name"] = label
        selected["label_index"] = label_index
        selected["current_binary_label"] = labels
        selected["pred_probability"] = probs
        selected["quality_self_confidence"] = quality_self
        selected["quality_normalized_margin"] = quality_margin
        selected["suspicion_self_confidence"] = suspicion_self
        selected["suspicion_normalized_margin"] = suspicion_margin
        selected["suspicion_percentile_self"] = percentile_high(suspicion_self)
        selected["suspicion_percentile_margin"] = percentile_high(suspicion_margin)
        selected["cl_hard_issue"] = hard_issue
        selected["estimator"] = estimator
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def score(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "full_test_study_predictions.csv"
    inference_manifest_path = args.output_dir / "inference_manifest.json"
    if not (args.output_dir / ".inference_complete").exists():
        raise FileNotFoundError("Inference completion marker is missing")
    if not prediction_path.exists() or not inference_manifest_path.exists():
        raise FileNotFoundError("Inference outputs are incomplete")
    inference_manifest = json.loads(inference_manifest_path.read_text(encoding="utf-8"))
    if sha256_file(prediction_path) != inference_manifest["prediction_sha256"]:
        raise ValueError("Frozen prediction file hash mismatch")
    if inference_manifest.get("expert_reference_used") is not False:
        raise ValueError("Inference manifest does not certify outcome blinding")

    frame = pd.read_csv(prediction_path)
    seed_probabilities: dict[int, np.ndarray] = {}
    for seed in DEFAULT_SEEDS:
        columns = [f"prob::seed_{seed}::{label}" for label in LABEL_NAMES]
        seed_probabilities[seed] = frame[columns].to_numpy(dtype=float)
    ensemble = np.mean(np.stack(list(seed_probabilities.values()), axis=0), axis=0)

    scored = []
    for seed, probabilities in seed_probabilities.items():
        scored.append(score_one_estimator(frame, f"seed_{seed}", probabilities))
    scored.append(score_one_estimator(frame, "ensemble", ensemble))
    score_frame = pd.concat(scored, ignore_index=True)
    score_path = args.output_dir / "full_test_cl_entry_scores.csv"
    score_frame.to_csv(score_path, index=False)
    manifest = {
        "protocol": PROTOCOL_NAME,
        "phase": "cl_scoring",
        "outcome_blind": True,
        "estimators": [f"seed_{seed}" for seed in DEFAULT_SEEDS] + ["ensemble"],
        "full_test_valid_entry_rows": int(score_frame["entry_key"].nunique()),
        "score_rows": int(len(score_frame)),
        "prediction_sha256": sha256_file(prediction_path),
        "score_sha256": sha256_file(score_path),
        "primary_score": "suspicion_percentile_self",
        "expert_reference_used": False,
    }
    write_json(args.output_dir / "scoring_manifest.json", manifest)
    (args.output_dir / ".scores_complete").write_text("complete\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def detection_metrics(
    frame: pd.DataFrame,
    score_column: str,
    cohort: str,
    estimator: str,
    method: str,
) -> dict[str, Any]:
    y = frame["true_issue"].astype(int).to_numpy()
    scores = frame[score_column].to_numpy(dtype=float)
    n = len(frame)
    issues = int(y.sum())
    correct = int(n - issues)
    prevalence = safe_div(issues, n)
    auprc = float(average_precision_score(y, scores)) if issues and correct else float("nan")
    auroc = float(roc_auc_score(y, scores)) if issues and correct else float("nan")
    top_n = max(1, int(round(n * 0.20)))
    ranked = frame.assign(_score=scores).sort_values(
        ["_score", "entry_key"], ascending=[False, True], kind="stable"
    )
    top = ranked.head(top_n)
    top_tp = int(top["true_issue"].sum())
    top_precision = safe_div(top_tp, top_n)
    top_recall = safe_div(top_tp, issues)
    top_enrichment = safe_div(top_precision, prevalence)
    hard = frame["cl_hard_issue"].eq(1)
    hard_n = int(hard.sum())
    hard_tp = int(frame.loc[hard, "true_issue"].sum())
    hard_precision = safe_div(hard_tp, hard_n)
    hard_recall = safe_div(hard_tp, issues)
    hard_enrichment = safe_div(hard_precision, prevalence)
    return {
        "cohort": cohort,
        "estimator": estimator,
        "method": method,
        "n_entries": n,
        "n_subjects": int(frame["subject_id"].nunique()),
        "true_issues": issues,
        "expert_correct": correct,
        "issue_prevalence": prevalence,
        "auprc": auprc,
        "auprc_minus_prevalence": auprc - prevalence if np.isfinite(auprc) else float("nan"),
        "auroc": auroc,
        "top20_n": top_n,
        "top20_true_issues": top_tp,
        "top20_precision": top_precision,
        "top20_recall": top_recall,
        "top20_enrichment": top_enrichment,
        "hard_flag_n": hard_n,
        "hard_flag_true_issues": hard_tp,
        "hard_flag_precision": hard_precision,
        "hard_flag_recall": hard_recall,
        "hard_flag_enrichment": hard_enrichment,
    }


def bootstrap_primary(frame: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    metric_names = [
        "auprc",
        "auprc_minus_prevalence",
        "auroc",
        "top20_precision",
        "top20_recall",
        "top20_enrichment",
        "hard_flag_precision",
        "hard_flag_recall",
        "hard_flag_enrichment",
    ]
    subjects = frame["subject_id"].drop_duplicates().to_numpy()
    groups = {subject: group for subject, group in frame.groupby("subject_id", sort=False)}
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in metric_names}
    for _ in range(replicates):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        replicate = pd.concat([groups[subject] for subject in sampled], ignore_index=True)
        row = detection_metrics(
            replicate,
            score_column="suspicion_percentile_self",
            cohort="bootstrap",
            estimator="ensemble",
            method="self_percentile",
        )
        for metric in metric_names:
            value = row[metric]
            if np.isfinite(value):
                values[metric].append(float(value))
    point = detection_metrics(
        frame,
        score_column="suspicion_percentile_self",
        cohort="all_entries",
        estimator="ensemble",
        method="self_percentile",
    )
    output = []
    for metric in metric_names:
        distribution = np.asarray(values[metric], dtype=float)
        if len(distribution):
            ci_lower = float(np.quantile(distribution, 0.025))
            ci_upper = float(np.quantile(distribution, 0.975))
        else:
            ci_lower = float("nan")
            ci_upper = float("nan")
        output.append(
            {
                "metric": metric,
                "point_estimate": point[metric],
                "ci_lower_95": ci_lower,
                "ci_upper_95": ci_upper,
                "valid_replicates": int(len(distribution)),
                "cluster": "subject_id",
            }
        )
    return pd.DataFrame(output)


def synthetic_positive_control(
    benchmark: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = benchmark[benchmark["true_issue"].eq(0)].copy().reset_index(drop=True)
    if len(base) != EXPECTED_BENCHMARK_CORRECT:
        raise ValueError("Synthetic positive control requires 371 expert-correct entries")
    rng = np.random.default_rng(seed)
    rows = []
    for rate in [0.05, 0.10, 0.20]:
        n_flip = max(1, int(round(len(base) * rate)))
        for replicate in range(replicates):
            flipped_indices = rng.choice(len(base), size=n_flip, replace=False)
            injected = np.zeros(len(base), dtype=int)
            injected[flipped_indices] = 1
            corrupted = base["current_binary_label"].to_numpy(dtype=int).copy()
            corrupted[flipped_indices] = 1 - corrupted[flipped_indices]
            probs = base["pred_probability"].to_numpy(dtype=float)
            quality = np.where(corrupted == 1, probs, 1.0 - probs)
            working = base[["entry_key", "label_name"]].copy()
            working["true_issue"] = injected
            working["score"] = 1.0 - quality
            working["score_percentile"] = working.groupby("label_name")["score"].rank(
                method="average", pct=True, ascending=True
            )
            ap = float(average_precision_score(injected, working["score_percentile"]))
            prevalence = float(injected.mean())
            top_n = max(1, int(round(len(working) * 0.20)))
            top = working.sort_values(
                ["score_percentile", "entry_key"], ascending=[False, True], kind="stable"
            ).head(top_n)
            precision = float(top["true_issue"].mean())
            rows.append(
                {
                    "flip_rate": rate,
                    "replicate": replicate,
                    "n_entries": len(working),
                    "n_injected_flips": n_flip,
                    "prevalence": prevalence,
                    "auprc": ap,
                    "auprc_minus_prevalence": ap - prevalence,
                    "top20_precision": precision,
                    "top20_enrichment": precision / prevalence,
                }
            )
    replicate_frame = pd.DataFrame(rows)
    summary = (
        replicate_frame.groupby("flip_rate", as_index=False)
        .agg(
            replicates=("replicate", "count"),
            n_injected_flips=("n_injected_flips", "first"),
            prevalence=("prevalence", "mean"),
            auprc_mean=("auprc", "mean"),
            auprc_ci_lower=("auprc", lambda x: np.quantile(x, 0.025)),
            auprc_ci_upper=("auprc", lambda x: np.quantile(x, 0.975)),
            top20_enrichment_mean=("top20_enrichment", "mean"),
            top20_enrichment_ci_lower=("top20_enrichment", lambda x: np.quantile(x, 0.025)),
            top20_enrichment_ci_upper=("top20_enrichment", lambda x: np.quantile(x, 0.975)),
        )
    )
    return replicate_frame, summary


def end_to_end_secondary(
    benchmark: pd.DataFrame,
    llm_evaluation_path: Path,
) -> pd.DataFrame:
    llm = pd.read_csv(llm_evaluation_path, usecols=["entry_key", "actual_action"])
    if llm["entry_key"].duplicated().any():
        raise ValueError("LLM evaluation contains duplicate entry keys")
    working = benchmark.merge(llm, on="entry_key", how="left", validate="one_to_one")
    if working["actual_action"].isna().any():
        raise ValueError("Missing prior LLM action for one or more benchmark entries")
    top_n = max(1, int(round(len(working) * 0.20)))
    ranked = working.sort_values(
        ["suspicion_percentile_self", "entry_key"],
        ascending=[False, True],
        kind="stable",
    )
    selected_keys = set(ranked.head(top_n)["entry_key"])
    selected = working["entry_key"].isin(selected_keys)
    post = working["current_binary_label"].astype(float).copy()
    relabel = selected & working["actual_action"].eq("relabel")
    mask = selected & working["actual_action"].eq("mask")
    post.loc[relabel] = 1.0 - post.loc[relabel]
    post.loc[mask] = np.nan
    retained = post.notna()
    post_correct = retained & post.eq(working["reference_binary_label"])
    baseline_correct = working["current_binary_label"].eq(working["reference_binary_label"])
    return pd.DataFrame(
        [
            {
                "policy": "ensemble_cl_top20_then_existing_llm_action",
                "n_entries": len(working),
                "selected_entries": int(selected.sum()),
                "selected_true_issues": int(working.loc[selected, "true_issue"].sum()),
                "baseline_accuracy": float(baseline_correct.mean()),
                "coverage": float(retained.mean()),
                "retained_accuracy": safe_div(post_correct.sum(), retained.sum()),
                "coverage_adjusted_correct_mass": float(post_correct.mean()),
                "delta_adjusted_vs_baseline": float(post_correct.mean() - baseline_correct.mean()),
            }
        ]
    )


def plot_detection(benchmark: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    primary = benchmark[benchmark["estimator"].eq("ensemble")]
    for issue, label, color in [(0, "Expert-correct", "#4C78A8"), (1, "Expert issue", "#E45756")]:
        values = primary.loc[primary["true_issue"].eq(issue), "suspicion_percentile_self"]
        axes[0].hist(values, bins=20, alpha=0.55, density=True, label=label, color=color)
    axes[0].set_xlabel("CL suspiciousness percentile within finding")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Expert issues should shift to the right")
    axes[0].legend(frameon=False)

    for estimator, color in [
        ("seed_13", "#8C8C8C"),
        ("seed_42", "#AAAAAA"),
        ("seed_97", "#BEBEBE"),
        ("seed_123", "#D0D0D0"),
        ("ensemble", "#D62728"),
    ]:
        frame = benchmark[benchmark["estimator"].eq(estimator)]
        precision, recall, _ = precision_recall_curve(
            frame["true_issue"], frame["suspicion_percentile_self"]
        )
        axes[1].plot(recall, precision, label=estimator, color=color, linewidth=2 if estimator == "ensemble" else 1)
    prevalence = EXPECTED_BENCHMARK_ISSUES / EXPECTED_BENCHMARK_ENTRIES
    axes[1].axhline(prevalence, color="black", linestyle="--", linewidth=1, label="No-skill prevalence")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Med-PaLM expert issue detection")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "medpalm_cl_detection.png", dpi=180)
    plt.close(figure)


def evaluate(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "full_test_cl_entry_scores.csv"
    scoring_manifest_path = args.output_dir / "scoring_manifest.json"
    if not (args.output_dir / ".scores_complete").exists():
        raise FileNotFoundError("CL scoring completion marker is missing")
    if not score_path.exists() or not scoring_manifest_path.exists():
        raise FileNotFoundError("CL scoring outputs are incomplete")
    scoring_manifest = json.loads(scoring_manifest_path.read_text(encoding="utf-8"))
    if sha256_file(score_path) != scoring_manifest["score_sha256"]:
        raise ValueError("Frozen CL score file hash mismatch")
    if scoring_manifest.get("expert_reference_used") is not False:
        raise ValueError("CL scoring was not outcome blind")

    blinded_path = require_path(args.benchmark_blinded, "--benchmark-blinded")
    reference_path = require_path(args.benchmark_reference, "--benchmark-reference")
    blinded = pd.read_csv(blinded_path)
    reference = pd.read_csv(reference_path)
    benchmark_reference = blinded[
        ["entry_key", "subject_id", "study_id", "finding", "current_binary_label"]
    ].merge(
        reference[
            [
                "entry_key",
                "subject_id",
                "study_id",
                "finding",
                "reference_binary_label",
                "expected_binary_action",
            ]
        ],
        on=["entry_key", "subject_id", "study_id", "finding"],
        how="inner",
        validate="one_to_one",
    )
    benchmark_reference["true_issue"] = benchmark_reference["expected_binary_action"].eq(
        "relabel"
    ).astype(int)
    if len(benchmark_reference) != EXPECTED_BENCHMARK_ENTRIES:
        raise ValueError("Unexpected compatible Med-PaLM benchmark size")
    if int(benchmark_reference["true_issue"].sum()) != EXPECTED_BENCHMARK_ISSUES:
        raise ValueError("Unexpected expert issue count")

    scores = pd.read_csv(score_path)
    benchmark = scores.merge(
        benchmark_reference,
        on=["entry_key", "subject_id", "study_id"],
        how="inner",
        validate="many_to_one",
        suffixes=("", "_reference"),
    )
    if len(benchmark) != EXPECTED_BENCHMARK_ENTRIES * 5:
        raise ValueError(f"Expected 2,490 estimator-entry rows, found {len(benchmark)}")
    if not benchmark["label_name"].eq(benchmark["finding"]).all():
        raise ValueError("Finding names differ between CL scores and expert reference")
    if not benchmark["current_binary_label"].eq(
        benchmark["current_binary_label_reference"]
    ).all():
        raise ValueError("Current binary labels differ between score and benchmark tables")

    count_table = pd.crosstab(
        benchmark_reference["finding"], benchmark_reference["true_issue"]
    ).reindex(columns=[0, 1], fill_value=0)
    supported_labels = sorted(
        count_table[
            (count_table.sum(axis=1) >= 20)
            & (count_table[0] >= 10)
            & (count_table[1] >= 10)
        ].index.tolist()
    )
    if supported_labels != ["Pleural Effusion", "Pneumonia", "Pneumothorax"]:
        raise ValueError(f"Unexpected supported-label set: {supported_labels}")

    method_columns = {
        "self_percentile": "suspicion_percentile_self",
        "self_raw": "suspicion_self_confidence",
        "margin_percentile": "suspicion_percentile_margin",
        "margin_raw": "suspicion_normalized_margin",
    }
    metric_rows = []
    for estimator, estimator_frame in benchmark.groupby("estimator", sort=True):
        cohorts: dict[str, pd.DataFrame] = {
            "all_entries": estimator_frame,
            "supported_labels": estimator_frame[
                estimator_frame["label_name"].isin(supported_labels)
            ],
        }
        for label, group in estimator_frame.groupby("label_name", sort=True):
            cohorts[f"label::{label}"] = group
        for cohort_name, cohort_frame in cohorts.items():
            for method, column in method_columns.items():
                metric_rows.append(
                    detection_metrics(cohort_frame, column, cohort_name, estimator, method)
                )
    metrics = pd.DataFrame(metric_rows)

    primary = benchmark[benchmark["estimator"].eq("ensemble")].copy()
    intervals = bootstrap_primary(
        primary,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    interval_map = intervals.set_index("metric")
    seed_rows = metrics[
        metrics["estimator"].isin([f"seed_{seed}" for seed in DEFAULT_SEEDS])
        & metrics["cohort"].eq("all_entries")
        & metrics["method"].eq("self_percentile")
    ]
    seed_positive = int((seed_rows["auprc_minus_prevalence"] > 0).sum())
    ap_supported = float(interval_map.loc["auprc_minus_prevalence", "ci_lower_95"] > 0)
    enrichment_supported = float(interval_map.loc["top20_enrichment", "ci_lower_95"] > 1)
    gate_pass = bool(ap_supported and enrichment_supported and seed_positive >= 3)
    if gate_pass:
        gate_label = "supported"
    else:
        primary_row = metrics[
            metrics["estimator"].eq("ensemble")
            & metrics["cohort"].eq("all_entries")
            & metrics["method"].eq("self_percentile")
        ].iloc[0]
        gate_label = (
            "suggestive"
            if primary_row["auprc_minus_prevalence"] > 0
            and primary_row["top20_enrichment"] > 1
            else "not_supported"
        )

    replicate_control, control_summary = synthetic_positive_control(
        primary,
        replicates=args.synthetic_replicates,
        seed=args.bootstrap_seed + 1,
    )
    llm_path = require_path(args.llm_evaluation, "--llm-evaluation")
    end_to_end = end_to_end_secondary(primary, llm_path)

    benchmark.to_csv(args.output_dir / "medpalm_cl_detection_rows_private.csv", index=False)
    metrics.to_csv(args.output_dir / "medpalm_cl_detection_metrics.csv", index=False)
    intervals.to_csv(args.output_dir / "medpalm_cl_detection_subject_bootstrap_ci.csv", index=False)
    count_table.rename(columns={0: "expert_correct", 1: "true_issue"}).to_csv(
        args.output_dir / "medpalm_detection_label_support.csv"
    )
    replicate_control.to_csv(
        args.output_dir / "medpalm_synthetic_positive_control_replicates.csv", index=False
    )
    control_summary.to_csv(
        args.output_dir / "medpalm_synthetic_positive_control_summary.csv", index=False
    )
    end_to_end.to_csv(args.output_dir / "medpalm_cl_llm_end_to_end_secondary.csv", index=False)
    plot_detection(benchmark, args.output_dir)

    primary_row = metrics[
        metrics["estimator"].eq("ensemble")
        & metrics["cohort"].eq("all_entries")
        & metrics["method"].eq("self_percentile")
    ].iloc[0].to_dict()
    summary = {
        "protocol": PROTOCOL_NAME,
        "status": gate_label,
        "gate_pass": gate_pass,
        "gate_definition": {
            "auprc_minus_prevalence_ci_lower_above_zero": bool(ap_supported),
            "top20_enrichment_ci_lower_above_one": bool(enrichment_supported),
            "individual_seeds_above_prevalence": seed_positive,
            "required_seed_count": 3,
        },
        "primary": primary_row,
        "supported_labels": supported_labels,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_cluster": "subject_id",
        "score_sha256": sha256_file(score_path),
        "benchmark_blinded_sha256": sha256_file(blinded_path),
        "benchmark_reference_sha256": sha256_file(reference_path),
        "interpretation_boundary": (
            "Conditional expert-adjudicated hard-case discrimination; not population precision/recall."
        ),
    }
    write_json(args.output_dir / "medpalm_cl_detection_summary.json", summary)
    (args.output_dir / ".evaluation_complete").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nEnd-to-end secondary:")
    print(end_to_end.to_string(index=False))


def main() -> None:
    args = parse_args()
    if args.mode == "infer":
        infer(args)
    elif args.mode == "score":
        score(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
