#!/usr/bin/env python3
"""Calibrate MIMIC-CXR DQS on expert-labelled test cohorts and transfer it.

The ``calibrate`` stage reads only fold-model predictions for the official
test split and the two predeclared expert references.  It freezes one global
self-confidence threshold per seed, fold model, and reference cohort.  The
``evaluate`` stage subsequently applies those frozen thresholds to the
existing MIMIC-CXR training-label trajectory.  Complete training-label truth
is never used or assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cleanlab.dataset import overall_label_health_score

from cxr_real_full_train_eval_cleanlab_xrv12 import LABEL_NAMES


PROTOCOL = "mimic_dqs_calibration_transfer_v1"
SEEDS = (7, 13, 42, 97, 123)
N_SPLITS = 4
EXPECTED_TRAIN_IMAGES = 237_717
EXPECTED_TEST_IMAGES = 3_414
EXPECTED_REFERENCES = {
    "radiologist_panel": {"entries": 1_796, "issues": 109, "studies": 563, "subjects": 251},
    "medpalm_hard_cases": {"entries": 498, "issues": 127, "studies": 457, "subjects": 179},
}
ARCHIVE_ROOT = Path(
    "/vol/bitbucket/yz3522/NoiseRate_results_archive/cxr_real_experiment/"
    "results_mobilenetv3_full_issue_pool_no_repeat_gpt54"
)
SEED_ROOTS = {
    7: ARCHIVE_ROOT / "20260823_additional5_v1/seed_7",
    13: ARCHIVE_ROOT / "20260820_seed13_v1/seed_13",
    42: ARCHIVE_ROOT / "20260823_additional5_v1/seed_42",
    97: ARCHIVE_ROOT / "20260823_additional5_v1/seed_97",
    123: ARCHIVE_ROOT / "20260823_additional5_v1/seed_123",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def parse_seeds(text: str) -> list[int]:
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if tuple(seeds) != SEEDS:
        raise ValueError(f"Formal seeds must be {SEEDS}, in that order")
    return seeds


def self_confidence(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("Self-confidence inputs must be aligned binary arrays")
    if not np.isfinite(probabilities).all() or not (
        (probabilities >= 0.0) & (probabilities <= 1.0)
    ).all():
        raise ValueError("Probabilities must lie in [0, 1]")
    return np.where(labels == 1, probabilities, 1.0 - probabilities)


def select_prevalence_matching_threshold(
    scores: np.ndarray, correct: np.ndarray
) -> tuple[float, float, int]:
    """Match total correct-label mass; ties use the higher threshold."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=np.int64)
    if scores.ndim != 1 or correct.ndim != 1 or len(scores) != len(correct) or not len(scores):
        raise ValueError("Calibration scores and targets must be aligned non-empty vectors")
    if not np.isfinite(scores).all() or not np.isin(correct, [0, 1]).all():
        raise ValueError("Calibration inputs are invalid")
    unique = np.unique(scores)
    candidates = np.concatenate(
        ([np.nextafter(unique[-1], np.inf)], unique, [np.nextafter(unique[0], -np.inf)])
    )
    target = int(correct.sum())
    best: tuple[int, float, int] | None = None
    best_threshold = float("nan")
    best_count = -1
    for threshold in candidates:
        count = int((scores >= threshold).sum())
        key = (abs(count - target), -float(threshold), count)
        if best is None or key < best:
            best = key
            best_threshold = float(threshold)
            best_count = count
    return best_threshold, best_count / len(scores), best_count


def parse_pipe_matrix(values: pd.Series, dtype: type) -> np.ndarray:
    return np.vstack([np.fromstring(str(value), sep="|", dtype=dtype) for value in values])


def load_score(score_root: Path, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    root = score_root / f"seed_{seed}"
    if not (root / ".score_complete").is_file():
        raise FileNotFoundError(f"Score marker is missing for seed {seed}")
    manifest_path = root / "score_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "protocol": PROTOCOL,
        "phase": "outcome_blind_fold_scoring",
        "expert_reference_used": False,
        "seed": seed,
        "n_splits": N_SPLITS,
        "train_images": EXPECTED_TRAIN_IMAGES,
        "test_images": EXPECTED_TEST_IMAGES,
        "label_names": LABEL_NAMES,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Unexpected {key} in score manifest for seed {seed}")
    for name, expected_hash in manifest["artifact_sha256"].items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Score artifact failed its hash check: {path}")
    reproduction = json.loads((root / "reproduction_summary.json").read_text(encoding="utf-8"))
    if reproduction.get("row_identity_exact") is not True:
        raise ValueError(f"Archived row identity was not reproduced for seed {seed}")
    action = np.load(root / "action_oof_predictions.npz")
    action_probs = action["probabilities"]
    action_fold = action["fold"]
    action_pool_id = action["pool_row_id"]
    if action_probs.shape != (EXPECTED_TRAIN_IMAGES, len(LABEL_NAMES)):
        raise ValueError(f"Unexpected action probability shape for seed {seed}")
    if action_fold.shape != (EXPECTED_TRAIN_IMAGES,) or set(np.unique(action_fold)) != set(
        range(1, N_SPLITS + 1)
    ):
        raise ValueError(f"Unexpected action folds for seed {seed}")
    test_index = pd.read_csv(root / "test_image_index.csv")
    test_predictions = np.load(root / "test_fold_predictions.npz")["probabilities"]
    if len(test_index) != EXPECTED_TEST_IMAGES or test_predictions.shape != (
        N_SPLITS,
        EXPECTED_TEST_IMAGES,
        len(LABEL_NAMES),
    ):
        raise ValueError(f"Unexpected test prediction dimensions for seed {seed}")
    return test_index, test_predictions, action_probs, np.column_stack([action_fold, action_pool_id])


def load_reference(path: Path, anchor: str) -> pd.DataFrame:
    reference = pd.read_csv(path)
    require_columns(
        reference,
        [
            "entry_key",
            "subject_id",
            "study_id",
            "label_name",
            "current_binary_label",
            "reference_binary_label",
            "true_issue",
        ],
        anchor,
    )
    if reference["entry_key"].duplicated().any():
        raise ValueError(f"{anchor} contains duplicate entry keys")
    if not reference["label_name"].isin(LABEL_NAMES).all():
        raise ValueError(f"{anchor} contains labels outside the twelve-finding task")
    observed = {
        "entries": int(len(reference)),
        "issues": int(reference["true_issue"].sum()),
        "studies": int(reference["study_id"].nunique()),
        "subjects": int(reference["subject_id"].nunique()),
    }
    if observed != EXPECTED_REFERENCES[anchor]:
        raise ValueError(f"{anchor} reference counts changed: {observed}")
    calculated_issue = reference["current_binary_label"].astype(int).ne(
        reference["reference_binary_label"].astype(int)
    )
    if not calculated_issue.eq(reference["true_issue"].astype(bool)).all():
        raise ValueError(f"{anchor} issue indicators are inconsistent")
    return reference


def expand_reference(reference: pd.DataFrame, test_index: pd.DataFrame) -> pd.DataFrame:
    image_index = test_index.reset_index(names="test_position")[
        ["test_position", "subject_id", "study_id", "dicom_id"]
    ]
    expanded = reference.merge(
        image_index,
        on=["subject_id", "study_id"],
        how="inner",
        validate="many_to_many",
    )
    if expanded.empty:
        raise ValueError("Reference has no overlap with test-image predictions")
    expanded["expanded_key"] = expanded["entry_key"].astype(str) + "::" + expanded[
        "dicom_id"
    ].astype(str)
    if expanded["expanded_key"].duplicated().any():
        raise ValueError("Expanded image-finding calibration keys are not unique")
    expanded["label_index"] = expanded["label_name"].map(
        {name: index for index, name in enumerate(LABEL_NAMES)}
    )
    return expanded


def calibrate(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    score_root = Path(args.score_root)
    seeds = parse_seeds(args.seeds)
    references = {
        "radiologist_panel": load_reference(Path(args.radiologist_reference), "radiologist_panel"),
        "medpalm_hard_cases": load_reference(Path(args.medpalm_reference), "medpalm_hard_cases"),
    }
    threshold_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    canonical_index: pd.DataFrame | None = None
    for seed in seeds:
        test_index, test_predictions, _, _ = load_score(score_root, seed)
        identity = test_index[["pool_row_id", "subject_id", "study_id", "dicom_id"]].astype(str)
        if canonical_index is None:
            canonical_index = identity
        elif not identity.equals(canonical_index):
            raise ValueError("Test-image identity differs across scoring seeds")
        for anchor, reference in references.items():
            expanded = expand_reference(reference, test_index)
            correct = expanded["current_binary_label"].astype(int).eq(
                expanded["reference_binary_label"].astype(int)
            ).astype(int).to_numpy()
            reference_rows.append(
                {
                    "seed": seed,
                    "anchor": anchor,
                    "study_finding_entries": len(reference),
                    "expanded_image_finding_entries": len(expanded),
                    "study_weighted_known_quality": 1.0 - float(reference["true_issue"].mean()),
                    "image_weighted_known_quality": float(correct.mean()),
                }
            )
            positions = expanded["test_position"].to_numpy(dtype=int)
            label_indices = expanded["label_index"].to_numpy(dtype=int)
            labels = expanded["current_binary_label"].to_numpy(dtype=int)
            for fold_id in range(1, N_SPLITS + 1):
                probability = test_predictions[fold_id - 1, positions, label_indices]
                scores = self_confidence(labels, probability)
                threshold, estimated_quality, estimated_count = select_prevalence_matching_threshold(
                    scores, correct
                )
                threshold_rows.append(
                    {
                        "anchor": anchor,
                        "seed": seed,
                        "fold_id": fold_id,
                        "threshold": threshold,
                        "calibration_entries": len(expanded),
                        "known_correct_entries": int(correct.sum()),
                        "estimated_correct_entries": estimated_count,
                        "known_quality": float(correct.mean()),
                        "estimated_quality": estimated_quality,
                        "absolute_prevalence_error": abs(estimated_quality - float(correct.mean())),
                    }
                )
    thresholds = pd.DataFrame(threshold_rows).sort_values(["anchor", "seed", "fold_id"])
    if len(thresholds) != len(references) * len(seeds) * N_SPLITS or thresholds[
        ["anchor", "seed", "fold_id"]
    ].duplicated().any():
        raise ValueError("Threshold table is incomplete")
    reference_summary = pd.DataFrame(reference_rows).sort_values(["anchor", "seed"])
    threshold_path = output / "calibration_thresholds_private.csv"
    reference_path = output / "calibration_reference_summary_private.csv"
    atomic_csv(thresholds, threshold_path)
    atomic_csv(reference_summary, reference_path)
    manifest = {
        "protocol": PROTOCOL,
        "stage": "calibration_frozen_before_training_trajectory_access",
        "seeds": seeds,
        "anchors": ["radiologist_panel", "medpalm_hard_cases"],
        "main_anchor": "radiologist_panel",
        "sensitivity_anchor": "medpalm_hard_cases",
        "threshold_design": "one global self-confidence threshold per anchor, seed, and fold model",
        "calibration_unit": "AP/PA image-finding entry expanded from expert study-finding labels",
        "radiologist_reference_sha256": sha256_file(Path(args.radiologist_reference)),
        "medpalm_reference_sha256": sha256_file(Path(args.medpalm_reference)),
        "thresholds_sha256": sha256_file(threshold_path),
        "reference_summary_sha256": sha256_file(reference_path),
        "program_sha256": sha256_file(Path(__file__)),
        "training_trajectory_accessed": False,
    }
    atomic_text(output / "calibration_manifest_private.json", json.dumps(manifest, indent=2) + "\n")
    atomic_text(output / ".calibration_frozen", manifest["thresholds_sha256"] + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def initial_state(root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    path = root / "llm_refine/loop_01/oof/train_cleanlab_sample_details.csv"
    frame = pd.read_csv(
        path,
        usecols=["pool_row_id", "binary_labels_for_detection", "valid_label_mask", "pred_probs"],
    )
    labels = parse_pipe_matrix(frame["binary_labels_for_detection"], float)
    valid = parse_pipe_matrix(frame["valid_label_mask"], int).astype(bool)
    archived_probs = parse_pipe_matrix(frame["pred_probs"], float)
    if labels.shape != valid.shape or labels.shape != archived_probs.shape:
        raise ValueError(f"Initial state matrices disagree for {root}")
    return frame, labels, valid, archived_probs


def apply_actions(
    root: Path,
    round_id: int,
    pool_to_index: dict[int, int],
    initial_labels: np.ndarray,
    initial_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    loop = root / f"llm_refine/loop_{round_id:02d}"
    relabels = pd.read_csv(loop / "applied_relabel_entries.csv")
    masks = pd.read_csv(loop / "applied_mask_entries.csv")
    keys = ["pool_row_id", "label_index"]
    if relabels.duplicated(keys).any() or masks.duplicated(keys).any():
        raise ValueError(f"Duplicate cumulative actions in {loop}")
    if not relabels[keys].merge(masks[keys], on=keys, how="inner").empty:
        raise ValueError(f"Relabel and mask actions overlap in {loop}")
    labels = initial_labels.copy()
    valid = initial_valid.copy()
    if not relabels.empty:
        rows = np.asarray([pool_to_index[int(value)] for value in relabels["pool_row_id"]])
        columns = relabels["label_index"].to_numpy(dtype=int)
        if not initial_valid[rows, columns].all():
            raise ValueError(f"Relabel actions include initially invalid entries in {loop}")
        raw = relabels["new_raw_label"].to_numpy(dtype=float)
        labels[rows, columns] = np.where(np.isin(raw, [1.0, -1.0]), 1.0, 0.0)
    if not masks.empty:
        rows = np.asarray([pool_to_index[int(value)] for value in masks["pool_row_id"]])
        columns = masks["label_index"].to_numpy(dtype=int)
        labels[rows, columns] = np.nan
        valid[rows, columns] = False
    return labels, valid, len(relabels), len(masks)


def official_dqs(labels: np.ndarray, valid: np.ndarray, probabilities: np.ndarray) -> float:
    y = labels[valid].astype(int)
    p = probabilities[valid]
    return float(
        overall_label_health_score(
            labels=y,
            pred_probs=np.column_stack([1.0 - p, p]),
            verbose=False,
        )
    )


def threshold_dqs(
    labels: np.ndarray,
    valid: np.ndarray,
    probabilities: np.ndarray,
    fold: np.ndarray,
    threshold_table: pd.DataFrame,
) -> float:
    lookup = threshold_table.set_index("fold_id")["threshold"].to_dict()
    if set(lookup) != set(range(1, N_SPLITS + 1)):
        raise ValueError("Fold threshold lookup is incomplete")
    thresholds = np.asarray([lookup[int(value)] for value in fold], dtype=float)[:, None]
    tiled_thresholds = np.broadcast_to(thresholds, labels.shape)
    return float(
        (
            self_confidence(labels[valid].astype(int), probabilities[valid])
            >= tiled_thresholds[valid]
        ).mean()
    )


def evaluate(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    score_root = Path(args.score_root)
    seeds = parse_seeds(args.seeds)
    threshold_path = Path(args.thresholds)
    observed_hash = sha256_file(threshold_path)
    if observed_hash != args.expected_thresholds_sha256:
        raise ValueError("Frozen threshold hash does not match the submitted hash")
    thresholds = pd.read_csv(threshold_path)
    require_columns(thresholds, ["anchor", "seed", "fold_id", "threshold"], "thresholds")
    if len(thresholds) != 2 * len(seeds) * N_SPLITS:
        raise ValueError("Frozen threshold table has an unexpected size")
    records: list[dict[str, Any]] = []
    reproduction_records: list[dict[str, Any]] = []
    for seed in seeds:
        root = SEED_ROOTS[seed]
        frame, labels0, valid0, archived_probs = initial_state(root)
        _, _, replay_probs, action_identity = load_score(score_root, seed)
        fold = action_identity[:, 0].astype(int)
        pool_id = action_identity[:, 1].astype(np.int64)
        archived_pool_id = frame["pool_row_id"].to_numpy(dtype=np.int64)
        if not np.array_equal(pool_id, archived_pool_id):
            raise ValueError(f"Action row identity differs for seed {seed}")
        reproduction = json.loads(
            (score_root / f"seed_{seed}" / "reproduction_summary.json").read_text(encoding="utf-8")
        )
        reproduction_records.append(
            {
                "seed": seed,
                "probability_mae": reproduction["flattened_probability_mae"],
                "probability_max_abs_error": reproduction["flattened_probability_max_abs_error"],
                "archived_dqs": reproduction["archived_dqs"],
                "replay_dqs": reproduction["replay_dqs"],
                "dqs_difference": reproduction["replay_dqs"] - reproduction["archived_dqs"],
            }
        )
        pool_to_index = {int(value): index for index, value in enumerate(archived_pool_id)}
        for round_id in range(0, 6):
            if round_id == 0:
                labels, valid, relabel_count, mask_count = labels0, valid0, 0, 0
            else:
                labels, valid, relabel_count, mask_count = apply_actions(
                    root, round_id, pool_to_index, labels0, valid0
                )
            archived_dqs = official_dqs(labels, valid, archived_probs)
            replay_dqs = official_dqs(labels, valid, replay_probs)
            for anchor in ["radiologist_panel", "medpalm_hard_cases"]:
                selected_thresholds = thresholds.loc[
                    thresholds["anchor"].eq(anchor) & thresholds["seed"].eq(seed),
                    ["fold_id", "threshold"],
                ]
                estimate = threshold_dqs(labels, valid, replay_probs, fold, selected_thresholds)
                records.append(
                    {
                        "seed": seed,
                        "round": round_id,
                        "anchor": anchor,
                        "calibrated_dqs": estimate,
                        "replay_uncalibrated_dqs": replay_dqs,
                        "archived_uncalibrated_dqs": archived_dqs,
                        "evaluable_entries": int(valid.sum()),
                        "cumulative_relabels": relabel_count,
                        "cumulative_masks": mask_count,
                    }
                )
    trajectory = pd.DataFrame(records).sort_values(["anchor", "round", "seed"])
    if len(trajectory) != 2 * len(seeds) * 6 or trajectory[
        ["anchor", "seed", "round"]
    ].duplicated().any():
        raise ValueError("Transferred trajectory is incomplete")
    summary = (
        trajectory.groupby(["anchor", "round"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            calibrated_dqs_mean=("calibrated_dqs", "mean"),
            calibrated_dqs_sd=("calibrated_dqs", "std"),
            replay_uncalibrated_dqs_mean=("replay_uncalibrated_dqs", "mean"),
            archived_uncalibrated_dqs_mean=("archived_uncalibrated_dqs", "mean"),
            evaluable_entries_mean=("evaluable_entries", "mean"),
        )
        .sort_values(["anchor", "round"])
    )
    if not summary["seeds"].eq(len(seeds)).all():
        raise ValueError("At least one trajectory state lacks five seeds")
    reproduction = pd.DataFrame(reproduction_records).sort_values("seed")
    trajectory_path = output / "mimic_calibrated_dqs_trajectories_private.csv"
    summary_path = output / "mimic_calibrated_dqs_summary.csv"
    reproduction_path = output / "round0_reproduction_summary.csv"
    atomic_csv(trajectory, trajectory_path)
    atomic_csv(summary, summary_path)
    atomic_csv(reproduction, reproduction_path)
    make_figure(summary, output / "mimic_calibrated_dqs_sensitivity")
    main_rows = summary[summary["anchor"].eq("radiologist_panel")].sort_values("round")
    sensitivity_rows = summary[summary["anchor"].eq("medpalm_hard_cases")].sort_values("round")
    analysis = {
        "protocol": PROTOCOL,
        "main_anchor": "radiologist_panel",
        "sensitivity_anchor": "medpalm_hard_cases",
        "radiologist_calibrated_round0": float(main_rows.iloc[0]["calibrated_dqs_mean"]),
        "radiologist_calibrated_round5": float(main_rows.iloc[-1]["calibrated_dqs_mean"]),
        "medpalm_calibrated_round0": float(sensitivity_rows.iloc[0]["calibrated_dqs_mean"]),
        "medpalm_calibrated_round5": float(sensitivity_rows.iloc[-1]["calibrated_dqs_mean"]),
        "mean_absolute_anchor_difference": float(
            np.mean(
                np.abs(
                    main_rows["calibrated_dqs_mean"].to_numpy()
                    - sensitivity_rows["calibrated_dqs_mean"].to_numpy()
                )
            )
        ),
        "interpretation": (
            "The radiologist panel is the predeclared main calibration candidate; Med-PaLM is a "
            "disagreement-selected sensitivity anchor. Neither supplies complete training-set truth, "
            "so absolute calibration error on the full MIMIC-CXR training set is not identifiable."
        ),
        "thresholds_sha256": observed_hash,
        "trajectory_sha256": sha256_file(trajectory_path),
        "summary_sha256": sha256_file(summary_path),
        "reproduction_sha256": sha256_file(reproduction_path),
        "figure_pdf_sha256": sha256_file(output / "mimic_calibrated_dqs_sensitivity.pdf"),
        "figure_png_sha256": sha256_file(output / "mimic_calibrated_dqs_sensitivity.png"),
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_text(output / "mimic_calibration_transfer_summary.json", json.dumps(analysis, indent=2) + "\n")
    atomic_text(output / ".evaluation_complete", "complete\n")
    print(json.dumps(analysis, indent=2), flush=True)


def make_figure(summary: pd.DataFrame, stem: Path) -> None:
    font_path = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
    if font_path.is_file():
        import matplotlib as mpl

        mpl.font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    colors = {"radiologist_panel": "#0077BB", "medpalm_hard_cases": "#EE7733"}
    labels = {
        "radiologist_panel": "Radiologist-panel calibration",
        "medpalm_hard_cases": "Med-PaLM hard-case calibration",
    }
    raw = summary[summary["anchor"].eq("radiologist_panel")].sort_values("round")
    ax.plot(
        raw["round"],
        raw["archived_uncalibrated_dqs_mean"],
        color="#777777",
        linestyle=":",
        marker="o",
        linewidth=1.5,
        label="Uncalibrated DQS",
    )
    for anchor in ["radiologist_panel", "medpalm_hard_cases"]:
        frame = summary[summary["anchor"].eq(anchor)].sort_values("round")
        x = frame["round"].to_numpy(dtype=float)
        y = frame["calibrated_dqs_mean"].to_numpy(dtype=float)
        sd = frame["calibrated_dqs_sd"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.7, color=colors[anchor], label=labels[anchor])
        ax.fill_between(x, y - sd, y + sd, color=colors[anchor], alpha=0.10, linewidth=0)
    ax.set(xlabel="Refinement round", ylabel="Dataset-quality estimate", xticks=range(0, 6))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def verify(args: argparse.Namespace) -> None:
    calibration = Path(args.calibration_dir)
    evaluation = Path(args.evaluation_dir)
    if not (calibration / ".calibration_frozen").is_file() or not (
        evaluation / ".evaluation_complete"
    ).is_file():
        raise FileNotFoundError("Calibration or evaluation completion marker is missing")
    threshold_path = calibration / "calibration_thresholds_private.csv"
    frozen_hash = (calibration / ".calibration_frozen").read_text(encoding="utf-8").strip()
    if sha256_file(threshold_path) != frozen_hash:
        raise ValueError("Frozen threshold file changed after calibration")
    summary = json.loads(
        (evaluation / "mimic_calibration_transfer_summary.json").read_text(encoding="utf-8")
    )
    checks = {
        "trajectory_sha256": evaluation / "mimic_calibrated_dqs_trajectories_private.csv",
        "summary_sha256": evaluation / "mimic_calibrated_dqs_summary.csv",
        "reproduction_sha256": evaluation / "round0_reproduction_summary.csv",
        "figure_pdf_sha256": evaluation / "mimic_calibrated_dqs_sensitivity.pdf",
        "figure_png_sha256": evaluation / "mimic_calibrated_dqs_sensitivity.png",
    }
    for key, path in checks.items():
        if sha256_file(path) != summary[key]:
            raise ValueError(f"Evaluation artifact changed after completion: {path}")
    print("MIMIC-CXR calibration-transfer postflight passed", flush=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    calibrate_parser = commands.add_parser("calibrate")
    calibrate_parser.add_argument("--score-root", required=True)
    calibrate_parser.add_argument("--radiologist-reference", required=True)
    calibrate_parser.add_argument("--medpalm-reference", required=True)
    calibrate_parser.add_argument("--output-dir", required=True)
    calibrate_parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--score-root", required=True)
    evaluate_parser.add_argument("--thresholds", required=True)
    evaluate_parser.add_argument("--expected-thresholds-sha256", required=True)
    evaluate_parser.add_argument("--output-dir", required=True)
    evaluate_parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--calibration-dir", required=True)
    verify_parser.add_argument("--evaluation-dir", required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    {"calibrate": calibrate, "evaluate": evaluate, "verify": verify}[args.command](args)


if __name__ == "__main__":
    main()
