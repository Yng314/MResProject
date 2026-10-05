#!/usr/bin/env python3
"""Five-loop VinDr top-20% refinement with frozen calibrated DQS thresholds.

This driver owns the refinement state only.  An external runner trains the OOF
models and supplies their output directories at initialization and after every
oracle update.  Calibration fitting is deliberately outside this program: this
program accepts, snapshots, and applies one already-frozen threshold per fold.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROTOCOL_NAME = "vindr_top20_calibrated_refinement_v1"
LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Lung Opacity",
    "Pleural effusion",
    "Pneumonia",
]
ACTION_IMAGES = 2_400
ACTION_ENTRIES = ACTION_IMAGES * len(LABELS)
INITIAL_ERRORS = 2_880
INITIAL_QUALITY = 0.80
N_FOLDS = 4
LOOPS = 5
TOP_FRACTION = 0.20
OOF_MARKER = ".blind_run_complete"
OOF_EVIDENCE = "entry_evidence.csv"
DEFAULT_SEEDS = (13, 42, 97, 123, 211, 307, 509, 701)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, allow_nan=False))


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def parse_seeds(text: str | Iterable[int]) -> list[int]:
    if isinstance(text, str):
        seeds = [int(token.strip()) for token in text.split(",") if token.strip()]
    else:
        seeds = [int(value) for value in text]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be non-empty and unique")
    return seeds


def make_entry_key(image_id: pd.Series, label_name: pd.Series) -> pd.Series:
    return image_id.astype(str) + "::" + label_name.astype(str)


def state_dir(output: Path) -> Path:
    return output / "state"


def loop_dir(output: Path, loop_id: int) -> Path:
    return output / f"loop_{loop_id:02d}"


def _validate_loop_id(loop_id: int) -> None:
    if not 1 <= int(loop_id) <= LOOPS:
        raise ValueError(f"loop_id must be within 1..{LOOPS}")


def _assert_absent(paths: Iterable[Path], stage: str) -> None:
    present = [str(path) for path in paths if path.exists()]
    if present:
        raise RuntimeError(
            f"{stage} has uncommitted output and will not overwrite it: {present}"
        )


def validate_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    require_columns(cohort, ["image_id", "image_path", "fold_id"] + LABELS, "action cohort")
    result = cohort.copy()
    result["image_id"] = result["image_id"].astype(str)
    if len(result) != ACTION_IMAGES or result["image_id"].duplicated().any():
        raise ValueError(f"Action cohort must contain {ACTION_IMAGES} unique images")
    forbidden = [
        column
        for column in result.columns
        if any(token in column.lower() for token in ("clean_label", "private_reference", "injected_error"))
    ]
    if forbidden:
        raise ValueError(f"Action cohort exposes private outcome columns: {forbidden}")
    for label in LABELS:
        if not result[label].isin([0, 1, False, True]).all():
            raise ValueError(f"Action cohort has non-binary values in {label}")
        result[label] = result[label].astype(int)
    result["fold_id"] = pd.to_numeric(result["fold_id"], errors="raise").astype(int)
    folds = sorted(result["fold_id"].unique().tolist())
    if folds != list(range(N_FOLDS)):
        raise ValueError(f"Action cohort folds must be 0..{N_FOLDS - 1}; found {folds}")
    return result


def cohort_entries(cohort: pd.DataFrame) -> pd.DataFrame:
    cohort = validate_cohort(cohort)
    parts: list[pd.DataFrame] = []
    for label_index, label_name in enumerate(LABELS):
        part = cohort[["image_id", "fold_id", label_name]].copy()
        part.columns = ["image_id", "fold_id", "current_label"]
        part["label_index"] = label_index
        part["label_name"] = label_name
        parts.append(part)
    entries = pd.concat(parts, ignore_index=True)
    entries["entry_key"] = make_entry_key(entries["image_id"], entries["label_name"])
    if len(entries) != ACTION_ENTRIES or entries["entry_key"].duplicated().any():
        raise RuntimeError("Action cohort did not expand to 14,400 unique label entries")
    return entries


def validate_private_reference(private: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        private,
        [
            "image_id",
            "label_name",
            "clean_label",
            "noisy_label",
            "injected_error",
            "flip_direction",
        ],
        "private reference",
    )
    result = private.copy()
    result["image_id"] = result["image_id"].astype(str)
    result["entry_key"] = make_entry_key(result["image_id"], result["label_name"])
    if len(result) != ACTION_ENTRIES or result["entry_key"].duplicated().any():
        raise ValueError("Private reference must contain 14,400 unique label entries")
    if set(result["label_name"].astype(str)) != set(LABELS):
        raise ValueError("Private reference label names do not match the locked findings")
    for column in ["clean_label", "noisy_label", "injected_error"]:
        if not result[column].isin([0, 1, False, True]).all():
            raise ValueError(f"Private reference has non-binary values in {column}")
        result[column] = result[column].astype(int)
    expected_issue = result["clean_label"].ne(result["noisy_label"]).astype(int)
    if not result["injected_error"].eq(expected_issue).all():
        raise ValueError("Private injected_error disagrees with clean/noisy labels")
    return result


def validate_evidence(cohort: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        evidence,
        [
            "image_id",
            "fold_id",
            "label_index",
            "label_name",
            "noisy_label",
            "oof_probability",
            "label_quality_self_confidence",
            "cl_issue",
            "self_confidence_suspicion",
            "cl_first_score",
        ],
        "OOF entry evidence",
    )
    result = evidence.copy()
    result["image_id"] = result["image_id"].astype(str)
    calculated_key = make_entry_key(result["image_id"], result["label_name"])
    if "entry_key" in result.columns and not result["entry_key"].astype(str).eq(calculated_key).all():
        raise ValueError("OOF evidence entry_key disagrees with image_id and label_name")
    result["entry_key"] = calculated_key
    if len(result) != ACTION_ENTRIES or result["entry_key"].duplicated().any():
        raise ValueError("OOF evidence must contain 14,400 unique label entries")
    expected_label_index = result["label_name"].map({name: i for i, name in enumerate(LABELS)})
    if expected_label_index.isna().any() or not result["label_index"].astype(int).eq(expected_label_index.astype(int)).all():
        raise ValueError("OOF evidence label_index disagrees with label_name")
    current = cohort_entries(cohort)[
        ["entry_key", "fold_id", "current_label"]
    ].rename(columns={"fold_id": "expected_fold_id"})
    result = result.merge(current, on="entry_key", validate="one_to_one")
    if not result["fold_id"].astype(int).eq(result["expected_fold_id"].astype(int)).all():
        raise ValueError("OOF evidence fold_id disagrees with the action cohort")
    if not result["noisy_label"].astype(int).eq(result["current_label"].astype(int)).all():
        raise ValueError("OOF evidence labels disagree with the current action cohort")
    probability = pd.to_numeric(result["oof_probability"], errors="coerce").to_numpy(dtype=float)
    confidence = pd.to_numeric(
        result["label_quality_self_confidence"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("OOF probabilities are not finite values within [0, 1]")
    if not np.isfinite(confidence).all() or not ((confidence >= 0) & (confidence <= 1)).all():
        raise ValueError("Label-quality self-confidence is not within [0, 1]")
    expected_confidence = np.where(
        result["current_label"].to_numpy(dtype=int) == 1,
        probability,
        1.0 - probability,
    )
    if not np.allclose(confidence, expected_confidence, rtol=1e-6, atol=2e-7):
        raise ValueError("Saved self-confidence disagrees with labels and OOF probabilities")
    if not result["cl_issue"].isin([0, 1, False, True]).all():
        raise ValueError("cl_issue must be binary")
    for column in ["self_confidence_suspicion", "cl_first_score"]:
        if not np.isfinite(pd.to_numeric(result[column], errors="coerce")).all():
            raise ValueError(f"OOF evidence contains non-finite {column}")
    result["cl_issue"] = result["cl_issue"].astype(bool)
    result = result.drop(columns=["expected_fold_id"])
    return result


def validate_oof_dir(cohort: pd.DataFrame, directory: Path) -> tuple[pd.DataFrame, Path]:
    marker = directory / OOF_MARKER
    evidence_path = directory / OOF_EVIDENCE
    if not marker.is_file():
        raise FileNotFoundError(f"OOF completion marker is missing: {marker}")
    if not evidence_path.is_file():
        raise FileNotFoundError(f"OOF entry evidence is missing: {evidence_path}")
    return validate_evidence(cohort, pd.read_csv(evidence_path)), evidence_path


def validate_frozen_thresholds(
    thresholds: pd.DataFrame, seed: int, expected_folds: Iterable[int]
) -> pd.DataFrame:
    require_columns(thresholds, ["fold_id", "threshold"], "frozen threshold table")
    result = thresholds.copy()
    if "seed" in result.columns:
        result = result.loc[result["seed"].astype(int).eq(int(seed))].copy()
    result["fold_id"] = pd.to_numeric(result["fold_id"], errors="raise").astype(int)
    result["threshold"] = pd.to_numeric(result["threshold"], errors="raise").astype(float)
    folds = sorted(int(value) for value in expected_folds)
    if len(result) != len(folds) or result["fold_id"].duplicated().any():
        raise ValueError("Frozen threshold table must contain one threshold per action fold")
    if sorted(result["fold_id"].tolist()) != folds:
        raise ValueError("Frozen threshold folds do not match the action cohort")
    if not result["threshold"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("Frozen thresholds must be within [0, 1]")
    result["seed"] = int(seed)
    return result[["seed", "fold_id", "threshold"]].sort_values("fold_id").reset_index(drop=True)


def true_state(cohort: pd.DataFrame, private: pd.DataFrame) -> dict[str, int | float]:
    entries = cohort_entries(cohort)[["entry_key", "current_label"]]
    truth = validate_private_reference(private)[["entry_key", "clean_label"]]
    joined = entries.merge(truth, on="entry_key", validate="one_to_one")
    incorrect = joined["current_label"].astype(int).ne(joined["clean_label"].astype(int))
    return {
        "remaining_errors": int(incorrect.sum()),
        "known_true_quality": float(1.0 - incorrect.mean()),
    }


def select_candidates(
    evidence: pd.DataFrame,
    reviewed_keys: set[str],
    top_fraction: float = TOP_FRACTION,
) -> tuple[pd.DataFrame, int]:
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be within (0, 1]")
    pool = evidence.loc[
        evidence["cl_issue"].astype(bool)
        & ~evidence["entry_key"].astype(str).isin(reviewed_keys)
    ].copy()
    selected_count = int(math.ceil(top_fraction * len(pool)))
    selected = pool.sort_values(
        ["cl_first_score", "self_confidence_suspicion", "entry_key"],
        ascending=[False, False, True],
        kind="stable",
    ).head(selected_count)
    if selected["entry_key"].duplicated().any() or selected["entry_key"].isin(reviewed_keys).any():
        raise RuntimeError("Selection contains duplicate or previously reviewed entries")
    return selected, int(len(pool))


def apply_oracle_labels(cohort: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    updated = cohort.copy()
    row_lookup = {
        str(image_id): index for index, image_id in enumerate(updated["image_id"].astype(str))
    }
    for row in selected.itertuples(index=False):
        index = row_lookup[str(row.image_id)]
        current = int(updated.at[index, str(row.label_name)])
        if current != int(row.current_label):
            raise ValueError(f"Selected entry changed before oracle review: {row.entry_key}")
        updated.at[index, str(row.label_name)] = int(row.clean_label)
    return validate_cohort(updated)


def official_uncalibrated_dqs(evidence: pd.DataFrame) -> float:
    try:
        from cleanlab.dataset import overall_label_health_score
    except ImportError as error:  # pragma: no cover - exercised in the formal environment
        raise RuntimeError("cleanlab is required to calculate the official uncalibrated DQS") from error
    labels = evidence["current_label"].to_numpy(dtype=int)
    probabilities = evidence["oof_probability"].to_numpy(dtype=float)
    return float(
        overall_label_health_score(
            labels=labels,
            pred_probs=np.column_stack([1.0 - probabilities, probabilities]),
            verbose=False,
        )
    )


def calibrated_dqs(evidence: pd.DataFrame, thresholds: pd.DataFrame) -> float:
    joined = evidence.merge(
        thresholds[["fold_id", "threshold"]],
        on="fold_id",
        how="left",
        validate="many_to_one",
    )
    if joined["threshold"].isna().any():
        raise ValueError("At least one action entry has no frozen fold threshold")
    return float(
        (
            joined["label_quality_self_confidence"].to_numpy(dtype=float)
            >= joined["threshold"].to_numpy(dtype=float)
        ).mean()
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _initial_manifest(output: Path) -> dict[str, Any]:
    marker = state_dir(output) / ".initialized"
    if not marker.is_file():
        raise FileNotFoundError(f"Initialization marker is missing: {marker}")
    manifest = _load_json(state_dir(output) / "initialization_manifest.json")
    if manifest.get("protocol") != PROTOCOL_NAME:
        raise RuntimeError("Output directory belongs to another protocol")
    return manifest


def _assert_seed(output: Path, seed: int) -> dict[str, Any]:
    manifest = _initial_manifest(output)
    if int(manifest["seed"]) != int(seed):
        raise ValueError("Seed differs from the initialized refinement state")
    return manifest


def current_cohort_path(output: Path, state_id: int) -> Path:
    if state_id == 0:
        return state_dir(output) / "initial_cohort_blind.csv"
    path = loop_dir(output, state_id) / "cohort_after_action_blind.csv"
    if not (loop_dir(output, state_id) / ".oracle_update_complete").is_file():
        raise FileNotFoundError(f"Oracle-updated state is incomplete at Loop {state_id}")
    return path


def evidence_manifest_path(output: Path, state_id: int) -> Path:
    if state_id == 0:
        return state_dir(output) / "initial_evidence_manifest.json"
    return loop_dir(output, state_id) / "post_action_evidence_manifest.json"


def evidence_for_state(output: Path, cohort: pd.DataFrame, state_id: int) -> pd.DataFrame:
    manifest_path = evidence_manifest_path(output, state_id)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Evidence manifest is missing for state {state_id}")
    manifest = _load_json(manifest_path)
    if int(manifest["state_id"]) != int(state_id):
        raise RuntimeError("Evidence manifest state id is inconsistent")
    directory = Path(manifest["oof_dir"])
    evidence, evidence_path = validate_oof_dir(cohort, directory)
    if sha256_file(evidence_path) != manifest["entry_evidence_sha256"]:
        raise RuntimeError(f"Frozen OOF evidence changed for state {state_id}")
    return evidence


def reviewed_keys_before(output: Path, loop_id: int) -> set[str]:
    _validate_loop_id(loop_id)
    frames: list[pd.DataFrame] = []
    for previous in range(1, loop_id):
        target = loop_dir(output, previous)
        if not (target / ".oracle_update_complete").is_file():
            raise FileNotFoundError(f"Prior oracle update is incomplete at Loop {previous}")
        frame = pd.read_csv(target / "review_history_blind.csv")
        frames.append(frame.loc[frame["first_review_loop"].astype(int).eq(previous)])
    if not frames:
        return set()
    reviewed = pd.concat(frames, ignore_index=True)
    if reviewed["entry_key"].duplicated().any():
        raise RuntimeError("An entry appears in more than one prior review loop")
    return set(reviewed["entry_key"].astype(str))


def _cumulative_history(output: Path, through_loop: int, private: bool) -> pd.DataFrame:
    suffix = "private" if private else "blind"
    frames: list[pd.DataFrame] = []
    for loop_id in range(1, through_loop + 1):
        path = loop_dir(output, loop_id) / f"selected_entries_{suffix}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame.insert(0, "first_review_loop", loop_id)
        frames.append(frame)
    if frames:
        result = pd.concat(frames, ignore_index=True)
    else:
        result = pd.DataFrame(columns=["first_review_loop", "entry_key"])
    if result["entry_key"].duplicated().any():
        raise RuntimeError("A reviewed entry re-entered a later loop")
    return result


def initialize(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    marker = state_dir(output) / ".initialized"
    if marker.is_file():
        manifest = _assert_seed(output, args.seed)
        checks = {
            Path(args.initial_cohort): manifest["source_cohort_sha256"],
            Path(args.private_reference): manifest["private_reference_sha256"],
            Path(args.thresholds): manifest["source_thresholds_sha256"],
            Path(args.initial_oof_dir) / OOF_EVIDENCE: manifest["initial_evidence_sha256"],
        }
        if any(not path.is_file() or sha256_file(path) != digest for path, digest in checks.items()):
            raise RuntimeError("Resume inputs differ from the frozen initialization inputs")
        print(json.dumps({**manifest, "resume_status": "already_complete"}, indent=2), flush=True)
        return
    if output.exists():
        raise FileExistsError(f"Uninitialized output directory already exists: {output}")

    cohort_path = Path(args.initial_cohort)
    private_path = Path(args.private_reference)
    thresholds_path = Path(args.thresholds)
    initial_oof = Path(args.initial_oof_dir)
    cohort = validate_cohort(pd.read_csv(cohort_path))
    private = validate_private_reference(pd.read_csv(private_path))
    entries = cohort_entries(cohort).merge(
        private[["entry_key", "noisy_label"]], on="entry_key", validate="one_to_one"
    )
    if not entries["current_label"].astype(int).eq(entries["noisy_label"].astype(int)).all():
        raise ValueError("Initial action cohort does not match private noisy labels")
    evidence, evidence_path = validate_oof_dir(cohort, initial_oof)
    thresholds = validate_frozen_thresholds(
        pd.read_csv(thresholds_path), args.seed, sorted(evidence["fold_id"].astype(int).unique())
    )

    output.mkdir(parents=True, exist_ok=False)
    state = state_dir(output)
    state.mkdir()
    cohort_snapshot = state / "initial_cohort_blind.csv"
    thresholds_snapshot = state / "frozen_fold_thresholds.csv"
    atomic_write_csv(cohort, cohort_snapshot)
    atomic_write_csv(thresholds, thresholds_snapshot)
    initial_evidence_manifest = {
        "protocol": PROTOCOL_NAME,
        "state_id": 0,
        "oof_dir": str(initial_oof.resolve()),
        "entry_evidence_sha256": sha256_file(evidence_path),
        "oof_marker_sha256": sha256_file(initial_oof / OOF_MARKER),
    }
    atomic_json(initial_evidence_manifest, state / "initial_evidence_manifest.json")
    initial_state = true_state(cohort, private)
    if int(initial_state["remaining_errors"]) != INITIAL_ERRORS or not np.isclose(
        float(initial_state["known_true_quality"]), INITIAL_QUALITY
    ):
        raise ValueError(
            "Initial action state must be the locked exact-r20 anchor "
            f"({INITIAL_ERRORS} errors, quality {INITIAL_QUALITY:.2f}); found {initial_state}"
        )
    manifest = {
        "protocol": PROTOCOL_NAME,
        "seed": int(args.seed),
        "action_images": ACTION_IMAGES,
        "action_entries": ACTION_ENTRIES,
        "folds": N_FOLDS,
        "loops": LOOPS,
        "top_fraction_of_current_unreviewed_issue_pool": TOP_FRACTION,
        "initial_errors": int(initial_state["remaining_errors"]),
        "initial_known_true_quality": float(initial_state["known_true_quality"]),
        "initial_issue_pool": int(evidence["cl_issue"].sum()),
        "source_cohort": str(cohort_path.resolve()),
        "source_cohort_sha256": sha256_file(cohort_path),
        "cohort_snapshot_sha256": sha256_file(cohort_snapshot),
        "private_reference_sha256": sha256_file(private_path),
        "source_thresholds_sha256": sha256_file(thresholds_path),
        "frozen_thresholds_sha256": sha256_file(thresholds_snapshot),
        "initial_evidence_sha256": sha256_file(evidence_path),
        "calibration_fitted_by_this_program": False,
        "program_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(manifest, state / "initialization_manifest.json")
    atomic_write_text(marker, "complete\n")
    print(json.dumps(manifest, indent=2), flush=True)


def select(args: argparse.Namespace) -> None:
    _validate_loop_id(args.loop_id)
    output = Path(args.output_dir)
    _assert_seed(output, args.seed)
    target = loop_dir(output, args.loop_id)
    marker = target / ".selection_complete"
    if marker.is_file():
        manifest = _load_json(target / "selection_manifest_blind.json")
        if sha256_file(target / "selected_entries_blind.csv") != manifest["selected_entries_sha256"]:
            raise RuntimeError("Frozen selection changed after completion")
        print(json.dumps({**manifest, "resume_status": "already_complete"}, indent=2), flush=True)
        return
    if target.exists():
        raise FileExistsError(f"Incomplete loop directory already exists: {target}")

    state_id = int(args.loop_id) - 1
    cohort_path = current_cohort_path(output, state_id)
    cohort = validate_cohort(pd.read_csv(cohort_path))
    evidence = evidence_for_state(output, cohort, state_id)
    reviewed = reviewed_keys_before(output, args.loop_id)
    selected, issue_pool = select_candidates(evidence, reviewed)
    target.mkdir(parents=True, exist_ok=False)
    columns = [
        "entry_key",
        "image_id",
        "fold_id",
        "label_index",
        "label_name",
        "current_label",
        "oof_probability",
        "label_quality_self_confidence",
        "cl_issue",
        "self_confidence_suspicion",
        "cl_first_score",
    ]
    selected_path = target / "selected_entries_blind.csv"
    atomic_write_csv(selected[columns], selected_path)
    manifest = {
        "protocol": PROTOCOL_NAME,
        "seed": int(args.seed),
        "loop": int(args.loop_id),
        "reviewed_before_loop": int(len(reviewed)),
        "current_unreviewed_issue_pool": int(issue_pool),
        "selected_entries": int(len(selected)),
        "top_fraction": TOP_FRACTION,
        "source_state": state_id,
        "source_cohort_sha256": sha256_file(cohort_path),
        "source_evidence_sha256": _load_json(evidence_manifest_path(output, state_id))[
            "entry_evidence_sha256"
        ],
        "selected_entries_sha256": sha256_file(selected_path),
        "private_reference_used": False,
    }
    atomic_json(manifest, target / "selection_manifest_blind.json")
    atomic_write_text(marker, "complete\n")
    print(json.dumps(manifest, indent=2), flush=True)


def oracle_update(args: argparse.Namespace) -> None:
    _validate_loop_id(args.loop_id)
    output = Path(args.output_dir)
    initialization = _assert_seed(output, args.seed)
    target = loop_dir(output, args.loop_id)
    if not (target / ".selection_complete").is_file():
        raise FileNotFoundError("Selection marker is missing")
    marker = target / ".oracle_update_complete"
    if marker.is_file():
        summary = _load_json(target / "oracle_summary_private.json")
        checks = {
            target / "cohort_after_action_blind.csv": summary["cohort_after_action_sha256"],
            target / "selected_entries_private.csv": summary["selected_entries_private_sha256"],
            target / "review_history_blind.csv": summary["review_history_blind_sha256"],
            target / "review_history_private.csv": summary["review_history_private_sha256"],
        }
        if any(sha256_file(path) != digest for path, digest in checks.items()):
            raise RuntimeError("Frozen oracle-update output changed after completion")
        print(json.dumps({**summary, "resume_status": "already_complete"}, indent=2), flush=True)
        return

    private_path = Path(args.private_reference)
    if sha256_file(private_path) != initialization["private_reference_sha256"]:
        raise RuntimeError("Private reference differs from the initialization reference")
    private = validate_private_reference(pd.read_csv(private_path))
    before_path = current_cohort_path(output, args.loop_id - 1)
    before = validate_cohort(pd.read_csv(before_path))
    selected_path = target / "selected_entries_blind.csv"
    selection_manifest = _load_json(target / "selection_manifest_blind.json")
    if sha256_file(selected_path) != selection_manifest["selected_entries_sha256"]:
        raise RuntimeError("Blind selection changed before oracle review")
    selected = pd.read_csv(selected_path)
    joined = selected.merge(
        private[["entry_key", "clean_label", "flip_direction"]],
        on="entry_key",
        how="left",
        validate="one_to_one",
    )
    if joined["clean_label"].isna().any():
        raise RuntimeError("One or more selected entries are absent from the private reference")
    joined["true_issue"] = joined["current_label"].astype(int).ne(
        joined["clean_label"].astype(int)
    ).astype(int)
    after = apply_oracle_labels(before, joined)
    before_state = true_state(before, private)
    after_state = true_state(after, private)
    corrected = int(joined["true_issue"].sum())
    if int(before_state["remaining_errors"]) - int(after_state["remaining_errors"]) != corrected:
        raise RuntimeError("Oracle review did not remove exactly the selected true errors")

    outputs = [
        target / "selected_entries_private.csv",
        target / "cohort_after_action_blind.csv",
        target / "review_history_blind.csv",
        target / "review_history_private.csv",
        target / "oracle_summary_private.json",
    ]
    _assert_absent(outputs, "oracle update")
    private_selected_path = target / "selected_entries_private.csv"
    cohort_after_path = target / "cohort_after_action_blind.csv"
    atomic_write_csv(joined, private_selected_path)
    atomic_write_csv(after, cohort_after_path)
    blind_history = _cumulative_history(output, args.loop_id, private=False)
    private_history = _cumulative_history(output, args.loop_id, private=True)
    blind_history_path = target / "review_history_blind.csv"
    private_history_path = target / "review_history_private.csv"
    atomic_write_csv(blind_history, blind_history_path)
    atomic_write_csv(private_history, private_history_path)
    summary = {
        "protocol": PROTOCOL_NAME,
        "seed": int(args.seed),
        "loop": int(args.loop_id),
        "selected_entries": int(len(joined)),
        "selected_true_errors": corrected,
        "selected_correct_labels_retained": int(len(joined) - corrected),
        "selection_precision": float(corrected / len(joined)) if len(joined) else None,
        "cumulative_reviewed_entries": int(len(blind_history)),
        "cumulative_corrected_errors": int(private_history["true_issue"].astype(int).sum()),
        "known_true_quality_before": float(before_state["known_true_quality"]),
        "known_true_quality_after": float(after_state["known_true_quality"]),
        "remaining_errors_before": int(before_state["remaining_errors"]),
        "remaining_errors_after": int(after_state["remaining_errors"]),
        "private_reference_sha256": sha256_file(private_path),
        "selected_entries_private_sha256": sha256_file(private_selected_path),
        "cohort_after_action_sha256": sha256_file(cohort_after_path),
        "review_history_blind_sha256": sha256_file(blind_history_path),
        "review_history_private_sha256": sha256_file(private_history_path),
    }
    atomic_json(summary, target / "oracle_summary_private.json")
    atomic_write_text(marker, "complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def finalize_loop(args: argparse.Namespace) -> None:
    _validate_loop_id(args.loop_id)
    output = Path(args.output_dir)
    _assert_seed(output, args.seed)
    target = loop_dir(output, args.loop_id)
    if not (target / ".oracle_update_complete").is_file():
        raise FileNotFoundError("Oracle-update marker is missing")
    marker = target / ".loop_complete"
    post_oof = Path(args.post_oof_dir)
    cohort_path = current_cohort_path(output, args.loop_id)
    cohort = validate_cohort(pd.read_csv(cohort_path))
    evidence, evidence_path = validate_oof_dir(cohort, post_oof)
    if marker.is_file():
        manifest = _load_json(target / "post_action_evidence_manifest.json")
        if str(post_oof.resolve()) != manifest["oof_dir"] or sha256_file(evidence_path) != manifest[
            "entry_evidence_sha256"
        ]:
            raise RuntimeError("Resume OOF directory differs from the frozen completed loop")
        print(json.dumps({**manifest, "resume_status": "already_complete"}, indent=2), flush=True)
        return

    manifest_path = target / "post_action_evidence_manifest.json"
    _assert_absent([manifest_path], "loop finalization")
    manifest = {
        "protocol": PROTOCOL_NAME,
        "seed": int(args.seed),
        "state_id": int(args.loop_id),
        "oof_dir": str(post_oof.resolve()),
        "cohort_sha256": sha256_file(cohort_path),
        "entry_evidence_sha256": sha256_file(evidence_path),
        "oof_marker_sha256": sha256_file(post_oof / OOF_MARKER),
        "cl_issue_entries": int(evidence["cl_issue"].sum()),
    }
    atomic_json(manifest, manifest_path)
    atomic_write_text(marker, "complete\n")
    print(json.dumps(manifest, indent=2), flush=True)


def _correlation(frame: pd.DataFrame, left: str, right: str, method: str) -> float | None:
    value = float(frame[[left, right]].corr(method=method).iloc[0, 1])
    return value if np.isfinite(value) else None


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def evaluate(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    initialization = _assert_seed(output, args.seed)
    marker = output / ".seed_evaluation_complete"
    if marker.is_file():
        summary = _load_json(output / "seed_evaluation_summary.json")
        if sha256_file(output / "dqs_trajectory_private.csv") != summary["trajectory_sha256"]:
            raise RuntimeError("Frozen evaluation trajectory changed after completion")
        print(json.dumps({**summary, "resume_status": "already_complete"}, indent=2), flush=True)
        return
    for loop_id in range(1, LOOPS + 1):
        if not (loop_dir(output, loop_id) / ".loop_complete").is_file():
            raise FileNotFoundError(f"Loop {loop_id} is incomplete")
    private_path = Path(args.private_reference)
    if sha256_file(private_path) != initialization["private_reference_sha256"]:
        raise RuntimeError("Private reference differs from the initialization reference")
    private = validate_private_reference(pd.read_csv(private_path))
    thresholds_path = state_dir(output) / "frozen_fold_thresholds.csv"
    if sha256_file(thresholds_path) != initialization["frozen_thresholds_sha256"]:
        raise RuntimeError("Frozen threshold snapshot changed after initialization")
    thresholds = validate_frozen_thresholds(
        pd.read_csv(thresholds_path), args.seed, range(N_FOLDS)
    )

    records: list[dict[str, Any]] = []
    for state_id in range(LOOPS + 1):
        cohort_path = current_cohort_path(output, state_id)
        cohort = validate_cohort(pd.read_csv(cohort_path))
        evidence = evidence_for_state(output, cohort, state_id)
        state = true_state(cohort, private)
        if state_id == 0:
            selected_entries = 0
            selected_true_errors = 0
            cumulative_reviews = 0
            cumulative_corrected = 0
        else:
            oracle = _load_json(loop_dir(output, state_id) / "oracle_summary_private.json")
            selected_entries = int(oracle["selected_entries"])
            selected_true_errors = int(oracle["selected_true_errors"])
            cumulative_reviews = int(oracle["cumulative_reviewed_entries"])
            cumulative_corrected = int(oracle["cumulative_corrected_errors"])
        uncalibrated = official_uncalibrated_dqs(evidence)
        calibrated = calibrated_dqs(evidence, thresholds)
        records.append(
            {
                "seed": int(args.seed),
                "loop": state_id,
                "selected_entries": selected_entries,
                "selected_true_errors": selected_true_errors,
                "cumulative_reviewed_entries": cumulative_reviews,
                "cumulative_corrected_errors": cumulative_corrected,
                "remaining_errors": int(state["remaining_errors"]),
                "known_true_quality": float(state["known_true_quality"]),
                "official_uncalibrated_dqs": uncalibrated,
                "calibrated_threshold_dqs": calibrated,
                "official_uncalibrated_absolute_error": abs(
                    uncalibrated - float(state["known_true_quality"])
                ),
                "calibrated_absolute_error": abs(
                    calibrated - float(state["known_true_quality"])
                ),
            }
        )
    trajectory = pd.DataFrame(records)
    trajectory_path = output / "dqs_trajectory_private.csv"
    summary_path = output / "seed_evaluation_summary.json"
    _assert_absent([trajectory_path, summary_path], "evaluation")
    atomic_write_csv(trajectory, trajectory_path)
    true_delta = np.diff(trajectory["known_true_quality"].to_numpy(dtype=float))
    uncalibrated_delta = np.diff(
        trajectory["official_uncalibrated_dqs"].to_numpy(dtype=float)
    )
    calibrated_delta = np.diff(trajectory["calibrated_threshold_dqs"].to_numpy(dtype=float))
    summary = {
        "protocol": PROTOCOL_NAME,
        "seed": int(args.seed),
        "loops": LOOPS,
        "initial_errors": int(initialization["initial_errors"]),
        "final_remaining_errors": int(trajectory.iloc[-1]["remaining_errors"]),
        "final_known_true_quality": float(trajectory.iloc[-1]["known_true_quality"]),
        "official_uncalibrated_mean_absolute_error": float(
            trajectory["official_uncalibrated_absolute_error"].mean()
        ),
        "calibrated_mean_absolute_error": float(trajectory["calibrated_absolute_error"].mean()),
        "official_uncalibrated_spearman": _correlation(
            trajectory, "known_true_quality", "official_uncalibrated_dqs", "spearman"
        ),
        "calibrated_spearman": _correlation(
            trajectory, "known_true_quality", "calibrated_threshold_dqs", "spearman"
        ),
        "official_uncalibrated_direction_agreement": int(
            (np.sign(true_delta) == np.sign(uncalibrated_delta)).sum()
        ),
        "calibrated_direction_agreement": int(
            (np.sign(true_delta) == np.sign(calibrated_delta)).sum()
        ),
        "direction_transitions": LOOPS,
        "private_reference_sha256": sha256_file(private_path),
        "frozen_thresholds_sha256": sha256_file(thresholds_path),
        "trajectory_sha256": sha256_file(trajectory_path),
        "calibration_fitted_by_this_program": False,
    }
    atomic_json(summary, summary_path)
    atomic_write_text(marker, "complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def exact_sign_flip_p(values: Iterable[float]) -> float:
    differences = np.asarray(list(values), dtype=float)
    differences = differences[np.isfinite(differences) & ~np.isclose(differences, 0.0)]
    if differences.size == 0:
        return 1.0
    observed = abs(float(differences.mean()))
    null = [
        abs(float(np.mean(differences * np.asarray(signs, dtype=float))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def aggregate(args: argparse.Namespace) -> None:
    experiment_root = Path(args.experiment_root)
    output = Path(args.aggregate_output)
    marker = output / ".aggregate_complete"
    seeds = parse_seeds(args.seeds)
    if marker.is_file():
        summary = _load_json(output / "aggregate_summary.json")
        checks = {
            output / "all_seed_dqs_trajectories_private.csv": summary["all_trajectories_sha256"],
            output / "seed_dqs_summary_private.csv": summary["seed_summary_sha256"],
            output / "loop_dqs_summary_private.csv": summary["loop_summary_sha256"],
        }
        if any(sha256_file(path) != digest for path, digest in checks.items()):
            raise RuntimeError("Frozen aggregate output changed after completion")
        if summary["seeds"] != seeds:
            raise RuntimeError("Requested seeds differ from the completed aggregate")
        print(json.dumps({**summary, "resume_status": "already_complete"}, indent=2), flush=True)
        return
    if output.exists():
        raise FileExistsError(f"Incomplete aggregate output directory already exists: {output}")

    trajectories: list[pd.DataFrame] = []
    seed_records: list[dict[str, Any]] = []
    for seed in seeds:
        seed_root = experiment_root / f"seed_{seed}"
        if not (seed_root / ".seed_evaluation_complete").is_file():
            raise FileNotFoundError(f"Seed evaluation is incomplete: {seed}")
        summary = _load_json(seed_root / "seed_evaluation_summary.json")
        if summary.get("protocol") != PROTOCOL_NAME or int(summary.get("seed", -1)) != seed:
            raise RuntimeError(f"Seed {seed} evaluation provenance is inconsistent")
        trajectory_path = seed_root / "dqs_trajectory_private.csv"
        if sha256_file(trajectory_path) != summary["trajectory_sha256"]:
            raise RuntimeError(f"Seed {seed} trajectory changed after evaluation")
        trajectory = pd.read_csv(trajectory_path)
        require_columns(
            trajectory,
            [
                "seed",
                "loop",
                "known_true_quality",
                "official_uncalibrated_dqs",
                "calibrated_threshold_dqs",
                "official_uncalibrated_absolute_error",
                "calibrated_absolute_error",
            ],
            f"seed {seed} DQS trajectory",
        )
        if len(trajectory) != LOOPS + 1 or trajectory["loop"].astype(int).tolist() != list(
            range(LOOPS + 1)
        ):
            raise RuntimeError(f"Seed {seed} does not contain the locked Loop 0-{LOOPS} trajectory")
        if not trajectory["seed"].astype(int).eq(seed).all():
            raise RuntimeError(f"Seed column is inconsistent for seed {seed}")
        trajectories.append(trajectory)
        seed_records.append(
            {
                "seed": seed,
                "official_uncalibrated_mae": float(
                    trajectory["official_uncalibrated_absolute_error"].mean()
                ),
                "calibrated_mae": float(trajectory["calibrated_absolute_error"].mean()),
                "mae_reduction": float(
                    trajectory["official_uncalibrated_absolute_error"].mean()
                    - trajectory["calibrated_absolute_error"].mean()
                ),
                "official_uncalibrated_spearman": summary["official_uncalibrated_spearman"],
                "calibrated_spearman": summary["calibrated_spearman"],
                "official_uncalibrated_direction_agreement": int(
                    summary["official_uncalibrated_direction_agreement"]
                ),
                "calibrated_direction_agreement": int(summary["calibrated_direction_agreement"]),
            }
        )

    all_trajectories = pd.concat(trajectories, ignore_index=True).sort_values(["seed", "loop"])
    seed_summary = pd.DataFrame(seed_records).sort_values("seed")
    loop_summary = (
        all_trajectories.groupby("loop", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            known_true_quality_mean=("known_true_quality", "mean"),
            known_true_quality_sd=("known_true_quality", "std"),
            official_uncalibrated_dqs_mean=("official_uncalibrated_dqs", "mean"),
            official_uncalibrated_dqs_sd=("official_uncalibrated_dqs", "std"),
            calibrated_dqs_mean=("calibrated_threshold_dqs", "mean"),
            calibrated_dqs_sd=("calibrated_threshold_dqs", "std"),
            official_uncalibrated_absolute_error_mean=(
                "official_uncalibrated_absolute_error",
                "mean",
            ),
            calibrated_absolute_error_mean=("calibrated_absolute_error", "mean"),
        )
        .sort_values("loop")
    )

    output.mkdir(parents=True, exist_ok=False)
    trajectories_path = output / "all_seed_dqs_trajectories_private.csv"
    seed_summary_path = output / "seed_dqs_summary_private.csv"
    loop_summary_path = output / "loop_dqs_summary_private.csv"
    atomic_write_csv(all_trajectories, trajectories_path)
    atomic_write_csv(seed_summary, seed_summary_path)
    atomic_write_csv(loop_summary, loop_summary_path)
    mae_reduction = seed_summary["mae_reduction"].to_numpy(dtype=float)
    summary = {
        "protocol": PROTOCOL_NAME,
        "seeds": seeds,
        "seed_count": len(seeds),
        "states_per_seed": LOOPS + 1,
        "official_uncalibrated_mae_mean_across_seeds": float(
            seed_summary["official_uncalibrated_mae"].mean()
        ),
        "official_uncalibrated_mae_sd_across_seeds": _finite_or_none(
            seed_summary["official_uncalibrated_mae"].std()
        ),
        "calibrated_mae_mean_across_seeds": float(seed_summary["calibrated_mae"].mean()),
        "calibrated_mae_sd_across_seeds": _finite_or_none(
            seed_summary["calibrated_mae"].std()
        ),
        "calibrated_mae_lower_seed_count": int((mae_reduction > 0).sum()),
        "mean_paired_mae_reduction": float(mae_reduction.mean()),
        "exact_paired_sign_flip_p": exact_sign_flip_p(mae_reduction),
        "all_trajectories_sha256": sha256_file(trajectories_path),
        "seed_summary_sha256": sha256_file(seed_summary_path),
        "loop_summary_sha256": sha256_file(loop_summary_path),
    }
    atomic_json(summary, output / "aggregate_summary.json")
    atomic_write_text(marker, "complete\n")
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    initialize_parser = subparsers.add_parser("initialize")
    initialize_parser.add_argument("--output-dir", type=Path, required=True)
    initialize_parser.add_argument("--initial-cohort", type=Path, required=True)
    initialize_parser.add_argument("--private-reference", type=Path, required=True)
    initialize_parser.add_argument("--initial-oof-dir", type=Path, required=True)
    initialize_parser.add_argument("--thresholds", type=Path, required=True)
    initialize_parser.add_argument("--seed", type=int, required=True)
    initialize_parser.set_defaults(function=initialize)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", type=Path, required=True)
    common.add_argument("--loop-id", type=int, required=True)
    common.add_argument("--seed", type=int, required=True)

    selection_parser = subparsers.add_parser("select", parents=[common])
    selection_parser.set_defaults(function=select)

    oracle_parser = subparsers.add_parser("oracle-update", parents=[common])
    oracle_parser.add_argument("--private-reference", type=Path, required=True)
    oracle_parser.set_defaults(function=oracle_update)

    finalizer = subparsers.add_parser("finalize-loop", parents=[common])
    finalizer.add_argument("--post-oof-dir", type=Path, required=True)
    finalizer.set_defaults(function=finalize_loop)

    evaluator = subparsers.add_parser("evaluate")
    evaluator.add_argument("--output-dir", type=Path, required=True)
    evaluator.add_argument("--private-reference", type=Path, required=True)
    evaluator.add_argument("--seed", type=int, required=True)
    evaluator.set_defaults(function=evaluate)

    aggregator = subparsers.add_parser("aggregate")
    aggregator.add_argument("--experiment-root", type=Path, required=True)
    aggregator.add_argument("--aggregate-output", type=Path, required=True)
    aggregator.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    aggregator.set_defaults(function=aggregate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
