#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from vindr_top20_calibrated_prepare import (
    ACTION_ENTRIES,
    ACTION_ERRORS,
    ACTION_IMAGES,
    LABELS,
    LOCKED_SEEDS,
    N_SPLITS,
    NOISE_RATE,
    PROBABILITY_COLUMNS,
    PROTOCOL_NAME,
    SCENARIO,
    SCORE_PROTOCOL_NAME,
    SENTINEL_ENTRIES,
    SENTINEL_ERRORS,
    SENTINEL_IMAGES,
    atomic_write_csv,
    atomic_write_text,
    calibrate,
    deterministic_split,
    make_corruption,
    materialize_evidence,
    prepare,
    select_prevalence_matching_threshold,
    sha256_file,
)


def clean_matrix(n_images: int) -> np.ndarray:
    row = np.arange(n_images)[:, None]
    column = np.arange(len(LABELS))[None, :]
    return ((row + column) % (column + 2) == 0).astype(np.int64)


def blind_frame(image_ids: list[str], labels: np.ndarray, sentinel: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "image_id": image_ids,
            "image_path": [f"{image_id}.png" for image_id in image_ids],
            "fold_id": -1 if sentinel else np.arange(len(image_ids)) % N_SPLITS,
        }
    )
    for label_index, label in enumerate(LABELS):
        frame[label] = labels[:, label_index]
    return frame


def private_frame(image_ids: list[str], clean: np.ndarray, noisy: np.ndarray) -> pd.DataFrame:
    rows = []
    for row_index, image_id in enumerate(image_ids):
        for label_index, label in enumerate(LABELS):
            is_error = int(clean[row_index, label_index] != noisy[row_index, label_index])
            rows.append(
                {
                    "image_id": image_id,
                    "label_name": label,
                    "clean_label": int(clean[row_index, label_index]),
                    "noisy_label": int(noisy[row_index, label_index]),
                    "injected_error": is_error,
                    "flip_direction": (
                        "none" if not is_error
                        else "0_to_1" if clean[row_index, label_index] == 0 else "1_to_0"
                    ),
                }
            )
    return pd.DataFrame(rows)


class VinDrTop20CalibratedPrepareTests(unittest.TestCase):
    def test_split_is_shared_and_order_invariant(self):
        ids = [f"image_{index:04d}" for index in range(3_000)]
        first = deterministic_split(ids)
        second = deterministic_split(reversed(ids))
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), ACTION_IMAGES)
        self.assertEqual(len(first[1]), SENTINEL_IMAGES)
        self.assertFalse(set(first[0]) & set(first[1]))

    def test_symmetric_corruption_is_exact_for_both_cohorts(self):
        for n_images, expected_errors in (
            (ACTION_IMAGES, ACTION_ERRORS),
            (SENTINEL_IMAGES, SENTINEL_ERRORS),
        ):
            clean = clean_matrix(n_images)
            noisy, injected, direction, counts = make_corruption(clean, 13, SCENARIO)
            self.assertEqual(int(injected.sum()), expected_errors)
            self.assertTrue((counts["injected_errors"] == int(NOISE_RATE * n_images)).all())
            self.assertTrue(np.array_equal(noisy != clean, injected))
            self.assertTrue(np.all(direction[injected] != "none"))

    def test_threshold_tie_uses_higher_threshold(self):
        scores = np.array([0.9, 0.7, 0.7, 0.2])
        correct = np.array([1, 1, 0, 0])
        threshold, _, count = select_prevalence_matching_threshold(scores, correct)
        self.assertEqual(threshold, 0.9)
        self.assertEqual(count, 1)

    def _write_score_and_prepared(self, root: Path, seed: int) -> tuple[Path, Path]:
        prepared = root / "prepared" / f"seed_{seed}" / "prepared"
        score = root / "scores" / f"seed_{seed}"
        prepared.mkdir(parents=True)
        score.mkdir(parents=True)
        action_ids = [f"action_{index:04d}" for index in range(ACTION_IMAGES)]
        sentinel_ids = [f"sentinel_{index:04d}" for index in range(SENTINEL_IMAGES)]
        action_labels = clean_matrix(ACTION_IMAGES)
        sentinel_clean = clean_matrix(SENTINEL_IMAGES)
        sentinel_noisy, _, _, _ = make_corruption(sentinel_clean, seed, SCENARIO)
        action = blind_frame(action_ids, action_labels)
        sentinel = blind_frame(sentinel_ids, sentinel_noisy, sentinel=True)
        sentinel_private = private_frame(sentinel_ids, sentinel_clean, sentinel_noisy)
        action_path = prepared / "blind_noisy_cohort.csv"
        sentinel_path = prepared / "sentinel_blind_cohort.csv"
        sentinel_private_path = prepared / "sentinel_private_reference.csv"
        atomic_write_csv(action, action_path)
        atomic_write_csv(sentinel, sentinel_path)
        atomic_write_csv(sentinel_private, sentinel_private_path)
        atomic_write_text(
            prepared / "prepare_summary.json",
            json.dumps(
                {
                    "protocol": PROTOCOL_NAME,
                    "seed": seed,
                    "blind_sha256": sha256_file(action_path),
                    "sentinel_blind_sha256": sha256_file(sentinel_path),
                    "sentinel_private_sha256": sha256_file(sentinel_private_path),
                }
            ),
        )
        atomic_write_text(prepared / ".prepare_complete", "complete\n")

        probabilities = np.linspace(0.01, 0.99, SENTINEL_IMAGES * len(LABELS)).reshape(
            SENTINEL_IMAGES, len(LABELS)
        )
        sentinel_parts = []
        for fold_id in range(N_SPLITS):
            part = pd.DataFrame({"image_id": sentinel_ids, "model_fold_id": fold_id})
            shifted = np.clip(probabilities + fold_id * 0.001, 0.0, 1.0)
            for label_index, column in enumerate(PROBABILITY_COLUMNS):
                part[column] = shifted[:, label_index]
            sentinel_parts.append(part)
        sentinel_folds = pd.concat(sentinel_parts, ignore_index=True)
        sentinel_folds_path = score / "sentinel_fold_predictions.csv"
        atomic_write_csv(sentinel_folds, sentinel_folds_path)
        action_oof = action[["image_id", "fold_id"]].copy()
        for label_index, column in enumerate(PROBABILITY_COLUMNS):
            action_oof[column] = 0.2 + label_index * 0.1
        action_oof_path = score / "action_oof_predictions.csv"
        atomic_write_csv(action_oof, action_oof_path)
        score_summary = {
            "protocol": SCORE_PROTOCOL_NAME,
            "seed": seed,
            "action_samples": ACTION_IMAGES,
            "action_entries": ACTION_ENTRIES,
            "sentinel_samples": SENTINEL_IMAGES,
            "sentinel_entries_per_fold_model": SENTINEL_ENTRIES,
            "n_splits": N_SPLITS,
            "blind_cohort_sha256": sha256_file(action_path),
            "sentinel_cohort_sha256": sha256_file(sentinel_path),
            "action_oof_sha256": sha256_file(action_oof_path),
            "sentinel_folds_sha256": sha256_file(sentinel_folds_path),
            "program_sha256": "score-program-hash",
            "source_helper_sha256": {"helper.py": "helper-hash"},
        }
        atomic_write_text(score / "score_summary.json", json.dumps(score_summary))
        atomic_write_text(score / ".score_complete", "complete\n")
        return prepared, score

    def test_calibrate_freezes_all_32_seed_fold_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in LOCKED_SEEDS:
                self._write_score_and_prepared(root, seed)
            output = root / "calibration"
            calibrate(
                argparse.Namespace(
                    output_dir=output,
                    score_root=root / "scores",
                    prepared_root=root / "prepared",
                    seeds=",".join(map(str, LOCKED_SEEDS)),
                )
            )
            thresholds = pd.read_csv(output / "calibration_thresholds_private.csv")
            self.assertEqual(len(thresholds), len(LOCKED_SEEDS) * N_SPLITS)
            self.assertFalse(thresholds[["seed", "fold_id"]].duplicated().any())
            frozen_hash = (output / ".calibration_frozen").read_text().strip()
            self.assertEqual(frozen_hash, sha256_file(output / "calibration_thresholds_private.csv"))
            manifest = json.loads((output / "calibration_manifest_private.json").read_text())
            self.assertFalse(manifest["action_private_reference_accessed"])

    def test_materialize_evidence_preserves_source_score_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, score = self._write_score_and_prepared(root, LOCKED_SEEDS[0])
            blind = pd.read_csv(prepared / "blind_noisy_cohort.csv")

            def fake_build_evidence(frame, labels, probabilities):
                oof = frame[["image_id", "fold_id"]].copy()
                for label_index, column in enumerate(PROBABILITY_COLUMNS):
                    oof[column] = probabilities[:, label_index]
                entries = []
                for row_index, row in frame.iterrows():
                    for label_index, label in enumerate(LABELS):
                        entries.append(
                            {
                                "image_id": row["image_id"],
                                "fold_id": int(row["fold_id"]),
                                "label_index": label_index,
                                "label_name": label,
                                "noisy_label": int(labels[row_index, label_index]),
                                "oof_probability": float(probabilities[row_index, label_index]),
                                "cl_issue": int((row_index + label_index) % 7 == 0),
                            }
                        )
                issue = np.array(
                    [[(row + label) % 7 == 0 for label in range(len(LABELS))] for row in range(len(frame))]
                )
                return oof, pd.DataFrame(entries), issue

            fake_module = types.SimpleNamespace(build_evidence=fake_build_evidence)
            output = root / "evidence"
            with patch.dict(sys.modules, {"vindr_mobilenet_oof": fake_module}):
                materialize_evidence(
                    argparse.Namespace(
                        output_dir=output,
                        blind_cohort=prepared / "blind_noisy_cohort.csv",
                        sentinel_cohort=prepared / "sentinel_blind_cohort.csv",
                        score_dir=score,
                        seed=LOCKED_SEEDS[0],
                    )
                )
            summary = json.loads((output / "blind_run_summary.json").read_text())
            source = json.loads((score / "score_summary.json").read_text())
            self.assertEqual(summary["source_action_oof_sha256"], source["action_oof_sha256"])
            self.assertEqual(summary["source_sentinel_folds_sha256"], source["sentinel_folds_sha256"])
            self.assertTrue((output / ".blind_run_complete").is_file())
            self.assertEqual(len(blind) * len(LABELS), len(pd.read_csv(output / "entry_evidence.csv")))


if __name__ == "__main__":
    unittest.main()
