#!/usr/bin/env python3

import argparse
import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from vindr_top20_calibrated_refinement import (
    ACTION_ENTRIES,
    ACTION_IMAGES,
    LABELS,
    aggregate,
    apply_oracle_labels,
    calibrated_dqs,
    evaluate,
    exact_sign_flip_p,
    finalize_loop,
    initialize,
    oracle_update,
    official_uncalibrated_dqs,
    select,
    select_candidates,
)


def write_oof_for_cohort(cohort_path: Path, oof_dir: Path, issue_count: int = 10) -> None:
    cohort = pd.read_csv(cohort_path)
    evidence_records: list[dict[str, object]] = []
    for label_index, label in enumerate(LABELS):
        for row_index, image_id in enumerate(cohort["image_id"].astype(str)):
            current = int(cohort.at[row_index, label])
            probability = 0.8 if current else 0.2
            issue = len(evidence_records) < issue_count
            suspicion = 1.0 - (probability if current else 1.0 - probability)
            evidence_records.append(
                {
                    "image_id": image_id,
                    "fold_id": int(cohort.at[row_index, "fold_id"]),
                    "label_index": label_index,
                    "label_name": label,
                    "noisy_label": current,
                    "oof_probability": probability,
                    "label_quality_self_confidence": probability if current else 1.0 - probability,
                    "cl_issue": int(issue),
                    "self_confidence_suspicion": suspicion,
                    "cl_first_score": (2.0 if issue else 0.0) + suspicion,
                }
            )
    oof_dir.mkdir()
    pd.DataFrame(evidence_records).to_csv(oof_dir / "entry_evidence.csv", index=False)
    (oof_dir / ".blind_run_complete").write_text("complete\n", encoding="utf-8")


def synthetic_locked_inputs(root: Path, seed: int = 13) -> tuple[Path, Path, Path, Path]:
    image_ids = [f"image-{index:04d}" for index in range(ACTION_IMAGES)]
    cohort = pd.DataFrame(
        {
            "image_id": image_ids,
            "image_path": [f"/images/{value}.png" for value in image_ids],
            "fold_id": np.arange(ACTION_IMAGES) % 4,
        }
    )
    for label in LABELS:
        cohort[label] = 0

    entries: list[dict[str, object]] = []
    entry_index = 0
    for label in LABELS:
        for row_index, image_id in enumerate(image_ids):
            injected = int(entry_index < 2_880)
            cohort.at[row_index, label] = injected
            entries.append(
                {
                    "image_id": image_id,
                    "label_name": label,
                    "clean_label": 0,
                    "noisy_label": injected,
                    "injected_error": injected,
                    "flip_direction": "0_to_1" if injected else "none",
                }
            )
            entry_index += 1
    private = pd.DataFrame(entries)

    cohort_path = root / "cohort.csv"
    private_path = root / "private.csv"
    thresholds_path = root / "thresholds.csv"
    oof_dir = root / "loop0_oof"
    cohort.to_csv(cohort_path, index=False)
    private.to_csv(private_path, index=False)
    write_oof_for_cohort(cohort_path, oof_dir)
    pd.DataFrame(
        {"seed": seed, "fold_id": [0, 1, 2, 3], "threshold": [0.5] * 4}
    ).to_csv(thresholds_path, index=False)
    return cohort_path, private_path, thresholds_path, oof_dir


class VinDrTop20CalibratedRefinementTests(unittest.TestCase):
    def test_selection_uses_ceiling_and_excludes_reviewed_issues(self) -> None:
        evidence = pd.DataFrame(
            {
                "entry_key": [f"e{i}" for i in range(12)],
                "cl_issue": [1] * 10 + [0, 0],
                "cl_first_score": [1.0] * 12,
                "self_confidence_suspicion": np.arange(12, dtype=float),
            }
        )
        selected, pool_size = select_candidates(evidence, {"e9", "e8"})
        self.assertEqual(pool_size, 8)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected["entry_key"].tolist(), ["e7", "e6"])

    def test_selection_tie_breaks_by_entry_key(self) -> None:
        evidence = pd.DataFrame(
            {
                "entry_key": ["c", "a", "b"],
                "cl_issue": [1, 1, 1],
                "cl_first_score": [2.1, 2.1, 2.1],
                "self_confidence_suspicion": [0.1, 0.1, 0.1],
            }
        )
        selected, pool_size = select_candidates(evidence, set(), top_fraction=0.34)
        self.assertEqual(pool_size, 3)
        self.assertEqual(selected["entry_key"].tolist(), ["a", "b"])

    def test_oracle_corrects_errors_and_retains_selected_correct_labels(self) -> None:
        image_ids = [f"i{index}" for index in range(ACTION_IMAGES)]
        cohort = pd.DataFrame(
            {
                "image_id": image_ids,
                "image_path": [f"{value}.png" for value in image_ids],
                "fold_id": np.arange(ACTION_IMAGES) % 4,
            }
        )
        for label in LABELS:
            cohort[label] = 0
        cohort.at[0, LABELS[0]] = 1
        selected = pd.DataFrame(
            {
                "entry_key": [f"{image_ids[0]}::{LABELS[0]}", f"{image_ids[1]}::{LABELS[0]}"],
                "image_id": image_ids[:2],
                "label_name": [LABELS[0], LABELS[0]],
                "current_label": [1, 0],
                "clean_label": [0, 0],
            }
        )
        updated = apply_oracle_labels(cohort, selected)
        self.assertEqual(int(updated.at[0, LABELS[0]]), 0)
        self.assertEqual(int(updated.at[1, LABELS[0]]), 0)
        self.assertTrue(updated[LABELS[1:]].eq(0).all().all())

    def test_calibrated_dqs_applies_fold_specific_frozen_thresholds(self) -> None:
        evidence = pd.DataFrame(
            {
                "fold_id": [0, 0, 1, 1],
                "label_quality_self_confidence": [0.7, 0.5, 0.8, 0.3],
            }
        )
        thresholds = pd.DataFrame({"fold_id": [0, 1], "threshold": [0.6, 0.75]})
        self.assertEqual(calibrated_dqs(evidence, thresholds), 0.5)

    def test_official_dqs_flattens_binary_entries_for_cleanlab(self) -> None:
        captured: dict[str, np.ndarray] = {}
        fake_dataset = types.ModuleType("cleanlab.dataset")

        def fake_health(*, labels, pred_probs, verbose):
            captured["labels"] = labels
            captured["pred_probs"] = pred_probs
            self.assertFalse(verbose)
            return 0.875

        fake_dataset.overall_label_health_score = fake_health
        fake_cleanlab = types.ModuleType("cleanlab")
        fake_cleanlab.dataset = fake_dataset
        evidence = pd.DataFrame(
            {"current_label": [0, 1], "oof_probability": [0.2, 0.9]}
        )
        with patch.dict(sys.modules, {"cleanlab": fake_cleanlab, "cleanlab.dataset": fake_dataset}):
            score = official_uncalibrated_dqs(evidence)
        self.assertEqual(score, 0.875)
        np.testing.assert_array_equal(captured["labels"], np.array([0, 1]))
        np.testing.assert_allclose(captured["pred_probs"], [[0.8, 0.2], [0.1, 0.9]])

    def test_initialize_and_select_are_resumable_with_complete_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cohort, private, thresholds, oof = synthetic_locked_inputs(root)
            output = root / "run" / "seed_13"
            init_args = argparse.Namespace(
                output_dir=output,
                initial_cohort=cohort,
                private_reference=private,
                initial_oof_dir=oof,
                thresholds=thresholds,
                seed=13,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                initialize(init_args)
                initialize(init_args)
            self.assertTrue((output / "state" / ".initialized").is_file())

            select_args = argparse.Namespace(output_dir=output, loop_id=1, seed=13)
            with contextlib.redirect_stdout(io.StringIO()):
                select(select_args)
                select(select_args)
            selected = pd.read_csv(output / "loop_01" / "selected_entries_blind.csv")
            self.assertEqual(len(selected), 2)
            self.assertTrue((output / "loop_01" / ".selection_complete").is_file())

    def test_complete_five_loop_state_and_aggregate_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cohort, private, thresholds, oof = synthetic_locked_inputs(root)
            experiment_root = root / "experiment"
            output = experiment_root / "seed_13"
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink):
                initialize(
                    argparse.Namespace(
                        output_dir=output,
                        initial_cohort=cohort,
                        private_reference=private,
                        initial_oof_dir=oof,
                        thresholds=thresholds,
                        seed=13,
                    )
                )
                for loop_id in range(1, 6):
                    select(argparse.Namespace(output_dir=output, loop_id=loop_id, seed=13))
                    oracle_update(
                        argparse.Namespace(
                            output_dir=output,
                            loop_id=loop_id,
                            private_reference=private,
                            seed=13,
                        )
                    )
                    post_oof = root / f"post_oof_{loop_id}"
                    write_oof_for_cohort(
                        output / f"loop_{loop_id:02d}" / "cohort_after_action_blind.csv",
                        post_oof,
                        issue_count=30,
                    )
                    finalize_loop(
                        argparse.Namespace(
                            output_dir=output,
                            loop_id=loop_id,
                            post_oof_dir=post_oof,
                            seed=13,
                        )
                    )

                fake_dataset = types.ModuleType("cleanlab.dataset")
                fake_dataset.overall_label_health_score = (
                    lambda *, labels, pred_probs, verbose: float(
                        np.where(labels == 1, pred_probs[:, 1], pred_probs[:, 0]).mean()
                    )
                )
                fake_cleanlab = types.ModuleType("cleanlab")
                fake_cleanlab.dataset = fake_dataset
                with patch.dict(
                    sys.modules, {"cleanlab": fake_cleanlab, "cleanlab.dataset": fake_dataset}
                ):
                    evaluate(
                        argparse.Namespace(
                            output_dir=output,
                            private_reference=private,
                            seed=13,
                        )
                    )
                aggregate_output = root / "aggregate"
                aggregate(
                    argparse.Namespace(
                        experiment_root=experiment_root,
                        aggregate_output=aggregate_output,
                        seeds="13",
                    )
                )
                aggregate(
                    argparse.Namespace(
                        experiment_root=experiment_root,
                        aggregate_output=aggregate_output,
                        seeds="13",
                    )
                )

            trajectory = pd.read_csv(output / "dqs_trajectory_private.csv")
            self.assertEqual(trajectory["loop"].tolist(), list(range(6)))
            histories = pd.read_csv(output / "loop_05" / "review_history_blind.csv")
            self.assertFalse(histories["entry_key"].duplicated().any())
            self.assertTrue((output / ".seed_evaluation_complete").is_file())
            self.assertTrue((aggregate_output / ".aggregate_complete").is_file())

    def test_exact_sign_flip_test_is_two_sided(self) -> None:
        self.assertEqual(exact_sign_flip_p([1.0, 1.0]), 0.5)
        self.assertEqual(exact_sign_flip_p([0.0, 0.0]), 1.0)


if __name__ == "__main__":
    unittest.main()
