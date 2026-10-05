import unittest

import numpy as np
import pandas as pd

from vindr_iterative_oracle_cleaning import (
    apply_oracle_labels,
    comparator_trajectory,
    select_candidates,
)


class VinDrIterativeOracleCleaningTests(unittest.TestCase):
    def test_selection_is_top_twenty_percent_of_unreviewed_hard_pool(self):
        evidence = pd.DataFrame(
            {
                "entry_key": [f"e{i}" for i in range(12)],
                "cl_issue": [1] * 10 + [0, 0],
                "cl_first_score": np.arange(12, dtype=float),
                "self_confidence_suspicion": np.arange(12, dtype=float),
            }
        )
        selected, pool = select_candidates(evidence, {"e9", "e8"}, 0.20)
        self.assertEqual(pool, 8)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected["entry_key"].tolist(), ["e7", "e6"])

    def test_selection_never_uses_non_hard_entries(self):
        evidence = pd.DataFrame(
            {
                "entry_key": ["hard", "soft"],
                "cl_issue": [1, 0],
                "cl_first_score": [2.1, 999.0],
                "self_confidence_suspicion": [0.1, 1.0],
            }
        )
        selected, pool = select_candidates(evidence, set(), 1.0)
        self.assertEqual(pool, 1)
        self.assertEqual(selected["entry_key"].tolist(), ["hard"])

    def test_selection_uses_ceiling_for_twenty_percent(self):
        evidence = pd.DataFrame(
            {
                "entry_key": [f"e{i}" for i in range(6)],
                "cl_issue": [1] * 6,
                "cl_first_score": np.arange(6, dtype=float),
                "self_confidence_suspicion": np.arange(6, dtype=float),
            }
        )
        selected, pool = select_candidates(evidence, set(), 0.20)
        self.assertEqual(pool, 6)
        self.assertEqual(len(selected), 2)

    def test_reviewed_entries_cannot_reenter_selection(self):
        evidence = pd.DataFrame(
            {
                "entry_key": ["reviewed", "new"],
                "cl_issue": [1, 1],
                "cl_first_score": [10.0, 1.0],
                "self_confidence_suspicion": [1.0, 0.1],
            }
        )
        selected, pool = select_candidates(evidence, {"reviewed"}, 1.0)
        self.assertEqual(pool, 1)
        self.assertEqual(selected["entry_key"].tolist(), ["new"])

    def test_oracle_changes_selected_label_to_clean_reference(self):
        cohort = pd.DataFrame(
            {
                "image_id": ["a"], "image_path": ["a.png"], "fold_id": [0],
                "Atelectasis": [1], "Cardiomegaly": [0], "Consolidation": [0],
                "Lung Opacity": [0], "Pleural effusion": [0], "Pneumonia": [0],
            }
        )
        selected = pd.DataFrame(
            {"entry_key": ["a::Atelectasis"], "image_id": ["a"],
             "label_name": ["Atelectasis"], "current_label": [1], "clean_label": [0]}
        )
        updated = apply_oracle_labels(cohort, selected)
        self.assertEqual(int(updated.loc[0, "Atelectasis"]), 0)
        self.assertEqual(int(updated.loc[0, "Cardiomegaly"]), 0)

    def test_comparator_trajectory_uses_fixed_error_denominator(self):
        frame = comparator_trajectory(np.array([1, 0, 1, 0]), [1, 2, 4], "test")
        self.assertEqual(frame["cumulative_true_issues"].tolist(), [1, 1, 2])
        self.assertAlmostEqual(float(frame.iloc[-1]["cumulative_recall"]), 2 / 3600)


if __name__ == "__main__":
    unittest.main()
