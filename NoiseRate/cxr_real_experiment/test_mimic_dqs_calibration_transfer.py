#!/usr/bin/env python3

import unittest

import numpy as np
import pandas as pd

from mimic_dqs_calibration_transfer import (
    select_prevalence_matching_threshold,
    self_confidence,
    threshold_dqs,
)


class MimicDqsCalibrationTransferTests(unittest.TestCase):
    def test_prevalence_matching_and_high_threshold_tie_break(self) -> None:
        scores = np.asarray([0.9, 0.8, 0.7, 0.6])
        correct = np.asarray([1, 1, 0, 0])
        threshold, quality, count = select_prevalence_matching_threshold(scores, correct)
        self.assertEqual(threshold, 0.8)
        self.assertEqual(count, 2)
        self.assertEqual(quality, 0.5)

    def test_self_confidence_uses_probability_of_existing_label(self) -> None:
        observed = self_confidence(np.asarray([1, 0]), np.asarray([0.8, 0.3]))
        np.testing.assert_allclose(observed, np.asarray([0.8, 0.7]))

    def test_threshold_dqs_uses_fold_thresholds_and_current_valid_entries(self) -> None:
        labels = np.asarray([[1, 0], [1, 0]], dtype=float)
        valid = np.asarray([[True, True], [True, False]])
        probabilities = np.asarray([[0.8, 0.3], [0.6, 0.9]])
        fold = np.asarray([1, 2])
        thresholds = pd.DataFrame(
            {"fold_id": [1, 2, 3, 4], "threshold": [0.75, 0.55, 0.5, 0.5]}
        )
        self.assertAlmostEqual(
            threshold_dqs(labels, valid, probabilities, fold, thresholds), 2.0 / 3.0
        )


if __name__ == "__main__":
    unittest.main()
