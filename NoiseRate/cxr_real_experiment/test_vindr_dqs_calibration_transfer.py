#!/usr/bin/env python3

import unittest

import numpy as np

from vindr_dqs_calibration_transfer import (
    select_prevalence_matching_threshold,
    self_confidence,
)


class VinDrDQSCalibrationTransferTests(unittest.TestCase):
    def test_self_confidence_uses_probability_of_observed_label(self):
        labels = np.array([1, 0, 1, 0])
        probabilities = np.array([0.8, 0.8, 0.2, 0.2])
        result = self_confidence(labels, probabilities)
        np.testing.assert_allclose(result, np.array([0.8, 0.2, 0.2, 0.8]))

    def test_threshold_matches_known_correct_mass(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        correct = np.array([1, 1, 1, 0, 0])
        threshold, quality, count = select_prevalence_matching_threshold(scores, correct)
        self.assertEqual(count, 3)
        self.assertAlmostEqual(quality, 0.6)
        self.assertEqual(int((scores >= threshold).sum()), 3)

    def test_threshold_tie_is_deterministic_and_conservative(self):
        scores = np.array([0.9, 0.7, 0.7, 0.2])
        correct = np.array([1, 1, 0, 0])
        first = select_prevalence_matching_threshold(scores, correct)
        second = select_prevalence_matching_threshold(scores, correct)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 0.9)
        self.assertEqual(first[2], 1)


if __name__ == "__main__":
    unittest.main()
