#!/usr/bin/env python3

import unittest

import numpy as np

from mimic_dqs_calibration_score import flattened_dqs


class TestMimicDqsCalibrationScore(unittest.TestCase):
    def test_flattened_dqs_ignores_masked_entries(self) -> None:
        labels = np.asarray([[1.0, np.nan], [0.0, 1.0]])
        valid = np.asarray([[True, False], [True, True]])
        probs = np.asarray([[0.9, 0.0], [0.1, 0.8]])
        self.assertAlmostEqual(flattened_dqs(labels, valid, probs), 1.0)

    def test_flattened_dqs_is_bounded(self) -> None:
        labels = np.asarray([[1.0], [0.0], [1.0], [0.0]])
        valid = np.ones_like(labels, dtype=bool)
        probs = np.asarray([[0.7], [0.4], [0.6], [0.2]])
        score = flattened_dqs(labels, valid, probs)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
