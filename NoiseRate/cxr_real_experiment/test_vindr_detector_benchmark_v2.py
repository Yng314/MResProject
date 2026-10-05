#!/usr/bin/env python3
"""Focused tests for the VinDr detector benchmark v2."""

import unittest

import numpy as np
import pandas as pd

from vindr_detector_benchmark_v2 import (
    active_label_cleaning_score,
    detector_metrics,
    fine_gmm_suspicion,
)


class DetectorBenchmarkV2Tests(unittest.TestCase):
    def test_alc_prioritises_confident_disagreement(self) -> None:
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.01, 0.90, 0.99, 0.10])
        scores = active_label_cleaning_score(labels, probabilities)
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[3], scores[2])

    def test_fine_gmm_detects_low_alignment_points(self) -> None:
        rng = np.random.default_rng(7)
        clean_zero = np.column_stack([np.full(80, 4.0), rng.normal(0, 0.05, (80, 3))])
        noisy_zero = np.column_stack([rng.normal(0, 0.05, (20, 3)), np.full(20, 4.0)])
        clean_one = np.column_stack([rng.normal(0, 0.05, (80, 1)), np.full(80, 4.0), rng.normal(0, 0.05, (80, 2))])
        noisy_one = np.column_stack([np.full(20, 4.0), rng.normal(0, 0.05, (20, 3))])
        features = np.vstack([clean_zero, noisy_zero, clean_one, noisy_one])
        labels = np.zeros((200, 1), dtype=int)
        labels[100:] = 1
        scores = fine_gmm_suspicion(features, labels, random_state=3)[:, 0]
        self.assertGreater(scores[80:100].mean(), scores[:80].mean())
        self.assertGreater(scores[180:].mean(), scores[100:180].mean())

    def test_detector_metrics_uses_exact_matched_budget(self) -> None:
        frame = pd.DataFrame(
            {
                "image_id": ["a", "b", "c", "d"],
                "label_name": ["x"] * 4,
                "injected_error": [1, 0, 1, 0],
                "score": [0.9, 0.1, 0.8, 0.2],
            }
        )
        result = detector_metrics(frame, "test", "score", 2, {}, "overall")
        self.assertEqual(result["errors_found"], 2)
        self.assertEqual(result["reviewed_entries"], 2)
        self.assertAlmostEqual(result["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
