#!/usr/bin/env python3
"""Focused unit tests for detector-alternative scoring."""

import unittest

import numpy as np
import pandas as pd

from vindr_detector_alternatives_benchmark import (
    active_label_cleaning_score,
    class_conditional_rank,
    simifeat_style_scores,
)


class DetectorAlternativeTests(unittest.TestCase):
    def test_alc_prioritises_confident_disagreement(self) -> None:
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.01, 0.90, 0.99, 0.10])
        scores = active_label_cleaning_score(labels, probabilities)
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[3], scores[2])

    def test_simifeat_style_prioritises_neighbour_disagreement(self) -> None:
        noisy = np.zeros((4, 6), dtype=int)
        noisy[:, 0] = [0, 0, 0, 1]
        neighbours = np.array(
            [
                [1, 2],
                [0, 2],
                [0, 1],
                [0, 1],
            ]
        )
        distances = np.full((4, 2), 0.1, dtype=float)
        score = simifeat_style_scores(noisy, neighbours, distances, [2])[2]
        self.assertGreater(score[3, 0], score[0, 0])

    def test_class_conditional_rank_does_not_mix_groups(self) -> None:
        frame = pd.DataFrame(
            {
                "label_name": ["A", "A", "B", "B"],
                "noisy_label": [0, 0, 1, 1],
                "score": [1.0, 2.0, 100.0, 200.0],
            }
        )
        ranks = class_conditional_rank(frame, "score").to_numpy()
        np.testing.assert_allclose(ranks, [0.5, 1.0, 0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
