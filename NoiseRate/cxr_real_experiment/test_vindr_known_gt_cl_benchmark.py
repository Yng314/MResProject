import unittest

import numpy as np
import pandas as pd

from vindr_known_gt_cl_benchmark import (
    binary_metrics,
    build_inner_split,
    exact_sign_flip_p,
    holm_adjust,
    parse_seeds,
    top_budget_mask,
)


class VinDrKnownGTBenchmarkTests(unittest.TestCase):
    def test_parse_seeds_rejects_duplicates(self):
        self.assertEqual(parse_seeds("13,42,97,123"), [13, 42, 97, 123])
        with self.assertRaises(ValueError):
            parse_seeds("13,13")

    def test_top_budget_mask_uses_deterministic_key_ties(self):
        frame = pd.DataFrame(
            {
                "image_id": ["b", "a", "c"],
                "label_name": ["L", "L", "L"],
                "score": [0.5, 0.5, 0.1],
            }
        )
        selected = top_budget_mask(frame, "score", 1)
        self.assertEqual(selected.tolist(), [False, True, False])

    def test_binary_metrics_detects_enrichment(self):
        truth = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
        prediction = np.array([1, 0, 0, 0, 0, 0], dtype=bool)
        score = np.array([0.9, 0.8, 0.4, 0.3, 0.2, 0.1])
        metrics = binary_metrics(truth, prediction, score)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["enrichment"], 3.0)
        self.assertEqual(metrics["auprc"], 1.0)

    def test_exact_sign_flip_resolution_for_four_agreeing_seeds(self):
        self.assertEqual(exact_sign_flip_p(np.ones(4)), 0.125)

    def test_holm_adjust_is_monotone_in_sorted_order(self):
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_inner_split_enforces_minimum_support(self):
        labels = np.zeros((100, 2), dtype=int)
        labels[:30, 0] = 1
        labels[20:60, 1] = 1
        inner_train, inner_validation, split_seed = build_inner_split(
            np.arange(100), labels, seed=13
        )
        self.assertGreaterEqual(split_seed, 13)
        for subset in [inner_train, inner_validation]:
            positive = labels[subset].sum(axis=0)
            negative = len(subset) - positive
            self.assertTrue(np.all(positive >= 3))
            self.assertTrue(np.all(negative >= 3))


if __name__ == "__main__":
    unittest.main()
