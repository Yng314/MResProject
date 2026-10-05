#!/usr/bin/env python3
"""Focused tests for the VinDr XRV OOF comparison."""

import unittest

import numpy as np
import pandas as pd

from vindr_xrv_oof_comparison import (
    blind_join,
    fixed_rank_fusion,
    percentile_rank,
    primary_contrasts,
)


class VinDrXRVOOFComparisonTests(unittest.TestCase):
    def test_percentile_rank_is_monotonic_and_tie_stable(self) -> None:
        ranks = percentile_rank(np.array([1.0, 3.0, 3.0, 2.0]))
        self.assertLess(ranks[0], ranks[3])
        self.assertEqual(ranks[1], ranks[2])
        self.assertLessEqual(ranks.max(), 1.0)

    def test_fusion_uses_equal_rank_weight(self) -> None:
        first = np.array([0.1, 0.9, 0.4])
        second = np.array([0.8, 0.2, 0.5])
        expected = 0.5 * percentile_rank(first) + 0.5 * percentile_rank(second)
        np.testing.assert_allclose(fixed_rank_fusion(first, second), expected)

    def test_blind_join_rejects_noisy_label_mismatch(self) -> None:
        mobile = pd.DataFrame(
            {
                "image_id": ["a"],
                "fold_id": [0],
                "label_index": [0],
                "label_name": ["Atelectasis"],
                "noisy_label": [0],
                "score_oof_cl": [0.2],
                "score_external_xrv": [0.3],
            }
        )
        xrv = pd.DataFrame(
            {
                "image_id": ["a"],
                "fold_id": [0],
                "label_index": [0],
                "label_name": ["Atelectasis"],
                "noisy_label": [1],
                "cl_first_score": [0.4],
            }
        )
        with self.assertRaises(ValueError):
            blind_join(mobile, xrv)

    def test_primary_contrasts_accepts_explicit_smoke_seed_count(self) -> None:
        rows = []
        for method, recall in {
            "mobilenet_oof_cl": 0.2,
            "xrv_oof_cl": 0.4,
            "direct_xrv": 0.3,
            "xrv_oof_direct_fusion": 0.5,
        }.items():
            rows.append(
                {
                    "regime": "balanced",
                    "rate_percent": 20,
                    "scope": "overall",
                    "seed": 887,
                    "method": method,
                    "recall": recall,
                }
            )
        result = primary_contrasts(pd.DataFrame(rows), expected_seeds=1)
        self.assertEqual(len(result), 3)
        self.assertTrue((result["paired_seeds"] == 1).all())


if __name__ == "__main__":
    unittest.main()
