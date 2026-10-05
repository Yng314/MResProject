#!/usr/bin/env python3
"""Regression tests for the VinDr self-optimization stress experiment."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from vindr_known_gt_cl_benchmark import LABELS
from vindr_self_optimization_stress import (
    cohort_entries,
    corrupt_from_orders,
    ordered_candidates,
    private_frame,
)


class SelfOptimizationStressTest(unittest.TestCase):
    def setUp(self) -> None:
        samples = 100
        self.image_ids = pd.Series([f"image_{index:03d}" for index in range(samples)])
        self.clean = np.zeros((samples, len(LABELS)), dtype=np.int64)
        for label_index in range(len(LABELS)):
            self.clean[: 10 + label_index, label_index] = 1
        records = []
        for row_index, image_id in enumerate(self.image_ids):
            for label_index, label in enumerate(LABELS):
                clean_label = int(self.clean[row_index, label_index])
                records.append(
                    {
                        "image_id": image_id,
                        "label_name": label,
                        "label_quality_self_confidence": (
                            row_index + 1 + label_index / 10
                        ) / 101,
                        "noisy_label": clean_label,
                    }
                )
        self.hardness = pd.DataFrame(records)

    def test_uniform_and_hard_match_counts(self) -> None:
        outputs = {}
        for structure in ("uniform", "hard"):
            orders = ordered_candidates(
                self.clean, self.image_ids, self.hardness, 11003, structure
            )
            outputs[structure] = corrupt_from_orders(self.clean, orders, 0.20)
        self.assertEqual(
            int(outputs["uniform"][1].sum()), int(outputs["hard"][1].sum())
        )
        uniform_counts = outputs["uniform"][3].sort_values("label_name").reset_index(drop=True)
        hard_counts = outputs["hard"][3].sort_values("label_name").reset_index(drop=True)
        pd.testing.assert_frame_equal(uniform_counts, hard_counts)
        self.assertEqual(int(outputs["uniform"][1].sum()), 120)

    def test_rates_are_nested_with_fixed_order(self) -> None:
        orders = ordered_candidates(self.clean, self.image_ids, self.hardness, 13007, "hard")
        low = corrupt_from_orders(self.clean, orders, 0.20)[1]
        medium = corrupt_from_orders(self.clean, orders, 0.30)[1]
        high = corrupt_from_orders(self.clean, orders, 0.40)[1]
        self.assertTrue(np.all(~low | medium))
        self.assertTrue(np.all(~medium | high))

    def test_private_reference_and_dynamic_entry_count(self) -> None:
        orders = ordered_candidates(self.clean, self.image_ids, self.hardness, 17011, "uniform")
        noisy, injected, direction, _ = corrupt_from_orders(self.clean, orders, 0.30)
        private = private_frame(self.image_ids, self.clean, noisy, injected, direction)
        self.assertEqual(len(private), 600)
        self.assertEqual(int(private["injected_error"].sum()), 180)
        cohort = pd.DataFrame({"image_id": self.image_ids, "image_path": "x.png", "fold_id": 0})
        for label_index, label in enumerate(LABELS):
            cohort[label] = noisy[:, label_index]
        entries = cohort_entries(cohort)
        self.assertEqual(len(entries), 600)
        self.assertFalse(entries["entry_key"].duplicated().any())


if __name__ == "__main__":
    unittest.main()
