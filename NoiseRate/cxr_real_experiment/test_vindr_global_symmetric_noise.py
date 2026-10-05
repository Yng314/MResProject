import unittest

import numpy as np
import pandas as pd

from vindr_global_symmetric_noise import (
    LABELS,
    calibration_table,
    make_corruption,
    scenario_definitions,
)


class VinDrGlobalSymmetricNoiseTests(unittest.TestCase):
    def setUp(self):
        self.clean = np.zeros((300, len(LABELS)), dtype=np.int64)
        supports = [25, 35, 45, 55, 65, 75]
        for label_index, support in enumerate(supports):
            self.clean[:support, label_index] = 1

    def test_scenario_grid_has_four_known_quality_anchors(self):
        scenarios = scenario_definitions([0.10, 0.20, 0.30])
        self.assertEqual(
            [scenario["scenario_id"] for scenario in scenarios],
            ["clean", "symmetric_entry_r10", "symmetric_entry_r20", "symmetric_entry_r30"],
        )

    def test_each_label_and_full_matrix_have_exact_noise_rate(self):
        for scenario in scenario_definitions([0.10, 0.20, 0.30])[1:]:
            _, injected, _, counts = make_corruption(self.clean, 13, scenario)
            expected_per_label = int(round(scenario["noise_rate"] * len(self.clean)))
            expected_total = int(round(scenario["noise_rate"] * self.clean.size))
            self.assertEqual(int(injected.sum()), expected_total)
            self.assertTrue((counts["injected_errors"] == expected_per_label).all())

    def test_class_conditional_rates_match_target_with_rounding(self):
        scenario = scenario_definitions([0.30])[1]
        _, _, _, counts = make_corruption(self.clean, 42, scenario)
        self.assertTrue((counts["false_positive_rate"] - 0.30).abs().max() < 0.01)
        self.assertTrue((counts["false_negative_rate"] - 0.30).abs().max() < 0.03)

    def test_corruption_is_deterministic(self):
        scenario = scenario_definitions([0.20])[1]
        first = make_corruption(self.clean, 211, scenario)
        second = make_corruption(self.clean, 211, scenario)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[2], second[2])

    def test_calibration_table_uses_known_quality_anchors(self):
        records = []
        for seed in [13, 42]:
            for rate, quality in [(0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]:
                records.append(
                    {
                        "seed": seed,
                        "seed_block": "test",
                        "noise_rate": rate,
                        "true_quality": quality,
                        "raw_entry_dqs": quality - 0.01,
                    }
                )
        calibration = calibration_table(pd.DataFrame(records))
        self.assertEqual(len(calibration), 2)
        self.assertTrue(calibration["strictly_decreasing_with_noise"].all())
        self.assertTrue((calibration["spearman_rho"] == 1.0).all())
        self.assertTrue((calibration["mae"] <= 0.0100001).all())


if __name__ == "__main__":
    unittest.main()
