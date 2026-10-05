import unittest

import numpy as np
import pandas as pd

from vindr_noise_direction_sensitivity import (
    LABELS,
    calibration_table,
    make_corruption,
    parse_rates,
    scenario_definitions,
)


class VinDrNoiseDirectionSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.clean = np.zeros((300, len(LABELS)), dtype=np.int64)
        supports = [50, 60, 70, 80, 90, 100]
        for index, support in enumerate(supports):
            self.clean[:support, index] = 1

    def test_scenario_grid_has_clean_plus_nine_noisy_scenarios(self):
        scenarios = scenario_definitions(parse_rates("0.10,0.20,0.30"))
        self.assertEqual(len(scenarios), 10)
        self.assertEqual(scenarios[0]["scenario_id"], "clean")
        self.assertEqual(
            {scenario["regime"] for scenario in scenarios[1:]},
            {"balanced", "fp_only", "fn_only"},
        )

    def test_regimes_have_matched_total_error_counts(self):
        scenarios = scenario_definitions([0.20])
        totals = {}
        for scenario in scenarios[1:]:
            _, injected, direction, counts = make_corruption(
                self.clean, 13, scenario
            )
            totals[scenario["regime"]] = int(injected.sum())
            self.assertEqual(int(injected.sum()), int(counts["injected_errors"].sum()))
            if scenario["regime"] == "balanced":
                self.assertEqual(int((direction == "0_to_1").sum()), int((direction == "1_to_0").sum()))
            elif scenario["regime"] == "fp_only":
                self.assertEqual(int((direction == "1_to_0").sum()), 0)
            elif scenario["regime"] == "fn_only":
                self.assertEqual(int((direction == "0_to_1").sum()), 0)
        self.assertEqual(len(set(totals.values())), 1)

    def test_corruption_is_deterministic(self):
        scenario = scenario_definitions([0.10])[2]
        first = make_corruption(self.clean, 211, scenario)
        second = make_corruption(self.clean, 211, scenario)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[2], second[2])

    def test_calibration_table_detects_monotonic_low_error_curve(self):
        records = []
        for seed in [13, 42]:
            records.append(
                {
                    "seed": seed,
                    "seed_block": "parent_overlap",
                    "regime": "clean",
                    "noise_rate": 0.0,
                    "true_quality": 1.0,
                    "raw_entry_dqs": 0.998,
                }
            )
            for regime in ["balanced", "fp_only", "fn_only"]:
                for rate, quality in [(0.10, 0.99), (0.20, 0.98), (0.30, 0.97)]:
                    records.append(
                        {
                            "seed": seed,
                            "seed_block": "parent_overlap",
                            "regime": regime,
                            "noise_rate": rate,
                            "true_quality": quality,
                            "raw_entry_dqs": quality - 0.002,
                        }
                    )
        calibration = calibration_table(pd.DataFrame(records))
        self.assertEqual(len(calibration), 6)
        self.assertTrue(calibration["strictly_decreasing_with_noise"].all())
        self.assertTrue((calibration["spearman_rho"] >= 0.99).all())
        self.assertTrue((calibration["mae"] <= 0.0021).all())


if __name__ == "__main__":
    unittest.main()
