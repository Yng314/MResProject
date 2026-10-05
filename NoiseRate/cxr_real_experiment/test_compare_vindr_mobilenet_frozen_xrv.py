import unittest

import pandas as pd

from compare_vindr_mobilenet_frozen_xrv import primary_decision


class CompareVinDrModelsTests(unittest.TestCase):
    def clean_frame(self, difference: float = -0.01) -> pd.DataFrame:
        return pd.DataFrame({"mobilenet_minus_frozen_clean_auroc": [difference] * 6})

    def fn_frame(self, difference: float) -> pd.DataFrame:
        return pd.DataFrame({"mobilenet_minus_frozen_fn_recall": [difference] * 6})

    def test_primary_gate_passes_for_six_consistent_large_improvements(self):
        result = primary_decision(self.fn_frame(0.12), self.clean_frame())
        self.assertEqual(result["two_sided_six_seed_exact_sign_flip_p"], 0.03125)
        self.assertTrue(result["model_limitation_explanation_supported"])

    def test_primary_gate_fails_below_practical_effect_threshold(self):
        result = primary_decision(self.fn_frame(0.08), self.clean_frame())
        self.assertFalse(result["gates"]["mean_fn_recall_improvement_at_least_0_10"])
        self.assertFalse(result["model_limitation_explanation_supported"])

    def test_primary_gate_fails_for_excess_clean_auroc_loss(self):
        result = primary_decision(self.fn_frame(0.12), self.clean_frame(-0.03))
        self.assertFalse(result["gates"]["clean_reference_auroc_not_more_than_0_02_lower"])
        self.assertFalse(result["model_limitation_explanation_supported"])


if __name__ == "__main__":
    unittest.main()
