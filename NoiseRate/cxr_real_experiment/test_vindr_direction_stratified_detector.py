import unittest

import pandas as pd

from vindr_direction_stratified_detector import direction_stratified_selection


class DirectionStratifiedSelectionTests(unittest.TestCase):
    def test_fixed_quota_is_exact_and_direction_specific(self):
        frame = pd.DataFrame(
            {
                "image_id": ["a", "b", "c", "d", "e", "f"],
                "label_name": ["x"] * 6,
                "noisy_label": [0, 0, 0, 1, 1, 1],
                "score_oof_label_incompatibility": [0, 0, 0, 0.1, 0.9, 0.8],
                "score_simifeat_style_k50": [0.2, 0.9, 0.7, 0, 0, 0],
            }
        )
        selected, negatives, positives = direction_stratified_selection(
            frame, budget=4, negative_quota=0.5
        )
        self.assertEqual((negatives, positives), (2, 2))
        self.assertEqual(set(frame.loc[selected, "image_id"]), {"b", "c", "e", "f"})

    def test_queue_shortfall_is_reallocated(self):
        frame = pd.DataFrame(
            {
                "image_id": ["a", "b", "c", "d"],
                "label_name": ["x"] * 4,
                "noisy_label": [0, 1, 1, 1],
                "score_oof_label_incompatibility": [0, 0.3, 0.2, 0.1],
                "score_simifeat_style_k50": [0.9, 0, 0, 0],
            }
        )
        selected, negatives, positives = direction_stratified_selection(
            frame, budget=3, negative_quota=1.0
        )
        self.assertEqual((negatives, positives), (1, 2))
        self.assertEqual(len(selected), 3)


if __name__ == "__main__":
    unittest.main()
