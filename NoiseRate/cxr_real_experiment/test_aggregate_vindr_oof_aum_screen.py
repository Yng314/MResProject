#!/usr/bin/env python3
"""Synthetic integration test for the AUM screen aggregator."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aggregate_vindr_oof_aum_screen import aggregate


class Args:
    root: str


class AUMAggregateTests(unittest.TestCase):
    def test_complete_two_seed_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in [13, 211]:
                seed_root = root / f"seed_{seed}"
                evaluation = seed_root / "private_evaluation"
                evaluation.mkdir(parents=True)
                (seed_root / ".benchmark_verified").write_text("verified\n")
                (seed_root / ".blind_run_complete").write_text("complete\n")
                (evaluation / ".evaluation_complete").write_text("complete\n")
                pd.DataFrame(
                    [
                        {"seed": seed, "method": "oof_label_incompatibility", "auprc": 0.8, "recall_at_error_count": 0.8},
                        {"seed": seed, "method": "active_label_cleaning", "auprc": 0.81, "recall_at_error_count": 0.8},
                        {"seed": seed, "method": "oof_aum", "auprc": 0.82, "recall_at_error_count": 0.81},
                    ]
                ).to_csv(evaluation / "detector_metrics.csv", index=False)
            args = Args()
            args.root = str(root)
            aggregate(args)
            self.assertTrue((root / "aggregate" / ".aggregate_complete").is_file())


if __name__ == "__main__":
    unittest.main()
