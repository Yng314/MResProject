#!/usr/bin/env python3

import unittest

import pandas as pd

from vindr_iterative_evidence_improvement import rank_entries
from vindr_matched_budget_refinement import apply_oracle, cumulative_review_endpoints


class MatchedBudgetRefinementTests(unittest.TestCase):
    def test_endpoints_exactly_partition_nondivisible_pool(self) -> None:
        endpoints = cumulative_review_endpoints(2333, 5)
        self.assertEqual(endpoints, [0, 466, 933, 1399, 1866, 2333])
        increments = [right - left for left, right in zip(endpoints, endpoints[1:])]
        self.assertEqual(sum(increments), 2333)
        self.assertLessEqual(max(increments) - min(increments), 1)

    def test_cl_first_endpoint_equals_issue_pool(self) -> None:
        evidence = pd.DataFrame(
            {
                "image_id": ["a", "b", "c", "d"],
                "label_name": ["x", "x", "x", "x"],
                "label_index": [0, 0, 0, 0],
                "cl_issue": [False, True, False, True],
                "cl_first_score": [0.99, 2.10, 0.98, 2.05],
                "self_confidence_suspicion": [0.99, 0.10, 0.98, 0.05],
                "entry_key": ["a::x", "b::x", "c::x", "d::x"],
            }
        )
        ranked = rank_entries(evidence)
        selected = set(ranked.head(2)["entry_key"])
        issues = set(evidence.loc[evidence["cl_issue"], "entry_key"])
        self.assertEqual(selected, issues)

    def test_oracle_updates_only_selected_entries(self) -> None:
        cohort = pd.DataFrame(
            {
                "image_id": ["a", "b"],
                "Finding": [1, 0],
                "untouched": [7, 8],
            }
        )
        selected = pd.DataFrame(
            {
                "entry_key": ["a::Finding"],
                "image_id": ["a"],
                "label_name": ["Finding"],
                "current_label": [1],
                "clean_label": [0],
            }
        )
        updated = apply_oracle(cohort, selected)
        self.assertEqual(updated["Finding"].tolist(), [0, 0])
        self.assertEqual(updated["untouched"].tolist(), [7, 8])


if __name__ == "__main__":
    unittest.main()
