import unittest

import pandas as pd
import torch

from vindr_densenet_oof import create_densenet
from vindr_iterative_evidence_improvement import (
    IMAGES,
    choose_probe_images,
    select_fixed_budget,
)


class DenseNetInitializationTest(unittest.TestCase):
    def test_branches_share_architecture_and_are_fully_trainable(self) -> None:
        cache = "/vol/gpudata/yz3522-llmtest/.cache/torchxrayvision"
        scratch = create_densenet("scratch", cache)
        pretrained = create_densenet("xrv_pretrained", cache)
        self.assertEqual(
            [tuple(parameter.shape) for parameter in scratch.parameters()],
            [tuple(parameter.shape) for parameter in pretrained.parameters()],
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in scratch.parameters()),
            sum(parameter.numel() for parameter in pretrained.parameters()),
        )
        self.assertTrue(all(parameter.requires_grad for parameter in scratch.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in pretrained.parameters()))
        with torch.no_grad():
            self.assertEqual(tuple(scratch(torch.zeros(1, 1, 224, 224)).shape), (1, 6))
            self.assertEqual(tuple(pretrained(torch.zeros(1, 1, 224, 224)).shape), (1, 6))


class IterativeSelectionTest(unittest.TestCase):
    def test_probe_split_is_deterministic_and_outcome_free(self) -> None:
        image_ids = [f"image_{index:04d}" for index in range(IMAGES)]
        first = choose_probe_images(image_ids, seed=13)
        second = choose_probe_images(reversed(image_ids), seed=13)
        other = choose_probe_images(image_ids, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 600)
        self.assertEqual(len(set(first)), 600)
        self.assertNotEqual(first, other)

    def test_selection_excludes_probe_and_previous_reviews(self) -> None:
        evidence = pd.DataFrame(
            {
                "entry_key": ["a::x", "b::x", "c::x", "d::x", "e::x"],
                "image_id": ["a", "b", "c", "d", "e"],
                "label_name": ["x"] * 5,
                "label_index": [0] * 5,
                "cl_first_score": [2.9, 2.8, 2.7, 0.9, 0.8],
                "self_confidence_suspicion": [0.9, 0.8, 0.7, 0.9, 0.8],
            }
        )
        selected, candidates = select_fixed_budget(
            evidence,
            reviewed_keys={"a::x"},
            probe_image_ids={"b"},
            budget=2,
        )
        self.assertEqual(candidates, 3)
        self.assertEqual(selected["entry_key"].tolist(), ["c::x", "d::x"])
        self.assertTrue(set(selected["entry_key"]).isdisjoint({"a::x", "b::x"}))


if __name__ == "__main__":
    unittest.main()
