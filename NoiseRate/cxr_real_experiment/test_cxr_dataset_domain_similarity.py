import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import torch

from cxr_dataset_domain_similarity import atomic_npz, c2st, frechet_distance, rbf_mmd_test


class DatasetDomainSimilarityTests(unittest.TestCase):
    def test_frechet_identity_is_zero(self):
        rng = np.random.default_rng(1)
        values = rng.normal(size=(100, 8))
        self.assertLess(frechet_distance(values, values), 1e-8)

    def test_c2st_separates_shifted_domains(self):
        rng = np.random.default_rng(2)
        first = rng.normal(loc=-2.0, size=(100, 6))
        second = rng.normal(loc=2.0, size=(100, 6))
        values = np.vstack([first, second])
        labels = np.repeat([0, 1], 100)
        result = c2st(values, labels, seed=3, bootstrap_draws=50)
        self.assertGreater(result["auc"], 0.99)
        self.assertGreater(result["ci_low"], 0.98)

    def test_mmd_accepts_integer_labels_with_float32_features(self):
        rng = np.random.default_rng(4)
        first = rng.normal(loc=-1.0, size=(20, 4)).astype(np.float32)
        second = rng.normal(loc=1.0, size=(20, 4)).astype(np.float32)
        values = np.vstack([first, second])
        labels = np.repeat([0, 1], 20)
        result = rbf_mmd_test(
            values,
            labels,
            device=torch.device("cpu"),
            seed=5,
            permutations=9,
            internal_repeats=3,
        )
        self.assertGreater(result["mmd2"], 0.0)
        self.assertGreaterEqual(result["permutation_p"], 0.0)
        self.assertLessEqual(result["permutation_p"], 1.0)

    def test_atomic_npz_stores_pickle_free_unicode_ids(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            atomic_npz(
                path,
                features=np.ones((2, 3), dtype=np.float32),
                source=np.array([0, 1]),
                image_id=np.array(["vindr-id", "mimic-id"], dtype="U64"),
            )
            with np.load(path, allow_pickle=False) as payload:
                self.assertFalse(payload["image_id"].dtype.hasobject)
                self.assertEqual(payload["image_id"].tolist(), ["vindr-id", "mimic-id"])


if __name__ == "__main__":
    unittest.main()
