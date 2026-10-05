#!/usr/bin/env python3
"""Focused tests for the OOF AUM screen."""

import unittest

import numpy as np

from vindr_oof_aum_screen import top_recall


class OOFAUMScreenTests(unittest.TestCase):
    def test_assigned_binary_margin_sign(self) -> None:
        labels = np.array([0, 1], dtype=float)
        logits = np.array([-2.0, 2.0])
        margins = (2.0 * labels - 1.0) * logits
        np.testing.assert_allclose(margins, [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
