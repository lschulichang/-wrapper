from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "splatnav-official"))

from ground_nav.path_utils import merge_collinear_ground_path  # noqa: E402


class CollinearSimplificationTest(unittest.TestCase):
    def test_merges_only_straight_forward_runs(self):
        path = np.array([
            [0.0, 0.0], [0.1, 0.0], [0.2, 0.0],
            [0.2, 0.1], [0.2, 0.2], [0.3, 0.2],
        ])
        result = merge_collinear_ground_path(path, max_segment_length=1.0)
        np.testing.assert_allclose(result, [
            [0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.3, 0.2],
        ])

    def test_respects_shared_segment_cap(self):
        path = np.column_stack([np.arange(7) * 0.2, np.zeros(7)])
        result = merge_collinear_ground_path(path, max_segment_length=0.5)
        self.assertTrue(np.all(np.linalg.norm(np.diff(result, axis=0), axis=1) <= 0.5 + 1e-12))
        np.testing.assert_allclose(result[[0, -1]], path[[0, -1]])

    def test_rejects_duplicate_consecutive_points(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            merge_collinear_ground_path(
                np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]), 1.0
            )


if __name__ == "__main__":
    unittest.main()
