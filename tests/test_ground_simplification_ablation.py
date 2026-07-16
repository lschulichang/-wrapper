from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "splatnav-official"))

from ground_nav.path_utils import (  # noqa: E402
    line_of_sight_supercover_free,
    merge_collinear_ground_path,
    simplify_ground_path_supercover,
    supercover_grid_indices,
)


class FakeGrid:
    def __init__(self, occupied):
        self.occupied = torch.as_tensor(occupied, dtype=torch.bool)
        self.shape = tuple(self.occupied.shape)

    def world_to_grid(self, point):
        return torch.round(torch.as_tensor(point)).to(torch.long)


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


class IntegerSupercoverTest(unittest.TestCase):
    def test_axis_aligned_and_reverse_are_symmetric(self):
        forward = supercover_grid_indices((0, 1), (3, 1))
        reverse = supercover_grid_indices((3, 1), (0, 1))
        self.assertEqual(forward, [(0, 1), (1, 1), (2, 1), (3, 1)])
        self.assertEqual(set(forward), set(reverse))

    def test_corner_crossing_includes_both_side_cells(self):
        cells = set(supercover_grid_indices((0, 0), (2, 2)))
        self.assertTrue({(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)} <= cells)

    def test_all_octants_have_reverse_symmetry(self):
        for goal in ((4, 1), (1, 4), (-1, 4), (-4, 1), (-4, -1),
                     (-1, -4), (1, -4), (4, -1)):
            forward = set(supercover_grid_indices((0, 0), goal))
            reverse = set(supercover_grid_indices(goal, (0, 0)))
            self.assertEqual(forward, reverse)

    def test_los_rejects_an_occupied_corner_touch(self):
        occupied = np.zeros((3, 3), dtype=bool)
        occupied[1, 0] = True
        grid = FakeGrid(occupied)
        self.assertFalse(line_of_sight_supercover_free(grid, (0, 0), (2, 2)))

    def test_greedy_supercover_preserves_endpoints_and_cap(self):
        grid = FakeGrid(np.zeros((8, 8), dtype=bool))
        path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                         [2.0, 1.0], [2.0, 2.0]])
        result = simplify_ground_path_supercover(path, grid, max_segment_length=2.1)
        np.testing.assert_allclose(result[[0, -1]], path[[0, -1]])
        self.assertLessEqual(np.max(np.linalg.norm(np.diff(result, axis=0), axis=1)), 2.1)


if __name__ == "__main__":
    unittest.main()
