#!/usr/bin/env python3
"""Unit tests for motion-direction-preserving Hybrid A* seed compression."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_hybrid_compression_ablation import (  # noqa: E402
    compress_direction_preserving,
    motion_direction_runs,
)


class FakeGrid:
    def __init__(self, shape=(20, 20)):
        self.shape = shape
        self.occupied = torch.zeros(shape, dtype=torch.bool)
        self.cell_sizes = torch.tensor([1.0, 1.0], dtype=torch.float64)

    def world_to_grid(self, point):
        return torch.as_tensor(
            np.rint(np.asarray(point)).astype(np.int64)
        )


class HybridCompressionAblationTest(unittest.TestCase):
    def test_forward_only_path_is_one_motion_run(self):
        poses = np.asarray(
            [
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
            ]
        )
        self.assertEqual(motion_direction_runs(poses), [(0, 2, 1)])

    def test_cusp_is_never_crossed(self):
        poses = np.asarray(
            [
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        self.assertEqual(
            motion_direction_runs(poses),
            [(0, 2, 1), (2, 4, -1)],
        )
        compressed, diagnostics = compress_direction_preserving(
            poses,
            FakeGrid(),
            max_segment_scene=10.0,
        )
        self.assertEqual(len(diagnostics), 2)
        self.assertTrue(
            np.any(
                np.all(
                    np.isclose(compressed, [3.0, 1.0]),
                    axis=1,
                )
            )
        )

    def test_occupied_cell_blocks_los_shortcut(self):
        poses = np.asarray(
            [
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
            ]
        )
        grid = FakeGrid()
        grid.occupied[2, 1] = True
        with self.assertRaisesRegex(RuntimeError, "no line-of-sight"):
            compress_direction_preserving(
                poses,
                grid,
                max_segment_scene=10.0,
            )


if __name__ == "__main__":
    unittest.main()
