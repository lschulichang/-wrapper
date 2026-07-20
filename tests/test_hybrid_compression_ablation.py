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
    select_distance_turn_stratified,
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

    def test_fixed_start_selection_balances_distance_and_turns(self):
        candidates = []
        for index in range(90):
            candidates.append(
                {
                    "candidate_id": f"candidate_{index:04d}",
                    "direct_distance_m": float(index + 1),
                    "reference_turn_count": int(index % 10),
                    "reference_path_length_m": float(index + 2),
                }
            )
        selected = select_distance_turn_stratified(
            candidates,
            sample_count=30,
        )
        self.assertEqual(len(selected), 30)
        self.assertEqual(
            len({row["candidate_id"] for row in selected}),
            30,
        )
        for layer in ("near", "medium", "far"):
            rows = [
                row
                for row in selected
                if row["distance_layer"] == layer
            ]
            self.assertEqual(len(rows), 10)
            self.assertGreaterEqual(
                len({row["reference_turn_count"] for row in rows}),
                8,
            )

    def test_fixed_start_selection_scales_to_120_trials(self):
        candidates = []
        for index in range(1200):
            candidates.append(
                {
                    "candidate_id": f"candidate_{index:04d}",
                    "direct_distance_m": float(index + 1),
                    "reference_turn_count": int(index % 100),
                    "reference_path_length_m": float(index + 2),
                }
            )
        selected = select_distance_turn_stratified(
            candidates,
            sample_count=120,
        )
        self.assertEqual(len(selected), 120)
        self.assertEqual(
            len({row["candidate_id"] for row in selected}),
            120,
        )
        for layer in ("near", "medium", "far"):
            rows = [
                row
                for row in selected
                if row["distance_layer"] == layer
            ]
            self.assertEqual(len(rows), 40)
            self.assertEqual(
                len({row["turn_target_quantile"] for row in rows}),
                40,
            )


if __name__ == "__main__":
    unittest.main()
