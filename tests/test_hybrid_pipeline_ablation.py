#!/usr/bin/env python3
"""Tests for the B/C/D Hybrid A* pipeline ablation helpers."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_hybrid_pipeline_ablation import (  # noqa: E402
    arclength_anchors,
    fit_plain_piecewise_bezier,
    free_space_statistics,
    sample_bezier_controls,
    select_manifest_rows,
)


class FakeGrid:
    def __init__(self):
        self.shape = (10, 10)
        self.lower_center = torch.tensor([0.5, 0.5], dtype=torch.float64)
        self.upper_center = torch.tensor([9.5, 9.5], dtype=torch.float64)
        self.cell_sizes = torch.tensor([1.0, 1.0], dtype=torch.float64)
        self.occupied = torch.zeros(self.shape, dtype=torch.bool)
        self.occupied[4, 4] = True


class HybridPipelineAblationTest(unittest.TestCase):
    def test_arclength_anchors_preserve_endpoints(self):
        poses = np.asarray(
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )
        anchors = arclength_anchors(poses, scale=1.0, spacing_m=0.3)
        np.testing.assert_allclose(anchors[0], poses[0, :2])
        np.testing.assert_allclose(anchors[-1], poses[-1, :2])
        self.assertGreater(len(anchors), 2)

    def test_plain_bezier_is_endpoint_and_tangent_constrained(self):
        poses = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.5, math.pi / 4.0],
                [2.0, 1.0, math.pi / 2.0],
            ]
        )
        controls, anchors = fit_plain_piecewise_bezier(
            poses,
            scale=1.0,
            spacing_m=0.5,
        )
        dense = sample_bezier_controls(controls, 20)
        np.testing.assert_allclose(dense[0, 0], poses[0, :2])
        np.testing.assert_allclose(dense[-1, -1], poses[-1, :2])
        start_tangent = controls[0, :, 1] - controls[0, :, 0]
        goal_tangent = controls[-1, :, -1] - controls[-1, :, -2]
        self.assertGreater(start_tangent[0], 0.0)
        self.assertAlmostEqual(start_tangent[1], 0.0, places=10)
        self.assertGreater(goal_tangent[1], 0.0)
        self.assertAlmostEqual(goal_tangent[0], 0.0, places=10)
        self.assertGreaterEqual(len(anchors), 3)

    def test_free_space_statistics_counts_occupied_and_outside(self):
        points = np.asarray([[1.5, 1.5], [4.5, 4.5], [12.0, 1.0]])
        result = free_space_statistics(points, FakeGrid())
        self.assertEqual(result["outside_free_sample_count"], 2)
        self.assertAlmostEqual(result["outside_free_sample_fraction"], 2.0 / 3.0)

    def test_manifest_selection_balances_distance_layers(self):
        rows = []
        for layer in ("near", "medium", "far"):
            for index in range(40):
                rows.append(
                    {
                        "trial_id": f"{layer}_{index:03d}",
                        "candidate_id": f"{layer}_{index:03d}",
                        "distance_layer": layer,
                        "goal_heading_mode": "free",
                    }
                )
        selected = select_manifest_rows(rows, 30)
        self.assertEqual(len(selected), 30)
        for layer in ("near", "medium", "far"):
            self.assertEqual(
                sum(row["distance_layer"] == layer for row in selected),
                10,
            )


if __name__ == "__main__":
    unittest.main()
