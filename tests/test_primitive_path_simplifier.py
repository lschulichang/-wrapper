#!/usr/bin/env python3
"""Tests for motion-primitive boundary simplification."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "splatnav-official"))

from ground_nav.primitive_path_simplifier import (  # noqa: E402
    simplify_hybrid_path_by_primitives,
)


class PrimitivePathSimplifierTest(unittest.TestCase):
    def path(self, curvatures):
        curvatures = np.asarray(curvatures, dtype=np.float64)
        boundaries = np.column_stack(
            [
                np.arange(len(curvatures) + 1, dtype=np.float64),
                np.zeros(len(curvatures) + 1),
                np.arange(len(curvatures) + 1, dtype=np.float64)
                * 0.1,
            ]
        )
        dense_x = np.linspace(
            0.0, float(len(curvatures)), 4 * len(curvatures) + 1
        )
        dense = np.column_stack(
            [
                dense_x,
                np.zeros_like(dense_x),
                0.1 * dense_x,
            ]
        )
        return SimpleNamespace(
            poses_scene=dense,
            primitive_boundary_poses_scene=boundaries,
            primitive_curvatures_1pm=curvatures,
        )

    def test_uses_primitive_boundaries_before_merging(self):
        result = simplify_hybrid_path_by_primitives(
            self.path([0.0, 0.0, 1.0, 1.0])
        )
        self.assertEqual(len(result.boundary_poses_scene), 5)
        np.testing.assert_allclose(
            result.merged_poses_scene[:, 0], [0.0, 2.0, 4.0]
        )
        np.testing.assert_allclose(
            result.merged_curvatures_1pm, [0.0, 1.0]
        )
        self.assertEqual(
            [row["primitive_count"] for row in result.curvature_runs],
            [2, 2],
        )

    def test_alternating_curvature_keeps_every_boundary(self):
        path = self.path([-1.0, 1.0, -1.0])
        result = simplify_hybrid_path_by_primitives(path)
        np.testing.assert_allclose(
            result.merged_poses_scene,
            path.primitive_boundary_poses_scene,
        )

    def test_tolerance_merges_numerically_equal_curvature(self):
        result = simplify_hybrid_path_by_primitives(
            self.path([1.0, 1.0 + 1e-10]),
            curvature_tolerance_1pm=1e-8,
        )
        self.assertEqual(len(result.curvature_runs), 1)
        self.assertEqual(len(result.merged_poses_scene), 2)

    def test_rejects_inconsistent_boundary_count(self):
        path = self.path([0.0, 1.0])
        path.primitive_boundary_poses_scene = (
            path.primitive_boundary_poses_scene[:-1]
        )
        with self.assertRaisesRegex(ValueError, "one more pose"):
            simplify_hybrid_path_by_primitives(path)


if __name__ == "__main__":
    unittest.main()
