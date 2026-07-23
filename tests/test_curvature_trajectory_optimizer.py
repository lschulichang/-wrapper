#!/usr/bin/env python3
"""Tests for the arc-length curvature-constrained trajectory optimizer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "splatnav-official"))

from ground_nav.curvature_trajectory_optimizer import (  # noqa: E402
    CurvatureTrajectoryOptimizer,
    TrajectoryOptimizerConfig,
)


def rectangular_corridor(x_min, x_max, y_min, y_max):
    A = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    b = np.asarray([x_max, -x_min, y_max, -y_min])
    return A, b


class CurvatureTrajectoryOptimizerTest(unittest.TestCase):
    def optimizer(self):
        return CurvatureTrajectoryOptimizer(
            TrajectoryOptimizerConfig(
                node_count=8,
                max_iterations=250,
                max_refinement_steps=0,
                dense_samples_per_segment=6,
            )
        )

    def test_straight_reference_satisfies_all_hard_constraints(self):
        x = np.linspace(0.0, 5.0, 11)
        reference = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
        result = self.optimizer().optimize(
            reference,
            [rectangular_corridor(-0.5, 5.5, -1.0, 1.0)],
            np.zeros(len(reference), dtype=int),
            start_pose=[0.0, 0.0, 0.0],
            goal_xy=[5.0, 0.0],
            min_turning_radius=2.0,
            max_curvature_rate=0.5,
        )
        self.assertTrue(result.success, result.failure_reason)
        np.testing.assert_allclose(result.poses[0], [0.0, 0.0, 0.0], atol=2e-5)
        np.testing.assert_allclose(result.poses[-1, :2], [5.0, 0.0], atol=2e-5)
        self.assertLessEqual(np.max(np.abs(result.curvature)), 0.95 / 2.0 + 2e-5)
        self.assertLessEqual(np.max(np.abs(result.curvature_rate)), 0.5 + 2e-5)
        self.assertTrue(np.all(np.diff(result.arc_length) > 0.0))
        self.assertEqual(len(result.path_to_corridor), len(result.poses))
        self.assertTrue(np.all(np.diff(result.path_to_corridor) >= 0))
        self.assertTrue(result.diagnostics[-1]["dense_corridor_feasible"])
        self.assertTrue(result.diagnostics[-1]["dense_dynamics_feasible"])

    def test_curvature_is_continuous_under_rate_control(self):
        x = np.linspace(0.0, 3.0, 9)
        reference = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
        result = self.optimizer().optimize(
            reference,
            [rectangular_corridor(-0.5, 3.5, -1.0, 1.0)],
            np.zeros(len(reference), dtype=int),
            start_pose=reference[0],
            goal_xy=reference[-1, :2],
            min_turning_radius=1.0,
            max_curvature_rate=0.25,
        )
        self.assertTrue(result.success, result.failure_reason)
        segment_lengths = np.diff(result.arc_length)
        np.testing.assert_allclose(
            np.diff(result.curvature),
            result.curvature_rate * segment_lengths,
            atol=3e-5,
        )

    def test_curved_reference_keeps_terminal_heading_free(self):
        radius = 3.0
        angle = np.linspace(0.0, np.pi / 3.0, 15)
        reference = np.column_stack(
            [
                radius * np.sin(angle),
                radius * (1.0 - np.cos(angle)),
                angle,
            ]
        )
        result = self.optimizer().optimize(
            reference,
            [rectangular_corridor(-0.5, 4.0, -0.5, 2.5)],
            np.zeros(len(reference), dtype=int),
            start_pose=reference[0],
            goal_xy=reference[-1, :2],
            min_turning_radius=2.5,
            max_curvature_rate=0.5,
        )
        self.assertTrue(result.success, result.failure_reason)
        np.testing.assert_allclose(result.poses[0], reference[0], atol=2e-5)
        np.testing.assert_allclose(result.poses[-1, :2], reference[-1, :2], atol=2e-5)
        self.assertLessEqual(np.max(np.abs(result.curvature)), 0.95 / 2.5 + 2e-5)
        # The API has no terminal-yaw equality; Hybrid yaw is a soft reference.
        self.assertTrue(np.isfinite(result.poses[-1, 2]))

    def test_rejects_nonmonotone_corridor_assignment(self):
        reference = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        )
        corridor = rectangular_corridor(-1.0, 3.0, -1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "monotone"):
            self.optimizer().optimize(
                reference,
                [corridor, corridor],
                [0, 1, 0],
                start_pose=reference[0],
                goal_xy=reference[-1, :2],
                min_turning_radius=1.0,
                max_curvature_rate=0.5,
            )


if __name__ == "__main__":
    unittest.main()
