#!/usr/bin/env python3
"""Integration tests for the three-layer ground-planning architecture."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "splatnav-official"))

from ground_nav.corridor_2d import PlanarCollisionSet  # noqa: E402
from ground_nav.curvature_trajectory_optimizer import (  # noqa: E402
    TrajectoryOptimizerConfig,
)
from ground_nav.hybrid_astar import (  # noqa: E402
    HybridAStarConfig,
    HybridAStarPlanner,
)
from ground_nav.planar_gaussians import PlanarGaussianSet  # noqa: E402
from ground_nav.three_layer_planner import (  # noqa: E402
    ThreeLayerGroundPlanner,
    ThreeLayerPlannerConfig,
)


class FakeGrid:
    def __init__(self, shape=(30, 30)):
        self.occupied = torch.zeros(shape, dtype=torch.bool)
        self.cell_sizes = torch.tensor([1.0, 1.0], dtype=torch.float64)
        self.lower_center = torch.tensor([0.5, 0.5], dtype=torch.float64)


def empty_ellipses():
    return PlanarGaussianSet(
        ids=torch.empty(0, dtype=torch.long),
        means=torch.empty((0, 2), dtype=torch.float64),
        covs=torch.empty((0, 2, 2), dtype=torch.float64),
        rots=torch.empty((0, 2, 2), dtype=torch.float64),
        scales=torch.empty((0, 2), dtype=torch.float64),
        z_min=0.0,
        z_max=1.0,
        confidence=1.0,
    )


class ThreeLayerGroundPlannerTest(unittest.TestCase):
    def build(self, max_iterations=250):
        hybrid = HybridAStarPlanner(
            FakeGrid(),
            scene_scale=1.0,
            config=HybridAStarConfig(
                min_turning_radius_m=2.0,
                analytic_expansion_interval=10,
                analytic_expansion_distance_cells=2.0,
            ),
        )
        collision_set = PlanarCollisionSet(
            empty_ellipses(),
            radius=0.25,
            stopping_distance=2.0,
        )
        return ThreeLayerGroundPlanner(
            hybrid,
            collision_set,
            ThreeLayerPlannerConfig(
                min_turning_radius_m=2.0,
                max_curvature_rate_1pm2=0.5,
                optimizer=TrajectoryOptimizerConfig(
                    node_count=8,
                    max_iterations=max_iterations,
                    max_refinement_steps=0,
                    dense_samples_per_segment=6,
                ),
            ),
        )

    def test_complete_three_layer_plan_reports_each_stage(self):
        planner = self.build()
        analytic_expansion = (
            planner.hybrid_planner._analytic_expansion_free_heading
        )
        attempt_count = 0

        def skip_initial_analytic_expansion(state, goal_xy):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                return None
            return analytic_expansion(state, goal_xy)

        planner.hybrid_planner._analytic_expansion_free_heading = (
            skip_initial_analytic_expansion
        )
        result = planner.plan([2.5, 2.5, 0.0], [12.5, 2.5])
        self.assertTrue(result.success, result.status())
        self.assertTrue(result.search_success)
        self.assertTrue(result.corridor_success)
        self.assertTrue(result.optimization_success)
        self.assertTrue(result.corridor_feasible)
        self.assertTrue(result.curvature_feasible)
        self.assertTrue(result.curvature_rate_feasible)
        self.assertFalse(result.fallback_used)
        self.assertEqual(
            result.trajectory_m.summary()["collocation_method"],
            "hermite_simpson",
        )
        self.assertTrue(
            result.trajectory_m.diagnostics[-1][
                "transition_overlap_feasible"
            ]
        )
        self.assertIsNotNone(result.simplification)
        self.assertEqual(
            len(result.corridor.path_to_corridor),
            len(result.simplification.boundary_poses_scene),
        )
        self.assertGreater(
            len(result.simplification.boundary_poses_scene),
            len(result.simplification.merged_poses_scene),
        )
        self.assertLessEqual(
            len(result.simplification.merged_poses_scene),
            len(
                result.hybrid_path.primitive_boundary_poses_scene
            ),
        )
        self.assertGreater(len(result.dense_output_poses_scene), 2)

    def test_optimizer_failure_uses_safe_hybrid_only_as_degraded_output(self):
        result = self.build(max_iterations=1).plan(
            [2.5, 2.5, 0.0], [12.5, 2.5]
        )
        self.assertTrue(result.search_success)
        self.assertTrue(result.corridor_success)
        self.assertFalse(result.optimization_success)
        self.assertTrue(result.fallback_used)
        self.assertFalse(result.success)
        self.assertEqual(
            result.failure_stage, "curvature_collocation"
        )
        np.testing.assert_allclose(
            result.output_poses_scene,
            result.hybrid_path.poses_scene,
        )
        self.assertIsNotNone(result.hybrid_path)


if __name__ == "__main__":
    unittest.main()
