#!/usr/bin/env python3
"""Tests for the forward-only Hybrid A* ground planner."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "splatnav-official"))

from ground_nav.bezier_2d import BezierPlanner2D  # noqa: E402
from ground_nav.corridor_2d import PlanarCollisionSet, build_planar_corridor  # noqa: E402
from ground_nav.hybrid_astar import (  # noqa: E402
    HybridAStarConfig,
    HybridAStarPlanner,
    _Record,
)
from ground_nav.planar_gaussians import PlanarGaussianSet  # noqa: E402


class FakeGrid:
    def __init__(self, shape=(30, 30), occupied=None):
        self.occupied = torch.zeros(shape, dtype=torch.bool)
        if occupied:
            for index in occupied:
                self.occupied[index] = True
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


class HybridAStarTest(unittest.TestCase):
    def config(self, **overrides):
        values = {
            "min_turning_radius_m": 2.0,
            "analytic_expansion_interval": 10,
            "analytic_expansion_distance_cells": 12.0,
            "max_expansions": 50_000,
        }
        values.update(overrides)
        return HybridAStarConfig(**values)

    def test_open_straight_path_reaches_exact_goal(self):
        planner = HybridAStarPlanner(FakeGrid(), 1.0, self.config())
        result = planner.plan([2.5, 2.5, 0.0], [12.5, 2.5, 0.0])
        np.testing.assert_allclose(result.poses_scene[0], [2.5, 2.5, 0.0], atol=1e-9)
        np.testing.assert_allclose(result.poses_scene[-1], [12.5, 2.5, 0.0], atol=1e-9)
        self.assertTrue(result.analytic_expansion_used)
        self.assertTrue(np.allclose(result.poses_scene[:, 1], 2.5, atol=1e-8))

    def test_quarter_turn_respects_minimum_radius(self):
        planner = HybridAStarPlanner(FakeGrid(), 1.0, self.config())
        result = planner.plan(
            [5.5, 5.5, 0.0],
            [7.5, 7.5, math.pi / 2.0],
        )
        np.testing.assert_allclose(
            result.poses_scene[-1],
            [7.5, 7.5, math.pi / 2.0],
            atol=1e-8,
        )
        self.assertLessEqual(
            np.max(np.abs(result.primitive_curvatures_1pm)),
            0.5 + 1e-12,
        )

    def test_free_goal_heading_reaches_exact_position(self):
        planner = HybridAStarPlanner(FakeGrid(), 1.0, self.config())
        result = planner.plan([5.5, 5.5, 0.0], [7.5, 7.5])
        np.testing.assert_allclose(
            result.poses_scene[-1, :2],
            [7.5, 7.5],
            atol=1e-8,
        )
        self.assertFalse(result.goal_heading_constrained)
        self.assertFalse(result.summary()["goal_heading_constrained"])
        self.assertTrue(np.isfinite(result.summary()["terminal_yaw_rad"]))

    def test_motion_checks_intermediate_samples(self):
        grid = FakeGrid(occupied=[(6, 5)])
        planner = HybridAStarPlanner(grid, 1.0, self.config())
        state = np.asarray([5.5, 5.5, 0.0])
        self.assertIsNone(
            planner._sample_motion(
                state,
                0.0,
                2.0,
                collision_check=True,
            )
        )

    def test_occupied_endpoint_is_rejected(self):
        planner = HybridAStarPlanner(
            FakeGrid(occupied=[(2, 2)]),
            1.0,
            self.config(),
        )
        with self.assertRaisesRegex(ValueError, "start pose"):
            planner.plan([2.5, 2.5, 0.0], [12.5, 2.5, 0.0])

    def test_search_routes_around_blocking_wall(self):
        occupied = [
            (15, j)
            for j in range(0, 25)
            if not 18 <= j <= 22
        ]
        grid = FakeGrid(shape=(35, 35), occupied=occupied)
        planner = HybridAStarPlanner(
            grid,
            1.0,
            self.config(analytic_expansion_distance_cells=6.0),
        )
        result = planner.plan([5.5, 5.5, 0.0], [27.5, 5.5, 0.0])
        indices = np.rint(result.poses_scene[:, :2] - 0.5).astype(int)
        self.assertTrue(np.all(~grid.occupied[indices[:, 0], indices[:, 1]].numpy()))
        self.assertGreater(float(np.max(result.poses_scene[:, 1])), 17.5)
        self.assertGreater(result.expanded_nodes, 1)

    def test_reconstruction_uses_immutable_parent_edges(self):
        planner = HybridAStarPlanner(FakeGrid(), 1.0, self.config())
        start = np.asarray([2.5, 2.5, 0.0])
        first_edge = np.asarray([[3.0, 2.5, 0.0], [4.0, 2.5, 0.0]])
        second_edge = np.asarray([[4.5, 2.5, 0.0], [5.5, 2.5, 0.0]])
        replacement_edge = np.asarray([[3.0, 3.0, 0.2], [4.0, 3.5, 0.2]])
        records = [
            _Record(start, 0.0, None, None, 0.0, planner.straight_index),
            _Record(first_edge[-1], 1.0, 0, first_edge, 0.0, planner.straight_index),
            _Record(second_edge[-1], 2.0, 1, second_edge, 0.0, planner.straight_index),
            # This represents a later lower-cost replacement for the same
            # discretized key as record 1. Existing descendants must keep
            # following record 1 and its already validated edge.
            _Record(
                replacement_edge[-1],
                0.5,
                0,
                replacement_edge,
                0.0,
                planner.straight_index,
            ),
        ]
        result = planner._reconstruct(
            records,
            2,
            (np.asarray([second_edge[-1]]), np.empty(0), 0.0),
            expanded_nodes=3,
            generated_nodes=4,
            goal_heading_constrained=False,
        )
        expected = np.vstack([start, first_edge, second_edge])
        np.testing.assert_allclose(result.poses_scene, expected, atol=1e-12)
        self.assertFalse(np.any(np.isclose(result.poses_scene[:, 1], 3.5)))

    def test_dense_hybrid_seed_enters_existing_corridor_without_extra_checks(self):
        planner = HybridAStarPlanner(FakeGrid(), 1.0, self.config())
        result = planner.plan([2.5, 2.5, 0.0], [12.5, 2.5])
        collision_set = PlanarCollisionSet(
            empty_ellipses(),
            radius=0.25,
            stopping_distance=2.0,
        )
        corridor = build_planar_corridor(result.xy, collision_set)
        bezier = BezierPlanner2D(degree=6, continuity_order=3)
        controls, feasible = bezier.optimize(
            corridor.polygons,
            result.xy[0],
            result.xy[-1],
            start_yaw=0.0,
            goal_yaw=float(result.poses_scene[-1, 2]),
            endpoint_tangent_min=0.05,
        )
        self.assertTrue(feasible)
        self.assertIsNotNone(controls)
        self.assertGreaterEqual(len(corridor.polygons), 1)
        start_tangent = controls[0, :, 1] - controls[0, :, 0]
        goal_tangent = controls[-1, :, -1] - controls[-1, :, -2]
        self.assertGreaterEqual(start_tangent[0], 0.05 - 1e-7)
        self.assertGreaterEqual(goal_tangent[0], 0.05 - 1e-7)
        self.assertAlmostEqual(start_tangent[1], 0.0, places=8)
        self.assertAlmostEqual(goal_tangent[1], 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
