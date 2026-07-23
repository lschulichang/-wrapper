#!/usr/bin/env python3
"""Tests for the single ellipse-distance GroundGrid route."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "splatnav-official"))

from ground_nav.ground_grid import GroundGrid  # noqa: E402
from ground_nav.planar_gaussians import PlanarGaussianSet  # noqa: E402


def ellipse_set(means, scales):
    means = torch.as_tensor(means, dtype=torch.float64)
    scales = torch.as_tensor(scales, dtype=torch.float64)
    count = len(means)
    rotations = torch.eye(
        2, dtype=torch.float64
    ).repeat(count, 1, 1)
    covariances = rotations @ torch.diag_embed(
        scales.square()
    ) @ rotations.transpose(1, 2)
    return PlanarGaussianSet(
        ids=torch.arange(count),
        means=means,
        covs=covariances,
        rots=rotations,
        scales=scales,
        z_min=0.0,
        z_max=1.0,
        confidence=1.0,
    )


def build_grid(ellipses, radius=0.0):
    return GroundGrid.from_projected_gaussians(
        ellipses,
        lower_xy=[0.0, 0.0],
        upper_xy=[10.0, 10.0],
        resolution_xy=[10, 10],
        footprint_radius=radius,
        z_floor_scene=0.0,
        robot_height=1.0,
        ground_clearance=0.0,
    )


class GroundGridTest(unittest.TestCase):
    def test_direct_constructor_is_disabled(self):
        with self.assertRaisesRegex(TypeError, "from_projected"):
            GroundGrid()

    def test_empty_projection_produces_free_grid(self):
        grid = build_grid(
            ellipse_set(
                np.empty((0, 2)),
                np.empty((0, 2)),
            )
        )
        self.assertEqual(grid.shape, (10, 10))
        self.assertFalse(bool(torch.any(grid.occupied).item()))
        self.assertEqual(
            grid.metadata.rasterization,
            "center_distance_plus_cell_circumradius",
        )

    def test_projected_ellipse_occupies_its_center_cell(self):
        grid = build_grid(
            ellipse_set([[5.5, 5.5]], [[0.4, 0.4]])
        )
        self.assertTrue(grid.is_occupied([5.5, 5.5]))
        self.assertFalse(grid.is_occupied([0.5, 0.5]))

    def test_footprint_radius_expands_occupied_region(self):
        ellipses = ellipse_set([[5.5, 5.5]], [[0.4, 0.4]])
        point = [7.5, 5.5]
        self.assertFalse(build_grid(ellipses, 0.0).is_occupied(point))
        self.assertTrue(build_grid(ellipses, 1.2).is_occupied(point))

    def test_point_to_ellipse_distance_is_euclidean(self):
        points = torch.tensor(
            [[0.0, 0.0], [3.0, 0.0]],
            dtype=torch.float64,
        )
        means = torch.zeros((2, 2), dtype=torch.float64)
        rotations = torch.eye(
            2, dtype=torch.float64
        ).repeat(2, 1, 1)
        scales = torch.ones((2, 2), dtype=torch.float64)
        distances = GroundGrid.point_to_ellipse_distance(
            points, means, rotations, scales
        )
        np.testing.assert_allclose(
            distances.numpy(), [0.0, 2.0], atol=1e-8
        )

    def test_height_metadata_must_match_projection(self):
        ellipses = ellipse_set([[5.5, 5.5]], [[0.4, 0.4]])
        with self.assertRaisesRegex(ValueError, "height metadata"):
            GroundGrid.from_projected_gaussians(
                ellipses,
                lower_xy=[0.0, 0.0],
                upper_xy=[10.0, 10.0],
                resolution_xy=10,
                footprint_radius=0.0,
                z_floor_scene=0.0,
                robot_height=0.8,
            )


if __name__ == "__main__":
    unittest.main()
