from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "splatnav-official"
sys.path.insert(0, str(UPSTREAM))

from ground_nav.bezier_2d import BezierPlanner2D  # noqa: E402
from ground_nav.corridor_2d import (  # noqa: E402
    PlanarCollisionSet,
    build_planar_corridor,
    compute_rotated_rectangle,
    continuous_circle_ellipse_test,
)
from ground_nav.path_utils import simplify_ground_path  # noqa: E402
from ground_nav.planar_gaussians import PlanarGaussianSet  # noqa: E402


class FakeGrid:
    def __init__(self):
        self.occupied = torch.zeros((20, 20), dtype=torch.bool)
        self.occupied[10, 5:15] = True
        self.cell_sizes = torch.tensor([0.1, 0.1])
        self.shape = (20, 20)

    def world_to_grid(self, point):
        return torch.round(torch.as_tensor(point) / self.cell_sizes).long().clamp(0, 19)

    def grid_to_world(self, index):
        return torch.as_tensor(index, dtype=torch.float32) * self.cell_sizes


def ellipse_set(means, scales):
    means = torch.tensor(means, dtype=torch.float32)
    scales = torch.tensor(scales, dtype=torch.float32)
    count = means.shape[0]
    rots = torch.eye(2).repeat(count, 1, 1)
    return PlanarGaussianSet(
        ids=torch.arange(count), means=means, covs=torch.diag_embed(scales.square()),
        rots=rots, scales=scales, z_min=0.0, z_max=1.0, confidence=1.0,
    )


class Milestone2DTest(unittest.TestCase):
    def test_height_filter_and_orthogonal_projection(self):
        covs = torch.tensor([
            [[4.0, 1.0, 0.2], [1.0, 2.0, 0.1], [0.2, 0.1, 0.04]],
            torch.eye(3),
        ])
        gsplat = SimpleNamespace(means=torch.tensor([[1.0, 2.0, 0.5], [3.0, 4.0, 3.0]]), covs=covs)
        projected = PlanarGaussianSet.from_gsplat(gsplat, 0.2, 0.8)
        self.assertEqual(len(projected), 1)
        np.testing.assert_allclose(projected.means.numpy(), [[1.0, 2.0]])
        np.testing.assert_allclose(projected.covs.numpy(), covs[:1, :2, :2].numpy())

    def test_rotated_rectangle_is_deterministic_and_contains_segment(self):
        segment = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        first = compute_rotated_rectangle(segment, 0.3)
        second = compute_rotated_rectangle(segment, 0.3)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)
        self.assertTrue(torch.all(first[0] @ segment.T <= first[1][:, None] + 1e-7))

    def test_continuous_circle_ellipse_collision(self):
        ellipses = ellipse_set([[0.0, 0.0]], [[0.25, 0.15]])
        crossing = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
        clear = torch.tensor([[-1.0, 1.0], [1.0, 1.0]])
        hit = continuous_circle_ellipse_test(crossing, ellipses.rots, ellipses.scales, ellipses.means, 0.1)
        safe = continuous_circle_ellipse_test(clear, ellipses.rots, ellipses.scales, ellipses.means, 0.1)
        self.assertFalse(bool(hit["is_not_intersect"][0]))
        self.assertTrue(bool(safe["is_not_intersect"][0]))

    def test_empty_candidate_set_has_complete_result(self):
        ellipses = PlanarGaussianSet(
            ids=torch.empty(0, dtype=torch.long), means=torch.empty((0, 2)), covs=torch.empty((0, 2, 2)),
            rots=torch.empty((0, 2, 2)), scales=torch.empty((0, 2)), z_min=0.0, z_max=1.0, confidence=1.0,
        )
        result = continuous_circle_ellipse_test(
            torch.tensor([[0.0, 0.0], [1.0, 0.0]]), ellipses.rots, ellipses.scales, ellipses.means, 0.1
        )
        self.assertEqual(result["deltas"].shape, (0, 2))
        self.assertEqual(result["Q_opt"].shape, (0, 2, 2))

    def test_simplification_uses_grid_line_of_sight_only(self):
        grid = FakeGrid()
        path = np.array([[0.1, 0.1], [0.5, 0.1], [0.9, 0.1], [0.9, 1.7], [1.5, 1.7]])
        simplified = simplify_ground_path(path, grid, 2.0)
        np.testing.assert_allclose(simplified[[0, -1]], path[[0, -1]])
        self.assertGreater(len(simplified), 2)

    def test_paper_style_corridor_contains_seed(self):
        ellipses = ellipse_set([[0.0, 0.35], [0.0, -0.35]], [[0.1, 0.1], [0.1, 0.1]])
        collision_set = PlanarCollisionSet(ellipses, radius=0.05, corridor_margin=0.5)
        path = np.array([[-0.8, 0.0], [0.0, 0.0], [0.8, 0.0]], dtype=np.float32)
        corridor = build_planar_corridor(path, collision_set)
        self.assertGreaterEqual(len(corridor.polygons), 1)
        for polygon, source in zip(corridor.polygons, corridor.source_segment_indices):
            segment = torch.tensor(path[source:source + 2])
            self.assertTrue(torch.all(polygon[0] @ segment.T <= polygon[1][:, None] + 1e-6))

    def test_2d_bezier_qp(self):
        A = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        polygons = [(A, torch.tensor([0.2, 1.0, 1.0, 1.0])), (A, torch.tensor([1.0, 0.2, 1.0, 1.0]))]
        planner = BezierPlanner2D(degree=6, continuity_order=3)
        controls, feasible = planner.optimize(polygons, [-0.8, 0.0], [0.8, 0.0])
        self.assertTrue(feasible)
        self.assertEqual(planner.last_solver_status, "Solved")
        np.testing.assert_allclose(controls[0, :, 0], [-0.8, 0.0], atol=1e-6)
        np.testing.assert_allclose(controls[-1, :, -1], [0.8, 0.0], atol=1e-6)
        dense = planner.sample(50)
        self.assertEqual(dense.shape, (2, 50, 2))


if __name__ == "__main__":
    unittest.main()
