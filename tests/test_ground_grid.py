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

from ground_nav.ground_grid import GroundGrid  # noqa: E402
from initialization.grid_utils import GSplatVoxel  # noqa: E402


def fake_voxel() -> SimpleNamespace:
    nx, ny, nz = 7, 6, 4
    lower = torch.tensor([0.0, 0.0, 0.0])
    upper = torch.tensor([7.0, 6.0, 4.0])
    cell_sizes = (upper - lower) / torch.tensor([nx, ny, nz])
    xs = torch.linspace(0.5, 6.5, nx)
    ys = torch.linspace(0.5, 5.5, ny)
    zs = torch.linspace(0.5, 3.5, nz)
    x, y, z = torch.meshgrid(xs, ys, zs, indexing="ij")
    centers = torch.stack([x, y, z], dim=-1)
    occupied = torch.zeros((nx, ny, nz), dtype=torch.bool)

    # A wall in the selected height band with a single opening at y index 3.
    occupied[3, :, 1] = True
    occupied[3, 3, 1] = False
    # This obstacle is above the selected band and must not be projected.
    occupied[1, 1, 3] = True

    return SimpleNamespace(
        non_navigable_grid=occupied,
        grid_centers=centers,
        cell_sizes=cell_sizes,
        radius=0.0,
    )


class GroundGridTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = GroundGrid(
            fake_voxel(),
            z_floor_scene=1.0,
            robot_height=1.0,
            footprint_radius=0.0,
        )

    def test_height_band_projection(self) -> None:
        self.assertTrue(self.grid.occupied[3, 0])
        self.assertFalse(self.grid.occupied[3, 3])
        self.assertFalse(self.grid.occupied[1, 1])
        self.assertEqual(self.grid.metadata.z_indices, (1,))

    def test_world_grid_round_trip_is_within_half_cell(self) -> None:
        point = torch.tensor([2.7, 4.2])
        recovered = self.grid.grid_to_world(self.grid.world_to_grid(point))
        error = torch.abs(recovered - point)
        self.assertTrue(torch.all(error <= self.grid.cell_sizes / 2 + 1e-6))

    def test_path_is_planar_and_collision_free(self) -> None:
        path = self.grid.create_path([0.5, 0.5], [6.5, 0.5])
        self.assertGreater(len(path), 2)
        self.assertEqual(path.shape[1], 2)
        for point in path:
            self.assertFalse(self.grid.is_occupied(point))
        self.assertTrue(np.any(np.isclose(path[:, 1], 3.5)))

    def test_occupied_endpoint_can_be_projected(self) -> None:
        projected_path = self.grid.create_path([3.5, 0.5], [6.5, 0.5])
        self.assertFalse(self.grid.is_occupied(projected_path[0]))

    def test_path_steps_are_strictly_four_connected(self) -> None:
        path = self.grid.create_path([0.5, 0.5], [6.5, 0.5])
        indices = np.stack([self.grid.world_to_grid(point).cpu().numpy() for point in path])
        self.assertTrue(np.all(np.abs(np.diff(indices, axis=0)).sum(axis=1) == 1))

    def test_invalid_height_band_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GroundGrid(fake_voxel(), z_floor_scene=8.0, robot_height=1.0, footprint_radius=0.0)

    def test_deprecated_z_floor_alias_matches_scene_named_argument(self) -> None:
        legacy = GroundGrid(fake_voxel(), z_floor=1.0, robot_height=1.0, footprint_radius=0.0)
        explicit = GroundGrid(
            fake_voxel(), z_floor_scene=1.0, robot_height=1.0, footprint_radius=0.0
        )
        torch.testing.assert_close(legacy.occupied, explicit.occupied)
        self.assertEqual(explicit.metadata.z_floor_scene, 1.0)
        self.assertEqual(explicit.metadata.z_floor, 1.0)

    def test_ground_clearance_excludes_floor_layer(self) -> None:
        voxel = fake_voxel()
        voxel.non_navigable_grid[2, 2, 0] = True
        grid = GroundGrid(
            voxel,
            z_floor_scene=0.0,
            robot_height=2.0,
            footprint_radius=0.0,
            ground_clearance=1.0,
        )
        self.assertFalse(grid.raw_occupied[2, 2])

    def test_disk_dilation_happens_only_in_xy(self) -> None:
        voxel = fake_voxel()
        voxel.non_navigable_grid.zero_()
        voxel.non_navigable_grid[3, 3, 1] = True
        grid = GroundGrid(
            voxel,
            z_floor_scene=1.0,
            robot_height=1.0,
            footprint_radius=0.6,
        )
        self.assertEqual(int(grid.raw_occupied.sum()), 1)
        self.assertTrue(grid.occupied[3, 3])
        self.assertTrue(grid.occupied[2, 3])
        self.assertTrue(grid.occupied[3, 2])

    def test_preinflated_source_grid_is_rejected(self) -> None:
        voxel = fake_voxel()
        voxel.radius = 0.1
        with self.assertRaises(ValueError):
            GroundGrid(voxel, z_floor_scene=1.0, robot_height=1.0, footprint_radius=0.1)

    def test_gsplat_voxel_supports_uninflated_radius_zero(self) -> None:
        gsplat = SimpleNamespace(
            means=torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32),
            covs=torch.eye(3, dtype=torch.float32).unsqueeze(0) * 0.01,
        )
        voxel = GSplatVoxel(
            gsplat,
            lower_bound=torch.tensor([0.0, 0.0, 0.0]),
            upper_bound=torch.tensor([1.0, 1.0, 1.0]),
            resolution=5,
            radius=0.0,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(voxel.non_navigable_grid.shape), (5, 5, 5))
        self.assertGreater(int(voxel.non_navigable_grid.sum()), 0)


if __name__ == "__main__":
    unittest.main()
