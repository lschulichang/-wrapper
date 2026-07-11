"""Project a 3D Splat-Nav voxel grid into a planar planning grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import dijkstra3d
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GroundGridMetadata:
    """Description of the height-band projection used to build a ground grid."""

    z_floor: float
    robot_height: float
    footprint_radius: float
    ground_clearance: float
    z_min: float
    z_max: float
    reference_z: float
    z_indices: tuple[int, ...]


class GroundGrid:
    """A 2D occupancy grid projected from an existing 3D ``Voxel`` object.

    The input must be an *uninflated* 3D occupancy grid. Obstacles intersecting
    ``[z_floor + ground_clearance, z_floor + robot_height]`` are projected with
    a logical OR along Z, then dilated only in XY by the circular footprint.
    """

    def __init__(
        self,
        voxel_grid: Any,
        z_floor: float,
        robot_height: float,
        footprint_radius: float,
        *,
        ground_clearance: float = 0.0,
        project_occupied_endpoints: bool = True,
    ) -> None:
        if robot_height <= 0:
            raise ValueError("robot_height must be positive")
        if footprint_radius < 0:
            raise ValueError("footprint_radius must be non-negative")
        if ground_clearance < 0:
            raise ValueError("ground_clearance must be non-negative")
        if ground_clearance >= robot_height:
            raise ValueError("ground_clearance must be smaller than robot_height")
        if voxel_grid.non_navigable_grid is None or voxel_grid.grid_centers is None:
            raise ValueError("voxel_grid must have initialized occupancy and centers")
        if voxel_grid.non_navigable_grid.ndim != 3:
            raise ValueError("voxel_grid.non_navigable_grid must be a 3D tensor")
        source_radius = float(getattr(voxel_grid, "radius", 0.0))
        if abs(source_radius) > 1e-9:
            raise ValueError(
                "GroundGrid requires an uninflated voxel grid (radius=0); "
                f"received radius={source_radius}"
            )

        self.voxel_grid = voxel_grid
        self.device = voxel_grid.non_navigable_grid.device
        self.project_occupied_endpoints = project_occupied_endpoints

        z_min = float(z_floor + ground_clearance)
        z_max = float(z_floor + robot_height)
        z_centers = voxel_grid.grid_centers[0, 0, :, 2]
        half_cell_z = voxel_grid.cell_sizes[2] / 2
        z_cell_min = z_centers - half_cell_z
        z_cell_max = z_centers + half_cell_z
        grid_z_min = float(z_cell_min[0].item())
        grid_z_max = float(z_cell_max[-1].item())
        tolerance = 1e-6
        if z_min < grid_z_min - tolerance or z_max > grid_z_max + tolerance:
            raise ValueError(
                f"height band [{z_min:.6f}, {z_max:.6f}] is not fully covered "
                f"by voxel bounds [{grid_z_min:.6f}, {grid_z_max:.6f}]"
            )
        # Strict overlap avoids selecting a neighboring cell that only touches
        # the height band at a zero-volume boundary.
        z_mask = (z_cell_max > z_min) & (z_cell_min < z_max)
        z_indices = torch.nonzero(z_mask, as_tuple=False).flatten()

        if z_indices.numel() == 0:
            bounds = (float(z_centers[0].item()), float(z_centers[-1].item()))
            raise ValueError(
                f"height band [{z_min:.6f}, {z_max:.6f}] intersects no voxel "
                f"cells; available z centers span [{bounds[0]:.6f}, {bounds[1]:.6f}]"
            )

        self.metadata = GroundGridMetadata(
            z_floor=float(z_floor),
            robot_height=float(robot_height),
            footprint_radius=float(footprint_radius),
            ground_clearance=float(ground_clearance),
            z_min=z_min,
            z_max=z_max,
            reference_z=float(z_floor + robot_height / 2),
            z_indices=tuple(int(i) for i in z_indices.cpu().tolist()),
        )

        self.xy_centers = voxel_grid.grid_centers[:, :, 0, :2]
        self.cell_sizes = voxel_grid.cell_sizes[:2]
        self.lower_center = self.xy_centers[0, 0]

        selected = voxel_grid.non_navigable_grid.index_select(2, z_indices)
        self.raw_occupied = torch.any(selected, dim=2)
        self.dilation_kernel = self._make_disk_kernel(float(footprint_radius))
        self.occupied = self._dilate(self.raw_occupied, self.dilation_kernel)
        self.free = ~self.occupied
        self.shape = tuple(int(v) for v in self.occupied.shape)

    def _make_disk_kernel(self, radius: float) -> torch.Tensor:
        """Build a conservative disk kernel for possibly non-square cells.

        A neighboring cell is included when its rectangular area intersects the
        radius-``r`` disk, rather than only when its center lies inside the disk.
        """

        dx, dy = (float(v) for v in self.cell_sizes.detach().cpu().tolist())
        nx = int(np.ceil(radius / dx + 0.5))
        ny = int(np.ceil(radius / dy + 0.5))
        ix = torch.arange(-nx, nx + 1, device=self.device, dtype=self.xy_centers.dtype)
        iy = torch.arange(-ny, ny + 1, device=self.device, dtype=self.xy_centers.dtype)
        gx, gy = torch.meshgrid(ix, iy, indexing="ij")
        closest_x = torch.clamp(torch.abs(gx) * dx - dx / 2, min=0.0)
        closest_y = torch.clamp(torch.abs(gy) * dy - dy / 2, min=0.0)
        return closest_x.square() + closest_y.square() <= radius**2 + 1e-12

    @staticmethod
    def _dilate(occupied: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        values = occupied.to(torch.float32)[None, None]
        weights = kernel.to(torch.float32)[None, None]
        padding = (kernel.shape[0] // 2, kernel.shape[1] // 2)
        return F.conv2d(values, weights, padding=padding).squeeze(0).squeeze(0) > 0

    @property
    def occupancy_2d(self) -> torch.Tensor:
        """Compatibility alias for the projected occupancy tensor."""

        return self.occupied

    def world_to_grid(self, point_xy: Sequence[float] | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Map an XY world coordinate to the nearest valid grid index."""

        point = torch.as_tensor(point_xy, dtype=self.xy_centers.dtype, device=self.device)
        if point.numel() < 2:
            raise ValueError("point_xy must contain at least x and y")
        point = point.flatten()[:2]

        indices = torch.round((point - self.lower_center) / self.cell_sizes).to(torch.long)
        max_indices = torch.tensor(self.shape, device=self.device, dtype=torch.long) - 1
        return torch.minimum(torch.maximum(indices, torch.zeros_like(indices)), max_indices)

    def grid_to_world(self, index_xy: Sequence[int] | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return the XY center of a planar grid cell."""

        index = torch.as_tensor(index_xy, dtype=torch.long, device=self.device).flatten()
        if index.numel() < 2:
            raise ValueError("index_xy must contain two indices")
        index = index[:2]
        if bool(torch.any(index < 0)) or index[0] >= self.shape[0] or index[1] >= self.shape[1]:
            raise IndexError(f"grid index {index.tolist()} is outside shape {self.shape}")
        return self.xy_centers[index[0], index[1]]

    def is_occupied(self, point_xy: Sequence[float] | np.ndarray | torch.Tensor) -> bool:
        index = self.world_to_grid(point_xy)
        return bool(self.occupied[index[0], index[1]].item())

    def find_closest_navigable(self, point_xy: Sequence[float] | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return the nearest free cell center to a world coordinate."""

        free_indices = torch.nonzero(self.free, as_tuple=False)
        if free_indices.numel() == 0:
            raise RuntimeError("ground grid contains no navigable cells")

        point = torch.as_tensor(point_xy, dtype=self.xy_centers.dtype, device=self.device).flatten()[:2]
        free_centers = self.xy_centers[free_indices[:, 0], free_indices[:, 1]]
        closest = torch.argmin(torch.linalg.norm(free_centers - point[None, :], dim=1))
        return free_centers[closest]

    def create_path(
        self,
        start_xy: Sequence[float] | np.ndarray | torch.Tensor,
        goal_xy: Sequence[float] | np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Plan a 4-connected planar path and return an ``N x 3`` world path.

        The returned Z coordinate is the cylinder center
        ``z_floor + robot_height / 2``. Using a depth-one 3D field with
        connectivity 6 gives four planar neighbors and reuses Splat-Nav's
        existing ``dijkstra3d`` dependency.
        """

        start = torch.as_tensor(start_xy, dtype=self.xy_centers.dtype, device=self.device).flatten()[:2]
        goal = torch.as_tensor(goal_xy, dtype=self.xy_centers.dtype, device=self.device).flatten()[:2]

        if self.is_occupied(start):
            if not self.project_occupied_endpoints:
                raise ValueError("start lies in an occupied ground-grid cell")
            start = self.find_closest_navigable(start)
        if self.is_occupied(goal):
            if not self.project_occupied_endpoints:
                raise ValueError("goal lies in an occupied ground-grid cell")
            goal = self.find_closest_navigable(goal)

        source_2d = self.world_to_grid(start).cpu().numpy().astype(np.int32)
        target_2d = self.world_to_grid(goal).cpu().numpy().astype(np.int32)
        source = np.array([source_2d[0], source_2d[1], 0], dtype=np.int32)
        target = np.array([target_2d[0], target_2d[1], 0], dtype=np.int32)
        field = self.occupied.detach().cpu().numpy()[:, :, None]

        try:
            indices = dijkstra3d.binary_dijkstra(
                field,
                source,
                target,
                connectivity=6,
                background_color=1,
            )
        except Exception as exc:
            raise RuntimeError("no feasible path exists in the projected ground grid") from exc

        if indices is None or len(indices) == 0:
            raise RuntimeError("no feasible path exists in the projected ground grid")

        indices = np.asarray(indices, dtype=np.int32)
        path_xy = self.xy_centers[
            torch.as_tensor(indices[:, 0], device=self.device),
            torch.as_tensor(indices[:, 1], device=self.device),
        ].detach().cpu().numpy()
        z = np.full((path_xy.shape[0], 1), self.metadata.reference_z, dtype=path_xy.dtype)
        return np.concatenate([path_xy, z], axis=1)

    def summary(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "raw_occupied_cells": int(self.raw_occupied.sum().item()),
            "occupied_cells": int(self.occupied.sum().item()),
            "free_cells": int(self.free.sum().item()),
            "occupied_fraction": float(self.occupied.float().mean().item()),
            "cell_sizes_xy": [float(v) for v in self.cell_sizes.detach().cpu().tolist()],
            "z_floor": self.metadata.z_floor,
            "robot_height": self.metadata.robot_height,
            "footprint_radius": self.metadata.footprint_radius,
            "ground_clearance": self.metadata.ground_clearance,
            "projection_z_min": self.metadata.z_min,
            "projection_z_max": self.metadata.z_max,
            "reference_z": self.metadata.reference_z,
            "z_indices": list(self.metadata.z_indices),
        }
