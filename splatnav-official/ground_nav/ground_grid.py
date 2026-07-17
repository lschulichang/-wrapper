"""Project a 3D Splat-Nav voxel grid into a planar planning grid."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GroundGridMetadata:
    """Description of the height-band projection used to build a ground grid."""

    z_floor_scene: float
    robot_height: float
    footprint_radius: float
    ground_clearance: float
    z_min: float
    z_max: float
    reference_z: float
    z_indices: tuple[int, ...]
    source: str = "legacy_voxel_projection"
    rasterization: str = "voxel_height_projection_xy_dilation"
    confidence: float | None = None
    cell_circumradius: float | None = None

    @property
    def z_floor(self) -> float:
        """Deprecated compatibility alias for the scene-coordinate floor Z."""

        return self.z_floor_scene


class GroundGrid:
    """A 2D occupancy grid projected from an existing 3D ``Voxel`` object.

    The input must be an *uninflated* 3D occupancy grid. Obstacles intersecting
    ``[z_floor_scene + ground_clearance, z_floor_scene + robot_height]`` are projected with
    a logical OR along Z, then dilated only in XY by the circular footprint.
    """

    def __init__(
        self,
        voxel_grid: Any,
        z_floor_scene: float | None = None,
        robot_height: float | None = None,
        footprint_radius: float | None = None,
        *,
        ground_clearance: float = 0.0,
        project_occupied_endpoints: bool = True,
        z_floor: float | None = None,
    ) -> None:
        if z_floor_scene is None:
            if z_floor is None:
                raise ValueError("z_floor_scene must be provided")
            z_floor_scene = float(z_floor)
        elif z_floor is not None:
            raise ValueError("provide only z_floor_scene; z_floor is a deprecated alias")
        if robot_height is None or footprint_radius is None:
            raise ValueError("robot_height and footprint_radius must be provided")
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

        z_min = float(z_floor_scene + ground_clearance)
        z_max = float(z_floor_scene + robot_height)
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
            z_floor_scene=float(z_floor_scene),
            robot_height=float(robot_height),
            footprint_radius=float(footprint_radius),
            ground_clearance=float(ground_clearance),
            z_min=z_min,
            z_max=z_max,
            reference_z=float(z_floor_scene + robot_height / 2),
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

    @classmethod
    def from_projected_gaussians(
        cls,
        ellipses: Any,
        lower_xy: Sequence[float] | np.ndarray | torch.Tensor,
        upper_xy: Sequence[float] | np.ndarray | torch.Tensor,
        resolution_xy: int | Sequence[int] | np.ndarray | torch.Tensor,
        footprint_radius: float,
        *,
        z_floor_scene: float | None = None,
        robot_height: float | None = None,
        ground_clearance: float = 0.0,
        project_occupied_endpoints: bool = False,
        distance_iterations: int = 40,
        max_pair_chunk: int = 1_000_000,
    ) -> "GroundGrid":
        """Rasterize height-filtered projected ellipses conservatively.

        A cell is occupied when the Euclidean distance from its center to any
        projected ellipse is no greater than the requested footprint radius
        plus the cell circumradius. The circumradius makes this a conservative
        rectangle-intersection test rather than a center-only approximation.
        """

        if footprint_radius < 0:
            raise ValueError("footprint_radius must be non-negative")
        if distance_iterations <= 0:
            raise ValueError("distance_iterations must be positive")
        if max_pair_chunk <= 0:
            raise ValueError("max_pair_chunk must be positive")
        if ellipses.means.ndim != 2 or ellipses.means.shape[1] != 2:
            raise ValueError("ellipses.means must have shape (N, 2)")
        if ellipses.rots.shape != (len(ellipses), 2, 2):
            raise ValueError("ellipses.rots must have shape (N, 2, 2)")
        if ellipses.scales.shape != (len(ellipses), 2):
            raise ValueError("ellipses.scales must have shape (N, 2)")
        if not bool(torch.all(torch.isfinite(ellipses.means)).item()):
            raise ValueError("ellipse means must be finite")
        if not bool(torch.all(torch.isfinite(ellipses.scales)).item()):
            raise ValueError("ellipse scales must be finite")
        if not bool(torch.all(ellipses.scales > 0).item()):
            raise ValueError("ellipse scales must be positive")

        device = ellipses.means.device
        dtype = ellipses.means.dtype
        lower = torch.as_tensor(lower_xy, dtype=dtype, device=device).flatten()
        upper = torch.as_tensor(upper_xy, dtype=dtype, device=device).flatten()
        if lower.numel() != 2 or upper.numel() != 2:
            raise ValueError("lower_xy and upper_xy must each contain two values")
        if not bool(torch.all(upper > lower).item()):
            raise ValueError("lower_xy and upper_xy must define a positive 2D extent")
        if isinstance(resolution_xy, int):
            resolution = torch.tensor(
                [resolution_xy, resolution_xy], device=device, dtype=torch.long
            )
        else:
            resolution = torch.as_tensor(
                resolution_xy, device=device, dtype=torch.long
            ).flatten()
        if resolution.numel() != 2 or not bool(torch.all(resolution > 0).item()):
            raise ValueError("resolution_xy must contain two positive integers")

        obj = cls.__new__(cls)
        obj.voxel_grid = None
        obj.ellipses = ellipses
        obj.device = device
        obj.project_occupied_endpoints = project_occupied_endpoints
        obj.cell_sizes = (upper - lower) / resolution.to(dtype)
        xs = lower[0] + (
            torch.arange(int(resolution[0]), device=device, dtype=dtype) + 0.5
        ) * obj.cell_sizes[0]
        ys = lower[1] + (
            torch.arange(int(resolution[1]), device=device, dtype=dtype) + 0.5
        ) * obj.cell_sizes[1]
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        obj.xy_centers = torch.stack([gx, gy], dim=-1)
        obj.lower_center = obj.xy_centers[0, 0]
        obj.shape = (int(resolution[0]), int(resolution[1]))
        cell_circumradius = 0.5 * float(torch.linalg.norm(obj.cell_sizes).item())
        obj.raw_occupied, obj.occupied = obj._rasterize_projected_gaussians(
            raw_threshold=cell_circumradius,
            occupied_threshold=float(footprint_radius) + cell_circumradius,
            distance_iterations=distance_iterations,
            max_pair_chunk=max_pair_chunk,
        )
        obj.free = ~obj.occupied
        obj.dilation_kernel = None
        z_min = float(ellipses.z_min)
        z_max = float(ellipses.z_max)
        if z_floor_scene is None:
            z_floor_scene = z_min
        if robot_height is None:
            robot_height = z_max - float(z_floor_scene)
        expected_z_min = float(z_floor_scene) + float(ground_clearance)
        expected_z_max = float(z_floor_scene) + float(robot_height)
        if not np.isclose(expected_z_min, z_min) or not np.isclose(expected_z_max, z_max):
            raise ValueError(
                "height metadata disagrees with the projected Gaussian height band"
            )
        obj.metadata = GroundGridMetadata(
            z_floor_scene=float(z_floor_scene),
            robot_height=float(robot_height),
            footprint_radius=float(footprint_radius),
            ground_clearance=float(ground_clearance),
            z_min=z_min,
            z_max=z_max,
            reference_z=0.5 * (z_min + z_max),
            z_indices=(),
            source="projected_gaussians",
            rasterization="center_distance_plus_cell_circumradius",
            confidence=float(ellipses.confidence),
            cell_circumradius=cell_circumradius,
        )
        return obj

    @staticmethod
    def point_to_ellipse_distance(
        points: torch.Tensor,
        means: torch.Tensor,
        rots: torch.Tensor,
        scales: torch.Tensor,
        iterations: int = 40,
    ) -> torch.Tensor:
        """Return Euclidean distances from points to filled 2D ellipses."""

        if points.shape != means.shape or points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points and means must both have shape (N, 2)")
        if rots.shape != (points.shape[0], 2, 2):
            raise ValueError("rots must have shape (N, 2, 2)")
        if scales.shape != points.shape:
            raise ValueError("scales must have shape (N, 2)")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if points.shape[0] == 0:
            return torch.empty(0, dtype=points.dtype, device=points.device)

        local = torch.einsum("nij,nj->ni", rots.transpose(1, 2), points - means)
        axis2 = scales.square().clamp_min(torch.finfo(scales.dtype).eps)
        normalized = torch.sum(local.square() / axis2, dim=-1)
        outside = normalized > 1.0
        distances = torch.zeros(points.shape[0], dtype=points.dtype, device=points.device)
        if not bool(torch.any(outside).item()):
            return distances

        y = local[outside]
        a2 = axis2[outside]
        lo = torch.zeros(y.shape[0], dtype=y.dtype, device=y.device)
        # This upper bound guarantees sum(a_i^2 y_i^2 / lambda^2) <= 1.
        hi = torch.sqrt(torch.sum(a2 * y.square(), dim=-1)).clamp_min(
            torch.finfo(y.dtype).eps
        )
        for _ in range(iterations):
            lam = 0.5 * (lo + hi)
            value = torch.sum(
                a2 * y.square() / (a2 + lam[:, None]).square(), dim=-1
            )
            lo = torch.where(value > 1.0, lam, lo)
            hi = torch.where(value <= 1.0, lam, hi)
        closest = a2 * y / (a2 + hi[:, None])
        distances[outside] = torch.linalg.norm(y - closest, dim=-1)
        return distances

    def _rasterize_projected_gaussians(
        self,
        *,
        raw_threshold: float,
        occupied_threshold: float,
        distance_iterations: int,
        max_pair_chunk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = torch.zeros(self.shape, dtype=torch.bool, device=self.device)
        occupied = torch.zeros_like(raw)
        if len(self.ellipses) == 0:
            return raw, occupied

        means = self.ellipses.means.detach().cpu().numpy()
        covs = self.ellipses.covs.detach().cpu().numpy()
        lower = (
            self.lower_center - 0.5 * self.cell_sizes
        ).detach().cpu().numpy()
        cell = self.cell_sizes.detach().cpu().numpy()
        max_extent = (
            np.sqrt(
                np.maximum(np.diagonal(covs, axis1=1, axis2=2), 0.0)
            )
            + occupied_threshold
        )
        min_center = means - max_extent
        max_center = means + max_extent
        lo = np.ceil((min_center - lower[None]) / cell[None] - 0.5).astype(
            np.int64
        )
        hi = np.floor((max_center - lower[None]) / cell[None] - 0.5).astype(
            np.int64
        )
        lo = np.maximum(lo, 0)
        hi = np.minimum(
            hi, np.asarray(self.shape, dtype=np.int64)[None] - 1
        )

        cell_chunks: list[np.ndarray] = []
        ellipse_chunks: list[np.ndarray] = []
        queued = 0

        def flush() -> None:
            nonlocal queued
            if queued == 0:
                return
            cell_ids_np = np.concatenate(cell_chunks)
            ellipse_ids_np = np.concatenate(ellipse_chunks)
            for start in range(0, len(cell_ids_np), max_pair_chunk):
                stop = min(start + max_pair_chunk, len(cell_ids_np))
                cell_ids = torch.as_tensor(
                    cell_ids_np[start:stop],
                    dtype=torch.long,
                    device=self.device,
                )
                ellipse_ids = torch.as_tensor(
                    ellipse_ids_np[start:stop],
                    dtype=torch.long,
                    device=self.device,
                )
                points = self.xy_centers.reshape(-1, 2)[cell_ids]
                distances = self.point_to_ellipse_distance(
                    points,
                    self.ellipses.means[ellipse_ids],
                    self.ellipses.rots[ellipse_ids],
                    self.ellipses.scales[ellipse_ids],
                    distance_iterations,
                )
                raw.view(-1)[
                    cell_ids[distances <= raw_threshold + 1e-12]
                ] = True
                occupied.view(-1)[
                    cell_ids[distances <= occupied_threshold + 1e-12]
                ] = True
            cell_chunks.clear()
            ellipse_chunks.clear()
            queued = 0

        ny = self.shape[1]
        for ellipse_id in range(len(self.ellipses)):
            if np.any(lo[ellipse_id] > hi[ellipse_id]):
                continue
            ix = np.arange(
                lo[ellipse_id, 0], hi[ellipse_id, 0] + 1, dtype=np.int64
            )
            iy = np.arange(
                lo[ellipse_id, 1], hi[ellipse_id, 1] + 1, dtype=np.int64
            )
            flat = (ix[:, None] * ny + iy[None, :]).reshape(-1)
            cell_chunks.append(flat)
            ellipse_chunks.append(
                np.full(flat.shape, ellipse_id, dtype=np.int64)
            )
            queued += flat.size
            if queued >= max_pair_chunk:
                flush()
        flush()
        return raw, occupied

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
        """Plan directly in the projected plane and return an ``N x 2`` path."""

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

        source = tuple(int(v) for v in self.world_to_grid(start).cpu().tolist())
        target = tuple(int(v) for v in self.world_to_grid(goal).cpu().tolist())
        indices = self._dijkstra_2d(source, target)
        if len(indices) == 0:
            raise RuntimeError("no feasible path exists in the projected ground grid")

        indices = np.asarray(indices, dtype=np.int32)
        return self.xy_centers[
            torch.as_tensor(indices[:, 0], device=self.device),
            torch.as_tensor(indices[:, 1], device=self.device),
        ].detach().cpu().numpy()

    def _dijkstra_2d(self, source: tuple[int, int], target: tuple[int, int]) -> list[tuple[int, int]]:
        """Metric-weighted four-neighbor Dijkstra on the 2D occupancy array."""

        occupied = self.occupied.detach().cpu().numpy()
        if occupied[source] or occupied[target]:
            return []
        distances = np.full(self.shape, np.inf, dtype=np.float64)
        parents = np.full((*self.shape, 2), -1, dtype=np.int32)
        distances[source] = 0.0
        serial = 0
        queue: list[tuple[float, int, int, int]] = [(0.0, serial, source[0], source[1])]
        dx, dy = (float(v) for v in self.cell_sizes.detach().cpu().tolist())
        neighbors = ((1, 0, dx), (0, 1, dy), (-1, 0, dx), (0, -1, dy))

        while queue:
            cost, _, i, j = heapq.heappop(queue)
            if cost > distances[i, j] + 1e-12:
                continue
            if (i, j) == target:
                break
            for di, dj, step_cost in neighbors:
                ni, nj = i + di, j + dj
                if not (0 <= ni < self.shape[0] and 0 <= nj < self.shape[1]):
                    continue
                if occupied[ni, nj]:
                    continue
                candidate = cost + step_cost
                if candidate + 1e-12 < distances[ni, nj]:
                    distances[ni, nj] = candidate
                    parents[ni, nj] = (i, j)
                    serial += 1
                    heapq.heappush(queue, (candidate, serial, ni, nj))

        if not np.isfinite(distances[target]):
            return []
        path = [target]
        current = target
        while current != source:
            parent = parents[current]
            if parent[0] < 0:
                return []
            current = (int(parent[0]), int(parent[1]))
            path.append(current)
        path.reverse()
        return path

    def summary(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "raw_occupied_cells": int(self.raw_occupied.sum().item()),
            "occupied_cells": int(self.occupied.sum().item()),
            "free_cells": int(self.free.sum().item()),
            "occupied_fraction": float(self.occupied.float().mean().item()),
            "cell_sizes_xy": [float(v) for v in self.cell_sizes.detach().cpu().tolist()],
            "z_floor_scene": self.metadata.z_floor_scene,
            "z_floor": self.metadata.z_floor_scene,
            "robot_height": self.metadata.robot_height,
            "footprint_radius": self.metadata.footprint_radius,
            "ground_clearance": self.metadata.ground_clearance,
            "projection_z_min": self.metadata.z_min,
            "projection_z_max": self.metadata.z_max,
            "reference_z": self.metadata.reference_z,
            "z_indices": list(self.metadata.z_indices),
            "source": self.metadata.source,
            "rasterization": self.metadata.rasterization,
            "confidence": self.metadata.confidence,
            "cell_circumradius": self.metadata.cell_circumradius,
            "search_backend": "native_2d_dijkstra_4_connected",
        }
