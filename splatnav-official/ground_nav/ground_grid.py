"""Ellipse-distance occupancy grid for the planar ground-planning mainline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class GroundGridMetadata:
    """Geometry used to build the single supported ground-planning grid."""

    z_floor_scene: float
    robot_height: float
    footprint_radius: float
    ground_clearance: float
    z_min: float
    z_max: float
    reference_z: float
    confidence: float
    cell_circumradius: float
    source: str = "projected_gaussians"
    rasterization: str = "center_distance_plus_cell_circumradius"


class GroundGrid:
    """Rasterize height-filtered Gaussian ellipses with footprint clearance.

    A cell is occupied when the Euclidean distance from its center to any
    filled projected ellipse is no greater than the robot footprint radius
    plus the cell circumradius.  This is the only map-construction route kept
    for the Hybrid A* ground-planning mainline.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise TypeError(
            "GroundGrid must be built with "
            "GroundGrid.from_projected_gaussians(...)"
        )

    @classmethod
    def from_projected_gaussians(
        cls,
        ellipses: Any,
        lower_xy: Sequence[float] | np.ndarray | torch.Tensor,
        upper_xy: Sequence[float] | np.ndarray | torch.Tensor,
        resolution_xy: int | Sequence[int] | np.ndarray | torch.Tensor,
        footprint_radius: float,
        *,
        z_floor_scene: float,
        robot_height: float,
        ground_clearance: float = 0.0,
        distance_iterations: int = 40,
        max_pair_chunk: int = 1_000_000,
    ) -> "GroundGrid":
        if footprint_radius < 0.0:
            raise ValueError("footprint_radius must be non-negative")
        if robot_height <= 0.0:
            raise ValueError("robot_height must be positive")
        if ground_clearance < 0.0 or ground_clearance >= robot_height:
            raise ValueError(
                "ground_clearance must lie in [0, robot_height)"
            )
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
        if not bool(torch.all(ellipses.scales > 0.0).item()):
            raise ValueError("ellipse scales must be positive")

        device = ellipses.means.device
        dtype = ellipses.means.dtype
        lower = torch.as_tensor(
            lower_xy, dtype=dtype, device=device
        ).flatten()
        upper = torch.as_tensor(
            upper_xy, dtype=dtype, device=device
        ).flatten()
        if lower.numel() != 2 or upper.numel() != 2:
            raise ValueError("lower_xy and upper_xy must contain two values")
        if not bool(torch.all(upper > lower).item()):
            raise ValueError("map bounds must define a positive 2D extent")

        if isinstance(resolution_xy, int):
            resolution = torch.tensor(
                [resolution_xy, resolution_xy],
                dtype=torch.long,
                device=device,
            )
        else:
            resolution = torch.as_tensor(
                resolution_xy, dtype=torch.long, device=device
            ).flatten()
        if resolution.numel() != 2 or not bool(
            torch.all(resolution > 0).item()
        ):
            raise ValueError("resolution_xy must contain two positive integers")

        z_min = float(z_floor_scene) + float(ground_clearance)
        z_max = float(z_floor_scene) + float(robot_height)
        if not np.isclose(z_min, float(ellipses.z_min)) or not np.isclose(
            z_max, float(ellipses.z_max)
        ):
            raise ValueError(
                "height metadata disagrees with projected Gaussian height band"
            )

        obj = cls.__new__(cls)
        obj.ellipses = ellipses
        obj.device = device
        obj.cell_sizes = (upper - lower) / resolution.to(dtype)
        xs = lower[0] + (
            torch.arange(
                int(resolution[0]), device=device, dtype=dtype
            )
            + 0.5
        ) * obj.cell_sizes[0]
        ys = lower[1] + (
            torch.arange(
                int(resolution[1]), device=device, dtype=dtype
            )
            + 0.5
        ) * obj.cell_sizes[1]
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        obj.xy_centers = torch.stack([gx, gy], dim=-1)
        obj.lower_center = obj.xy_centers[0, 0]
        obj.shape = (int(resolution[0]), int(resolution[1]))
        obj.lower_edge = lower
        obj.upper_edge = upper

        cell_circumradius = 0.5 * float(
            torch.linalg.norm(obj.cell_sizes).item()
        )
        obj.raw_occupied, obj.occupied = (
            obj._rasterize_projected_gaussians(
                raw_threshold=cell_circumradius,
                occupied_threshold=(
                    float(footprint_radius) + cell_circumradius
                ),
                distance_iterations=distance_iterations,
                max_pair_chunk=max_pair_chunk,
            )
        )
        obj.free = ~obj.occupied
        obj.metadata = GroundGridMetadata(
            z_floor_scene=float(z_floor_scene),
            robot_height=float(robot_height),
            footprint_radius=float(footprint_radius),
            ground_clearance=float(ground_clearance),
            z_min=z_min,
            z_max=z_max,
            reference_z=0.5 * (z_min + z_max),
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
        """Return Euclidean distance from each point to its filled ellipse."""

        if points.shape != means.shape or points.ndim != 2:
            raise ValueError("points and means must have shape (N, 2)")
        if points.shape[1] != 2:
            raise ValueError("points and means must have shape (N, 2)")
        if rots.shape != (points.shape[0], 2, 2):
            raise ValueError("rots must have shape (N, 2, 2)")
        if scales.shape != points.shape:
            raise ValueError("scales must have shape (N, 2)")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if points.shape[0] == 0:
            return torch.empty(
                0, dtype=points.dtype, device=points.device
            )

        local = torch.einsum(
            "nij,nj->ni", rots.transpose(1, 2), points - means
        )
        axis2 = scales.square().clamp_min(torch.finfo(scales.dtype).eps)
        normalized = torch.sum(local.square() / axis2, dim=-1)
        outside = normalized > 1.0
        distances = torch.zeros(
            points.shape[0], dtype=points.dtype, device=points.device
        )
        if not bool(torch.any(outside).item()):
            return distances

        values = local[outside]
        outside_axis2 = axis2[outside]
        lower = torch.zeros(
            values.shape[0], dtype=values.dtype, device=values.device
        )
        upper = torch.sqrt(
            torch.sum(outside_axis2 * values.square(), dim=-1)
        ).clamp_min(torch.finfo(values.dtype).eps)
        for _ in range(iterations):
            multiplier = 0.5 * (lower + upper)
            equation = torch.sum(
                outside_axis2
                * values.square()
                / (outside_axis2 + multiplier[:, None]).square(),
                dim=-1,
            )
            lower = torch.where(equation > 1.0, multiplier, lower)
            upper = torch.where(equation <= 1.0, multiplier, upper)
        closest = (
            outside_axis2
            * values
            / (outside_axis2 + upper[:, None])
        )
        distances[outside] = torch.linalg.norm(
            values - closest, dim=-1
        )
        return distances

    def _rasterize_projected_gaussians(
        self,
        *,
        raw_threshold: float,
        occupied_threshold: float,
        distance_iterations: int,
        max_pair_chunk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = torch.zeros(
            self.shape, dtype=torch.bool, device=self.device
        )
        occupied = torch.zeros_like(raw)
        if len(self.ellipses) == 0:
            return raw, occupied

        means = self.ellipses.means.detach().cpu().numpy()
        covariances = self.ellipses.covs.detach().cpu().numpy()
        lower = self.lower_edge.detach().cpu().numpy()
        cell = self.cell_sizes.detach().cpu().numpy()
        maximum_extent = (
            np.sqrt(
                np.maximum(
                    np.diagonal(
                        covariances, axis1=1, axis2=2
                    ),
                    0.0,
                )
            )
            + occupied_threshold
        )
        minimum_center = means - maximum_extent
        maximum_center = means + maximum_extent
        lower_index = np.ceil(
            (minimum_center - lower[None]) / cell[None] - 0.5
        ).astype(np.int64)
        upper_index = np.floor(
            (maximum_center - lower[None]) / cell[None] - 0.5
        ).astype(np.int64)
        lower_index = np.maximum(lower_index, 0)
        upper_index = np.minimum(
            upper_index,
            np.asarray(self.shape, dtype=np.int64)[None] - 1,
        )

        cell_chunks: list[np.ndarray] = []
        ellipse_chunks: list[np.ndarray] = []
        queued = 0

        def flush() -> None:
            nonlocal queued
            if queued == 0:
                return
            cell_ids_numpy = np.concatenate(cell_chunks)
            ellipse_ids_numpy = np.concatenate(ellipse_chunks)
            for start in range(0, len(cell_ids_numpy), max_pair_chunk):
                stop = min(
                    start + max_pair_chunk, len(cell_ids_numpy)
                )
                cell_ids = torch.as_tensor(
                    cell_ids_numpy[start:stop],
                    dtype=torch.long,
                    device=self.device,
                )
                ellipse_ids = torch.as_tensor(
                    ellipse_ids_numpy[start:stop],
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
                    cell_ids[
                        distances <= occupied_threshold + 1e-12
                    ]
                ] = True
            cell_chunks.clear()
            ellipse_chunks.clear()
            queued = 0

        cells_y = self.shape[1]
        for ellipse_id in range(len(self.ellipses)):
            if np.any(
                lower_index[ellipse_id] > upper_index[ellipse_id]
            ):
                continue
            x_indices = np.arange(
                lower_index[ellipse_id, 0],
                upper_index[ellipse_id, 0] + 1,
                dtype=np.int64,
            )
            y_indices = np.arange(
                lower_index[ellipse_id, 1],
                upper_index[ellipse_id, 1] + 1,
                dtype=np.int64,
            )
            flat = (
                x_indices[:, None] * cells_y + y_indices[None, :]
            ).reshape(-1)
            cell_chunks.append(flat)
            ellipse_chunks.append(
                np.full(flat.shape, ellipse_id, dtype=np.int64)
            )
            queued += flat.size
            if queued >= max_pair_chunk:
                flush()
        flush()
        return raw, occupied

    def world_to_grid(
        self,
        point_xy: Sequence[float] | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        point = torch.as_tensor(
            point_xy,
            dtype=self.xy_centers.dtype,
            device=self.device,
        ).flatten()
        if point.numel() < 2:
            raise ValueError("point_xy must contain x and y")
        indices = torch.round(
            (point[:2] - self.lower_center) / self.cell_sizes
        ).to(torch.long)
        maximum = (
            torch.tensor(
                self.shape, device=self.device, dtype=torch.long
            )
            - 1
        )
        return torch.minimum(
            torch.maximum(indices, torch.zeros_like(indices)),
            maximum,
        )

    def grid_to_world(
        self,
        index_xy: Sequence[int] | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        index = torch.as_tensor(
            index_xy, dtype=torch.long, device=self.device
        ).flatten()
        if index.numel() < 2:
            raise ValueError("index_xy must contain two indices")
        index = index[:2]
        if bool(torch.any(index < 0)) or bool(
            index[0] >= self.shape[0] or index[1] >= self.shape[1]
        ):
            raise IndexError(
                f"grid index {index.tolist()} is outside shape {self.shape}"
            )
        return self.xy_centers[index[0], index[1]]

    def is_occupied(
        self,
        point_xy: Sequence[float] | np.ndarray | torch.Tensor,
    ) -> bool:
        point = torch.as_tensor(
            point_xy, dtype=self.xy_centers.dtype, device=self.device
        ).flatten()[:2]
        if bool(torch.any(point < self.lower_edge)) or bool(
            torch.any(point >= self.upper_edge)
        ):
            return True
        index = self.world_to_grid(point)
        return bool(self.occupied[index[0], index[1]].item())

    def summary(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "raw_occupied_cells": int(
                self.raw_occupied.sum().item()
            ),
            "occupied_cells": int(self.occupied.sum().item()),
            "free_cells": int(self.free.sum().item()),
            "occupied_fraction": float(
                self.occupied.float().mean().item()
            ),
            "cell_sizes_xy": [
                float(value)
                for value in self.cell_sizes.detach().cpu().tolist()
            ],
            "z_floor_scene": self.metadata.z_floor_scene,
            "robot_height": self.metadata.robot_height,
            "footprint_radius": self.metadata.footprint_radius,
            "ground_clearance": self.metadata.ground_clearance,
            "projection_z_min": self.metadata.z_min,
            "projection_z_max": self.metadata.z_max,
            "reference_z": self.metadata.reference_z,
            "source": self.metadata.source,
            "rasterization": self.metadata.rasterization,
            "confidence": self.metadata.confidence,
            "cell_circumradius": self.metadata.cell_circumradius,
        }
