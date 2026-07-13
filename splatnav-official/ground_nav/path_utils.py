"""Ground-path geometry helpers for the corridor planning pipeline."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch


def circumscribed_sphere_radius(footprint_radius: float, robot_height: float) -> float:
    if footprint_radius < 0 or robot_height <= 0:
        raise ValueError("footprint_radius must be non-negative and robot_height positive")
    return float(np.hypot(footprint_radius, robot_height / 2.0))


def gaussian_height_mask(gsplat, z_min: float, z_max: float) -> torch.Tensor:
    """Select 1-sigma Gaussian AABBs overlapping the navigable body band."""
    if z_min >= z_max:
        raise ValueError("z_min must be smaller than z_max")
    z_sigma = torch.sqrt(torch.clamp(gsplat.covs[:, 2, 2], min=0.0))
    gaussian_min = gsplat.means[:, 2] - z_sigma
    gaussian_max = gsplat.means[:, 2] + z_sigma
    return (gaussian_max > z_min) & (gaussian_min < z_max)


def line_of_sight_free(grid, start_xy, goal_xy) -> bool:
    """Conservatively check all cells touched by a world-space line segment."""
    start = np.asarray(start_xy, dtype=np.float64)
    goal = np.asarray(goal_xy, dtype=np.float64)
    distance = np.linalg.norm(goal - start)
    spacing = 0.25 * float(torch.min(grid.cell_sizes).item())
    count = max(2, int(np.ceil(distance / spacing)) + 1)
    points = start[None, :] + np.linspace(0.0, 1.0, count)[:, None] * (goal - start)[None, :]
    occupied = grid.occupied.detach().cpu().numpy()
    for point in points:
        index = grid.world_to_grid(point).detach().cpu().numpy()
        # Include immediate neighbors near a cell boundary for a conservative
        # supercover approximation.
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                i, j = int(index[0] + di), int(index[1] + dj)
                if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1] and occupied[i, j]:
                    center = grid.grid_to_world([i, j]).detach().cpu().numpy()
                    half = grid.cell_sizes.detach().cpu().numpy() / 2
                    closest = np.maximum(np.abs(point - center) - half, 0.0)
                    if np.linalg.norm(closest) <= spacing:
                        return False
    return True


def simplify_ground_path(
    path: np.ndarray,
    grid,
    max_segment_length: float,
    exact_segment_validator: Callable[[np.ndarray], bool] | None = None,
) -> np.ndarray:
    """Greedily select the farthest visible and exactly safe waypoint."""
    path = np.asarray(path)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        raise ValueError("path must have shape (N, 3), N >= 2")
    if max_segment_length <= 0:
        raise ValueError("max_segment_length must be positive")
    simplified = [path[0]]
    root = 0
    while root < len(path) - 1:
        chosen = None
        for candidate in range(len(path) - 1, root, -1):
            segment = path[[root, candidate]]
            if np.linalg.norm(segment[1, :2] - segment[0, :2]) > max_segment_length + 1e-9:
                continue
            if not line_of_sight_free(grid, segment[0, :2], segment[1, :2]):
                continue
            if exact_segment_validator is not None and not exact_segment_validator(segment):
                continue
            chosen = candidate
            break
        if chosen is None:
            raise RuntimeError(f"no safe simplified segment can leave path index {root}")
        simplified.append(path[chosen])
        root = chosen
    return np.asarray(simplified, dtype=path.dtype)
