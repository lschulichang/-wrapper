"""Deterministic path simplification on an inflated 2D ground grid."""

from __future__ import annotations

import numpy as np
import torch


def line_of_sight_free(grid, start_xy, goal_xy) -> bool:
    """Conservatively test a segment against every touched inflated cell."""

    start = np.asarray(start_xy, dtype=np.float64)
    goal = np.asarray(goal_xy, dtype=np.float64)
    distance = float(np.linalg.norm(goal - start))
    spacing = 0.25 * float(torch.min(grid.cell_sizes).item())
    count = max(2, int(np.ceil(distance / spacing)) + 1)
    points = start[None] + np.linspace(0.0, 1.0, count)[:, None] * (goal - start)[None]
    occupied = grid.occupied.detach().cpu().numpy()
    half = grid.cell_sizes.detach().cpu().numpy() / 2.0
    for point in points:
        index = grid.world_to_grid(point).detach().cpu().numpy()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                i, j = int(index[0] + di), int(index[1] + dj)
                if not (0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]):
                    continue
                if not occupied[i, j]:
                    continue
                center = grid.grid_to_world([i, j]).detach().cpu().numpy()
                closest = np.maximum(np.abs(point - center) - half, 0.0)
                if np.linalg.norm(closest) <= spacing:
                    return False
    return True


def supercover_grid_indices(start_index, goal_index) -> list[tuple[int, int]]:
    """Return every integer grid cell touched by a centre-to-centre segment.

    When the segment passes exactly through a grid corner, both side-adjacent
    cells are included in addition to the diagonal cell.  This closed-cell
    convention is deliberately conservative for occupancy-grid collision
    checking and uses integer arithmetic only.
    """

    start = np.asarray(start_index, dtype=np.int64).reshape(-1)
    goal = np.asarray(goal_index, dtype=np.int64).reshape(-1)
    if start.size < 2 or goal.size < 2:
        raise ValueError("grid indices must contain two coordinates")

    x, y = int(start[0]), int(start[1])
    goal_x, goal_y = int(goal[0]), int(goal[1])
    dx, dy = goal_x - x, goal_y - y
    nx, ny = abs(dx), abs(dy)
    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
    ix = iy = 0
    cells: list[tuple[int, int]] = [(x, y)]

    def append_unique(cell: tuple[int, int]) -> None:
        if cell != cells[-1]:
            cells.append(cell)

    while ix < nx or iy < ny:
        decision = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx
        if decision == 0:
            # The segment crosses a grid corner.  Closed-cell supercover
            # includes the two cells touching that corner from either side.
            if ix < nx:
                append_unique((x + step_x, y))
            if iy < ny:
                append_unique((x, y + step_y))
            if ix < nx:
                x += step_x
                ix += 1
            if iy < ny:
                y += step_y
                iy += 1
            append_unique((x, y))
        elif decision < 0:
            x += step_x
            ix += 1
            append_unique((x, y))
        else:
            y += step_y
            iy += 1
            append_unique((x, y))
    return cells


def line_of_sight_supercover_free(grid, start_xy, goal_xy) -> bool:
    """Test LOS by enumerating all inflated-grid cells touched by the segment."""

    start_index = grid.world_to_grid(start_xy).detach().cpu().numpy()
    goal_index = grid.world_to_grid(goal_xy).detach().cpu().numpy()
    occupied = grid.occupied.detach().cpu().numpy()
    for i, j in supercover_grid_indices(start_index, goal_index):
        if not (0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]):
            return False
        if occupied[i, j]:
            return False
    return True


def _simplify_ground_path(
    path: np.ndarray,
    grid,
    max_segment_length: float,
    visibility_test,
) -> np.ndarray:
    """Shared farthest-visible greedy simplifier."""

    path = np.asarray(path)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("path must have shape (N, 2), N >= 2")
    if max_segment_length <= 0:
        raise ValueError("max_segment_length must be positive")
    simplified = [path[0]]
    root = 0
    while root < len(path) - 1:
        chosen = None
        for candidate in range(len(path) - 1, root, -1):
            segment = path[[root, candidate]]
            if np.linalg.norm(segment[1] - segment[0]) > max_segment_length + 1e-9:
                continue
            if not visibility_test(grid, segment[0], segment[1]):
                continue
            chosen = candidate
            break
        if chosen is None:
            raise RuntimeError(f"no line-of-sight segment can leave path index {root}")
        simplified.append(path[chosen])
        root = chosen
    return np.asarray(simplified, dtype=path.dtype)


def simplify_ground_path(path: np.ndarray, grid, max_segment_length: float) -> np.ndarray:
    """Greedily simplify using the legacy sampled LOS test (strategy C)."""

    return _simplify_ground_path(path, grid, max_segment_length, line_of_sight_free)


def simplify_ground_path_supercover(
    path: np.ndarray,
    grid,
    max_segment_length: float,
) -> np.ndarray:
    """Greedily simplify using integer supercover LOS (strategy D)."""

    return _simplify_ground_path(
        path,
        grid,
        max_segment_length,
        line_of_sight_supercover_free,
    )


def merge_collinear_ground_path(path: np.ndarray, max_segment_length: float) -> np.ndarray:
    """Remove only collinear forward points while respecting a segment cap.

    This deliberately performs no visibility shortcut: every output segment is
    a subset of one straight run in the original four-connected grid path.
    """

    path = np.asarray(path)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("path must have shape (N, 2), N >= 2")
    if max_segment_length <= 0:
        raise ValueError("max_segment_length must be positive")
    if np.any(np.linalg.norm(np.diff(path, axis=0), axis=1) <= 1e-12):
        raise ValueError("path contains duplicate consecutive points")

    result = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        direction = path[anchor + 1] - path[anchor]
        chosen = anchor + 1
        while chosen + 1 < len(path):
            next_step = path[chosen + 1] - path[chosen]
            cross = direction[0] * next_step[1] - direction[1] * next_step[0]
            scale = max(np.linalg.norm(direction) * np.linalg.norm(next_step), 1.0)
            same_direction = float(np.dot(direction, next_step)) > 0.0
            within_cap = (
                np.linalg.norm(path[chosen + 1] - path[anchor])
                <= max_segment_length + 1e-9
            )
            if abs(float(cross)) > 1e-10 * scale or not same_direction or not within_cap:
                break
            chosen += 1
        result.append(path[chosen])
        anchor = chosen
    return np.asarray(result, dtype=path.dtype)
