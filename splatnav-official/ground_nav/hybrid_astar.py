"""Forward-only Hybrid A* for the projected 2D ground grid."""

from __future__ import annotations

import heapq
import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""

    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def angle_difference(first: float, second: float) -> float:
    """Return the absolute wrapped difference between two angles."""

    return abs(wrap_angle(float(first) - float(second)))


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

    @classmethod
    def from_value(cls, value: "Pose2D | Sequence[float]") -> "Pose2D":
        if isinstance(value, cls):
            return cls(float(value.x), float(value.y), wrap_angle(value.yaw))
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size != 3:
            raise ValueError("pose must contain exactly x, y, and yaw")
        if not np.all(np.isfinite(array)):
            raise ValueError("pose values must be finite")
        return cls(float(array[0]), float(array[1]), wrap_angle(float(array[2])))

    def as_array(self) -> np.ndarray:
        return np.asarray([self.x, self.y, wrap_angle(self.yaw)], dtype=np.float64)


@dataclass(frozen=True)
class HybridAStarConfig:
    """Physical and discretization parameters for forward Hybrid A*."""

    heading_bins: int = 72
    min_turning_radius_m: float = 0.20
    primitive_length_cells: float = 1.5
    collision_step_cells: float = 0.5
    curvature_fractions: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    steering_cost_weight: float = 0.05
    steering_change_cost_weight: float = 0.10
    analytic_expansion_interval: int = 50
    analytic_expansion_distance_cells: float = 12.0
    free_goal_heading_bins: int = 24
    max_expansions: int = 200_000

    def __post_init__(self) -> None:
        if self.heading_bins < 4:
            raise ValueError("heading_bins must be at least 4")
        if self.min_turning_radius_m <= 0.0:
            raise ValueError("min_turning_radius_m must be positive")
        if self.primitive_length_cells <= 0.0:
            raise ValueError("primitive_length_cells must be positive")
        if self.collision_step_cells <= 0.0:
            raise ValueError("collision_step_cells must be positive")
        if not self.curvature_fractions:
            raise ValueError("at least one curvature fraction is required")
        if any(not np.isfinite(value) or abs(value) > 1.0 for value in self.curvature_fractions):
            raise ValueError("curvature fractions must be finite and lie in [-1, 1]")
        if not any(abs(value) <= 1e-12 for value in self.curvature_fractions):
            raise ValueError("curvature_fractions must include a straight primitive")
        if self.steering_cost_weight < 0.0 or self.steering_change_cost_weight < 0.0:
            raise ValueError("steering cost weights must be non-negative")
        if self.analytic_expansion_interval <= 0:
            raise ValueError("analytic_expansion_interval must be positive")
        if self.analytic_expansion_distance_cells <= 0.0:
            raise ValueError("analytic_expansion_distance_cells must be positive")
        if self.free_goal_heading_bins < 4:
            raise ValueError("free_goal_heading_bins must be at least 4")
        if self.max_expansions <= 0:
            raise ValueError("max_expansions must be positive")


@dataclass(frozen=True)
class HybridPath:
    """Dense path samples plus compact search diagnostics."""

    poses_scene: np.ndarray
    node_poses_scene: np.ndarray
    primitive_curvatures_1pm: np.ndarray
    total_cost_m: float
    expanded_nodes: int
    generated_nodes: int
    analytic_expansion_used: bool
    goal_heading_constrained: bool
    config: dict

    @property
    def xy(self) -> np.ndarray:
        return self.poses_scene[:, :2]

    def summary(self) -> dict:
        lengths = np.linalg.norm(np.diff(self.poses_scene[:, :2], axis=0), axis=1)
        scene_scale = float(self.config["scene_scale"])
        return {
            "backend": "hybrid_astar_forward_dubins",
            "dense_pose_count": int(len(self.poses_scene)),
            "search_node_pose_count": int(len(self.node_poses_scene)),
            "primitive_count": int(len(self.primitive_curvatures_1pm)),
            "path_length_scene": float(lengths.sum()),
            "path_length_m": float(lengths.sum() / scene_scale),
            "total_cost_m": float(self.total_cost_m),
            "expanded_nodes": int(self.expanded_nodes),
            "generated_nodes": int(self.generated_nodes),
            "analytic_expansion_used": bool(self.analytic_expansion_used),
            "goal_heading_constrained": bool(self.goal_heading_constrained),
            "terminal_yaw_rad": float(self.poses_scene[-1, 2]),
            "config": dict(self.config),
        }


@dataclass
class _Record:
    state: np.ndarray
    g_m: float
    parent: tuple[int, int, int] | None
    curvature_scene: float
    steering_index: int


class HybridAStarPlanner:
    """Plan a forward-only curvature-bounded path on ``GroundGrid.occupied``."""

    _DUBINS_TYPES = (
        ("L", "S", "L"),
        ("R", "S", "R"),
        ("L", "S", "R"),
        ("R", "S", "L"),
        ("R", "L", "R"),
        ("L", "R", "L"),
    )

    def __init__(
        self,
        grid,
        scene_scale: float,
        config: HybridAStarConfig | None = None,
    ) -> None:
        if scene_scale <= 0.0:
            raise ValueError("scene_scale must be positive")
        self.grid = grid
        self.scene_scale = float(scene_scale)
        self.config = HybridAStarConfig() if config is None else config
        self.occupied = np.asarray(grid.occupied.detach().cpu(), dtype=bool)
        self.shape = tuple(int(value) for value in self.occupied.shape)
        self.cell_sizes = np.asarray(
            grid.cell_sizes.detach().cpu(), dtype=np.float64
        ).reshape(2)
        self.lower_center = np.asarray(
            grid.lower_center.detach().cpu(), dtype=np.float64
        ).reshape(2)
        self.lower_edge = self.lower_center - 0.5 * self.cell_sizes
        self.upper_edge = (
            self.lower_center
            + (np.asarray(self.shape, dtype=np.float64) - 0.5) * self.cell_sizes
        )
        self.heading_step = 2.0 * math.pi / self.config.heading_bins
        self.primitive_length_scene = (
            self.config.primitive_length_cells * float(np.max(self.cell_sizes))
        )
        self.collision_step_scene = (
            self.config.collision_step_cells * float(np.min(self.cell_sizes))
        )
        self.turning_radius_scene = (
            self.config.min_turning_radius_m * self.scene_scale
        )
        self.curvatures_scene = np.asarray(
            self.config.curvature_fractions, dtype=np.float64
        ) / self.turning_radius_scene
        self.straight_index = int(
            np.argmin(np.abs(np.asarray(self.config.curvature_fractions)))
        )

    def _point_index(self, point: Sequence[float]) -> tuple[int, int] | None:
        point_array = np.asarray(point, dtype=np.float64).reshape(2)
        if np.any(point_array < self.lower_edge) or np.any(point_array >= self.upper_edge):
            return None
        index = np.rint(
            (point_array - self.lower_center) / self.cell_sizes
        ).astype(np.int64)
        if np.any(index < 0) or np.any(index >= np.asarray(self.shape)):
            return None
        return int(index[0]), int(index[1])

    def _heading_index(self, yaw: float) -> int:
        normalized = (wrap_angle(yaw) + math.pi) % (2.0 * math.pi)
        return int(math.floor(normalized / self.heading_step)) % self.config.heading_bins

    def _state_key(
        self, state: np.ndarray
    ) -> tuple[int, int, int] | None:
        index = self._point_index(state[:2])
        if index is None:
            return None
        return index[0], index[1], self._heading_index(state[2])

    def _state_is_free(self, state: np.ndarray) -> bool:
        index = self._point_index(state[:2])
        return index is not None and not bool(self.occupied[index])

    @staticmethod
    def _integrate(
        state: np.ndarray, curvature_scene: float, distance_scene: float
    ) -> np.ndarray:
        x, y, yaw = (float(value) for value in state)
        if abs(curvature_scene) <= 1e-12:
            return np.asarray(
                [
                    x + distance_scene * math.cos(yaw),
                    y + distance_scene * math.sin(yaw),
                    wrap_angle(yaw),
                ],
                dtype=np.float64,
            )
        yaw_next = yaw + curvature_scene * distance_scene
        x_next = x + (math.sin(yaw_next) - math.sin(yaw)) / curvature_scene
        y_next = y + (-math.cos(yaw_next) + math.cos(yaw)) / curvature_scene
        return np.asarray([x_next, y_next, wrap_angle(yaw_next)], dtype=np.float64)

    def _sample_motion(
        self,
        state: np.ndarray,
        curvature_scene: float,
        distance_scene: float,
        *,
        collision_check: bool = True,
    ) -> np.ndarray | None:
        if distance_scene < -1e-12:
            raise ValueError("forward motion distance must be non-negative")
        if distance_scene <= 1e-12:
            return np.asarray([state], dtype=np.float64)
        count = max(1, int(math.ceil(distance_scene / self.collision_step_scene)))
        distances = np.linspace(
            distance_scene / count, distance_scene, count, dtype=np.float64
        )
        samples = np.stack(
            [self._integrate(state, curvature_scene, value) for value in distances]
        )
        if collision_check and any(not self._state_is_free(sample) for sample in samples):
            return None
        return samples

    def _goal_cost_map(self, goal_index: tuple[int, int]) -> np.ndarray:
        costs = np.full(self.shape, np.inf, dtype=np.float64)
        costs[goal_index] = 0.0
        queue: list[tuple[float, int, int]] = [(0.0, goal_index[0], goal_index[1])]
        steps = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                steps.append(
                    (di, dj, math.hypot(di * self.cell_sizes[0], dj * self.cell_sizes[1]))
                )
        while queue:
            cost, i, j = heapq.heappop(queue)
            if cost > costs[i, j] + 1e-12:
                continue
            for di, dj, step_cost in steps:
                ni, nj = i + di, j + dj
                if not (0 <= ni < self.shape[0] and 0 <= nj < self.shape[1]):
                    continue
                if self.occupied[ni, nj]:
                    continue
                candidate = cost + step_cost
                if candidate + 1e-12 < costs[ni, nj]:
                    costs[ni, nj] = candidate
                    heapq.heappush(queue, (candidate, ni, nj))
        return costs

    @staticmethod
    def _mod2pi(angle: float) -> float:
        return float(angle) % (2.0 * math.pi)

    @classmethod
    def _dubins_parameters(
        cls, alpha: float, beta: float, distance: float
    ) -> list[tuple[tuple[str, str, str], tuple[float, float, float]]]:
        sa, sb = math.sin(alpha), math.sin(beta)
        ca, cb = math.cos(alpha), math.cos(beta)
        cab = math.cos(alpha - beta)
        d2 = distance * distance
        candidates = []

        p2 = 2.0 + d2 - 2.0 * cab + 2.0 * distance * (sa - sb)
        if p2 >= -1e-12:
            p = math.sqrt(max(0.0, p2))
            tmp = math.atan2(cb - ca, distance + sa - sb)
            candidates.append(
                (("L", "S", "L"), (cls._mod2pi(-alpha + tmp), p, cls._mod2pi(beta - tmp)))
            )

        p2 = 2.0 + d2 - 2.0 * cab + 2.0 * distance * (-sa + sb)
        if p2 >= -1e-12:
            p = math.sqrt(max(0.0, p2))
            tmp = math.atan2(ca - cb, distance - sa + sb)
            candidates.append(
                (("R", "S", "R"), (cls._mod2pi(alpha - tmp), p, cls._mod2pi(-beta + tmp)))
            )

        p2 = -2.0 + d2 + 2.0 * cab + 2.0 * distance * (sa + sb)
        if p2 >= -1e-12:
            p = math.sqrt(max(0.0, p2))
            tmp = (
                math.atan2(-ca - cb, distance + sa + sb)
                - math.atan2(-2.0, p)
            )
            candidates.append(
                (("L", "S", "R"), (cls._mod2pi(-alpha + tmp), p, cls._mod2pi(-beta + tmp)))
            )

        p2 = d2 - 2.0 + 2.0 * cab - 2.0 * distance * (sa + sb)
        if p2 >= -1e-12:
            p = math.sqrt(max(0.0, p2))
            tmp = (
                math.atan2(ca + cb, distance - sa - sb)
                - math.atan2(2.0, p)
            )
            candidates.append(
                (("R", "S", "L"), (cls._mod2pi(alpha - tmp), p, cls._mod2pi(beta - tmp)))
            )

        value = (
            6.0 - d2 + 2.0 * cab + 2.0 * distance * (sa - sb)
        ) / 8.0
        if abs(value) <= 1.0 + 1e-12:
            p = cls._mod2pi(2.0 * math.pi - math.acos(float(np.clip(value, -1.0, 1.0))))
            t = cls._mod2pi(
                alpha
                - math.atan2(ca - cb, distance - sa + sb)
                + 0.5 * p
            )
            q = cls._mod2pi(alpha - beta - t + p)
            candidates.append((("R", "L", "R"), (t, p, q)))

        value = (
            6.0 - d2 + 2.0 * cab + 2.0 * distance * (-sa + sb)
        ) / 8.0
        if abs(value) <= 1.0 + 1e-12:
            p = cls._mod2pi(2.0 * math.pi - math.acos(float(np.clip(value, -1.0, 1.0))))
            t = cls._mod2pi(
                -alpha
                - math.atan2(ca - cb, distance + sa - sb)
                + 0.5 * p
            )
            q = cls._mod2pi(beta - alpha - t + p)
            candidates.append((("L", "R", "L"), (t, p, q)))
        return candidates

    def _integrate_dubins_segments(
        self,
        start: np.ndarray,
        path_types: tuple[str, str, str],
        normalized_lengths: tuple[float, float, float],
        *,
        collision_check: bool,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        current = np.asarray(start, dtype=np.float64)
        dense = [current.copy()]
        curvatures_m = []
        for path_type, normalized_length in zip(path_types, normalized_lengths):
            distance_scene = float(normalized_length) * self.turning_radius_scene
            if distance_scene <= 1e-12:
                continue
            if path_type == "L":
                curvature_scene = 1.0 / self.turning_radius_scene
            elif path_type == "R":
                curvature_scene = -1.0 / self.turning_radius_scene
            else:
                curvature_scene = 0.0
            samples = self._sample_motion(
                current,
                curvature_scene,
                distance_scene,
                collision_check=collision_check,
            )
            if samples is None:
                return None
            dense.extend(samples)
            current = samples[-1]
            curvatures_m.append(curvature_scene * self.scene_scale)
        return np.asarray(dense, dtype=np.float64), np.asarray(curvatures_m, dtype=np.float64)

    def _dubins_shortest(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        *,
        validate_endpoint: bool = False,
    ) -> tuple[tuple[str, str, str], tuple[float, float, float], float] | None:
        dx, dy = goal[:2] - start[:2]
        distance_scene = math.hypot(float(dx), float(dy))
        direction = math.atan2(float(dy), float(dx)) if distance_scene > 1e-12 else 0.0
        normalized_distance = distance_scene / self.turning_radius_scene
        alpha = self._mod2pi(float(start[2]) - direction)
        beta = self._mod2pi(float(goal[2]) - direction)
        candidates = self._dubins_parameters(alpha, beta, normalized_distance)
        if not validate_endpoint:
            if not candidates:
                return None
            path_types, lengths = min(candidates, key=lambda item: sum(item[1]))
            return (
                path_types,
                lengths,
                sum(lengths) * self.turning_radius_scene,
            )

        valid = []
        for path_types, lengths in candidates:
            integrated = self._integrate_dubins_segments(
                start, path_types, lengths, collision_check=False
            )
            if integrated is None:
                continue
            end = integrated[0][-1]
            position_error = float(np.linalg.norm(end[:2] - goal[:2]))
            yaw_error = angle_difference(end[2], goal[2])
            tolerance = max(1e-7, 1e-7 * self.turning_radius_scene)
            if position_error <= tolerance and yaw_error <= 1e-7:
                valid.append((sum(lengths), path_types, lengths))
        if not valid:
            return None
        normalized_total, path_types, lengths = min(valid, key=lambda item: item[0])
        return path_types, lengths, normalized_total * self.turning_radius_scene

    def _analytic_expansion(
        self, state: np.ndarray, goal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        solution = self._dubins_shortest(
            state,
            goal,
            validate_endpoint=True,
        )
        if solution is None:
            return None
        path_types, lengths, total_scene = solution
        integrated = self._integrate_dubins_segments(
            state, path_types, lengths, collision_check=True
        )
        if integrated is None:
            return None
        poses, curvatures_m = integrated
        poses[-1] = goal
        return poses, curvatures_m, total_scene / self.scene_scale

    def _analytic_expansion_free_heading(
        self,
        state: np.ndarray,
        goal_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Connect to an exact goal position while choosing the terminal yaw."""

        delta = np.asarray(goal_xy, dtype=np.float64) - state[:2]
        bearing = (
            math.atan2(float(delta[1]), float(delta[0]))
            if np.linalg.norm(delta) > 1e-12
            else float(state[2])
        )
        regular = np.linspace(
            -math.pi,
            math.pi,
            self.config.free_goal_heading_bins,
            endpoint=False,
            dtype=np.float64,
        )
        headings = np.concatenate(
            [regular, [wrap_angle(state[2]), wrap_angle(bearing)]]
        )
        unique_headings = []
        seen = set()
        for heading in headings:
            key = int(round(wrap_angle(float(heading)) * 1e12))
            if key in seen:
                continue
            seen.add(key)
            unique_headings.append(wrap_angle(float(heading)))

        candidates = []
        for heading in unique_headings:
            goal = np.asarray(
                [goal_xy[0], goal_xy[1], heading],
                dtype=np.float64,
            )
            solution = self._dubins_shortest(
                state,
                goal,
                validate_endpoint=True,
            )
            if solution is not None:
                candidates.append((solution[2], goal))

        for _, goal in sorted(candidates, key=lambda item: item[0]):
            analytic = self._analytic_expansion(state, goal)
            if analytic is not None:
                return analytic
        return None

    def _heuristic_m(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        goal_cost_scene: np.ndarray,
    ) -> float:
        index = self._point_index(state[:2])
        if index is None:
            return math.inf
        holonomic = float(goal_cost_scene[index]) / self.scene_scale
        if not np.isfinite(holonomic):
            return math.inf
        if len(goal) == 2:
            return holonomic
        dubins = self._dubins_shortest(state, goal)
        nonholonomic = (
            float(np.linalg.norm(state[:2] - goal[:2])) / self.scene_scale
            if dubins is None
            else dubins[2] / self.scene_scale
        )
        return max(holonomic, nonholonomic)

    def _reconstruct(
        self,
        records: dict[tuple[int, int, int], _Record],
        terminal_key: tuple[int, int, int],
        analytic: tuple[np.ndarray, np.ndarray, float],
        expanded_nodes: int,
        generated_nodes: int,
        goal_heading_constrained: bool,
    ) -> HybridPath:
        keys = []
        key = terminal_key
        while key is not None:
            keys.append(key)
            key = records[key].parent
        keys.reverse()
        node_poses = np.stack([records[item].state for item in keys])
        dense = [node_poses[0]]
        primitive_curvatures_m = []
        for parent_key, child_key in zip(keys[:-1], keys[1:]):
            parent = records[parent_key]
            child = records[child_key]
            samples = self._sample_motion(
                parent.state,
                child.curvature_scene,
                self.primitive_length_scene,
                collision_check=False,
            )
            if samples is None:
                raise RuntimeError("failed to reconstruct a validated Hybrid A* edge")
            dense.extend(samples)
            primitive_curvatures_m.append(child.curvature_scene * self.scene_scale)

        analytic_poses, analytic_curvatures_m, analytic_cost_m = analytic
        if len(analytic_poses) > 1:
            dense.extend(analytic_poses[1:])
        primitive_curvatures_m.extend(analytic_curvatures_m.tolist())
        dense_array = np.asarray(dense, dtype=np.float64)
        duplicate = np.linalg.norm(np.diff(dense_array[:, :2], axis=0), axis=1) <= 1e-12
        if np.any(duplicate):
            keep = np.concatenate([[True], ~duplicate])
            dense_array = dense_array[keep]
        return HybridPath(
            poses_scene=dense_array,
            node_poses_scene=np.vstack([node_poses, analytic_poses[-1]]),
            primitive_curvatures_1pm=np.asarray(
                primitive_curvatures_m, dtype=np.float64
            ),
            total_cost_m=float(records[terminal_key].g_m + analytic_cost_m),
            expanded_nodes=expanded_nodes,
            generated_nodes=generated_nodes,
            analytic_expansion_used=True,
            goal_heading_constrained=goal_heading_constrained,
            config={**asdict(self.config), "scene_scale": self.scene_scale},
        )

    def plan(
        self,
        start_pose: Pose2D | Sequence[float],
        goal_pose: Pose2D | Sequence[float],
    ) -> HybridPath:
        """Return a path to a goal pose or to a goal position with free yaw."""

        start = Pose2D.from_value(start_pose).as_array()
        if isinstance(goal_pose, Pose2D):
            goal = goal_pose.as_array()
            goal_heading_constrained = True
        else:
            goal_values = np.asarray(
                goal_pose,
                dtype=np.float64,
            ).reshape(-1)
            if goal_values.size not in (2, 3):
                raise ValueError(
                    "goal must contain x and y, with an optional yaw"
                )
            if not np.all(np.isfinite(goal_values)):
                raise ValueError("goal values must be finite")
            if goal_values.size == 2:
                goal = goal_values
                goal_heading_constrained = False
            else:
                goal = Pose2D.from_value(goal_values).as_array()
                goal_heading_constrained = True
        start_index = self._point_index(start[:2])
        goal_index = self._point_index(goal[:2])
        if start_index is None or goal_index is None:
            raise ValueError("start and goal must lie inside the ground-grid bounds")
        if self.occupied[start_index]:
            raise ValueError("start pose lies in an occupied ground-grid cell")
        if self.occupied[goal_index]:
            raise ValueError("goal pose lies in an occupied ground-grid cell")

        goal_cost_scene = self._goal_cost_map(goal_index)
        if not np.isfinite(goal_cost_scene[start_index]):
            raise RuntimeError("start and goal are disconnected in the projected ground grid")

        start_key = self._state_key(start)
        if start_key is None:
            raise RuntimeError("failed to index a valid start pose")
        records = {
            start_key: _Record(
                state=start,
                g_m=0.0,
                parent=None,
                curvature_scene=0.0,
                steering_index=self.straight_index,
            )
        }
        start_h = self._heuristic_m(start, goal, goal_cost_scene)
        queue = [(start_h, 0.0, 0, start_key)]
        serial = 0
        expanded_nodes = 0
        generated_nodes = 1
        analytic_distance_scene = (
            self.config.analytic_expansion_distance_cells
            * float(np.max(self.cell_sizes))
        )

        while queue and expanded_nodes < self.config.max_expansions:
            _, queued_g, _, key = heapq.heappop(queue)
            record = records.get(key)
            if record is None or queued_g > record.g_m + 1e-12:
                continue
            expanded_nodes += 1
            distance_to_goal = float(np.linalg.norm(record.state[:2] - goal[:2]))
            should_try_analytic = (
                expanded_nodes == 1
                or expanded_nodes % self.config.analytic_expansion_interval == 0
                or distance_to_goal <= analytic_distance_scene
            )
            if should_try_analytic:
                analytic = (
                    self._analytic_expansion(record.state, goal)
                    if goal_heading_constrained
                    else self._analytic_expansion_free_heading(
                        record.state,
                        goal[:2],
                    )
                )
                if analytic is not None:
                    return self._reconstruct(
                        records,
                        key,
                        analytic,
                        expanded_nodes,
                        generated_nodes,
                        goal_heading_constrained,
                    )

            previous_fraction = self.config.curvature_fractions[record.steering_index]
            for steering_index, curvature_scene in enumerate(self.curvatures_scene):
                samples = self._sample_motion(
                    record.state,
                    float(curvature_scene),
                    self.primitive_length_scene,
                    collision_check=True,
                )
                if samples is None:
                    continue
                child_state = samples[-1]
                child_key = self._state_key(child_state)
                if child_key is None:
                    continue
                fraction = self.config.curvature_fractions[steering_index]
                length_m = self.primitive_length_scene / self.scene_scale
                edge_cost = length_m * (
                    1.0
                    + self.config.steering_cost_weight * abs(fraction)
                    + self.config.steering_change_cost_weight
                    * abs(fraction - previous_fraction)
                )
                child_g = record.g_m + edge_cost
                previous = records.get(child_key)
                if previous is not None and child_g + 1e-12 >= previous.g_m:
                    continue
                heuristic = self._heuristic_m(child_state, goal, goal_cost_scene)
                if not np.isfinite(heuristic):
                    continue
                records[child_key] = _Record(
                    state=child_state,
                    g_m=child_g,
                    parent=key,
                    curvature_scene=float(curvature_scene),
                    steering_index=steering_index,
                )
                serial += 1
                generated_nodes += 1
                heapq.heappush(
                    queue,
                    (child_g + heuristic, child_g, serial, child_key),
                )

        if expanded_nodes >= self.config.max_expansions:
            raise RuntimeError(
                f"Hybrid A* exceeded max_expansions={self.config.max_expansions}"
            )
        raise RuntimeError(
            "Hybrid A* found no collision-free path to the goal position"
        )
