"""Arc-length direct-collocation optimizer for forward ground trajectories."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import Bounds, minimize


def _wrap_angle(values):
    return (np.asarray(values, dtype=np.float64) + math.pi) % (2.0 * math.pi) - math.pi


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class TrajectoryOptimizerConfig:
    """Numerical and objective settings for arc-length collocation."""

    node_count: int = 32
    curvature_margin: float = 0.95
    length_weight: float = 0.10
    reference_position_weight: float = 10.0
    reference_heading_weight: float = 0.50
    curvature_weight: float = 0.50
    curvature_rate_weight: float = 5.0
    terminal_heading_weight: float = 0.20
    max_iterations: int = 500
    ftol: float = 1e-8
    constraint_tolerance: float = 2e-5
    dense_samples_per_segment: int = 8
    max_refinement_steps: int = 2
    refinement_factor: float = 1.5
    curvature_rate_relaxation: float = 1.5

    def __post_init__(self) -> None:
        if self.node_count < 4:
            raise ValueError("node_count must be at least four")
        if not 0.0 < self.curvature_margin <= 1.0:
            raise ValueError("curvature_margin must lie in (0, 1]")
        weights = (
            self.length_weight,
            self.reference_position_weight,
            self.reference_heading_weight,
            self.curvature_weight,
            self.curvature_rate_weight,
            self.terminal_heading_weight,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("objective weights must be finite and non-negative")
        if self.max_iterations <= 0 or self.ftol <= 0.0:
            raise ValueError("solver iteration and tolerance settings must be positive")
        if self.constraint_tolerance <= 0.0:
            raise ValueError("constraint_tolerance must be positive")
        if self.dense_samples_per_segment < 2:
            raise ValueError("dense_samples_per_segment must be at least two")
        if self.max_refinement_steps < 0 or self.refinement_factor <= 1.0:
            raise ValueError("invalid refinement settings")
        if self.curvature_rate_relaxation < 1.0:
            raise ValueError("curvature_rate_relaxation must be at least one")


@dataclass(frozen=True)
class TrajectoryResult:
    success: bool
    poses: np.ndarray
    curvature: np.ndarray
    curvature_rate: np.ndarray
    arc_length: np.ndarray
    objective: float | None
    solve_time: float
    failure_reason: str | None
    path_to_corridor: np.ndarray
    dense_poses: np.ndarray
    dense_arc_length: np.ndarray
    diagnostics: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "success": bool(self.success),
            "node_count": int(len(self.poses)),
            "dense_pose_count": int(len(self.dense_poses)),
            "path_length": (
                float(self.arc_length[-1]) if len(self.arc_length) else None
            ),
            "max_abs_curvature": (
                float(np.max(np.abs(self.curvature)))
                if len(self.curvature)
                else None
            ),
            "max_abs_curvature_rate": (
                float(np.max(np.abs(self.curvature_rate)))
                if len(self.curvature_rate)
                else None
            ),
            "objective": self.objective,
            "solve_time": float(self.solve_time),
            "failure_reason": self.failure_reason,
            "attempt_count": len(self.diagnostics),
        }


@dataclass(frozen=True)
class _Layout:
    nodes: int

    @property
    def segments(self) -> int:
        return self.nodes - 1

    @property
    def size(self) -> int:
        return 4 * self.nodes + 2 * self.segments

    def pack(self, x, y, heading, curvature, curvature_rate, lengths):
        return np.concatenate([x, y, heading, curvature, curvature_rate, lengths])

    def unpack(self, values):
        n, m = self.nodes, self.segments
        x = values[0:n]
        y = values[n : 2 * n]
        heading = values[2 * n : 3 * n]
        curvature = values[3 * n : 4 * n]
        curvature_rate = values[4 * n : 4 * n + m]
        lengths = values[4 * n + m : 4 * n + 2 * m]
        return x, y, heading, curvature, curvature_rate, lengths


def _dynamics(state: np.ndarray, curvature_rate: float) -> np.ndarray:
    return np.asarray(
        [
            math.cos(float(state[2])),
            math.sin(float(state[2])),
            float(state[3]),
            float(curvature_rate),
        ],
        dtype=np.float64,
    )


def _integrate_constant_rate(
    state: np.ndarray,
    curvature_rate: float,
    distance: float,
    steps: int,
) -> np.ndarray:
    current = np.asarray(state, dtype=np.float64).copy()
    step = float(distance) / int(steps)
    for _ in range(int(steps)):
        k1 = _dynamics(current, curvature_rate)
        k2 = _dynamics(current + 0.5 * step * k1, curvature_rate)
        k3 = _dynamics(current + 0.5 * step * k2, curvature_rate)
        k4 = _dynamics(current + step * k3, curvature_rate)
        current = current + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return current


def _normalize_corridors(corridors) -> list[tuple[np.ndarray, np.ndarray]]:
    normalized = []
    for index, (A, b) in enumerate(corridors):
        A = _as_numpy(A)
        b = _as_numpy(b).reshape(-1)
        if A.ndim != 2 or A.shape[1] != 2 or A.shape[0] != len(b):
            raise ValueError(f"corridor {index} has inconsistent A and b")
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
            raise ValueError(f"corridor {index} contains non-finite values")
        normalized.append((A, b))
    if not normalized:
        raise ValueError("at least one corridor is required")
    return normalized


def _reference_arc_length(reference: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(reference[:, :2], axis=0), axis=1)
    if np.any(lengths <= 1e-10):
        raise ValueError("reference path must not contain duplicate adjacent positions")
    return np.concatenate([[0.0], np.cumsum(lengths)])


def _mesh_from_reference(
    reference: np.ndarray,
    reference_arc: np.ndarray,
    reference_mapping: np.ndarray,
    node_count: int,
):
    uniform = np.linspace(0.0, reference_arc[-1], node_count)
    transitions = np.flatnonzero(np.diff(reference_mapping) != 0) + 1
    candidates = np.sort(np.concatenate([uniform, reference_arc[transitions]]))
    merge_tolerance = max(1e-9, 1e-9 * float(reference_arc[-1]))
    keep = np.concatenate([[True], np.diff(candidates) > merge_tolerance])
    mesh_arc = candidates[keep]
    mesh_arc[0] = 0.0
    mesh_arc[-1] = reference_arc[-1]
    x = np.interp(mesh_arc, reference_arc, reference[:, 0])
    y = np.interp(mesh_arc, reference_arc, reference[:, 1])
    unwrapped_heading = np.unwrap(reference[:, 2])
    heading = np.interp(mesh_arc, reference_arc, unwrapped_heading)
    node_indices = np.searchsorted(reference_arc, mesh_arc, side="right") - 1
    node_indices = np.clip(node_indices, 0, len(reference) - 1)
    node_mapping = reference_mapping[node_indices]
    midpoint_arc = 0.5 * (mesh_arc[:-1] + mesh_arc[1:])
    segment_indices = np.searchsorted(reference_arc, midpoint_arc, side="right") - 1
    segment_indices = np.clip(segment_indices, 0, len(reference) - 2)
    segment_mapping = reference_mapping[segment_indices]
    return (
        np.column_stack([x, y, heading]),
        mesh_arc,
        node_mapping.astype(np.int64),
        segment_mapping.astype(np.int64),
    )


def _initial_curvature(
    heading: np.ndarray,
    lengths: np.ndarray,
    curvature_limit: float,
    curvature_rate_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    segment_curvature = np.diff(heading) / lengths
    curvature = np.empty(len(heading), dtype=np.float64)
    curvature[0] = segment_curvature[0]
    curvature[-1] = segment_curvature[-1]
    if len(curvature) > 2:
        curvature[1:-1] = 0.5 * (
            segment_curvature[:-1] + segment_curvature[1:]
        )
    curvature = np.clip(curvature, -curvature_limit, curvature_limit)
    for _ in range(3):
        for index, distance in enumerate(lengths):
            delta = curvature_rate_limit * distance
            curvature[index + 1] = np.clip(
                curvature[index + 1],
                curvature[index] - delta,
                curvature[index] + delta,
            )
        for index in range(len(lengths) - 1, -1, -1):
            delta = curvature_rate_limit * lengths[index]
            curvature[index] = np.clip(
                curvature[index],
                curvature[index + 1] - delta,
                curvature[index + 1] + delta,
            )
    curvature_rate = np.diff(curvature) / lengths
    return curvature, curvature_rate


class CurvatureTrajectoryOptimizer:
    """Solve a corridor-constrained forward trajectory in the arc-length domain."""

    def __init__(self, config: TrajectoryOptimizerConfig | None = None) -> None:
        self.config = TrajectoryOptimizerConfig() if config is None else config

    def optimize(
        self,
        reference_path: np.ndarray,
        corridors,
        path_to_corridor: Sequence[int],
        start_pose: Sequence[float] | None,
        goal_xy: Sequence[float] | None,
        min_turning_radius: float,
        max_curvature_rate: float,
    ) -> TrajectoryResult:
        begin = time.perf_counter()
        reference = np.asarray(reference_path, dtype=np.float64)
        if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) < 2:
            raise ValueError("reference_path must have shape (N, 3), N >= 2")
        if not np.all(np.isfinite(reference)):
            raise ValueError("reference_path values must be finite")
        corridor_list = _normalize_corridors(corridors)
        reference_mapping = np.asarray(path_to_corridor, dtype=np.int64).reshape(-1)
        if len(reference_mapping) != len(reference):
            raise ValueError("path_to_corridor must have one entry per reference pose")
        if np.any(reference_mapping < 0) or np.any(reference_mapping >= len(corridor_list)):
            raise ValueError("path_to_corridor contains an invalid corridor index")
        if np.any(np.diff(reference_mapping) < 0):
            raise ValueError("path_to_corridor must be monotone")
        if min_turning_radius <= 0.0 or max_curvature_rate <= 0.0:
            raise ValueError("turning radius and curvature-rate limit must be positive")
        start = (
            reference[0]
            if start_pose is None
            else np.asarray(start_pose, dtype=np.float64).reshape(3)
        )
        goal = (
            reference[-1, :2]
            if goal_xy is None
            else np.asarray(goal_xy, dtype=np.float64).reshape(2)
        )
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
            raise ValueError("start_pose and goal_xy must be finite")
        reference_arc = _reference_arc_length(reference)
        curvature_limit = self.config.curvature_margin / float(min_turning_radius)
        diagnostics = []
        last_candidate = None

        for attempt in range(self.config.max_refinement_steps + 1):
            node_count = int(
                math.ceil(self.config.node_count * self.config.refinement_factor**attempt)
            )
            rate_limit = float(max_curvature_rate)
            if attempt == self.config.max_refinement_steps and attempt > 0:
                rate_limit *= self.config.curvature_rate_relaxation
            (
                reference_nodes,
                reference_node_arc,
                node_mapping,
                segment_mapping,
            ) = _mesh_from_reference(
                reference,
                reference_arc,
                reference_mapping,
                node_count,
            )
            layout = _Layout(len(reference_nodes))
            reference_lengths = np.diff(reference_node_arc)
            initial_curvature, initial_rate = _initial_curvature(
                reference_nodes[:, 2],
                reference_lengths,
                curvature_limit,
                rate_limit,
            )
            initial = layout.pack(
                reference_nodes[:, 0],
                reference_nodes[:, 1],
                reference_nodes[:, 2],
                initial_curvature,
                initial_rate,
                reference_lengths,
            )

            lower = np.full(layout.size, -np.inf, dtype=np.float64)
            upper = np.full(layout.size, np.inf, dtype=np.float64)
            n, m = layout.nodes, layout.segments
            lower[3 * n : 4 * n] = -curvature_limit
            upper[3 * n : 4 * n] = curvature_limit
            lower[4 * n : 4 * n + m] = -rate_limit
            upper[4 * n : 4 * n + m] = rate_limit
            lower[4 * n + m :] = np.maximum(1e-4, 0.25 * reference_lengths)
            upper[4 * n + m :] = 2.0 * reference_lengths

            length_weight = self.config.length_weight * (0.5**attempt)
            reference_weight = self.config.reference_position_weight * (1.5**attempt)

            def objective(values):
                x, y, heading, curvature, rate, lengths = layout.unpack(values)
                position_error = (x - reference_nodes[:, 0]) ** 2 + (
                    y - reference_nodes[:, 1]
                ) ** 2
                heading_error = heading - reference_nodes[:, 2]
                curvature_energy = 0.5 * np.sum(
                    (curvature[:-1] ** 2 + curvature[1:] ** 2) * lengths
                )
                return float(
                    length_weight * np.sum(lengths)
                    + reference_weight * np.sum(position_error)
                    + self.config.reference_heading_weight
                    * np.sum(heading_error**2)
                    + self.config.curvature_weight * curvature_energy
                    + self.config.curvature_rate_weight
                    * np.sum(rate**2 * lengths)
                    + self.config.terminal_heading_weight * heading_error[-1] ** 2
                )

            def equality(values):
                x, y, heading, curvature, rate, lengths = layout.unpack(values)
                rows = [
                    x[0] - start[0],
                    y[0] - start[1],
                    heading[0] - np.unwrap([reference[0, 2], start[2]])[-1],
                    x[-1] - goal[0],
                    y[-1] - goal[1],
                ]
                for index in range(m):
                    state = np.asarray(
                        [x[index], y[index], heading[index], curvature[index]]
                    )
                    predicted = _integrate_constant_rate(
                        state, rate[index], lengths[index], steps=4
                    )
                    following = np.asarray(
                        [
                            x[index + 1],
                            y[index + 1],
                            heading[index + 1],
                            curvature[index + 1],
                        ]
                    )
                    rows.extend(following - predicted)
                return np.asarray(rows, dtype=np.float64)

            def inequality(values):
                x, y, _, _, _, _ = layout.unpack(values)
                rows = []
                for index, corridor_index in enumerate(node_mapping):
                    A, b = corridor_list[int(corridor_index)]
                    rows.extend(b - A @ np.asarray([x[index], y[index]]))
                for index, corridor_index in enumerate(segment_mapping):
                    A, b = corridor_list[int(corridor_index)]
                    midpoint = np.asarray(
                        [0.5 * (x[index] + x[index + 1]), 0.5 * (y[index] + y[index + 1])]
                    )
                    rows.extend(b - A @ midpoint)
                return np.asarray(rows, dtype=np.float64)

            solution = minimize(
                objective,
                initial,
                method="SLSQP",
                bounds=Bounds(lower, upper),
                constraints=(
                    {"type": "eq", "fun": equality},
                    {"type": "ineq", "fun": inequality},
                ),
                options={
                    "maxiter": self.config.max_iterations,
                    "ftol": self.config.ftol,
                    "disp": False,
                },
            )
            candidate = np.asarray(solution.x, dtype=np.float64)
            last_candidate = (
                candidate,
                layout,
                node_mapping,
                segment_mapping,
                rate_limit,
            )
            equality_error = float(np.max(np.abs(equality(candidate))))
            minimum_margin = float(np.min(inequality(candidate)))
            validation = self._validate_candidate(
                candidate,
                layout,
                corridor_list,
                node_mapping,
                segment_mapping,
                curvature_limit,
                rate_limit,
            )
            attempt_row = {
                "attempt": attempt,
                "node_count": layout.nodes,
                "solver_success": bool(solution.success),
                "solver_status": int(solution.status),
                "solver_message": str(solution.message),
                "iterations": int(solution.nit),
                "objective": float(solution.fun),
                "max_equality_error": equality_error,
                "minimum_corridor_margin": minimum_margin,
                "curvature_rate_limit": rate_limit,
                **validation,
            }
            diagnostics.append(attempt_row)
            feasible = (
                solution.success
                and equality_error <= self.config.constraint_tolerance
                and minimum_margin >= -self.config.constraint_tolerance
                and validation["dense_corridor_feasible"]
                and validation["curvature_feasible"]
                and validation["curvature_rate_feasible"]
                and validation["dense_dynamics_feasible"]
            )
            if feasible:
                return self._result_from_candidate(
                    candidate,
                    layout,
                    node_mapping,
                    segment_mapping,
                    float(solution.fun),
                    time.perf_counter() - begin,
                    diagnostics,
                    corridor_list,
                )

        failure = diagnostics[-1]["solver_message"] if diagnostics else "no solve attempt"
        if last_candidate is None:
            return self._failure_result(time.perf_counter() - begin, failure, diagnostics)
        candidate, layout, node_mapping, segment_mapping, _ = last_candidate
        partial = self._result_from_candidate(
            candidate,
            layout,
            node_mapping,
            segment_mapping,
            None,
            time.perf_counter() - begin,
            diagnostics,
            corridor_list,
            success=False,
            failure_reason=failure,
        )
        return partial

    def _dense_rollout(self, values, layout: _Layout, segment_mapping):
        x, y, heading, curvature, rate, lengths = layout.unpack(values)
        dense_states = [np.asarray([x[0], y[0], heading[0], curvature[0]])]
        dense_arc = [0.0]
        dense_mapping = [int(segment_mapping[0])]
        endpoint_errors = []
        cumulative = 0.0
        for index in range(layout.segments):
            start = np.asarray([x[index], y[index], heading[index], curvature[index]])
            for sample in range(1, self.config.dense_samples_per_segment + 1):
                distance = lengths[index] * sample / self.config.dense_samples_per_segment
                state = _integrate_constant_rate(
                    start,
                    rate[index],
                    distance,
                    steps=max(1, sample),
                )
                dense_states.append(state)
                dense_arc.append(cumulative + distance)
                dense_mapping.append(int(segment_mapping[index]))
            expected = np.asarray(
                [x[index + 1], y[index + 1], heading[index + 1], curvature[index + 1]]
            )
            endpoint_errors.append(float(np.max(np.abs(dense_states[-1] - expected))))
            cumulative += lengths[index]
        return (
            np.asarray(dense_states),
            np.asarray(dense_arc),
            np.asarray(dense_mapping, dtype=np.int64),
            max(endpoint_errors, default=0.0),
        )

    def _validate_candidate(
        self,
        values,
        layout,
        corridors,
        node_mapping,
        segment_mapping,
        curvature_limit,
        rate_limit,
    ) -> dict:
        _, _, _, curvature, rate, _ = layout.unpack(values)
        dense, _, dense_mapping, dense_error = self._dense_rollout(
            values, layout, segment_mapping
        )
        margins = []
        for state, corridor_index in zip(dense, dense_mapping):
            A, b = corridors[int(corridor_index)]
            margins.extend(b - A @ state[:2])
        minimum_dense_margin = float(np.min(margins))
        tolerance = self.config.constraint_tolerance
        return {
            "dense_corridor_feasible": minimum_dense_margin >= -tolerance,
            "minimum_dense_corridor_margin": minimum_dense_margin,
            "curvature_feasible": bool(
                np.max(np.abs(curvature)) <= curvature_limit + tolerance
            ),
            "curvature_rate_feasible": bool(
                np.max(np.abs(rate)) <= rate_limit + tolerance
            ),
            "dense_dynamics_feasible": dense_error <= 5.0 * tolerance,
            "max_dense_dynamics_error": dense_error,
        }

    def _result_from_candidate(
        self,
        values,
        layout,
        node_mapping,
        segment_mapping,
        objective,
        solve_time,
        diagnostics,
        corridors,
        *,
        success=True,
        failure_reason=None,
    ) -> TrajectoryResult:
        x, y, heading, curvature, rate, lengths = layout.unpack(values)
        arc = np.concatenate([[0.0], np.cumsum(lengths)])
        dense, dense_arc, dense_mapping, _ = self._dense_rollout(
            values, layout, segment_mapping
        )
        poses = np.column_stack([x, y, _wrap_angle(heading)])
        dense_poses = np.column_stack([dense[:, :2], _wrap_angle(dense[:, 2])])
        return TrajectoryResult(
            success=bool(success),
            poses=poses,
            curvature=np.asarray(curvature).copy(),
            curvature_rate=np.asarray(rate).copy(),
            arc_length=arc,
            objective=None if objective is None else float(objective),
            solve_time=float(solve_time),
            failure_reason=failure_reason,
            path_to_corridor=np.asarray(node_mapping, dtype=np.int64).copy(),
            dense_poses=dense_poses,
            dense_arc_length=dense_arc,
            diagnostics=list(diagnostics),
        )

    @staticmethod
    def _failure_result(solve_time, failure_reason, diagnostics):
        return TrajectoryResult(
            success=False,
            poses=np.empty((0, 3)),
            curvature=np.empty(0),
            curvature_rate=np.empty(0),
            arc_length=np.empty(0),
            objective=None,
            solve_time=float(solve_time),
            failure_reason=str(failure_reason),
            path_to_corridor=np.empty(0, dtype=np.int64),
            dense_poses=np.empty((0, 3)),
            dense_arc_length=np.empty(0),
            diagnostics=list(diagnostics),
        )


def optimize(
    reference_path,
    corridors,
    path_to_corridor,
    start_pose,
    goal_xy,
    min_turning_radius,
    max_curvature_rate,
    *,
    config: TrajectoryOptimizerConfig | None = None,
) -> TrajectoryResult:
    """Functional wrapper around :class:`CurvatureTrajectoryOptimizer`."""

    return CurvatureTrajectoryOptimizer(config).optimize(
        reference_path,
        corridors,
        path_to_corridor,
        start_pose,
        goal_xy,
        min_turning_radius,
        max_curvature_rate,
    )
