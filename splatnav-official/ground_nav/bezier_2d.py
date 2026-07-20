"""Convex 2D Bezier corridor optimization using the paper's QP structure."""

from __future__ import annotations

import math

import clarabel
import numpy as np
import scipy.special
from scipy import sparse


def _derivative_endpoint(degree: int, order: int, at_end: bool):
    factor = math.factorial(degree) / math.factorial(degree - order)
    offset = degree - order if at_end else 0
    indices = np.arange(order + 1) + offset
    coeffs = factor * np.array([(-1) ** (order - j) * scipy.special.comb(order, j) for j in range(order + 1)])
    return indices, coeffs


class BezierPlanner2D:
    def __init__(self, degree: int = 6, continuity_order: int = 3):
        if degree < 1 or not 0 <= continuity_order <= degree:
            raise ValueError("invalid degree or continuity order")
        self.degree = degree
        self.continuity_order = continuity_order
        self.control_points = None
        self.last_solver_status = None

    def _index(self, section: int, dimension: int, control: int) -> int:
        count = self.degree + 1
        return section * 2 * count + dimension * count + control

    def optimize(
        self,
        polygons,
        start,
        goal,
        *,
        start_yaw: float | None = None,
        goal_yaw: float | None = None,
        endpoint_tangent_min: float = 0.0,
    ):
        sections = len(polygons)
        if sections == 0:
            raise ValueError("at least one polygon is required")
        if start_yaw is not None and not np.isfinite(start_yaw):
            raise ValueError("start_yaw must be finite")
        if goal_yaw is not None and not np.isfinite(goal_yaw):
            raise ValueError("goal_yaw must be finite")
        if not np.isfinite(endpoint_tangent_min) or endpoint_tangent_min < 0.0:
            raise ValueError("endpoint_tangent_min must be finite and non-negative")
        count = self.degree + 1
        variables = sections * 2 * count
        difference = np.eye(count - 1, count, k=1) - np.eye(count - 1, count)
        block = difference.T @ difference
        P = sparse.block_diag([block] * (sections * 2), format="csc")

        G_rows, h = [], []
        for section, (A, b) in enumerate(polygons):
            A = np.asarray(A.detach().cpu() if hasattr(A, "detach") else A, dtype=np.float64)
            b = np.asarray(b.detach().cpu() if hasattr(b, "detach") else b, dtype=np.float64)
            for control in range(count):
                for row, bound in zip(A, b):
                    values = np.zeros(variables)
                    values[self._index(section, 0, control)] = row[0]
                    values[self._index(section, 1, control)] = row[1]
                    G_rows.append(values)
                    h.append(bound)

        C_rows, d = [], []
        for dimension in range(2):
            row = np.zeros(variables); row[self._index(0, dimension, 0)] = 1.0
            C_rows.append(row); d.append(float(start[dimension]))
            row = np.zeros(variables); row[self._index(sections - 1, dimension, self.degree)] = 1.0
            C_rows.append(row); d.append(float(goal[dimension]))
        for section, previous_control, next_control, yaw in (
            (0, 0, 1, start_yaw),
            (sections - 1, self.degree - 1, self.degree, goal_yaw),
        ):
            if yaw is None:
                continue
            direction = np.array([np.cos(yaw), np.sin(yaw)])
            perpendicular = np.array([-direction[1], direction[0]])
            tangent_row = np.zeros(variables)
            forward_row = np.zeros(variables)
            for dimension in range(2):
                tangent_row[self._index(section, dimension, next_control)] = (
                    perpendicular[dimension]
                )
                tangent_row[self._index(section, dimension, previous_control)] = (
                    -perpendicular[dimension]
                )
                # Clarabel's nonnegative cone encodes Gx <= h. This row
                # requires the endpoint derivative to point forward along yaw.
                forward_row[self._index(section, dimension, next_control)] = (
                    -direction[dimension]
                )
                forward_row[self._index(section, dimension, previous_control)] = (
                    direction[dimension]
                )
            C_rows.append(tangent_row); d.append(0.0)
            G_rows.append(forward_row); h.append(-float(endpoint_tangent_min))
        for section in range(sections - 1):
            for order in range(self.continuity_order + 1):
                left_indices, left_coeffs = _derivative_endpoint(self.degree, order, True)
                right_indices, right_coeffs = _derivative_endpoint(self.degree, order, False)
                for dimension in range(2):
                    row = np.zeros(variables)
                    for idx, value in zip(left_indices, left_coeffs):
                        row[self._index(section, dimension, int(idx))] += value
                    for idx, value in zip(right_indices, right_coeffs):
                        row[self._index(section + 1, dimension, int(idx))] -= value
                    C_rows.append(row); d.append(0.0)

        G = np.asarray(G_rows); h = np.asarray(h)
        C = np.asarray(C_rows); d = np.asarray(d)
        A_problem = sparse.csc_matrix(np.concatenate([C, G], axis=0))
        b_problem = np.concatenate([d, h])
        cones = [clarabel.ZeroConeT(len(d)), clarabel.NonnegativeConeT(len(h))]
        settings = clarabel.DefaultSettings(); settings.verbose = False
        solution = clarabel.DefaultSolver(P, np.zeros(variables), A_problem, b_problem, cones, settings).solve()
        self.last_solver_status = str(solution.status)
        if self.last_solver_status != "Solved":
            self.control_points = None
            return None, False
        flat = np.asarray(solution.x)
        self.control_points = flat.reshape(sections, 2, count)
        return self.control_points, True

    def sample(self, samples_per_section: int = 100):
        if self.control_points is None:
            raise RuntimeError("optimize must succeed before sampling")
        t = np.linspace(0.0, 1.0, samples_per_section)
        basis = np.stack([
            scipy.special.comb(self.degree, j) * (1.0 - t) ** (self.degree - j) * t**j
            for j in range(self.degree + 1)
        ])
        return np.stack([(section @ basis).T for section in self.control_points])
