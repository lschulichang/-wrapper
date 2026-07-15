 """Arc-length and time parameterization for planar Bezier paths."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def wrap_angle(angle):
    """Wrap angles to ``[-pi, pi)`` for scalars or arrays."""

    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _bernstein_basis(degree: int, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.stack([
        math.comb(degree, index)
        * (1.0 - values) ** (degree - index)
        * values**index
        for index in range(degree + 1)
    ])


def evaluate_bezier_geometry(control_points: np.ndarray, values: np.ndarray):
    """Evaluate position and first two parameter derivatives.

    ``control_points`` has shape ``(2, degree + 1)`` and ``values`` has
    shape ``(N,)``. Returned arrays have shape ``(N, 2)``.
    """

    control = np.asarray(control_points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if control.ndim != 2 or control.shape[0] != 2 or control.shape[1] < 3:
        raise ValueError("control_points must have shape (2, degree + 1), degree >= 2")
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    degree = control.shape[1] - 1
    position = (control @ _bernstein_basis(degree, values)).T
    first_control = degree * np.diff(control, axis=1)
    first = (first_control @ _bernstein_basis(degree - 1, values)).T
    second_control = degree * (degree - 1) * np.diff(control, n=2, axis=1)
    second = (second_control @ _bernstein_basis(degree - 2, values)).T
    return position, first, second


@dataclass(frozen=True)
class ReferenceState:
    t: float
    x: float
    y: float
    theta: float
    s_m: float
    curvature: float
    v: float
    omega: float

    def as_array(self) -> np.ndarray:
        return np.array([
            self.t, self.x, self.y, self.theta,
            self.s_m, self.curvature, self.v, self.omega,
        ], dtype=np.float64)


class TimedTrajectory:
    """A forward-only, curvature-aware timed reference trajectory."""

    columns = (
        "t_s", "x_scene", "y_scene", "theta_rad",
        "s_m", "curvature_1pm", "v_mps", "omega_radps",
    )

    def __init__(self, data: np.ndarray, heading_unwrapped: np.ndarray, metadata: dict):
        data = np.asarray(data, dtype=np.float64)
        heading_unwrapped = np.asarray(heading_unwrapped, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != 8 or len(data) < 2:
            raise ValueError("timed trajectory data must have shape (N, 8), N >= 2")
        if heading_unwrapped.shape != (len(data),):
            raise ValueError("heading_unwrapped must match trajectory length")
        if not np.all(np.diff(data[:, 0]) > 0.0):
            raise ValueError("trajectory timestamps must be strictly increasing")
        self.data = data
        self.heading_unwrapped = heading_unwrapped
        self.metadata = dict(metadata)

    @classmethod
    def from_bezier(
        cls,
        control_points,
        *,
        scale: float,
        ds_meters: float = 0.02,
        max_speed_mps: float = 0.20,
        max_accel_mps2: float = 0.30,
        max_omega_radps: float = 1.0,
        lookup_samples_per_section: int = 501,
        derivative_epsilon: float = 1e-8,
    ):
        controls_scene = np.asarray(control_points, dtype=np.float64)
        if controls_scene.ndim != 3 or controls_scene.shape[1] != 2 or controls_scene.shape[2] < 3:
            raise ValueError("control_points must have shape (sections, 2, degree + 1)")
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        if ds_meters <= 0.0 or max_speed_mps <= 0.0 or max_accel_mps2 <= 0.0:
            raise ValueError("distance, speed, and acceleration limits must be positive")
        if max_omega_radps <= 0.0:
            raise ValueError("max_omega_radps must be positive")
        if lookup_samples_per_section < 50:
            raise ValueError("lookup_samples_per_section must be at least 50")

        controls_m = controls_scene / float(scale)
        lookup_global, lookup_points = [], []
        for section, control in enumerate(controls_m):
            values = np.linspace(0.0, 1.0, lookup_samples_per_section)
            position, first, _ = evaluate_bezier_geometry(control, values)
            speed_parameter = np.linalg.norm(first, axis=1)
            if np.any(speed_parameter < derivative_epsilon):
                bad = int(np.argmin(speed_parameter))
                raise ValueError(
                    f"degenerate Bezier derivative in section {section} at u={values[bad]:.8f}: "
                    f"norm={speed_parameter[bad]:.3e}"
                )
            if section > 0:
                values = values[1:]
                position = position[1:]
            lookup_global.append(section + values)
            lookup_points.append(position)
        global_parameter = np.concatenate(lookup_global)
        points_m = np.concatenate(lookup_points)
        increments = np.linalg.norm(np.diff(points_m, axis=0), axis=1)
        if np.any(increments <= 1e-12):
            raise ValueError("Bezier arc-length lookup contains duplicate points")
        cumulative = np.concatenate([[0.0], np.cumsum(increments)])
        total_length = float(cumulative[-1])
        if total_length <= ds_meters:
            raise ValueError("Bezier path is too short for the requested arc-length spacing")

        uniform_s = np.arange(0.0, total_length, ds_meters, dtype=np.float64)
        if total_length - uniform_s[-1] > 1e-10:
            uniform_s = np.append(uniform_s, total_length)
        else:
            uniform_s[-1] = total_length
        uniform_global = np.interp(uniform_s, cumulative, global_parameter)
        sections = controls_m.shape[0]
        section_ids = np.minimum(np.floor(uniform_global).astype(np.int64), sections - 1)
        local_values = uniform_global - section_ids
        local_values[section_ids == sections - 1] = np.minimum(local_values[section_ids == sections - 1], 1.0)
        local_values[-1] = 1.0

        positions_m = np.empty((len(uniform_s), 2), dtype=np.float64)
        first = np.empty_like(positions_m)
        second = np.empty_like(positions_m)
        for section in range(sections):
            mask = section_ids == section
            if not np.any(mask):
                continue
            positions_m[mask], first[mask], second[mask] = evaluate_bezier_geometry(
                controls_m[section], local_values[mask]
            )
        derivative_norm = np.linalg.norm(first, axis=1)
        if np.any(derivative_norm < derivative_epsilon):
            bad = int(np.argmin(derivative_norm))
            raise ValueError(f"degenerate resampled derivative at index {bad}")
        heading_unwrapped = np.unwrap(np.arctan2(first[:, 1], first[:, 0]))
        curvature = (
            first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        ) / derivative_norm**3
        if not np.all(np.isfinite(curvature)):
            raise ValueError("non-finite curvature in timed trajectory")

        speed_cap = np.minimum(
            max_speed_mps,
            max_omega_radps / np.maximum(np.abs(curvature), 1e-12),
        )
        speeds = speed_cap.copy()
        speeds[0] = 0.0
        for index in range(1, len(speeds)):
            ds = uniform_s[index] - uniform_s[index - 1]
            reachable = np.sqrt(max(0.0, speeds[index - 1] ** 2 + 2.0 * max_accel_mps2 * ds))
            speeds[index] = min(speeds[index], reachable)
        speeds[-1] = 0.0
        for index in range(len(speeds) - 2, -1, -1):
            ds = uniform_s[index + 1] - uniform_s[index]
            reachable = np.sqrt(max(0.0, speeds[index + 1] ** 2 + 2.0 * max_accel_mps2 * ds))
            speeds[index] = min(speeds[index], reachable)
        distance_steps = np.diff(uniform_s)
        denominators = speeds[:-1] + speeds[1:]
        if np.any(denominators <= 1e-12):
            raise ValueError("speed profile contains an unreachable zero-speed interval")
        time_steps = 2.0 * distance_steps / denominators
        timestamps = np.concatenate([[0.0], np.cumsum(time_steps)])
        omega = speeds * curvature
        positions_scene = positions_m * float(scale)
        data = np.column_stack([
            timestamps,
            positions_scene,
            wrap_angle(heading_unwrapped),
            uniform_s,
            curvature,
            speeds,
            omega,
        ])
        metadata = {
            "scale": float(scale),
            "ds_meters": float(ds_meters),
            "max_speed_mps": float(max_speed_mps),
            "max_accel_mps2": float(max_accel_mps2),
            "max_omega_radps": float(max_omega_radps),
            "lookup_samples_per_section": int(lookup_samples_per_section),
            "total_length_m": total_length,
            "total_time_s": float(timestamps[-1]),
            "max_abs_curvature_1pm": float(np.max(np.abs(curvature))),
        }
        return cls(data, heading_unwrapped, metadata)

    @property
    def duration(self) -> float:
        return float(self.data[-1, 0])

    @property
    def goal(self) -> ReferenceState:
        return self.query(self.duration)

    def query(self, time_s: float) -> ReferenceState:
        time_s = float(np.clip(time_s, self.data[0, 0], self.data[-1, 0]))
        timestamps = self.data[:, 0]
        values = [time_s]
        for column in (1, 2):
            values.append(float(np.interp(time_s, timestamps, self.data[:, column])))
        theta_unwrapped = float(np.interp(time_s, timestamps, self.heading_unwrapped))
        values.append(float(wrap_angle(theta_unwrapped)))
        for column in (4, 5, 6, 7):
            values.append(float(np.interp(time_s, timestamps, self.data[:, column])))
        return ReferenceState(*values)

