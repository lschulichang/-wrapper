"""Kanayama trajectory tracking with bounded unicycle commands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .timed_trajectory import ReferenceState, wrap_angle


@dataclass(frozen=True)
class TrackingError:
    e_x: float
    e_y: float
    e_theta: float
    position: float

    def as_array(self) -> np.ndarray:
        return np.array([self.e_x, self.e_y, self.e_theta, self.position], dtype=np.float64)


class KanayamaController:
    def __init__(
        self,
        *,
        scale: float,
        k_x: float = 1.0,
        k_y: float = 2.0,
        k_theta: float = 2.0,
        max_speed_mps: float = 0.20,
        max_omega_radps: float = 1.0,
        max_accel_mps2: float = 0.30,
        max_alpha_radps2: float = 2.0,
        terminal_position_gain: float = 1.0,
        terminal_bearing_gain: float = 2.0,
        terminal_heading_gain: float = 2.0,
        position_tolerance_m: float = 0.02,
        heading_tolerance_rad: float = np.deg2rad(3.0),
    ):
        values = (
            scale, k_x, k_y, k_theta, max_speed_mps, max_omega_radps,
            max_accel_mps2, max_alpha_radps2, terminal_position_gain,
            terminal_bearing_gain, terminal_heading_gain,
            position_tolerance_m, heading_tolerance_rad,
        )
        if any(float(value) <= 0.0 for value in values):
            raise ValueError("controller scales, gains, limits, and tolerances must be positive")
        self.scale = float(scale)
        self.k_x = float(k_x)
        self.k_y = float(k_y)
        self.k_theta = float(k_theta)
        self.max_speed = float(max_speed_mps)
        self.max_omega = float(max_omega_radps)
        self.max_accel = float(max_accel_mps2)
        self.max_alpha = float(max_alpha_radps2)
        self.terminal_position_gain = float(terminal_position_gain)
        self.terminal_bearing_gain = float(terminal_bearing_gain)
        self.terminal_heading_gain = float(terminal_heading_gain)
        self.position_tolerance = float(position_tolerance_m)
        self.heading_tolerance = float(heading_tolerance_rad)

    def error(self, state, reference: ReferenceState) -> TrackingError:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (3,):
            raise ValueError("state must have shape (3,)")
        dx = (reference.x - state[0]) / self.scale
        dy = (reference.y - state[1]) / self.scale
        cosine, sine = np.cos(state[2]), np.sin(state[2])
        e_x = cosine * dx + sine * dy
        e_y = -sine * dx + cosine * dy
        e_theta = float(wrap_angle(reference.theta - state[2]))
        return TrackingError(float(e_x), float(e_y), e_theta, float(np.hypot(dx, dy)))

    def _limit(self, raw_control, previous_control, dt: float) -> np.ndarray:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        raw = np.asarray(raw_control, dtype=np.float64)
        previous = np.asarray(previous_control, dtype=np.float64)
        if raw.shape != (2,) or previous.shape != (2,):
            raise ValueError("controls must have shape (2,)")
        target_v = float(np.clip(raw[0], 0.0, self.max_speed))
        target_omega = float(np.clip(raw[1], -self.max_omega, self.max_omega))
        v = np.clip(target_v, previous[0] - self.max_accel * dt, previous[0] + self.max_accel * dt)
        omega = np.clip(
            target_omega,
            previous[1] - self.max_alpha * dt,
            previous[1] + self.max_alpha * dt,
        )
        return np.array([
            np.clip(v, 0.0, self.max_speed),
            np.clip(omega, -self.max_omega, self.max_omega),
        ], dtype=np.float64)

    def compute(self, state, reference: ReferenceState, previous_control, dt: float):
        error = self.error(state, reference)
        raw = np.array([
            reference.v * np.cos(error.e_theta) + self.k_x * error.e_x,
            reference.omega + reference.v * (
                self.k_y * error.e_y + self.k_theta * np.sin(error.e_theta)
            ),
        ])
        return self._limit(raw, previous_control, dt), error

    def compute_terminal(self, state, goal: ReferenceState, previous_control, dt: float):
        """Settle position first, then use the unicycle's in-place rotation."""

        error = self.error(state, goal)
        heading_error = float(wrap_angle(goal.theta - float(state[2])))
        reached = error.position <= self.position_tolerance and abs(heading_error) <= self.heading_tolerance
        if reached:
            raw = np.zeros(2)
        elif error.position > self.position_tolerance:
            dx = (goal.x - float(state[0])) / self.scale
            dy = (goal.y - float(state[1])) / self.scale
            bearing_error = float(wrap_angle(np.arctan2(dy, dx) - float(state[2])))
            raw = np.array([
                min(self.max_speed, self.terminal_position_gain * error.position)
                * max(0.0, np.cos(bearing_error)),
                self.terminal_bearing_gain * bearing_error,
            ])
        else:
            raw = np.array([0.0, self.terminal_heading_gain * heading_error])
        return self._limit(raw, previous_control, dt), error, reached
