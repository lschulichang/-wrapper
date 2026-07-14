"""Exact discrete kinematics for a planar unicycle."""

from __future__ import annotations

import numpy as np

from .timed_trajectory import wrap_angle


class UnicycleModel:
    """Integrate metric controls while storing positions in scene units."""

    def __init__(self, scale: float, angular_epsilon: float = 1e-9):
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        if angular_epsilon <= 0.0:
            raise ValueError("angular_epsilon must be positive")
        self.scale = float(scale)
        self.angular_epsilon = float(angular_epsilon)

    def step(self, state, control, dt: float) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        control = np.asarray(control, dtype=np.float64)
        if state.shape != (3,) or control.shape != (2,):
            raise ValueError("state and control must have shapes (3,) and (2,)")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        x, y, theta = state
        speed_scene = self.scale * float(control[0])
        omega = float(control[1])
        if abs(omega) <= self.angular_epsilon:
            x_next = x + speed_scene * np.cos(theta) * dt
            y_next = y + speed_scene * np.sin(theta) * dt
        else:
            theta_next = theta + omega * dt
            ratio = speed_scene / omega
            x_next = x + ratio * (np.sin(theta_next) - np.sin(theta))
            y_next = y - ratio * (np.cos(theta_next) - np.cos(theta))
        return np.array([x_next, y_next, float(wrap_angle(theta + omega * dt))], dtype=np.float64)

