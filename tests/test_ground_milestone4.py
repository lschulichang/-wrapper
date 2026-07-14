from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "splatnav-official"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(SCRIPTS))

from ground_nav.kanayama_controller import KanayamaController  # noqa: E402
from ground_nav.timed_trajectory import (  # noqa: E402
    ReferenceState,
    TimedTrajectory,
    evaluate_bezier_geometry,
    wrap_angle,
)
from ground_nav.unicycle_model import UnicycleModel  # noqa: E402
from smoke_ground_unicycle import corridor_statistics  # noqa: E402


def straight_controls(scale=1.0):
    return scale * np.array([[[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]]], dtype=np.float64)


class TimedTrajectoryTest(unittest.TestCase):
    def test_analytic_derivatives_match_finite_differences(self):
        control = np.array([[0.0, 0.4, 1.0], [0.0, 0.8, 0.0]])
        u = np.array([0.37])
        position, first, second = evaluate_bezier_geometry(control, u)
        h = 1e-5
        plus = evaluate_bezier_geometry(control, u + h)[0]
        minus = evaluate_bezier_geometry(control, u - h)[0]
        first_fd = (plus - minus) / (2.0 * h)
        second_fd = (plus - 2.0 * position + minus) / h**2
        np.testing.assert_allclose(first, first_fd, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(second, second_fd, rtol=2e-5, atol=2e-5)

    def test_straight_line_time_parameterization(self):
        trajectory = TimedTrajectory.from_bezier(
            straight_controls(), scale=1.0, ds_meters=0.02,
            max_speed_mps=0.2, max_accel_mps2=0.3, max_omega_radps=1.0,
        )
        self.assertEqual(trajectory.data.shape[1], 8)
        self.assertTrue(np.all(np.diff(trajectory.data[:, 0]) > 0.0))
        self.assertAlmostEqual(trajectory.data[0, 6], 0.0)
        self.assertAlmostEqual(trajectory.data[-1, 6], 0.0)
        np.testing.assert_allclose(trajectory.data[:, 5], 0.0, atol=1e-10)
        np.testing.assert_allclose(trajectory.data[:, 7], 0.0, atol=1e-10)
        self.assertLessEqual(np.max(trajectory.data[:, 6]), 0.2 + 1e-12)
        acceleration = np.abs(np.diff(trajectory.data[:, 6]) / np.diff(trajectory.data[:, 0]))
        self.assertLessEqual(np.max(acceleration), 0.3 + 1e-10)
        self.assertAlmostEqual(trajectory.metadata["total_length_m"], 1.0, places=7)

    def test_physical_result_is_invariant_to_scene_scale(self):
        first = TimedTrajectory.from_bezier(straight_controls(1.0), scale=1.0)
        second = TimedTrajectory.from_bezier(straight_controls(0.25), scale=0.25)
        np.testing.assert_allclose(first.data[:, 0], second.data[:, 0], atol=1e-10)
        np.testing.assert_allclose(first.data[:, 4:], second.data[:, 4:], atol=1e-10)
        np.testing.assert_allclose(first.data[:, 1:3], second.data[:, 1:3] / 0.25, atol=1e-10)

    def test_quarter_circle_curvature(self):
        k = 4.0 * (np.sqrt(2.0) - 1.0) / 3.0
        control = np.array([[[1.0, 1.0, k, 0.0], [0.0, k, 1.0, 1.0]]])
        trajectory = TimedTrajectory.from_bezier(control, scale=1.0, ds_meters=0.01)
        self.assertLess(np.max(np.abs(trajectory.data[:, 5] - 1.0)), 0.04)

    def test_degenerate_bezier_fails(self):
        control = np.array([[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]])
        with self.assertRaisesRegex(ValueError, "degenerate"):
            TimedTrajectory.from_bezier(control, scale=1.0)


class UnicycleAndControllerTest(unittest.TestCase):
    def test_unicycle_straight_step(self):
        model = UnicycleModel(scale=0.25)
        state = model.step([1.0, 2.0, np.pi / 2.0], [0.4, 0.0], 0.5)
        np.testing.assert_allclose(state[:2], [1.0, 2.05], atol=1e-12)
        self.assertAlmostEqual(state[2], np.pi / 2.0)

    def test_unicycle_exact_arc_step(self):
        model = UnicycleModel(scale=1.0)
        state = model.step([0.0, 0.0, 0.0], [1.0, 1.0], np.pi / 2.0)
        np.testing.assert_allclose(state[:2], [1.0, 1.0], atol=1e-12)
        self.assertAlmostEqual(state[2], np.pi / 2.0)

    def test_angle_wrap(self):
        self.assertAlmostEqual(float(wrap_angle(3.0 * np.pi)), -np.pi)
        self.assertAlmostEqual(float(wrap_angle(-3.0 * np.pi)), -np.pi)

    def test_kanayama_uses_metric_error_and_rate_limits(self):
        controller = KanayamaController(scale=0.25)
        reference = ReferenceState(0.0, 0.025, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0)
        command, error = controller.compute(
            np.array([0.0, 0.0, 0.0]), reference, np.zeros(2), 0.1
        )
        self.assertAlmostEqual(error.e_x, 0.1)
        self.assertAlmostEqual(command[0], 0.03)
        self.assertAlmostEqual(command[1], 0.0)

    def test_terminal_heading_settles(self):
        controller = KanayamaController(scale=1.0)
        model = UnicycleModel(scale=1.0)
        goal = ReferenceState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        state = np.array([0.0, 0.0, 0.5])
        previous = np.zeros(2)
        reached = False
        for _ in range(300):
            command, _, pose_reached = controller.compute_terminal(state, goal, previous, 0.02)
            if pose_reached and np.linalg.norm(command) <= 1e-6:
                reached = True
                break
            state = model.step(state, command, 0.02)
            previous = command
        self.assertTrue(reached)
        self.assertLessEqual(abs(float(wrap_angle(state[2]))), np.deg2rad(3.0))

    def test_kanayama_tracks_curved_reference(self):
        k = 4.0 * (np.sqrt(2.0) - 1.0) / 3.0
        control = np.array([[[1.0, 1.0, k, 0.0], [0.0, k, 1.0, 1.0]]])
        trajectory = TimedTrajectory.from_bezier(control, scale=1.0, ds_meters=0.01)
        controller = KanayamaController(scale=1.0)
        model = UnicycleModel(scale=1.0)
        initial = trajectory.query(0.0)
        state = np.array([initial.x, initial.y, initial.theta])
        previous = np.zeros(2)
        lateral_errors = []
        time_s = 0.0
        while time_s <= trajectory.duration + 1e-12:
            reference = trajectory.query(time_s)
            command, error = controller.compute(state, reference, previous, 0.02)
            lateral_errors.append(error.e_y)
            state = model.step(state, command, 0.02)
            previous = command
            time_s += 0.02
        lateral_errors = np.asarray(lateral_errors)
        self.assertLess(np.sqrt(np.mean(lateral_errors**2)), 0.03)
        self.assertLess(np.max(np.abs(lateral_errors)), 0.08)


class PipelineVerificationTest(unittest.TestCase):
    def test_corridor_statistics_use_halfspace_union(self):
        # Two adjacent squares represented exactly as milestone-2 A, b rows.
        first = (
            np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
            np.array([1.0, 0.0, 1.0, 0.0]),
        )
        second = (
            np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
            np.array([2.0, -1.0, 1.0, 0.0]),
        )
        flags, metrics = corridor_statistics(
            np.array([[0.5, 0.5], [1.5, 0.5], [2.1, 0.5]]),
            [first, second],
            scale=0.5,
        )
        np.testing.assert_array_equal(flags, [True, True, False])
        self.assertEqual(metrics["outside_sample_count"], 1)
        self.assertAlmostEqual(metrics["max_violation_m"], 0.2)


if __name__ == "__main__":
    unittest.main()
