#!/usr/bin/env python3
"""Time-parameterize a milestone-2 path and track it with a unicycle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "splatnav-official"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(UPSTREAM))

from ground_nav.corridor_2d import PlanarCollisionSet  # noqa: E402
from ground_nav.kanayama_controller import KanayamaController  # noqa: E402
from ground_nav.path_utils import line_of_sight_free  # noqa: E402
from ground_nav.planar_gaussians import PlanarGaussianSet  # noqa: E402
from ground_nav.timed_trajectory import TimedTrajectory  # noqa: E402
from ground_nav.unicycle_model import UnicycleModel  # noqa: E402
from smoke_ground_corridor_2d import (  # noqa: E402
    configure_map_axis,
    polygon_vertices,
    projected_ellipse_polylines,
)


class SavedGroundGrid:
    """The minimal GroundGrid interface required for continuous LOS checks."""

    def __init__(self, archive):
        self.occupied = torch.as_tensor(archive["occupied"], dtype=torch.bool)
        self.cell_sizes = torch.as_tensor(archive["cell_sizes"], dtype=torch.float64)
        self.lower_center = torch.as_tensor(archive["lower_center"], dtype=torch.float64)
        self.shape = tuple(int(value) for value in archive["shape"])

    def world_to_grid(self, point):
        point = torch.as_tensor(point, dtype=torch.float64).flatten()[:2]
        index = torch.round((point - self.lower_center) / self.cell_sizes).to(torch.long)
        maximum = torch.tensor(self.shape, dtype=torch.long) - 1
        return torch.minimum(torch.maximum(index, torch.zeros_like(index)), maximum)

    def grid_to_world(self, index):
        index = torch.as_tensor(index, dtype=torch.float64).flatten()[:2]
        return self.lower_center + index * self.cell_sizes


def load_polygons(path: Path):
    polygons = []
    for rows in json.loads(path.read_text()):
        array = np.asarray(rows, dtype=np.float64)
        polygons.append((array[:, :2], array[:, 2]))
    if not polygons:
        raise ValueError("no safety polygons were saved by milestone 2")
    return polygons


def load_projected_ellipses(path: Path, planning_result: dict, device):
    archive = np.load(path)
    ids = torch.as_tensor(archive["ids"], dtype=torch.long, device=device)
    means = torch.as_tensor(archive["means"], dtype=torch.float32, device=device)
    covs = torch.as_tensor(archive["covs"], dtype=torch.float32, device=device)
    eigenvalues, eigenvectors = torch.linalg.eigh(covs)
    eigenvalues = torch.clamp(eigenvalues, min=1e-12)
    scene = planning_result["scene_units"]
    return PlanarGaussianSet(
        ids=ids,
        means=means,
        covs=covs,
        rots=eigenvectors,
        scales=torch.sqrt(eigenvalues),
        z_min=float(scene["z_min"]),
        z_max=float(scene["z_max"]),
        confidence=1.0,
    )


def inside_corridor(point, polygons, tolerance=1e-6):
    return any(np.all(A @ point <= b + tolerance) for A, b in polygons)


def corridor_statistics(points, polygons, scale, tolerance=1e-6):
    """Measure membership in the union of milestone-2 half-space polygons."""
    points = np.asarray(points, dtype=np.float64)
    signed_violations = np.asarray([
        min(float(np.max(A @ point - b)) for A, b in polygons)
        for point in points
    ])
    inside = signed_violations <= tolerance
    positive = np.maximum(signed_violations, 0.0)
    return inside, {
        "sample_count": int(len(points)),
        "outside_sample_count": int(np.count_nonzero(~inside)),
        "inside_fraction": float(np.mean(inside)),
        "max_violation_scene": float(np.max(positive)),
        "max_violation_m": float(np.max(positive) / scale),
        "minimum_union_clearance_scene": float(max(0.0, -np.max(signed_violations))),
        "minimum_union_clearance_m": float(max(0.0, -np.max(signed_violations)) / scale),
    }


def segment_safety_flags(points, collision_set, grid):
    exact_flags, grid_flags = [], []
    with torch.no_grad():
        for first, second in zip(points[:-1], points[1:]):
            segment = np.stack([first, second])
            exact_flags.append(collision_set.segment_is_safe(segment))
            grid_flags.append(line_of_sight_free(grid, first, second))
    return np.asarray(exact_flags, dtype=bool), np.asarray(grid_flags, dtype=bool)


def flag_statistics(flags):
    flags = np.asarray(flags, dtype=bool)
    return {
        "segment_count": int(len(flags)),
        "violating_segment_count": int(np.count_nonzero(~flags)),
        "safe_fraction": float(np.mean(flags)) if len(flags) else 1.0,
    }


def json_default(item):
    if isinstance(item, np.generic):
        return item.item()
    raise TypeError(f"unsupported JSON value: {type(item).__name__}")


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default))


def plot_timing(trajectory, output_dir, max_speed, max_omega):
    data = trajectory.data
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(data[:, 0], data[:, 6], color="tab:blue")
    axes[0, 0].axhline(max_speed, color="tab:red", linestyle="--", label="limit")
    axes[0, 0].set(xlabel="time [s]", ylabel="v_d [m/s]", title="Reference speed")
    axes[0, 0].legend()
    axes[0, 1].plot(data[:, 0], data[:, 7], color="tab:orange")
    axes[0, 1].axhline(max_omega, color="tab:red", linestyle="--")
    axes[0, 1].axhline(-max_omega, color="tab:red", linestyle="--", label="limits")
    axes[0, 1].set(xlabel="time [s]", ylabel="omega_d [rad/s]", title="Reference angular speed")
    axes[0, 1].legend()
    axes[1, 0].plot(data[:, 4], data[:, 5], color="tab:green")
    axes[1, 0].set(xlabel="arc length [m]", ylabel="curvature [1/m]", title="Path curvature")
    axes[1, 1].plot(data[:, 0], data[:, 4], color="tab:purple")
    axes[1, 1].set(xlabel="time [s]", ylabel="arc length [m]", title="Arc-length progress")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "timing_and_curvature.png", dpi=180)
    plt.close(fig)


def plot_tracking_overview(
    trajectory,
    aligned_reference,
    executed,
    errors,
    ellipses,
    polygons,
    extent,
    radius_scene,
    result,
    bad_point_indices,
    output_dir,
):
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.add_collection(LineCollection(
        projected_ellipse_polylines(ellipses),
        colors="tab:red", linewidths=0.25, alpha=0.10, rasterized=True,
        label="projected ellipses", zorder=1,
    ))
    polygon_label = True
    for A, b in polygons:
        vertices = polygon_vertices(A, b)
        if vertices is not None:
            ax.fill(
                vertices[:, 0], vertices[:, 1], color="tab:green", alpha=0.16,
                label="safe polygons" if polygon_label else None, zorder=2,
            )
            polygon_label = False
    ax.plot(
        trajectory.data[:, 1], trajectory.data[:, 2], color="tab:orange", linewidth=3.0,
        alpha=0.80, label="timed Bezier reference", zorder=3,
    )
    ax.plot(
        executed[:, 1], executed[:, 2], color="tab:purple", linewidth=1.8,
        linestyle="--", label="unicycle execution", zorder=4,
    )
    connector_count = min(10, len(executed))
    for index in np.unique(np.linspace(0, len(executed) - 1, connector_count).astype(int)):
        ax.plot(
            [aligned_reference[index, 1], executed[index, 1]],
            [aligned_reference[index, 2], executed[index, 2]],
            color="0.35", linewidth=0.7, alpha=0.55, zorder=3,
        )
    pose_count = min(15, len(executed))
    for index in np.unique(np.linspace(0, len(executed) - 1, pose_count).astype(int)):
        x, y, theta = executed[index, 1:4]
        ax.add_patch(Circle((x, y), radius_scene, fill=False, color="0.1", linewidth=0.7, alpha=0.55, zorder=5))
        ax.arrow(
            x, y, radius_scene * 0.85 * np.cos(theta), radius_scene * 0.85 * np.sin(theta),
            width=radius_scene * 0.025, head_width=radius_scene * 0.20,
            color="0.1", alpha=0.60, length_includes_head=True, zorder=5,
        )
    ax.scatter(executed[0, 1], executed[0, 2], marker="*", s=110, color="tab:green", label="start", zorder=6)
    ax.scatter(executed[-1, 1], executed[-1, 2], marker="*", s=110, color="tab:red", label="finish", zorder=6)
    if bad_point_indices:
        indices = np.asarray(sorted(bad_point_indices), dtype=int)
        ax.scatter(
            executed[indices, 1], executed[indices, 2], marker="x", s=55,
            color="red", label="corridor/exact warning", zorder=7,
        )
    configure_map_axis(ax, extent, "Timed reference and unicycle execution")
    ax.legend(loc="upper right")
    metrics = (
        f"RMSE ey: {result['rmse_lateral_m'] * 100:.2f} cm\n"
        f"max |ey|: {result['max_lateral_error_m'] * 100:.2f} cm\n"
        f"final position: {result['final_position_error_m'] * 100:.2f} cm\n"
        f"final heading: {np.rad2deg(result['final_heading_error_rad']):.2f} deg\n"
        f"exact collision-free: {result['continuous_ellipse_safe']}\n"
        f"reference corridor: {result['reference_inside_corridor']}\n"
        f"execution corridor: {result['corridor']['execution']['inside_fraction'] * 100:.2f}%\n"
        f"reached-goal: {result['reached_goal']}"
    )
    ax.text(
        0.015, 0.015, metrics, transform=ax.transAxes, va="bottom", ha="left", fontsize=8.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82}, zorder=8,
    )
    worst = int(np.argmax(errors[:, 4]))
    center = 0.5 * (executed[worst, 1:3] + aligned_reference[worst, 1:3])
    half_width = max(0.03 * result["scale"], 3.0 * errors[worst, 4] * result["scale"])
    inset = ax.inset_axes([0.61, 0.08, 0.35, 0.30])
    inset.plot(trajectory.data[:, 1], trajectory.data[:, 2], color="tab:orange", linewidth=2.2)
    inset.plot(executed[:, 1], executed[:, 2], color="tab:purple", linestyle="--", linewidth=1.5)
    inset.plot(
        [aligned_reference[worst, 1], executed[worst, 1]],
        [aligned_reference[worst, 2], executed[worst, 2]], color="black", linewidth=1.0,
    )
    inset.scatter(aligned_reference[worst, 1], aligned_reference[worst, 2], color="tab:orange", s=20)
    inset.scatter(executed[worst, 1], executed[worst, 2], color="tab:purple", s=20)
    inset.set_xlim(center[0] - half_width, center[0] + half_width)
    inset.set_ylim(center[1] - half_width, center[1] + half_width)
    inset.set_aspect("equal")
    inset.set_title(f"max error {errors[worst, 4] * 100:.2f} cm", fontsize=8)
    inset.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "reference_and_executed_trajectory.png", dpi=180)
    plt.close(fig)


def plot_errors_and_controls(errors, controls, max_speed, max_omega, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes[0, 0].plot(errors[:, 0], errors[:, 1], label="ex")
    axes[0, 0].plot(errors[:, 0], errors[:, 2], label="ey")
    axes[0, 0].set(ylabel="position error [m]", title="Body-frame position errors")
    axes[0, 0].legend()
    axes[0, 1].plot(errors[:, 0], np.rad2deg(errors[:, 3]), color="tab:green")
    axes[0, 1].set(ylabel="heading error [deg]", title="Heading error")
    axes[1, 0].plot(controls[:, 0], controls[:, 1], label="reference")
    axes[1, 0].plot(controls[:, 0], controls[:, 3], linestyle="--", label="command")
    axes[1, 0].axhline(max_speed, color="tab:red", linestyle=":", label="limit")
    axes[1, 0].axhline(0.0, color="tab:red", linestyle=":")
    axes[1, 0].set(xlabel="time [s]", ylabel="v [m/s]", title="Linear speed")
    axes[1, 0].legend()
    axes[1, 1].plot(controls[:, 0], controls[:, 2], label="reference")
    axes[1, 1].plot(controls[:, 0], controls[:, 4], linestyle="--", label="command")
    axes[1, 1].axhline(max_omega, color="tab:red", linestyle=":", label="limits")
    axes[1, 1].axhline(-max_omega, color="tab:red", linestyle=":")
    axes[1, 1].set(xlabel="time [s]", ylabel="omega [rad/s]", title="Angular speed")
    axes[1, 1].legend()
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "tracking_errors_and_controls.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument("--ds-meters", type=float, default=0.02)
    parser.add_argument("--max-speed-mps", type=float, default=0.20)
    parser.add_argument("--max-accel-mps2", type=float, default=0.30)
    parser.add_argument("--max-omega-radps", type=float, default=1.0)
    parser.add_argument("--max-alpha-radps2", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--settle-timeout", type=float, default=5.0)
    parser.add_argument("--kx", type=float, default=1.0)
    parser.add_argument("--ky", type=float, default=2.0)
    parser.add_argument("--ktheta", type=float, default=2.0)
    args = parser.parse_args()
    assets = args.assets_dir.resolve()
    required = (
        "control_points.npy", "projected_gaussians.npz", "polygons.json",
        "ground_grid.npz", "result.json", "verification.json",
    )
    missing = [name for name in required if not (assets / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing milestone-2 assets: {missing}")

    planning_result = json.loads((assets / "result.json").read_text())
    planning_verification = json.loads((assets / "verification.json").read_text())
    save_json(assets / "planning_result.json", planning_result)
    save_json(assets / "planning_verification.json", planning_verification)
    scale = float(planning_result["scale"])
    try:
        trajectory = TimedTrajectory.from_bezier(
            np.load(assets / "control_points.npy"),
            scale=scale,
            ds_meters=args.ds_meters,
            max_speed_mps=args.max_speed_mps,
            max_accel_mps2=args.max_accel_mps2,
            max_omega_radps=args.max_omega_radps,
        )
    except Exception as error:
        failure = {"failure_reason": f"time parameterization failed: {error}"}
        save_json(assets / "tracking_result.json", failure)
        planning_result["tracking"] = failure
        planning_result["failure_reason"] = failure["failure_reason"]
        save_json(assets / "result.json", planning_result)
        raise

    controller = KanayamaController(
        scale=scale,
        k_x=args.kx,
        k_y=args.ky,
        k_theta=args.ktheta,
        max_speed_mps=args.max_speed_mps,
        max_omega_radps=args.max_omega_radps,
        max_accel_mps2=args.max_accel_mps2,
        max_alpha_radps2=args.max_alpha_radps2,
    )
    model = UnicycleModel(scale)
    initial = trajectory.query(0.0)
    state = np.array([initial.x, initial.y, initial.theta], dtype=np.float64)
    previous_control = np.zeros(2, dtype=np.float64)
    maximum_time = trajectory.duration + args.settle_timeout
    executed_rows, control_rows, error_rows, reference_rows = [], [], [], []
    reached_goal = False
    time_s = 0.0
    while time_s <= maximum_time + 1e-12:
        reference = trajectory.query(time_s)
        if time_s < trajectory.duration - 1e-12:
            command, error = controller.compute(state, reference, previous_control, args.dt)
            pose_reached = False
        else:
            command, error, pose_reached = controller.compute_terminal(
                state, trajectory.goal, previous_control, args.dt
            )
        executed_rows.append([time_s, *state])
        control_rows.append([time_s, reference.v, reference.omega, *command])
        error_rows.append([time_s, *error.as_array()])
        reference_rows.append(reference.as_array())
        if pose_reached and np.linalg.norm(command) <= 1e-6:
            reached_goal = True
            break
        state = model.step(state, command, args.dt)
        previous_control = command
        time_s += args.dt

    executed = np.asarray(executed_rows, dtype=np.float64)
    controls = np.asarray(control_rows, dtype=np.float64)
    errors = np.asarray(error_rows, dtype=np.float64)
    aligned_reference = np.asarray(reference_rows, dtype=np.float64)
    np.save(assets / "reference_trajectory.npy", trajectory.data)
    np.save(assets / "executed_trajectory.npy", executed)
    np.save(assets / "controls.npy", controls)
    np.save(assets / "tracking_errors.npy", errors)

    polygons = load_polygons(assets / "polygons.json")
    grid_archive = np.load(assets / "ground_grid.npz")
    grid = SavedGroundGrid(grid_archive)
    extent = np.asarray(grid_archive["extent"], dtype=np.float64).tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ellipses = load_projected_ellipses(assets / "projected_gaussians.npz", planning_result, device)
    collision_set = PlanarCollisionSet(
        ellipses,
        radius=float(planning_result["scene_units"]["circle_radius"]),
        corridor_margin=float(planning_result["scene_units"]["corridor_margin"]),
        iterations=10,
    )
    reference_exact_flags, reference_grid_flags = segment_safety_flags(
        trajectory.data[:, 1:3], collision_set, grid
    )
    execution_exact_flags, execution_grid_flags = segment_safety_flags(
        executed[:, 1:3], collision_set, grid
    )
    reference_corridor_flags, reference_corridor = corridor_statistics(
        trajectory.data[:, 1:3], polygons, scale
    )
    execution_corridor_flags, execution_corridor = corridor_statistics(
        executed[:, 1:3], polygons, scale
    )
    # Red crosses represent a violation of the continuous geometry model or the
    # reference corridor. The inflated grid is a conservative seed-search model
    # and is reported separately instead of being mislabeled as a collision.
    bad_points = set(np.flatnonzero(~execution_corridor_flags).tolist())
    bad_points.update((np.flatnonzero(~execution_exact_flags) + 1).tolist())

    command_dt = np.diff(controls[:, 0])
    command_accel = np.abs(np.diff(controls[:, 3])) / command_dt if len(controls) > 1 else np.array([0.0])
    command_alpha = np.abs(np.diff(controls[:, 4])) / command_dt if len(controls) > 1 else np.array([0.0])
    reference_dt = np.diff(trajectory.data[:, 0])
    reference_accel = np.abs(np.diff(trajectory.data[:, 6])) / reference_dt
    final_heading_error = abs(float(errors[-1, 3]))
    tracking_result = {
        "scale": scale,
        "device": str(device),
        "reference_samples": int(len(trajectory.data)),
        "execution_samples": int(len(executed)),
        "reference_duration_s": trajectory.duration,
        "execution_duration_s": float(executed[-1, 0]),
        "path_length_m": float(trajectory.metadata["total_length_m"]),
        "max_abs_curvature_1pm": float(trajectory.metadata["max_abs_curvature_1pm"]),
        "max_reference_speed_mps": float(np.max(trajectory.data[:, 6])),
        "max_reference_omega_radps": float(np.max(np.abs(trajectory.data[:, 7]))),
        "max_reference_accel_mps2": float(np.max(reference_accel)),
        "max_command_speed_mps": float(np.max(controls[:, 3])),
        "max_command_omega_radps": float(np.max(np.abs(controls[:, 4]))),
        "max_command_accel_mps2": float(np.max(command_accel)),
        "max_command_alpha_radps2": float(np.max(command_alpha)),
        "rmse_lateral_m": float(np.sqrt(np.mean(errors[:, 2] ** 2))),
        "max_lateral_error_m": float(np.max(np.abs(errors[:, 2]))),
        "final_position_error_m": float(errors[-1, 4]),
        "final_heading_error_rad": final_heading_error,
        "reference_continuous_ellipse_safe": bool(np.all(reference_exact_flags)),
        "reference_inflated_grid_safe": bool(np.all(reference_grid_flags)),
        "reference_inside_corridor": bool(np.all(reference_corridor_flags)),
        "continuous_ellipse_safe": bool(np.all(execution_exact_flags)),
        "inflated_grid_safe": bool(np.all(execution_grid_flags)),
        "inside_corridor": bool(np.all(execution_corridor_flags)),
        "corridor": {
            "reference": reference_corridor,
            "execution": execution_corridor,
        },
        "segment_checks": {
            "reference": {
                "continuous_projected_ellipses": flag_statistics(reference_exact_flags),
                "inflated_grid": flag_statistics(reference_grid_flags),
            },
            "execution": {
                "continuous_projected_ellipses": flag_statistics(execution_exact_flags),
                "inflated_grid": flag_statistics(execution_grid_flags),
            },
        },
        "safety_model_roles": {
            "inflated_grid": "conservative milestone-2 seed-search diagnostic",
            "safety_polygons": "hard constraint for the milestone-2 Bezier reference; execution diagnostic without a tracking tube",
            "projected_ellipses": "authoritative continuous collision model for reference and execution",
        },
        "reached_goal": bool(reached_goal),
        "limits": {
            "max_speed_mps": args.max_speed_mps,
            "max_accel_mps2": args.max_accel_mps2,
            "max_omega_radps": args.max_omega_radps,
            "max_alpha_radps2": args.max_alpha_radps2,
            "dt_s": args.dt,
        },
        "failure_reason": None,
    }
    tolerance = 1e-6
    required_verification = {
        "time_parameterization_valid": bool(np.all(np.diff(trajectory.data[:, 0]) > 0.0)),
        "reference_speed_within_limit": tracking_result["max_reference_speed_mps"] <= args.max_speed_mps + tolerance,
        "reference_omega_within_limit": tracking_result["max_reference_omega_radps"] <= args.max_omega_radps + tolerance,
        "reference_accel_within_limit": tracking_result["max_reference_accel_mps2"] <= args.max_accel_mps2 + 1e-5,
        "command_speed_within_limit": tracking_result["max_command_speed_mps"] <= args.max_speed_mps + tolerance,
        "command_omega_within_limit": tracking_result["max_command_omega_radps"] <= args.max_omega_radps + tolerance,
        "command_accel_within_limit": tracking_result["max_command_accel_mps2"] <= args.max_accel_mps2 + 1e-5,
        "command_alpha_within_limit": tracking_result["max_command_alpha_radps2"] <= args.max_alpha_radps2 + 1e-5,
        "reference_continuous_ellipse_safe": tracking_result["reference_continuous_ellipse_safe"],
        "reference_inside_corridor": tracking_result["reference_inside_corridor"],
        "execution_continuous_ellipse_safe": tracking_result["continuous_ellipse_safe"],
        "lateral_rmse_within_3cm": tracking_result["rmse_lateral_m"] <= 0.03,
        "max_lateral_error_within_8cm": tracking_result["max_lateral_error_m"] <= 0.08,
        "final_position_within_2cm": tracking_result["final_position_error_m"] <= 0.02,
        "final_heading_within_3deg": tracking_result["final_heading_error_rad"] <= np.deg2rad(3.0),
        "reached_goal": tracking_result["reached_goal"],
    }
    diagnostic_verification = {
        "reference_inflated_grid_safe": tracking_result["reference_inflated_grid_safe"],
        "execution_inflated_grid_safe": tracking_result["inflated_grid_safe"],
        "execution_inside_reference_corridor": tracking_result["inside_corridor"],
    }
    accepted = all(required_verification.values())
    tracking_verification = {
        **required_verification,
        "accepted": accepted,
        "diagnostic_only": diagnostic_verification,
    }
    if not accepted:
        failed = [key for key, value in required_verification.items() if not value]
        tracking_result["failure_reason"] = f"milestone-4 verification failed: {failed}"
    save_json(assets / "tracking_result.json", tracking_result)
    save_json(assets / "tracking_verification.json", tracking_verification)
    planning_result["tracking"] = tracking_result
    planning_result["failure_reason"] = tracking_result["failure_reason"]
    planning_verification["tracking"] = tracking_verification
    save_json(assets / "result.json", planning_result)
    save_json(assets / "verification.json", planning_verification)

    plot_timing(trajectory, assets, args.max_speed_mps, args.max_omega_radps)
    plot_tracking_overview(
        trajectory, aligned_reference, executed, errors, ellipses, polygons, extent,
        float(planning_result["scene_units"]["circle_radius"]), tracking_result,
        bad_points, assets,
    )
    plot_errors_and_controls(errors, controls, args.max_speed_mps, args.max_omega_radps, assets)
    print(json.dumps(
        {"tracking": tracking_result, "verification": tracking_verification},
        indent=2,
        default=json_default,
    ))
    if not accepted:
        raise SystemExit(tracking_result["failure_reason"])


if __name__ == "__main__":
    main()
