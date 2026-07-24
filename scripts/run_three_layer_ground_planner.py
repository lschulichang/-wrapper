#!/usr/bin/env python3
"""Run the single Hybrid A* -> corridor -> curvature-planning mainline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "splatnav-official"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(UPSTREAM))

from ground_nav import (  # noqa: E402
    GroundGrid,
    HybridAStarConfig,
    HybridAStarPlanner,
    PlanarCollisionSet,
    PlanarGaussianSet,
    ThreeLayerGroundPlanner,
    ThreeLayerPlannerConfig,
    TrajectoryOptimizerConfig,
    compute_stopping_distance,
)
from smoke_splatplan import SCENE_PRESETS  # noqa: E402
from splat.splat_utils import GSplatLoader  # noqa: E402


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def save_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        ),
        encoding="utf-8",
    )


def halfspace_polygon(A, b, tolerance=1e-8):
    """Return ordered vertices of a bounded 2D half-space polygon."""

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    vertices = []
    for left in range(len(b)):
        for right in range(left + 1, len(b)):
            matrix = np.stack([A[left], A[right]])
            determinant = float(np.linalg.det(matrix))
            if abs(determinant) <= 1e-12:
                continue
            point = np.linalg.solve(
                matrix, np.asarray([b[left], b[right]])
            )
            if np.all(A @ point <= b + tolerance):
                vertices.append(point)
    if len(vertices) < 3:
        return None
    unique = np.unique(
        np.round(np.asarray(vertices), decimals=12), axis=0
    )
    if len(unique) < 3:
        return None
    center = np.mean(unique, axis=0)
    angles = np.arctan2(
        unique[:, 1] - center[1],
        unique[:, 0] - center[0],
    )
    return unique[np.argsort(angles)]


def save_visualizations(
    output_dir: Path,
    result,
    curvature_limit: float,
    curvature_rate_limit: float,
) -> list[str]:
    """Save standard figures for every three-layer experiment."""

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    figure, axis = plt.subplots(figsize=(9.0, 7.0))
    if result.corridor is not None:
        for index, (A, b) in enumerate(
            result.corridor.corridors
        ):
            polygon = halfspace_polygon(
                np.asarray(A.detach().cpu()),
                np.asarray(b.detach().cpu()),
            )
            if polygon is None:
                continue
            axis.fill(
                polygon[:, 0],
                polygon[:, 1],
                color="#4C78A8",
                alpha=0.10,
                label=(
                    "GS safety corridors"
                    if index == 0
                    else None
                ),
            )
            axis.plot(
                np.r_[polygon[:, 0], polygon[0, 0]],
                np.r_[polygon[:, 1], polygon[0, 1]],
                color="#4C78A8",
                alpha=0.35,
                linewidth=0.8,
            )
    if result.hybrid_path is not None:
        path = result.hybrid_path.poses_scene
        axis.plot(
            path[:, 0],
            path[:, 1],
            linestyle="--",
            color="#7F7F7F",
            linewidth=1.1,
            label="Hybrid A* dense path",
        )
    if result.simplification is not None:
        path = result.simplification.boundary_poses_scene
        axis.plot(
            path[:, 0],
            path[:, 1],
            "o-",
            color="#F58518",
            markersize=3.0,
            linewidth=1.0,
            label="Motion-primitive boundaries",
        )
    if len(result.dense_output_poses_scene):
        path = result.dense_output_poses_scene
        axis.plot(
            path[:, 0],
            path[:, 1],
            color="#E45756",
            linewidth=2.0,
            label=(
                "Optimized trajectory"
                if result.success
                else "Hybrid fallback"
            ),
        )
        axis.scatter(
            path[0, 0],
            path[0, 1],
            marker="o",
            s=55,
            color="#54A24B",
            label="Start",
            zorder=5,
        )
        axis.scatter(
            path[-1, 0],
            path[-1, 1],
            marker="*",
            s=95,
            color="#B279A2",
            label="Goal",
            zorder=5,
        )
    axis.set_title(
        "Three-layer ground planning"
        + (" — success" if result.success else " — fallback")
    )
    axis.set_xlabel("Scene x")
    axis.set_ylabel("Scene y")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    overview_path = figure_dir / "planning_overview.png"
    figure.savefig(overview_path, dpi=180)
    plt.close(figure)
    generated.append(str(overview_path.relative_to(output_dir)))

    if result.trajectory_m is not None:
        trajectory = result.trajectory_m
        figure, axes = plt.subplots(
            2, 1, figsize=(9.0, 6.5), sharex=True
        )
        axes[0].plot(
            trajectory.arc_length,
            trajectory.curvature,
            color="#4C78A8",
            linewidth=1.8,
        )
        axes[0].axhline(
            curvature_limit,
            color="#E45756",
            linestyle="--",
            linewidth=1.0,
            label="limits",
        )
        axes[0].axhline(
            -curvature_limit,
            color="#E45756",
            linestyle="--",
            linewidth=1.0,
        )
        axes[0].set_ylabel(r"$\kappa$ (m$^{-1}$)")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="best")
        axes[1].plot(
            trajectory.arc_length,
            trajectory.curvature_rate,
            color="#F58518",
            linewidth=1.8,
        )
        axes[1].axhline(
            curvature_rate_limit,
            color="#E45756",
            linestyle="--",
            linewidth=1.0,
            label="limits",
        )
        axes[1].axhline(
            -curvature_rate_limit,
            color="#E45756",
            linestyle="--",
            linewidth=1.0,
        )
        axes[1].set_xlabel("Arc length s (m)")
        axes[1].set_ylabel(r"$\sigma$ (m$^{-2}$)")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="best")
        figure.suptitle(
            "Curvature and curvature-rate profiles"
        )
        figure.tight_layout()
        profile_path = (
            figure_dir / "curvature_profiles.png"
        )
        figure.savefig(profile_path, dpi=180)
        plt.close(figure)
        generated.append(
            str(profile_path.relative_to(output_dir))
        )

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    timing_names = list(result.timings)
    timing_values = [
        float(result.timings[name]) for name in timing_names
    ]
    labels = [
        name.removesuffix("_s").replace("_", " ")
        for name in timing_names
    ]
    bars = axis.barh(
        labels, timing_values, color="#72B7B2"
    )
    axis.bar_label(
        bars,
        labels=[f"{value:.3f} s" for value in timing_values],
        padding=3,
        fontsize=8,
    )
    axis.set_xlabel("Wall-clock time (s)")
    axis.set_title("Planning stage timing")
    axis.grid(True, axis="x", alpha=0.25)
    figure.tight_layout()
    timing_path = figure_dir / "stage_timings.png"
    figure.savefig(timing_path, dpi=180)
    plt.close(figure)
    generated.append(str(timing_path.relative_to(output_dir)))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--scene",
        default="old_union",
        choices=sorted(SCENE_PRESETS),
    )
    parser.add_argument(
        "--start-pose",
        nargs=3,
        required=True,
        type=float,
        metavar=("X", "Y", "YAW"),
    )
    parser.add_argument(
        "--goal-xy",
        nargs=2,
        required=True,
        type=float,
        metavar=("X", "Y"),
    )
    parser.add_argument("--z-floor-scene", type=float, default=-0.15)
    parser.add_argument(
        "--robot-height-meters", type=float, default=0.10
    )
    parser.add_argument(
        "--footprint-radius-meters", type=float, default=0.15
    )
    parser.add_argument(
        "--ground-clearance-meters", type=float, default=0.02
    )
    parser.add_argument("--max-speed-mps", type=float, default=0.20)
    parser.add_argument(
        "--max-brake-decel-mps2", type=float, default=0.30
    )
    parser.add_argument(
        "--min-turning-radius-meters", type=float, default=0.20
    )
    parser.add_argument(
        "--max-curvature-rate-1pm2", type=float, default=10.0
    )
    parser.add_argument(
        "--hybrid-heading-bins", type=int, default=72
    )
    parser.add_argument(
        "--hybrid-max-expansions", type=int, default=200_000
    )
    parser.add_argument(
        "--optimizer-nodes", type=int, default=12
    )
    parser.add_argument(
        "--optimizer-max-iterations", type=int, default=500
    )
    parser.add_argument(
        "--optimizer-max-refinements", type=int, default=2
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_pose = np.asarray(args.start_pose, dtype=np.float64)
    goal_xy = np.asarray(args.goal_xy, dtype=np.float64)
    if not np.all(np.isfinite(start_pose)) or not np.all(
        np.isfinite(goal_xy)
    ):
        raise ValueError("start pose and goal position must be finite")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    transform = json.loads(
        (
            args.config.resolve().parent
            / "dataparser_transforms.json"
        ).read_text(encoding="utf-8")
    )
    scale = float(transform["scale"])
    height = args.robot_height_meters * scale
    radius = args.footprint_radius_meters * scale
    clearance = args.ground_clearance_meters * scale
    stopping_distance_m = compute_stopping_distance(
        args.max_speed_mps,
        args.max_brake_decel_mps2,
    )
    preset = SCENE_PRESETS[args.scene]
    lower = torch.tensor(
        preset["lower_bound"], device=device
    )
    upper = torch.tensor(
        preset["upper_bound"], device=device
    )
    preprocessing = {}

    synchronize(device)
    begin = time.perf_counter()
    gsplat = GSplatLoader(args.config, device)
    synchronize(device)
    preprocessing["gsplat_load_s"] = (
        time.perf_counter() - begin
    )

    begin = time.perf_counter()
    ellipses = PlanarGaussianSet.from_gsplat(
        gsplat,
        args.z_floor_scene + clearance,
        args.z_floor_scene + height,
        confidence=1.0,
    )
    synchronize(device)
    preprocessing["height_band_projection_s"] = (
        time.perf_counter() - begin
    )

    begin = time.perf_counter()
    grid = GroundGrid.from_projected_gaussians(
        ellipses,
        lower_xy=lower[:2],
        upper_xy=upper[:2],
        resolution_xy=preset["resolution"],
        footprint_radius=radius,
        z_floor_scene=args.z_floor_scene,
        robot_height=height,
        ground_clearance=clearance,
    )
    synchronize(device)
    preprocessing["ellipse_distance_grid_s"] = (
        time.perf_counter() - begin
    )

    collision_set = PlanarCollisionSet(
        ellipses,
        radius=radius,
        stopping_distance=stopping_distance_m * scale,
        iterations=10,
    )
    hybrid = HybridAStarPlanner(
        grid,
        scene_scale=scale,
        config=HybridAStarConfig(
            heading_bins=args.hybrid_heading_bins,
            min_turning_radius_m=(
                args.min_turning_radius_meters
            ),
            max_expansions=args.hybrid_max_expansions,
        ),
    )
    planner = ThreeLayerGroundPlanner(
        hybrid,
        collision_set,
        ThreeLayerPlannerConfig(
            min_turning_radius_m=(
                args.min_turning_radius_meters
            ),
            max_curvature_rate_1pm2=(
                args.max_curvature_rate_1pm2
            ),
            optimizer=TrajectoryOptimizerConfig(
                node_count=args.optimizer_nodes,
                max_iterations=args.optimizer_max_iterations,
                max_refinement_steps=(
                    args.optimizer_max_refinements
                ),
            ),
        ),
    )

    result = planner.plan(start_pose, goal_xy)
    output_kind = (
        "optimized_path"
        if result.success
        else (
            "hybrid_fallback"
            if result.fallback_used
            else "no_path"
        )
    )
    np.save(
        args.output_dir / "selected_output_poses_scene.npy",
        result.dense_output_poses_scene,
    )
    if result.hybrid_path is not None:
        np.save(
            args.output_dir / "hybrid_path_poses_scene.npy",
            result.hybrid_path.poses_scene,
        )
    if result.trajectory_m is not None:
        np.save(
            args.output_dir / "optimized_path_poses_m.npy",
            result.trajectory_m.dense_poses,
        )
        np.save(
            args.output_dir / "collocation_poses_m.npy",
            result.trajectory_m.poses,
        )
        np.save(
            args.output_dir / "collocation_arc_length_m.npy",
            result.trajectory_m.arc_length,
        )
        np.save(
            args.output_dir / "collocation_curvature_1pm.npy",
            result.trajectory_m.curvature,
        )
        np.save(
            args.output_dir
            / "collocation_curvature_rate_1pm2.npy",
            result.trajectory_m.curvature_rate,
        )
    curvature_limit = (
        0.95 / args.min_turning_radius_meters
    )
    visualizations = save_visualizations(
        args.output_dir,
        result,
        curvature_limit=curvature_limit,
        curvature_rate_limit=args.max_curvature_rate_1pm2,
    )

    report = {
        "mainline": (
            "3D Gaussian Splat -> height-band planar ellipses -> "
            "ellipse-distance GroundGrid -> forward-only Hybrid A* -> "
            "motion-primitive boundaries -> ordered GS corridor -> "
            "Hermite-Simpson arc-length direct collocation -> "
            "curvature, curvature-rate, and corridor verification"
        ),
        "corridor_reference": "motion_primitive_boundaries",
        "status": result.status(),
        "output_kind": output_kind,
        "visualizations": visualizations,
        "scene": args.scene,
        "device": str(device),
        "scene_scale": scale,
        "start_pose_scene": start_pose.tolist(),
        "goal_xy_scene": goal_xy.tolist(),
        "preprocessing": preprocessing,
        "ground_grid": grid.summary(),
        "hybrid_path": (
            None
            if result.hybrid_path is None
            else result.hybrid_path.summary()
        ),
        "simplification": (
            None
            if result.simplification is None
            else result.simplification.summary()
        ),
        "corridor": (
            None
            if result.corridor is None
            else {
                "corridor_count": len(
                    result.corridor.corridors
                ),
                "overlap_count": len(result.corridor.overlaps),
                "overlaps": result.corridor.overlaps,
            }
        ),
        "optimizer": (
            None
            if result.trajectory_m is None
            else result.trajectory_m.summary()
        ),
        "optimizer_diagnostics": (
            []
            if result.trajectory_m is None
            else result.trajectory_m.diagnostics
        ),
        "constraints": {
            "min_turning_radius_m": (
                args.min_turning_radius_meters
            ),
            "curvature_limit_1pm": (
                curvature_limit
            ),
            "max_curvature_rate_1pm2": (
                args.max_curvature_rate_1pm2
            ),
            "terminal_heading": "free_with_soft_reference",
        },
    }
    save_json(args.output_dir / "result.json", report)
    print(json.dumps(report, indent=2, default=json_default))


if __name__ == "__main__":
    main()
