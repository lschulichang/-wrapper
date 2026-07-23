#!/usr/bin/env python3
"""Run the single Hybrid A* -> corridor -> curvature-planning mainline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


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
        "--optimizer-nodes", type=int, default=32
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

    report = {
        "mainline": (
            "3D Gaussian Splat -> height-band planar ellipses -> "
            "ellipse-distance GroundGrid -> forward-only Hybrid A* -> "
            "ordered GS corridor -> arc-length direct collocation -> "
            "curvature, curvature-rate, and corridor verification"
        ),
        "status": result.status(),
        "output_kind": output_kind,
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
                0.95 / args.min_turning_radius_meters
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
