#!/usr/bin/env python3
"""Run the planar Gaussian corridor and 2D Bezier milestone."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import scipy.optimize
import scipy.spatial
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "splatnav-official"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(UPSTREAM))

from ground_nav.bezier_2d import BezierPlanner2D  # noqa: E402
from ground_nav.corridor_2d import (  # noqa: E402
    PlanarCollisionSet,
    build_planar_corridor,
    compute_stopping_distance,
)
from ground_nav.ground_grid import GroundGrid  # noqa: E402
from ground_nav.path_utils import simplify_ground_path  # noqa: E402
from ground_nav.planar_gaussians import PlanarGaussianSet  # noqa: E402
from initialization.grid_utils import GSplatVoxel  # noqa: E402
from smoke_splatplan import SCENE_PRESETS  # noqa: E402
from splat.splat_utils import GSplatLoader  # noqa: E402


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def polygon_vertices(A, b):
    A = np.asarray(A); b = np.asarray(b)
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    result = scipy.optimize.linprog(
        np.array([0.0, 0.0, -1.0]), A_ub=np.hstack([A, norms]), b_ub=b,
        bounds=[(None, None), (None, None), (0.0, None)], method="highs",
    )
    if not result.success or result.x[2] <= 1e-9:
        return None
    try:
        hs = scipy.spatial.HalfspaceIntersection(np.hstack([A, -b[:, None]]), result.x[:2])
        return hs.intersections[scipy.spatial.ConvexHull(hs.intersections).vertices]
    except Exception:
        return None


def max_control_violation(controls, polygons):
    value = -np.inf
    for control, (A, b) in zip(controls, polygons):
        A = A.detach().cpu().numpy(); b = b.detach().cpu().numpy()
        value = max(value, float(np.max(A @ control - b[:, None])))
    return value


def projected_ellipse_polylines(ellipses, samples: int = 32):
    """Return low-cost polylines for the height-filtered projected ellipses."""

    if samples < 8:
        raise ValueError("ellipse samples must be at least 8")
    angles = torch.linspace(
        0.0,
        2.0 * torch.pi,
        samples + 1,
        device=ellipses.means.device,
        dtype=ellipses.means.dtype,
    )
    unit_circle = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    local = ellipses.scales[:, None, :] * unit_circle[None, :, :]
    world = ellipses.means[:, None, :] + torch.einsum("nij,nkj->nki", ellipses.rots, local)
    return world.detach().cpu().numpy()


def configure_map_axis(ax, extent, title):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scene", default="old_union", choices=sorted(SCENE_PRESETS))
    parser.add_argument(
        "--z-floor-scene", "--z-floor", dest="z_floor_scene", type=float, default=-0.15,
        help="Floor Z in normalized scene coordinates; --z-floor is a deprecated alias",
    )
    parser.add_argument("--robot-height-meters", type=float, default=0.10)
    parser.add_argument("--footprint-radius-meters", type=float, default=0.15)
    parser.add_argument("--ground-clearance-meters", type=float, default=0.02)
    parser.add_argument("--max-speed-mps", type=float, default=0.20)
    parser.add_argument("--max-brake-decel-mps2", type=float, default=0.30)
    parser.add_argument(
        "--corridor-margin-meters",
        type=float,
        default=None,
        help="Deprecated manual override; default is vmax^2/(2*max_brake_decel)",
    )
    parser.add_argument("--max-segment-meters", type=float, default=1.0)
    parser.add_argument("--start", nargs=2, type=float, required=True)
    parser.add_argument("--goal", nargs=2, type=float, required=True)
    parser.add_argument(
        "--legacy-voxel-diagnostic",
        action="store_true",
        help="Build the former 3D-voxel projection only for comparison outputs",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = json.loads((args.config.resolve().parent / "dataparser_transforms.json").read_text())
    scale = float(transform["scale"])
    height = args.robot_height_meters * scale
    radius = args.footprint_radius_meters * scale
    clearance = args.ground_clearance_meters * scale
    computed_stopping_distance_m = compute_stopping_distance(
        args.max_speed_mps, args.max_brake_decel_mps2
    )
    if args.corridor_margin_meters is None:
        stopping_distance_m = computed_stopping_distance_m
        stopping_distance_source = "vmax_squared_over_2_brake_decel"
    else:
        if args.corridor_margin_meters < 0.0:
            raise ValueError("corridor-margin-meters must be non-negative")
        stopping_distance_m = float(args.corridor_margin_meters)
        stopping_distance_source = "manual_corridor_margin_override"
    stopping_distance = stopping_distance_m * scale
    preset = SCENE_PRESETS[args.scene]
    lower = torch.tensor(preset["lower_bound"], device=device)
    upper = torch.tensor(preset["upper_bound"], device=device)
    lower[2] = min(float(lower[2]), args.z_floor_scene)
    upper[2] = max(float(upper[2]), args.z_floor_scene + height)
    z_min = float(args.z_floor_scene + clearance)
    z_max = float(args.z_floor_scene + height)

    timings = {}
    t0 = time.time(); gsplat = GSplatLoader(args.config, device); sync(device); timings["gsplat_load"] = time.time() - t0
    t0 = time.time(); ellipses = PlanarGaussianSet.from_gsplat(
        gsplat, z_min, z_max, confidence=1.0
    ); sync(device); timings["height_filter_full_projection"] = time.time() - t0
    t0 = time.time(); grid = GroundGrid.from_projected_gaussians(
        ellipses,
        lower_xy=lower[:2],
        upper_xy=upper[:2],
        resolution_xy=preset["resolution"],
        footprint_radius=radius,
        z_floor_scene=args.z_floor_scene,
        robot_height=height,
        ground_clearance=clearance,
        project_occupied_endpoints=False,
    ); sync(device); timings["projected_ellipse_rasterization"] = time.time() - t0
    legacy_grid = None
    if args.legacy_voxel_diagnostic:
        t0 = time.time()
        voxel = GSplatVoxel(
            gsplat, lower, upper, preset["resolution"], 0.0, device
        )
        sync(device)
        timings["legacy_voxel"] = time.time() - t0
        t0 = time.time()
        legacy_grid = GroundGrid(
            voxel,
            args.z_floor_scene,
            height,
            radius,
            ground_clearance=clearance,
            project_occupied_endpoints=False,
        )
        sync(device)
        timings["legacy_height_projection_xy_dilation"] = time.time() - t0
    collision_set = PlanarCollisionSet(
        ellipses, radius=radius, stopping_distance=stopping_distance, iterations=10
    )

    start, goal = np.asarray(args.start, np.float32), np.asarray(args.goal, np.float32)
    if grid.is_occupied(start) or grid.is_occupied(goal):
        raise RuntimeError("start or goal is occupied; automatic projection is disabled")
    t0 = time.time(); raw_seed = grid.create_path(start, goal); timings["dijkstra_2d"] = time.time() - t0
    t0 = time.time(); simplified = simplify_ground_path(raw_seed, grid, args.max_segment_meters * scale); timings["simplification"] = time.time() - t0
    segments = np.stack([simplified[:-1], simplified[1:]], axis=1)
    t0 = time.time(); exact_seed = [collision_set.segment_is_safe(segment) for segment in segments]; sync(device); timings["seed_exact_verification"] = time.time() - t0
    if not all(exact_seed):
        bad = [i for i, safe in enumerate(exact_seed) if not safe]
        raise RuntimeError(f"inflated grid and projected ellipses disagree on simplified segments {bad}")

    t0 = time.time(); corridor = build_planar_corridor(simplified, collision_set); sync(device); timings["corridor"] = time.time() - t0
    planner = BezierPlanner2D(degree=6, continuity_order=3)
    t0 = time.time(); controls, feasible = planner.optimize(corridor.polygons, simplified[0], simplified[-1]); timings["bezier_qp"] = time.time() - t0
    if not feasible:
        raise RuntimeError(f"2D Bezier QP failed: {planner.last_solver_status}")
    dense = planner.sample(100)

    dense_safe = True
    for section in dense:
        for p0, p1 in zip(section[:-1], section[1:]):
            if not collision_set.segment_is_safe(np.stack([p0, p1])):
                dense_safe = False
                break
        if not dense_safe:
            break
    control_violation = max_control_violation(controls, corridor.polygons)
    endpoint_error = max(np.linalg.norm(dense[0, 0] - simplified[0]), np.linalg.norm(dense[-1, -1] - simplified[-1]))
    verification = {
        "path_is_strictly_2d": bool(raw_seed.shape[1] == simplified.shape[1] == dense.shape[2] == 2),
        "simplified_seed_exact_safe": bool(all(exact_seed)),
        "qp_status": planner.last_solver_status,
        "max_control_point_violation": control_violation,
        "max_endpoint_error": float(endpoint_error),
        "dense_trajectory_exact_safe": bool(dense_safe),
    }

    polygon_json = [
        np.concatenate([A.detach().cpu().numpy(), b.detach().cpu().numpy()[:, None]], axis=1).tolist()
        for A, b in corridor.polygons
    ]
    np.save(args.output_dir / "raw_seed.npy", raw_seed)
    np.save(args.output_dir / "simplified_seed.npy", simplified)
    np.save(args.output_dir / "control_points.npy", controls)
    np.save(args.output_dir / "trajectory.npy", dense)
    np.savez_compressed(
        args.output_dir / "projected_gaussians.npz",
        ids=ellipses.ids.detach().cpu().numpy(),
        means=ellipses.means.detach().cpu().numpy(),
        covs=ellipses.covs.detach().cpu().numpy(),
        rots=ellipses.rots.detach().cpu().numpy(),
        scales=ellipses.scales.detach().cpu().numpy(),
        z_min=np.asarray(ellipses.z_min, dtype=np.float64),
        z_max=np.asarray(ellipses.z_max, dtype=np.float64),
        confidence=np.asarray(ellipses.confidence, dtype=np.float64),
        projection_mode=np.asarray(ellipses.projection_mode),
    )
    np.savez_compressed(
        args.output_dir / "ground_grid.npz",
        raw_occupied=grid.raw_occupied.detach().cpu().numpy(),
        occupied=grid.occupied.detach().cpu().numpy(),
        lower_center=grid.lower_center.detach().cpu().numpy(),
        cell_sizes=grid.cell_sizes.detach().cpu().numpy(),
        shape=np.asarray(grid.shape, dtype=np.int64),
        extent=np.asarray([
            float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1])
        ], dtype=np.float64),
        scale=np.asarray(scale, dtype=np.float64),
        source=np.asarray(grid.metadata.source),
        rasterization=np.asarray(grid.metadata.rasterization),
        footprint_radius_scene=np.asarray(radius, dtype=np.float64),
        cell_circumradius_scene=np.asarray(
            grid.metadata.cell_circumradius, dtype=np.float64
        ),
    )
    (args.output_dir / "polygons.json").write_text(json.dumps(polygon_json, indent=2))
    (args.output_dir / "verification.json").write_text(json.dumps(verification, indent=2))
    result = {
        "scene": args.scene, "device": str(device), "scale": scale,
        "physical": {
            "robot_height_m": args.robot_height_meters, "projected_circle_radius_m": args.footprint_radius_meters,
            "ground_clearance_m": args.ground_clearance_meters,
            "max_speed_mps": args.max_speed_mps,
            "max_brake_decel_mps2": args.max_brake_decel_mps2,
            "computed_stopping_distance_m": computed_stopping_distance_m,
            "stopping_distance_m": stopping_distance_m,
            "stopping_distance_source": stopping_distance_source,
            "corridor_margin_m": stopping_distance_m,
        },
        "scene_units": {
            "height": height, "circle_radius": radius, "ground_clearance": clearance,
            "stopping_distance": stopping_distance,
            "collision_box_lateral_half_width": radius + stopping_distance,
            "corridor_margin": stopping_distance,
            "z_floor_scene": args.z_floor_scene,
            "z_min": grid.metadata.z_min, "z_max": grid.metadata.z_max,
        },
        "gaussians_total": int(gsplat.means.shape[0]), "projected_ellipses": len(ellipses),
        "map": {
            **grid.summary(),
            "projection_mode": ellipses.projection_mode,
            "legacy_voxel_diagnostic": bool(args.legacy_voxel_diagnostic),
        },
        "raw_seed_points": len(raw_seed), "simplified_seed_points": len(simplified),
        "polygon_count": len(corridor.polygons), "timings": timings,
        "qp_status": planner.last_solver_status, "failure_reason": None, "verification": verification,
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2))

    raw_occupancy = grid.raw_occupied.detach().cpu().numpy()
    occupancy = grid.occupied.detach().cpu().numpy()
    extent = [float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1])]

    # Figure 1: one continuous source model and its two conservative grids.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    axes[0].add_collection(LineCollection(
        projected_ellipse_polylines(ellipses),
        colors="tab:red",
        linewidths=0.25,
        alpha=0.25,
        rasterized=True,
    ))
    axes[1].imshow(raw_occupancy.T, origin="lower", extent=extent, cmap="Greys", aspect="equal")
    axes[2].imshow(occupancy.T, origin="lower", extent=extent, cmap="Greys", aspect="equal")
    configure_map_axis(axes[0], extent, "Height-filtered projected ellipses")
    configure_map_axis(axes[1], extent, "Conservative ellipse rasterization")
    configure_map_axis(axes[2], extent, "Robot-footprint search grid")
    fig.tight_layout()
    fig.savefig(args.output_dir / "planar_map_generation.png", dpi=180)
    plt.close(fig)

    if legacy_grid is not None:
        legacy_raw = legacy_grid.raw_occupied.detach().cpu().numpy()
        legacy_occupied = legacy_grid.occupied.detach().cpu().numpy()
        false_positive = legacy_occupied & ~occupancy
        false_negative = ~legacy_occupied & occupancy
        np.savez_compressed(
            args.output_dir / "legacy_voxel_projection.npz",
            raw_occupied=legacy_raw,
            occupied=legacy_occupied,
            extent=np.asarray(extent, dtype=np.float64),
        )
        comparison = {
            "authority": "projected_gaussians",
            "legacy_false_positive_cells": int(false_positive.sum()),
            "legacy_false_negative_cells": int(false_negative.sum()),
            "legacy_occupied_cells": int(legacy_occupied.sum()),
            "projected_gaussian_occupied_cells": int(occupancy.sum()),
        }
        (args.output_dir / "map_model_comparison.json").write_text(
            json.dumps(comparison, indent=2)
        )
        result["map"]["legacy_comparison"] = comparison
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2))
        fig, axes = plt.subplots(2, 2, figsize=(12, 11), sharex=True, sharey=True)
        layers = (
            (legacy_occupied, "Legacy 3D-voxel projection", "Greys"),
            (occupancy, "Projected-Gaussian search grid", "Greys"),
            (false_positive, "Legacy false positives", "Reds"),
            (false_negative, "Legacy false negatives", "Blues"),
        )
        for axis, (layer, title, cmap) in zip(axes.ravel(), layers):
            axis.imshow(
                layer.T, origin="lower", extent=extent, cmap=cmap,
                aspect="equal", interpolation="nearest",
            )
            configure_map_axis(axis, extent, title)
        fig.tight_layout()
        fig.savefig(args.output_dir / "legacy_map_comparison.png", dpi=180)
        plt.close(fig)

    # Figure 2: the direct 2D Dijkstra path and its deterministic LOS simplification.
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(occupancy.T, origin="lower", extent=extent, cmap="Greys", alpha=0.72, aspect="equal")
    ax.plot(raw_seed[:, 0], raw_seed[:, 1], color="tab:blue", linewidth=1.5, label="2D Dijkstra")
    ax.plot(
        simplified[:, 0], simplified[:, 1], "o--", color="tab:orange", linewidth=1.8,
        markersize=4.5, label="LOS simplified",
    )
    configure_map_axis(ax, extent, "Projected-Gaussian search grid and 2D seed paths")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "dijkstra_and_simplified_path.png", dpi=180)
    plt.close(fig)

    # Figure 3: continuous projected-Gaussian geometry, convex corridor, and Bezier result.
    fig, ax = plt.subplots(figsize=(8, 7))
    ellipse_lines = LineCollection(
        projected_ellipse_polylines(ellipses),
        colors="tab:red",
        linewidths=0.28,
        alpha=0.22,
        rasterized=True,
        label="height-filtered projected ellipses",
        zorder=1,
    )
    ax.add_collection(ellipse_lines)
    polygon_label = True
    for A, b in corridor.polygons:
        vertices = polygon_vertices(A.detach().cpu().numpy(), b.detach().cpu().numpy())
        if vertices is not None:
            ax.fill(
                vertices[:, 0], vertices[:, 1], color="tab:green", alpha=0.18,
                label="safe polygons" if polygon_label else None, zorder=2,
            )
            polygon_label = False
    ax.plot(
        simplified[:, 0], simplified[:, 1], "o--", color="tab:blue", linewidth=1.6,
        markersize=4.5, label="simplified seed", zorder=3,
    )
    ax.plot(
        dense[:, :, 0].ravel(), dense[:, :, 1].ravel(), color="tab:orange",
        linewidth=2.0, label="2D Bezier", zorder=4,
    )
    configure_map_axis(ax, extent, "Projected ellipses, safe corridor, and 2D Bezier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "projected_ellipses_corridor_bezier.png", dpi=180)
    plt.close(fig)

    print(json.dumps(result, indent=2))
    if not all([
        verification["path_is_strictly_2d"], verification["simplified_seed_exact_safe"],
        verification["qp_status"] == "Solved", verification["max_control_point_violation"] <= 1e-5,
        verification["max_endpoint_error"] <= 1e-6, verification["dense_trajectory_exact_safe"],
    ]):
        raise SystemExit("2D milestone verification failed")


if __name__ == "__main__":
    main()
