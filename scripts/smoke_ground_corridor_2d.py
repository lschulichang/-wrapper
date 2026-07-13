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
import numpy as np
import scipy.optimize
import scipy.spatial
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "splatnav-official"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(UPSTREAM))

from ground_nav.bezier_2d import BezierPlanner2D  # noqa: E402
from ground_nav.corridor_2d import PlanarCollisionSet, build_planar_corridor  # noqa: E402
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scene", default="old_union", choices=sorted(SCENE_PRESETS))
    parser.add_argument("--z-floor", type=float, default=-0.15)
    parser.add_argument("--robot-height-meters", type=float, default=0.10)
    parser.add_argument("--footprint-radius-meters", type=float, default=0.15)
    parser.add_argument("--ground-clearance-meters", type=float, default=0.02)
    parser.add_argument("--corridor-margin-meters", type=float, default=0.10)
    parser.add_argument("--max-segment-meters", type=float, default=1.0)
    parser.add_argument("--start", nargs=2, type=float, required=True)
    parser.add_argument("--goal", nargs=2, type=float, required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = json.loads((args.config.resolve().parent / "dataparser_transforms.json").read_text())
    scale = float(transform["scale"])
    height = args.robot_height_meters * scale
    radius = args.footprint_radius_meters * scale
    clearance = args.ground_clearance_meters * scale
    margin = args.corridor_margin_meters * scale
    preset = SCENE_PRESETS[args.scene]
    lower = torch.tensor(preset["lower_bound"], device=device)
    upper = torch.tensor(preset["upper_bound"], device=device)
    lower[2] = min(float(lower[2]), args.z_floor)
    upper[2] = max(float(upper[2]), args.z_floor + height)

    timings = {}
    t0 = time.time(); gsplat = GSplatLoader(args.config, device); sync(device); timings["gsplat_load"] = time.time() - t0
    t0 = time.time(); voxel = GSplatVoxel(gsplat, lower, upper, preset["resolution"], 0.0, device); sync(device); timings["voxel"] = time.time() - t0
    t0 = time.time(); grid = GroundGrid(
        voxel, args.z_floor, height, radius, ground_clearance=clearance, project_occupied_endpoints=False
    ); sync(device); timings["height_projection_xy_dilation"] = time.time() - t0
    t0 = time.time(); ellipses = PlanarGaussianSet.from_gsplat(gsplat, grid.metadata.z_min, grid.metadata.z_max); sync(device); timings["ellipse_projection"] = time.time() - t0
    collision_set = PlanarCollisionSet(ellipses, radius=radius, corridor_margin=margin, iterations=10)

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
        ids=ellipses.ids.detach().cpu().numpy(), means=ellipses.means.detach().cpu().numpy(), covs=ellipses.covs.detach().cpu().numpy(),
    )
    (args.output_dir / "polygons.json").write_text(json.dumps(polygon_json, indent=2))
    (args.output_dir / "verification.json").write_text(json.dumps(verification, indent=2))
    result = {
        "scene": args.scene, "device": str(device), "scale": scale,
        "physical": {
            "robot_height_m": args.robot_height_meters, "projected_circle_radius_m": args.footprint_radius_meters,
            "ground_clearance_m": args.ground_clearance_meters, "corridor_margin_m": args.corridor_margin_meters,
        },
        "scene_units": {
            "height": height, "circle_radius": radius, "ground_clearance": clearance,
            "corridor_margin": margin, "z_min": grid.metadata.z_min, "z_max": grid.metadata.z_max,
        },
        "gaussians_total": int(gsplat.means.shape[0]), "projected_ellipses": len(ellipses),
        "raw_seed_points": len(raw_seed), "simplified_seed_points": len(simplified),
        "polygon_count": len(corridor.polygons), "timings": timings,
        "qp_status": planner.last_solver_status, "failure_reason": None, "verification": verification,
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2))

    occupancy = grid.occupied.detach().cpu().numpy()
    extent = [float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    axes[0].imshow(grid.raw_occupied.cpu().numpy().T, origin="lower", extent=extent, cmap="Greys", aspect="equal")
    axes[0].plot(raw_seed[:, 0], raw_seed[:, 1], label="2D Dijkstra")
    axes[1].imshow(occupancy.T, origin="lower", extent=extent, cmap="Greys", aspect="equal")
    axes[1].plot(raw_seed[:, 0], raw_seed[:, 1], alpha=0.4, label="raw")
    axes[1].plot(simplified[:, 0], simplified[:, 1], "o-", label="LOS simplified")
    for ax in axes: ax.legend(); ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout(); fig.savefig(args.output_dir / "ground_grid_and_seed.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7)); ax.imshow(occupancy.T, origin="lower", extent=extent, cmap="Greys", alpha=0.45, aspect="equal")
    for A, b in corridor.polygons:
        vertices = polygon_vertices(A.detach().cpu().numpy(), b.detach().cpu().numpy())
        if vertices is not None: ax.fill(vertices[:, 0], vertices[:, 1], color="tab:green", alpha=0.18)
    ax.plot(simplified[:, 0], simplified[:, 1], "o--", label="seed")
    ax.plot(dense[:, :, 0].ravel(), dense[:, :, 1].ravel(), color="tab:orange", label="2D Bezier")
    ax.legend(); ax.set_aspect("equal"); fig.tight_layout(); fig.savefig(args.output_dir / "corridor_and_trajectory.png", dpi=180); plt.close(fig)

    print(json.dumps(result, indent=2))
    if not all([
        verification["path_is_strictly_2d"], verification["simplified_seed_exact_safe"],
        verification["qp_status"] == "Solved", verification["max_control_point_violation"] <= 1e-5,
        verification["max_endpoint_error"] <= 1e-6, verification["dense_trajectory_exact_safe"],
    ]):
        raise SystemExit("2D milestone verification failed")


if __name__ == "__main__":
    main()
