#!/usr/bin/env python3
"""Run milestone 2: ground seed, GS corridor, and fixed-height Bezier QP."""

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
import scipy.special
import scipy.spatial
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "splatnav-official"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(UPSTREAM))

from ellipsoids.intersection_utils import compute_intersection_linear_motion  # noqa: E402
from ground_nav.ground_grid import GroundGrid  # noqa: E402
from ground_nav.path_utils import circumscribed_sphere_radius, gaussian_height_mask, simplify_ground_path  # noqa: E402
from initialization.grid_utils import GSplatVoxel  # noqa: E402
from polytopes.collision_set import GSplatCollisionSet  # noqa: E402
from smoke_splatplan import SCENE_PRESETS  # noqa: E402
from splat.splat_utils import GSplatLoader  # noqa: E402
from splatplan.splatplan import SplatPlan  # noqa: E402
from splatplan.spline_utils import SplinePlanner  # noqa: E402


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def exact_segment_safe(segment, collision_set) -> bool:
    segment_t = torch.as_tensor(segment, dtype=torch.float32, device=collision_set.device)
    candidates = collision_set.compute_set_one_step(segment_t)
    if candidates["means"].shape[0] == 0:
        return True
    output = compute_intersection_linear_motion(
        segment_t[0], segment_t[1] - segment_t[0], candidates["rots"], candidates["scales"],
        candidates["means"], R_B=None, S_B=float(collision_set.radius), collision_type="sphere", N=10,
    )
    return bool(torch.all(output["is_not_intersect"]).item())


def dense_bezier(coeffs, samples=100):
    degree = coeffs.shape[-1] - 1
    t = np.linspace(0.0, 1.0, samples)
    basis = np.stack([scipy.special.comb(degree, i) * (1-t)**(degree-i) * t**i for i in range(degree+1)])
    return np.stack([(section @ basis).T for section in coeffs])


def slice_vertices(polytope, z):
    matrix = np.asarray(polytope)
    A, b = matrix[:, :3], matrix[:, 3]
    A2, b2 = A[:, :2], b - A[:, 2] * z
    norms = np.linalg.norm(A2, axis=1, keepdims=True)
    c = np.array([0.0, 0.0, -1.0])
    Aub = np.hstack([A2, norms])
    result = scipy.optimize.linprog(c, A_ub=Aub, b_ub=b2, bounds=[(None,None),(None,None),(0,None)], method="highs")
    if not result.success or result.x[2] <= 1e-8:
        return None
    halfspaces = np.hstack([A2, -b2[:, None]])
    try:
        hs = scipy.spatial.HalfspaceIntersection(halfspaces, result.x[:2])
        vertices = hs.intersections
        return vertices[scipy.spatial.ConvexHull(vertices).vertices]
    except Exception:
        return None


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
    preset = SCENE_PRESETS[args.scene]
    transform = json.loads((args.config.resolve().parent / "dataparser_transforms.json").read_text())
    scale = float(transform["scale"])
    height = args.robot_height_meters * scale
    footprint = args.footprint_radius_meters * scale
    clearance = args.ground_clearance_meters * scale
    sphere_radius_m = circumscribed_sphere_radius(args.footprint_radius_meters, args.robot_height_meters)
    sphere_radius = sphere_radius_m * scale
    corridor_margin = args.corridor_margin_meters * scale

    timings = {}
    t0 = time.time(); gsplat = GSplatLoader(args.config, device); sync(device); timings["gsplat_load"] = time.time()-t0
    lower = torch.tensor(preset["lower_bound"], device=device)
    upper = torch.tensor(preset["upper_bound"], device=device)
    upper[2] = max(float(upper[2]), args.z_floor + height)
    t0 = time.time(); raw_voxel = GSplatVoxel(gsplat, lower, upper, preset["resolution"], 0.0, device); sync(device); timings["voxel"] = time.time()-t0
    t0 = time.time(); grid = GroundGrid(raw_voxel, args.z_floor, height, sphere_radius, ground_clearance=clearance, project_occupied_endpoints=False); sync(device); timings["projection_dilation"] = time.time()-t0

    mask = gaussian_height_mask(gsplat, grid.metadata.z_min, grid.metadata.z_max)
    collision_set = GSplatCollisionSet(gsplat, 0.0, 1.0, float(sphere_radius), device, primitive_mask=mask, corridor_margin=float(corridor_margin))
    start, goal = np.asarray(args.start, np.float32), np.asarray(args.goal, np.float32)
    if grid.is_occupied(start) or grid.is_occupied(goal):
        raise RuntimeError("start or goal is occupied; automatic projection is disabled")
    t0 = time.time(); raw_seed = grid.create_path(start, goal); timings["seed_search"] = time.time()-t0
    t0 = time.time(); simplified = simplify_ground_path(raw_seed, grid, args.max_segment_meters*scale, lambda seg: exact_segment_safe(seg, collision_set)); sync(device); timings["simplify_validate"] = time.time()-t0

    spline = SplinePlanner(spline_deg=6, N_sec=10, device=device)
    robot = {"radius": float(sphere_radius), "vmax": 0.0, "amax": 1.0}
    env = {"lower_bound": lower, "upper_bound": upper, "resolution": preset["resolution"]}
    planner = SplatPlan(gsplat, robot, env, spline, device, seed_planner=grid, collision_set=collision_set)
    planner.raise_on_infeasible = True
    t0 = time.time(); output = planner.generate_from_seed(simplified, fixed_z=grid.metadata.reference_z, time_seed=timings["seed_search"]); sync(device); timings["corridor_and_qp"] = time.time()-t0

    coeffs = np.asarray(output["coeffs"])
    trajectory = np.asarray(output["traj"])
    dense = dense_bezier(coeffs, 100)
    poly_mats = np.asarray(output["polytopes"], dtype=object)
    control_violation = 0.0
    for coeff, matrix in zip(coeffs, output["polytopes"]):
        matrix = np.asarray(matrix); A, b = matrix[:, :3], matrix[:, 3]
        control_violation = max(control_violation, float(np.max(A @ coeff - b[:, None])))
    dense_safe = True
    for section in dense:
        for p0, p1 in zip(section[:-1], section[1:]):
            if not exact_segment_safe(np.stack([p0, p1]), collision_set):
                dense_safe = False; break
        if not dense_safe: break

    np.save(args.output_dir / "raw_seed.npy", raw_seed)
    np.save(args.output_dir / "simplified_seed.npy", simplified)
    np.save(args.output_dir / "trajectory.npy", trajectory)
    np.save(args.output_dir / "trajectory_dense.npy", dense)
    (args.output_dir / "polytopes.json").write_text(json.dumps(output["polytopes"], indent=2))

    verification = {
        "seed_segments_exact_safe": all(exact_segment_safe(seg, collision_set) for seg in np.stack([simplified[:-1], simplified[1:]], axis=1)),
        "qp_feasible": bool(output["feasible"]),
        "qp_status": output["qp_status"],
        "num_polytopes": int(output["num_polytopes"]),
        "max_control_point_violation": control_violation,
        "max_z_error": float(np.max(np.abs(dense[:,:,2] - grid.metadata.reference_z))),
        "max_vz": float(np.max(np.abs(trajectory[:,5]))),
        "max_az": float(np.max(np.abs(trajectory[:,8]))),
        "max_jz": float(np.max(np.abs(trajectory[:,11]))),
        "dense_trajectory_exact_safe": dense_safe,
    }
    (args.output_dir / "verification.json").write_text(json.dumps(verification, indent=2))
    result = {
        "scene": args.scene, "device": str(device), "scale": scale,
        "physical": {"height_m": args.robot_height_meters, "footprint_radius_m": args.footprint_radius_meters, "sphere_radius_m": sphere_radius_m, "clearance_m": args.ground_clearance_meters, "corridor_margin_m": args.corridor_margin_meters},
        "scene_units": {"height": height, "footprint_radius": footprint, "sphere_radius": sphere_radius, "clearance": clearance, "corridor_margin": corridor_margin, "reference_z": grid.metadata.reference_z},
        "gaussians_total": int(gsplat.means.shape[0]), "gaussians_filtered": int(mask.sum()),
        "raw_seed_points": len(raw_seed), "simplified_seed_points": len(simplified),
        "num_polytopes": output["num_polytopes"], "timings": timings,
        "qp_status": output["qp_status"], "failure_reason": output["failure_reason"],
        "verification": verification,
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2))

    occ = grid.occupied.detach().cpu().numpy(); raw_occ = grid.raw_occupied.detach().cpu().numpy()
    extent = [float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1])]
    fig, axes = plt.subplots(1,2,figsize=(14,6),sharex=True,sharey=True)
    axes[0].imshow(raw_occ.T,origin='lower',extent=extent,cmap='Greys',aspect='equal'); axes[0].plot(raw_seed[:,0],raw_seed[:,1],label='raw seed')
    axes[1].imshow(occ.T,origin='lower',extent=extent,cmap='Greys',aspect='equal'); axes[1].plot(raw_seed[:,0],raw_seed[:,1],alpha=.4,label='raw'); axes[1].plot(simplified[:,0],simplified[:,1],'o-',label='simplified')
    for ax in axes: ax.legend(); ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.tight_layout(); fig.savefig(args.output_dir/'ground_grid.png',dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,7)); ax.imshow(occ.T,origin='lower',extent=extent,cmap='Greys',alpha=.45,aspect='equal')
    for matrix in output["polytopes"]:
        vertices = slice_vertices(matrix, grid.metadata.reference_z)
        if vertices is not None: ax.fill(vertices[:,0],vertices[:,1],alpha=.18,color='tab:green')
    ax.plot(simplified[:,0],simplified[:,1],'o--',label='seed'); ax.plot(dense[:,:,0].ravel(),dense[:,:,1].ravel(),label='Bezier'); ax.legend(); ax.set_aspect('equal'); fig.tight_layout(); fig.savefig(args.output_dir/'corridor_and_trajectory.png',dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(11,7)); values=[dense[:,:,2].ravel(),trajectory[:,5],trajectory[:,8],trajectory[:,11]]; names=['z','vz','az','jz']
    for ax,val,name in zip(axes.ravel(),values,names): ax.plot(val); ax.set_title(name); ax.grid(True,alpha=.3)
    fig.tight_layout(); fig.savefig(args.output_dir/'fixed_z_check.png',dpi=180); plt.close(fig)
    print(json.dumps(result, indent=2))
    if not all([verification["seed_segments_exact_safe"], verification["qp_feasible"], verification["max_control_point_violation"] <= 1e-5, verification["max_z_error"] <= 1e-6, verification["dense_trajectory_exact_safe"]]):
        raise SystemExit("milestone 2 verification failed")


if __name__ == "__main__": main()
