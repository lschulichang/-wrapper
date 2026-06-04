#!/usr/bin/env python3
"""Launch a viser viewer for selected trial trajectories/corridors from eval_results.json."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib as mpl
import numpy as np
import scipy
import torch
import trimesh
import viser
import viser.transforms as tf

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_upstream_root() -> Path:
    candidates = [
        PROJECT_ROOT / "splatnav-official",
        PROJECT_ROOT / "third_party" / "splatnav-official",
        Path("/data/howard/repos/splatnav-official"),
    ]
    for path in candidates:
        if (path / "splatplan").is_dir() and (path / "splat").is_dir():
            return path
    raise FileNotFoundError(f"Upstream source not found. Tried: {candidates}")


UPSTREAM_ROOT = resolve_upstream_root()
sys.path.insert(0, str(UPSTREAM_ROOT))

from polytopes.polytopes_utils import find_interior  # noqa: E402
from splat.splat_utils import GSplatLoader  # noqa: E402


def create_polytope_trimesh(polytopes: list[tuple[np.ndarray, np.ndarray]], colors=None):
    mesh = None
    for i, (A, b) in enumerate(polytopes):
        pt = find_interior(A, b)
        halfspaces = np.concatenate([A, -b[..., None]], axis=-1)
        hs = scipy.spatial.HalfspaceIntersection(halfspaces, pt, incremental=False, qhull_options=None)
        qhull_pts = hs.intersections
        poly_mesh = trimesh.convex.convex_hull(qhull_pts)

        if colors is not None:
            poly_mesh.visual.face_colors = colors[i]
            poly_mesh.visual.vertex_colors = colors[i]

        if mesh is None:
            mesh = poly_mesh
        else:
            mesh += poly_mesh

    return mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eval_json", type=Path, required=True)
    parser.add_argument("--trial_indices", type=str, default="")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--show_ellipsoid_mesh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.eval_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    records = payload.get("trial_records", [])
    selected = []

    requested = []
    if args.trial_indices.strip():
        requested = [int(x.strip()) for x in args.trial_indices.split(",") if x.strip()]

    if requested:
        idx_set = set(requested)
        for rec in records:
            if rec.get("trial_index") in idx_set and "output_full" in rec:
                selected.append(rec)
    else:
        # default: one success + one failure (if available)
        succ = next((r for r in records if r.get("feasible") and "output_full" in r), None)
        fail = next((r for r in records if (not r.get("feasible")) and "output_full" in r), None)
        if succ is not None:
            selected.append(succ)
        if fail is not None and fail is not succ:
            selected.append(fail)

    if not selected:
        raise RuntimeError("No trial records with output_full found for visualization")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gsplat = GSplatLoader(args.config, device)

    server = viser.ViserServer(host=args.host, port=args.port)
    rotation = tf.SO3.from_x_radians(0.0).wxyz

    splat_kwargs = dict(
        name="/splats",
        centers=gsplat.means.cpu().numpy(),
        covariances=gsplat.covs.cpu().numpy(),
        rgbs=gsplat.colors.cpu().numpy(),
        opacities=gsplat.opacities.cpu().numpy(),
        wxyz=rotation,
    )
    try:
        if hasattr(server.scene, "add_gaussian_splats"):
            server.scene.add_gaussian_splats(**splat_kwargs)
        elif hasattr(server.scene, "_add_gaussian_splats"):
            server.scene._add_gaussian_splats(**splat_kwargs)
        else:
            print("[viser] gaussian splats API unavailable; skip splat rendering")
    except Exception as exc:  # noqa: BLE001
        print(f"[viser] gaussian splats skipped due to error: {exc}")

    cmap = mpl.cm.get_cmap("jet")

    for i, rec in enumerate(selected):
        out = rec["output_full"]
        trial_idx = rec.get("trial_index")
        feasible = rec.get("feasible")

        traj = np.asarray(out["traj"], dtype=np.float32)
        traj_color = (20, 160, 20) if feasible else (200, 40, 40)
        server.scene.add_spline_catmull_rom(
            name=f"/trial_{trial_idx}_traj",
            positions=traj[:, :3],
            line_width=4.0,
            color=traj_color,
            wxyz=rotation,
        )

        polytopes_raw = out.get("polytopes", [])
        polytopes = [
            (np.asarray(poly, dtype=np.float64)[..., :3], np.asarray(poly, dtype=np.float64)[..., 3])
            for poly in polytopes_raw
        ]

        corridor_colors = np.array([cmap(x) for x in np.linspace(0, 1, len(polytopes))])[..., :3]
        corridor_colors = np.concatenate([corridor_colors, 0.15 * np.ones((len(polytopes), 1))], axis=-1)
        corridor_colors = (255 * corridor_colors).astype(np.uint8)

        corridor_mesh = create_polytope_trimesh(polytopes, colors=corridor_colors)
        server.scene.add_mesh_trimesh(
            name=f"/trial_{trial_idx}_corridor_feasible_{int(bool(feasible))}",
            mesh=corridor_mesh,
            wxyz=rotation,
            visible=True,
        )

    print(f"[viser] ready host={args.host} port={args.port}")
    print(f"[viser] selected_trials={[r.get('trial_index') for r in selected]}")

    while True:
        time.sleep(10)


if __name__ == "__main__":
    main()
