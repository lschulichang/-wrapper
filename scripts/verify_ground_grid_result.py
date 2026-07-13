#!/usr/bin/env python3
"""Verify a smoke_ground_grid result without rebuilding the GSplat grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Result JSON from smoke_ground_grid.py")
    args = parser.parse_args()

    data = json.loads(args.result.read_text(encoding="utf-8"))
    occupancy = np.load(data["occupancy_file"])
    path = np.load(data["path_file"])
    lower = np.asarray(data["voxel_bounds_used"]["lower_bound"][:2])
    upper = np.asarray(data["voxel_bounds_used"]["upper_bound"][:2])
    cell_sizes = (upper - lower) / np.asarray(occupancy.shape)
    lower_center = lower + cell_sizes / 2
    if path.ndim != 2 or path.shape[1] != 2:
        raise SystemExit(f"expected a strictly 2D path with shape (N, 2), got {path.shape}")
    indices = np.rint((path - lower_center) / cell_sizes).astype(np.int64)

    collision_points = occupancy[indices[:, 0], indices[:, 1]]
    index_steps = np.abs(np.diff(indices, axis=0))
    path_length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    report = {
        "occupancy_shape": list(occupancy.shape),
        "path_shape": list(path.shape),
        "collision_points": int(collision_points.sum()),
        "four_connected": bool(np.all(index_steps.sum(axis=1) == 1)),
        "strictly_2d": True,
        "path_length_scene_units": float(path_length),
        "start_index": indices[0].tolist(),
        "goal_index": indices[-1].tolist(),
    }
    scale = data.get("unit_conversion", {}).get("scene_units_per_meter")
    if scale:
        report["path_length_meters"] = float(path_length / scale)
    print(json.dumps(report, indent=2))

    if report["collision_points"] != 0:
        raise SystemExit("path intersects inflated occupancy")
    if not report["four_connected"]:
        raise SystemExit("path contains a non-4-connected step")


if __name__ == "__main__":
    main()
