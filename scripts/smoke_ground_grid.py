#!/usr/bin/env python3
"""Build a projected ground grid and plan one path in an existing scene."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "splatnav-official"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(UPSTREAM_ROOT))

from ground_nav.ground_grid import GroundGrid  # noqa: E402
from initialization.grid_utils import GSplatVoxel  # noqa: E402
from run_output_utils import resolve_experiment_output  # noqa: E402
from smoke_splatplan import SCENE_PRESETS, to_serializable  # noqa: E402
from splat.splat_utils import GSplatLoader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument(
        "--z-floor-scene", "--z-floor", dest="z_floor_scene", required=True, type=float,
        help="Floor Z in normalized scene coordinates; --z-floor is a deprecated alias",
    )
    parser.add_argument("--robot-height-meters", required=True, type=float)
    parser.add_argument("--footprint-radius-meters", required=True, type=float)
    parser.add_argument("--ground-clearance-meters", type=float, default=0.0)
    parser.add_argument("--start", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--output", type=Path, help="Optional JSON path; existing prefixes are never overwritten")
    args = parser.parse_args()

    output_path, auto_run_dir, redirect_reason = resolve_experiment_output(
        args.output,
        PROJECT_ROOT,
        topic=f"ground_grid_milestone1_{args.scene}",
    )
    if redirect_reason is not None:
        print(f"Output protection: {redirect_reason}")
        print(f"Allocated new run directory: {auto_run_dir}")

    import torch

    if args.scene not in SCENE_PRESETS:
        raise ValueError(f"unsupported scene {args.scene!r}; choose from {sorted(SCENE_PRESETS)}")

    transform_path = args.config.resolve().parent / "dataparser_transforms.json"
    transform_data = json.loads(transform_path.read_text(encoding="utf-8"))
    scene_units_per_meter = float(transform_data["scale"])
    robot_height = args.robot_height_meters * scene_units_per_meter
    footprint_radius = args.footprint_radius_meters * scene_units_per_meter
    ground_clearance = args.ground_clearance_meters * scene_units_per_meter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset = SCENE_PRESETS[args.scene]
    gsplat = GSplatLoader(args.config, device)
    lower_bound = torch.tensor(preset["lower_bound"], device=device)
    upper_bound = torch.tensor(preset["upper_bound"], device=device)
    # The original presets were designed for drone paths and may not cover the
    # complete cylinder height. Expand Z so the requested body band is never
    # silently truncated; preserve the preset XY planning region.
    lower_bound[2] = min(float(lower_bound[2].item()), args.z_floor_scene)
    upper_bound[2] = max(float(upper_bound[2].item()), args.z_floor_scene + robot_height)
    raw_voxel = GSplatVoxel(
        gsplat,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        resolution=preset["resolution"],
        radius=0.0,
        device=device,
    )
    grid = GroundGrid(
        raw_voxel,
        z_floor_scene=args.z_floor_scene,
        robot_height=robot_height,
        footprint_radius=footprint_radius,
        ground_clearance=ground_clearance,
        project_occupied_endpoints=False,
    )

    preset_center = np.asarray(preset["mean_config"], dtype=np.float32)[:2]
    radius = float(preset["radius_config"])
    start = np.asarray(args.start if args.start else preset_center + [radius, 0.0], dtype=np.float32)
    goal = np.asarray(args.goal if args.goal else preset_center - [radius, 0.0], dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_occupancy_path = output_path.with_suffix(".raw_occupancy.npy")
    occupancy_path = output_path.with_suffix(".occupancy.npy")
    path_file = output_path.with_suffix(".path.npy")
    figure_file = output_path.with_suffix(".png")
    np.save(raw_occupancy_path, grid.raw_occupied.detach().cpu().numpy())
    np.save(occupancy_path, grid.occupied.detach().cpu().numpy())

    raw_occupancy = grid.raw_occupied.detach().cpu().numpy()
    occupancy = grid.occupied.detach().cpu().numpy()
    free = ~occupancy
    component_labels, component_count = ndimage.label(
        free,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
    )
    component_sizes = np.bincount(component_labels.ravel())[1:]

    start_was_occupied = grid.is_occupied(start)
    goal_was_occupied = grid.is_occupied(goal)
    start_used = start
    goal_used = goal
    start_index = grid.world_to_grid(start_used).detach().cpu().numpy()
    goal_index = grid.world_to_grid(goal_used).detach().cpu().numpy()
    start_component = int(component_labels[start_index[0], start_index[1]])
    goal_component = int(component_labels[goal_index[0], goal_index[1]])

    path = None
    path_error = None
    try:
        path = grid.create_path(start, goal)
        np.save(path_file, path)
    except (RuntimeError, ValueError) as exc:
        path_error = str(exc)
    extent = [
        float(grid.xy_centers[0, 0, 0].item() - grid.cell_sizes[0].item() / 2),
        float(grid.xy_centers[-1, 0, 0].item() + grid.cell_sizes[0].item() / 2),
        float(grid.xy_centers[0, 0, 1].item() - grid.cell_sizes[1].item() / 2),
        float(grid.xy_centers[0, -1, 1].item() + grid.cell_sizes[1].item() / 2),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for ax, image, title in zip(
        axes,
        (raw_occupancy, occupancy),
        ("height-band projection (before XY dilation)", "disk-inflated grid and 2D path"),
    ):
        ax.imshow(
            image.T,
            origin="lower",
            extent=extent,
            cmap="Greys",
            interpolation="nearest",
            alpha=0.8,
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    if path is not None:
        axes[1].plot(path[:, 0], path[:, 1], color="tab:blue", linewidth=2, label="ground path")
    axes[1].scatter([start_used[0]], [start_used[1]], color="tab:green", marker="o", label="start used")
    axes[1].scatter([goal_used[0]], [goal_used[1]], color="tab:red", marker="x", label="goal used")
    axes[1].legend(loc="best")
    fig.suptitle(f"{args.scene}: cylinder footprint ground planning")
    fig.tight_layout()
    fig.savefig(figure_file, dpi=180)
    plt.close(fig)

    result = {
        "scene": args.scene,
        "device": str(device),
        "unit_conversion": {
            "scene_units_per_meter": scene_units_per_meter,
            "z_floor_scene_units": args.z_floor_scene,
            "robot_height_meters": args.robot_height_meters,
            "robot_height_scene_units": robot_height,
            "footprint_radius_meters": args.footprint_radius_meters,
            "footprint_radius_scene_units": footprint_radius,
            "ground_clearance_meters": args.ground_clearance_meters,
            "ground_clearance_scene_units": ground_clearance,
        },
        "start_requested": start.tolist(),
        "goal_requested": goal.tolist(),
        "start_used": start_used.tolist(),
        "goal_used": goal_used.tolist(),
        "start_was_occupied": bool(start_was_occupied),
        "goal_was_occupied": bool(goal_was_occupied),
        "start_component": start_component,
        "goal_component": goal_component,
        "free_component_count": int(component_count),
        "largest_free_component_cells": int(component_sizes.max()) if component_sizes.size else 0,
        "path_feasible": path is not None,
        "path_error": path_error,
        "path_points": 0 if path is None else int(len(path)),
        "path_start": None if path is None else path[0].tolist(),
        "path_goal": None if path is None else path[-1].tolist(),
        "path_is_strictly_2d": False if path is None else bool(path.ndim == 2 and path.shape[1] == 2),
        "ground_grid": grid.summary(),
        "raw_occupancy_file": str(raw_occupancy_path),
        "occupancy_file": str(occupancy_path),
        "path_file": str(path_file),
        "comparison_figure": str(figure_file),
        "scene_preset": to_serializable(preset),
        "voxel_bounds_used": {
            "lower_bound": lower_bound.detach().cpu().tolist(),
            "upper_bound": upper_bound.detach().cpu().tolist(),
        },
    }
    result_text = json.dumps(result, indent=2)
    output_path.write_text(result_text, encoding="utf-8")
    print(result_text)

    if auto_run_dir is not None:
        verifier = PROJECT_ROOT / "scripts" / "verify_ground_grid_result.py"
        verification = subprocess.run(
            [sys.executable, str(verifier), str(output_path)],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        verification_path = auto_run_dir / "assets" / "verification.json"
        verification_path.write_text(verification.stdout, encoding="utf-8")
        with (auto_run_dir / "stdout.log").open("a", encoding="utf-8") as stream:
            stream.write(result_text + "\n" + verification.stdout + verification.stderr)
        with (auto_run_dir / "notes.md").open("a", encoding="utf-8") as stream:
            stream.write(f"\n- Output protection: {redirect_reason}.\n")
            stream.write(f"- Result JSON: {output_path}.\n")
            stream.write(f"- Verification exit code: {verification.returncode}.\n")
        print(verification.stdout, end="")
        print(f"run_dir={auto_run_dir}")
        print(f"result_json={output_path}")
        if verification.returncode != 0:
            raise RuntimeError(f"result verification failed: {verification.stderr}")


if __name__ == "__main__":
    main()
