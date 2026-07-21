#!/usr/bin/env python3
"""Evaluate raw Hybrid A* paths without a safety corridor on fixed goals."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "splatnav-official"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(UPSTREAM))

from ground_nav.corridor_2d import (  # noqa: E402
    PlanarCollisionSet,
    compute_stopping_distance,
)
from ground_nav.ground_grid import GroundGrid  # noqa: E402
from ground_nav.hybrid_astar import (  # noqa: E402
    HybridAStarConfig,
    HybridAStarPlanner,
)
from ground_nav.planar_gaussians import PlanarGaussianSet  # noqa: E402
from run_hybrid_compression_ablation import (  # noqa: E402
    append_jsonl,
    numeric_summary,
    save_json,
    sync,
)
from smoke_ground_corridor_2d import configure_map_axis  # noqa: E402
from smoke_splatplan import SCENE_PRESETS  # noqa: E402
from splat.splat_utils import GSplatLoader  # noqa: E402


LAYERS = ("near", "medium", "far")


def select_manifest_rows(rows: list[dict], sample_count: int) -> list[dict]:
    if sample_count <= 0 or sample_count % len(LAYERS):
        raise ValueError("sample_count must be a positive multiple of three")
    selected = []
    per_layer = sample_count // len(LAYERS)
    for layer in LAYERS:
        layer_rows = [row for row in rows if row["distance_layer"] == layer]
        if len(layer_rows) < per_layer:
            raise ValueError(f"manifest has only {len(layer_rows)} {layer} rows")
        indices = np.linspace(0, len(layer_rows) - 1, per_layer).round().astype(int)
        selected.extend(dict(layer_rows[index]) for index in indices)
    if len({row["trial_id"] for row in selected}) != len(selected):
        raise ValueError("selected trial identifiers are not unique")
    if any(row.get("goal_heading_mode") != "free" for row in selected):
        raise ValueError("manifest must use free goal heading")
    return selected


def grid_sample_statistics(points: np.ndarray, grid) -> dict:
    points = np.asarray(points, dtype=np.float64)
    lower_center = np.asarray(grid.lower_center.detach().cpu(), dtype=np.float64)
    cell_sizes = np.asarray(grid.cell_sizes.detach().cpu(), dtype=np.float64)
    shape = np.asarray(grid.shape, dtype=np.int64)
    lower_edge = lower_center - 0.5 * cell_sizes
    upper_edge = lower_center + (shape - 0.5) * cell_sizes
    inside = np.all(
        (points >= lower_edge[None]) & (points < upper_edge[None]), axis=1
    )
    indices = np.rint((points - lower_center[None]) / cell_sizes[None]).astype(int)
    clipped = np.clip(indices, 0, shape - 1)
    occupied = np.asarray(grid.occupied.detach().cpu(), dtype=bool)
    safe = inside & ~occupied[clipped[:, 0], clipped[:, 1]]
    unsafe_indices = np.flatnonzero(~safe)
    return {
        "grid_sample_count": int(len(points)),
        "grid_unsafe_sample_count": int(len(unsafe_indices)),
        "grid_unsafe_sample_fraction": float(len(unsafe_indices) / len(points)),
        "grid_collision_free": bool(len(unsafe_indices) == 0),
        "grid_unsafe_sample_indices": unsafe_indices.tolist(),
    }


def continuous_segment_statistics(points: np.ndarray, collision_set) -> dict:
    points = np.asarray(points, dtype=np.float64)
    safe = []
    with torch.no_grad():
        for first, second in zip(points[:-1], points[1:]):
            safe.append(collision_set.segment_is_safe(np.stack([first, second])))
    safe = np.asarray(safe, dtype=bool)
    unsafe_indices = np.flatnonzero(~safe)
    return {
        "continuous_segment_count": int(len(safe)),
        "continuous_unsafe_segment_count": int(len(unsafe_indices)),
        "continuous_unsafe_segment_fraction": (
            float(len(unsafe_indices) / len(safe)) if len(safe) else 0.0
        ),
        "continuous_collision_free": bool(len(unsafe_indices) == 0),
        "continuous_unsafe_segment_indices": unsafe_indices.tolist(),
    }


def evaluate_path(hybrid_path, grid, collision_set, scale: float) -> dict:
    poses = np.asarray(hybrid_path.poses_scene, dtype=np.float64)
    xy = poses[:, :2]
    lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    curvatures = np.asarray(
        hybrid_path.primitive_curvatures_1pm, dtype=np.float64
    )
    record = {
        "dense_pose_count": int(len(poses)),
        "path_length_m": float(lengths.sum() / scale),
        "primitive_count": int(len(curvatures)),
        "max_abs_primitive_curvature_1pm": (
            float(np.max(np.abs(curvatures))) if len(curvatures) else 0.0
        ),
        "curvature_change_count": (
            int(np.count_nonzero(np.abs(np.diff(curvatures)) > 1e-9))
            if len(curvatures) > 1
            else 0
        ),
    }
    record.update(grid_sample_statistics(xy, grid))
    begin = time.perf_counter()
    record.update(continuous_segment_statistics(xy, collision_set))
    record["continuous_collision_check_s"] = time.perf_counter() - begin
    record["direct_path_safe"] = bool(
        record["grid_collision_free"] and record["continuous_collision_free"]
    )
    return record


def build_summary(records: list[dict], sample_count: int) -> dict:
    planned = [row for row in records if row["search_success"]]
    metric_rows = [{**row, "success": True} for row in planned]
    metrics = (
        "time_hybrid_astar_s",
        "continuous_collision_check_s",
        "path_length_m",
        "dense_pose_count",
        "primitive_count",
        "expanded_nodes",
        "generated_nodes",
        "max_abs_primitive_curvature_1pm",
        "curvature_change_count",
        "grid_unsafe_sample_fraction",
        "continuous_unsafe_segment_fraction",
    )
    by_layer = {}
    for layer in LAYERS:
        rows = [row for row in records if row["distance_layer"] == layer]
        layer_planned = [row for row in rows if row["search_success"]]
        by_layer[layer] = {
            "trial_count": len(rows),
            "search_success_count": sum(row["search_success"] for row in rows),
            "grid_collision_free_count": sum(
                row.get("grid_collision_free", False) for row in layer_planned
            ),
            "continuous_collision_free_count": sum(
                row.get("continuous_collision_free", False) for row in layer_planned
            ),
            "continuous_collision_trial_count": sum(
                not row.get("continuous_collision_free", False)
                for row in layer_planned
            ),
        }
    return {
        "sample_count": sample_count,
        "search_success_count": int(len(planned)),
        "search_failure_count": int(sample_count - len(planned)),
        "search_success_rate": float(len(planned) / sample_count),
        "grid_collision_free_count": int(
            sum(row["grid_collision_free"] for row in planned)
        ),
        "grid_collision_free_rate_over_all_trials": float(
            sum(row["grid_collision_free"] for row in planned) / sample_count
        ),
        "continuous_collision_free_count": int(
            sum(row["continuous_collision_free"] for row in planned)
        ),
        "continuous_collision_trial_count": int(
            sum(not row["continuous_collision_free"] for row in planned)
        ),
        "continuous_collision_free_rate_over_all_trials": float(
            sum(row["continuous_collision_free"] for row in planned) / sample_count
        ),
        "direct_path_safe_count": int(
            sum(row["direct_path_safe"] for row in planned)
        ),
        "direct_path_safe_rate_over_all_trials": float(
            sum(row["direct_path_safe"] for row in planned) / sample_count
        ),
        "metrics": {
            key: numeric_summary(metric_rows, key) for key in metrics
        },
        "by_distance_layer": by_layer,
        "failure_reasons": dict(
            Counter(
                row["failure_reason"]
                for row in records
                if not row["search_success"]
            )
        ),
    }


def plot_examples(output_dir, selected, artifacts, grid, extent):
    chosen = []
    for layer in LAYERS:
        rows = [row for row in selected if row["distance_layer"] == layer]
        collision_rows = [
            row
            for row in rows
            if row["trial_id"] in artifacts
            and artifacts[row["trial_id"]]["unsafe_segment_indices"]
        ]
        chosen.append((collision_rows or rows)[len(collision_rows or rows) // 2])
    occupied = np.asarray(grid.occupied.detach().cpu(), dtype=float).T
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, trial in zip(axes, chosen):
        axis.imshow(
            occupied,
            origin="lower",
            cmap="binary",
            extent=extent,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        artifact = artifacts.get(trial["trial_id"])
        if artifact is not None:
            path = artifact["path"]
            axis.plot(path[:, 0], path[:, 1], color="tab:blue", linewidth=1.2)
            for index in artifact["unsafe_segment_indices"]:
                axis.plot(
                    path[index : index + 2, 0],
                    path[index : index + 2, 1],
                    color="tab:red",
                    linewidth=3.0,
                )
        configure_map_axis(
            axis,
            extent,
            f"{trial['distance_layer']} / {trial['trial_id']}",
        )
    fig.tight_layout()
    fig.savefig(output_dir / "representative_raw_hybrid_paths.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--selected-goals-json", required=True, type=Path)
    parser.add_argument("--scene", default="old_union", choices=sorted(SCENE_PRESETS))
    parser.add_argument("--sample-count", type=int, default=120)
    parser.add_argument("--z-floor-scene", type=float, default=-0.15)
    parser.add_argument("--robot-height-meters", type=float, default=0.10)
    parser.add_argument("--footprint-radius-meters", type=float, default=0.15)
    parser.add_argument("--ground-clearance-meters", type=float, default=0.02)
    parser.add_argument("--max-speed-mps", type=float, default=0.20)
    parser.add_argument("--max-brake-decel-mps2", type=float, default=0.30)
    parser.add_argument("--min-turning-radius-meters", type=float, default=0.20)
    parser.add_argument("--hybrid-heading-bins", type=int, default=72)
    parser.add_argument("--hybrid-max-expansions", type=int, default=200000)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.selected_goals_json.read_text(encoding="utf-8"))
    selected = select_manifest_rows(manifest, args.sample_count)
    save_json(args.output_dir / "selected_goals.json", selected)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale = float(
        json.loads(
            (args.config.resolve().parent / "dataparser_transforms.json").read_text()
        )["scale"]
    )
    height = args.robot_height_meters * scale
    radius = args.footprint_radius_meters * scale
    clearance = args.ground_clearance_meters * scale
    stopping_distance_m = compute_stopping_distance(
        args.max_speed_mps, args.max_brake_decel_mps2
    )
    preset = SCENE_PRESETS[args.scene]
    lower = torch.tensor(preset["lower_bound"], device=device)
    upper = torch.tensor(preset["upper_bound"], device=device)
    preprocessing = {}

    sync(device)
    begin = time.perf_counter()
    gsplat = GSplatLoader(args.config, device)
    sync(device)
    preprocessing["gsplat_load_s"] = time.perf_counter() - begin
    begin = time.perf_counter()
    ellipses = PlanarGaussianSet.from_gsplat(
        gsplat,
        args.z_floor_scene + clearance,
        args.z_floor_scene + height,
        confidence=1.0,
    )
    sync(device)
    preprocessing["height_filter_projection_s"] = time.perf_counter() - begin
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
        project_occupied_endpoints=False,
        rasterization="ellipse_distance",
    )
    sync(device)
    preprocessing["rasterization_ellipse_distance_s"] = time.perf_counter() - begin
    collision_set = PlanarCollisionSet(
        ellipses,
        radius=radius,
        stopping_distance=stopping_distance_m * scale,
        iterations=10,
    )
    planner = HybridAStarPlanner(
        grid,
        scene_scale=scale,
        config=HybridAStarConfig(
            heading_bins=args.hybrid_heading_bins,
            min_turning_radius_m=args.min_turning_radius_meters,
            max_expansions=args.hybrid_max_expansions,
        ),
    )

    records = []
    artifacts = {}
    record_path = args.output_dir / "records.jsonl"
    for index, trial in enumerate(selected):
        print(f"[{index + 1:03d}/{len(selected):03d}] {trial['trial_id']}", flush=True)
        record = {
            "trial_id": trial["trial_id"],
            "candidate_id": trial["candidate_id"],
            "distance_layer": trial["distance_layer"],
            "goal_heading_mode": "free",
            "requested_goal_yaw_rad": None,
            "search_success": False,
            "failure_reason": None,
        }
        try:
            begin = time.perf_counter()
            path = planner.plan(
                [*trial["start_scene"], trial["start_yaw_rad"]],
                trial["goal_scene"],
            )
            record["time_hybrid_astar_s"] = time.perf_counter() - begin
            record.update(path.summary())
            record.update(evaluate_path(path, grid, collision_set, scale))
            record["search_success"] = True
            artifacts[trial["trial_id"]] = {
                "path": path.xy,
                "unsafe_segment_indices": record[
                    "continuous_unsafe_segment_indices"
                ],
            }
        except Exception as error:
            record["failure_reason"] = str(error)
        records.append(record)
        append_jsonl(record_path, record)
        print(
            json.dumps(
                {
                    "search": record["search_success"],
                    "grid_safe": record.get("grid_collision_free"),
                    "continuous_safe": record.get("continuous_collision_free"),
                    "unsafe_segments": record.get(
                        "continuous_unsafe_segment_count"
                    ),
                }
            ),
            flush=True,
        )

    summary = build_summary(records, args.sample_count)
    summary.update(
        {
            "scene": args.scene,
            "device": str(device),
            "scale": scale,
            "preprocessing": preprocessing,
            "protocol": {
                "selected_goals_json": str(args.selected_goals_json),
                "sample_count": args.sample_count,
                "distance_layers": list(LAYERS),
                "trials_per_layer": args.sample_count // len(LAYERS),
                "fixed_start": selected[0]["start_scene"],
                "fixed_start_yaw_rad": selected[0]["start_yaw_rad"],
                "goal_heading_mode": "free",
                "path_postprocessing": "none",
                "collision_models": {
                    "planner": "footprint-inflated ellipse-distance GroundGrid",
                    "postcheck": (
                        "piecewise-linear dense Hybrid path under continuous "
                        "circle-ellipse test"
                    ),
                    "footprint_radius_m": args.footprint_radius_meters,
                    "stopping_distance_m": stopping_distance_m,
                },
            },
        }
    )
    save_json(args.output_dir / "summary.json", summary)
    save_json(
        args.output_dir / "failures.json",
        [row for row in records if not row["search_success"]],
    )
    save_json(
        args.output_dir / "collision_trials.json",
        [
            row
            for row in records
            if row["search_success"] and not row["continuous_collision_free"]
        ],
    )
    upper_center = grid.lower_center + (
        torch.as_tensor(
            grid.shape,
            dtype=grid.cell_sizes.dtype,
            device=grid.cell_sizes.device,
        )
        - 1
    ) * grid.cell_sizes
    extent = [
        float(grid.lower_center[0] - 0.5 * grid.cell_sizes[0]),
        float(upper_center[0] + 0.5 * grid.cell_sizes[0]),
        float(grid.lower_center[1] - 0.5 * grid.cell_sizes[1]),
        float(upper_center[1] + 0.5 * grid.cell_sizes[1]),
    ]
    plot_examples(args.output_dir, selected, artifacts, grid, extent)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
