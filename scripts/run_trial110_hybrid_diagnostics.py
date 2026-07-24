#!/usr/bin/env python3
"""Repeat one Hybrid A* trial and diagnose wall-clock long tails."""

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
    PlanarGaussianSet,
)
from smoke_splatplan import SCENE_PRESETS  # noqa: E402
from splat.splat_utils import GSplatLoader  # noqa: E402


COUNT_KEYS = (
    "expanded_nodes",
    "generated_nodes",
    "collision_check_count",
    "motion_collision_check_count",
    "open_list_max_length",
    "analytic_expansion_attempt_count",
    "analytic_expansion_success_count",
    "dubins_connection_attempt_count",
    "dubins_connection_success_count",
    "dubins_solver_call_count",
    "dubins_path_candidate_count",
    "free_heading_candidate_count",
)
TIME_KEYS = (
    "hybrid_astar_wall_time_s",
    "hybrid_astar_cpu_time_s",
    "goal_cost_map_time_s",
    "collision_check_time_s",
    "motion_collision_check_time_s",
    "heuristic_time_s",
    "analytic_expansion_time_s",
    "dubins_connection_time_s",
    "dubins_solver_time_s",
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def save_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_jsonl(path: Path, value) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def numeric_summary(values) -> dict | None:
    array = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and np.isfinite(float(value))
        ],
        dtype=np.float64,
    )
    if len(array) == 0:
        return None
    mean = float(np.mean(array))
    return {
        "count": int(len(array)),
        "mean": mean,
        "std": float(np.std(array)),
        "coefficient_of_variation": (
            float(np.std(array) / mean)
            if abs(mean) > 1e-15
            else 0.0
        ),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def select_trial(rows: list[dict], trial_id: str) -> dict:
    matches = [row for row in rows if row["trial_id"] == trial_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one manifest row for {trial_id}"
        )
    trial = dict(matches[0])
    if trial.get("goal_heading_mode") != "free":
        raise ValueError("trial must use free goal heading")
    return trial


def save_visualizations(
    records: list[dict], output_dir: Path
) -> list[str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    repeats = np.asarray(
        [row["repeat_index"] for row in records]
    )

    figure, axes = plt.subplots(
        2, 1, figsize=(10.0, 8.0), sharex=True
    )
    width = 0.38
    wall = np.asarray(
        [row["hybrid_astar_wall_time_s"] for row in records]
    )
    cpu = np.asarray(
        [row["hybrid_astar_cpu_time_s"] for row in records]
    )
    axes[0].bar(
        repeats - width / 2,
        wall,
        width=width,
        color="#4C78A8",
        label="wall time",
    )
    axes[0].bar(
        repeats + width / 2,
        cpu,
        width=width,
        color="#F58518",
        label="process CPU time",
    )
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Repeated Hybrid A* runtime")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend()
    expanded = np.asarray(
        [row["expanded_nodes"] for row in records]
    )
    generated_nodes = np.asarray(
        [row["generated_nodes"] for row in records]
    )
    axes[1].plot(
        repeats,
        expanded,
        "o-",
        color="#54A24B",
        label="expanded nodes",
    )
    axes[1].plot(
        repeats,
        generated_nodes,
        "s-",
        color="#E45756",
        label="generated nodes",
    )
    axes[1].set_xlabel("Repeat")
    axes[1].set_ylabel("Node count")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    axes[1].set_xticks(repeats)
    figure.tight_layout()
    path = figure_dir / "runtime_and_nodes.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    generated.append(str(path.relative_to(output_dir)))

    figure, axes = plt.subplots(
        2, 1, figsize=(10.0, 8.0), sharex=True
    )
    axes[0].plot(
        repeats,
        [row["collision_check_count"] for row in records],
        "o-",
        label="state collision checks",
        color="#4C78A8",
    )
    axes[0].plot(
        repeats,
        [row["open_list_max_length"] for row in records],
        "s-",
        label="maximum open-list length",
        color="#F58518",
    )
    axes[0].set_ylabel("Count")
    axes[0].set_title("Collision and open-list workload")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        repeats,
        [
            row["analytic_expansion_attempt_count"]
            for row in records
        ],
        "o-",
        label="analytic triggers",
        color="#54A24B",
    )
    axes[1].plot(
        repeats,
        [
            row["dubins_connection_attempt_count"]
            for row in records
        ],
        "s-",
        label="collision-checked Dubins attempts",
        color="#E45756",
    )
    axes[1].plot(
        repeats,
        [
            row["dubins_connection_success_count"]
            for row in records
        ],
        "^-",
        label="Dubins successes",
        color="#B279A2",
    )
    axes[1].set_xlabel("Repeat")
    axes[1].set_ylabel("Count")
    axes[1].set_xticks(repeats)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    path = figure_dir / "collision_open_dubins_counts.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    generated.append(str(path.relative_to(output_dir)))

    figure, axis = plt.subplots(figsize=(10.0, 5.8))
    timing_series = (
        ("goal cost map", "goal_cost_map_time_s", "#4C78A8"),
        (
            "motion sampling + collision",
            "motion_collision_check_time_s",
            "#F58518",
        ),
        (
            "analytic expansion",
            "analytic_expansion_time_s",
            "#E45756",
        ),
        ("Dubins solver", "dubins_solver_time_s", "#B279A2"),
    )
    for label, key, color in timing_series:
        axis.plot(
            repeats,
            [row[key] for row in records],
            "o-",
            label=label,
            color=color,
        )
    axis.set_xlabel("Repeat")
    axis.set_ylabel("Cumulative time within plan (s)")
    axis.set_title(
        "Internal timing (categories may overlap)"
    )
    axis.set_xticks(repeats)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = figure_dir / "internal_timing.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    generated.append(str(path.relative_to(output_dir)))

    figure, axis = plt.subplots(figsize=(9.0, 7.0))
    for row in records:
        path_array = np.load(row["path_file"])
        axis.plot(
            path_array[:, 0],
            path_array[:, 1],
            linewidth=1.0,
            alpha=0.45,
        )
    axis.scatter(
        records[0]["start_scene"][0],
        records[0]["start_scene"][1],
        s=65,
        color="#54A24B",
        label="start",
        zorder=5,
    )
    axis.scatter(
        records[0]["goal_scene"][0],
        records[0]["goal_scene"][1],
        s=90,
        marker="*",
        color="#B279A2",
        label="goal",
        zorder=5,
    )
    axis.set_xlabel("Scene x")
    axis.set_ylabel("Scene y")
    axis.set_title("Hybrid paths from 10 identical repeats")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = figure_dir / "repeated_path_overlay.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    generated.append(str(path.relative_to(output_dir)))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--selected-goals-json", required=True, type=Path
    )
    parser.add_argument("--trial-id", default="trial_110")
    parser.add_argument("--repeat-count", type=int, default=10)
    parser.add_argument(
        "--scene",
        default="old_union",
        choices=sorted(SCENE_PRESETS),
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
    parser.add_argument(
        "--min-turning-radius-meters", type=float, default=0.20
    )
    parser.add_argument(
        "--hybrid-heading-bins", type=int, default=72
    )
    parser.add_argument(
        "--hybrid-max-expansions", type=int, default=200_000
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.repeat_count <= 0:
        raise ValueError("repeat_count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        args.selected_goals_json.read_text(encoding="utf-8")
    )
    trial = select_trial(manifest, args.trial_id)
    save_json(args.output_dir / "selected_trial.json", trial)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    scale = float(
        json.loads(
            (
                args.config.resolve().parent
                / "dataparser_transforms.json"
            ).read_text(encoding="utf-8")
        )["scale"]
    )
    height = args.robot_height_meters * scale
    radius = args.footprint_radius_meters * scale
    clearance = args.ground_clearance_meters * scale
    preset = SCENE_PRESETS[args.scene]
    lower = torch.tensor(preset["lower_bound"], device=device)
    upper = torch.tensor(preset["upper_bound"], device=device)
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
    planner = HybridAStarPlanner(
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

    start_scene = np.asarray(
        trial["start_scene"], dtype=np.float64
    )
    goal_scene = np.asarray(
        trial["goal_scene"], dtype=np.float64
    )
    direct_distance_m = float(
        np.linalg.norm(goal_scene - start_scene) / scale
    )
    path_dir = args.output_dir / "hybrid_paths"
    path_dir.mkdir(parents=True, exist_ok=True)
    record_path = args.output_dir / "records.jsonl"
    records = []

    for repeat_index in range(1, args.repeat_count + 1):
        print(
            f"[{repeat_index:02d}/{args.repeat_count:02d}] "
            f"{args.trial_id}",
            flush=True,
        )
        wall_begin = time.perf_counter()
        cpu_begin = time.process_time()
        path = planner.plan(
            [
                *trial["start_scene"],
                trial["start_yaw_rad"],
            ],
            trial["goal_scene"],
        )
        cpu_time = time.process_time() - cpu_begin
        wall_time = time.perf_counter() - wall_begin
        diagnostics = path.diagnostics
        path_file = (
            path_dir / f"repeat_{repeat_index:02d}.npy"
        )
        np.save(path_file, path.poses_scene)
        path_length_m = float(path.arc_lengths_m[-1])
        record = {
            "repeat_index": repeat_index,
            "trial_id": args.trial_id,
            "search_success": True,
            "start_scene": trial["start_scene"],
            "goal_scene": trial["goal_scene"],
            "direct_distance_m": direct_distance_m,
            "hybrid_astar_wall_time_s": float(wall_time),
            "hybrid_astar_cpu_time_s": float(cpu_time),
            "wall_minus_cpu_time_s": float(
                wall_time - cpu_time
            ),
            "wall_to_cpu_ratio": float(
                wall_time / max(cpu_time, 1e-12)
            ),
            "path_length_m": path_length_m,
            "detour_factor": float(
                path_length_m / direct_distance_m
            ),
            "path_file": str(path_file),
            **diagnostics,
        }
        records.append(record)
        append_jsonl(record_path, record)
        print(
            json.dumps(
                {
                    "wall_s": wall_time,
                    "cpu_s": cpu_time,
                    "expanded": record["expanded_nodes"],
                    "generated": record["generated_nodes"],
                    "collision_checks": record[
                        "collision_check_count"
                    ],
                    "open_max": record[
                        "open_list_max_length"
                    ],
                    "analytic_attempts": record[
                        "analytic_expansion_attempt_count"
                    ],
                    "dubins_attempts": record[
                        "dubins_connection_attempt_count"
                    ],
                    "dubins_successes": record[
                        "dubins_connection_success_count"
                    ],
                    "path_length_m": path_length_m,
                    "detour_factor": record[
                        "detour_factor"
                    ],
                }
            ),
            flush=True,
        )

    visualizations = save_visualizations(
        records, args.output_dir
    )
    summaries = {
        key: numeric_summary([row[key] for row in records])
        for key in (
            *TIME_KEYS,
            *COUNT_KEYS,
            "path_length_m",
            "direct_distance_m",
            "detour_factor",
            "wall_minus_cpu_time_s",
            "wall_to_cpu_ratio",
        )
    }
    summary = {
        "scene": args.scene,
        "device": str(device),
        "scene_scale": scale,
        "trial_id": args.trial_id,
        "repeat_count": args.repeat_count,
        "success_count": len(records),
        "metrics": summaries,
        "all_counts_identical": {
            key: len({row[key] for row in records}) == 1
            for key in COUNT_KEYS
        },
        "all_paths_identical": all(
            np.array_equal(
                np.load(records[0]["path_file"]),
                np.load(row["path_file"]),
            )
            for row in records[1:]
        ),
        "preprocessing": preprocessing,
        "visualizations": visualizations,
        "protocol": {
            "selected_goals_json": str(
                args.selected_goals_json
            ),
            "fixed_start_pose": True,
            "goal_heading_mode": "free",
            "stops_after": "Hybrid_A_star",
            "safety_corridor_run": False,
            "trajectory_optimization_run": False,
            "same_planner_instance_reused": True,
            "diagnostics_reset_before_each_plan": True,
            "timer_note": (
                "internal timing categories overlap; "
                "wall time includes diagnostic instrumentation"
            ),
            "heading_bins": args.hybrid_heading_bins,
            "min_turning_radius_m": (
                args.min_turning_radius_meters
            ),
            "max_expansions": args.hybrid_max_expansions,
        },
    }
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
