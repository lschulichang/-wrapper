#!/usr/bin/env python3
"""Run the full three-layer planner on a reproducible medium-distance batch."""

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


def append_jsonl(path: Path, value) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                value,
                ensure_ascii=False,
                default=json_default,
            )
            + "\n"
        )


def select_medium_rows(rows: list[dict], sample_count: int) -> list[dict]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    medium = [
        dict(row)
        for row in rows
        if row.get("distance_layer") == "medium"
    ]
    if len(medium) < sample_count:
        raise ValueError(
            f"manifest has only {len(medium)} medium rows"
        )
    indices = np.linspace(
        0,
        len(medium) - 1,
        sample_count,
    ).round().astype(int)
    selected = [medium[index] for index in indices]
    if len({row["trial_id"] for row in selected}) != len(selected):
        raise ValueError("selected trial identifiers are not unique")
    if any(
        row.get("goal_heading_mode") != "free"
        for row in selected
    ):
        raise ValueError("manifest must use free goal heading")
    return selected


def numeric_summary(records: list[dict], key: str):
    values = np.asarray(
        [
            float(record[key])
            for record in records
            if record.get(key) is not None
            and np.isfinite(float(record[key]))
        ],
        dtype=np.float64,
    )
    if len(values) == 0:
        return None
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "sum": float(np.sum(values)),
    }


def build_summary(
    records: list[dict],
    preprocessing: dict,
    protocol: dict,
) -> dict:
    stage_counts = Counter(
        record.get("failure_stage") or "success"
        for record in records
    )
    numeric_keys = (
        "total_plan_time_s",
        "hybrid_astar_time_s",
        "ordered_corridor_time_s",
        "curvature_collocation_time_s",
        "hybrid_path_length_m",
        "optimized_path_length_m",
        "detour_factor",
        "primitive_boundary_pose_count",
        "corridor_count",
        "optimizer_node_count",
        "optimizer_iteration_count",
        "max_abs_curvature_1pm",
        "max_abs_curvature_rate_1pm2",
        "minimum_dense_corridor_margin_m",
        "minimum_transition_overlap_margin_m",
        "max_collocation_defect",
    )
    return {
        "sample_count": len(records),
        "search_success_count": int(
            sum(row["search_success"] for row in records)
        ),
        "corridor_success_count": int(
            sum(row["corridor_success"] for row in records)
        ),
        "optimization_success_count": int(
            sum(row["optimization_success"] for row in records)
        ),
        "complete_success_count": int(
            sum(row["complete_success"] for row in records)
        ),
        "fallback_count": int(
            sum(row["fallback_used"] for row in records)
        ),
        "failure_stage_counts": dict(stage_counts),
        "metrics": {
            key: numeric_summary(records, key)
            for key in numeric_keys
        },
        "preprocessing": preprocessing,
        "protocol": protocol,
    }


def save_visualizations(
    records: list[dict],
    output_dir: Path,
    paths: list[tuple[str, np.ndarray, bool]],
    curvature_limit: float,
    rate_limit: float,
) -> list[str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    x = np.arange(1, len(records) + 1)
    labels = [row["trial_id"] for row in records]

    status_names = (
        "search_success",
        "corridor_success",
        "optimization_success",
        "complete_success",
    )
    status = np.asarray(
        [
            [float(row[name]) for name in status_names]
            for row in records
        ]
    )
    fig, axis = plt.subplots(figsize=(10, 10))
    image = axis.imshow(
        status,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    axis.set_xticks(range(len(status_names)))
    axis.set_xticklabels(
        ["Hybrid A*", "corridor", "optimizer", "complete"]
    )
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels)
    axis.set_title("Three-layer planning success by trial")
    fig.colorbar(image, ax=axis, ticks=[0, 1])
    fig.tight_layout()
    path = figure_dir / "success_matrix.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(str(path.relative_to(output_dir)))

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 13),
        constrained_layout=True,
    )
    axes[0].bar(
        x,
        [
            row.get("hybrid_astar_time_s", np.nan)
            for row in records
        ],
        label="Hybrid A*",
        color="#4c78a8",
    )
    axes[0].bar(
        x,
        [
            row.get("ordered_corridor_time_s", np.nan)
            for row in records
        ],
        bottom=[
            row.get("hybrid_astar_time_s", 0.0)
            for row in records
        ],
        label="corridor",
        color="#f58518",
    )
    axes[0].set_title("Search and corridor time")
    axes[0].set_ylabel("Time (s)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        x,
        [
            row.get("curvature_collocation_time_s", np.nan)
            for row in records
        ],
        color="#54a24b",
    )
    axes[1].set_title("Hermite-Simpson solve time")
    axes[1].set_ylabel("Time (s)")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(
        x - 0.2,
        [
            row.get("primitive_boundary_pose_count", np.nan)
            for row in records
        ],
        width=0.4,
        label="primitive boundaries",
        color="#b279a2",
    )
    axes[2].bar(
        x + 0.2,
        [
            row.get("corridor_count", np.nan)
            for row in records
        ],
        width=0.4,
        label="GS corridors",
        color="#e45756",
    )
    axes[2].set_title("Mainline problem size")
    axes[2].set_ylabel("Count")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=60, ha="right")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.25)
    path = figure_dir / "timing_and_problem_size.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(str(path.relative_to(output_dir)))

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 11),
        constrained_layout=True,
    )
    axes[0].bar(
        x,
        [
            row.get("max_abs_curvature_1pm", np.nan)
            for row in records
        ],
        color="#4c78a8",
    )
    axes[0].axhline(
        curvature_limit,
        color="#e45756",
        linestyle="--",
        label="limit",
    )
    axes[0].set_ylabel(r"$|\kappa|_{\max}$ (m$^{-1}$)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        x,
        [
            row.get("max_abs_curvature_rate_1pm2", np.nan)
            for row in records
        ],
        color="#f58518",
    )
    axes[1].axhline(
        rate_limit,
        color="#e45756",
        linestyle="--",
        label="limit",
    )
    axes[1].set_ylabel(r"$|\sigma|_{\max}$ (m$^{-2}$)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    margins = np.asarray(
        [
            row.get("minimum_dense_corridor_margin_m", np.nan)
            for row in records
        ],
        dtype=np.float64,
    )
    axes[2].bar(x, margins, color="#54a24b")
    axes[2].axhline(0.0, color="#e45756", linestyle="--")
    axes[2].set_ylabel("Minimum corridor margin (m)")
    axes[2].set_xlabel("Trial")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=60, ha="right")
    axes[2].grid(axis="y", alpha=0.25)
    path = figure_dir / "constraint_verification.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(str(path.relative_to(output_dir)))

    fig, axis = plt.subplots(figsize=(10, 8))
    for trial_id, poses, success in paths:
        axis.plot(
            poses[:, 0],
            poses[:, 1],
            color="#4c78a8" if success else "#e45756",
            alpha=0.55,
            linewidth=1.2,
            label=None,
        )
        axis.scatter(
            poses[-1, 0],
            poses[-1, 1],
            s=12,
            color="#54a24b" if success else "#e45756",
        )
    if paths:
        start = paths[0][1][0]
        axis.scatter(
            start[0],
            start[1],
            s=70,
            marker="o",
            color="black",
            label="fixed start",
            zorder=5,
        )
    axis.set_title(
        "Optimized paths (blue) and degraded fallbacks (red)"
    )
    axis.set_xlabel("Scene x")
    axis.set_ylabel("Scene y")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend()
    path = figure_dir / "trajectory_overlay.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(str(path.relative_to(output_dir)))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--selected-goals-json",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--scene",
        default="old_union",
        choices=sorted(SCENE_PRESETS),
    )
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--z-floor-scene", type=float, default=-0.15)
    parser.add_argument(
        "--robot-height-meters",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--footprint-radius-meters",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--ground-clearance-meters",
        type=float,
        default=0.02,
    )
    parser.add_argument("--max-speed-mps", type=float, default=0.20)
    parser.add_argument(
        "--max-brake-decel-mps2",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--min-turning-radius-meters",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--max-curvature-rate-1pm2",
        type=float,
        default=10.0,
    )
    parser.add_argument("--hybrid-heading-bins", type=int, default=72)
    parser.add_argument(
        "--hybrid-max-expansions",
        type=int,
        default=200_000,
    )
    parser.add_argument("--optimizer-nodes", type=int, default=12)
    parser.add_argument(
        "--optimizer-max-iterations",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--optimizer-max-refinements",
        type=int,
        default=2,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path_dir = args.output_dir / "paths"
    path_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        args.selected_goals_json.read_text(encoding="utf-8")
    )
    selected = select_medium_rows(manifest, args.sample_count)
    save_json(args.output_dir / "selected_goals.json", selected)

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
    stopping_distance_m = compute_stopping_distance(
        args.max_speed_mps,
        args.max_brake_decel_mps2,
    )
    preset = SCENE_PRESETS[args.scene]
    lower = torch.tensor(preset["lower_bound"], device=device)
    upper = torch.tensor(preset["upper_bound"], device=device)
    preprocessing = {}

    synchronize(device)
    begin = time.perf_counter()
    gsplat = GSplatLoader(args.config, device)
    synchronize(device)
    preprocessing["gsplat_load_s"] = time.perf_counter() - begin
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
            min_turning_radius_m=args.min_turning_radius_meters,
            max_expansions=args.hybrid_max_expansions,
        ),
    )
    planner = ThreeLayerGroundPlanner(
        hybrid,
        collision_set,
        ThreeLayerPlannerConfig(
            min_turning_radius_m=args.min_turning_radius_meters,
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

    records = []
    plot_paths = []
    record_path = args.output_dir / "records.jsonl"
    for index, trial in enumerate(selected):
        print(
            f"[{index + 1:03d}/{len(selected):03d}] "
            f"{trial['trial_id']}",
            flush=True,
        )
        begin = time.perf_counter()
        result = planner.plan(
            [
                *trial["start_scene"],
                trial["start_yaw_rad"],
            ],
            trial["goal_scene"],
        )
        total_time = time.perf_counter() - begin
        status = result.status()
        record = {
            "trial_id": trial["trial_id"],
            "candidate_id": trial["candidate_id"],
            "distance_layer": trial["distance_layer"],
            "direct_distance_m": trial["direct_distance_m"],
            "search_success": status["search_success"],
            "corridor_success": status["corridor_success"],
            "optimization_success": status[
                "optimization_success"
            ],
            "complete_success": status[
                "complete_planning_success"
            ],
            "fallback_used": status["fallback_used"],
            "failure_stage": status["failure_stage"],
            "failure_reason": status["failure_reason"],
            "total_plan_time_s": total_time,
            "hybrid_astar_time_s": status["timings"].get(
                "hybrid_astar_s"
            ),
            "ordered_corridor_time_s": status["timings"].get(
                "ordered_corridor_s"
            ),
            "curvature_collocation_time_s": status[
                "timings"
            ].get("curvature_collocation_s"),
        }
        if result.hybrid_path is not None:
            hybrid_summary = result.hybrid_path.summary()
            record.update(
                {
                    "hybrid_path_length_m": hybrid_summary[
                        "path_length_m"
                    ],
                    "hybrid_dense_pose_count": hybrid_summary[
                        "dense_pose_count"
                    ],
                    "expanded_nodes": hybrid_summary[
                        "expanded_nodes"
                    ],
                    "generated_nodes": hybrid_summary[
                        "generated_nodes"
                    ],
                }
            )
        if result.simplification is not None:
            simplification = result.simplification.summary()
            record["primitive_boundary_pose_count"] = (
                simplification["primitive_boundary_pose_count"]
            )
        if result.corridor is not None:
            record["corridor_count"] = len(
                result.corridor.corridors
            )
            record["corridor_overlap_count"] = len(
                result.corridor.overlaps
            )
        if result.trajectory_m is not None:
            optimizer = result.trajectory_m.summary()
            record.update(
                {
                    "optimized_path_length_m": optimizer[
                        "path_length"
                    ],
                    "optimizer_node_count": optimizer[
                        "node_count"
                    ],
                    "max_abs_curvature_1pm": optimizer[
                        "max_abs_curvature"
                    ],
                    "max_abs_curvature_rate_1pm2": optimizer[
                        "max_abs_curvature_rate"
                    ],
                }
            )
            if optimizer["path_length"] is not None:
                record["detour_factor"] = (
                    optimizer["path_length"]
                    / trial["direct_distance_m"]
                )
            if result.trajectory_m.diagnostics:
                diagnostics = result.trajectory_m.diagnostics[-1]
                record.update(
                    {
                        "optimizer_iteration_count": diagnostics.get(
                            "iterations"
                        ),
                        "minimum_dense_corridor_margin_m": (
                            diagnostics.get(
                                "minimum_dense_corridor_margin"
                            )
                        ),
                        "minimum_transition_overlap_margin_m": (
                            diagnostics.get(
                                "minimum_transition_overlap_margin"
                            )
                        ),
                        "max_collocation_defect": diagnostics.get(
                            "max_collocation_defect"
                        ),
                        "transition_node_count": diagnostics.get(
                            "transition_node_count"
                        ),
                    }
                )
        if len(result.dense_output_poses_scene):
            output_path = (
                path_dir / f"{trial['trial_id']}_output_scene.npy"
            )
            np.save(output_path, result.dense_output_poses_scene)
            record["output_path_file"] = str(
                output_path.relative_to(args.output_dir)
            )
            plot_paths.append(
                (
                    trial["trial_id"],
                    result.dense_output_poses_scene.copy(),
                    bool(result.success),
                )
            )
        records.append(record)
        append_jsonl(record_path, record)
        print(
            json.dumps(
                {
                    "complete": record["complete_success"],
                    "failure_stage": record["failure_stage"],
                    "hybrid_s": record.get(
                        "hybrid_astar_time_s"
                    ),
                    "corridors": record.get("corridor_count"),
                    "optimizer_nodes": record.get(
                        "optimizer_node_count"
                    ),
                    "optimizer_s": record.get(
                        "curvature_collocation_time_s"
                    ),
                }
            ),
            flush=True,
        )

    curvature_limit = (
        0.95 / args.min_turning_radius_meters
    )
    protocol = {
        "scene": args.scene,
        "device": str(device),
        "scene_scale": scale,
        "selected_goals_json": str(args.selected_goals_json),
        "sample_count": args.sample_count,
        "distance_layer": "medium",
        "goal_heading_mode": "free",
        "corridor_reference": "motion_primitive_boundaries",
        "collocation_method": "hermite_simpson",
        "solver": "SLSQP",
        "curvature_limit_1pm": curvature_limit,
        "max_curvature_rate_1pm2": (
            args.max_curvature_rate_1pm2
        ),
        "optimizer_nodes": args.optimizer_nodes,
        "optimizer_max_iterations": (
            args.optimizer_max_iterations
        ),
        "optimizer_max_refinements": (
            args.optimizer_max_refinements
        ),
        "stops_after": "final_constraint_verification",
    }
    visualizations = save_visualizations(
        records,
        args.output_dir,
        plot_paths,
        curvature_limit,
        args.max_curvature_rate_1pm2,
    )
    summary = build_summary(
        records,
        preprocessing,
        protocol,
    )
    summary["visualizations"] = visualizations
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
