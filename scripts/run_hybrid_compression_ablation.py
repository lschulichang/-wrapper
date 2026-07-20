#!/usr/bin/env python3
"""Paired 30-case ablation of Hybrid A* seed compression."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
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
from ground_nav.hybrid_astar import (  # noqa: E402
    HybridAStarConfig,
    HybridAStarPlanner,
)
from ground_nav.path_utils import simplify_ground_path_supercover  # noqa: E402
from ground_nav.planar_gaussians import PlanarGaussianSet  # noqa: E402
from ground_nav.timed_trajectory import evaluate_bezier_geometry  # noqa: E402
from run_grid_rasterization_ablation import (  # noqa: E402
    endpoint_indices,
    generate_candidates,
    path_length_m,
    select_stratified,
)
from smoke_ground_corridor_2d import (  # noqa: E402
    configure_map_axis,
    max_control_violation,
)
from smoke_splatplan import SCENE_PRESETS  # noqa: E402
from splat.splat_utils import GSplatLoader  # noqa: E402


VARIANTS = ("dense_direct", "direction_preserving_los")


def sync(device: torch.device) -> None:
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


def wrapped_angle_delta(values: np.ndarray) -> np.ndarray:
    return (np.diff(values) + np.pi) % (2.0 * np.pi) - np.pi


def motion_directions(poses: np.ndarray) -> np.ndarray:
    """Return +1/-1 for each nonzero path step using the pose heading."""

    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3 or len(poses) < 2:
        raise ValueError("poses must have shape (N, 3), N >= 2")
    steps = np.diff(poses[:, :2], axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("poses contain an adjacent duplicate")
    headings = np.column_stack(
        [np.cos(poses[:-1, 2]), np.sin(poses[:-1, 2])]
    )
    projections = np.sum(steps * headings, axis=1)
    if np.any(np.abs(projections) <= 1e-10 * lengths):
        raise ValueError("a path step is orthogonal to its motion heading")
    return np.where(projections > 0.0, 1, -1).astype(np.int8)


def motion_direction_runs(poses: np.ndarray) -> list[tuple[int, int, int]]:
    """Return inclusive pose-index runs without crossing a gear change."""

    directions = motion_directions(poses)
    runs: list[tuple[int, int, int]] = []
    start_pose = 0
    for step_index in range(1, len(directions)):
        if directions[step_index] == directions[step_index - 1]:
            continue
        runs.append(
            (start_pose, step_index, int(directions[step_index - 1]))
        )
        start_pose = step_index
    runs.append((start_pose, len(poses) - 1, int(directions[-1])))
    return runs


def compress_direction_preserving(
    poses: np.ndarray,
    grid: GroundGrid,
    max_segment_scene: float,
) -> tuple[np.ndarray, list[dict]]:
    """Apply conservative LOS compression independently inside each gear run."""

    poses = np.asarray(poses, dtype=np.float64)
    compressed: list[np.ndarray] = []
    diagnostics = []
    for start, end, direction in motion_direction_runs(poses):
        run = poses[start : end + 1, :2]
        simplified = simplify_ground_path_supercover(
            run,
            grid,
            max_segment_scene,
        )
        if compressed and np.allclose(
            compressed[-1], simplified[0], atol=1e-12, rtol=0.0
        ):
            compressed.extend(simplified[1:])
        else:
            compressed.extend(simplified)
        diagnostics.append(
            {
                "start_pose_index": int(start),
                "end_pose_index": int(end),
                "direction": "forward" if direction > 0 else "reverse",
                "input_points": int(len(run)),
                "output_points": int(len(simplified)),
            }
        )
    result = np.asarray(compressed, dtype=np.float64)
    if not np.allclose(result[0], poses[0, :2]):
        raise RuntimeError("compression changed the start point")
    if not np.allclose(result[-1], poses[-1, :2]):
        raise RuntimeError("compression changed the goal point")
    return result, diagnostics


def endpoint_yaws(reference_path: np.ndarray) -> tuple[float, float]:
    first = reference_path[1] - reference_path[0]
    last = reference_path[-1] - reference_path[-2]
    return (
        float(math.atan2(first[1], first[0])),
        float(math.atan2(last[1], last[0])),
    )


def trajectory_quality(
    controls: np.ndarray,
    dense: np.ndarray,
    scale: float,
) -> dict:
    values = np.linspace(0.0, 1.0, 101)
    curvature_parts = []
    for control in controls:
        _, first, second = evaluate_bezier_geometry(control / scale, values)
        speed = np.linalg.norm(first, axis=1)
        valid = speed > 1e-10
        curvature = np.full(len(values), np.nan)
        curvature[valid] = (
            first[valid, 0] * second[valid, 1]
            - first[valid, 1] * second[valid, 0]
        ) / speed[valid] ** 3
        curvature_parts.append(np.abs(curvature))
    curvature = np.concatenate(curvature_parts)
    finite = curvature[np.isfinite(curvature)]
    return {
        "bezier_length_m": float(
            sum(
                np.linalg.norm(np.diff(section, axis=0), axis=1).sum()
                for section in dense
            )
            / scale
        ),
        "max_abs_curvature_1pm": (
            float(np.max(finite)) if len(finite) else None
        ),
        "p95_abs_curvature_1pm": (
            float(np.quantile(finite, 0.95)) if len(finite) else None
        ),
        "degenerate_curvature_samples": int(
            np.count_nonzero(~np.isfinite(curvature))
        ),
    }


def run_variant(
    *,
    trial: dict,
    variant: str,
    hybrid_poses: np.ndarray,
    grid: GroundGrid,
    collision_set: PlanarCollisionSet,
    scale: float,
    max_segment_scene: float,
    start_yaw: float,
    goal_yaw: float,
    dense_samples: int,
    device: torch.device,
) -> tuple[dict, dict | None]:
    record = {
        "trial_id": trial["trial_id"],
        "candidate_id": trial["candidate_id"],
        "complexity_layer": trial["complexity_layer"],
        "variant": variant,
        "success": False,
        "failure_stage": None,
        "failure_reason": None,
    }
    artifact = None
    stage = "seed_preparation"
    try:
        begin = time.perf_counter()
        if variant == "dense_direct":
            seed = np.asarray(hybrid_poses[:, :2], dtype=np.float64)
            direction_runs = [
                {
                    "direction": "forward",
                    "input_points": int(len(seed)),
                    "output_points": int(len(seed)),
                }
            ]
        elif variant == "direction_preserving_los":
            seed, direction_runs = compress_direction_preserving(
                hybrid_poses,
                grid,
                max_segment_scene,
            )
        else:
            raise ValueError(f"unknown variant {variant}")
        record["time_seed_preparation_s"] = time.perf_counter() - begin
        record["seed_points"] = int(len(seed))
        record["seed_segments"] = int(len(seed) - 1)
        record["seed_path_length_m"] = path_length_m(seed, scale)
        record["seed_point_ratio"] = float(len(seed) / len(hybrid_poses))
        record["motion_direction_run_count"] = int(len(direction_runs))
        record["max_seed_segment_m"] = float(
            np.max(np.linalg.norm(np.diff(seed, axis=0), axis=1)) / scale
        )

        stage = "corridor"
        sync(device)
        begin = time.perf_counter()
        corridor = build_planar_corridor(seed, collision_set)
        sync(device)
        record["time_corridor_s"] = time.perf_counter() - begin
        record["polygon_count"] = int(len(corridor.polygons))
        created = [
            row for row in corridor.diagnostics if row.get("created_polygon")
        ]
        record["candidate_gaussian_total"] = int(
            sum(row.get("candidate_count", 0) for row in created)
        )
        record["halfspace_total"] = int(
            sum(row.get("halfspace_count", 0) for row in created)
        )

        stage = "bezier_qp"
        planner = BezierPlanner2D(degree=6, continuity_order=3)
        begin = time.perf_counter()
        controls, feasible = planner.optimize(
            corridor.polygons,
            seed[0],
            seed[-1],
            start_yaw=start_yaw,
            goal_yaw=goal_yaw,
            endpoint_tangent_min=(
                0.25 * float(torch.min(grid.cell_sizes).item())
            ),
        )
        record["time_qp_s"] = time.perf_counter() - begin
        record["qp_status"] = planner.last_solver_status
        if not feasible:
            raise RuntimeError(f"QP status {planner.last_solver_status}")
        dense = planner.sample(dense_samples)
        record["max_control_point_violation"] = max_control_violation(
            controls,
            corridor.polygons,
        )
        record["max_endpoint_error_m"] = float(
            max(
                np.linalg.norm(dense[0, 0] - seed[0]),
                np.linalg.norm(dense[-1, -1] - seed[-1]),
            )
            / scale
        )
        if (
            record["max_control_point_violation"] > 1e-5
            or record["max_endpoint_error_m"] > 1e-6
        ):
            raise RuntimeError("Bezier control-point verification failed")
        record.update(trajectory_quality(controls, dense, scale))
        record["time_downstream_s"] = float(
            record["time_seed_preparation_s"]
            + record["time_corridor_s"]
            + record["time_qp_s"]
        )
        record["success"] = True
        artifact = {
            "hybrid": hybrid_poses[:, :2],
            "seed": seed,
            "dense": dense,
        }
    except Exception as error:
        record["failure_stage"] = stage
        record["failure_reason"] = str(error)
        record["time_downstream_s"] = float(
            sum(
                record.get(key, 0.0)
                for key in (
                    "time_seed_preparation_s",
                    "time_corridor_s",
                    "time_qp_s",
                )
            )
        )
    return record, artifact


def numeric_summary(rows: list[dict], key: str) -> dict | None:
    values = np.asarray(
        [
            row[key]
            for row in rows
            if row.get("success") and row.get(key) is not None
        ],
        dtype=np.float64,
    )
    if not len(values):
        return None
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def paired_delta_summary(values: list[float]) -> dict | None:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return None
    result = {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }
    if np.any(np.abs(array) > 1e-12):
        test = scipy.stats.wilcoxon(array, zero_method="wilcox")
        result["wilcoxon_statistic"] = float(test.statistic)
        result["wilcoxon_p"] = float(test.pvalue)
    else:
        result["wilcoxon_statistic"] = None
        result["wilcoxon_p"] = None
    return result


def build_summary(
    records: list[dict],
    search_records: list[dict],
    sample_count: int,
) -> dict:
    metrics = (
        "seed_points",
        "seed_point_ratio",
        "seed_path_length_m",
        "max_seed_segment_m",
        "polygon_count",
        "candidate_gaussian_total",
        "halfspace_total",
        "time_seed_preparation_s",
        "time_corridor_s",
        "time_qp_s",
        "time_downstream_s",
        "bezier_length_m",
        "max_abs_curvature_1pm",
        "p95_abs_curvature_1pm",
    )
    summary = {
        "sample_count": int(sample_count),
        "paired_variants_per_trial": list(VARIANTS),
        "search": {
            "success_count": int(
                sum(row["success"] for row in search_records)
            ),
            "failure_count": int(
                sum(not row["success"] for row in search_records)
            ),
            "metrics": {
                key: numeric_summary(
                    [
                        {**row, "success": row["success"]}
                        for row in search_records
                    ],
                    key,
                )
                for key in (
                    "time_hybrid_astar_s",
                    "path_length_m",
                    "dense_pose_count",
                    "expanded_nodes",
                    "generated_nodes",
                )
            },
        },
        "variants": {},
    }
    for variant in VARIANTS:
        rows = [row for row in records if row["variant"] == variant]
        successful = [row for row in rows if row["success"]]
        failure_stages = sorted(
            {
                row["failure_stage"]
                for row in rows
                if row.get("failure_stage")
            }
        )
        summary["variants"][variant] = {
            "success_count": int(len(successful)),
            "failure_count": int(len(rows) - len(successful)),
            "success_rate_over_all_trials": float(
                len(successful) / sample_count
            ),
            "failure_stages": {
                stage: int(
                    sum(row.get("failure_stage") == stage for row in rows)
                )
                for stage in failure_stages
            },
            "metrics": {
                key: numeric_summary(rows, key) for key in metrics
            },
        }

    by_trial: dict[str, dict[str, dict]] = {}
    for row in records:
        by_trial.setdefault(row["trial_id"], {})[row["variant"]] = row
    paired = [
        variants
        for variants in by_trial.values()
        if all(
            name in variants and variants[name]["success"]
            for name in VARIANTS
        )
    ]
    dense_only = sum(
        variants.get("dense_direct", {}).get("success", False)
        and not variants.get("direction_preserving_los", {}).get(
            "success", False
        )
        for variants in by_trial.values()
    )
    compressed_only = sum(
        variants.get("direction_preserving_los", {}).get("success", False)
        and not variants.get("dense_direct", {}).get("success", False)
        for variants in by_trial.values()
    )
    discordant = dense_only + compressed_only
    summary["paired_success"] = {
        "both_success_count": int(len(paired)),
        "dense_only_success": int(dense_only),
        "compressed_only_success": int(compressed_only),
        "discordant_count": int(discordant),
        "exact_two_sided_p": (
            float(
                scipy.stats.binomtest(
                    dense_only,
                    discordant,
                    p=0.5,
                ).pvalue
            )
            if discordant
            else None
        ),
    }
    delta_metrics = (
        "seed_points",
        "polygon_count",
        "time_corridor_s",
        "time_qp_s",
        "time_downstream_s",
        "bezier_length_m",
        "max_abs_curvature_1pm",
    )
    summary["paired_deltas_compressed_minus_dense"] = {
        key: paired_delta_summary(
            [
                rows["direction_preserving_los"][key]
                - rows["dense_direct"][key]
                for rows in paired
                if rows["direction_preserving_los"].get(key) is not None
                and rows["dense_direct"].get(key) is not None
            ]
        )
        for key in delta_metrics
    }
    return summary


def save_csv(path: Path, records: list[dict]) -> None:
    fields = sorted({key for row in records for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def plot_results(
    output_dir: Path,
    records: list[dict],
    selected: list[dict],
    artifacts: dict[tuple[str, str], dict],
    grid: GroundGrid,
    extent: list[float],
) -> None:
    successful = {
        variant: [
            row
            for row in records
            if row["variant"] == variant and row["success"]
        ]
        for variant in VARIANTS
    }
    labels = ["Dense direct", "Direction-preserving LOS"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(
        labels,
        [len(successful[name]) / 30.0 for name in VARIANTS],
    )
    axes[0].set(ylim=(0, 1.05), title="End-to-end success rate")
    for axis, metric, title in (
        (axes[1], "seed_points", "Seed point count"),
        (axes[2], "polygon_count", "Safety polygon count"),
    ):
        axis.boxplot(
            [
                [row[metric] for row in successful[name]]
                for name in VARIANTS
            ],
            tick_labels=labels,
        )
        axis.set_title(title)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    fig.savefig(output_dir / "success_and_corridor_size.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, metric, title in (
        (axes[0], "time_corridor_s", "Corridor time [s]"),
        (axes[1], "time_qp_s", "QP time [s]"),
        (axes[2], "max_abs_curvature_1pm", "Max |curvature| [1/m]"),
    ):
        axis.boxplot(
            [
                [
                    row[metric]
                    for row in successful[name]
                    if row.get(metric) is not None
                ]
                for name in VARIANTS
            ],
            tick_labels=labels,
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    fig.savefig(output_dir / "timing_and_curvature.png", dpi=180)
    plt.close(fig)

    representatives = []
    for layer in ("low", "medium", "high"):
        rows = sorted(
            [
                row
                for row in selected
                if row["complexity_layer"] == layer
            ],
            key=lambda row: row["reference_path_length_m"],
        )
        representatives.append(rows[len(rows) // 2])
    occupied = grid.occupied.detach().cpu().numpy()
    fig, axes = plt.subplots(3, 2, figsize=(13, 17), sharex=True, sharey=True)
    for row_index, trial in enumerate(representatives):
        for column_index, variant in enumerate(VARIANTS):
            axis = axes[row_index, column_index]
            axis.imshow(
                occupied.T,
                origin="lower",
                extent=extent,
                cmap="Greys",
                interpolation="nearest",
                aspect="equal",
            )
            artifact = artifacts.get((trial["trial_id"], variant))
            if artifact is None:
                matching = next(
                    (
                        row
                        for row in records
                        if row["trial_id"] == trial["trial_id"]
                        and row["variant"] == variant
                    ),
                    None,
                )
                message = (
                    "Hybrid A* failed"
                    if matching is None
                    else f"{matching['failure_stage']}\n"
                    f"{matching['failure_reason']}"
                )
                axis.text(
                    0.5,
                    0.5,
                    message,
                    color="red",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            else:
                axis.plot(
                    artifact["hybrid"][:, 0],
                    artifact["hybrid"][:, 1],
                    color="tab:blue",
                    linewidth=0.9,
                    label="Hybrid A* dense",
                )
                axis.plot(
                    artifact["seed"][:, 0],
                    artifact["seed"][:, 1],
                    "o--",
                    color="tab:orange",
                    markersize=2.5,
                    label="corridor seed",
                )
                for section_index, section in enumerate(artifact["dense"]):
                    axis.plot(
                        section[:, 0],
                        section[:, 1],
                        color="tab:red",
                        linewidth=1.5,
                        label="Bezier" if section_index == 0 else None,
                    )
                axis.legend(fontsize=7)
            configure_map_axis(
                axis,
                extent,
                f"{trial['complexity_layer']} / {variant}",
            )
    fig.tight_layout()
    fig.savefig(output_dir / "representative_paths.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--scene", default="old_union", choices=sorted(SCENE_PRESETS)
    )
    parser.add_argument("--z-floor-scene", type=float, default=-0.15)
    parser.add_argument("--robot-height-meters", type=float, default=0.10)
    parser.add_argument("--footprint-radius-meters", type=float, default=0.15)
    parser.add_argument("--ground-clearance-meters", type=float, default=0.02)
    parser.add_argument("--max-speed-mps", type=float, default=0.20)
    parser.add_argument("--max-brake-decel-mps2", type=float, default=0.30)
    parser.add_argument("--max-segment-meters", type=float, default=1.0)
    parser.add_argument("--endpoint-clearance-meters", type=float, default=0.05)
    parser.add_argument("--min-distance-meters", type=float, default=2.0)
    parser.add_argument("--candidate-count", type=int, default=300)
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--dense-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--min-turning-radius-meters", type=float, default=0.20)
    parser.add_argument("--hybrid-heading-bins", type=int, default=72)
    parser.add_argument("--hybrid-max-expansions", type=int, default=200000)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.sample_count != 30:
        raise ValueError("the standard paired ablation requires 30 trials")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale = float(
        json.loads(
            (
                args.config.resolve().parent / "dataparser_transforms.json"
            ).read_text()
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
    z_min = args.z_floor_scene + clearance
    z_max = args.z_floor_scene + height
    preprocessing = {}

    sync(device)
    begin = time.perf_counter()
    gsplat = GSplatLoader(args.config, device)
    sync(device)
    preprocessing["gsplat_load_s"] = time.perf_counter() - begin
    begin = time.perf_counter()
    ellipses = PlanarGaussianSet.from_gsplat(
        gsplat,
        z_min,
        z_max,
        confidence=1.0,
    )
    sync(device)
    preprocessing["height_filter_projection_s"] = time.perf_counter() - begin

    grids = {}
    for rasterization in ("ellipse_distance", "aabb_disk_dilation"):
        begin = time.perf_counter()
        grids[rasterization] = GroundGrid.from_projected_gaussians(
            ellipses,
            lower_xy=lower[:2],
            upper_xy=upper[:2],
            resolution_xy=preset["resolution"],
            footprint_radius=radius,
            z_floor_scene=args.z_floor_scene,
            robot_height=height,
            ground_clearance=clearance,
            project_occupied_endpoints=False,
            rasterization=rasterization,
        )
        sync(device)
        preprocessing[f"rasterization_{rasterization}_s"] = (
            time.perf_counter() - begin
        )
    grid = grids["ellipse_distance"]
    collision_set = PlanarCollisionSet(
        ellipses,
        radius=radius,
        stopping_distance=stopping_distance_m * scale,
        iterations=10,
    )

    indices, endpoint_info = endpoint_indices(
        grid,
        grids["aabb_disk_dilation"],
        scale,
        args.endpoint_clearance_meters,
    )
    begin = time.perf_counter()
    candidates, candidate_info = generate_candidates(
        grid,
        indices,
        scale,
        args.candidate_count,
        args.min_distance_meters,
        args.seed,
    )
    preprocessing["candidate_generation_s"] = time.perf_counter() - begin
    hybrid_config = HybridAStarConfig(
        heading_bins=args.hybrid_heading_bins,
        min_turning_radius_m=args.min_turning_radius_meters,
        max_expansions=args.hybrid_max_expansions,
    )
    hybrid = HybridAStarPlanner(
        grid,
        scene_scale=scale,
        config=hybrid_config,
    )
    eligible_candidates = []
    excluded_start_poses = []
    for candidate in candidates:
        start_yaw, goal_yaw = endpoint_yaws(candidate["reference_path"])
        candidate["start_yaw_rad"] = start_yaw
        candidate["goal_yaw_rad"] = goal_yaw
        feasible_indices = hybrid.feasible_forward_primitive_indices(
            [*candidate["start_scene"], start_yaw]
        )
        candidate["feasible_start_primitive_indices"] = list(
            feasible_indices
        )
        candidate["feasible_start_primitive_count"] = int(
            len(feasible_indices)
        )
        candidate["feasible_start_curvature_fractions"] = [
            float(hybrid.config.curvature_fractions[index])
            for index in feasible_indices
        ]
        if feasible_indices:
            eligible_candidates.append(candidate)
            continue
        excluded_start_poses.append(
            {
                **{
                    key: value
                    for key, value in candidate.items()
                    if key != "reference_path"
                },
                "exclusion_reason": (
                    "collision-free start pose has no feasible forward "
                    "motion primitive"
                ),
            }
        )
    if len(eligible_candidates) < args.sample_count:
        raise RuntimeError(
            f"only {len(eligible_candidates)} candidates remain after "
            "the forward-primitive feasibility filter"
        )
    selected = select_stratified(
        eligible_candidates,
        args.sample_count,
    )
    if any(
        trial["feasible_start_primitive_count"] == 0
        for trial in selected
    ):
        raise RuntimeError(
            "formal sample contains a zero-forward-primitive start pose"
        )
    save_json(
        args.output_dir / "excluded_start_poses.json",
        excluded_start_poses,
    )
    save_json(
        args.output_dir / "selected_pairs.json",
        [
            {
                **{
                    key: value
                    for key, value in trial.items()
                    if key != "reference_path"
                },
                "reference_path": trial["reference_path"].tolist(),
            }
            for trial in selected
        ],
    )
    representative_ids = set()
    for layer in ("low", "medium", "high"):
        rows = sorted(
            [
                row
                for row in selected
                if row["complexity_layer"] == layer
            ],
            key=lambda row: row["reference_path_length_m"],
        )
        representative_ids.add(rows[len(rows) // 2]["trial_id"])

    records: list[dict] = []
    search_records: list[dict] = []
    artifacts: dict[tuple[str, str], dict] = {}
    record_path = args.output_dir / "trial_records.jsonl"
    search_path = args.output_dir / "search_records.jsonl"
    for trial_index, trial in enumerate(selected):
        print(
            f"[{trial_index + 1:02d}/30] {trial['trial_id']} Hybrid A*",
            flush=True,
        )
        search_record = {
            "trial_id": trial["trial_id"],
            "candidate_id": trial["candidate_id"],
            "complexity_layer": trial["complexity_layer"],
            "start_yaw_rad": trial["start_yaw_rad"],
            "goal_yaw_rad": trial["goal_yaw_rad"],
            "feasible_start_primitive_count": trial[
                "feasible_start_primitive_count"
            ],
            "feasible_start_primitive_indices": trial[
                "feasible_start_primitive_indices"
            ],
            "success": False,
            "failure_reason": None,
        }
        hybrid_path = None
        try:
            begin = time.perf_counter()
            hybrid_path = hybrid.plan(
                [
                    *trial["start_scene"],
                    trial["start_yaw_rad"],
                ],
                [
                    *trial["goal_scene"],
                    trial["goal_yaw_rad"],
                ],
            )
            search_record["time_hybrid_astar_s"] = (
                time.perf_counter() - begin
            )
            search_record.update(hybrid_path.summary())
            search_record["success"] = True
        except Exception as error:
            search_record["failure_reason"] = str(error)
        search_records.append(search_record)
        append_jsonl(search_path, search_record)

        if hybrid_path is None:
            print(
                json.dumps(
                    {
                        "search_success": False,
                        "failure": search_record["failure_reason"],
                    }
                ),
                flush=True,
            )
            continue

        order = (
            VARIANTS
            if trial_index % 2 == 0
            else tuple(reversed(VARIANTS))
        )
        for execution_order, variant in enumerate(order):
            record, artifact = run_variant(
                trial=trial,
                variant=variant,
                hybrid_poses=hybrid_path.poses_scene,
                grid=grid,
                collision_set=collision_set,
                scale=scale,
                max_segment_scene=args.max_segment_meters * scale,
                start_yaw=trial["start_yaw_rad"],
                goal_yaw=trial["goal_yaw_rad"],
                dense_samples=args.dense_samples,
                device=device,
            )
            record["execution_order_within_pair"] = int(execution_order)
            record["time_hybrid_astar_s"] = search_record[
                "time_hybrid_astar_s"
            ]
            records.append(record)
            append_jsonl(record_path, record)
            if (
                trial["trial_id"] in representative_ids
                and artifact is not None
            ):
                artifacts[(trial["trial_id"], variant)] = artifact
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "success": record["success"],
                        "points": record.get("seed_points"),
                        "polygons": record.get("polygon_count"),
                        "downstream_s": record.get("time_downstream_s"),
                        "failure": record.get("failure_reason"),
                    }
                ),
                flush=True,
            )

    save_csv(args.output_dir / "paired_metrics.csv", records)
    failures = {
        "search": [row for row in search_records if not row["success"]],
        "downstream": [row for row in records if not row["success"]],
    }
    save_json(args.output_dir / "failures.json", failures)
    summary = build_summary(records, search_records, args.sample_count)
    summary.update(
        {
            "scene": args.scene,
            "device": str(device),
            "scale": scale,
            "preprocessing": preprocessing,
            "endpoint_selection": endpoint_info,
            "candidate_generation": candidate_info,
            "start_pose_prefilter": {
                "definition": (
                    "exclude a collision-free start pose when every configured "
                    "forward Hybrid A* motion primitive collides"
                ),
                "generated_candidate_count": int(len(candidates)),
                "eligible_candidate_count": int(len(eligible_candidates)),
                "excluded_candidate_count": int(
                    len(excluded_start_poses)
                ),
                "formal_selected_count": int(len(selected)),
            },
            "physical_parameters": {
                "robot_height_m": args.robot_height_meters,
                "footprint_radius_m": args.footprint_radius_meters,
                "ground_clearance_m": args.ground_clearance_meters,
                "max_speed_mps": args.max_speed_mps,
                "max_brake_decel_mps2": args.max_brake_decel_mps2,
                "stopping_distance_m": stopping_distance_m,
                "max_segment_m": args.max_segment_meters,
                "min_turning_radius_m": args.min_turning_radius_meters,
            },
            "protocol": {
                "seed": args.seed,
                "candidate_count": args.candidate_count,
                "sample_count": args.sample_count,
                "strata": ["low", "medium", "high"],
                "endpoint_yaw_source": (
                    "first and last segment directions of the deterministic "
                    "ellipse-distance Dijkstra reference path"
                ),
                "continuous_posthoc_collision_review": False,
                "main_pipeline_modified": False,
                "zero_forward_primitive_starts_in_formal_sample": False,
                "direction_preservation_note": (
                    "Compression is independently applied inside each "
                    "forward/reverse run. The current Hybrid A* planner is "
                    "forward-only, so all observed runs are forward."
                ),
            },
        }
    )
    save_json(args.output_dir / "summary.json", summary)
    extent = [
        float(lower[0]),
        float(upper[0]),
        float(lower[1]),
        float(upper[1]),
    ]
    plot_results(
        args.output_dir,
        records,
        selected,
        artifacts,
        grid,
        extent,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
