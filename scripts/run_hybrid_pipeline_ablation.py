#!/usr/bin/env python3
"""B/C/D ablation for raw Hybrid, corridor Bezier, and curvature collocation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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

from ground_nav.bezier_2d import BezierPlanner2D  # noqa: E402
from ground_nav.corridor_2d import (  # noqa: E402
    PlanarCollisionSet,
    build_planar_corridor,
    compute_stopping_distance,
)
from ground_nav.curvature_trajectory_optimizer import (  # noqa: E402
    CurvatureTrajectoryOptimizer,
    TrajectoryOptimizerConfig,
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
    trajectory_quality,
)
from smoke_ground_corridor_2d import (  # noqa: E402
    configure_map_axis,
    max_control_violation,
)
from smoke_splatplan import SCENE_PRESETS  # noqa: E402
from splat.splat_utils import GSplatLoader  # noqa: E402


GROUPS = (
    "B_hybrid_raw",
    "C_corridor_bezier",
    "D_curvature_optimizer",
)


def wrapped_differences(values: np.ndarray) -> np.ndarray:
    return (np.diff(values) + np.pi) % (2.0 * np.pi) - np.pi


def path_length_m(points: np.ndarray, scale: float) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum() / scale)


def flatten_sections(sections: np.ndarray) -> np.ndarray:
    parts = [sections[0]]
    parts.extend(section[1:] for section in sections[1:])
    return np.concatenate(parts, axis=0)


def continuous_collision_statistics(
    points: np.ndarray,
    collision_set: PlanarCollisionSet,
) -> dict:
    safe = []
    with torch.no_grad():
        for first, second in zip(points[:-1], points[1:]):
            safe.append(
                collision_set.segment_is_safe(np.stack([first, second]))
            )
    safe = np.asarray(safe, dtype=bool)
    violating = int(np.count_nonzero(~safe))
    return {
        "continuous_segment_count": int(len(safe)),
        "continuous_unsafe_segment_count": violating,
        "continuous_unsafe_segment_fraction": (
            float(violating / len(safe)) if len(safe) else 0.0
        ),
        "collision_free": bool(violating == 0),
    }


def free_space_statistics(points: np.ndarray, grid: GroundGrid) -> dict:
    points = np.asarray(points, dtype=np.float64)
    lower_center = grid.lower_center.detach().cpu().numpy()
    cell_sizes = grid.cell_sizes.detach().cpu().numpy()
    shape = np.asarray(grid.shape, dtype=np.int64)
    lower_edge = lower_center - 0.5 * cell_sizes
    upper_edge = lower_center + (shape - 0.5) * cell_sizes
    inside_bounds = np.all(
        (points >= lower_edge[None]) & (points < upper_edge[None]),
        axis=1,
    )
    indices = np.rint((points - lower_center[None]) / cell_sizes[None]).astype(int)
    valid_indices = np.clip(indices, 0, shape - 1)
    occupied = grid.occupied.detach().cpu().numpy().astype(bool)
    inside_free = inside_bounds & ~occupied[
        valid_indices[:, 0], valid_indices[:, 1]
    ]
    outside_count = int(np.count_nonzero(~inside_free))
    return {
        "curve_sample_count": int(len(points)),
        "outside_free_sample_count": outside_count,
        "outside_free_sample_fraction": float(outside_count / len(points)),
        "inside_free_sample_fraction": float(np.mean(inside_free)),
    }


def raw_record(
    trial: dict,
    hybrid_path,
    grid: GroundGrid,
    collision_set: PlanarCollisionSet,
    scale: float,
    search_time: float,
    min_turning_radius_m: float,
) -> dict:
    signed_curvatures = hybrid_path.primitive_curvatures_1pm
    curvatures = np.abs(signed_curvatures)
    curvature_changes = (
        int(np.count_nonzero(np.abs(np.diff(signed_curvatures)) > 1e-6))
        if len(curvatures) > 1
        else 0
    )
    record = {
        "trial_id": trial["trial_id"],
        "candidate_id": trial["candidate_id"],
        "distance_layer": trial["distance_layer"],
        "group": "B_hybrid_raw",
        "search_success": True,
        "corridor_success": None,
        "optimization_success": None,
        "tracking_success": None,
        "path_length_m": path_length_m(hybrid_path.xy, scale),
        "pose_count": int(len(hybrid_path.poses_scene)),
        "primitive_count": int(len(hybrid_path.primitive_curvatures_1pm)),
        "curvature_change_count": curvature_changes,
        "total_abs_heading_change_rad": float(
            np.abs(wrapped_differences(hybrid_path.poses_scene[:, 2])).sum()
        ),
        "max_abs_curvature_1pm": (
            float(np.max(curvatures)) if len(curvatures) else 0.0
        ),
        "p95_abs_curvature_1pm": (
            float(np.quantile(curvatures, 0.95)) if len(curvatures) else 0.0
        ),
        "time_hybrid_astar_s": float(search_time),
    }
    record.update(continuous_collision_statistics(hybrid_path.xy, collision_set))
    record.update(free_space_statistics(hybrid_path.xy, grid))
    record["continuous_safe"] = bool(record["collision_free"])
    record["grid_safe"] = bool(record["outside_free_sample_count"] == 0)
    record["curvature_feasible"] = bool(
        record["max_abs_curvature_1pm"]
        <= 1.0 / float(min_turning_radius_m) + 1e-9
    )
    record["curvature_rate_feasible"] = False
    record["success"] = bool(
        record["continuous_safe"]
        and record["grid_safe"]
        and record["curvature_feasible"]
    )
    return record


def corridor_bezier_record(
    trial: dict,
    hybrid_path,
    grid: GroundGrid,
    collision_set: PlanarCollisionSet,
    scale: float,
    min_turning_radius_m: float,
    samples_per_section: int,
    device: torch.device,
) -> tuple[dict, dict | None]:
    record = {
        "trial_id": trial["trial_id"],
        "candidate_id": trial["candidate_id"],
        "distance_layer": trial["distance_layer"],
        "group": "C_corridor_bezier",
        "search_success": True,
        "corridor_success": False,
        "success": False,
        "optimization_success": False,
        "grid_safe": False,
        "continuous_safe": False,
        "curvature_feasible": False,
        "curvature_rate_feasible": False,
        "tracking_success": None,
        "failure_stage": None,
        "failure_reason": None,
    }
    seed = hybrid_path.xy
    stage = "corridor"
    try:
        sync(device)
        begin = time.perf_counter()
        corridor = build_planar_corridor(hybrid_path.poses_scene, collision_set)
        sync(device)
        record["time_corridor_s"] = time.perf_counter() - begin
        record["polygon_count"] = int(len(corridor.polygons))
        record["corridor_success"] = True
        record["overlap_count"] = int(len(corridor.overlaps))
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
            start_yaw=float(hybrid_path.poses_scene[0, 2]),
            goal_yaw=None,
            endpoint_tangent_min=(
                0.25 * float(torch.min(grid.cell_sizes).item())
            ),
        )
        record["time_qp_s"] = time.perf_counter() - begin
        record["qp_status"] = planner.last_solver_status
        if not feasible:
            raise RuntimeError(f"QP status {planner.last_solver_status}")
        dense = planner.sample(samples_per_section)
        record["max_control_point_violation"] = max_control_violation(
            controls,
            corridor.polygons,
        )
        if record["max_control_point_violation"] > 1e-5:
            raise RuntimeError("Bezier control-point verification failed")
        record.update(trajectory_quality(controls, dense, scale))
        record["path_length_m"] = record["bezier_length_m"]
        record["path_length_ratio_to_hybrid"] = float(
            record["path_length_m"] / path_length_m(seed, scale)
        )
        record["optimization_success"] = True
        record["time_downstream_s"] = float(
            record["time_corridor_s"] + record["time_qp_s"]
        )

        stage = "curvature_check"
        curvature_limit = 1.0 / float(min_turning_radius_m)
        record["curvature_limit_1pm"] = curvature_limit
        record["curvature_margin_1pm"] = float(
            curvature_limit - record["max_abs_curvature_1pm"]
        )
        record["curvature_feasible"] = bool(
            record["max_abs_curvature_1pm"] <= curvature_limit + 1e-9
            and record["degenerate_curvature_samples"] == 0
        )
        if not record["curvature_feasible"]:
            record["failure_stage"] = stage
            record["failure_reason"] = (
                "Bezier curvature exceeds forward Hybrid A* limit"
            )
        flat = flatten_sections(dense)
        record.update(continuous_collision_statistics(flat, collision_set))
        record.update(free_space_statistics(flat, grid))
        record["continuous_safe"] = bool(record["collision_free"])
        record["grid_safe"] = bool(record["outside_free_sample_count"] == 0)
        record["success"] = bool(
            record["optimization_success"]
            and record["grid_safe"]
            and record["continuous_safe"]
            and record["curvature_feasible"]
        )
        return record, {
            "hybrid": seed,
            "curve": flat,
        }
    except Exception as error:
        record["failure_stage"] = stage
        record["failure_reason"] = str(error)
        return record, None


def curvature_optimizer_record(
    trial: dict,
    hybrid_path,
    grid: GroundGrid,
    collision_set: PlanarCollisionSet,
    scale: float,
    min_turning_radius_m: float,
    max_curvature_rate_1pm2: float,
    optimizer_nodes: int,
    optimizer_max_iterations: int,
    optimizer_max_refinements: int,
    device: torch.device,
) -> tuple[dict, dict | None]:
    record = {
        "trial_id": trial["trial_id"],
        "candidate_id": trial["candidate_id"],
        "distance_layer": trial["distance_layer"],
        "group": "D_curvature_optimizer",
        "search_success": True,
        "corridor_success": False,
        "optimization_success": False,
        "grid_safe": False,
        "continuous_safe": False,
        "curvature_feasible": False,
        "curvature_rate_feasible": False,
        "tracking_success": None,
        "fallback_used": False,
        "success": False,
        "failure_stage": None,
        "failure_reason": None,
    }
    stage = "corridor"
    try:
        sync(device)
        begin = time.perf_counter()
        corridor = build_planar_corridor(
            hybrid_path.poses_scene,
            collision_set,
        )
        sync(device)
        record["time_corridor_s"] = time.perf_counter() - begin
        record["corridor_success"] = True
        record["polygon_count"] = int(len(corridor.corridors))
        record["overlap_count"] = int(len(corridor.overlaps))

        stage = "curvature_optimizer"
        reference_m = np.asarray(hybrid_path.poses_scene, dtype=np.float64).copy()
        reference_m[:, :2] /= scale
        corridors_m = [
            (
                np.asarray(A.detach().cpu(), dtype=np.float64),
                np.asarray(b.detach().cpu(), dtype=np.float64) / scale,
            )
            for A, b in corridor.corridors
        ]
        optimizer = CurvatureTrajectoryOptimizer(
            TrajectoryOptimizerConfig(
                node_count=optimizer_nodes,
                max_iterations=optimizer_max_iterations,
                max_refinement_steps=optimizer_max_refinements,
            )
        )
        print(
            json.dumps(
                {
                    "trial_id": trial["trial_id"],
                    "stage": "curvature_optimizer",
                    "reference_pose_count": int(len(reference_m)),
                    "corridor_count": int(len(corridor.corridors)),
                    "requested_nodes": int(optimizer_nodes),
                    "max_iterations": int(optimizer_max_iterations),
                    "max_refinements": int(optimizer_max_refinements),
                }
            ),
            flush=True,
        )
        trajectory = optimizer.optimize(
            reference_m,
            corridors_m,
            corridor.path_to_corridor,
            start_pose=reference_m[0],
            goal_xy=reference_m[-1, :2],
            min_turning_radius=min_turning_radius_m,
            max_curvature_rate=max_curvature_rate_1pm2,
        )
        record["time_optimizer_s"] = trajectory.solve_time
        record["optimizer_attempt_count"] = len(trajectory.diagnostics)
        print(
            json.dumps(
                {
                    "trial_id": trial["trial_id"],
                    "stage": "curvature_optimizer_done",
                    "success": trajectory.success,
                    "solve_time_s": trajectory.solve_time,
                    "attempt_count": len(trajectory.diagnostics),
                    "failure_reason": trajectory.failure_reason,
                }
            ),
            flush=True,
        )
        if trajectory.diagnostics:
            record["optimizer_status"] = trajectory.diagnostics[-1][
                "solver_message"
            ]
            record["max_dynamics_error"] = trajectory.diagnostics[-1][
                "max_dense_dynamics_error"
            ]
            record["min_corridor_margin_m"] = trajectory.diagnostics[-1][
                "minimum_dense_corridor_margin"
            ]
            record["applied_curvature_rate_limit_1pm2"] = trajectory.diagnostics[
                -1
            ]["curvature_rate_limit"]
        if not trajectory.success:
            raw_continuous = continuous_collision_statistics(
                hybrid_path.xy, collision_set
            )
            raw_grid = free_space_statistics(hybrid_path.xy, grid)
            raw_curvature_ok = bool(
                np.max(np.abs(hybrid_path.primitive_curvatures_1pm))
                <= 1.0 / min_turning_radius_m + 1e-9
            )
            record.update(raw_continuous)
            record.update(raw_grid)
            record["continuous_safe"] = bool(raw_continuous["collision_free"])
            record["grid_safe"] = bool(
                raw_grid["outside_free_sample_count"] == 0
            )
            record["curvature_feasible"] = raw_curvature_ok
            record["fallback_used"] = bool(
                record["continuous_safe"]
                and record["grid_safe"]
                and raw_curvature_ok
            )
            record["failure_stage"] = stage
            record["failure_reason"] = trajectory.failure_reason
            return record, {
                "hybrid": hybrid_path.xy,
                "curve": hybrid_path.xy,
                "fallback": True,
            }

        record["optimization_success"] = True
        dense_scene = np.asarray(trajectory.dense_poses, dtype=np.float64).copy()
        dense_scene[:, :2] *= scale
        record["path_length_m"] = float(trajectory.arc_length[-1])
        record["path_length_ratio_to_hybrid"] = float(
            record["path_length_m"] / path_length_m(hybrid_path.xy, scale)
        )
        record["max_abs_curvature_1pm"] = float(
            np.max(np.abs(trajectory.curvature))
        )
        record["max_abs_curvature_rate_1pm2"] = float(
            np.max(np.abs(trajectory.curvature_rate))
        )
        curvature_limit = 0.95 / min_turning_radius_m
        applied_rate_limit = float(
            trajectory.diagnostics[-1]["curvature_rate_limit"]
        )
        record["curvature_limit_1pm"] = curvature_limit
        record["curvature_feasible"] = bool(
            record["max_abs_curvature_1pm"] <= curvature_limit + 1e-6
        )
        record["curvature_rate_feasible"] = bool(
            record["max_abs_curvature_rate_1pm2"]
            <= applied_rate_limit + 1e-6
        )
        record.update(
            continuous_collision_statistics(dense_scene[:, :2], collision_set)
        )
        record.update(free_space_statistics(dense_scene[:, :2], grid))
        record["continuous_safe"] = bool(record["collision_free"])
        record["grid_safe"] = bool(record["outside_free_sample_count"] == 0)
        record["success"] = bool(
            record["optimization_success"]
            and record["grid_safe"]
            and record["continuous_safe"]
            and record["curvature_feasible"]
            and record["curvature_rate_feasible"]
        )
        if not record["success"]:
            record["failure_stage"] = "verification"
            record["failure_reason"] = "optimized trajectory failed final checks"
        return record, {
            "hybrid": hybrid_path.xy,
            "curve": dense_scene[:, :2],
        }
    except Exception as error:
        record["failure_stage"] = stage
        record["failure_reason"] = str(error)
        return record, None


def select_manifest_rows(rows: list[dict], sample_count: int) -> list[dict]:
    if sample_count <= 0 or sample_count % 3:
        raise ValueError("sample_count must be a positive multiple of three")
    selected = []
    per_layer = sample_count // 3
    for layer in ("near", "medium", "far"):
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


def summary_for_records(
    search_records: list[dict],
    records: list[dict],
    sample_count: int,
) -> dict:
    metrics = (
        "path_length_m",
        "max_abs_curvature_1pm",
        "p95_abs_curvature_1pm",
        "time_hybrid_astar_s",
        "time_bezier_s",
        "time_collision_check_s",
        "time_corridor_s",
        "time_qp_s",
        "time_downstream_s",
        "time_optimizer_s",
        "pose_count",
        "primitive_count",
        "anchor_count",
        "section_count",
        "polygon_count",
        "outside_free_sample_fraction",
        "continuous_unsafe_segment_fraction",
        "path_length_ratio_to_hybrid",
        "max_abs_curvature_rate_1pm2",
        "min_corridor_margin_m",
    )
    result = {
        "sample_count": int(sample_count),
        "groups": list(GROUPS),
        "search": {
            "success_count": int(sum(row["success"] for row in search_records)),
            "failure_count": int(sum(not row["success"] for row in search_records)),
            "metrics": {
                key: numeric_summary(search_records, key)
                for key in (
                    "time_hybrid_astar_s",
                    "path_length_m",
                    "dense_pose_count",
                    "expanded_nodes",
                    "generated_nodes",
                )
            },
        },
        "results": {},
    }
    for group in GROUPS:
        rows = [row for row in records if row["group"] == group]
        generated = [
            row
            for row in rows
            if group == "B_hybrid_raw" or row.get("optimization_success", False)
        ]
        metric_rows = [
            {**row, "success": True}
            for row in generated
        ]
        entry = {
            "record_count": int(len(rows)),
            "success_count": int(sum(row["success"] for row in rows)),
            "success_rate_over_all_trials": float(
                sum(row["success"] for row in rows) / sample_count
            ),
            "generated_count": int(len(generated)),
            "search_success_count": int(
                sum(row.get("search_success", False) for row in rows)
            ),
            "corridor_success_count": int(
                sum(row.get("corridor_success") is True for row in rows)
            ),
            "optimization_success_count": int(
                sum(row.get("optimization_success") is True for row in rows)
            ),
            "grid_safe_count": int(
                sum(row.get("grid_safe", False) for row in rows)
            ),
            "continuous_safe_count": int(
                sum(row.get("continuous_safe", False) for row in rows)
            ),
            "curvature_feasible_count": int(
                sum(row.get("curvature_feasible", False) for row in rows)
            ),
            "curvature_rate_feasible_count": int(
                sum(row.get("curvature_rate_feasible", False) for row in rows)
            ),
            "tracking_success_count": int(
                sum(row.get("tracking_success") is True for row in rows)
            ),
            "fallback_count": int(
                sum(row.get("fallback_used", False) for row in rows)
            ),
            "metrics": {key: numeric_summary(metric_rows, key) for key in metrics},
        }
        result["results"][group] = entry
    return result


def save_csv(path: Path, records: list[dict]) -> None:
    fields = sorted({key for row in records for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def plot_representatives(
    output_dir: Path,
    selected: list[dict],
    artifacts: dict[tuple[str, str], dict],
    grid: GroundGrid,
    extent: list[float],
) -> None:
    representative_ids = {}
    for layer in ("near", "medium", "far"):
        rows = [row for row in selected if row["distance_layer"] == layer]
        representative_ids[layer] = rows[len(rows) // 2]["trial_id"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    occupied = grid.occupied.detach().cpu().numpy().T.astype(float)
    for row_index, layer in enumerate(("near", "medium", "far")):
        trial_id = representative_ids[layer]
        for column_index, group in enumerate(GROUPS):
            axis = axes[row_index, column_index]
            axis.imshow(
                occupied,
                origin="lower",
                cmap="binary",
                extent=extent,
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            artifact = artifacts.get((trial_id, group))
            if artifact is not None:
                axis.plot(
                    artifact["hybrid"][:, 0],
                    artifact["hybrid"][:, 1],
                    color="tab:blue",
                    linewidth=1.0,
                    label="Hybrid A*",
                )
                if "curve" in artifact:
                    axis.plot(
                        artifact["curve"][:, 0],
                        artifact["curve"][:, 1],
                        color="tab:red",
                        linewidth=1.5,
                        label=(
                            "Hybrid fallback"
                            if artifact.get("fallback", False)
                            else (
                                "Bezier"
                                if group == "C_corridor_bezier"
                                else "Curvature trajectory"
                            )
                        ),
                    )
                axis.legend(fontsize=7)
            configure_map_axis(axis, extent, f"{layer} / {group}")
    fig.tight_layout()
    fig.savefig(output_dir / "representative_paths.png", dpi=180)
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
    parser.add_argument("--corridor-bezier-samples-per-section", type=int, default=100)
    parser.add_argument("--max-curvature-rate-1pm2", type=float, default=10.0)
    parser.add_argument("--trajectory-optimizer-nodes", type=int, default=32)
    parser.add_argument("--trajectory-optimizer-max-iterations", type=int, default=500)
    parser.add_argument("--trajectory-optimizer-max-refinements", type=int, default=2)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.selected_goals_json.read_text(encoding="utf-8"))
    selected = select_manifest_rows(manifest, args.sample_count)
    save_json(args.output_dir / "selected_goals.json", selected)

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
    hybrid = HybridAStarPlanner(
        grid,
        scene_scale=scale,
        config=HybridAStarConfig(
            heading_bins=args.hybrid_heading_bins,
            min_turning_radius_m=args.min_turning_radius_meters,
            max_expansions=args.hybrid_max_expansions,
        ),
    )

    search_records = []
    records = []
    artifacts = {}
    search_path = args.output_dir / "search_records.jsonl"
    record_path = args.output_dir / "group_records.jsonl"
    representative_ids = {
        rows[len(rows) // 2]["trial_id"]
        for layer in ("near", "medium", "far")
        for rows in [[row for row in selected if row["distance_layer"] == layer]]
    }

    for index, trial in enumerate(selected):
        print(f"[{index + 1:03d}/{len(selected):03d}] {trial['trial_id']}", flush=True)
        search_record = {
            "trial_id": trial["trial_id"],
            "candidate_id": trial["candidate_id"],
            "distance_layer": trial["distance_layer"],
            "goal_heading_mode": "free",
            "requested_goal_yaw_rad": None,
            "success": False,
            "search_success": False,
            "failure_reason": None,
        }
        hybrid_path = None
        try:
            begin = time.perf_counter()
            hybrid_path = hybrid.plan(
                [*trial["start_scene"], trial["start_yaw_rad"]],
                trial["goal_scene"],
            )
            search_record["time_hybrid_astar_s"] = time.perf_counter() - begin
            search_record.update(hybrid_path.summary())
            search_record["success"] = True
            search_record["search_success"] = True
        except Exception as error:
            search_record["failure_reason"] = str(error)
        search_records.append(search_record)
        append_jsonl(search_path, search_record)
        if hybrid_path is None:
            print(json.dumps({"search_success": False, "failure": search_record["failure_reason"]}), flush=True)
            continue

        b_record = raw_record(
            trial,
            hybrid_path,
            grid,
            collision_set,
            scale,
            search_record["time_hybrid_astar_s"],
            args.min_turning_radius_meters,
        )
        c_record, c_artifact = corridor_bezier_record(
            trial,
            hybrid_path,
            grid,
            collision_set,
            scale,
            args.min_turning_radius_meters,
            args.corridor_bezier_samples_per_section,
            device,
        )
        d_record, d_artifact = curvature_optimizer_record(
            trial,
            hybrid_path,
            grid,
            collision_set,
            scale,
            args.min_turning_radius_meters,
            args.max_curvature_rate_1pm2,
            args.trajectory_optimizer_nodes,
            args.trajectory_optimizer_max_iterations,
            args.trajectory_optimizer_max_refinements,
            device,
        )
        trial_records = (b_record, c_record, d_record)
        for record in trial_records:
            records.append(record)
            append_jsonl(record_path, record)
        if trial["trial_id"] in representative_ids:
            artifacts[(trial["trial_id"], "B_hybrid_raw")] = {
                "hybrid": hybrid_path.xy,
            }
            if c_artifact is not None:
                artifacts[(trial["trial_id"], "C_corridor_bezier")] = c_artifact
            if d_artifact is not None:
                artifacts[(trial["trial_id"], "D_curvature_optimizer")] = d_artifact
        print(
            json.dumps(
                {
                    "B": b_record["success"],
                    "C_generated": c_record["success"],
                    "C_collision_free": c_record.get("collision_free"),
                    "C_outside_fraction": c_record.get("outside_free_sample_fraction"),
                    "D_optimized": d_record.get("optimization_success"),
                    "D_curvature_feasible": d_record.get("curvature_feasible"),
                    "D_max_curvature": d_record.get("max_abs_curvature_1pm"),
                    "D_max_curvature_rate": d_record.get(
                        "max_abs_curvature_rate_1pm2"
                    ),
                }
            ),
            flush=True,
        )

    summary = summary_for_records(search_records, records, args.sample_count)
    summary.update(
        {
            "scene": args.scene,
            "device": str(device),
            "scale": scale,
            "preprocessing": preprocessing,
            "protocol": {
                "selected_goals_json": str(args.selected_goals_json),
                "sample_count": args.sample_count,
                "distance_layers": ["near", "medium", "far"],
                "trials_per_layer": args.sample_count // 3,
                "goal_heading_mode": "free",
                "corridor_bezier": {
                    "type": "degree-6 Bezier QP in ordered GS corridors",
                    "samples_per_section": args.corridor_bezier_samples_per_section,
                    "start_heading": "fixed",
                    "terminal_heading": "free",
                },
                "full_method": (
                    "Hybrid A* -> ordered GS safety corridor -> arc-length "
                    "direct collocation with curvature and curvature-rate constraints"
                ),
                "curvature_limit_1pm": 0.95 / args.min_turning_radius_meters,
                "max_curvature_rate_1pm2": args.max_curvature_rate_1pm2,
                "trajectory_optimizer_nodes": args.trajectory_optimizer_nodes,
                "trajectory_optimizer_max_iterations": (
                    args.trajectory_optimizer_max_iterations
                ),
                "trajectory_optimizer_max_refinements": (
                    args.trajectory_optimizer_max_refinements
                ),
            },
        }
    )
    save_json(args.output_dir / "summary.json", summary)
    save_json(
        args.output_dir / "failures.json",
        {
            "search": [row for row in search_records if not row["success"]],
            "groups": [
                row
                for row in records
                if row.get("failure_reason") is not None
            ],
        },
    )
    save_csv(args.output_dir / "group_records.csv", records)
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
    plot_representatives(args.output_dir, selected, artifacts, grid, extent)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
