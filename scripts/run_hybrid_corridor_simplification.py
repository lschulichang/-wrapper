#!/usr/bin/env python3
"""Compare corridor counts through primitive-aware Hybrid path simplification."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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
    build_planar_corridor,
    compute_stopping_distance,
    simplify_hybrid_path_by_primitives,
)
from smoke_splatplan import SCENE_PRESETS  # noqa: E402
from splat.splat_utils import GSplatLoader  # noqa: E402


LAYERS = ("near", "medium", "far")
CORRIDOR_VARIANTS = (
    ("dense", "poses_scene"),
    ("primitive_boundaries", "boundary_poses_scene"),
    ("merged_curvature_runs", "merged_poses_scene"),
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
        handle.write(
            json.dumps(value, ensure_ascii=False) + "\n"
        )


def select_manifest_rows(
    rows: list[dict], sample_count: int
) -> list[dict]:
    if sample_count <= 0 or sample_count % len(LAYERS):
        raise ValueError(
            "sample_count must be a positive multiple of three"
        )
    selected = []
    per_layer = sample_count // len(LAYERS)
    for layer in LAYERS:
        layer_rows = [
            row for row in rows if row["distance_layer"] == layer
        ]
        if len(layer_rows) < per_layer:
            raise ValueError(
                f"manifest has only {len(layer_rows)} {layer} rows"
            )
        indices = np.linspace(
            0, len(layer_rows) - 1, per_layer
        ).round().astype(int)
        selected.extend(
            dict(layer_rows[index]) for index in indices
        )
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
    }


def build_corridor_record(
    path: np.ndarray,
    collision_set: PlanarCollisionSet,
    device: torch.device,
) -> tuple[dict, object | None]:
    synchronize(device)
    begin = time.perf_counter()
    try:
        corridor = build_planar_corridor(path, collision_set)
        synchronize(device)
        return (
            {
                "success": True,
                "path_pose_count": int(len(path)),
                "corridor_count": int(len(corridor.corridors)),
                "overlap_count": int(len(corridor.overlaps)),
                "time_s": float(time.perf_counter() - begin),
                "failure_reason": None,
            },
            corridor,
        )
    except Exception as error:
        synchronize(device)
        return (
            {
                "success": False,
                "path_pose_count": int(len(path)),
                "corridor_count": None,
                "overlap_count": None,
                "time_s": float(time.perf_counter() - begin),
                "failure_reason": str(error),
            },
            None,
        )


def build_summary(records: list[dict], sample_count: int) -> dict:
    searched = [row for row in records if row["search_success"]]
    result = {
        "sample_count": int(sample_count),
        "search_success_count": int(len(searched)),
        "search_failure_count": int(sample_count - len(searched)),
        "simplification": {
            key: numeric_summary(searched, key)
            for key in (
                "primitive_simplification_time_s",
                "dense_pose_count",
                "primitive_count",
                "primitive_boundary_pose_count",
                "curvature_run_count",
                "merged_pose_count",
            )
        },
        "corridors": {},
    }
    for variant, _ in CORRIDOR_VARIANTS:
        success_key = f"{variant}_corridor_success"
        count_key = f"{variant}_corridor_count"
        time_key = f"{variant}_corridor_time_s"
        result["corridors"][variant] = {
            "success_count": int(
                sum(row.get(success_key, False) for row in searched)
            ),
            "failure_count": int(
                sum(
                    not row.get(success_key, False)
                    for row in searched
                )
            ),
            "corridor_count": numeric_summary(
                searched, count_key
            ),
            "time_s": numeric_summary(searched, time_key),
        }

    paired = [
        row
        for row in searched
        if all(
            row.get(f"{variant}_corridor_success", False)
            for variant, _ in CORRIDOR_VARIANTS
        )
    ]
    result["paired_all_corridors_success_count"] = len(paired)
    result["corridor_count_change"] = {
        "dense_to_primitive_boundaries": numeric_summary(
            paired,
            "dense_to_primitive_boundary_corridor_change",
        ),
        "primitive_boundaries_to_merged_curvature_runs": (
            numeric_summary(
                paired,
                "primitive_boundary_to_merged_corridor_change",
            )
        ),
        "dense_to_merged_curvature_runs": numeric_summary(
            paired,
            "dense_to_merged_corridor_change",
        ),
        "dense_to_merged_reduction_fraction": numeric_summary(
            paired,
            "dense_to_merged_corridor_reduction_fraction",
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--selected-goals-json", required=True, type=Path
    )
    parser.add_argument(
        "--scene",
        default="old_union",
        choices=sorted(SCENE_PRESETS),
    )
    parser.add_argument("--sample-count", type=int, default=3)
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
    parser.add_argument("--max-speed-mps", type=float, default=0.20)
    parser.add_argument(
        "--max-brake-decel-mps2", type=float, default=0.30
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
    parser.add_argument(
        "--curvature-tolerance-1pm", type=float, default=1e-9
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        args.selected_goals_json.read_text(encoding="utf-8")
    )
    selected = select_manifest_rows(manifest, args.sample_count)
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
            min_turning_radius_m=(
                args.min_turning_radius_meters
            ),
            max_expansions=args.hybrid_max_expansions,
        ),
    )

    records = []
    record_path = args.output_dir / "records.jsonl"
    for index, trial in enumerate(selected):
        print(
            f"[{index + 1:03d}/{len(selected):03d}] "
            f"{trial['trial_id']}",
            flush=True,
        )
        record = {
            "trial_id": trial["trial_id"],
            "candidate_id": trial["candidate_id"],
            "distance_layer": trial["distance_layer"],
            "search_success": False,
            "failure_reason": None,
        }
        try:
            begin = time.perf_counter()
            hybrid_path = hybrid.plan(
                [
                    *trial["start_scene"],
                    trial["start_yaw_rad"],
                ],
                trial["goal_scene"],
            )
            record["hybrid_astar_time_s"] = (
                time.perf_counter() - begin
            )
            record["search_success"] = True
            begin = time.perf_counter()
            simplification = simplify_hybrid_path_by_primitives(
                hybrid_path,
                curvature_tolerance_1pm=(
                    args.curvature_tolerance_1pm
                ),
            )
            record["primitive_simplification_time_s"] = (
                time.perf_counter() - begin
            )
            record.update(simplification.summary())
            paths = {
                "dense": hybrid_path.poses_scene,
                "primitive_boundaries": (
                    simplification.boundary_poses_scene
                ),
                "merged_curvature_runs": (
                    simplification.merged_poses_scene
                ),
            }
            for variant, _ in CORRIDOR_VARIANTS:
                corridor_record, _ = build_corridor_record(
                    paths[variant], collision_set, device
                )
                record[f"{variant}_corridor_success"] = (
                    corridor_record["success"]
                )
                record[f"{variant}_corridor_count"] = (
                    corridor_record["corridor_count"]
                )
                record[f"{variant}_corridor_time_s"] = (
                    corridor_record["time_s"]
                )
                record[f"{variant}_corridor_failure_reason"] = (
                    corridor_record["failure_reason"]
                )

            if all(
                record[f"{variant}_corridor_success"]
                for variant, _ in CORRIDOR_VARIANTS
            ):
                dense_count = record["dense_corridor_count"]
                boundary_count = record[
                    "primitive_boundaries_corridor_count"
                ]
                merged_count = record[
                    "merged_curvature_runs_corridor_count"
                ]
                record[
                    "dense_to_primitive_boundary_corridor_change"
                ] = boundary_count - dense_count
                record[
                    "primitive_boundary_to_merged_corridor_change"
                ] = merged_count - boundary_count
                record["dense_to_merged_corridor_change"] = (
                    merged_count - dense_count
                )
                record[
                    "dense_to_merged_corridor_reduction_fraction"
                ] = (
                    (dense_count - merged_count) / dense_count
                    if dense_count
                    else 0.0
                )
        except Exception as error:
            record["failure_reason"] = str(error)
        records.append(record)
        append_jsonl(record_path, record)
        print(
            json.dumps(
                {
                    "search": record["search_success"],
                    "dense_poses": record.get("dense_pose_count"),
                    "primitive_boundaries": record.get(
                        "primitive_boundary_pose_count"
                    ),
                    "merged_poses": record.get(
                        "merged_pose_count"
                    ),
                    "dense_corridors": record.get(
                        "dense_corridor_count"
                    ),
                    "boundary_corridors": record.get(
                        "primitive_boundaries_corridor_count"
                    ),
                    "merged_corridors": record.get(
                        "merged_curvature_runs_corridor_count"
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
            "scene_scale": scale,
            "preprocessing": preprocessing,
            "protocol": {
                "selected_goals_json": str(
                    args.selected_goals_json
                ),
                "sample_count": args.sample_count,
                "goal_heading_mode": "free",
                "stops_after": "ordered_GS_safety_corridor",
                "trajectory_optimization_run": False,
                "simplification": [
                    "replace dense samples with exact motion-primitive boundaries",
                    "merge maximal consecutive equal-curvature primitive runs",
                ],
                "curvature_tolerance_1pm": (
                    args.curvature_tolerance_1pm
                ),
            },
        }
    )
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
