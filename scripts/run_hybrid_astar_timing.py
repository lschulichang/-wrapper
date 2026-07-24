#!/usr/bin/env python3
"""Benchmark Hybrid A* only on balanced fixed-start/free-goal trials."""

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


LAYERS = ("near", "medium", "far")
LAYER_COLORS = {
    "near": "#4C78A8",
    "medium": "#F58518",
    "far": "#E45756",
}


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
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def correlation(records: list[dict], key: str) -> float | None:
    pairs = [
        (float(row[key]), float(row["hybrid_astar_time_s"]))
        for row in records
        if row.get("search_success")
        and row.get(key) is not None
        and np.isfinite(float(row[key]))
    ]
    if len(pairs) < 2:
        return None
    values = np.asarray(pairs, dtype=np.float64)
    if np.std(values[:, 0]) <= 1e-12:
        return None
    return float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])


def save_visualizations(
    records: list[dict], output_dir: Path
) -> list[str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    successful = [
        row for row in records if row.get("search_success")
    ]
    generated = []

    figure, axis = plt.subplots(figsize=(12.0, 5.5))
    positions = np.arange(len(records))
    times = [
        (
            float(row["hybrid_astar_time_s"])
            if row.get("search_success")
            else 0.0
        )
        for row in records
    ]
    colors = [
        (
            LAYER_COLORS[row["distance_layer"]]
            if row.get("search_success")
            else "#9D9D9D"
        )
        for row in records
    ]
    axis.bar(positions, times, color=colors)
    if successful:
        median = float(
            np.median(
                [
                    row["hybrid_astar_time_s"]
                    for row in successful
                ]
            )
        )
        axis.axhline(
            median,
            color="#222222",
            linestyle="--",
            linewidth=1.0,
            label=f"median = {median:.3f} s",
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [row["trial_id"] for row in records],
        rotation=70,
        ha="right",
        fontsize=7,
    )
    axis.set_ylabel("Hybrid A* time (s)")
    axis.set_title("Hybrid A* wall-clock time by trial")
    axis.grid(True, axis="y", alpha=0.25)
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, color=LAYER_COLORS[layer]
        )
        for layer in LAYERS
    ]
    labels = list(LAYERS)
    if successful:
        line_handle, = axis.plot(
            [], [], color="#222222", linestyle="--"
        )
        handles.append(line_handle)
        labels.append(f"median = {median:.3f} s")
    axis.legend(handles, labels, loc="best")
    figure.tight_layout()
    path = figure_dir / "hybrid_timing_by_trial.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    generated.append(str(path.relative_to(output_dir)))

    figure, axes = plt.subplots(
        1, 2, figsize=(11.0, 5.0)
    )
    layer_values = [
        [
            row["hybrid_astar_time_s"]
            for row in successful
            if row["distance_layer"] == layer
        ]
        for layer in LAYERS
    ]
    all_values = [
        row["hybrid_astar_time_s"] for row in successful
    ]
    box = axes[0].boxplot(
        [*layer_values, all_values],
        tick_labels=[*LAYERS, "all"],
        patch_artist=True,
        showmeans=True,
    )
    for patch, color in zip(
        box["boxes"],
        [
            *(LAYER_COLORS[layer] for layer in LAYERS),
            "#72B7B2",
        ],
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.70)
    axes[0].set_ylabel("Hybrid A* time (s)")
    axes[0].set_title("Timing distribution")
    axes[0].grid(True, axis="y", alpha=0.25)
    for layer in LAYERS:
        values = [
            row["hybrid_astar_time_s"]
            for row in successful
            if row["distance_layer"] == layer
        ]
        axes[1].hist(
            values,
            bins=max(3, min(8, len(values))),
            alpha=0.45,
            color=LAYER_COLORS[layer],
            label=layer,
        )
    axes[1].set_xlabel("Hybrid A* time (s)")
    axes[1].set_ylabel("Trial count")
    axes[1].set_title("Timing histogram")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    path = figure_dir / "hybrid_timing_distribution.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    generated.append(str(path.relative_to(output_dir)))

    figure, axes = plt.subplots(
        1, 2, figsize=(11.5, 5.0)
    )
    for layer in LAYERS:
        rows = [
            row
            for row in successful
            if row["distance_layer"] == layer
        ]
        axes[0].scatter(
            [row["path_length_m"] for row in rows],
            [row["hybrid_astar_time_s"] for row in rows],
            color=LAYER_COLORS[layer],
            label=layer,
            s=38,
            alpha=0.85,
        )
        axes[1].scatter(
            [row["expanded_nodes"] for row in rows],
            [row["hybrid_astar_time_s"] for row in rows],
            color=LAYER_COLORS[layer],
            label=layer,
            s=38,
            alpha=0.85,
        )
    path_corr = correlation(successful, "path_length_m")
    node_corr = correlation(successful, "expanded_nodes")
    axes[0].set_xlabel("Hybrid path length (m)")
    axes[0].set_ylabel("Hybrid A* time (s)")
    axes[0].set_title(
        "Time vs path length"
        + (
            ""
            if path_corr is None
            else f"  (r={path_corr:.2f})"
        )
    )
    axes[1].set_xlabel("Expanded nodes")
    axes[1].set_ylabel("Hybrid A* time (s)")
    axes[1].set_title(
        "Time vs expanded nodes"
        + (
            ""
            if node_corr is None
            else f"  (r={node_corr:.2f})"
        )
    )
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.tight_layout()
    path = figure_dir / "hybrid_timing_scaling.png"
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
    parser.add_argument(
        "--scene",
        default="old_union",
        choices=sorted(SCENE_PRESETS),
    )
    parser.add_argument("--sample-count", type=int, default=30)
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        args.selected_goals_json.read_text(encoding="utf-8")
    )
    selected = select_manifest_rows(
        manifest, args.sample_count
    )
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

    records = []
    record_path = args.output_dir / "records.jsonl"
    path_dir = args.output_dir / "hybrid_paths"
    path_dir.mkdir(parents=True, exist_ok=True)
    loop_begin = time.perf_counter()
    for index, trial in enumerate(selected):
        print(
            f"[{index + 1:03d}/{len(selected):03d}] "
            f"{trial['trial_id']} {trial['distance_layer']}",
            flush=True,
        )
        record = {
            "trial_id": trial["trial_id"],
            "candidate_id": trial["candidate_id"],
            "distance_layer": trial["distance_layer"],
            "start_scene": trial["start_scene"],
            "start_yaw_rad": trial["start_yaw_rad"],
            "goal_scene": trial["goal_scene"],
            "direct_distance_m": trial["direct_distance_m"],
            "reference_path_length_m": trial[
                "reference_path_length_m"
            ],
            "search_success": False,
            "hybrid_astar_time_s": None,
            "failure_reason": None,
        }
        try:
            begin = time.perf_counter()
            path = planner.plan(
                [
                    *trial["start_scene"],
                    trial["start_yaw_rad"],
                ],
                trial["goal_scene"],
            )
            elapsed = time.perf_counter() - begin
            summary = path.summary()
            record.update(
                {
                    "search_success": True,
                    "hybrid_astar_time_s": float(elapsed),
                    "path_length_m": summary["path_length_m"],
                    "dense_pose_count": summary[
                        "dense_pose_count"
                    ],
                    "primitive_count": summary[
                        "primitive_count"
                    ],
                    "expanded_nodes": summary[
                        "expanded_nodes"
                    ],
                    "generated_nodes": summary[
                        "generated_nodes"
                    ],
                    "terminal_yaw_rad": summary[
                        "terminal_yaw_rad"
                    ],
                }
            )
            np.save(
                path_dir / f"{trial['trial_id']}.npy",
                path.poses_scene,
            )
        except Exception as error:
            record["failure_reason"] = str(error)
        records.append(record)
        append_jsonl(record_path, record)
        print(
            json.dumps(
                {
                    "success": record["search_success"],
                    "time_s": record["hybrid_astar_time_s"],
                    "expanded_nodes": record.get(
                        "expanded_nodes"
                    ),
                    "path_length_m": record.get(
                        "path_length_m"
                    ),
                    "failure_reason": record["failure_reason"],
                }
            ),
            flush=True,
        )
    loop_wall_time_s = time.perf_counter() - loop_begin

    successful = [
        row for row in records if row["search_success"]
    ]
    start_signatures = {
        (
            *map(float, row["start_scene"]),
            float(row["start_yaw_rad"]),
        )
        for row in records
    }
    visualizations = save_visualizations(
        records, args.output_dir
    )
    summary = {
        "scene": args.scene,
        "device": str(device),
        "scene_scale": scale,
        "sample_count": int(args.sample_count),
        "search_success_count": int(len(successful)),
        "search_failure_count": int(
            len(records) - len(successful)
        ),
        "success_rate": float(
            len(successful) / len(records)
        ),
        "hybrid_astar_time_s": numeric_summary(
            [
                row["hybrid_astar_time_s"]
                for row in successful
            ]
        ),
        "time_by_distance_layer_s": {
            layer: numeric_summary(
                [
                    row["hybrid_astar_time_s"]
                    for row in successful
                    if row["distance_layer"] == layer
                ]
            )
            for layer in LAYERS
        },
        "path_length_m": numeric_summary(
            [row.get("path_length_m") for row in successful]
        ),
        "expanded_nodes": numeric_summary(
            [row.get("expanded_nodes") for row in successful]
        ),
        "correlation": {
            "time_vs_path_length": correlation(
                successful, "path_length_m"
            ),
            "time_vs_expanded_nodes": correlation(
                successful, "expanded_nodes"
            ),
        },
        "loop_wall_time_s": float(loop_wall_time_s),
        "preprocessing": preprocessing,
        "visualizations": visualizations,
        "protocol": {
            "selected_goals_json": str(
                args.selected_goals_json
            ),
            "balanced_distance_layers": {
                layer: int(
                    sum(
                        row["distance_layer"] == layer
                        for row in records
                    )
                )
                for layer in LAYERS
            },
            "fixed_start_pose": len(start_signatures) == 1,
            "goal_heading_mode": "free",
            "stops_after": "Hybrid_A_star",
            "safety_corridor_run": False,
            "trajectory_optimization_run": False,
            "hybrid_astar_timer_excludes_preprocessing": True,
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
