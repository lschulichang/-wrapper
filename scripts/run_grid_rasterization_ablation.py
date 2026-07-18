#!/usr/bin/env python3
"""Paired 30-case comparison of two projected-Gaussian grid rasterizers."""

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
import scipy.ndimage
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
from ground_nav.path_utils import simplify_ground_path_supercover  # noqa: E402
from ground_nav.planar_gaussians import PlanarGaussianSet  # noqa: E402
from smoke_ground_corridor_2d import (  # noqa: E402
    configure_map_axis,
    max_control_violation,
)
from smoke_splatplan import SCENE_PRESETS  # noqa: E402
from splat.splat_utils import GSplatLoader  # noqa: E402


METHODS = ("ellipse_distance", "aabb_disk_dilation")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def save_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def path_length_m(path: np.ndarray, scale: float) -> float:
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum() / scale)


def path_turn_count(path: np.ndarray) -> int:
    if len(path) < 3:
        return 0
    directions = np.sign(np.diff(path, axis=0)).astype(np.int8)
    return int(
        np.count_nonzero(
            np.any(directions[1:] != directions[:-1], axis=1)
        )
    )


def endpoint_indices(
    ellipse_grid: GroundGrid,
    aabb_grid: GroundGrid,
    scale: float,
    clearance_m: float,
) -> tuple[np.ndarray, dict]:
    ellipse_free = ellipse_grid.free.detach().cpu().numpy().astype(bool)
    aabb_free = aabb_grid.free.detach().cpu().numpy().astype(bool)
    structure = np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
    )
    labels, component_count = scipy.ndimage.label(
        ellipse_free, structure=structure
    )
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    if len(sizes) <= 1 or int(sizes.max()) == 0:
        raise RuntimeError("ellipse-distance grid has no free component")
    largest_label = int(np.argmax(sizes))
    common_free = ellipse_free & aabb_free
    sampling_m = ellipse_grid.cell_sizes.detach().cpu().numpy() / scale
    clearance = scipy.ndimage.distance_transform_edt(
        common_free, sampling=sampling_m
    )
    eligible = (
        (labels == largest_label)
        & common_free
        & (clearance >= clearance_m)
    )
    eligible[[0, -1], :] = False
    eligible[:, [0, -1]] = False
    indices = np.argwhere(eligible)
    if len(indices) < 60:
        raise RuntimeError(
            f"only {len(indices)} cells are valid endpoints on both maps"
        )
    return indices, {
        "ellipse_component_count": int(component_count),
        "ellipse_largest_component_cells": int(sizes[largest_label]),
        "common_free_cells": int(common_free.sum()),
        "eligible_common_endpoint_cells": int(len(indices)),
        "endpoint_clearance_m": float(clearance_m),
    }


def generate_candidates(
    grid: GroundGrid,
    indices: np.ndarray,
    scale: float,
    count: int,
    min_distance_m: float,
    seed: int,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    candidates: list[dict] = []
    seen: set[tuple] = set()
    attempts = 0
    maximum_attempts = max(10000, count * 1000)
    while len(candidates) < count and attempts < maximum_attempts:
        attempts += 1
        chosen = rng.choice(len(indices), size=2, replace=False)
        first = tuple(int(v) for v in indices[chosen[0]])
        second = tuple(int(v) for v in indices[chosen[1]])
        pair_key = tuple(sorted((first, second)))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        start = grid.grid_to_world(first).detach().cpu().numpy()
        goal = grid.grid_to_world(second).detach().cpu().numpy()
        direct_m = float(np.linalg.norm(goal - start) / scale)
        if direct_m < min_distance_m:
            continue
        try:
            path = grid.create_path(start, goal)
        except (RuntimeError, ValueError):
            continue
        candidates.append(
            {
                "candidate_id": f"candidate_{len(candidates):04d}",
                "start_grid": list(first),
                "goal_grid": list(second),
                "start_scene": start.tolist(),
                "goal_scene": goal.tolist(),
                "start_meters": (start / scale).tolist(),
                "goal_meters": (goal / scale).tolist(),
                "reference_path": path,
                "direct_distance_m": direct_m,
                "reference_path_length_m": path_length_m(path, scale),
                "reference_path_points": int(len(path)),
                "reference_turn_count": path_turn_count(path),
            }
        )
    if len(candidates) != count:
        raise RuntimeError(
            f"generated only {len(candidates)} of {count} candidates"
        )
    return candidates, {
        "seed": int(seed),
        "sampling_attempts": int(attempts),
        "candidate_count": int(len(candidates)),
        "minimum_endpoint_distance_m": float(min_distance_m),
    }


def select_stratified(candidates: list[dict], sample_count: int) -> list[dict]:
    if sample_count % 3:
        raise ValueError("sample_count must be divisible by three")
    ordered = sorted(
        candidates,
        key=lambda row: (
            row["reference_turn_count"],
            row["reference_path_length_m"],
        ),
    )
    layers = np.array_split(np.asarray(ordered, dtype=object), 3)
    selected: list[dict] = []
    per_layer = sample_count // 3
    for label, layer in zip(("low", "medium", "high"), layers):
        values = sorted(
            layer.tolist(), key=lambda row: row["reference_path_length_m"]
        )
        used: set[str] = set()
        for quantile in np.linspace(0.05, 0.95, per_layer):
            desired = quantile * max(len(values) - 1, 1)
            ranks = sorted(
                range(len(values)),
                key=lambda index: (abs(index - desired), index),
            )
            for rank in ranks:
                candidate = values[rank]
                if candidate["candidate_id"] in used:
                    continue
                chosen = dict(candidate)
                chosen["complexity_layer"] = label
                chosen["selection_quantile"] = float(quantile)
                selected.append(chosen)
                used.add(candidate["candidate_id"])
                break
    if len(selected) != sample_count:
        raise RuntimeError(
            f"selected only {len(selected)} of {sample_count} pairs"
        )
    for index, row in enumerate(selected):
        row["trial_id"] = f"trial_{index:03d}"
    return selected


def run_method(
    candidate: dict,
    method: str,
    grid: GroundGrid,
    collision_set: PlanarCollisionSet,
    scale: float,
    max_segment_scene: float,
    dense_samples: int,
    device: torch.device,
) -> tuple[dict, dict | None]:
    record = {
        "trial_id": candidate["trial_id"],
        "candidate_id": candidate["candidate_id"],
        "complexity_layer": candidate["complexity_layer"],
        "method": method,
        "success": False,
        "failure_stage": None,
        "failure_reason": None,
        "direct_distance_m": candidate["direct_distance_m"],
        "reference_turn_count": candidate["reference_turn_count"],
    }
    artifact = None
    stage = "dijkstra"
    start = np.asarray(candidate["start_scene"], dtype=np.float32)
    goal = np.asarray(candidate["goal_scene"], dtype=np.float32)
    try:
        sync(device)
        begin = time.perf_counter()
        raw = grid.create_path(start, goal)
        sync(device)
        record["time_dijkstra_s"] = time.perf_counter() - begin
        record["raw_path_points"] = int(len(raw))
        record["raw_path_length_m"] = path_length_m(raw, scale)
        record["raw_turn_count"] = path_turn_count(raw)

        stage = "simplification"
        begin = time.perf_counter()
        simplified = simplify_ground_path_supercover(
            raw, grid, max_segment_scene
        )
        record["time_simplification_s"] = time.perf_counter() - begin
        record["simplified_points"] = int(len(simplified))
        record["simplified_path_length_m"] = path_length_m(
            simplified, scale
        )

        stage = "corridor"
        sync(device)
        begin = time.perf_counter()
        corridor = build_planar_corridor(simplified, collision_set)
        sync(device)
        record["time_corridor_s"] = time.perf_counter() - begin
        record["polygon_count"] = int(len(corridor.polygons))

        stage = "bezier_qp"
        planner = BezierPlanner2D(degree=6, continuity_order=3)
        begin = time.perf_counter()
        controls, feasible = planner.optimize(
            corridor.polygons, simplified[0], simplified[-1]
        )
        record["time_qp_s"] = time.perf_counter() - begin
        record["qp_status"] = planner.last_solver_status
        if not feasible:
            raise RuntimeError(f"QP status {planner.last_solver_status}")
        dense = planner.sample(dense_samples)
        record["max_control_point_violation"] = max_control_violation(
            controls, corridor.polygons
        )
        endpoint_error = max(
            np.linalg.norm(dense[0, 0] - simplified[0]),
            np.linalg.norm(dense[-1, -1] - simplified[-1]),
        )
        record["max_endpoint_error_m"] = float(endpoint_error / scale)
        if (
            record["max_control_point_violation"] > 1e-5
            or record["max_endpoint_error_m"] > 1e-6
        ):
            raise RuntimeError("Bezier control-point verification failed")
        record["bezier_length_m"] = float(
            sum(
                np.linalg.norm(np.diff(section, axis=0), axis=1).sum()
                for section in dense
            )
            / scale
        )
        record["time_pipeline_s"] = float(
            record["time_dijkstra_s"]
            + record["time_simplification_s"]
            + record["time_corridor_s"]
            + record["time_qp_s"]
        )
        record["success"] = True
        artifact = {
            "raw": raw,
            "simplified": simplified,
            "dense": dense,
        }
    except Exception as error:
        record["failure_stage"] = stage
        record["failure_reason"] = str(error)
        record["time_pipeline_s"] = float(
            sum(
                record.get(name, 0.0)
                for name in (
                    "time_dijkstra_s",
                    "time_simplification_s",
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
    if len(values) == 0:
        return None
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def build_summary(
    records: list[dict],
    grids: dict[str, GroundGrid],
    raster_times: dict[str, float],
    sample_count: int,
) -> dict:
    summary = {"sample_count": int(sample_count), "methods": {}}
    metric_keys = (
        "time_dijkstra_s",
        "time_pipeline_s",
        "raw_path_length_m",
        "simplified_path_length_m",
        "bezier_length_m",
        "raw_path_points",
        "simplified_points",
        "polygon_count",
    )
    for method in METHODS:
        rows = [row for row in records if row["method"] == method]
        successes = [row for row in rows if row["success"]]
        grid = grids[method]
        summary["methods"][method] = {
            "rasterization": grid.metadata.rasterization,
            "rasterization_time_s": float(raster_times[method]),
            "raw_occupied_cells": int(grid.raw_occupied.sum().item()),
            "occupied_cells": int(grid.occupied.sum().item()),
            "free_cells": int(grid.free.sum().item()),
            "success_count": int(len(successes)),
            "failure_count": int(len(rows) - len(successes)),
            "success_rate": float(len(successes) / max(len(rows), 1)),
            "failure_stages": {
                stage: int(
                    sum(row.get("failure_stage") == stage for row in rows)
                )
                for stage in sorted(
                    {
                        row.get("failure_stage")
                        for row in rows
                        if row.get("failure_stage")
                    }
                )
            },
            "metrics": {
                key: numeric_summary(rows, key) for key in metric_keys
            },
        }
    paired = []
    by_trial: dict[str, dict[str, dict]] = {}
    for row in records:
        by_trial.setdefault(row["trial_id"], {})[row["method"]] = row
    for trial_id, methods in by_trial.items():
        if all(
            method in methods and methods[method].get("success")
            for method in METHODS
        ):
            ellipse = methods["ellipse_distance"]
            aabb = methods["aabb_disk_dilation"]
            paired.append(
                {
                    "trial_id": trial_id,
                    "path_length_delta_m_aabb_minus_ellipse": (
                        aabb["raw_path_length_m"]
                        - ellipse["raw_path_length_m"]
                    ),
                    "pipeline_time_delta_s_aabb_minus_ellipse": (
                        aabb["time_pipeline_s"]
                        - ellipse["time_pipeline_s"]
                    ),
                    "dijkstra_time_delta_s_aabb_minus_ellipse": (
                        aabb["time_dijkstra_s"]
                        - ellipse["time_dijkstra_s"]
                    ),
                }
            )
    summary["paired_both_success_count"] = int(len(paired))
    ellipse_only_success = sum(
        methods.get("ellipse_distance", {}).get("success", False)
        and not methods.get("aabb_disk_dilation", {}).get(
            "success", False
        )
        for methods in by_trial.values()
    )
    aabb_only_success = sum(
        methods.get("aabb_disk_dilation", {}).get("success", False)
        and not methods.get("ellipse_distance", {}).get("success", False)
        for methods in by_trial.values()
    )
    discordant = ellipse_only_success + aabb_only_success
    summary["paired_success_test"] = {
        "ellipse_only_success": int(ellipse_only_success),
        "aabb_only_success": int(aabb_only_success),
        "discordant_pairs": int(discordant),
        "exact_two_sided_p": (
            float(
                scipy.stats.binomtest(
                    ellipse_only_success,
                    discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
            if discordant
            else None
        ),
    }
    for key in (
        "path_length_delta_m_aabb_minus_ellipse",
        "pipeline_time_delta_s_aabb_minus_ellipse",
        "dijkstra_time_delta_s_aabb_minus_ellipse",
    ):
        values = np.asarray([row[key] for row in paired], dtype=np.float64)
        result = None
        if len(values):
            result = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
            }
            nonzero = values[np.abs(values) > 1e-12]
            if len(nonzero):
                alternative = (
                    "greater"
                    if key
                    == "path_length_delta_m_aabb_minus_ellipse"
                    else "two-sided"
                )
                test = scipy.stats.wilcoxon(
                    values,
                    zero_method="wilcox",
                    alternative=alternative,
                )
                result["wilcoxon_alternative"] = alternative
                result["wilcoxon_statistic"] = float(test.statistic)
                result["wilcoxon_p"] = float(test.pvalue)
            else:
                result["wilcoxon_alternative"] = None
                result["wilcoxon_statistic"] = None
                result["wilcoxon_p"] = None
        summary[key] = result
    path_deltas = np.asarray(
        [
            row["path_length_delta_m_aabb_minus_ellipse"]
            for row in paired
        ],
        dtype=np.float64,
    )
    summary["paired_path_direction_counts"] = {
        "aabb_longer": int(np.count_nonzero(path_deltas > 1e-9)),
        "equal": int(np.count_nonzero(np.abs(path_deltas) <= 1e-9)),
        "aabb_shorter": int(np.count_nonzero(path_deltas < -1e-9)),
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
    grids: dict[str, GroundGrid],
    records: list[dict],
    selected: list[dict],
    artifacts: dict[tuple[str, str], dict],
    extent: list[float],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for axis, method in zip(axes, METHODS):
        occupied = grids[method].occupied.detach().cpu().numpy()
        axis.imshow(
            occupied.T,
            origin="lower",
            extent=extent,
            cmap="Greys",
            interpolation="nearest",
            aspect="equal",
        )
        for candidate in selected:
            start = candidate["start_scene"]
            goal = candidate["goal_scene"]
            axis.plot(
                [start[0], goal[0]],
                [start[1], goal[1]],
                ".",
                color="tab:orange",
                alpha=0.5,
                markersize=3,
            )
        configure_map_axis(
            axis, extent, f"{method}: 30 common endpoint pairs"
        )
    fig.tight_layout()
    fig.savefig(output_dir / "maps_and_selected_pairs.png", dpi=180)
    plt.close(fig)

    by_trial: dict[str, dict[str, dict]] = {}
    for row in records:
        by_trial.setdefault(row["trial_id"], {})[row["method"]] = row
    both = [
        methods
        for methods in by_trial.values()
        if all(
            method in methods and methods[method].get("success")
            for method in METHODS
        )
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if both:
        ellipse_lengths = [
            row["ellipse_distance"]["raw_path_length_m"] for row in both
        ]
        aabb_lengths = [
            row["aabb_disk_dilation"]["raw_path_length_m"] for row in both
        ]
        axes[0].scatter(ellipse_lengths, aabb_lengths, alpha=0.75)
        limits = [
            min(ellipse_lengths + aabb_lengths),
            max(ellipse_lengths + aabb_lengths),
        ]
        axes[0].plot(limits, limits, "k--", linewidth=1)
        axes[0].set_xlabel("Ellipse-distance path length (m)")
        axes[0].set_ylabel("AABB path length (m)")
        axes[0].set_title("Paired Dijkstra path length")
        ellipse_times = [
            row["ellipse_distance"]["time_pipeline_s"] for row in both
        ]
        aabb_times = [
            row["aabb_disk_dilation"]["time_pipeline_s"] for row in both
        ]
        axes[1].scatter(ellipse_times, aabb_times, alpha=0.75)
        limits = [
            min(ellipse_times + aabb_times),
            max(ellipse_times + aabb_times),
        ]
        axes[1].plot(limits, limits, "k--", linewidth=1)
        axes[1].set_xlabel("Ellipse-distance pipeline time (s)")
        axes[1].set_ylabel("AABB pipeline time (s)")
        axes[1].set_title("Paired downstream time")
    else:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "No pair succeeded with both methods",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
    fig.tight_layout()
    fig.savefig(output_dir / "paired_path_and_timing.png", dpi=180)
    plt.close(fig)

    timing_names = (
        "time_dijkstra_s",
        "time_simplification_s",
        "time_corridor_s",
        "time_qp_s",
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, timing_name in zip(axes.ravel(), timing_names):
        values = [
            [
                row[timing_name]
                for row in records
                if row["method"] == method
                and row.get("success")
                and timing_name in row
            ]
            for method in METHODS
        ]
        axis.boxplot(values, tick_labels=["Ellipse", "AABB"], showfliers=True)
        axis.set_title(timing_name)
        axis.set_ylabel("seconds")
    fig.tight_layout()
    fig.savefig(output_dir / "timing_breakdown.png", dpi=180)
    plt.close(fig)

    representatives = []
    for label in ("low", "medium", "high"):
        rows = [
            row for row in selected if row["complexity_layer"] == label
        ]
        rows = sorted(rows, key=lambda row: row["reference_path_length_m"])
        representatives.append(rows[len(rows) // 2])
    fig, axes = plt.subplots(3, 2, figsize=(13, 17), sharex=True, sharey=True)
    for row_index, candidate in enumerate(representatives):
        for column_index, method in enumerate(METHODS):
            axis = axes[row_index, column_index]
            occupied = grids[method].occupied.detach().cpu().numpy()
            axis.imshow(
                occupied.T,
                origin="lower",
                extent=extent,
                cmap="Greys",
                aspect="equal",
                interpolation="nearest",
            )
            artifact = artifacts.get((candidate["trial_id"], method))
            if artifact is None:
                matching = next(
                    row
                    for row in records
                    if row["trial_id"] == candidate["trial_id"]
                    and row["method"] == method
                )
                axis.text(
                    0.5,
                    0.5,
                    f"FAILED\n{matching.get('failure_stage')}\n"
                    f"{matching.get('failure_reason')}",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    color="red",
                )
            else:
                raw = artifact["raw"]
                simplified = artifact["simplified"]
                dense = artifact["dense"]
                axis.plot(
                    raw[:, 0],
                    raw[:, 1],
                    color="tab:blue",
                    linewidth=1,
                    label="Dijkstra",
                )
                axis.plot(
                    simplified[:, 0],
                    simplified[:, 1],
                    "o--",
                    color="tab:orange",
                    markersize=2.5,
                    label="simplified",
                )
                for section_index, section in enumerate(dense):
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
                f"{candidate['complexity_layer']} / {method}",
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
    parser.add_argument("--max-segment-meters", type=float, default=1.0)
    parser.add_argument(
        "--endpoint-clearance-meters", type=float, default=0.05
    )
    parser.add_argument("--min-distance-meters", type=float, default=2.0)
    parser.add_argument("--candidate-count", type=int, default=300)
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--dense-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.sample_count != 30:
        raise ValueError("this standard experiment requires 30 paired cases")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform_path = (
        args.config.resolve().parent / "dataparser_transforms.json"
    )
    scale = float(json.loads(transform_path.read_text())["scale"])
    height = args.robot_height_meters * scale
    radius = args.footprint_radius_meters * scale
    clearance = args.ground_clearance_meters * scale
    stopping_distance_m = compute_stopping_distance(
        args.max_speed_mps, args.max_brake_decel_mps2
    )
    stopping_distance = stopping_distance_m * scale
    preset = SCENE_PRESETS[args.scene]
    lower = torch.tensor(preset["lower_bound"], device=device)
    upper = torch.tensor(preset["upper_bound"], device=device)
    z_min = args.z_floor_scene + clearance
    z_max = args.z_floor_scene + height
    timings = {}

    sync(device)
    begin = time.perf_counter()
    gsplat = GSplatLoader(args.config, device)
    sync(device)
    timings["gsplat_load_s"] = time.perf_counter() - begin
    begin = time.perf_counter()
    ellipses = PlanarGaussianSet.from_gsplat(
        gsplat, z_min, z_max, confidence=1.0
    )
    sync(device)
    timings["height_filter_projection_s"] = time.perf_counter() - begin

    grids: dict[str, GroundGrid] = {}
    raster_times = {}
    for method in METHODS:
        sync(device)
        begin = time.perf_counter()
        grids[method] = GroundGrid.from_projected_gaussians(
            ellipses,
            lower_xy=lower[:2],
            upper_xy=upper[:2],
            resolution_xy=preset["resolution"],
            footprint_radius=radius,
            z_floor_scene=args.z_floor_scene,
            robot_height=height,
            ground_clearance=clearance,
            project_occupied_endpoints=False,
            rasterization=method,
        )
        sync(device)
        raster_times[method] = time.perf_counter() - begin
    collision_set = PlanarCollisionSet(
        ellipses,
        radius=radius,
        stopping_distance=stopping_distance,
        iterations=10,
    )

    indices, endpoint_info = endpoint_indices(
        grids["ellipse_distance"],
        grids["aabb_disk_dilation"],
        scale,
        args.endpoint_clearance_meters,
    )
    begin = time.perf_counter()
    candidates, candidate_info = generate_candidates(
        grids["ellipse_distance"],
        indices,
        scale,
        args.candidate_count,
        args.min_distance_meters,
        args.seed,
    )
    timings["candidate_generation_s"] = time.perf_counter() - begin
    selected = select_stratified(candidates, args.sample_count)

    save_json(
        args.output_dir / "candidate_pairs.json",
        {
            **candidate_info,
            "pairs": [
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "reference_path"
                }
                for candidate in candidates
            ],
        },
    )
    save_json(
        args.output_dir / "selected_pairs.json",
        [
            {
                **{
                    key: value
                    for key, value in candidate.items()
                    if key != "reference_path"
                },
                "reference_path": candidate["reference_path"].tolist(),
            }
            for candidate in selected
        ],
    )

    representative_ids = set()
    for label in ("low", "medium", "high"):
        rows = sorted(
            [
                row
                for row in selected
                if row["complexity_layer"] == label
            ],
            key=lambda row: row["reference_path_length_m"],
        )
        representative_ids.add(rows[len(rows) // 2]["trial_id"])

    records = []
    artifacts = {}
    for trial_index, candidate in enumerate(selected):
        order = METHODS if trial_index % 2 == 0 else METHODS[::-1]
        for execution_order, method in enumerate(order):
            print(
                f"[{trial_index + 1:02d}/30] "
                f"{candidate['trial_id']} {method}",
                flush=True,
            )
            record, artifact = run_method(
                candidate,
                method,
                grids[method],
                collision_set,
                scale,
                args.max_segment_meters * scale,
                args.dense_samples,
                device,
            )
            record["execution_order_within_pair"] = int(execution_order)
            records.append(record)
            if (
                candidate["trial_id"] in representative_ids
                and artifact is not None
            ):
                artifacts[(candidate["trial_id"], method)] = artifact
            print(
                json.dumps(
                    {
                        "success": record["success"],
                        "path_m": record.get("raw_path_length_m"),
                        "pipeline_s": record.get("time_pipeline_s"),
                        "failure": record.get("failure_reason"),
                    }
                ),
                flush=True,
            )

    save_csv(args.output_dir / "paired_metrics.csv", records)
    failures = [row for row in records if not row["success"]]
    save_json(args.output_dir / "failures.json", failures)
    summary = build_summary(
        records, grids, raster_times, args.sample_count
    )
    ellipse_occupied = (
        grids["ellipse_distance"].occupied.detach().cpu().numpy()
    )
    aabb_occupied = (
        grids["aabb_disk_dilation"].occupied.detach().cpu().numpy()
    )
    summary.update(
        {
            "scene": args.scene,
            "device": str(device),
            "scale": scale,
            "gaussians_total": int(gsplat.means.shape[0]),
            "projected_ellipses": int(len(ellipses)),
            "preprocessing": timings,
            "endpoint_selection": endpoint_info,
            "candidate_generation": candidate_info,
            "map_differences": {
                "aabb_only_occupied_cells": int(
                    (aabb_occupied & ~ellipse_occupied).sum()
                ),
                "ellipse_only_occupied_cells": int(
                    (ellipse_occupied & ~aabb_occupied).sum()
                ),
            },
            "physical_parameters": {
                "robot_height_m": args.robot_height_meters,
                "footprint_radius_m": args.footprint_radius_meters,
                "ground_clearance_m": args.ground_clearance_meters,
                "max_speed_mps": args.max_speed_mps,
                "max_brake_decel_mps2": args.max_brake_decel_mps2,
                "stopping_distance_m": stopping_distance_m,
                "max_segment_m": args.max_segment_meters,
            },
            "selection_policy": {
                "shared_endpoint_requirement": (
                    "both endpoints free in both maps"
                ),
                "reference_component": (
                    "largest ellipse-distance free component"
                ),
                "stratification": (
                    "10 low, 10 medium, 10 high by reference turns "
                    "and path length"
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
        grids,
        records,
        selected,
        artifacts,
        extent,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
