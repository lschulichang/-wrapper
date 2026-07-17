#!/usr/bin/env python3
"""Paired diagnostic ablation for four planar path simplification strategies."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
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
from ground_nav.path_utils import (  # noqa: E402
    merge_collinear_ground_path,
    simplify_ground_path,
    simplify_ground_path_supercover,
)
from ground_nav.planar_gaussians import PlanarGaussianSet  # noqa: E402
from ground_nav.timed_trajectory import evaluate_bezier_geometry  # noqa: E402
from initialization.grid_utils import GSplatVoxel  # noqa: E402
from smoke_ground_corridor_2d import (  # noqa: E402
    configure_map_axis,
    max_control_violation,
    polygon_vertices,
)
from smoke_splatplan import SCENE_PRESETS  # noqa: E402
from splat.splat_utils import GSplatLoader  # noqa: E402


STRATEGIES = ("A_none", "B_collinear", "C_sampled_los", "D_supercover_los")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default))


def metric_path(path, scale):
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)) / scale)


def path_metrics(path, scale):
    steps = np.diff(path, axis=0)
    # The ground planner is four-connected, so a turn is exactly a change in
    # the signed grid-step direction.  Normalising by a global axis maximum
    # can make this dependent on map resolution and is unnecessary.
    directions = np.sign(steps).astype(np.int8)
    turns = int(np.count_nonzero(np.any(directions[1:] != directions[:-1], axis=1))) if len(steps) > 1 else 0
    direct = float(np.linalg.norm(path[-1] - path[0]) / scale)
    length = metric_path(path, scale)
    return {
        "raw_point_count": int(len(path)),
        "raw_segment_count": int(len(path) - 1),
        "raw_path_length_m": length,
        "direct_distance_m": direct,
        "raw_turn_count": turns,
        "tortuosity": float(length / max(direct, 1e-12)),
    }


def largest_component_endpoints(grid, scale, endpoint_clearance_m):
    occupied = grid.occupied.detach().cpu().numpy().astype(bool)
    free = ~occupied
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, count = scipy.ndimage.label(free, structure=structure)
    if count == 0:
        raise RuntimeError("projected grid has no free connected component")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    sampling_m = grid.cell_sizes.detach().cpu().numpy() / scale
    clearance = scipy.ndimage.distance_transform_edt(free, sampling=sampling_m)
    eligible = (labels == largest) & (clearance >= endpoint_clearance_m)
    eligible[[0, -1], :] = False
    eligible[:, [0, -1]] = False
    indices = np.argwhere(eligible)
    if len(indices) < 60:
        raise RuntimeError(f"only {len(indices)} endpoint cells satisfy the clearance constraint")
    return indices, {
        "component_count": int(count),
        "largest_component_cells": int(sizes[largest]),
        "eligible_endpoint_cells": int(len(indices)),
        "endpoint_clearance_m": float(endpoint_clearance_m),
    }


def generate_candidate_paths(grid, endpoint_indices, scale, count, min_distance_m, seed):
    rng = np.random.default_rng(seed)
    candidates = []
    seen = set()
    attempts = 0
    maximum_attempts = max(10000, count * 1000)
    while len(candidates) < count and attempts < maximum_attempts:
        attempts += 1
        chosen = rng.choice(len(endpoint_indices), size=2, replace=False)
        first_index = tuple(int(v) for v in endpoint_indices[chosen[0]])
        second_index = tuple(int(v) for v in endpoint_indices[chosen[1]])
        pair_key = tuple(sorted((first_index, second_index)))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        start = grid.grid_to_world(first_index).detach().cpu().numpy()
        goal = grid.grid_to_world(second_index).detach().cpu().numpy()
        if float(np.linalg.norm(goal - start) / scale) < min_distance_m:
            continue
        try:
            raw_path = grid.create_path(start, goal)
        except Exception:
            continue
        metrics = path_metrics(raw_path, scale)
        candidates.append({
            "candidate_id": f"candidate_{len(candidates):04d}",
            "start_grid": list(first_index),
            "goal_grid": list(second_index),
            "start_scene": start.tolist(),
            "goal_scene": goal.tolist(),
            "start_meters": (start / scale).tolist(),
            "goal_meters": (goal / scale).tolist(),
            "raw_path": raw_path,
            **metrics,
        })
    if len(candidates) < count:
        raise RuntimeError(f"generated only {len(candidates)} of {count} requested candidates")
    return candidates, {"sampling_attempts": attempts, "candidate_count": len(candidates)}


def select_stratified_candidates(candidates, sample_count):
    """Select paired trials using only inflated-grid path properties.

    The projected and footprint-dilated occupancy grid is the safety model for
    Dijkstra and path simplification.  Deliberately do not pre-filter complete
    raw paths with the continuous circle--ellipse test; the exact geometry is
    retained where it is required to build corridor separating lines and to
    verify the final Bezier trajectory.
    """

    if sample_count % 3 != 0:
        raise ValueError("sample_count must be divisible by three")
    ordered = sorted(candidates, key=lambda row: (row["raw_turn_count"], row["raw_path_length_m"]))
    layers = np.array_split(np.asarray(ordered, dtype=object), 3)
    per_layer = sample_count // 3
    selected = []
    labels = ("low", "medium", "high")
    for label, values in zip(labels, layers):
        values = sorted(values.tolist(), key=lambda row: row["raw_path_length_m"])
        targets = np.linspace(0.05, 0.95, per_layer)
        used = set()
        for target in targets:
            desired = target * max(len(values) - 1, 1)
            ranks = sorted(range(len(values)), key=lambda index: (abs(index - desired), index))
            chosen = None
            for rank in ranks:
                candidate = values[rank]
                candidate_id = candidate["candidate_id"]
                if candidate_id in used or any(row["candidate_id"] == candidate_id for row in selected):
                    continue
                chosen = dict(candidate)
                chosen["complexity_layer"] = label
                chosen["length_target_quantile"] = float(target)
                used.add(candidate_id)
                selected.append(chosen)
                break
            if chosen is None:
                raise RuntimeError(f"could not select {per_layer} exact-safe samples for {label} layer")
    for index, row in enumerate(selected):
        row["trial_id"] = f"trial_{index:03d}"
    return selected


def choose_simplified_path(strategy, raw_path, grid, max_segment_scene):
    if strategy == "A_none":
        return np.array(raw_path, copy=True)
    if strategy == "B_collinear":
        return merge_collinear_ground_path(raw_path, max_segment_scene)
    if strategy == "C_sampled_los":
        return simplify_ground_path(raw_path, grid, max_segment_scene)
    if strategy == "D_supercover_los":
        return simplify_ground_path_supercover(raw_path, grid, max_segment_scene)
    raise ValueError(f"unknown strategy {strategy}")


def bezier_quality(controls, dense, scale):
    trajectory_length = sum(
        np.sum(np.linalg.norm(np.diff(section, axis=0), axis=1)) for section in dense
    ) / scale
    curvature_values = []
    values = np.linspace(0.0, 1.0, 101)
    for control in controls:
        _, first, second = evaluate_bezier_geometry(control / scale, values)
        norm = np.linalg.norm(first, axis=1)
        valid = norm > 1e-10
        curvature = np.full(len(values), np.inf)
        curvature[valid] = (
            first[valid, 0] * second[valid, 1]
            - first[valid, 1] * second[valid, 0]
        ) / norm[valid] ** 3
        curvature_values.append(curvature)
    curvature_values = np.abs(np.concatenate(curvature_values))
    finite_curvature = curvature_values[np.isfinite(curvature_values)]
    return {
        "bezier_length_m": float(trajectory_length),
        "curvature_degenerate_sample_count": int(np.count_nonzero(~np.isfinite(curvature_values))),
        "max_abs_curvature_1pm": (
            float(np.max(finite_curvature)) if len(finite_curvature) else None
        ),
        "p95_abs_curvature_1pm": (
            float(np.quantile(finite_curvature, 0.95)) if len(finite_curvature) else None
        ),
    }


def dense_exact_verification(collision_set, corridor, simplified, dense):
    """Verify sampled Bezier chords against the complete relevant ellipse set.

    Each corridor polygon is built inside the shrunken candidate rectangle of
    its source seed segment.  Therefore ellipses rejected by that rectangle
    cannot intersect a robot disk whose centre remains in the polygon.  Reusing
    this candidate set avoids scanning every projected Gaussian for every dense
    chord while preserving the same continuous circle--ellipse test.
    """

    for section_index, section in enumerate(dense):
        source = corridor.source_segment_indices[section_index]
        candidate_data = collision_set.candidates(simplified[[source, source + 1]])
        for first, second in zip(section[:-1], section[1:]):
            segment = torch.as_tensor(
                np.stack([first, second]),
                dtype=collision_set.ellipses.means.dtype,
                device=collision_set.device,
            )
            data = dict(candidate_data)
            data["segment"] = segment
            exact = collision_set.exact_test(segment, data)
            if not bool(torch.all(exact["is_not_intersect"]).item()):
                return False
    return True


def run_variant(candidate, strategy, grid, collision_set, scale, max_segment_scene, dense_samples, device):
    raw_path = candidate["raw_path"]
    record = {
        "trial_id": candidate["trial_id"],
        "candidate_id": candidate["candidate_id"],
        "complexity_layer": candidate["complexity_layer"],
        "strategy": strategy,
        "success": False,
        "failure_stage": None,
        "failure_reason": None,
        **{
            key: value
            for key, value in candidate.items()
            if (key.startswith("raw_") and key != "raw_path")
            or key in ("direct_distance_m", "tortuosity")
        },
    }
    artifact = None
    stage = "simplification"
    try:
        sync(device); start_time = time.perf_counter()
        simplified = choose_simplified_path(strategy, raw_path, grid, max_segment_scene)
        sync(device); record["time_simplification_s"] = time.perf_counter() - start_time
        lengths = np.linalg.norm(np.diff(simplified, axis=0), axis=1)
        if not np.allclose(simplified[0], raw_path[0]) or not np.allclose(simplified[-1], raw_path[-1]):
            raise RuntimeError("simplification changed an endpoint")
        if np.any(lengths <= 1e-12) or float(np.max(lengths)) > max_segment_scene + 1e-8:
            raise RuntimeError("simplified path contains an invalid segment")
        record.update({
            "simplified_point_count": int(len(simplified)),
            "simplified_segment_count": int(len(simplified) - 1),
            "point_compression_ratio": float(len(simplified) / len(raw_path)),
            "simplified_path_length_m": metric_path(simplified, scale),
            "max_segment_length_m": float(np.max(lengths) / scale),
        })

        stage = "corridor"
        sync(device); start_time = time.perf_counter()
        corridor = build_planar_corridor(simplified, collision_set)
        sync(device); record["time_corridor_s"] = time.perf_counter() - start_time
        created = [row for row in corridor.diagnostics if row.get("created_polygon")]
        record.update({
            "polygon_count": int(len(corridor.polygons)),
            "candidate_gaussian_total": int(sum(row.get("candidate_count", 0) for row in created)),
            "candidate_gaussian_mean": float(np.mean([row.get("candidate_count", 0) for row in created])),
            "candidate_gaussian_max": int(max(row.get("candidate_count", 0) for row in created)),
            "halfspace_total": int(sum(row.get("halfspace_count", 0) for row in created)),
        })

        stage = "bezier_qp"
        planner = BezierPlanner2D(degree=6, continuity_order=3)
        start_time = time.perf_counter()
        controls, feasible = planner.optimize(corridor.polygons, simplified[0], simplified[-1])
        record["time_qp_s"] = time.perf_counter() - start_time
        record["qp_status"] = planner.last_solver_status
        if not feasible:
            raise RuntimeError(f"QP status {planner.last_solver_status}")
        record["qp_variable_count"] = int(controls.size)
        dense = planner.sample(dense_samples)
        record["max_control_point_violation"] = max_control_violation(controls, corridor.polygons)
        record["max_endpoint_error_m"] = float(max(
            np.linalg.norm(dense[0, 0] - simplified[0]),
            np.linalg.norm(dense[-1, -1] - simplified[-1]),
        ) / scale)
        if record["max_control_point_violation"] > 1e-5 or record["max_endpoint_error_m"] > 1e-6:
            raise RuntimeError("QP geometry verification failed")

        stage = "dense_exact_verification"
        sync(device); start_time = time.perf_counter()
        dense_safe = dense_exact_verification(
            collision_set, corridor, simplified, dense
        )
        sync(device); record["time_dense_exact_s"] = time.perf_counter() - start_time
        record["dense_trajectory_exact_safe"] = bool(dense_safe)
        if not dense_safe:
            raise RuntimeError("dense Bezier trajectory failed exact collision verification")

        record.update(bezier_quality(controls, dense, scale))
        record["time_downstream_s"] = float(
            record["time_simplification_s"]
            + record["time_corridor_s"]
            + record["time_qp_s"]
        )
        record["time_after_simplification_s"] = float(
            record["time_corridor_s"]
            + record["time_qp_s"]
        )
        record["success"] = True
        artifact = {"simplified": simplified, "dense": dense, "polygons": corridor.polygons}
    except Exception as error:
        record["failure_stage"] = stage
        record["failure_reason"] = str(error)
        present = [record.get(key, 0.0) for key in (
            "time_simplification_s", "time_corridor_s", "time_qp_s"
        )]
        record["time_downstream_s"] = float(sum(present))
    return record, artifact


def summarize_values(values, rng):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return None
    resampled = rng.choice(values, size=(2000, len(values)), replace=True)
    medians = np.median(resampled, axis=1)
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "bootstrap_median_ci95": [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))],
    }


def paired_statistics(records):
    by_trial = {}
    for row in records:
        if row["success"]:
            by_trial.setdefault(row["trial_id"], {})[row["strategy"]] = row["time_downstream_s"]
    complete = [values for values in by_trial.values() if all(key in values for key in STRATEGIES)]
    if len(complete) < 3:
        return {"complete_pair_count": len(complete), "friedman": None, "wilcoxon": {}}
    arrays = {key: np.asarray([row[key] for row in complete]) for key in STRATEGIES}
    friedman = scipy.stats.friedmanchisquare(*(arrays[key] for key in STRATEGIES))
    raw = []
    for first, second in itertools.combinations(STRATEGIES, 2):
        result = scipy.stats.wilcoxon(arrays[first], arrays[second], alternative="two-sided")
        raw.append((f"{first}_vs_{second}", float(result.statistic), float(result.pvalue)))
    ordered = sorted(enumerate(raw), key=lambda item: item[1][2])
    corrected = [None] * len(raw)
    running = 0.0
    for rank, (original_index, (_, _, pvalue)) in enumerate(ordered):
        adjusted = min(1.0, pvalue * (len(raw) - rank))
        running = max(running, adjusted)
        corrected[original_index] = running
    return {
        "complete_pair_count": len(complete),
        "friedman": {"statistic": float(friedman.statistic), "pvalue": float(friedman.pvalue)},
        "wilcoxon": {
            name: {"statistic": statistic, "raw_pvalue": pvalue, "holm_pvalue": corrected[index]}
            for index, (name, statistic, pvalue) in enumerate(raw)
        },
    }


def build_summary(records, sample_count, seed):
    metrics = (
        "time_simplification_s", "time_corridor_s", "time_qp_s",
        "time_downstream_s", "time_after_simplification_s", "simplified_point_count",
        "polygon_count", "candidate_gaussian_total",
        "halfspace_total", "bezier_length_m", "max_abs_curvature_1pm",
    )
    summary = {"sample_count": sample_count, "strategies": {}, "paired_statistics": paired_statistics(records)}
    rng = np.random.default_rng(seed + 91)
    for strategy in STRATEGIES:
        rows = [row for row in records if row["strategy"] == strategy]
        successful = [row for row in rows if row["success"]]
        collision_failures = sum(
            row.get("failure_stage") == "dense_exact_verification"
            or (
                row.get("failure_stage") == "corridor"
                and "not circle-ellipse safe" in (row.get("failure_reason") or "")
            )
            for row in rows
        )
        summary["strategies"][strategy] = {
            "success_count": len(successful),
            "success_rate": len(successful) / sample_count,
            "collision_failure_count": int(collision_failures),
            "metrics": {
                key: summarize_values(
                    [row[key] for row in successful if row.get(key) is not None], rng
                )
                for key in metrics
            },
        }
    eligible = []
    for strategy, values in summary["strategies"].items():
        if values["success_count"] >= math.ceil(0.95 * sample_count) and values["collision_failure_count"] == 0:
            eligible.append(strategy)
    if not eligible:
        summary["recommendation"] = {"strategy": None, "reason": "no strategy met the safety and 95% success gates"}
        return summary
    medians = {key: summary["strategies"][key]["metrics"]["time_downstream_s"]["median"] for key in eligible}
    best_median = min(medians.values())
    tied = [key for key in eligible if medians[key] <= best_median * 1.10 + 1e-12]
    if len(tied) > 1:
        p95 = {key: summary["strategies"][key]["metrics"]["time_downstream_s"]["p95"] for key in tied}
        best_p95 = min(p95.values())
        tied = [key for key in tied if p95[key] <= best_p95 * 1.10 + 1e-12]
    if len(tied) > 1:
        tied.sort(key=lambda key: (
            summary["strategies"][key]["metrics"]["polygon_count"]["median"],
            summary["strategies"][key]["metrics"]["bezier_length_m"]["median"],
            key,
        ))
    summary["recommendation"] = {
        "strategy": tied[0],
        "reason": "passed safety/success gates and won the median, p95, polygon-count, trajectory-length decision order",
    }
    return summary


def save_records(output_dir, records):
    scalar_keys = sorted({key for row in records for key, value in row.items() if np.isscalar(value) or value is None})
    with (output_dir / "paired_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key) for key in scalar_keys})


def plot_timing(output_dir, records):
    successful = {key: [row for row in records if row["strategy"] == key and row["success"]] for key in STRATEGIES}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    datasets = [[row["time_downstream_s"] for row in successful[key]] for key in STRATEGIES]
    if all(datasets):
        axes[0].boxplot(datasets, tick_labels=STRATEGIES)
    else:
        axes[0].text(0.5, 0.5, "insufficient successful runs", ha="center", va="center")
        axes[0].set_xticks(range(1, len(STRATEGIES) + 1), STRATEGIES)
    axes[0].set(ylabel="time [s]", title="Paired downstream time")
    components = ("time_simplification_s", "time_corridor_s", "time_qp_s")
    bottoms = np.zeros(len(STRATEGIES))
    for component in components:
        values = np.array([
            np.median([row[component] for row in successful[key]])
            if successful[key] else 0.0
            for key in STRATEGIES
        ])
        axes[1].bar(STRATEGIES, values, bottom=bottoms, label=component.replace("time_", "").replace("_s", ""))
        bottoms += values
    axes[1].set(ylabel="median time [s]", title="Median time breakdown")
    axes[1].legend(fontsize=8)
    for axis in axes: axis.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(output_dir / "timing_breakdown.png", dpi=180); plt.close(fig)


def plot_complexity(output_dir, records):
    successful = {key: [row for row in records if row["strategy"] == key and row["success"]] for key in STRATEGIES}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, metric, title in zip(
        axes,
        ("simplified_point_count", "polygon_count", "candidate_gaussian_total"),
        ("Simplified points", "Safety polygons", "Candidate Gaussians"),
    ):
        datasets = [[row[metric] for row in successful[key]] for key in STRATEGIES]
        if all(datasets):
            axis.boxplot(datasets, tick_labels=STRATEGIES)
        else:
            axis.text(0.5, 0.5, "insufficient successful runs", ha="center", va="center")
            axis.set_xticks(range(1, len(STRATEGIES) + 1), STRATEGIES)
        axis.set_title(title); axis.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(output_dir / "compression_and_corridor.png", dpi=180); plt.close(fig)


def plot_quality(output_dir, records, sample_count):
    successful = {key: [row for row in records if row["strategy"] == key and row["success"]] for key in STRATEGIES}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(STRATEGIES, [len(successful[key]) / sample_count for key in STRATEGIES])
    axes[0].set(ylim=(0, 1.05), ylabel="success rate", title="End-to-end planning success")
    for axis, metric, title in (
        (axes[1], "bezier_length_m", "Bezier length [m]"),
        (axes[2], "max_abs_curvature_1pm", "Max |curvature| [1/m]"),
    ):
        datasets = [
            [row[metric] for row in successful[key] if row.get(metric) is not None]
            for key in STRATEGIES
        ]
        if all(datasets):
            axis.boxplot(datasets, tick_labels=STRATEGIES)
        else:
            axis.text(0.5, 0.5, "insufficient successful runs", ha="center", va="center")
            axis.set_xticks(range(1, len(STRATEGIES) + 1), STRATEGIES)
        axis.set_title(title)
    for axis in axes: axis.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(output_dir / "trajectory_quality.png", dpi=180); plt.close(fig)


def plot_representatives(output_dir, representatives, artifacts, occupied, extent):
    fig, axes = plt.subplots(3, len(STRATEGIES), figsize=(20, 14))
    for row_index, candidate in enumerate(representatives):
        for column_index, strategy in enumerate(STRATEGIES):
            axis = axes[row_index, column_index]
            axis.imshow(occupied.T, origin="lower", extent=extent, cmap="Greys", alpha=0.65, interpolation="nearest")
            raw = candidate["raw_path"]
            axis.plot(raw[:, 0], raw[:, 1], color="tab:blue", linewidth=0.8, alpha=0.5, label="raw Dijkstra")
            artifact = artifacts.get((candidate["trial_id"], strategy))
            if artifact is not None:
                simplified = artifact["simplified"]
                axis.plot(simplified[:, 0], simplified[:, 1], "o--", color="tab:orange", markersize=2.5, label="seed")
                for A, b in artifact["polygons"]:
                    vertices = polygon_vertices(A.detach().cpu().numpy(), b.detach().cpu().numpy())
                    if vertices is not None:
                        axis.fill(vertices[:, 0], vertices[:, 1], color="tab:green", alpha=0.12)
                dense = artifact["dense"]
                for section in dense:
                    axis.plot(section[:, 0], section[:, 1], color="tab:red", linewidth=1.7)
            configure_map_axis(axis, extent, f"{candidate['complexity_layer']} / {strategy}")
            if row_index == 0 and column_index == 0: axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output_dir / "representative_paths.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scene", default="old_union", choices=sorted(SCENE_PRESETS))
    parser.add_argument(
        "--z-floor-scene", "--z-floor", dest="z_floor_scene", type=float, default=-0.15,
        help="Floor Z in normalized scene coordinates; --z-floor is a deprecated alias",
    )
    parser.add_argument("--robot-height-meters", type=float, default=0.10)
    parser.add_argument("--footprint-radius-meters", type=float, default=0.15)
    parser.add_argument("--ground-clearance-meters", type=float, default=0.02)
    parser.add_argument("--max-speed-mps", type=float, default=0.20)
    parser.add_argument("--max-brake-decel-mps2", type=float, default=0.30)
    parser.add_argument(
        "--corridor-margin-meters",
        type=float,
        default=None,
        help="Deprecated manual override; default is vmax^2/(2*max_brake_decel)",
    )
    parser.add_argument("--max-segment-meters", type=float, default=1.0)
    parser.add_argument("--endpoint-clearance-meters", type=float, default=0.05)
    parser.add_argument("--min-distance-meters", type=float, default=2.0)
    parser.add_argument("--candidate-count", type=int, default=300)
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--dense-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--baseline-start", nargs=2, type=float,
        default=(0.7209999561, 0.2604999840),
    )
    parser.add_argument(
        "--baseline-goal", nargs=2, type=float,
        default=(-0.5350000262, 0.2300000042),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.sample_count % 3 or args.candidate_count < args.sample_count * 3:
        raise ValueError("sample_count must be divisible by 3 and candidate_count must be at least 3x larger")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = json.loads((args.config.resolve().parent / "dataparser_transforms.json").read_text())
    scale = float(transform["scale"])
    height = args.robot_height_meters * scale
    radius = args.footprint_radius_meters * scale
    clearance = args.ground_clearance_meters * scale
    computed_stopping_distance_m = compute_stopping_distance(
        args.max_speed_mps, args.max_brake_decel_mps2
    )
    if args.corridor_margin_meters is None:
        stopping_distance_m = computed_stopping_distance_m
        stopping_distance_source = "vmax_squared_over_2_brake_decel"
    else:
        if args.corridor_margin_meters < 0.0:
            raise ValueError("corridor-margin-meters must be non-negative")
        stopping_distance_m = float(args.corridor_margin_meters)
        stopping_distance_source = "manual_corridor_margin_override"
    stopping_distance = stopping_distance_m * scale
    maximum_segment = args.max_segment_meters * scale
    preset = SCENE_PRESETS[args.scene]
    lower = torch.tensor(preset["lower_bound"], device=device)
    upper = torch.tensor(preset["upper_bound"], device=device)
    lower[2] = min(float(lower[2]), args.z_floor_scene)
    upper[2] = max(float(upper[2]), args.z_floor_scene + height)

    preprocessing = {}
    sync(device); start_time = time.perf_counter(); gsplat = GSplatLoader(args.config, device); sync(device)
    preprocessing["gsplat_load_s"] = time.perf_counter() - start_time
    start_time = time.perf_counter(); voxel = GSplatVoxel(gsplat, lower, upper, preset["resolution"], 0.0, device); sync(device)
    preprocessing["voxel_s"] = time.perf_counter() - start_time
    start_time = time.perf_counter(); grid = GroundGrid(
        voxel, args.z_floor_scene, height, radius,
        ground_clearance=clearance, project_occupied_endpoints=False,
    ); sync(device); preprocessing["height_projection_xy_dilation_s"] = time.perf_counter() - start_time
    start_time = time.perf_counter(); ellipses = PlanarGaussianSet.from_gsplat(
        gsplat, grid.metadata.z_min, grid.metadata.z_max
    ); sync(device); preprocessing["ellipse_projection_s"] = time.perf_counter() - start_time
    collision_set = PlanarCollisionSet(
        ellipses, radius=radius, stopping_distance=stopping_distance, iterations=10
    )

    endpoint_indices, component_info = largest_component_endpoints(
        grid, scale, args.endpoint_clearance_meters
    )
    start_time = time.perf_counter()
    candidates, sampling_info = generate_candidate_paths(
        grid, endpoint_indices, scale, args.candidate_count, args.min_distance_meters, args.seed
    )
    preprocessing["candidate_generation_s"] = time.perf_counter() - start_time
    start_time = time.perf_counter()
    selected = select_stratified_candidates(candidates, args.sample_count)
    preprocessing["candidate_selection_s"] = time.perf_counter() - start_time

    # Preserve the fixed old_union pair used by milestones 2/4 as a separate
    # reproducibility anchor.  It is intentionally excluded from the paired
    # random-sample statistics below.
    baseline_start = np.asarray(args.baseline_start, dtype=np.float32)
    baseline_goal = np.asarray(args.baseline_goal, dtype=np.float32)
    baseline_path = grid.create_path(baseline_start, baseline_goal)
    baseline = {
        "start_scene": baseline_start.tolist(),
        "goal_scene": baseline_goal.tolist(),
        "start_grid": grid.world_to_grid(baseline_start).detach().cpu().tolist(),
        "goal_grid": grid.world_to_grid(baseline_goal).detach().cpu().tolist(),
        "raw_path": baseline_path.tolist(),
        "complete_path_exact_prefiltered": False,
        "included_in_statistics": False,
        **path_metrics(baseline_path, scale),
    }

    candidate_json = [{key: value for key, value in row.items() if key != "raw_path"} for row in candidates]
    selected_json = []
    for row in selected:
        selected_json.append({
            **{key: value for key, value in row.items() if key != "raw_path"},
            "raw_path": row["raw_path"].tolist(),
        })
    save_json(args.output_dir / "candidate_pairs.json", {
        "seed": args.seed, "component": component_info, "sampling": sampling_info, "pairs": candidate_json,
    })
    save_json(args.output_dir / "selected_pairs.json", selected_json)
    save_json(args.output_dir / "fixed_baseline_pair.json", baseline)

    representatives = []
    for label in ("low", "medium", "high"):
        rows = sorted([row for row in selected if row["complexity_layer"] == label], key=lambda row: row["raw_path_length_m"])
        representatives.append(rows[len(rows) // 2])
    representative_ids = {row["trial_id"] for row in representatives}

    records = []
    artifacts = {}
    # Rotate the execution order across trials so each strategy occupies every
    # within-trial position equally often over each block of four trials.
    orders = tuple(
        STRATEGIES[offset:] + STRATEGIES[:offset]
        for offset in range(len(STRATEGIES))
    )
    for trial_index, candidate in enumerate(selected):
        for execution_order, strategy in enumerate(orders[trial_index % len(orders)]):
            print(f"[{trial_index + 1:02d}/{len(selected)}] {candidate['trial_id']} {strategy}", flush=True)
            record, artifact = run_variant(
                candidate, strategy, grid, collision_set, scale,
                maximum_segment, args.dense_samples, device,
            )
            record["execution_order_within_trial"] = int(execution_order)
            records.append(record)
            if candidate["trial_id"] in representative_ids and artifact is not None:
                artifacts[(candidate["trial_id"], strategy)] = artifact
            print(json.dumps({
                "success": record["success"],
                "downstream_s": record.get("time_downstream_s"),
                "points": record.get("simplified_point_count"),
                "polygons": record.get("polygon_count"),
                "failure": record.get("failure_reason"),
            }), flush=True)

    save_records(args.output_dir, records)
    summary = build_summary(records, args.sample_count, args.seed)
    summary.update({
        "scene": args.scene,
        "device": str(device),
        "scale": scale,
        "z_floor_scene": args.z_floor_scene,
        "preprocessing": preprocessing,
        "physical_parameters": {
            "robot_height_m": args.robot_height_meters,
            "footprint_radius_m": args.footprint_radius_meters,
            "ground_clearance_m": args.ground_clearance_meters,
            "max_speed_mps": args.max_speed_mps,
            "max_brake_decel_mps2": args.max_brake_decel_mps2,
            "computed_stopping_distance_m": computed_stopping_distance_m,
            "stopping_distance_m": stopping_distance_m,
            "stopping_distance_source": stopping_distance_source,
            "corridor_margin_m": stopping_distance_m,
            "max_segment_m": args.max_segment_meters,
            "endpoint_clearance_m": args.endpoint_clearance_meters,
            "minimum_endpoint_distance_m": args.min_distance_meters,
        },
        "dataset": {
            **component_info,
            **sampling_info,
            "selected_count": len(selected),
            "complete_path_exact_prefilter_enabled": False,
        },
    })
    save_json(args.output_dir / "summary.json", summary)
    failures = [row for row in records if not row["success"]]
    save_json(args.output_dir / "failures.json", failures)
    extent = [float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1])]
    plot_timing(args.output_dir, records)
    plot_complexity(args.output_dir, records)
    plot_quality(args.output_dir, records, args.sample_count)
    plot_representatives(
        args.output_dir, representatives, artifacts,
        grid.occupied.detach().cpu().numpy(), extent,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
