#!/usr/bin/env python3
"""Extended Splat-Plan evaluation with per-segment candidate stats and corridor connectivity checks."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_upstream_root() -> Path:
    candidates = [
        PROJECT_ROOT / "splatnav-official",
        PROJECT_ROOT / "third_party" / "splatnav-official",
        Path("/data/howard/repos/splatnav-official"),
    ]
    for path in candidates:
        if (path / "splatplan").is_dir() and (path / "splat").is_dir():
            return path
    raise FileNotFoundError(f"Upstream source not found. Tried: {candidates}")


UPSTREAM_ROOT = resolve_upstream_root()
sys.path.insert(0, str(UPSTREAM_ROOT))

from splat.splat_utils import GSplatLoader  # noqa: E402
from splatplan.splatplan import SplatPlan  # noqa: E402
from splatplan.spline_utils import SplinePlanner  # noqa: E402


SCENE_PRESETS = {
    "old_union": {
        "radius_config": 1.35 / 2,
        "radius_z": 0.01,
        "mean_config": np.array([0.14, 0.23, -0.15], dtype=np.float32),
        "radius": 0.01,
        "amax": 0.1,
        "vmax": 0.1,
        "lower_bound": [-0.8, -0.7, -0.2],
        "upper_bound": [1.0, 1.0, -0.1],
        "resolution": 100,
    },
    "stonehenge": {
        "radius_config": 0.784 / 2,
        "radius_z": 0.01,
        "mean_config": np.array([-0.08, -0.03, 0.05], dtype=np.float32),
        "radius": 0.01,
        "amax": 0.1,
        "vmax": 0.1,
        "lower_bound": [-0.5, -0.5, 0.0],
        "upper_bound": [0.5, 0.5, 0.3],
        "resolution": 150,
    },
    "statues": {
        "radius_config": 0.475,
        "radius_z": 0.03,
        "mean_config": np.array([-0.064, -0.0064, -0.025], dtype=np.float32),
        "radius": 0.03,
        "amax": 0.1,
        "vmax": 0.1,
        "lower_bound": [-0.5, -0.5, -0.3],
        "upper_bound": [0.5, 0.5, 0.2],
        "resolution": 100,
    },
    "flight": {
        "radius_config": 0.545 / 2,
        "radius_z": 0.06,
        "mean_config": np.array([0.19, 0.01, -0.02], dtype=np.float32),
        "radius": 0.02,
        "amax": 0.1,
        "vmax": 0.1,
        "lower_bound": [-1.33, -0.5, -0.17],
        "upper_bound": [1.0, 0.5, 0.26],
        "resolution": 100,
    },
}


def traj_length_from_output(output: dict[str, Any]) -> float | None:
    traj = output.get("traj")
    if not isinstance(traj, list) or len(traj) < 2:
        return None
    pts = np.asarray(traj, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return None
    diffs = pts[1:, :3] - pts[:-1, :3]
    seg_lens = np.linalg.norm(diffs, axis=1)
    return float(np.sum(seg_lens))


def polytope_intersection_nonempty(poly1: list[list[float]], poly2: list[list[float]]) -> bool:
    arr1 = np.asarray(poly1, dtype=np.float64)
    arr2 = np.asarray(poly2, dtype=np.float64)
    A = np.vstack([arr1[:, :3], arr2[:, :3]])
    b = np.concatenate([arr1[:, 3], arr2[:, 3]])
    c = np.zeros(3, dtype=np.float64)
    res = scipy.optimize.linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * 3, method="highs")
    return bool(res.success)


def analyze_corridor_connectivity(polytopes: list[list[list[float]]]) -> dict[str, Any]:
    if not isinstance(polytopes, list) or len(polytopes) <= 1:
        return {
            "adjacent_pairs": max(len(polytopes) - 1, 0),
            "gap_pair_count": 0,
            "gap_pair_indices": [],
            "has_gap": False,
        }

    gap_indices: list[int] = []
    for i in range(len(polytopes) - 1):
        try:
            ok = polytope_intersection_nonempty(polytopes[i], polytopes[i + 1])
        except Exception:
            ok = False
        if not ok:
            gap_indices.append(i)

    return {
        "adjacent_pairs": len(polytopes) - 1,
        "gap_pair_count": len(gap_indices),
        "gap_pair_indices": gap_indices,
        "has_gap": len(gap_indices) > 0,
    }


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="old_union", choices=sorted(SCENE_PRESETS))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--save_full_output", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    p = SCENE_PRESETS[args.scene]

    robot_config = {"radius": p["radius"], "vmax": p["vmax"], "amax": p["amax"]}
    voxel_config = {
        "lower_bound": torch.tensor(p["lower_bound"], device=device),
        "upper_bound": torch.tensor(p["upper_bound"], device=device),
        "resolution": p["resolution"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0_load = time.time()
    gsplat = GSplatLoader(args.config, device)
    spline_planner = SplinePlanner(spline_deg=6, device=device)
    planner = SplatPlan(gsplat, robot_config, voxel_config, spline_planner, device)
    load_time = time.time() - t0_load

    t = np.linspace(0.0, 2.0 * math.pi, args.trials, endpoint=False, dtype=np.float32)
    tz = 10.0 * np.linspace(0.0, 2.0 * math.pi, args.trials, endpoint=False, dtype=np.float32)

    starts = np.stack(
        [
            p["radius_config"] * np.cos(t),
            p["radius_config"] * np.sin(t),
            p["radius_z"] * np.sin(tz),
        ],
        axis=-1,
    ).astype(np.float32)
    starts = starts + p["mean_config"][None, :]

    goals = np.stack(
        [
            p["radius_config"] * np.cos(t + math.pi),
            p["radius_config"] * np.sin(t + math.pi),
            p["radius_z"] * np.sin(tz + math.pi),
        ],
        axis=-1,
    ).astype(np.float32)
    goals = goals + p["mean_config"][None, :]

    rng = np.random.default_rng(args.seed)
    order = np.arange(args.trials)
    rng.shuffle(order)

    trial_records: list[dict[str, Any]] = []
    plan_times: list[float] = []
    path_lengths: list[float] = []
    feasible_count = 0
    all_candidate_counts: list[int] = []

    for rank, idx in enumerate(order.tolist(), start=1):
        start = starts[idx]
        goal = goals[idx]
        x = torch.tensor(start, dtype=torch.float32, device=device)
        g = torch.tensor(goal, dtype=torch.float32, device=device)

        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.synchronize()

        error = None
        output = None
        try:
            output = planner.generate_path(x, g)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

        if device.type == "cuda":
            torch.cuda.synchronize()
        plan_time = time.time() - t0
        plan_times.append(plan_time)

        if output is None:
            feasible = False
            traj_len = None
            num_polytopes = None
            segment_debug_summary = None
            corridor_connectivity = None
            candidate_counts_this_trial: list[int] = []
            exact_hits_this_trial: list[int] = []
        else:
            feasible = bool(output.get("feasible", False))
            traj_len = traj_length_from_output(output)
            if traj_len is not None:
                path_lengths.append(traj_len)
            num_polytopes = output.get("num_polytopes")
            segment_debug_summary = output.get("segment_debug_summary")
            segment_debug = output.get("segment_debug", [])
            candidate_counts_this_trial = [
                int(row.get("candidate_count", 0))
                for row in segment_debug
                if row.get("created_polytope", False)
            ]
            exact_hits_this_trial = [
                int(row.get("exact_hit_count", 0))
                for row in segment_debug
                if row.get("created_polytope", False)
            ]
            all_candidate_counts.extend(candidate_counts_this_trial)
            corridor_connectivity = analyze_corridor_connectivity(output.get("polytopes", []))

        feasible_count += int(feasible)

        rec: dict[str, Any] = {
            "order_rank": rank,
            "trial_index": idx,
            "start": start.tolist(),
            "goal": goal.tolist(),
            "plan_time_sec": plan_time,
            "feasible": feasible,
            "traj_length_m": traj_len,
            "num_polytopes": num_polytopes,
            "candidate_counts_per_polytope": candidate_counts_this_trial,
            "exact_hits_per_polytope": exact_hits_this_trial,
            "segment_debug_summary": segment_debug_summary,
            "corridor_connectivity": corridor_connectivity,
            "error": error,
        }

        if args.save_full_output and output is not None:
            rec["output_full"] = to_builtin(output)

        trial_records.append(rec)
        print(
            f"[eval] {rank:03d}/{args.trials} idx={idx:03d} feasible={feasible} "
            f"plan_time={plan_time:.4f}s traj_len={traj_len}"
        )

    safety_rate = feasible_count / float(args.trials)

    connectivity_failures = [
        r
        for r in trial_records
        if isinstance(r.get("corridor_connectivity"), dict)
        and r["corridor_connectivity"].get("has_gap", False)
    ]

    infeasible_with_gap = [
        r
        for r in trial_records
        if (not r.get("feasible", False))
        and isinstance(r.get("corridor_connectivity"), dict)
        and r["corridor_connectivity"].get("has_gap", False)
    ]

    first_success = next((r for r in trial_records if r.get("feasible", False)), None)
    first_failure = next((r for r in trial_records if not r.get("feasible", False)), None)

    summary = {
        "scene": args.scene,
        "config": str(args.config),
        "trials": args.trials,
        "seed": args.seed,
        "device": str(device),
        "load_time_sec": load_time,
        "safety_rate": safety_rate,
        "feasible_count": feasible_count,
        "infeasible_count": args.trials - feasible_count,
        "plan_time_stats": summarize(plan_times),
        "traj_length_stats": summarize(path_lengths),
        "candidate_count_stats": summarize([float(v) for v in all_candidate_counts]),
        "candidate_count_total_samples": len(all_candidate_counts),
        "corridor_gap_trial_count": len(connectivity_failures),
        "infeasible_with_gap_count": len(infeasible_with_gap),
        "first_success_trial_index": None if first_success is None else first_success["trial_index"],
        "first_failure_trial_index": None if first_failure is None else first_failure["trial_index"],
    }

    out_json = {
        "summary": summary,
        "trial_records": trial_records,
    }

    (args.output_dir / "eval_results.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    # candidate distribution artifacts
    cand_csv = args.output_dir / "candidate_counts_all_segments.csv"
    cand_csv.write_text("candidate_count\n" + "\n".join(str(v) for v in all_candidate_counts) + "\n", encoding="utf-8")

    if all_candidate_counts:
        plt.figure(figsize=(8, 5), dpi=150)
        plt.hist(all_candidate_counts, bins=30)
        plt.xlabel("Candidate Gaussian Count per Polytope Build")
        plt.ylabel("Frequency")
        plt.title(f"{args.scene} Candidate Gaussian Distribution")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.output_dir / "candidate_count_hist.png")
        plt.close()

    md_lines = []
    md_lines.append("# Extended SplatPlan Eval")
    md_lines.append("")
    md_lines.append(f"- scene: {args.scene}")
    md_lines.append(f"- config: `{args.config}`")
    md_lines.append(f"- trials: {args.trials}")
    md_lines.append(f"- seed: {args.seed}")
    md_lines.append(f"- safety_rate: {summary['safety_rate']:.4f}")
    md_lines.append(f"- feasible_count: {summary['feasible_count']}")
    md_lines.append(f"- infeasible_count: {summary['infeasible_count']}")
    md_lines.append("")
    md_lines.append("## Core Metrics")
    md_lines.append("")
    md_lines.append(f"- plan_time_mean_sec: {summary['plan_time_stats']['mean']}")
    md_lines.append(f"- path_length_mean_m: {summary['traj_length_stats']['mean']}")
    md_lines.append(f"- candidate_count_mean: {summary['candidate_count_stats']['mean']}")
    md_lines.append(f"- corridor_gap_trial_count: {summary['corridor_gap_trial_count']}")
    md_lines.append(f"- infeasible_with_gap_count: {summary['infeasible_with_gap_count']}")
    md_lines.append("")
    md_lines.append("## Case Picks")
    md_lines.append("")
    md_lines.append(f"- first_success_trial_index: {summary['first_success_trial_index']}")
    md_lines.append(f"- first_failure_trial_index: {summary['first_failure_trial_index']}")

    (args.output_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[eval] saved_json={args.output_dir / 'eval_results.json'}")
    print(f"[eval] saved_summary={args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
