#!/usr/bin/env python3
"""Run official-style multi-trial SplatPlan/SFC batch experiments."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

# Keep upstream untouched and imported as dependency.
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

from SFC.corridor_utils import SafeFlightCorridor  # noqa: E402
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


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def build_planner(
    config_path: Path, scene_name: str, method: str, device: torch.device
) -> tuple[Any, dict[str, Any]]:
    if scene_name not in SCENE_PRESETS:
        raise ValueError(f"Unsupported scene '{scene_name}', available: {sorted(SCENE_PRESETS)}")

    p = SCENE_PRESETS[scene_name]
    robot_config = {"radius": p["radius"], "vmax": p["vmax"], "amax": p["amax"]}
    voxel_config = {
        "lower_bound": torch.tensor(p["lower_bound"], device=device),
        "upper_bound": torch.tensor(p["upper_bound"], device=device),
        "resolution": p["resolution"],
    }

    t0 = time.time()
    gsplat = GSplatLoader(config_path, device)
    load_time = time.time() - t0

    spline_planner = SplinePlanner(spline_deg=6, device=device)
    if method == "splatplan":
        planner = SplatPlan(gsplat, robot_config, voxel_config, spline_planner, device)
    elif method.startswith("sfc-"):
        mode = int(method.split("-")[1])
        planner = SafeFlightCorridor(gsplat, robot_config, voxel_config, spline_planner, device, mode=mode)
    else:
        raise ValueError("method must be splatplan or sfc-1..sfc-4")

    return planner, {
        "scene_name": scene_name,
        "method": method,
        "load_time_sec": load_time,
        "robot_config": robot_config,
        "voxel_config": {
            "lower_bound": p["lower_bound"],
            "upper_bound": p["upper_bound"],
            "resolution": p["resolution"],
        },
    }


def compact_output(out: Any, include_traj: bool) -> dict[str, Any]:
    if not isinstance(out, dict):
        return {"raw_type": str(type(out)), "raw": to_serializable(out)}

    keep_keys = [
        "feasible",
        "num_polytopes",
        "times_astar",
        "times_collision_set",
        "times_ellipsoid",
        "times_polytope",
        "times_opt",
    ]
    if include_traj:
        keep_keys.extend(["path", "traj"])

    compact = {k: to_serializable(out[k]) for k in keep_keys if k in out}
    if "path" in compact and isinstance(compact["path"], list):
        compact["path_len"] = len(compact["path"])
    if "traj" in compact and isinstance(compact["traj"], list):
        compact["traj_len"] = len(compact["traj"])
    return compact


def run_method(
    planner: Any,
    scene_name: str,
    method: str,
    trials: int,
    seed: int,
    keep_traj_limit: int,
    device: torch.device,
) -> dict[str, Any]:
    p = SCENE_PRESETS[scene_name]

    t = np.linspace(0.0, 2.0 * math.pi, trials, endpoint=False, dtype=np.float32)
    tz = 10.0 * np.linspace(0.0, 2.0 * math.pi, trials, endpoint=False, dtype=np.float32)

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

    rng = np.random.default_rng(seed)
    order = np.arange(trials)
    rng.shuffle(order)

    trial_records: list[dict[str, Any]] = []
    feasible_count = 0
    plan_times: list[float] = []
    saved_traj_count = 0

    for i in order.tolist():
        start = starts[i]
        goal = goals[i]
        x = torch.tensor(start, dtype=torch.float32, device=device)
        g = torch.tensor(goal, dtype=torch.float32, device=device)

        try:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            out = planner.generate_path(x, g)
            if device.type == "cuda":
                torch.cuda.synchronize()
            plan_time = time.time() - t0

            include_traj = saved_traj_count < keep_traj_limit
            out_c = compact_output(out, include_traj=include_traj)
            feasible = bool(out_c.get("feasible", False))
            feasible_count += int(feasible)
            plan_times.append(plan_time)
            if include_traj and ("traj" in out_c or "path" in out_c):
                saved_traj_count += 1

            trial_records.append(
                {
                    "trial_index": i,
                    "start": start.tolist(),
                    "goal": goal.tolist(),
                    "plan_time_sec": plan_time,
                    "output": out_c,
                }
            )
        except Exception as exc:  # noqa: BLE001
            trial_records.append(
                {
                    "trial_index": i,
                    "start": start.tolist(),
                    "goal": goal.tolist(),
                    "plan_time_sec": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if plan_times:
        avg_time = float(np.mean(plan_times))
        p95_time = float(np.percentile(plan_times, 95))
    else:
        avg_time = None
        p95_time = None

    return {
        "method": method,
        "scene_name": scene_name,
        "trials": trials,
        "feasible_count": feasible_count,
        "feasible_rate": feasible_count / float(trials),
        "avg_plan_time_sec": avg_time,
        "p95_plan_time_sec": p95_time,
        "saved_traj_count": saved_traj_count,
        "trial_records": trial_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to nerfstudio config.yml")
    parser.add_argument("--scene", default="old_union", choices=sorted(SCENE_PRESETS.keys()))
    parser.add_argument(
        "--methods",
        default="sfc-1,sfc-2,sfc-3,sfc-4,splatplan",
        help="Comma separated methods, e.g. sfc-1,sfc-2,splatplan",
    )
    parser.add_argument("--trials", type=int, default=120, help="Trials per method")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--keep_traj_limit",
        type=int,
        default=20,
        help="Max number of trial records per method that keep path/traj arrays",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    run_started = time.time()

    method_results = []
    for method in methods:
        print(f"[batch] building planner method={method}")
        planner, planner_meta = build_planner(args.config, args.scene, method, device)
        print(f"[batch] running method={method} trials={args.trials}")
        result = run_method(planner, args.scene, method, args.trials, args.seed, args.keep_traj_limit, device)
        result["planner_meta"] = planner_meta
        method_results.append(result)

    out = {
        "meta": {
            "scene": args.scene,
            "config": str(args.config),
            "methods": methods,
            "trials_per_method": args.trials,
            "seed": args.seed,
            "device": str(device),
            "run_time_sec": time.time() - run_started,
        },
        "results": method_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[batch] saved={args.output}")
    for r in method_results:
        print(
            f"[batch] {r['method']} feasible_rate={r['feasible_rate']:.3f} "
            f"avg_plan_time_sec={r['avg_plan_time_sec']}"
        )


if __name__ == "__main__":
    main()
