#!/usr/bin/env python3
"""Run a lightweight Splat-Plan/SFC smoke test on a single scene."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

# Keep upstream repository untouched and imported as a dependency.
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
    config_path: Path,
    scene_name: str,
    method: str,
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    if scene_name not in SCENE_PRESETS:
        raise ValueError(f"Unsupported scene '{scene_name}'. Available: {list(SCENE_PRESETS)}")

    p = SCENE_PRESETS[scene_name]

    robot_config = {
        "radius": p["radius"],
        "vmax": p["vmax"],
        "amax": p["amax"],
    }
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
        raise ValueError("method must be one of: splatplan, sfc-1, sfc-2, sfc-3, sfc-4")

    meta = {
        "scene_name": scene_name,
        "method": method,
        "load_time_sec": load_time,
        "robot_config": robot_config,
        "voxel_config": {
            "lower_bound": p["lower_bound"],
            "upper_bound": p["upper_bound"],
            "resolution": p["resolution"],
        },
        "radius_config": p["radius_config"],
        "mean_config": p["mean_config"].tolist(),
    }

    return planner, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to nerfstudio config.yml")
    parser.add_argument("--scene", required=True, choices=sorted(SCENE_PRESETS.keys()))
    parser.add_argument("--method", default="sfc-1", help="splatplan or sfc-1..sfc-4")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    planner, meta = build_planner(args.config, args.scene, args.method, device)

    p = SCENE_PRESETS[args.scene]
    start = p["mean_config"] + np.array([p["radius_config"], 0.0, 0.0], dtype=np.float32)
    goal = p["mean_config"] + np.array([-p["radius_config"], 0.0, 0.0], dtype=np.float32)

    x = torch.tensor(start, dtype=torch.float32, device=device)
    g = torch.tensor(goal, dtype=torch.float32, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    output = planner.generate_path(x, g)
    if device.type == "cuda":
        torch.cuda.synchronize()
    plan_time = time.time() - t0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "meta": meta,
        "device": str(device),
        "plan_time_sec": plan_time,
        "start": start.tolist(),
        "goal": goal.tolist(),
        "output": to_serializable(output),
    }
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved: {args.output}")
    print(f"plan_time_sec={plan_time:.4f}")


if __name__ == "__main__":
    main()
