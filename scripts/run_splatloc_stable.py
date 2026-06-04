#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

OFFICIAL_ROOT_DEFAULT = Path(os.environ.get("SPLATNAV_OFFICIAL_ROOT", "/data/howard/repos/splatnav-official")).resolve()
if str(OFFICIAL_ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_ROOT_DEFAULT))

from ns_utils.nerfstudio_utils import GaussianSplat
from pose_estimator.utils import SE3error, execute_PnP_RANSAC, load_dataset, POI_Detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable Splat-Loc runner with fallback when PnP fails")
    parser.add_argument("--official_root", type=Path, default=OFFICIAL_ROOT_DEFAULT)
    parser.add_argument("--scene", type=str, default="old_union")
    parser.add_argument(
        "--config_rel",
        type=Path,
        default=Path("outputs/old_union2/splatfacto/2024-09-02_151414/config.yml"),
    )
    parser.add_argument("--dataset_rel", type=Path, default=Path("data/old_union2"))
    parser.add_argument("--detector", type=str, choices=["sift", "orb", "surf"], default="sift")
    parser.add_argument("--max_frames", type=int, default=0, help="0 means all frames")
    parser.add_argument("--init_mode", type=str, choices=["previous", "gt"], default="previous")
    return parser.parse_args()


def choose_detector(name: str) -> POI_Detector:
    if name == "sift":
        return POI_Detector.SIFT
    if name == "orb":
        return POI_Detector.ORB
    if name == "surf":
        return POI_Detector.SURF
    raise ValueError(f"Unsupported detector: {name}")


def main() -> int:
    args = parse_args()

    official_root = args.official_root.resolve()
    config_path = (official_root / args.config_rel).resolve()
    dataset_path = (official_root / args.dataset_rel).resolve()
    scene_name = args.scene
    detector = choose_detector(args.detector)

    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gsplat = GaussianSplat(
        config_path=config_path,
        res_factor=None,
        test_mode="test",
        dataset_mode="val",
        device=device,
    )

    _, _, cam_K = gsplat.get_camera_intrinsics()
    cam_K = cam_K.to(device)

    dataset = load_dataset(data_path=dataset_path, dataset_mode="all")
    num_total = len(dataset)
    num_trials = min(num_total, args.max_frames) if args.max_frames > 0 else num_total

    dataset_images = [dataset.get_image_float32(i) for i in range(num_trials)]

    pose_error = np.inf * np.ones((num_trials + 1, 2), dtype=np.float64)
    est_pose_over_time = []
    fallback_count = 0
    success_count = 0
    local_est_pose = None

    print(f"[stable-loc] official_root={official_root}")
    print(f"[stable-loc] config={config_path}")
    print(f"[stable-loc] dataset={dataset_path}")
    print(f"[stable-loc] detector={args.detector}")
    print(f"[stable-loc] frames={num_trials}")

    for idx in range(num_trials):
        cam_rgb = dataset_images[idx].to(device)

        gt_pose = torch.eye(4)
        gt_pose[:3] = dataset.cameras.camera_to_worlds[idx].to(device)
        gt_pose = gt_pose.cpu().numpy()

        if args.init_mode == "gt" or idx == 0 or local_est_pose is None:
            init_guess = torch.tensor(gt_pose, device=device).float()
        else:
            init_guess = torch.tensor(local_est_pose, device=device).float()

        try:
            local_est_pose = execute_PnP_RANSAC(
                gsplat,
                init_guess,
                camera_intrinsics_K=cam_K,
                rgb_input=cam_rgb,
                feature_detector=detector,
                save_image=False,
                pnp_figure_filename="figures/pnp_init_guess.png",
                print_stats=False,
                visualize_PnP_matches=False,
                pnp_matches_figure_filename="figures/pnp_matches.png",
            )
            success_count += 1
        except Exception as exc:
            fallback_count += 1
            local_est_pose = init_guess.cpu().numpy()
            print(f"[stable-loc] frame={idx + 1:05d} fallback={type(exc).__name__} msg={exc}")

        err = SE3error(gt_pose, local_est_pose)
        pose_error[idx, :] = err
        est_pose_over_time.append(local_est_pose)

        if idx == 0 or (idx + 1) % 10 == 0 or idx == num_trials - 1:
            print(
                f"[stable-loc] progress={idx + 1}/{num_trials} "
                f"rot_err_deg={err[0]:.6f} trans_err_m={err[1]:.6f}"
            )

    fig_dir = official_root / "figures" / scene_name / "test_runs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(6, 6), dpi=200)
    ax1.plot(pose_error[:, 0], label="Rotation Error")
    ax2.plot(pose_error[:, 1], label="Translation Error")
    ax1.set_xlabel("Timestep")
    ax2.set_xlabel("Timestep")
    ax1.set_ylabel("Rotation Error (degrees)")
    ax2.set_ylabel("Translation Error (m)")
    ax1.grid()
    ax2.grid()
    fig.savefig(fig_dir / "error_stats.png")
    plt.close(fig)

    result_dir = official_root / "results" / scene_name / "test_runs"
    result_dir.mkdir(parents=True, exist_ok=True)

    est_json = {f"frame_{idx + 1:05d}": pose.tolist() for idx, pose in enumerate(est_pose_over_time)}
    with open(result_dir / "est_pose.json", "w", encoding="utf-8") as fh:
        json.dump(est_json, fh, indent=2)

    meta = {
        "scene": scene_name,
        "frames_total": num_trials,
        "success_count": success_count,
        "fallback_count": fallback_count,
        "detector": args.detector,
        "init_mode": args.init_mode,
    }
    with open(result_dir / "stable_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[stable-loc] done frames_total={num_trials} success={success_count} fallback={fallback_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
