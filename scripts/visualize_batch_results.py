#!/usr/bin/env python3
"""Generate PNG visualizations from batch experiment JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def collect_plan_times(records: list[dict]) -> list[float]:
    vals = []
    for rec in records:
        v = rec.get("plan_time_sec")
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals


def collect_example_trajs(records: list[dict], max_count: int = 10) -> list[np.ndarray]:
    trajs = []
    for rec in records:
        out = rec.get("output", {})
        traj = out.get("traj")
        if isinstance(traj, list) and len(traj) >= 2:
            arr = np.asarray(traj, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                trajs.append(arr[:, :3])
        if len(trajs) >= max_count:
            break
    return trajs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Batch JSON path")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory for PNG outputs")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results: list[dict] = data.get("results", [])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    methods = [r.get("method", "unknown") for r in results]
    feasible_rates = [float(r.get("feasible_rate", 0.0)) for r in results]

    # Figure 1: feasible rate bar
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(methods, feasible_rates, color="#3B82F6")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("Feasible Rate")
    ax1.set_title("Feasible Rate by Method")
    ax1.grid(axis="y", alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(args.output_dir / "feasible_rate.png", dpi=160)
    plt.close(fig1)

    # Figure 2: plan time boxplot
    time_groups = []
    time_labels = []
    for r in results:
        times = collect_plan_times(r.get("trial_records", []))
        if times:
            time_groups.append(times)
            time_labels.append(r.get("method", "unknown"))

    if time_groups:
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        ax2.boxplot(time_groups, labels=time_labels, showfliers=False)
        ax2.set_ylabel("Plan Time (s)")
        ax2.set_title("Plan Time Distribution")
        ax2.grid(axis="y", alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(args.output_dir / "plan_time_boxplot.png", dpi=160)
        plt.close(fig2)

    # Figure 3: XY trajectory overlay (first few per method)
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    palette = plt.get_cmap("tab10")
    plotted_any = False
    for idx, r in enumerate(results):
        method = r.get("method", f"m{idx}")
        trajs = collect_example_trajs(r.get("trial_records", []), max_count=6)
        color = palette(idx % 10)
        for t in trajs:
            ax3.plot(t[:, 0], t[:, 1], color=color, alpha=0.4, linewidth=1.2)
            plotted_any = True
        if trajs:
            ax3.plot([], [], color=color, label=method)

    if plotted_any:
        ax3.set_title("Trajectory XY Overlay (Examples)")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.grid(alpha=0.3)
        ax3.legend()
        ax3.axis("equal")
        fig3.tight_layout()
        fig3.savefig(args.output_dir / "traj_xy_overlay.png", dpi=180)
    plt.close(fig3)

    # Markdown summary for quick read
    summary_lines = [
        "# Batch Summary",
        "",
        f"- input: `{args.input}`",
        "",
        "| method | feasible_rate | avg_plan_time_sec | p95_plan_time_sec |",
        "|---|---:|---:|---:|",
    ]
    for r in results:
        summary_lines.append(
            f"| {r.get('method')} | {float(r.get('feasible_rate', 0.0)):.3f} | "
            f"{r.get('avg_plan_time_sec')} | {r.get('p95_plan_time_sec')} |"
        )

    (args.output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[viz] output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
