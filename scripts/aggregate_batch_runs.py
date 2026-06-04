#!/usr/bin/env python3
"""Aggregate overnight batch results into CSV/Markdown/plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_rows(root: Path, scene: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = sorted(root.glob(f"*_formal_batch_{scene}_iter*/batch_results.json"))
    for fp in candidates:
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        meta = data.get("meta", {})
        results = data.get("results", [])
        iter_name = fp.parent.name
        for item in results:
            rows.append(
                {
                    "iter_dir": iter_name,
                    "scene": meta.get("scene", scene),
                    "method": item.get("method"),
                    "trials": item.get("trials"),
                    "feasible_rate": item.get("feasible_rate"),
                    "avg_plan_time_sec": item.get("avg_plan_time_sec"),
                    "p95_plan_time_sec": item.get("p95_plan_time_sec"),
                    "saved_traj_count": item.get("saved_traj_count"),
                    "source_json": str(fp),
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_csv.write_text("", encoding="utf-8")
        return

    keys = [
        "iter_dir",
        "scene",
        "method",
        "trials",
        "feasible_rate",
        "avg_plan_time_sec",
        "p95_plan_time_sec",
        "saved_traj_count",
        "source_json",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_markdown(rows: list[dict[str, Any]], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Overnight Summary",
        "",
        "| iter_dir | method | feasible_rate | avg_plan_time_sec | p95_plan_time_sec | trials |",
        "|---|---|---:|---:|---:|---:|",
    ]
    if rows:
        for r in rows:
            lines.append(
                f"| {r['iter_dir']} | {r['method']} | {float(r.get('feasible_rate') or 0.0):.3f} | "
                f"{r.get('avg_plan_time_sec')} | {r.get('p95_plan_time_sec')} | {r.get('trials')} |"
            )
    else:
        lines.append("| - | - | - | - | - | - |")

    # Best row by feasible_rate then avg_plan_time
    if rows:
        best = sorted(
            rows,
            key=lambda x: (
                -(float(x.get("feasible_rate") or 0.0)),
                float(x.get("avg_plan_time_sec") or 1e18),
            ),
        )[0]
        lines.extend(
            [
                "",
                "## Best Candidate",
                "",
                f"- iter_dir: `{best['iter_dir']}`",
                f"- method: `{best['method']}`",
                f"- feasible_rate: `{float(best.get('feasible_rate') or 0.0):.3f}`",
                f"- avg_plan_time_sec: `{best.get('avg_plan_time_sec')}`",
                f"- source_json: `{best['source_json']}`",
            ]
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    # group by method with iteration order index
    methods = sorted({r["method"] for r in rows if r.get("method")})
    iter_order = sorted({r["iter_dir"] for r in rows})
    iter_to_idx = {name: i for i, name in enumerate(iter_order)}

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    for m in methods:
        series = sorted((r for r in rows if r.get("method") == m), key=lambda x: iter_to_idx[x["iter_dir"]])
        xs = [iter_to_idx[r["iter_dir"]] for r in series]
        ys = [float(r.get("feasible_rate") or 0.0) for r in series]
        ax1.plot(xs, ys, marker="o", label=m)
    ax1.set_title("Feasible Rate over Iterations")
    ax1.set_xlabel("Iteration Index")
    ax1.set_ylabel("Feasible Rate")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(out_dir / "feasible_rate_over_iters.png", dpi=170)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for m in methods:
        series = sorted((r for r in rows if r.get("method") == m), key=lambda x: iter_to_idx[x["iter_dir"]])
        xs = [iter_to_idx[r["iter_dir"]] for r in series]
        ys = [float(r.get("avg_plan_time_sec") or 0.0) for r in series]
        ax2.plot(xs, ys, marker="o", label=m)
    ax2.set_title("Average Plan Time over Iterations")
    ax2.set_xlabel("Iteration Index")
    ax2.set_ylabel("Avg Plan Time (s)")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "avg_plan_time_over_iters.png", dpi=170)
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_root", type=Path, default=Path("/home/howard/Splat-Nav/runs"))
    parser.add_argument("--scene", type=str, default="old_union")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.runs_root, args.scene)
    rows = sorted(rows, key=lambda x: (x["iter_dir"], x["method"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "overnight_summary.csv")
    write_markdown(rows, args.output_dir / "overnight_summary.md")
    plot(rows, args.output_dir)
    print(f"[agg] rows={len(rows)}")
    print(f"[agg] output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
