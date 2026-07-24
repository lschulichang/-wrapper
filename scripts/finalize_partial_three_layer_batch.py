#!/usr/bin/env python3
"""Finalize charts and statistics for a deliberately stopped batch run."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_three_layer_medium_batch import (  # noqa: E402
    build_summary,
    save_json,
    save_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--requested-sample-count",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--termination-exit-code",
        type=int,
        default=143,
    )
    args = parser.parse_args()

    record_path = args.output_dir / "records.jsonl"
    records = [
        json.loads(line)
        for line in record_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    selected = json.loads(
        (args.output_dir / "selected_goals.json").read_text(
            encoding="utf-8"
        )
    )
    completed_ids = {record["trial_id"] for record in records}
    unfinished_ids = [
        row["trial_id"]
        for row in selected
        if row["trial_id"] not in completed_ids
    ]
    paths = []
    for record in records:
        relative_path = record.get("output_path_file")
        if relative_path is None:
            continue
        poses = np.load(args.output_dir / relative_path)
        paths.append(
            (
                record["trial_id"],
                poses,
                bool(record["complete_success"]),
            )
        )

    curvature_limit = 4.75
    rate_limit = 10.0
    protocol = {
        "scene": "old_union",
        "distance_layer": "medium",
        "goal_heading_mode": "free",
        "corridor_reference": "motion_primitive_boundaries",
        "collocation_method": "hermite_simpson",
        "solver": "SLSQP",
        "curvature_limit_1pm": curvature_limit,
        "max_curvature_rate_1pm2": rate_limit,
        "optimizer_nodes": 12,
        "optimizer_max_iterations": 500,
        "optimizer_max_refinements": 2,
        "stops_after": "final_constraint_verification",
        "partial_run": True,
        "statistics_scope": "completed_records_only",
    }
    visualizations = save_visualizations(
        records,
        args.output_dir,
        paths,
        curvature_limit,
        rate_limit,
    )
    summary = build_summary(
        records,
        {
            "available": False,
            "reason": (
                "batch was terminated before the normal final "
                "summary was written"
            ),
        },
        protocol,
    )
    summary.update(
        {
            "requested_sample_count": int(
                args.requested_sample_count
            ),
            "completed_sample_count": int(len(records)),
            "unfinished_sample_count": int(
                args.requested_sample_count - len(records)
            ),
            "completion_fraction": float(
                len(records) / args.requested_sample_count
            ),
            "terminated_early": True,
            "termination_exit_code": int(
                args.termination_exit_code
            ),
            "last_completed_trial": (
                records[-1]["trial_id"] if records else None
            ),
            "unfinished_trial_ids": unfinished_ids,
            "finalized_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "visualizations": visualizations,
        }
    )
    save_json(args.output_dir / "summary.json", summary)
    save_json(
        args.output_dir / "termination.json",
        {
            "terminated_early": True,
            "termination_exit_code": int(
                args.termination_exit_code
            ),
            "statistics_scope": "completed_records_only",
            "completed_trial_ids": [
                record["trial_id"] for record in records
            ],
            "unfinished_trial_ids": unfinished_ids,
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
