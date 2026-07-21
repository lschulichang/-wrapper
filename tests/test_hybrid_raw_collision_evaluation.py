from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "splatnav-official"))

from run_hybrid_raw_collision_evaluation import (  # noqa: E402
    build_summary,
    continuous_segment_statistics,
    evaluate_path,
    grid_sample_statistics,
    select_manifest_rows,
)


class FakeGrid:
    def __init__(self):
        self.lower_center = torch.tensor([0.0, 0.0])
        self.cell_sizes = torch.tensor([1.0, 1.0])
        self.shape = (4, 4)
        self.occupied = torch.zeros(self.shape, dtype=torch.bool)
        self.occupied[2, 2] = True


class FakeCollisionSet:
    def segment_is_safe(self, segment):
        return not (
            float(segment[0, 0]) < 1.5 <= float(segment[1, 0])
        )


class HybridRawCollisionEvaluationTest(unittest.TestCase):
    def test_manifest_selection_is_balanced_and_free_heading(self):
        rows = []
        for layer in ("near", "medium", "far"):
            for index in range(4):
                rows.append(
                    {
                        "trial_id": f"{layer}_{index}",
                        "distance_layer": layer,
                        "goal_heading_mode": "free",
                    }
                )
        selected = select_manifest_rows(rows, 6)
        self.assertEqual(
            {layer: sum(row["distance_layer"] == layer for row in selected)
             for layer in ("near", "medium", "far")},
            {"near": 2, "medium": 2, "far": 2},
        )

    def test_grid_and_continuous_statistics_report_violations(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 2.0]])
        grid = grid_sample_statistics(points, FakeGrid())
        continuous = continuous_segment_statistics(points, FakeCollisionSet())
        self.assertFalse(grid["grid_collision_free"])
        self.assertEqual(grid["grid_unsafe_sample_indices"], [2])
        self.assertFalse(continuous["continuous_collision_free"])
        self.assertEqual(continuous["continuous_unsafe_segment_indices"], [1])

    def test_evaluate_path_combines_both_safety_models(self):
        path = SimpleNamespace(
            poses_scene=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.4]]
            ),
            primitive_curvatures_1pm=np.array([0.0, 2.5]),
        )
        result = evaluate_path(path, FakeGrid(), FakeCollisionSet(), scale=1.0)
        self.assertTrue(result["grid_collision_free"])
        self.assertFalse(result["continuous_collision_free"])
        self.assertFalse(result["direct_path_safe"])
        self.assertAlmostEqual(result["max_abs_primitive_curvature_1pm"], 2.5)

    def test_summary_keeps_failures_in_denominator(self):
        records = [
            {
                "distance_layer": layer,
                "search_success": True,
                "grid_collision_free": True,
                "continuous_collision_free": layer != "far",
                "direct_path_safe": layer != "far",
            }
            for layer in ("near", "medium", "far")
        ]
        summary = build_summary(records, 3)
        self.assertEqual(summary["search_success_count"], 3)
        self.assertEqual(summary["continuous_collision_free_count"], 2)
        self.assertAlmostEqual(
            summary["continuous_collision_free_rate_over_all_trials"], 2 / 3
        )


if __name__ == "__main__":
    unittest.main()
