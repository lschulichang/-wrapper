"""Simplify a Hybrid A* path with its motion-primitive structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrimitivePathSimplification:
    """Primitive boundaries and maximal equal-curvature runs."""

    boundary_poses_scene: np.ndarray
    merged_poses_scene: np.ndarray
    merged_curvatures_1pm: np.ndarray
    curvature_runs: list[dict]
    dense_pose_count: int
    primitive_count: int

    def summary(self) -> dict:
        return {
            "dense_pose_count": int(self.dense_pose_count),
            "primitive_count": int(self.primitive_count),
            "primitive_boundary_pose_count": int(
                len(self.boundary_poses_scene)
            ),
            "curvature_run_count": int(len(self.curvature_runs)),
            "merged_pose_count": int(len(self.merged_poses_scene)),
            "dense_to_boundary_removed_count": int(
                self.dense_pose_count
                - len(self.boundary_poses_scene)
            ),
            "boundary_to_merged_removed_count": int(
                len(self.boundary_poses_scene)
                - len(self.merged_poses_scene)
            ),
        }


def simplify_hybrid_path_by_primitives(
    hybrid_path,
    *,
    curvature_tolerance_1pm: float = 1e-9,
) -> PrimitivePathSimplification:
    """Use primitive boundaries, then merge adjacent equal-curvature runs.

    A run containing primitives ``[i, j]`` is represented by its start
    boundary ``i`` and end boundary ``j + 1``.  Headings at retained
    boundaries are preserved exactly.
    """

    if curvature_tolerance_1pm < 0.0:
        raise ValueError(
            "curvature_tolerance_1pm must be non-negative"
        )
    dense = np.asarray(
        hybrid_path.poses_scene, dtype=np.float64
    )
    boundaries = np.asarray(
        hybrid_path.primitive_boundary_poses_scene,
        dtype=np.float64,
    )
    curvatures = np.asarray(
        hybrid_path.primitive_curvatures_1pm,
        dtype=np.float64,
    ).reshape(-1)
    if dense.ndim != 2 or dense.shape[1] != 3 or len(dense) < 2:
        raise ValueError(
            "Hybrid path dense poses must have shape (N, 3), N >= 2"
        )
    if (
        boundaries.ndim != 2
        or boundaries.shape[1] != 3
        or len(boundaries) != len(curvatures) + 1
    ):
        raise ValueError(
            "primitive boundaries must have one more pose than curvatures"
        )
    if not np.all(np.isfinite(boundaries)) or not np.all(
        np.isfinite(curvatures)
    ):
        raise ValueError(
            "primitive boundaries and curvatures must be finite"
        )
    if not np.allclose(
        boundaries[0], dense[0], atol=1e-10, rtol=0.0
    ) or not np.allclose(
        boundaries[-1], dense[-1], atol=1e-10, rtol=0.0
    ):
        raise ValueError(
            "primitive boundaries must preserve dense-path endpoints"
        )
    if len(curvatures) == 0:
        raise ValueError("Hybrid path contains no motion primitive")

    runs: list[dict] = []
    retained = [boundaries[0].copy()]
    merged_curvatures = []
    run_start = 0
    for primitive_index in range(1, len(curvatures)):
        if np.isclose(
            curvatures[primitive_index],
            curvatures[run_start],
            atol=curvature_tolerance_1pm,
            rtol=0.0,
        ):
            continue
        run_end = primitive_index - 1
        retained.append(boundaries[run_end + 1].copy())
        merged_curvatures.append(float(curvatures[run_start]))
        runs.append(
            {
                "start_primitive_index": int(run_start),
                "end_primitive_index": int(run_end),
                "primitive_count": int(run_end - run_start + 1),
                "curvature_1pm": float(curvatures[run_start]),
                "start_boundary_index": int(run_start),
                "end_boundary_index": int(run_end + 1),
            }
        )
        run_start = primitive_index

    run_end = len(curvatures) - 1
    retained.append(boundaries[-1].copy())
    merged_curvatures.append(float(curvatures[run_start]))
    runs.append(
        {
            "start_primitive_index": int(run_start),
            "end_primitive_index": int(run_end),
            "primitive_count": int(run_end - run_start + 1),
            "curvature_1pm": float(curvatures[run_start]),
            "start_boundary_index": int(run_start),
            "end_boundary_index": int(run_end + 1),
        }
    )
    merged = np.asarray(retained, dtype=np.float64)
    if len(merged) != len(merged_curvatures) + 1:
        raise RuntimeError(
            "merged boundary count is inconsistent with curvature runs"
        )
    return PrimitivePathSimplification(
        boundary_poses_scene=boundaries.copy(),
        merged_poses_scene=merged,
        merged_curvatures_1pm=np.asarray(
            merged_curvatures, dtype=np.float64
        ),
        curvature_runs=runs,
        dense_pose_count=int(len(dense)),
        primitive_count=int(len(curvatures)),
    )
