"""Single ground-planning mainline: Hybrid A*, corridor, collocation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .corridor_2d import CorridorResult, build_planar_corridor
from .curvature_trajectory_optimizer import (
    CurvatureTrajectoryOptimizer,
    TrajectoryOptimizerConfig,
    TrajectoryResult,
)
from .hybrid_astar import HybridPath


@dataclass(frozen=True)
class ThreeLayerPlannerConfig:
    min_turning_radius_m: float = 0.20
    max_curvature_rate_1pm2: float = 10.0
    optimizer: TrajectoryOptimizerConfig = field(
        default_factory=TrajectoryOptimizerConfig
    )

    def __post_init__(self) -> None:
        if self.min_turning_radius_m <= 0.0:
            raise ValueError("min_turning_radius_m must be positive")
        if self.max_curvature_rate_1pm2 <= 0.0:
            raise ValueError(
                "max_curvature_rate_1pm2 must be positive"
            )


@dataclass(frozen=True)
class ThreeLayerPlanResult:
    """Stage status and the selected mainline or degraded fallback output."""

    search_success: bool
    corridor_success: bool
    optimization_success: bool
    corridor_feasible: bool
    curvature_feasible: bool
    curvature_rate_feasible: bool
    fallback_used: bool
    failure_stage: str | None
    failure_reason: str | None
    hybrid_path: HybridPath | None
    corridor: CorridorResult | None
    trajectory_m: TrajectoryResult | None
    output_poses_scene: np.ndarray
    dense_output_poses_scene: np.ndarray
    timings: dict

    @property
    def success(self) -> bool:
        return bool(
            self.search_success
            and self.corridor_success
            and self.optimization_success
            and self.corridor_feasible
            and self.curvature_feasible
            and self.curvature_rate_feasible
            and not self.fallback_used
        )

    def status(self) -> dict:
        return {
            "search_success": bool(self.search_success),
            "corridor_success": bool(self.corridor_success),
            "optimization_success": bool(self.optimization_success),
            "corridor_feasible": bool(self.corridor_feasible),
            "curvature_feasible": bool(self.curvature_feasible),
            "curvature_rate_feasible": bool(
                self.curvature_rate_feasible
            ),
            "fallback_used": bool(self.fallback_used),
            "complete_planning_success": bool(self.success),
            "failure_stage": self.failure_stage,
            "failure_reason": self.failure_reason,
            "timings": dict(self.timings),
        }


class ThreeLayerGroundPlanner:
    """Execute the only retained planar ground-planning route."""

    def __init__(
        self,
        hybrid_planner,
        collision_set,
        config: ThreeLayerPlannerConfig | None = None,
    ) -> None:
        self.hybrid_planner = hybrid_planner
        self.collision_set = collision_set
        self.config = (
            ThreeLayerPlannerConfig() if config is None else config
        )
        self.optimizer = CurvatureTrajectoryOptimizer(
            self.config.optimizer
        )
        self.scene_scale = float(hybrid_planner.scene_scale)
        if self.scene_scale <= 0.0:
            raise ValueError(
                "Hybrid planner scene_scale must be positive"
            )

    def plan(
        self,
        start_pose: Sequence[float],
        goal_xy: Sequence[float],
    ) -> ThreeLayerPlanResult:
        timings: dict[str, float] = {}
        hybrid_path = None
        corridor = None
        trajectory = None
        stage = "hybrid_astar"
        try:
            begin = time.perf_counter()
            hybrid_path = self.hybrid_planner.plan(
                start_pose, goal_xy
            )
            timings["hybrid_astar_s"] = (
                time.perf_counter() - begin
            )

            stage = "ordered_corridor"
            begin = time.perf_counter()
            corridor = build_planar_corridor(
                hybrid_path.poses_scene,
                self.collision_set,
            )
            timings["ordered_corridor_s"] = (
                time.perf_counter() - begin
            )

            stage = "curvature_collocation"
            reference_m = np.asarray(
                hybrid_path.poses_scene, dtype=np.float64
            ).copy()
            reference_m[:, :2] /= self.scene_scale
            corridors_m = [
                (
                    np.asarray(
                        A.detach().cpu(), dtype=np.float64
                    ),
                    np.asarray(
                        b.detach().cpu(), dtype=np.float64
                    )
                    / self.scene_scale,
                )
                for A, b in corridor.corridors
            ]
            trajectory = self.optimizer.optimize(
                reference_m,
                corridors_m,
                corridor.path_to_corridor,
                start_pose=reference_m[0],
                goal_xy=reference_m[-1, :2],
                min_turning_radius=(
                    self.config.min_turning_radius_m
                ),
                max_curvature_rate=(
                    self.config.max_curvature_rate_1pm2
                ),
            )
            timings["curvature_collocation_s"] = (
                trajectory.solve_time
            )
            if not trajectory.success:
                return self._fallback_result(
                    hybrid_path,
                    corridor,
                    trajectory,
                    timings,
                    stage,
                    trajectory.failure_reason,
                )

            stage = "constraint_verification"
            diagnostics = (
                trajectory.diagnostics[-1]
                if trajectory.diagnostics
                else {}
            )
            tolerance = self.config.optimizer.constraint_tolerance
            corridor_feasible = bool(
                diagnostics.get("dense_corridor_feasible", False)
                and diagnostics.get(
                    "minimum_dense_corridor_margin", -np.inf
                )
                >= -tolerance
            )
            curvature_limit = (
                self.config.optimizer.curvature_margin
                / self.config.min_turning_radius_m
            )
            curvature_feasible = bool(
                len(trajectory.curvature)
                and np.max(np.abs(trajectory.curvature))
                <= curvature_limit + tolerance
            )
            curvature_rate_feasible = bool(
                len(trajectory.curvature_rate)
                and np.max(np.abs(trajectory.curvature_rate))
                <= self.config.max_curvature_rate_1pm2 + tolerance
            )
            if not all(
                (
                    corridor_feasible,
                    curvature_feasible,
                    curvature_rate_feasible,
                )
            ):
                return self._fallback_result(
                    hybrid_path,
                    corridor,
                    trajectory,
                    timings,
                    stage,
                    "optimized path failed final constraint verification",
                )

            output, dense_output = self._trajectory_scene_arrays(
                trajectory
            )
            return ThreeLayerPlanResult(
                search_success=True,
                corridor_success=True,
                optimization_success=True,
                corridor_feasible=True,
                curvature_feasible=True,
                curvature_rate_feasible=True,
                fallback_used=False,
                failure_stage=None,
                failure_reason=None,
                hybrid_path=hybrid_path,
                corridor=corridor,
                trajectory_m=trajectory,
                output_poses_scene=output,
                dense_output_poses_scene=dense_output,
                timings=timings,
            )
        except Exception as error:
            if hybrid_path is not None:
                return self._fallback_result(
                    hybrid_path,
                    corridor,
                    trajectory,
                    timings,
                    stage,
                    str(error),
                )
            return ThreeLayerPlanResult(
                search_success=False,
                corridor_success=False,
                optimization_success=False,
                corridor_feasible=False,
                curvature_feasible=False,
                curvature_rate_feasible=False,
                fallback_used=False,
                failure_stage=stage,
                failure_reason=str(error),
                hybrid_path=None,
                corridor=None,
                trajectory_m=None,
                output_poses_scene=np.empty((0, 3)),
                dense_output_poses_scene=np.empty((0, 3)),
                timings=timings,
            )

    def _fallback_result(
        self,
        hybrid_path,
        corridor,
        trajectory,
        timings,
        failure_stage,
        failure_reason,
    ) -> ThreeLayerPlanResult:
        fallback = np.asarray(
            hybrid_path.poses_scene, dtype=np.float64
        ).copy()
        return ThreeLayerPlanResult(
            search_success=True,
            corridor_success=corridor is not None,
            optimization_success=False,
            corridor_feasible=False,
            curvature_feasible=self._hybrid_curvature_is_feasible(
                hybrid_path
            ),
            curvature_rate_feasible=False,
            fallback_used=True,
            failure_stage=failure_stage,
            failure_reason=(
                None
                if failure_reason is None
                else str(failure_reason)
            ),
            hybrid_path=hybrid_path,
            corridor=corridor,
            trajectory_m=trajectory,
            output_poses_scene=fallback,
            dense_output_poses_scene=fallback.copy(),
            timings=dict(timings),
        )

    def _trajectory_scene_arrays(
        self, trajectory: TrajectoryResult
    ) -> tuple[np.ndarray, np.ndarray]:
        poses = np.asarray(
            trajectory.poses, dtype=np.float64
        ).copy()
        dense = np.asarray(
            trajectory.dense_poses, dtype=np.float64
        ).copy()
        poses[:, :2] *= self.scene_scale
        dense[:, :2] *= self.scene_scale
        return poses, dense

    def _hybrid_curvature_is_feasible(
        self, hybrid_path: HybridPath
    ) -> bool:
        curvatures = np.asarray(
            hybrid_path.primitive_curvatures_1pm,
            dtype=np.float64,
        )
        return bool(
            len(curvatures) == 0
            or np.max(np.abs(curvatures))
            <= 1.0 / self.config.min_turning_radius_m + 1e-9
        )
