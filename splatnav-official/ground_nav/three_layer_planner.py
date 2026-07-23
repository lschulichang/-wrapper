"""Three-layer ground planner: Hybrid A*, safe corridor, precise trajectory."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

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
            raise ValueError("max_curvature_rate_1pm2 must be positive")


@dataclass(frozen=True)
class ThreeLayerPlanResult:
    search_success: bool
    corridor_success: bool
    optimization_success: bool
    grid_safe: bool
    continuous_safe: bool
    curvature_feasible: bool
    curvature_rate_feasible: bool
    tracking_success: bool | None
    fallback_used: bool
    failure_stage: str | None
    failure_reason: str | None
    hybrid_path: HybridPath | None
    corridor: CorridorResult | None
    trajectory_m: TrajectoryResult | None
    trajectory_poses_scene: np.ndarray
    dense_trajectory_poses_scene: np.ndarray
    timings: dict

    @property
    def success(self) -> bool:
        return bool(
            self.search_success
            and self.corridor_success
            and self.optimization_success
            and self.grid_safe
            and self.continuous_safe
            and self.curvature_feasible
            and self.curvature_rate_feasible
        )

    def status(self) -> dict:
        return {
            "search_success": self.search_success,
            "corridor_success": self.corridor_success,
            "optimization_success": self.optimization_success,
            "grid_safe": self.grid_safe,
            "continuous_safe": self.continuous_safe,
            "curvature_feasible": self.curvature_feasible,
            "curvature_rate_feasible": self.curvature_rate_feasible,
            "tracking_success": self.tracking_success,
            "fallback_used": self.fallback_used,
            "complete_planning_success": self.success,
            "failure_stage": self.failure_stage,
            "failure_reason": self.failure_reason,
            "timings": dict(self.timings),
        }


class ThreeLayerGroundPlanner:
    """Keep topology search, safe space, and precise trajectory decoupled."""

    def __init__(
        self,
        hybrid_planner,
        collision_set,
        config: ThreeLayerPlannerConfig | None = None,
    ) -> None:
        self.hybrid_planner = hybrid_planner
        self.collision_set = collision_set
        self.config = ThreeLayerPlannerConfig() if config is None else config
        self.optimizer = CurvatureTrajectoryOptimizer(self.config.optimizer)
        self.scene_scale = float(hybrid_planner.scene_scale)
        if self.scene_scale <= 0.0:
            raise ValueError("Hybrid planner scene_scale must be positive")

    def plan(
        self,
        start_pose: Sequence[float],
        goal_xy: Sequence[float],
    ) -> ThreeLayerPlanResult:
        timings = {}
        hybrid_path = None
        corridor = None
        trajectory = None
        stage = "search"
        try:
            begin = time.perf_counter()
            hybrid_path = self.hybrid_planner.plan(start_pose, goal_xy)
            timings["search_s"] = time.perf_counter() - begin

            stage = "corridor"
            begin = time.perf_counter()
            corridor = build_planar_corridor(
                hybrid_path.poses_scene,
                self.collision_set,
            )
            timings["corridor_s"] = time.perf_counter() - begin

            stage = "optimization"
            reference_m = np.asarray(hybrid_path.poses_scene, dtype=np.float64).copy()
            reference_m[:, :2] /= self.scene_scale
            corridors_m = [
                (
                    np.asarray(A.detach().cpu(), dtype=np.float64),
                    np.asarray(b.detach().cpu(), dtype=np.float64) / self.scene_scale,
                )
                for A, b in corridor.corridors
            ]
            trajectory = self.optimizer.optimize(
                reference_m,
                corridors_m,
                corridor.path_to_corridor,
                start_pose=[
                    float(start_pose[0]) / self.scene_scale,
                    float(start_pose[1]) / self.scene_scale,
                    float(start_pose[2]),
                ],
                goal_xy=np.asarray(goal_xy, dtype=np.float64) / self.scene_scale,
                min_turning_radius=self.config.min_turning_radius_m,
                max_curvature_rate=self.config.max_curvature_rate_1pm2,
            )
            timings["optimization_s"] = trajectory.solve_time
            if not trajectory.success:
                fallback_safe = self._hybrid_fallback_is_safe(hybrid_path)
                return self._result(
                    hybrid_path,
                    corridor,
                    trajectory,
                    timings,
                    optimization_success=False,
                    grid_safe=fallback_safe[0],
                    continuous_safe=fallback_safe[1],
                    curvature_feasible=self._hybrid_curvature_is_feasible(hybrid_path),
                    curvature_rate_feasible=False,
                    fallback_used=all(fallback_safe)
                    and self._hybrid_curvature_is_feasible(hybrid_path),
                    failure_stage=stage,
                    failure_reason=trajectory.failure_reason,
                )

            stage = "verification"
            poses_scene, dense_scene = self._trajectory_scene_arrays(trajectory)
            grid_safe = self._grid_path_is_safe(dense_scene[:, :2])
            continuous_safe = self._continuous_path_is_safe(dense_scene[:, :2])
            curvature_limit = (
                self.config.optimizer.curvature_margin
                / self.config.min_turning_radius_m
            )
            curvature_feasible = bool(
                np.max(np.abs(trajectory.curvature))
                <= curvature_limit + self.config.optimizer.constraint_tolerance
            )
            curvature_rate_feasible = bool(
                np.max(np.abs(trajectory.curvature_rate))
                <= self.config.max_curvature_rate_1pm2
                * self.config.optimizer.curvature_rate_relaxation
                + self.config.optimizer.constraint_tolerance
            )
            if not all(
                (
                    grid_safe,
                    continuous_safe,
                    curvature_feasible,
                    curvature_rate_feasible,
                )
            ):
                fallback_safe = self._hybrid_fallback_is_safe(hybrid_path)
                return self._result(
                    hybrid_path,
                    corridor,
                    trajectory,
                    timings,
                    optimization_success=True,
                    grid_safe=grid_safe,
                    continuous_safe=continuous_safe,
                    curvature_feasible=curvature_feasible,
                    curvature_rate_feasible=curvature_rate_feasible,
                    fallback_used=all(fallback_safe)
                    and self._hybrid_curvature_is_feasible(hybrid_path),
                    failure_stage=stage,
                    failure_reason="optimized trajectory failed final verification",
                )
            return self._result(
                hybrid_path,
                corridor,
                trajectory,
                timings,
                optimization_success=True,
                grid_safe=True,
                continuous_safe=True,
                curvature_feasible=True,
                curvature_rate_feasible=True,
                fallback_used=False,
                failure_stage=None,
                failure_reason=None,
            )
        except Exception as error:
            search_success = hybrid_path is not None
            corridor_success = corridor is not None
            return ThreeLayerPlanResult(
                search_success=search_success,
                corridor_success=corridor_success,
                optimization_success=False,
                grid_safe=False,
                continuous_safe=False,
                curvature_feasible=False,
                curvature_rate_feasible=False,
                tracking_success=None,
                fallback_used=False,
                failure_stage=stage,
                failure_reason=str(error),
                hybrid_path=hybrid_path,
                corridor=corridor,
                trajectory_m=trajectory,
                trajectory_poses_scene=np.empty((0, 3)),
                dense_trajectory_poses_scene=np.empty((0, 3)),
                timings=timings,
            )

    def _result(
        self,
        hybrid_path,
        corridor,
        trajectory,
        timings,
        *,
        optimization_success,
        grid_safe,
        continuous_safe,
        curvature_feasible,
        curvature_rate_feasible,
        fallback_used,
        failure_stage,
        failure_reason,
    ):
        if optimization_success:
            poses_scene, dense_scene = self._trajectory_scene_arrays(trajectory)
        else:
            poses_scene, dense_scene = np.empty((0, 3)), np.empty((0, 3))
        return ThreeLayerPlanResult(
            search_success=True,
            corridor_success=True,
            optimization_success=bool(optimization_success),
            grid_safe=bool(grid_safe),
            continuous_safe=bool(continuous_safe),
            curvature_feasible=bool(curvature_feasible),
            curvature_rate_feasible=bool(curvature_rate_feasible),
            tracking_success=None,
            fallback_used=bool(fallback_used),
            failure_stage=failure_stage,
            failure_reason=failure_reason,
            hybrid_path=hybrid_path,
            corridor=corridor,
            trajectory_m=trajectory,
            trajectory_poses_scene=poses_scene,
            dense_trajectory_poses_scene=dense_scene,
            timings=dict(timings),
        )

    def _trajectory_scene_arrays(self, trajectory):
        if trajectory is None or len(trajectory.poses) == 0:
            return np.empty((0, 3)), np.empty((0, 3))
        poses = np.asarray(trajectory.poses, dtype=np.float64).copy()
        dense = np.asarray(trajectory.dense_poses, dtype=np.float64).copy()
        poses[:, :2] *= self.scene_scale
        dense[:, :2] *= self.scene_scale
        return poses, dense

    def _grid_path_is_safe(self, points_scene):
        return all(
            self.hybrid_planner._state_is_free(
                np.asarray([point[0], point[1], 0.0])
            )
            for point in points_scene
        )

    def _continuous_path_is_safe(self, points_scene):
        with torch.no_grad():
            return all(
                self.collision_set.segment_is_safe(np.stack([first, second]))
                for first, second in zip(points_scene[:-1], points_scene[1:])
            )

    def _hybrid_fallback_is_safe(self, hybrid_path):
        points = hybrid_path.xy
        return self._grid_path_is_safe(points), self._continuous_path_is_safe(points)

    def _hybrid_curvature_is_feasible(self, hybrid_path):
        if len(hybrid_path.primitive_curvatures_1pm) == 0:
            return True
        return bool(
            np.max(np.abs(hybrid_path.primitive_curvatures_1pm))
            <= 1.0 / self.config.min_turning_radius_m + 1e-9
        )
