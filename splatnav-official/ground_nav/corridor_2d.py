"""Paper-style circle-ellipse collision sets and convex corridors in 2D."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ellipsoids.intersection_utils import compute_K_parameters_sphere, compute_sphere_ellipsoid_Q


def compute_rotated_rectangle(segment: torch.Tensor, half_margin: float):
    delta = segment[1] - segment[0]
    length = torch.linalg.norm(delta)
    if float(length) <= 0:
        raise ValueError("segment endpoints must be distinct")
    e1 = delta / length
    e2 = torch.stack([-e1[1], e1[0]])
    midpoint = 0.5 * (segment[0] + segment[1])
    A = torch.stack([e1, e2, -e1, -e2])
    extents = torch.stack([
        length / 2.0 + torch.as_tensor(half_margin, device=segment.device, dtype=segment.dtype),
        torch.as_tensor(half_margin, device=segment.device, dtype=segment.dtype),
    ])
    b = torch.cat([A[:2] @ midpoint + extents, A[2:] @ midpoint + extents])
    return A, b, midpoint


def ellipse_halfspace_intersection(means, rots, scales, A, b):
    if means.shape[0] == 0:
        return torch.zeros(0, dtype=torch.bool, device=means.device)
    numerator = means @ A.T - b[None]
    rotated_normals = rots.transpose(1, 2) @ A.T
    support = torch.linalg.norm(rotated_normals * scales[..., None], dim=1).clamp_min(1e-12)
    return (numerator / support <= 1.0).all(dim=-1)


def continuous_circle_ellipse_test(segment, rots, scales, means, radius: float, iterations: int = 10):
    """Dimension-independent implementation of the paper's Algorithm 1."""

    if means.shape[0] == 0:
        empty = torch.empty(0, device=segment.device, dtype=segment.dtype)
        return {
            "seedpoint": torch.empty((0, 2), device=segment.device, dtype=segment.dtype),
            "deltas": torch.empty((0, 2), device=segment.device, dtype=segment.dtype),
            "Q_opt": torch.empty((0, 2, 2), device=segment.device, dtype=segment.dtype),
            "K_opt": empty,
            "mu_A": means,
            "is_not_intersect": empty.bool(),
        }
    x0 = segment[0]
    delta_x = segment[1] - segment[0]
    lower = torch.zeros(means.shape[0], device=means.device, dtype=means.dtype)
    upper = torch.ones_like(lower)
    lambdas, Phi, kappa = compute_K_parameters_sphere(rots, scales, float(radius))
    for _ in range(iterations):
        s = 0.5 * (lower + upper)
        Q_sqrt = compute_sphere_ellipsoid_Q(rots, scales, float(radius), s)
        q_delta = (Q_sqrt @ delta_x[:, None]).squeeze(-1)
        q_diff = (Q_sqrt @ (means - x0)[..., None]).squeeze(-1)
        denominator = torch.sum(q_delta.square(), dim=-1)
        numerator = torch.sum(q_delta * q_diff, dim=-1)
        t = torch.where(denominator > 1e-18, numerator / denominator, torch.zeros_like(numerator)).clamp(0.0, 1.0)
        seedpoint = x0[None] + t[:, None] * delta_x[None]
        delta = seedpoint - means
        v = torch.sum(delta[..., None] * Phi, dim=-2)
        grad_num = kappa - 2.0 * (s * kappa)[:, None] - (s.square())[:, None] * (lambdas - kappa)
        grad_den = (kappa + s[:, None] * (lambdas - kappa)).square().clamp_min(1e-18)
        gradient = torch.sum((grad_num / grad_den) * v.square(), dim=-1)
        lower = torch.where(gradient >= 0.0, s, lower)
        upper = torch.where(gradient < 0.0, s, upper)
    Q = Q_sqrt.transpose(-2, -1) @ Q_sqrt
    K = torch.einsum("ni,nij,nj->n", delta, Q, delta)
    return {
        "seedpoint": seedpoint,
        "deltas": delta,
        "Q_opt": Q,
        "K_opt": K,
        "mu_A": means,
        "is_not_intersect": K >= 1.0,
    }


class PlanarCollisionSet:
    def __init__(self, ellipses, radius: float, corridor_margin: float, iterations: int = 10):
        if radius < 0 or corridor_margin < 0:
            raise ValueError("radius and corridor_margin must be non-negative")
        self.ellipses = ellipses
        self.radius = float(radius)
        self.corridor_margin = float(corridor_margin)
        self.iterations = int(iterations)
        self.device = ellipses.means.device

    def candidates(self, segment):
        segment = torch.as_tensor(segment, dtype=self.ellipses.means.dtype, device=self.device)
        A, b, midpoint = compute_rotated_rectangle(segment, self.radius + self.corridor_margin)
        keep = ellipse_halfspace_intersection(
            self.ellipses.means, self.ellipses.rots, self.ellipses.scales, A, b
        )
        return {
            "ids": self.ellipses.ids[keep],
            "means": self.ellipses.means[keep],
            "rots": self.ellipses.rots[keep],
            "scales": self.ellipses.scales[keep],
            "A_box": A,
            "b_box": b,
            "b_box_shrunk": b - self.radius,
            "segment": segment,
            "midpoint": midpoint,
        }

    def exact_test(self, segment, candidate_data=None):
        data = self.candidates(segment) if candidate_data is None else candidate_data
        return continuous_circle_ellipse_test(
            data["segment"], data["rots"], data["scales"], data["means"], self.radius, self.iterations
        )

    def segment_is_safe(self, segment) -> bool:
        result = self.exact_test(segment)
        return bool(torch.all(result["is_not_intersect"]).item())


def _supporting_line(delta, Q, K, mean):
    delta_Q = Q @ delta
    rhs = torch.sqrt(K.clamp_min(0.0)) + torch.dot(delta_Q, mean)
    return -delta_Q, -rhs


@dataclass
class PlanarCorridor:
    polygons: list[tuple[torch.Tensor, torch.Tensor]]
    source_segment_indices: list[int]
    diagnostics: list[dict]


def build_planar_corridor(path: np.ndarray, collision_set: PlanarCollisionSet, tolerance: float = 1e-6):
    path = np.asarray(path)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("path must have shape (N, 2), N >= 2")
    segments = np.stack([path[:-1], path[1:]], axis=1)
    polygons = []
    source_indices = []
    diagnostics = []
    current = None
    for index, segment_np in enumerate(segments):
        segment = torch.as_tensor(segment_np, dtype=collision_set.ellipses.means.dtype, device=collision_set.device)
        contained = False
        if current is not None:
            contained = bool(torch.all(current[0] @ segment.T <= current[1][:, None] + tolerance).item())
        create = index == 0 or index == len(segments) - 1 or not contained
        row = {"segment_index": index, "contained": contained, "created_polygon": create}
        if not create:
            diagnostics.append(row)
            continue
        data = collision_set.candidates(segment)
        exact = collision_set.exact_test(segment, data)
        if not bool(torch.all(exact["is_not_intersect"]).item()):
            raise RuntimeError(f"simplified segment {index} is not circle-ellipse safe")
        means, rots, scales = data["means"], data["rots"], data["scales"]
        deltas, matrices, values = exact["deltas"], exact["Q_opt"], exact["K_opt"]
        cut_A, cut_b = [], []
        while values.numel() > 0:
            selected = int(torch.argmin(values).item())
            a, b = _supporting_line(deltas[selected], matrices[selected], values[selected], means[selected])
            cut_A.append(a)
            cut_b.append(b)
            norm = torch.linalg.norm(a).clamp_min(1e-12)
            keep = ellipse_halfspace_intersection(
                means, rots, scales, (a / norm)[None], (b / norm + collision_set.radius)[None]
            )
            keep[selected] = False
            means, rots, scales = means[keep], rots[keep], scales[keep]
            deltas, matrices, values = deltas[keep], matrices[keep], values[keep]
        if cut_A:
            A = torch.cat([torch.stack(cut_A), data["A_box"]], dim=0)
            b = torch.cat([torch.stack(cut_b), data["b_box_shrunk"]], dim=0)
        else:
            A, b = data["A_box"], data["b_box_shrunk"]
        norms = torch.linalg.norm(A, dim=-1).clamp_min(1e-12)
        A, b = A / norms[:, None], b / norms
        if not bool(torch.all(A @ segment.T <= b[:, None] + tolerance).item()):
            raise RuntimeError(f"polygon {len(polygons)} does not contain its seed segment")
        current = (A, b)
        polygons.append(current)
        source_indices.append(index)
        row.update({"candidate_count": int(data["means"].shape[0]), "halfspace_count": int(A.shape[0])})
        diagnostics.append(row)
    return PlanarCorridor(polygons, source_indices, diagnostics)
