"""Height-filtered orthogonal projections of 3D Gaussian ellipsoids."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlanarGaussianSet:
    ids: torch.Tensor
    means: torch.Tensor
    covs: torch.Tensor
    rots: torch.Tensor
    scales: torch.Tensor
    z_min: float
    z_max: float
    confidence: float

    @classmethod
    def from_gsplat(cls, gsplat, z_min: float, z_max: float, confidence: float = 1.0):
        if z_min >= z_max:
            raise ValueError("z_min must be smaller than z_max")
        if confidence <= 0:
            raise ValueError("confidence must be positive")
        z_sigma = confidence * torch.sqrt(torch.clamp(gsplat.covs[:, 2, 2], min=0.0))
        mask = ((gsplat.means[:, 2] + z_sigma) > z_min) & ((gsplat.means[:, 2] - z_sigma) < z_max)
        ids = torch.arange(gsplat.means.shape[0], device=gsplat.means.device)[mask]
        covs = gsplat.covs[mask, :2, :2] * confidence**2
        eigenvalues, eigenvectors = torch.linalg.eigh(covs)
        eigenvalues = torch.clamp(eigenvalues, min=1e-12)
        return cls(
            ids=ids,
            means=gsplat.means[mask, :2],
            covs=covs,
            rots=eigenvectors,
            scales=torch.sqrt(eigenvalues),
            z_min=float(z_min),
            z_max=float(z_max),
            confidence=float(confidence),
        )

    def __len__(self) -> int:
        return int(self.means.shape[0])
