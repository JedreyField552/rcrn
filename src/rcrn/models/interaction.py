"""Interaction covariance, perturbation noise and the gate statistic.

Ref: Methods, "Reaction-norm coupling and the decline-to-condition gate", Eq. (2);
Algorithm 2 step 3 (Sigma_s, sigma_delta^2(p) and nu_p are estimated by
cross-validation inside the training partition).

Sigma_s is the covariance of the susceptibility directions over the candidate
list, shrunk towards a scaled identity so that its inverse stays defined when the
route count approaches the candidate count. The per-record noise term grows as the
plaque gets smaller, which is the reading the manuscript gives for the two largest
transport losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

EPSILON = 1e-8


@dataclass(slots=True)
class InteractionSurfaces:
    """Everything Eq. (2) needs, plus the signal it compares against."""

    covariance: Tensor
    noise_variance: Tensor
    prior: Tensor
    signal: Tensor
    statistic: Tensor


def shrinkage_covariance(susceptibility: Tensor, strength: float = 0.1) -> Tensor:
    """Ledoit-Wolf style shrinkage of the route covariance towards a scaled identity."""

    if susceptibility.dim() != 2:
        raise ValueError("expected a (candidate, route) tensor")
    centred = susceptibility - susceptibility.mean(dim=0, keepdim=True)
    samples = centred.shape[0]
    if samples < 2:
        return torch.eye(susceptibility.shape[1], dtype=susceptibility.dtype) * strength
    empirical = centred.transpose(0, 1) @ centred / (samples - 1)
    target = torch.eye(empirical.shape[0], dtype=empirical.dtype, device=empirical.device) * (
        torch.trace(empirical) / empirical.shape[0]
    )
    return (1.0 - strength) * empirical + strength * target


def perturbation_noise(
    perturbation: Tensor,
    plaque_volume_mm3: Tensor,
    reference_volume_mm3: float,
    floor: float = 1e-4,
) -> Tensor:
    """Residual variance of the perturbation vector, scaled by the sampled area."""

    if perturbation.dim() != 2:
        raise ValueError("expected a (record, route) tensor")
    route_variance = torch.var(perturbation, dim=-1, unbiased=False)
    pooled = torch.clamp(route_variance.mean(), min=floor)
    scale = torch.clamp(
        reference_volume_mm3 / torch.clamp(plaque_volume_mm3, min=1e-3), min=0.05, max=20.0
    )
    return pooled * scale


def interaction_statistic(signal: Tensor, noise_variance: Tensor, prior: Tensor) -> Tensor:
    """The normalised interaction-variance ratio of Eq. (2)."""

    denominator = signal + noise_variance + prior
    return signal / (denominator + EPSILON)


class InteractionEstimator:
    """Estimate the interaction surfaces and evaluate the gate statistic."""

    def __init__(self, prior_strength: float = 0.1, shrinkage: float = 0.1) -> None:
        self.prior_strength = prior_strength
        self.shrinkage = shrinkage
        self._covariance: Tensor | None = None

    @property
    def covariance(self) -> Tensor:
        if self._covariance is None:
            raise RuntimeError("interaction surfaces are not fitted")
        return self._covariance

    def fit(self, susceptibility: Tensor) -> Tensor:
        self._covariance = shrinkage_covariance(susceptibility, self.shrinkage)
        return self._covariance

    def evaluate(
        self,
        perturbation: Tensor,
        plaque_volume_mm3: Tensor,
        reference_volume_mm3: float,
    ) -> InteractionSurfaces:
        covariance = self.covariance
        signal = torch.einsum("pr,rs,ps->p", perturbation, covariance, perturbation)
        noise = perturbation_noise(perturbation, plaque_volume_mm3, reference_volume_mm3)
        prior = torch.full_like(
            noise, self.prior_strength * float(torch.trace(covariance)) / covariance.shape[0]
        )
        statistic = interaction_statistic(signal, noise, prior)
        return InteractionSurfaces(
            covariance=covariance,
            noise_variance=noise,
            prior=prior,
            signal=signal,
            statistic=statistic,
        )
