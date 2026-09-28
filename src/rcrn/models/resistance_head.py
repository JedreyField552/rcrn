"""Plaque resistance head and the perturbation vector.

Ref: Methods, "The plaque composition encoder" (r(xp) in R^3_+, delta_p =
r_hat(xp) - r0 with r0 the route-wise average resistance at the training level).

The positive link keeps every route coordinate a resistance. The route mean r0 is
a buffer rather than a parameter, so it can be refreshed from the training pool
without disturbing the fitted heads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class ResistanceOutput:
    """Positive route resistances and their displacement from the training mean."""

    resistance: Tensor
    perturbation: Tensor


class ResistanceHead(nn.Module):
    """Map plaque tokens onto the route resistances and centre them at r0."""

    def __init__(self, hidden_dim: int, route_count: int, dropout: float = 0.0) -> None:
        super().__init__()
        if route_count < 1:
            raise ValueError("at least one route is required")
        self.route_count = route_count
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, route_count)
        self.route_mean: Tensor
        self.register_buffer("route_mean", torch.ones(route_count))

    def forward(self, pooled_tokens: Tensor) -> ResistanceOutput:
        normalized: Tensor = self.norm(pooled_tokens)
        dropped: Tensor = self.dropout(normalized)
        raw: Tensor = self.head(dropped)
        resistance: Tensor = torch.nn.functional.softplus(raw) + 1e-3
        perturbation: Tensor = resistance - self.route_mean.unsqueeze(0)
        return ResistanceOutput(resistance=resistance, perturbation=perturbation)

    @torch.no_grad()
    def refresh_route_mean(self, resistance: Tensor, momentum: float = 1.0) -> None:
        """Update the route-wise average resistance from a training batch."""

        if resistance.dim() != 2:
            raise ValueError("expected a (record, route) tensor")
        batch_mean = resistance.mean(dim=0)
        blended = (1.0 - momentum) * self.route_mean + momentum * batch_mean
        self.route_mean.copy_(blended)
