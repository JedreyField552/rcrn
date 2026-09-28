"""Three-route transport-susceptibility head.

Ref: Methods, "The regime-route molecular encoder" (si = B hi with hi in R^d_cap
and B in R^{3 x d_cap}; the vector is normalised so route attribution does not get
confused with its magnitude; the same branch yields the plaque-blind direct effect
ai).

B is the three-way head of the paper: a low-rank or full map from the coupling
state onto the transport routes. The coupling module owns only the contraction
against the plaque resistance vector, so a Tier 1 removal of the route head is one
constructor argument and a Tier 3 rank cap is another.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class RouteFields:
    """Susceptibility direction, its magnitude and the plaque-blind direct effect."""

    susceptibility: Tensor
    magnitude: Tensor
    direct_effect: Tensor

    def route_count(self) -> int:
        return int(self.susceptibility.shape[-1])


class RouteOperator(nn.Module):
    """B in R^{route x d_cap}, either dense or factorised at a declared rank."""

    def __init__(self, route_count: int, d_cap: int, rank: int) -> None:
        super().__init__()
        self.route_count = route_count
        self.d_cap = d_cap
        self.rank = rank
        if rank <= 0 or rank >= min(route_count, d_cap):
            self.factor_left = nn.Parameter(torch.empty(route_count, d_cap))
            self.factor_right: nn.Parameter | None = None
        else:
            self.factor_left = nn.Parameter(torch.empty(route_count, rank))
            self.factor_right = nn.Parameter(torch.empty(d_cap, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / max(1, self.factor_left.shape[-1]) ** 0.5
        with torch.no_grad():
            self.factor_left.uniform_(-bound, bound)
            if self.factor_right is not None:
                self.factor_right.uniform_(-bound, bound)

    def matrix(self) -> Tensor:
        if self.factor_right is None:
            return self.factor_left
        product: Tensor = self.factor_left @ self.factor_right.transpose(0, 1)
        return product

    def effective_rank(self) -> int:
        return self.rank if self.factor_right is not None else min(self.route_count, self.d_cap)

    def forward(self, graph_state: Tensor) -> Tensor:
        projection: Tensor = graph_state @ self.matrix().transpose(0, 1)
        return projection


class RouteHead(nn.Module):
    """Project the coupling state onto the transport routes and the direct effect."""

    def __init__(
        self,
        d_cap: int,
        route_count: int,
        rank: int = 3,
        normalise_routes: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if route_count < 1:
            raise ValueError("at least one route is required")
        self.route_count = route_count
        self.normalise_routes = normalise_routes
        self.norm = nn.LayerNorm(d_cap)
        self.operator = RouteOperator(route_count, d_cap, rank)
        self.direct = nn.Linear(d_cap, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_state: Tensor) -> RouteFields:
        if graph_state.dim() != 2:
            raise ValueError("expected a (molecule, d_cap) tensor")
        normalized: Tensor = self.norm(graph_state)
        dropped: Tensor = self.dropout(normalized)
        projected: Tensor = self.operator(dropped)
        magnitude = torch.linalg.norm(projected, dim=-1)
        if self.normalise_routes:
            direction = projected / (magnitude.unsqueeze(-1) + 1e-9)
        else:
            direction = projected
        direct: Tensor = self.direct(dropped).squeeze(-1)
        return RouteFields(susceptibility=direction, magnitude=magnitude, direct_effect=direct)
