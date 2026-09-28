"""Reaction-norm coupling and its named alternatives.

Ref: Methods, "Reaction-norm coupling and the decline-to-condition gate"; Eq. (1);
Algorithm 1 step 1; Table 2 Tier 1 and Tier 3 substitution rows.

The default fusion contracts the susceptibility direction against the plaque
resistance, giving the reaction norm of Eq. (1), and reports the route-wise split
of that contraction. The alternatives are the ones the manuscript names: a plain
concatenation of the two branches, a bilinear concatenation, and cross-attention
over the plaque tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from rcrn.models.route_head import RouteFields

COUPLING_KINDS: tuple[str, ...] = (
    "reaction_norm",
    "plain_concat",
    "bilinear_concat",
    "cross_attention",
)


@dataclass(slots=True)
class CouplingOutput:
    """Susceptibility, the pair score and the route-wise split of that score."""

    susceptibility: Tensor
    magnitude: Tensor
    direct_effect: Tensor
    pair_score: Tensor
    route_contribution: Tensor


class ReactionNormCoupling(nn.Module):
    """Fuse the molecular route vector with the plaque resistance vector."""

    def __init__(
        self,
        route_count: int,
        kind: str = "reaction_norm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kind not in COUPLING_KINDS:
            raise ValueError(f"unknown coupling {kind}")
        self.kind = kind
        self.route_count = route_count
        self.molecular_linear = nn.Linear(route_count, route_count, bias=False)
        self.resistance_linear = nn.Linear(route_count, route_count, bias=False)
        self.bilinear = nn.Linear(route_count, route_count, bias=False)
        self.cross_query = nn.Linear(route_count, route_count)
        self.cross_output = nn.Linear(route_count, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        fields: RouteFields,
        resistance: Tensor,
        plaque_tokens: Tensor | None = None,
    ) -> CouplingOutput:
        direction = self.dropout(fields.susceptibility)
        direct = fields.direct_effect
        if direction.shape[-1] != resistance.shape[-1]:
            raise ValueError("molecular and plaque route counts differ")
        if self.kind == "reaction_norm":
            contraction = -(resistance @ direction.transpose(0, 1))
            pair_score = direct.unsqueeze(0) + contraction
            route_contribution = -direction.unsqueeze(0) * resistance.unsqueeze(1)
        elif self.kind == "plain_concat":
            molecular = self.molecular_linear(direction)
            plaque = self.resistance_linear(resistance)
            route_contribution = molecular.unsqueeze(0) + plaque.unsqueeze(1)
            pair_score = direct.unsqueeze(0) + route_contribution.sum(-1)
        elif self.kind == "bilinear_concat":
            molecular = self.molecular_linear(direction)
            plaque = self.resistance_linear(resistance)
            interaction = torch.einsum("mr,pr->pmr", self.bilinear(direction), resistance)
            route_contribution = molecular.unsqueeze(0) + plaque.unsqueeze(1) + interaction
            pair_score = direct.unsqueeze(0) + route_contribution.sum(-1)
        else:
            if plaque_tokens is None:
                raise ValueError("cross-attention coupling needs plaque tokens")
            query = torch.tanh(self.cross_query(direction)).unsqueeze(0)
            scores = query @ plaque_tokens.transpose(1, 2)
            attention = torch.softmax(scores, dim=-1)
            attended = attention @ plaque_tokens
            pair_score = direct.unsqueeze(0) + self.cross_output(attended).squeeze(-1)
            route_contribution = attended * direction.unsqueeze(0)
        return CouplingOutput(
            susceptibility=direction,
            magnitude=fields.magnitude,
            direct_effect=direct,
            pair_score=pair_score,
            route_contribution=route_contribution,
        )


def route_share(route_contribution: Tensor) -> Tensor:
    """Absolute share of the score change carried by each transport route."""

    magnitude = torch.abs(route_contribution)
    total = magnitude.sum(dim=-1, keepdim=True)
    return magnitude / (total + 1e-9)
