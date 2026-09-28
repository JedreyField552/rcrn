"""Scale-free branch-consistency term.

Ref: Eq. (4) (Lconsist penalises inconsistent within-distinction orderings between
the branches while allowing the two branches a scale discrepancy).

The penalty compares the sign of two within-plaque orderings: the one implied by
the plaque-blind direct effect and the one implied by the route contraction. It is
normalised by both magnitudes, so a branch that is merely larger is not penalised.
"""

from __future__ import annotations

import torch
from torch import Tensor

EPSILON = 1e-8


def branch_consistency_loss(
    direct_effect: Tensor,
    susceptibility: Tensor,
    resistance: Tensor,
    margin: float = 1.0,
    pair_budget: int = 64,
) -> Tensor:
    """Squared hinge on the disagreement between the two within-plaque orderings."""

    if direct_effect.dim() != 1 or susceptibility.dim() != 2 or resistance.dim() != 2:
        raise ValueError("unexpected tensor ranks")
    candidate_count = direct_effect.shape[0]
    if candidate_count < 2:
        return direct_effect.sum() * 0.0
    index = torch.arange(candidate_count, device=direct_effect.device)
    left = index.unsqueeze(0)
    right = index.unsqueeze(1)
    keep = left < right
    left_index = left.expand(candidate_count, candidate_count)[keep][:pair_budget]
    right_index = right.expand(candidate_count, candidate_count)[keep][:pair_budget]
    direct_delta = direct_effect[left_index] - direct_effect[right_index]
    route_delta = (susceptibility[left_index] - susceptibility[right_index]) @ resistance.transpose(
        0, 1
    )
    numerator = direct_delta.unsqueeze(1) * route_delta
    denominator = direct_delta.abs().unsqueeze(1) * torch.abs(route_delta) + EPSILON
    agreement = numerator / denominator
    penalty = torch.clamp(margin - agreement, min=0.0) ** 2
    return penalty.mean()


def ordering_discordance(
    direct_effect: Tensor, susceptibility: Tensor, resistance: Tensor
) -> Tensor:
    """Share of within-plaque pairs the two branches order differently."""

    candidate_count = direct_effect.shape[0]
    if candidate_count < 2:
        return direct_effect.sum() * 0.0
    direct_delta = direct_effect.unsqueeze(0) - direct_effect.unsqueeze(1)
    route_delta = (
        susceptibility.unsqueeze(0) - susceptibility.unsqueeze(1)
    ) @ resistance.transpose(0, 1)
    agreement = torch.sign(direct_delta.unsqueeze(-1) * route_delta)
    per_pair = (agreement < 0).to(direct_effect.dtype).mean(dim=-1)
    upper = torch.triu(torch.ones_like(per_pair, dtype=torch.bool), diagonal=1)
    return per_pair[upper].mean()
