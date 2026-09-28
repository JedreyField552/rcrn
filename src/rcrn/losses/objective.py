"""Composite training objective.

Ref: Eq. (4):

    L = lambda1 L_rank + lambda2 L_prop + lambda3 L_reg + lambda4 L_consist

The weights are fixed on the training folds and are never re-adjusted on the test
fold; the training loop enforces that by reading them once from configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rcrn.losses.consistency import branch_consistency_loss
from rcrn.losses.property_prior import property_prior_loss
from rcrn.losses.rank import RankPairSet, agreement_weighted_rank_loss, concordance_rate
from rcrn.losses.route_sparsity import group_sparsity_loss
from rcrn.models.rcrn import RCRNOutput


@dataclass(slots=True)
class LossWeights:
    """The four non-negative weights of Eq. (4)."""

    rank: float = 1.0
    property: float = 0.5
    route_sparsity: float = 0.01
    consistency: float = 0.2

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.rank, self.property, self.route_sparsity, self.consistency)


@dataclass(slots=True)
class LossTargets:
    """Everything the four terms need beyond the model output."""

    observed_order: Tensor
    agreement_weights: Tensor
    auxiliary_targets: Tensor
    rank_pairs: RankPairSet | None = None


@dataclass(slots=True)
class LossBreakdown:
    """Per-term values of the composite objective."""

    total: Tensor
    rank: Tensor
    property: Tensor
    route_sparsity: Tensor
    consistency: Tensor
    concordance: Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "rank": float(self.rank.detach()),
            "property": float(self.property.detach()),
            "route_sparsity": float(self.route_sparsity.detach()),
            "consistency": float(self.consistency.detach()),
            "concordance": float(self.concordance.detach()),
        }


@dataclass(slots=True)
class ObjectiveSettings:
    """Surrogate, temperature, pair budget and the group count of the sparsity term."""

    surrogate: str = "logistic"
    temperature: float = 0.5
    pair_budget: int = 256
    groups: int = 3
    property_beta: float = 0.25
    consistency_margin: float = 1.0


class CompositeObjective:
    """Assemble the four terms of Eq. (4)."""

    def __init__(self, weights: LossWeights, settings: ObjectiveSettings) -> None:
        if min(weights.as_tuple()) < 0.0:
            raise ValueError("loss weights must be non-negative")
        self.weights = weights
        self.settings = settings

    def __call__(
        self,
        output: RCRNOutput,
        targets: LossTargets,
        property_prediction: Tensor,
    ) -> LossBreakdown:
        if targets.rank_pairs is None:
            raise ValueError("rank pairs must be prepared before the objective is evaluated")
        rank_term = agreement_weighted_rank_loss(
            output.coupling.pair_score,
            targets.rank_pairs,
            self.settings.temperature,
            self.settings.surrogate,
        )
        property_term = property_prior_loss(
            property_prediction, targets.auxiliary_targets, self.settings.property_beta
        )
        sparsity_term = group_sparsity_loss(
            output.route_fields.susceptibility, self.settings.groups
        )
        consistency_term = branch_consistency_loss(
            output.route_fields.direct_effect,
            output.route_fields.susceptibility,
            output.resistance.resistance,
            margin=self.settings.consistency_margin,
        )
        total = (
            self.weights.rank * rank_term
            + self.weights.property * property_term
            + self.weights.route_sparsity * sparsity_term
            + self.weights.consistency * consistency_term
        )
        return LossBreakdown(
            total=total,
            rank=rank_term,
            property=property_term,
            route_sparsity=sparsity_term,
            consistency=consistency_term,
            concordance=concordance_rate(output.coupling.pair_score, targets.rank_pairs),
        )


def zero_objective_reference(output: RCRNOutput) -> Tensor:
    """A differentiable zero anchored to the model output, for empty batches."""

    return output.coupling.pair_score.sum() * 0.0 + torch.zeros(
        (), dtype=output.coupling.pair_score.dtype
    )
