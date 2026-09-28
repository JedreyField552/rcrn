"""Objective terms."""

from __future__ import annotations

from rcrn.losses.consistency import branch_consistency_loss, ordering_discordance
from rcrn.losses.objective import (
    CompositeObjective,
    LossBreakdown,
    LossTargets,
    LossWeights,
    ObjectiveSettings,
    zero_objective_reference,
)
from rcrn.losses.property_prior import property_prior_loss
from rcrn.losses.rank import (
    RankPairSet,
    agreement_weighted_rank_loss,
    build_pairs,
    concordance_rate,
)
from rcrn.losses.route_sparsity import coordinate_sparsity_loss, group_sparsity_loss

__all__ = [
    "CompositeObjective",
    "LossBreakdown",
    "LossTargets",
    "LossWeights",
    "ObjectiveSettings",
    "RankPairSet",
    "agreement_weighted_rank_loss",
    "branch_consistency_loss",
    "build_pairs",
    "concordance_rate",
    "coordinate_sparsity_loss",
    "group_sparsity_loss",
    "ordering_discordance",
    "property_prior_loss",
    "zero_objective_reference",
]
