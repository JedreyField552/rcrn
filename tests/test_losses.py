"""Objective terms."""

from __future__ import annotations

import numpy as np
import torch

from rcrn.losses.consistency import branch_consistency_loss, ordering_discordance
from rcrn.losses.objective import (
    CompositeObjective,
    LossTargets,
    LossWeights,
    ObjectiveSettings,
    zero_objective_reference,
)
from rcrn.losses.property_prior import property_prior_loss
from rcrn.losses.rank import (
    agreement_weighted_rank_loss,
    build_pairs,
    concordance_rate,
)
from rcrn.losses.route_sparsity import coordinate_sparsity_loss, group_sparsity_loss
from rcrn.models.composition_head import CompositionOutput
from rcrn.models.coupling import CouplingOutput
from rcrn.models.graph_state import MolecularEmbedding
from rcrn.models.plaque_encoder import PlaqueEmbedding
from rcrn.models.rcrn import RCRNOutput
from rcrn.models.resistance_head import ResistanceOutput
from rcrn.models.route_head import RouteFields


def test_pairs_follow_the_observed_order() -> None:
    scores = torch.zeros(2, 4)
    order = torch.tensor([[2, 0, 3, 1], [0, 1, 2, 3]])
    weights = torch.tensor([0.5, 0.9])
    pairs = build_pairs(scores, order, weights, 3)
    assert pairs.size() == 6
    assert pairs.preferred.tolist()[:3] == [2, 2, 2]
    assert pairs.dispreferred.tolist()[:3] == [0, 3, 1]


def test_rank_loss_matches_explicit_enumeration() -> None:
    torch.manual_seed(0)
    scores = torch.randn(2, 4, dtype=torch.float64)
    order = torch.tensor([[3, 1, 0, 2], [1, 2, 3, 0]])
    weights = torch.tensor([0.4, 0.8], dtype=torch.float64)
    temperature = 0.5
    pairs = build_pairs(scores, order, weights, 6)
    computed = agreement_weighted_rank_loss(scores, pairs, temperature).item()
    manual = 0.0
    for plaque in range(2):
        rows = [
            position for position in range(pairs.size()) if pairs.plaque_index[position] == plaque
        ]
        inner = 0.0
        for position in rows:
            margin = (
                scores[plaque, pairs.dispreferred[position]]
                - scores[plaque, pairs.preferred[position]]
            ) / temperature
            inner += float(np.log1p(np.exp(float(margin))))
        manual += float(weights[plaque]) * inner / len(rows)
    assert abs(computed - manual / 2) < 1e-12


def test_empty_pair_set_is_not_a_failure() -> None:
    scores = torch.zeros(1, 2)
    order = torch.tensor([[0, 1]])
    pairs = build_pairs(scores, order, torch.ones(1), 0)
    assert pairs.size() == 0
    assert agreement_weighted_rank_loss(scores, pairs, 0.5).item() == 0.0


def test_concordance_rate_is_one_for_a_perfect_order() -> None:
    scores = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    order = torch.tensor([[0, 1, 2, 3]])
    pairs = build_pairs(scores, order, torch.ones(1), 6)
    assert concordance_rate(scores, pairs).item() == 1.0


def test_property_prior_ignores_missing_labels() -> None:
    prediction = torch.tensor([1.0, 2.0, 3.0])
    labelled = torch.tensor([1.0, float("nan"), 3.0])
    loss = property_prior_loss(prediction, labelled, beta=0.25)
    assert loss.item() == 0.0
    shifted = property_prior_loss(prediction, torch.tensor([0.0, 0.0, 0.0]), beta=0.25)
    assert shifted.item() > 0.0


def test_group_sparsity_matches_the_column_norm_sum() -> None:
    route = torch.tensor([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    assert group_sparsity_loss(route, 3).item() == 5.0
    assert coordinate_sparsity_loss(route).item() > 0.0


def test_consistency_penalises_discordant_branches() -> None:
    direct = torch.tensor([3.0, 2.0, 1.0])
    susceptibility = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    resistance = torch.tensor([[0.0, 0.0, 1.0]])
    agreeing = branch_consistency_loss(direct, susceptibility, resistance, margin=1.0)
    flipped = branch_consistency_loss(-direct, susceptibility, resistance, margin=1.0)
    assert flipped.item() > agreeing.item()


def test_ordering_discordance_reaches_one_half_when_reversed() -> None:
    direct = torch.tensor([3.0, 2.0, 1.0])
    susceptibility = torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 2.0], [0.0, 0.0, 1.0]])
    resistance = torch.tensor([[0.0, 0.0, -1.0]])
    assert ordering_discordance(direct, susceptibility, resistance).item() == 1.0


def _output() -> RCRNOutput:
    torch.manual_seed(0)
    molecules, plaques = 4, 3
    direct = torch.randn(molecules)
    direction = torch.nn.functional.normalize(torch.randn(molecules, 3), dim=-1)
    resistance = torch.rand(plaques, 3) + 0.5
    pair = (direct.unsqueeze(0) - resistance @ direction.transpose(0, 1)).requires_grad_(True)
    return RCRNOutput(
        molecules=MolecularEmbedding(
            node_states=torch.randn(6, 8),
            graph_state=torch.randn(molecules, 8),
            graph_index=torch.tensor([0, 0, 1, 2, 3, 3]),
            graph_count=molecules,
        ),
        plaque=PlaqueEmbedding(
            patch_tokens=torch.randn(plaques, 5, 8),
            register_tokens=torch.randn(plaques, 2, 8),
            pooled=torch.randn(plaques, 8),
        ),
        resistance=ResistanceOutput(
            resistance=resistance, perturbation=torch.zeros_like(resistance)
        ),
        composition=CompositionOutput(index=torch.rand(plaques), is_probability=False),
        route_fields=RouteFields(
            susceptibility=direction,
            magnitude=torch.linalg.norm(torch.randn(molecules, 3), dim=-1),
            direct_effect=direct,
        ),
        coupling=CouplingOutput(
            susceptibility=direction,
            magnitude=torch.linalg.norm(torch.randn(molecules, 3), dim=-1),
            direct_effect=direct,
            pair_score=pair,
            route_contribution=torch.randn(plaques, molecules, 3),
        ),
    )


def test_composite_objective_sums_the_weighted_terms() -> None:
    output = _output()
    weights = LossWeights(rank=1.0, property=0.5, route_sparsity=0.01, consistency=0.2)
    objective = CompositeObjective(weights, ObjectiveSettings(groups=3))
    order = torch.tensor([[3, 1, 0, 2], [0, 1, 2, 3], [2, 3, 1, 0]])
    targets = LossTargets(
        observed_order=order,
        agreement_weights=torch.tensor([0.5, 0.6, 0.7]),
        auxiliary_targets=torch.tensor([0.1, 0.2, 0.3, 0.4]),
        rank_pairs=build_pairs(output.coupling.pair_score, order, torch.tensor([0.5, 0.6, 0.7]), 6),
    )
    prediction = torch.tensor([0.1, 0.2, 0.3, 0.4])
    breakdown = objective(output, targets, prediction)
    manual = (
        weights.rank * breakdown.rank
        + weights.property * breakdown.property
        + weights.route_sparsity * breakdown.route_sparsity
        + weights.consistency * breakdown.consistency
    )
    assert torch.allclose(breakdown.total, manual, atol=1e-9)
    assert breakdown.as_dict()["total"] == float(breakdown.total.detach())


def test_negative_weights_are_rejected() -> None:
    try:
        CompositeObjective(LossWeights(rank=-1.0), ObjectiveSettings())
    except ValueError:
        return
    raise AssertionError("a negative loss weight must be rejected")


def test_zero_reference_stays_differentiable() -> None:
    output = _output()
    reference = zero_objective_reference(output)
    assert reference.item() == 0.0
    assert reference.grad_fn is not None


def test_objective_needs_pairs() -> None:
    output = _output()
    objective = CompositeObjective(LossWeights(), ObjectiveSettings())
    targets = LossTargets(
        observed_order=torch.zeros(3, 4, dtype=torch.long),
        agreement_weights=torch.ones(3),
        auxiliary_targets=torch.zeros(4),
    )
    try:
        objective(output, targets, torch.zeros(4))
    except ValueError:
        return
    raise AssertionError("the objective must refuse an unprepared pair set")
