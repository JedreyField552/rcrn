"""Gated scoring and per-plaque orderings."""

from __future__ import annotations

import numpy as np
import torch

from rcrn.models.coupling import CouplingOutput
from rcrn.models.gate import GateDecision
from rcrn.models.scorer import (
    GatedScorer,
    PlaqueOrdering,
    deferral_share,
    degenerate_orderings,
)


def _coupling(plaque_count: int = 3, molecules: int = 5) -> CouplingOutput:
    torch.manual_seed(0)
    direct = torch.tensor([0.5, 0.1, 0.9, 0.2, 0.7])
    pair = direct.unsqueeze(0) + torch.randn(plaque_count, molecules) * 0.2
    return CouplingOutput(
        susceptibility=torch.nn.functional.normalize(torch.randn(molecules, 3), dim=-1),
        magnitude=torch.linalg.norm(torch.randn(molecules, 3), dim=-1),
        direct_effect=direct,
        pair_score=pair,
        route_contribution=torch.randn(plaque_count, molecules, 3),
    )


def test_orderings_follow_the_score() -> None:
    coupling = _coupling()
    scorer = GatedScorer(("A", "B", "C", "D", "E"))
    orderings = scorer.order(("R0", "R1", "R2"), coupling)
    assert len(orderings) == 3
    for position, ordering in enumerate(orderings):
        assert ordering.conditioned is True
        assert (
            ordering.ordering[0]
            == ("A", "B", "C", "D", "E")[int(np.argmax(coupling.pair_score[position].numpy()))]
        )
        assert ordering.route_share is not None
        assert np.allclose(ordering.route_share.sum(axis=-1), 1.0)


def test_deferred_records_fall_back_to_the_blind_order() -> None:
    coupling = _coupling()
    scorer = GatedScorer(("A", "B", "C", "D", "E"))
    decision = GateDecision(
        conditioned=np.array([True, False, True]),
        threshold=0.5,
        conditioning_share=2 / 3,
        deferral_share=1 / 3,
        false_conditioning_rate=0.0,
        signal_reference_used=False,
    )
    orderings = scorer.order(("R0", "R1", "R2"), coupling, None, decision)
    assert orderings[1].conditioned is False
    assert orderings[1].route_share is None
    blind = np.argsort(-coupling.direct_effect.numpy(), kind="stable")
    assert orderings[1].order == tuple(int(index) for index in blind)


def test_blind_only_scorer_never_conditions() -> None:
    coupling = _coupling()
    scorer = GatedScorer(("A", "B", "C", "D", "E"), blind_only=True)
    orderings = scorer.order(("R0",), coupling)
    assert orderings[0].conditioned is False


def test_ties_are_detected() -> None:
    ordering = PlaqueOrdering(
        record_id="R",
        scores=np.zeros(4),
        order=(0, 1, 2, 3),
        ordering=("A", "B", "C", "D"),
        conditioned=True,
        statistic=0.0,
    )
    assert ordering.has_ties() is True
    assert degenerate_orderings((ordering,)) == 1


def test_deferral_share_counts_fallbacks() -> None:
    coupling = _coupling()
    orderings = GatedScorer(("A", "B", "C", "D", "E")).order(("R0", "R1"), coupling)
    assert deferral_share(orderings) == 0.0


def test_top_k_depth_is_respected() -> None:
    coupling = _coupling()
    ordering = GatedScorer(("A", "B", "C", "D", "E")).order(("R0",), coupling)[0]
    assert len(ordering.top_k(2)) == 2
    assert ordering.top_k(2) == ordering.order[:2]
