"""Reaction-norm coupling and its alternatives."""

from __future__ import annotations

import pytest
import torch

from rcrn.models.coupling import COUPLING_KINDS, ReactionNormCoupling, route_share
from rcrn.models.route_head import RouteFields


def _fields(molecules: int = 4, routes: int = 3) -> RouteFields:
    torch.manual_seed(0)
    direction = torch.nn.functional.normalize(torch.randn(molecules, routes), dim=-1)
    return RouteFields(
        susceptibility=direction,
        magnitude=torch.linalg.norm(torch.randn(molecules, routes), dim=-1),
        direct_effect=torch.randn(molecules),
    )


def test_reaction_norm_equals_the_negative_contraction() -> None:
    fields = _fields()
    resistance = torch.rand(3, 3) + 0.5
    coupling = ReactionNormCoupling(3, kind="reaction_norm")
    with torch.no_grad():
        for parameter in coupling.parameters():
            parameter.zero_()
    output = coupling(fields, resistance)
    expected = fields.direct_effect.unsqueeze(0) - resistance @ fields.susceptibility.transpose(
        0, 1
    )
    assert torch.allclose(output.pair_score, expected, atol=1e-6)


def test_pair_score_orientation_is_plaque_by_candidate() -> None:
    output = ReactionNormCoupling(3).forward(_fields(5), torch.rand(7, 3))
    assert output.pair_score.shape == (7, 5)


def test_route_contribution_sums_to_the_conditioned_part() -> None:
    fields = _fields()
    resistance = torch.rand(4, 3) + 0.5
    output = ReactionNormCoupling(3, kind="reaction_norm")(fields, resistance)
    conditioned = output.pair_score - output.direct_effect.unsqueeze(0)
    assert torch.allclose(output.route_contribution.sum(-1), conditioned, atol=1e-6)


def test_route_share_is_a_distribution() -> None:
    share = route_share(torch.rand(2, 5, 3))
    assert torch.allclose(share.sum(-1), torch.ones(2, 5), atol=1e-6)


@pytest.mark.parametrize("kind", COUPLING_KINDS)
def test_every_coupling_kind_runs(kind: str) -> None:
    coupling = ReactionNormCoupling(3, kind=kind)
    tokens = torch.randn(4, 6, 3)
    output = coupling(_fields(), torch.rand(4, 3), tokens)
    assert output.pair_score.shape == (4, 4)
    assert torch.isfinite(output.pair_score).all()


def test_plain_concatenation_differs_from_the_reaction_norm() -> None:
    fields = _fields()
    resistance = torch.rand(3, 3) + 0.5
    torch.manual_seed(1)
    norm = ReactionNormCoupling(3, kind="reaction_norm")(fields, resistance)
    torch.manual_seed(1)
    concat = ReactionNormCoupling(3, kind="plain_concat")(fields, resistance)
    assert not torch.allclose(norm.pair_score, concat.pair_score, atol=1e-6)


def test_cross_attention_requires_tokens() -> None:
    with pytest.raises(ValueError):
        ReactionNormCoupling(3, kind="cross_attention")(_fields(), torch.rand(2, 3))


def test_route_count_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        ReactionNormCoupling(3)(_fields(routes=3), torch.rand(2, 4))


def test_unknown_coupling_is_rejected() -> None:
    with pytest.raises(ValueError):
        ReactionNormCoupling(3, kind="not_a_coupling")
