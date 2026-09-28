"""Component-necessity arithmetic and transport degradation."""

from __future__ import annotations

import numpy as np
import pytest

from rcrn.evaluation.ablation import (
    AblationRow,
    control_collapse,
    interaction_ratio,
    interaction_ratio_band,
    joint_rows,
    largest_single_loss,
    null_rows,
    parameter_share,
    ratio_within_band,
    removal_rows,
    retained_share,
    retained_share_band,
    synergistic_joint,
)
from rcrn.metrics.transport import (
    degradation_class,
    losses_exceeding,
    paired_marginal_divergence,
    transport_cost_profile,
    transport_row,
    transport_summary,
)


def _row(
    name: str, tier: str, delta: float, metric: float = 0.6, parameters: float = 150.0
) -> AblationRow:
    return AblationRow(
        tier=tier,
        variant=name,
        metric=metric,
        delta=delta,
        parameters_m=parameters,
        linked_equation="eq",
    )


def test_interaction_ratio_is_joint_over_the_sum() -> None:
    assert interaction_ratio(0.14, (0.10, 0.04)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        interaction_ratio(0.1, (0.0, 0.0))


def test_rounding_band_brackets_the_printed_ratio() -> None:
    low, high = interaction_ratio_band(0.14, (0.10, 0.0))
    assert low <= 1.39 <= high
    assert ratio_within_band(1.39, 0.14, (0.10, 0.0)) is True
    assert ratio_within_band(5.0, 0.14, (0.10, 0.0)) is False


def test_band_widens_with_the_rounding_unit() -> None:
    narrow = interaction_ratio_band(0.14, (0.10, 0.04), rounding=0.001)
    wide = interaction_ratio_band(0.14, (0.10, 0.04), rounding=0.01)
    assert wide[1] - wide[0] > narrow[1] - narrow[0]


def test_parameter_share_reports_the_removed_fraction() -> None:
    assert parameter_share(150.7, 146.9) == pytest.approx(0.0252, abs=1e-4)
    with pytest.raises(ValueError):
        parameter_share(0.0, 1.0)


def test_retained_share_matches_the_definitions() -> None:
    assert retained_share(0.67, 0.69) == pytest.approx(0.9710, abs=1e-4)
    low, high = retained_share_band(0.67, 0.69)
    assert low <= 0.973 <= high


def test_tier_helpers_partition_the_rows() -> None:
    rows = (
        _row("removal", "tier_1_removal", -0.10),
        _row("joint", "tier_2_joint", -0.14),
        _row("substitution", "tier_3_substitution", -0.12),
        _row("control", "tier_0_control", -0.68, metric=0.01),
    )
    assert [row.variant for row in removal_rows(rows)] == ["removal"]
    assert [row.variant for row in joint_rows(rows)] == ["joint"]
    assert len(rows) - 3 == 1
    assert largest_single_loss(rows).variant == "removal"


def test_null_rows_respect_the_tolerance() -> None:
    rows = (_row("gate", "tier_1_removal", 0.0), _row("coupling", "tier_1_removal", -0.10))
    assert [row.variant for row in null_rows(rows)] == ["gate"]


def test_synergy_requires_exceeding_the_sum() -> None:
    joint = _row("joint", "tier_2_joint", -0.14)
    first = _row("a", "tier_1_removal", -0.10)
    second = _row("b", "tier_1_removal", 0.00)
    assert synergistic_joint(joint, first, second) is True
    assert synergistic_joint(_row("joint", "tier_2_joint", -0.09), first, second) is False


def test_control_collapse_needs_every_control_below_the_ceiling() -> None:
    rows = (
        _row("c1", "tier_0_control", -0.68, metric=0.01),
        _row("c2", "tier_0_control", -0.41, metric=0.29),
    )
    assert control_collapse(rows, ceiling=0.30) is True
    assert control_collapse((rows[1],), ceiling=0.10) is False


def test_degradation_class_threshold() -> None:
    assert degradation_class(-2.4) == "graceful"
    assert degradation_class(-5.1) == "moderate"
    assert degradation_class(-5.0) == "graceful"


def test_transport_row_prefers_the_paired_difference() -> None:
    row = transport_row("condition", 100, 0.69, 0.63, paired_deltas=np.array([-0.04, -0.06]))
    assert row.paired_delta_pp == pytest.approx(-5.0)
    assert row.marginal_delta_pp == pytest.approx(-6.0)
    assert row.degradation == "graceful"


def test_transport_row_falls_back_to_the_marginal_difference() -> None:
    row = transport_row("condition", 10, 0.69, 0.63)
    assert row.paired_delta_pp == row.marginal_delta_pp


def test_transport_summary_counts_and_extremes() -> None:
    rows = (
        transport_row("a", 10, 0.69, 0.67),
        transport_row("b", 10, 0.69, 0.59),
    )
    summary = transport_summary(rows)
    assert summary["conditions"] == 2.0
    assert summary["moderate"] == 1.0
    assert summary["largest_loss_pp"] == pytest.approx(10.0)
    assert losses_exceeding(rows) == ("b",)


def test_divergence_and_profile_are_aligned_with_the_rows() -> None:
    rows = (
        transport_row("a", 10, 0.69, 0.67, paired_deltas=np.array([-0.05, -0.05])),
        transport_row("b", 10, 0.69, 0.61),
    )
    divergence = paired_marginal_divergence(rows)
    assert divergence.shape == (2,)
    assert divergence[0] == pytest.approx(-3.0)
    profile = transport_cost_profile(rows)
    assert set(profile) == {"a", "b"}
