"""Arithmetic inside the manuscript's own tables.

These checks recompute relations between numbers the article prints. A failure here
is a discrepancy inside the article, so the module is kept separate from the code
verdict and reported that way.
"""

from __future__ import annotations

import pytest

from rcrn.evaluation.manuscript import (
    ABLATION_DELTAS,
    ABLATION_METRICS,
    ASSAY_MAE,
    FULL_PARAMETERS_M,
    GATE_CONDITIONING_SHARE_PCT,
    GATE_FALSE_CONDITIONING_PCT,
    PRIMARY_RHO,
    REGION_RECORDS,
    SITE_RECORDS,
    TOTAL_RECORDS,
    TRANSPORT,
    WITHOUT_COUPLING_PARAMETERS_M,
    ablation_delta_agrees_with_metrics,
    all_consistency_results,
    arm_weighted_pool_matches_pooled_row,
    assay_family_ordering,
    caco2_margin_approximates_one_hundredth,
    cohort_partitions_sum,
    coupling_parameter_share_matches,
    fit_and_validation_recover_region_one,
    gate_calibration_improves,
    gate_shares_complement,
    interaction_ratio_reachable,
    parameter_attribution_bounds,
    quartile_agreement_falls,
    rank_floor_reachable,
    reference_level_below_increment,
    region_weighted_pool_matches_pooled_row,
    reported_increment_matches_columns,
    retained_margin_share_reachable,
    route_share_range_ordered,
    tier3_substitution_ordering,
    transport_largest_loss,
    transport_moderate_count,
)


def test_arm_and_site_partitions_recover_the_total() -> None:
    assert sum(SITE_RECORDS.values()) == TOTAL_RECORDS
    assert sum(REGION_RECORDS.values()) == TOTAL_RECORDS
    assert cohort_partitions_sum().holds is True


def test_fit_and_validation_tile_region_one() -> None:
    assert fit_and_validation_recover_region_one().holds is True


def test_arm_weighted_pools_that_hold() -> None:
    assert arm_weighted_pool_matches_pooled_row(
        "rho", {"prospective": 0.67, "retrospective": 0.70, "pooled": 0.69}
    ).holds


def test_region_weighted_increment_does_not_reproduce_the_pooled_row() -> None:
    """The pooled increment is not the size-weighted mean of the two regional ones."""

    result = region_weighted_pool_matches_pooled_row(
        "increment", {"region_I": 0.072, "region_II": 0.061, "pooled": 0.071}
    )
    assert result.holds is False
    assert "0.0690" in result.detail


def test_region_weighted_crossing_share_does_not_reproduce_the_pooled_row() -> None:
    result = region_weighted_pool_matches_pooled_row(
        "mcid_crossing", {"region_I": 74.8, "region_II": 69.4, "pooled": 73.9}
    )
    assert result.holds is False


def test_region_weighted_calibration_slope_does_not_reproduce_the_pooled_row() -> None:
    result = region_weighted_pool_matches_pooled_row(
        "calibration_slope", {"region_I": 1.04, "region_II": 1.09, "pooled": 1.04}
    )
    assert result.holds is False


def test_gate_shares_partition_the_cohort() -> None:
    assert GATE_CONDITIONING_SHARE_PCT + GATE_FALSE_CONDITIONING_PCT == 100.0
    assert gate_shares_complement().holds is True


def test_coupling_parameter_share_matches_the_printed_value() -> None:
    assert coupling_parameter_share_matches().holds is True
    assert (
        abs((FULL_PARAMETERS_M - WITHOUT_COUPLING_PARAMETERS_M) / FULL_PARAMETERS_M * 100 - 2.5)
        < 0.1
    )


def test_every_joint_row_ratio_is_reachable_from_the_printed_losses() -> None:
    for name in (
        "coupling_and_gate",
        "route_and_composition",
        "coupling_and_composition",
        "route_and_rank",
    ):
        assert interaction_ratio_reachable(name).holds is True


def test_every_removal_delta_agrees_with_its_metric() -> None:
    for name in ABLATION_DELTAS:
        assert ablation_delta_agrees_with_metrics(name).holds is True, name
    assert ABLATION_METRICS["without_interaction_variance_gate"] == PRIMARY_RHO


def test_retained_share_is_reachable() -> None:
    assert retained_margin_share_reachable().holds is True


def test_transport_table_yields_four_moderate_conditions() -> None:
    assert len(TRANSPORT) == 8
    assert transport_moderate_count().holds is True
    assert transport_largest_loss().holds is True


def test_assay_errors_order_as_the_article_describes() -> None:
    assert assay_family_ordering().holds is True
    assert ASSAY_MAE["descriptor_family"] < ASSAY_MAE["graph_family"]
    assert caco2_margin_approximates_one_hundredth().holds is True


def test_rank_floor_is_arithmetically_reachable() -> None:
    assert rank_floor_reachable().holds is True


def test_reported_increment_lies_inside_its_interval() -> None:
    assert reported_increment_matches_columns().holds is True


def test_mechanistic_readings_are_ordered() -> None:
    assert route_share_range_ordered().holds is True
    assert quartile_agreement_falls().holds is True


def test_gate_records_the_same_rank_column_with_a_lower_slope() -> None:
    assert gate_calibration_improves().holds is True


def test_reference_level_sits_between_the_two_arms() -> None:
    assert reference_level_below_increment().holds is True


def test_tier3_substitution_ordering_holds() -> None:
    assert tier3_substitution_ordering().holds is True


def test_parameter_attribution_leaves_a_positive_shared_budget() -> None:
    assert parameter_attribution_bounds().holds is True


def test_consistency_family_reports_four_discrepancies() -> None:
    results = all_consistency_results()
    failing = [result.name for result in results if not result.holds]
    assert len(results) == 39
    assert set(failing) == {
        "arm_weighted_pool_calibration_slope",
        "region_weighted_pool_increment",
        "region_weighted_pool_mcid_crossing",
        "region_weighted_pool_calibration_slope",
    }


@pytest.mark.parametrize("name", sorted(ABLATION_DELTAS))
def test_removal_details_are_recomputed_not_transcribed(name: str) -> None:
    detail = ablation_delta_agrees_with_metrics(name).detail
    assert "metric difference" in detail
