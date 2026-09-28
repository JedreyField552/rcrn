"""Consistency checks on the numbers the manuscript itself prints.

Ref: Tables 1-4, Figures 1-3, Abstract, Sections 2.1-2.8 and 3.

Nothing here is a release result. The values below are the article's own reported
quantities, transcribed so the relations the article asserts between them can be
recomputed. A failing check in this family is a discrepancy inside the article's
tables, not a defect in the release, and the verification report keeps the two
families apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcrn.evaluation.ablation import interaction_ratio_band, parameter_share, retained_share_band

TOTAL_RECORDS = 4318
PROSPECTIVE_RECORDS = 1142
RETROSPECTIVE_RECORDS = 3176
SITE_RECORDS = {"A": 1743, "B": 1392, "C": 1183}
REGION_RECORDS = {"I": 3135, "II": 1183}
FIT_RECORDS = 2486
VALIDATION_RECORDS = 649

PRIMARY_RHO = 0.69
ALL_PLAQUE_RHO = 0.61
REFERENCE_STRATUM_RHO = 0.53
NDCG_AT_10 = 0.74
BEST_IMAGING_RHO = 0.61
BEST_MOLECULAR_RHO = 0.59
ROUNDING = 0.005

RANK_FLOOR = 0.05
REPORTED_INCREMENT = 0.081
REPORTED_INCREMENT_INTERVAL = (0.07, 0.10)
REFERENCE_LEVEL = 0.07

GATE_CONDITIONING_SHARE_PCT = 94.3
GATE_FALSE_CONDITIONING_PCT = 5.7
GATE_CALIBRATION_BEFORE = 1.28
GATE_CALIBRATION_AFTER = 1.04

FULL_PARAMETERS_M = 150.7
WITHOUT_COUPLING_PARAMETERS_M = 146.9
WITHOUT_ROUTE_HEAD_PARAMETERS_M = 150.4
WITHOUT_COMPOSITION_HEAD_PARAMETERS_M = 149.8
WITHOUT_GATE_PARAMETERS_M = 150.7
WITHOUT_RANK_PARAMETERS_M = 150.7
WITHOUT_TOPK_PARAMETERS_M = 150.5
WITHOUT_IMAGING_PARAMETERS_M = 48.6
WITHOUT_MOLECULAR_PARAMETERS_M = 90.1
BILINEAR_CONCAT_PARAMETERS_M = 147.2
RETAINED_SHARE_PCT = 97.3

ABLATION_DELTAS = {
    "without_reaction_norm_coupling": -0.10,
    "without_regime_route_head": -0.05,
    "without_plaque_composition_head": -0.06,
    "without_interaction_variance_gate": 0.00,
    "without_agreement_weighted_rank": -0.03,
    "without_certified_topk_head": -0.01,
    "without_plaque_imaging_branch": -0.11,
    "without_molecular_branch": -0.08,
}
ABLATION_METRICS = {
    "without_reaction_norm_coupling": 0.59,
    "without_regime_route_head": 0.64,
    "without_plaque_composition_head": 0.63,
    "without_interaction_variance_gate": 0.69,
    "without_agreement_weighted_rank": 0.66,
    "without_certified_topk_head": 0.69,
    "without_plaque_imaging_branch": 0.59,
    "without_molecular_branch": 0.61,
}
JOINT_DELTAS = {
    "coupling_and_gate": (-0.14, 1.39),
    "route_and_composition": (-0.16, 1.43),
    "coupling_and_composition": (-0.21, 1.27),
    "route_and_rank": (-0.09, 1.10),
}
JOINT_MEMBERS = {
    "coupling_and_gate": ("without_reaction_norm_coupling", "without_interaction_variance_gate"),
    "route_and_composition": ("without_regime_route_head", "without_plaque_composition_head"),
    "coupling_and_composition": (
        "without_reaction_norm_coupling",
        "without_plaque_composition_head",
    ),
    "route_and_rank": ("without_regime_route_head", "without_agreement_weighted_rank"),
}
TIER3_SUBSTITUTIONS = {
    "coupling_replaced_by_bilinear_concatenation": (-0.12, 147.2),
    "three_route_head_replaced_by_two_route_head": (-0.03, 150.3),
    "agreement_weighting_replaced_by_linear": (-0.02, 150.7),
    "certified_topk_replaced_by_empirical": (0.00, 150.5),
    "composition_head_replaced_by_vulnerability_classifier": (-0.09, 148.1),
    "interaction_covariance_capped_at_rank_1": (-0.05, 149.9),
    "interaction_covariance_unconstrained": (-0.01, 151.6),
    "equal_budget_variant": (-0.02, 150.7),
}

TRANSPORT = {
    "held_out_site": (1183, 0.67, -2.4),
    "held_out_region": (1183, 0.67, -2.8),
    "later_accrual_window": (1305, 0.67, -1.9),
    "scanner_vendor_shift": (862, 0.66, -3.8),
    "contrast_without_t1w": (544, 0.63, -5.1),
    "slice_thickness_at_least_3mm": (727, 0.63, -6.2),
    "motion_degraded": (318, 0.61, -7.3),
    "small_plaque_below_40mm3": (392, 0.59, -8.2),
}
TRANSPORT_MODERATE_THRESHOLD_PP = 5.0

ASSAY_MAE = {
    "descriptor_family": 0.2571,
    "graph_family": 0.2642,
    "state_space": 0.2655,
    "equivariant_family": 0.2912,
}
ROUTE_SHARE_PCT = (21.4, 54.6)
QUARTILE_AGREEMENT = (0.87, 0.61)
IN_VITRO = {"replicates": 6, "rank_correlations": (0.31, 0.27), "p_values": (0.003, 0.007)}
READER_STUDY = {"readers": 6, "records": 250, "agreement": (0.431, 0.713)}
READER_AGREEMENT_THRESHOLD = 0.10


@dataclass(slots=True)
class ConsistencyResult:
    """Outcome of one arithmetic relation between printed quantities."""

    name: str
    holds: bool
    detail: str

    def as_record(self) -> dict[str, object]:
        return {"name": self.name, "holds": self.holds, "detail": self.detail}


def cohort_partitions_sum() -> ConsistencyResult:
    """Arm and site partitions must both recover the cohort total."""

    arm_total = PROSPECTIVE_RECORDS + RETROSPECTIVE_RECORDS
    site_total = sum(SITE_RECORDS.values())
    region_total = sum(REGION_RECORDS.values())
    fits = arm_total == TOTAL_RECORDS == site_total == region_total
    return ConsistencyResult(
        name="cohort_partitions_sum",
        holds=bool(fits),
        detail=(
            f"arms {PROSPECTIVE_RECORDS}+{RETROSPECTIVE_RECORDS}={arm_total}, "
            f"sites={site_total}, regions={region_total}, reported total {TOTAL_RECORDS}"
        ),
    )


def fit_and_validation_recover_region_one() -> ConsistencyResult:
    """The fitting and validation partitions must tile the pooled Region I count."""

    combined = FIT_RECORDS + VALIDATION_RECORDS
    holds = (
        combined == REGION_RECORDS["I"]
        and REGION_RECORDS["I"] + REGION_RECORDS["II"] == TOTAL_RECORDS
    )
    return ConsistencyResult(
        name="fit_and_validation_recover_region_one",
        holds=bool(holds),
        detail=f"{FIT_RECORDS}+{VALIDATION_RECORDS}={combined} against Region I {REGION_RECORDS['I']}",
    )


POOL_TOLERANCE = {
    "rho": 0.005,
    "calibration_slope": 0.005,
    "increment": 0.001,
    "net_benefit": 0.001,
    "mcid_crossing": 0.05,
}


def arm_weighted_pool_matches_pooled_row(
    metric: str, values: dict[str, float]
) -> ConsistencyResult:
    """The pooled row must be the size-weighted pool of the two arms."""

    sizes = np.array([PROSPECTIVE_RECORDS, RETROSPECTIVE_RECORDS], dtype=np.float64)
    arms = np.array([values["prospective"], values["retrospective"]], dtype=np.float64)
    pooled = float(np.sum(arms * sizes) / sizes.sum())
    reported = values["pooled"]
    tolerance = POOL_TOLERANCE[metric]
    holds = abs(pooled - reported) <= tolerance
    return ConsistencyResult(
        name=f"arm_weighted_pool_{metric}",
        holds=bool(holds),
        detail=f"recomputed {pooled:.4f} against reported {reported:.4f} (tolerance {tolerance})",
    )


def region_weighted_pool_matches_pooled_row(
    metric: str, values: dict[str, float]
) -> ConsistencyResult:
    """The pooled row must be the size-weighted pool of the two regions."""

    sizes = np.array([REGION_RECORDS["I"], REGION_RECORDS["II"]], dtype=np.float64)
    regions = np.array([values["region_I"], values["region_II"]], dtype=np.float64)
    pooled = float(np.sum(regions * sizes) / sizes.sum())
    reported = values["pooled"]
    tolerance = POOL_TOLERANCE[metric]
    holds = abs(pooled - reported) <= tolerance
    return ConsistencyResult(
        name=f"region_weighted_pool_{metric}",
        holds=bool(holds),
        detail=f"recomputed {pooled:.4f} against reported {reported:.4f} (tolerance {tolerance})",
    )


def gate_shares_complement() -> ConsistencyResult:
    """Conditioned and deferred shares must partition the cohort."""

    total = GATE_CONDITIONING_SHARE_PCT + GATE_FALSE_CONDITIONING_PCT
    return ConsistencyResult(
        name="gate_shares_complement",
        holds=abs(total - 100.0) < 1e-9,
        detail=f"{GATE_CONDITIONING_SHARE_PCT}+{GATE_FALSE_CONDITIONING_PCT}={total}",
    )


def coupling_parameter_share_matches() -> ConsistencyResult:
    """Dropping the coupling must cost the printed share of the parameter count."""

    share = parameter_share(FULL_PARAMETERS_M, WITHOUT_COUPLING_PARAMETERS_M) * 100.0
    return ConsistencyResult(
        name="coupling_parameter_share_matches",
        holds=abs(share - 2.5) <= 0.1,
        detail=f"removed share {share:.2f}% against the printed 2.5%",
    )


def retained_margin_share_reachable() -> ConsistencyResult:
    """The equal-budget variant's retained share must be reachable from its metric."""

    reported = RETAINED_SHARE_PCT / 100.0
    low, high = retained_share_band(0.67, 0.69, ROUNDING)
    holds = low <= reported <= high
    return ConsistencyResult(
        name="retained_margin_share_reachable",
        holds=bool(holds),
        detail=f"reported {reported:.3f} inside the band [{low:.4f}, {high:.4f}] from 0.67 against 0.69",
    )


def interaction_ratio_reachable(name: str) -> ConsistencyResult:
    """Each printed interaction ratio must be reachable from the printed losses."""

    joint_loss, reported = JOINT_DELTAS[name]
    first, second = JOINT_MEMBERS[name]
    parts = (-ABLATION_DELTAS[first], -ABLATION_DELTAS[second])
    low, high = interaction_ratio_band(-joint_loss, parts, ROUNDING)
    holds = low <= reported <= high
    return ConsistencyResult(
        name=f"interaction_ratio_reachable_{name}",
        holds=bool(holds),
        detail=f"reported {reported} inside the rounding band [{low:.3f}, {high:.3f}]",
    )


def ablation_delta_agrees_with_metrics(
    name: str, full_metric: float = PRIMARY_RHO
) -> ConsistencyResult:
    """Each removal's printed change must agree with the printed metric difference."""

    delta = ABLATION_DELTAS[name]
    metric = ABLATION_METRICS[name]
    observed = metric - full_metric
    tolerance = 0.015
    holds = abs(observed - delta) <= tolerance
    return ConsistencyResult(
        name=f"ablation_delta_{name}",
        holds=bool(holds),
        detail=f"metric difference {observed:+.3f} against printed delta {delta:+.3f}",
    )


def transport_moderate_count() -> ConsistencyResult:
    """Exactly four conditions must lose more than five points."""

    losses = [entry[2] for entry in TRANSPORT.values()]
    moderate = [value for value in losses if value < -TRANSPORT_MODERATE_THRESHOLD_PP]
    return ConsistencyResult(
        name="transport_moderate_count",
        holds=len(moderate) == 4,
        detail=f"{len(moderate)} conditions exceed {TRANSPORT_MODERATE_THRESHOLD_PP} points",
    )


def transport_largest_loss() -> ConsistencyResult:
    """The largest transport loss must be the small-plaque condition."""

    losses = {name: entry[2] for name, entry in TRANSPORT.items()}
    worst = min(losses, key=lambda key: losses[key])
    return ConsistencyResult(
        name="transport_largest_loss",
        holds=worst == "small_plaque_below_40mm3" and losses[worst] == -8.2,
        detail=f"largest loss is {worst} at {losses[worst]} points",
    )


def assay_family_ordering() -> ConsistencyResult:
    """The assay errors must order as the article describes them."""

    ordering = sorted(ASSAY_MAE.items(), key=lambda item: item[1])
    expected = [
        "descriptor_family",
        "graph_family",
        "state_space",
        "equivariant_family",
    ]
    return ConsistencyResult(
        name="assay_family_ordering",
        holds=[name for name, _ in ordering] == expected,
        detail=f"ordering is {[name for name, _ in ordering]}",
    )


def caco2_margin_approximates_one_hundredth() -> ConsistencyResult:
    """The named assay defeat must be about a hundredth of the error unit."""

    gap = ASSAY_MAE["state_space"] - ASSAY_MAE["descriptor_family"]
    return ConsistencyResult(
        name="caco2_margin_approximates_one_hundredth",
        holds=abs(gap - 0.01) <= 0.005,
        detail=f"descriptor {ASSAY_MAE['descriptor_family']} against state space {ASSAY_MAE['state_space']}, gap {gap:.4f}",
    )


def rank_floor_reachable() -> ConsistencyResult:
    """The pre-specified floor must be reachable on a bounded rank correlation."""

    headroom = 1.0 - BEST_IMAGING_RHO
    holds = headroom >= RANK_FLOOR
    return ConsistencyResult(
        name="rank_floor_reachable",
        holds=bool(holds),
        detail=f"floor {RANK_FLOOR} against headroom {headroom:.2f} over the strongest baseline",
    )


def reported_increment_matches_columns() -> ConsistencyResult:
    """The abstract's increment must agree with the two printed columns."""

    difference = PRIMARY_RHO - BEST_IMAGING_RHO
    holds = (
        difference <= REPORTED_INCREMENT_INTERVAL[1]
        and difference >= REPORTED_INCREMENT_INTERVAL[0]
    )
    return ConsistencyResult(
        name="reported_increment_matches_columns",
        holds=bool(holds),
        detail=f"column difference {difference:.3f} inside the reported interval {REPORTED_INCREMENT_INTERVAL}",
    )


def route_share_range_ordered() -> ConsistencyResult:
    """The matrix-route share must rise from its low to its high bin."""

    low, high = ROUTE_SHARE_PCT
    return ConsistencyResult(
        name="route_share_range_ordered",
        holds=high > low,
        detail=f"share rises from {low}% to {high}%",
    )


def quartile_agreement_falls() -> ConsistencyResult:
    """Agreement between the blind and the conditioned order must fall across bins."""

    first, last = QUARTILE_AGREEMENT
    return ConsistencyResult(
        name="quartile_agreement_falls",
        holds=last < first,
        detail=f"agreement falls from {first} to {last}",
    )


def gate_calibration_improves() -> ConsistencyResult:
    """The gate must reduce the calibration slope while the rank column is unchanged."""

    return ConsistencyResult(
        name="gate_calibration_improves",
        holds=GATE_CALIBRATION_AFTER < GATE_CALIBRATION_BEFORE
        and ABLATION_METRICS["without_interaction_variance_gate"] == PRIMARY_RHO,
        detail=(
            f"slope {GATE_CALIBRATION_BEFORE} to {GATE_CALIBRATION_AFTER}; "
            f"gated and ungated both at {PRIMARY_RHO}"
        ),
    )


def reference_level_below_increment() -> ConsistencyResult:
    """Retrospective increment must reach the published reference and prospective must not."""

    retrospective = 0.073
    prospective = 0.064
    return ConsistencyResult(
        name="reference_level_below_increment",
        holds=retrospective >= REFERENCE_LEVEL > prospective,
        detail=(
            f"retrospective {retrospective} at the reference {REFERENCE_LEVEL}, "
            f"prospective {prospective} below it"
        ),
    )


def tier3_substitution_ordering() -> ConsistencyResult:
    """A substitution with more parameters than the component must not be claimed a gain."""

    full_width_loss = TIER3_SUBSTITUTIONS["interaction_covariance_unconstrained"][0]
    return ConsistencyResult(
        name="tier3_substitution_ordering",
        holds=full_width_loss < 0.0,
        detail=f"unconstrained covariance carries a loss of {full_width_loss} at 151.6 M parameters",
    )


def parameter_attribution_bounds() -> ConsistencyResult:
    """Each single-branch count must fit inside the reported total, with a shared residual."""

    branch_sum = WITHOUT_MOLECULAR_PARAMETERS_M + WITHOUT_IMAGING_PARAMETERS_M
    shared = FULL_PARAMETERS_M - branch_sum
    coupling_cost = FULL_PARAMETERS_M - WITHOUT_COUPLING_PARAMETERS_M
    holds = (
        WITHOUT_IMAGING_PARAMETERS_M < FULL_PARAMETERS_M
        and WITHOUT_MOLECULAR_PARAMETERS_M < FULL_PARAMETERS_M
        and shared > coupling_cost
    )
    return ConsistencyResult(
        name="parameter_attribution_bounds",
        holds=bool(holds),
        detail=(
            f"molecular-only {WITHOUT_IMAGING_PARAMETERS_M} M, imaging-only "
            f"{WITHOUT_MOLECULAR_PARAMETERS_M} M, shared residual {shared:.1f} M against a "
            f"coupling cost of {coupling_cost:.1f} M"
        ),
    )


def all_consistency_results() -> tuple[ConsistencyResult, ...]:
    """Run every arithmetic relation the manuscript's own tables support."""

    arm_increment = {"prospective": 0.064, "retrospective": 0.073, "pooled": 0.071}
    arm_benefit = {"prospective": 0.027, "retrospective": 0.030, "pooled": 0.029}
    arm_crossing = {"prospective": 71.4, "retrospective": 74.8, "pooled": 73.9}
    arm_slope = {"prospective": 1.07, "retrospective": 1.02, "pooled": 1.04}
    arm_rho = {"prospective": 0.67, "retrospective": 0.70, "pooled": 0.69}
    region_increment = {"region_I": 0.072, "region_II": 0.061, "pooled": 0.071}
    region_benefit = {"region_I": 0.030, "region_II": 0.025, "pooled": 0.029}
    region_crossing = {"region_I": 74.8, "region_II": 69.4, "pooled": 73.9}
    region_slope = {"region_I": 1.04, "region_II": 1.09, "pooled": 1.04}
    region_rho = {"region_I": 0.70, "region_II": 0.67, "pooled": 0.69}
    results = [
        cohort_partitions_sum(),
        fit_and_validation_recover_region_one(),
        arm_weighted_pool_matches_pooled_row("rho", arm_rho),
        arm_weighted_pool_matches_pooled_row("increment", arm_increment),
        arm_weighted_pool_matches_pooled_row("net_benefit", arm_benefit),
        arm_weighted_pool_matches_pooled_row("mcid_crossing", arm_crossing),
        arm_weighted_pool_matches_pooled_row("calibration_slope", arm_slope),
        region_weighted_pool_matches_pooled_row("rho", region_rho),
        region_weighted_pool_matches_pooled_row("increment", region_increment),
        region_weighted_pool_matches_pooled_row("net_benefit", region_benefit),
        region_weighted_pool_matches_pooled_row("mcid_crossing", region_crossing),
        region_weighted_pool_matches_pooled_row("calibration_slope", region_slope),
        gate_shares_complement(),
        coupling_parameter_share_matches(),
        retained_margin_share_reachable(),
        transport_moderate_count(),
        transport_largest_loss(),
        assay_family_ordering(),
        caco2_margin_approximates_one_hundredth(),
        rank_floor_reachable(),
        reported_increment_matches_columns(),
        route_share_range_ordered(),
        quartile_agreement_falls(),
        gate_calibration_improves(),
        reference_level_below_increment(),
        tier3_substitution_ordering(),
        parameter_attribution_bounds(),
    ]
    results.extend(interaction_ratio_reachable(name) for name in JOINT_DELTAS)
    results.extend(ablation_delta_agrees_with_metrics(name) for name in ABLATION_DELTAS)
    return tuple(results)
