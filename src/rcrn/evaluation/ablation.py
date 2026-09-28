"""Component-necessity arithmetic.

Ref: Table 2 and its caption; Results, Section 2.3; Discussion (the joint removals
are synergistic rather than stacked, and a substitution with more parameters than
the component it replaces loses more support).

The interaction ratio is the joint loss divided by the sum of the two single
losses. Because the manuscript prints the losses to two decimals, the ratio is only
determined up to a rounding band; the band is computed here rather than assumed, so
a reported ratio can be checked against the table it comes from.
"""

from __future__ import annotations

from dataclasses import dataclass

ABLATION_TIERS: tuple[str, ...] = (
    "tier_0_control",
    "tier_1_removal",
    "tier_2_joint",
    "tier_3_substitution",
)


@dataclass(slots=True)
class AblationRow:
    """One variant of the necessity table."""

    tier: str
    variant: str
    metric: float
    delta: float
    parameters_m: float
    linked_equation: str

    def loss(self) -> float:
        return -self.delta


def interaction_ratio(joint_loss: float, single_losses: tuple[float, float]) -> float:
    """Joint loss over the sum of the separate losses."""

    total = sum(single_losses)
    if total <= 0.0:
        raise ValueError("the separate losses must be positive")
    return joint_loss / total


def interaction_ratio_band(
    joint_loss: float,
    single_losses: tuple[float, float],
    rounding: float = 0.005,
) -> tuple[float, float]:
    """Range of the ratio consistent with the printed losses at a given rounding."""

    numerator_high = joint_loss + rounding
    numerator_low = max(joint_loss - rounding, 0.0)
    denominator_low = sum(single_losses) - 2.0 * rounding
    denominator_high = sum(single_losses) + 2.0 * rounding
    if denominator_low <= 0.0:
        raise ValueError("the separate losses are too small for this rounding")
    return (numerator_low / denominator_high, numerator_high / denominator_low)


def ratio_within_band(
    reported: float,
    joint_loss: float,
    single_losses: tuple[float, float],
    rounding: float = 0.005,
) -> bool:
    """Whether a reported ratio is reachable from the printed losses."""

    low, high = interaction_ratio_band(joint_loss, single_losses, rounding)
    return low <= reported <= high


def ratio_from_rows(
    reported: float,
    joint: AblationRow,
    first: AblationRow,
    second: AblationRow,
    rounding: float = 0.005,
) -> bool:
    """Convenience wrapper over :func:`ratio_within_band` for three table rows."""

    return ratio_within_band(reported, joint.loss(), (first.loss(), second.loss()), rounding)


def parameter_share(full_parameters_m: float, reduced_parameters_m: float) -> float:
    """Fraction of the parameter count removed by dropping a component."""

    if full_parameters_m <= 0.0:
        raise ValueError("the full parameter count must be positive")
    return (full_parameters_m - reduced_parameters_m) / full_parameters_m


def retained_share(variant_metric: float, full_metric: float) -> float:
    """Metric retained by a variant relative to the full model."""

    if full_metric == 0.0:
        raise ValueError("the full-model metric must be non-zero")
    return variant_metric / full_metric


def retained_share_band(
    variant_metric: float, full_metric: float, rounding: float = 0.005
) -> tuple[float, float]:
    """Range of the retained share consistent with two printed metrics."""

    return (
        (variant_metric - rounding) / (full_metric + rounding),
        (variant_metric + rounding) / (full_metric - rounding),
    )


def largest_single_loss(rows: tuple[AblationRow, ...]) -> AblationRow:
    """The removal with the largest single cost, which the manuscript names."""

    removals = [row for row in rows if row.tier == "tier_1_removal"]
    if not removals:
        raise ValueError("no removal rows to compare")
    return min(removals, key=lambda row: row.delta)


def null_rows(rows: tuple[AblationRow, ...], tolerance: float = 0.005) -> tuple[AblationRow, ...]:
    """Variants whose change is null at the reported precision."""

    return tuple(row for row in rows if abs(row.delta) <= tolerance)


def synergistic_joint(
    joint: AblationRow, first: AblationRow, second: AblationRow, tolerance: float = 0.005
) -> bool:
    """Whether a joint removal costs more than its two separate removals together."""

    return joint.loss() > (first.loss() + second.loss()) + tolerance


def joint_rows(rows: tuple[AblationRow, ...]) -> tuple[AblationRow, ...]:
    """The Tier 2 rows of the necessity table."""

    return tuple(row for row in rows if row.tier == "tier_2_joint")


def removal_rows(rows: tuple[AblationRow, ...]) -> tuple[AblationRow, ...]:
    """The Tier 1 rows of the necessity table."""

    return tuple(row for row in rows if row.tier == "tier_1_removal")


def substitution_rows(rows: tuple[AblationRow, ...]) -> tuple[AblationRow, ...]:
    """The Tier 3 rows of the necessity table."""

    return tuple(row for row in rows if row.tier == "tier_3_substitution")


def control_rows(rows: tuple[AblationRow, ...]) -> tuple[AblationRow, ...]:
    """The Tier 0 negative-control rows, which must collapse towards chance."""

    return tuple(row for row in rows if row.tier == "tier_0_control")


def control_collapse(rows: tuple[AblationRow, ...], ceiling: float) -> bool:
    """Whether every negative control sits at or below the chance ceiling."""

    controls = control_rows(rows)
    if not controls:
        return False
    return all(row.metric <= ceiling for row in controls)
