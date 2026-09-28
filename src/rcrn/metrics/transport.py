"""Transport degradation over the out-of-distribution folds.

Ref: Table 3; Figure 3; Results, Section 2.5.

Every condition entered no fitting step, and the reported change is the mean paired
difference within records, which can differ from the difference of the two marginal
columns. Both quantities are carried here so the divergence the table caption
describes is visible rather than implied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEGRADATION_CLASSES: tuple[str, ...] = ("graceful", "moderate")
MODERATE_THRESHOLD_PP = 5.0


@dataclass(slots=True)
class TransportRow:
    """One out-of-distribution condition and its cost in percentage points."""

    condition: str
    records: int
    in_distribution: float
    out_of_distribution: float
    paired_delta_pp: float
    marginal_delta_pp: float
    degradation: str

    def cost_pp(self) -> float:
        return -self.marginal_delta_pp


def degradation_class(delta_pp: float, threshold: float = MODERATE_THRESHOLD_PP) -> str:
    """A loss of more than the threshold counts as moderate, otherwise graceful."""

    return "moderate" if delta_pp < -threshold else "graceful"


def transport_row(
    condition: str,
    records: int,
    in_distribution: float,
    out_of_distribution: float,
    paired_deltas: np.ndarray | None = None,
    threshold: float = MODERATE_THRESHOLD_PP,
) -> TransportRow:
    """Build one row, preferring the within-record paired difference when given."""

    marginal = (out_of_distribution - in_distribution) * 100.0
    if paired_deltas is None or paired_deltas.size == 0:
        paired = marginal
    else:
        paired = float(np.mean(paired_deltas) * 100.0)
    return TransportRow(
        condition=condition,
        records=int(records),
        in_distribution=float(in_distribution),
        out_of_distribution=float(out_of_distribution),
        paired_delta_pp=paired,
        marginal_delta_pp=float(marginal),
        degradation=degradation_class(paired, threshold),
    )


def transport_summary(
    rows: tuple[TransportRow, ...], threshold: float = MODERATE_THRESHOLD_PP
) -> dict[str, float]:
    """Count graceful and moderate conditions and find the largest loss."""

    if not rows:
        return {"conditions": 0.0, "graceful": 0.0, "moderate": 0.0, "largest_loss_pp": 0.0}
    delta = np.array([row.paired_delta_pp for row in rows], dtype=np.float64)
    moderate = int(np.sum(delta < -threshold))
    return {
        "conditions": float(len(rows)),
        "graceful": float(len(rows) - moderate),
        "moderate": float(moderate),
        "largest_loss_pp": float(-delta.min()),
        "smallest_loss_pp": float(-delta.max()),
    }


def paired_marginal_divergence(rows: tuple[TransportRow, ...]) -> np.ndarray:
    """Signed gap between the paired and the marginal difference of every row."""

    return np.array([row.paired_delta_pp - row.marginal_delta_pp for row in rows], dtype=np.float64)


def losses_exceeding(
    rows: tuple[TransportRow, ...], threshold: float = MODERATE_THRESHOLD_PP
) -> tuple[str, ...]:
    """Conditions whose paired loss exceeds the degradation threshold."""

    return tuple(row.condition for row in rows if row.degradation == "moderate")


def transport_cost_profile(rows: tuple[TransportRow, ...]) -> dict[str, float]:
    """Loss attributed to each acquisition axis, for the discussion of Figure 3."""

    profile: dict[str, float] = {}
    for row in rows:
        profile[row.condition] = row.paired_delta_pp
    return profile
