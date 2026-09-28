"""Out-of-distribution evaluation over site, region, accrual and acquisition folds.

Ref: Table 3; Figure 3; Results, Section 2.5.

A fold is a record selection, not a fitting step: the same fitted scorer is applied
unchanged and the paired within-record difference is reported alongside the
difference of the two marginal columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcrn.metrics.ranking import spearman_rho
from rcrn.metrics.transport import MODERATE_THRESHOLD_PP, TransportRow, transport_row


@dataclass(slots=True)
class FoldComparison:
    """In-distribution and out-of-distribution agreement on one fold."""

    condition: str
    records: int
    in_distribution: float
    out_of_distribution: float
    paired_deltas: np.ndarray

    def as_row(self, threshold: float = MODERATE_THRESHOLD_PP) -> TransportRow:
        return transport_row(
            self.condition,
            self.records,
            self.in_distribution,
            self.out_of_distribution,
            self.paired_deltas,
            threshold,
        )


def paired_agreement_deltas(
    reference_scores: np.ndarray,
    reference_targets: np.ndarray,
    fold_scores: np.ndarray,
    fold_targets: np.ndarray,
) -> np.ndarray:
    """Per-record change in rank agreement for the records present in both sets."""

    if reference_scores.shape != fold_scores.shape:
        raise ValueError("the two score matrices must be aligned")
    deltas = np.zeros(reference_scores.shape[0], dtype=np.float64)
    for row in range(reference_scores.shape[0]):
        deltas[row] = spearman_rho(fold_scores[row], fold_targets[row]) - spearman_rho(
            reference_scores[row], reference_targets[row]
        )
    return deltas


def fold_comparison(
    condition: str,
    in_distribution_scores: np.ndarray,
    in_distribution_targets: np.ndarray,
    fold_scores: np.ndarray,
    fold_targets: np.ndarray,
) -> FoldComparison:
    """Compare one condition against the pooled in-distribution reference."""

    in_distribution = float(
        np.mean(
            [
                spearman_rho(in_distribution_scores[row], in_distribution_targets[row])
                for row in range(in_distribution_scores.shape[0])
            ]
        )
    )
    out_of_distribution = float(
        np.mean(
            [
                spearman_rho(fold_scores[row], fold_targets[row])
                for row in range(fold_scores.shape[0])
            ]
        )
    )
    deltas = paired_agreement_deltas(
        in_distribution_scores, in_distribution_targets, fold_scores, fold_targets
    )
    return FoldComparison(
        condition=condition,
        records=int(fold_scores.shape[0]),
        in_distribution=in_distribution,
        out_of_distribution=out_of_distribution,
        paired_deltas=deltas,
    )


def retained_ordering_share(
    in_distribution: float, out_of_distribution: float, reference_loss_pp: float
) -> float:
    """Share of the field's own cross-site loss that the condition stays inside."""

    observed_loss = (in_distribution - out_of_distribution) * 100.0
    if reference_loss_pp <= 0.0:
        raise ValueError("the reference loss must be positive")
    return observed_loss / reference_loss_pp


def acquisition_axis_attribution(rows: tuple[TransportRow, ...]) -> dict[str, float]:
    """Total loss attributed to the imaging axis rather than to the coupling."""

    imaging_axes = {
        "contrast_without_t1w",
        "slice_thickness_at_least_3mm",
        "motion_degraded",
        "small_plaque_below_40mm3",
        "scanner_vendor_shift",
    }
    attribution = {"imaging_axis": 0.0, "identity_or_time_axis": 0.0}
    for row in rows:
        target = "imaging_axis" if row.condition in imaging_axes else "identity_or_time_axis"
        attribution[target] += -row.paired_delta_pp
    return attribution
