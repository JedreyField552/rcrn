"""Decline-to-condition gate.

Ref: Methods, "Reaction-norm coupling and the decline-to-condition gate", Eq. (1)
and Eq. (2); Table 2 Tier 1 (without the interaction-variance gate is null on the
primary stratum but reduces the calibration slope); Discussion (the gate verifies
conditioning for most records and defers the rest).

The threshold is chosen on held-out training folds so that the share of records
with no usable interaction signal that still get conditioned stays under the
declared tolerance. Because the feasible set of the gated scorer is a subset of the
ungated one, the constraint can only be reported as a cost, never as a gain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rcrn.models.interaction import InteractionSurfaces

DEFAULT_GRID_SIZE = 201


@dataclass(slots=True)
class GateDecision:
    """Which records are conditioned and which fall back to the plaque-blind score."""

    conditioned: np.ndarray
    threshold: float
    conditioning_share: float
    deferral_share: float
    false_conditioning_rate: float
    signal_reference_used: bool

    def is_degenerate(self) -> bool:
        return self.conditioning_share <= 0.0


@dataclass(slots=True)
class GateCalibration:
    """Threshold search record."""

    threshold: float
    tolerance: float
    false_conditioning_rate: float
    candidates_tried: int
    signal_reference_used: bool = False
    grid: np.ndarray = field(default_factory=lambda: np.zeros(0))


def false_conditioning_rate(
    statistic: np.ndarray, no_signal: np.ndarray, threshold: float
) -> float:
    """Share of no-signal records that the gate would still condition on."""

    if no_signal.sum() == 0:
        return 0.0
    conditioned = statistic[no_signal] > threshold
    return float(np.mean(conditioned))


def calibrate_threshold(
    surfaces: InteractionSurfaces,
    signal_reference: np.ndarray | None,
    tolerance: float,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> GateCalibration:
    """Smallest threshold whose false-conditioning rate stays under tolerance."""

    statistic = surfaces.statistic.detach().cpu().numpy()
    if signal_reference is None:
        floor = float(np.quantile(statistic, 0.25))
        no_signal = statistic <= floor
        used_reference = False
    else:
        reference = np.asarray(signal_reference, dtype=np.float64)
        floor = float(np.quantile(reference, 0.25))
        no_signal = reference <= floor
        used_reference = True
    grid = np.quantile(statistic, np.linspace(0.0, 1.0, grid_size))
    grid = np.unique(grid)
    chosen = float(grid[-1])
    rate = 0.0
    for candidate in grid:
        rate = false_conditioning_rate(statistic, no_signal, float(candidate))
        if rate <= tolerance:
            chosen = float(candidate)
            break
    return GateCalibration(
        threshold=chosen,
        tolerance=tolerance,
        false_conditioning_rate=rate,
        candidates_tried=int(grid.size),
        signal_reference_used=used_reference,
        grid=grid,
    )


class DeclineToConditionGate:
    """Apply a calibrated threshold to an interaction statistic."""

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = threshold
        self.calibration: GateCalibration | None = None

    def calibrate(
        self,
        surfaces: InteractionSurfaces,
        signal_reference: np.ndarray | None,
        tolerance: float,
        grid_size: int = DEFAULT_GRID_SIZE,
    ) -> GateCalibration:
        calibration = calibrate_threshold(surfaces, signal_reference, tolerance, grid_size)
        self.calibration = calibration
        self.threshold = calibration.threshold
        return calibration

    def decide(
        self,
        surfaces: InteractionSurfaces,
        signal_reference: np.ndarray | None = None,
    ) -> GateDecision:
        if self.threshold is None:
            raise RuntimeError("the gate threshold has not been calibrated")
        statistic = surfaces.statistic.detach().cpu().numpy()
        conditioned = statistic > self.threshold
        if signal_reference is None:
            floor = float(np.quantile(statistic, 0.25))
            no_signal = statistic <= floor
            used_reference = False
        else:
            reference = np.asarray(signal_reference, dtype=np.float64)
            no_signal = reference <= float(np.quantile(reference, 0.25))
            used_reference = True
        return GateDecision(
            conditioned=conditioned,
            threshold=float(self.threshold),
            conditioning_share=float(np.mean(conditioned)),
            deferral_share=float(1.0 - np.mean(conditioned)),
            false_conditioning_rate=false_conditioning_rate(
                statistic, no_signal, float(self.threshold)
            ),
            signal_reference_used=used_reference,
        )
