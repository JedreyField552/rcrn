"""Site-stratified bootstrap intervals.

Ref: Results, Section 2.1 (the primary interval is a bootstrap interval over 2,000
resamples of the plaque records); Methods, "Evaluation protocol and statistical
analysis" (blocks are drawn while respecting the within-block structure).

Resampling happens inside each site so that a resample keeps the site composition
of the cohort. The paired variant resamples record positions once and evaluates
both scorers on the same draw, which is what a paired interval needs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 2000


@dataclass(slots=True)
class Interval:
    """A point estimate with its bootstrap interval."""

    point: float
    low: float
    high: float
    resamples: int
    alpha: float

    def half_width(self) -> float:
        return 0.5 * (self.high - self.low)

    def excludes(self, value: float) -> bool:
        return value < self.low or value > self.high

    def as_record(self) -> dict[str, float]:
        return {
            "point": self.point,
            "low": self.low,
            "high": self.high,
            "resamples": float(self.resamples),
            "alpha": self.alpha,
        }


def stratified_indices(blocks: Sequence[str], resamples: int, seed: int) -> list[np.ndarray]:
    """Draw resamples that keep every block's size and membership structure."""

    if resamples < 1:
        raise ValueError("at least one resample is required")
    labels = np.asarray(list(blocks))
    unique = sorted(set(labels.tolist()))
    groups = {label: np.flatnonzero(labels == label) for label in unique}
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    for _ in range(resamples):
        pieces = [
            rng.choice(groups[label], size=groups[label].size, replace=True) for label in unique
        ]
        draws.append(np.concatenate(pieces))
    return draws


def bootstrap_interval(
    statistic: Callable[[np.ndarray], float],
    blocks: Sequence[str],
    point_value: float,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval of a statistic over a stratified resample."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    draws = stratified_indices(blocks, resamples, seed)
    values = np.array([statistic(index) for index in draws], dtype=np.float64)
    low, high = np.percentile(values, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return Interval(
        point=float(point_value),
        low=float(low),
        high=float(high),
        resamples=resamples,
        alpha=alpha,
    )


def paired_bootstrap_interval(
    statistic: Callable[[np.ndarray], float],
    blocks: Sequence[str],
    point_value: float,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Alias of the stratified interval, kept explicit for the paired contrasts."""

    return bootstrap_interval(statistic, blocks, point_value, resamples, alpha, seed)


def stratified_ranks(blocks: Sequence[str]) -> np.ndarray:
    """Block identifier of every record, as an integer code."""

    labels = np.asarray(list(blocks))
    unique = sorted(set(labels.tolist()))
    lookup = {label: position for position, label in enumerate(unique)}
    return np.array([lookup[label] for label in labels.tolist()], dtype=np.int64)


def minimum_detectable_shift(interval: Interval) -> float:
    """Half the interval width, used as the granularity the design can resolve."""

    return interval.half_width()
