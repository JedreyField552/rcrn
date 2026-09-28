"""Multiplicity control.

Ref: Results, Section 2.1 (Holm-Bonferroni over twelve planned tests, false
discovery rate 0.05 for the exploratory set); Methods, "Evaluation protocol and
statistical analysis".

The Holm step-down carries the running *maximum* of the scaled p-values in p-value
order, not the running minimum: the family-wise guarantee comes from the fact that
the adjusted values are non-decreasing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MULTIPLICITY_METHODS: tuple[str, ...] = ("holm", "bonferroni", "benjamini_hochberg")


@dataclass(slots=True)
class AdjustedFamily:
    """Adjusted p-values together with the rejections at the chosen level."""

    method: str
    level: float
    raw: np.ndarray
    adjusted: np.ndarray
    rejected: np.ndarray

    def rejection_count(self) -> int:
        return int(self.rejected.sum())

    def as_records(self) -> tuple[dict[str, float | bool], ...]:
        return tuple(
            {
                "raw": float(self.raw[index]),
                "adjusted": float(self.adjusted[index]),
                "rejected": bool(self.rejected[index]),
            }
            for index in range(self.raw.size)
        )


def holm_bonferroni(pvalues: np.ndarray, level: float = 0.05) -> np.ndarray:
    """Holm step-down adjusted p-values."""

    raw = np.asarray(pvalues, dtype=np.float64)
    if raw.ndim != 1:
        raise ValueError("expected a one-dimensional p-value vector")
    count = raw.size
    if count == 0:
        return raw.copy()
    order = np.argsort(raw, kind="stable")
    scaled = raw[order] * np.arange(count, 0, -1)
    running = np.maximum.accumulate(scaled)
    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = np.clip(running, 0.0, 1.0)
    return adjusted


def bonferroni(pvalues: np.ndarray) -> np.ndarray:
    """Single-step Bonferroni adjustment."""

    raw = np.asarray(pvalues, dtype=np.float64)
    return np.clip(raw * raw.size, 0.0, 1.0)


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values for a false discovery rate."""

    raw = np.asarray(pvalues, dtype=np.float64)
    if raw.ndim != 1:
        raise ValueError("expected a one-dimensional p-value vector")
    count = raw.size
    if count == 0:
        return raw.copy()
    order = np.argsort(-raw, kind="stable")
    ranks = np.arange(count, 0, -1)
    scaled = raw[order] * count / ranks
    running = np.minimum.accumulate(scaled)
    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = np.clip(running, 0.0, 1.0)
    return adjusted


def adjust_family(pvalues: np.ndarray, method: str = "holm", level: float = 0.05) -> AdjustedFamily:
    """Adjust a family of planned comparisons and mark the rejections."""

    if method not in MULTIPLICITY_METHODS:
        raise ValueError(f"unknown multiplicity method {method}")
    raw = np.asarray(pvalues, dtype=np.float64)
    if method == "holm":
        adjusted = holm_bonferroni(raw, level)
    elif method == "bonferroni":
        adjusted = bonferroni(raw)
    else:
        adjusted = benjamini_hochberg(raw)
    return AdjustedFamily(
        method=method,
        level=level,
        raw=raw,
        adjusted=adjusted,
        rejected=adjusted <= level,
    )


def planned_family_size(config_value: int | None = None) -> int:
    """The number of planned tests the confirmatory family carries."""

    if config_value is None:
        return 1
    if config_value < 1:
        raise ValueError("a family must contain at least one test")
    return int(config_value)
