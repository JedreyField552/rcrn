"""Reader-agreement statistics.

Ref: Results, Section 2.6 (reader agreement before and after the prioritisation
ordering); Discussion (the hard-stratum agreement constrains every imaging-based
claim); Methods, "Evaluation protocol and statistical analysis" (the first
human-factors study uses six readers under identical blinded conditions with a
washout period).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def cohen_kappa(first: Sequence[int], second: Sequence[int], levels: int | None = None) -> float:
    """Chance-corrected agreement between two raters on a nominal scale."""

    left = np.asarray(first, dtype=np.int64)
    right = np.asarray(second, dtype=np.int64)
    if left.shape != right.shape:
        raise ValueError("rater vectors must be aligned")
    if left.size == 0:
        return 0.0
    categories = levels if levels is not None else int(max(left.max(), right.max())) + 1
    confusion = np.zeros((categories, categories), dtype=np.float64)
    for observed, predicted in zip(left.tolist(), right.tolist(), strict=True):
        confusion[observed, predicted] += 1.0
    total = confusion.sum()
    observed_agreement = float(np.trace(confusion) / total)
    row = confusion.sum(axis=1) / total
    column = confusion.sum(axis=0) / total
    expected_agreement = float(np.sum(row * column))
    if expected_agreement >= 1.0:
        return 1.0
    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)


def weighted_kappa(
    first: Sequence[int], second: Sequence[int], levels: int | None = None, quadratic: bool = True
) -> float:
    """Chance-corrected agreement with a graded disagreement weight."""

    left = np.asarray(first, dtype=np.int64)
    right = np.asarray(second, dtype=np.int64)
    if left.shape != right.shape:
        raise ValueError("rater vectors must be aligned")
    if left.size == 0:
        return 0.0
    categories = levels if levels is not None else int(max(left.max(), right.max())) + 1
    confusion = np.zeros((categories, categories), dtype=np.float64)
    for observed, predicted in zip(left.tolist(), right.tolist(), strict=True):
        confusion[observed, predicted] += 1.0
    index = np.arange(categories, dtype=np.float64)
    distance = np.abs(index[:, None] - index[None, :])
    weights = (distance / max(1.0, categories - 1)) ** (2.0 if quadratic else 1.0)
    total = confusion.sum()
    expected = np.outer(confusion.sum(axis=1), confusion.sum(axis=0)) / total
    observed_disagreement = float(np.sum(weights * confusion) / total)
    expected_disagreement = float(np.sum(weights * expected) / total)
    if expected_disagreement == 0.0:
        return 1.0
    return 1.0 - observed_disagreement / expected_disagreement


def fleiss_kappa(ratings: np.ndarray) -> float:
    """Agreement across an arbitrary number of raters on a nominal scale.

    ``ratings`` holds one row per subject and one column per category, counting how
    many raters chose that category.
    """

    table = np.asarray(ratings, dtype=np.float64)
    if table.ndim != 2:
        raise ValueError("expected a subject by category count table")
    subjects, _ = table.shape
    if subjects == 0:
        return 0.0
    raters = table.sum(axis=1)
    if not np.allclose(raters, raters[0]):
        raise ValueError("Fleiss' kappa expects a constant rater count per subject")
    if raters[0] < 2:
        return 0.0
    per_subject = (np.sum(table**2, axis=1) - raters) / (raters * (raters - 1.0))
    observed = float(per_subject.mean())
    total = table.sum()
    marginals = table.sum(axis=0) / total
    expected = float(np.sum(marginals**2))
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def agreement_improvement(before: float, after: float) -> float:
    """Absolute change in an agreement coefficient."""

    return float(after - before)


def agreement_confidence(coefficient: float, threshold: float) -> str:
    """Label an agreement coefficient against a published reference level."""

    if coefficient >= threshold:
        return "at_or_above_reference"
    return "below_reference"
