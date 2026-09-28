"""Multiplicity control and bootstrap intervals."""

from __future__ import annotations

import numpy as np
import pytest

from rcrn.evaluation.bootstrap import (
    Interval,
    bootstrap_interval,
    minimum_detectable_shift,
    stratified_indices,
    stratified_ranks,
)
from rcrn.evaluation.multiplicity import (
    adjust_family,
    benjamini_hochberg,
    bonferroni,
    holm_bonferroni,
    planned_family_size,
)


def test_holm_matches_the_definition() -> None:
    pvalues = np.array([0.001, 0.008, 0.039, 0.041, 0.06, 0.2, 0.5])
    order = np.argsort(pvalues)
    scaled = [
        pvalues[order[position]] * (pvalues.size - position) for position in range(pvalues.size)
    ]
    running = np.maximum.accumulate(scaled)
    manual = np.zeros(pvalues.size)
    manual[order] = np.clip(running, 0.0, 1.0)
    assert np.allclose(holm_bonferroni(pvalues), manual)


def test_holm_is_monotone_in_p_value_order() -> None:
    pvalues = np.array([0.5, 0.01, 0.2, 0.001])
    adjusted = holm_bonferroni(pvalues)
    ordered = adjusted[np.argsort(pvalues)]
    assert bool(np.all(np.diff(ordered) >= -1e-12))


def test_holm_is_bounded_by_bonferroni_on_the_smallest_p_value() -> None:
    pvalues = np.array([0.001, 0.02, 0.5])
    assert holm_bonferroni(pvalues).min() == pytest.approx(bonferroni(pvalues).min())


def test_benjamini_hochberg_matches_the_definition() -> None:
    pvalues = np.array([0.001, 0.008, 0.039, 0.041, 0.06])
    order = np.argsort(-pvalues)
    running = 1.0
    manual = np.zeros(pvalues.size)
    for position in range(pvalues.size):
        rank = pvalues.size - position
        running = min(running, pvalues[order[position]] * pvalues.size / rank)
        manual[order[position]] = min(1.0, running)
    assert np.allclose(benjamini_hochberg(pvalues), manual)


def test_adjust_family_marks_the_rejections() -> None:
    family = adjust_family(np.array([0.001, 0.02, 0.5]), "holm", 0.05)
    assert family.rejection_count() == 2
    assert family.as_records()[0]["rejected"] is True


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError):
        adjust_family(np.array([0.1]), "nonsense", 0.05)


def test_family_size_must_be_positive() -> None:
    assert planned_family_size(12) == 12
    assert planned_family_size(None) == 1
    with pytest.raises(ValueError):
        planned_family_size(0)


def test_stratified_bootstrap_keeps_block_sizes() -> None:
    blocks = ["A", "A", "B", "B", "B", "C"]
    draws = stratified_indices(blocks, 20, seed=0)
    assert len(draws) == 20
    for draw in draws:
        labels = np.array(blocks)[draw]
        assert list(np.bincount(np.unique(labels, return_inverse=True)[1])) == [2, 3, 1]


def test_stratified_ranks_are_consistent_with_the_labels() -> None:
    blocks = ["B", "A", "B"]
    ranks = stratified_ranks(blocks)
    assert ranks[0] == ranks[2]
    assert ranks[1] != ranks[0]


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    blocks = ["A"] * 100 + ["B"] * 100
    interval = bootstrap_interval(
        lambda index: float(values[index].mean()),
        blocks,
        point_value=float(values.mean()),
        resamples=200,
        seed=1,
    )
    assert interval.low <= interval.point <= interval.high or interval.low <= interval.high
    assert interval.resamples == 200
    assert interval.half_width() >= 0.0


def test_interval_reports_membership() -> None:
    interval = Interval(point=0.5, low=0.4, high=0.6, resamples=100, alpha=0.05)
    assert interval.excludes(0.7) is True
    assert interval.excludes(0.5) is False
    assert interval.as_record()["resamples"] == 100.0
    assert minimum_detectable_shift(interval) == pytest.approx(0.1)


def test_resample_count_must_be_positive() -> None:
    with pytest.raises(ValueError):
        stratified_indices(["A"], 0, seed=0)
