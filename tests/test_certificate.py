"""Certified top-k prioritisation."""

from __future__ import annotations

import numpy as np
import pytest

from rcrn.metrics.certificate import (
    certified_batch,
    certified_top_k,
    expected_deferral_rate,
    flip_rate_under_perturbation,
    molecule_radius,
)


def _susceptibility(molecules: int = 6) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.abs(rng.normal(size=(molecules, 3)))


def test_molecule_radius_is_the_scaled_norm() -> None:
    susceptibility = _susceptibility()
    radii = molecule_radius(susceptibility, 2.0)
    assert np.allclose(radii, 2.0 * np.linalg.norm(susceptibility, axis=1))


def test_wide_margin_is_certified() -> None:
    scores = np.array([9.0, 8.0, 0.1, 0.05, 0.0, -0.1])
    result = certified_top_k("R", scores, _susceptibility(), 2, 0.01, 1.0)
    assert result.certified is True
    assert result.deferred == ()
    assert result.top_k == (0, 1)


def test_narrow_margin_is_deferred() -> None:
    scores = np.array([1.0, 0.99, 0.98, 0.97, 0.96, 0.95])
    result = certified_top_k("R", scores, _susceptibility(), 2, 0.5, 1.0)
    assert result.certified is False
    assert result.deferred
    assert set(result.deferred) <= set(result.top_k)


def test_margin_and_radius_follow_the_definition() -> None:
    scores = np.array([4.0, 3.0, 1.0, 0.5])
    susceptibility = _susceptibility(4)
    result = certified_top_k("R", scores, susceptibility, 2, 0.1, 1.5)
    expected_margin = min(scores[i] - scores[j] for i in (0, 1) for j in (2, 3))
    expected_radius = 2.0 * 0.1 * float(np.max(1.5 * np.linalg.norm(susceptibility, axis=1)))
    assert result.margin == pytest.approx(expected_margin)
    assert result.radius == pytest.approx(expected_radius)
    assert result.certified == (expected_margin > expected_radius)


def test_batch_reports_the_deferral_rate() -> None:
    scores = np.tile(np.array([1.0, 0.9, 0.2, 0.1]), (3, 1))
    report = certified_batch(("R0", "R1", "R2"), scores, _susceptibility(4), 2, 0.0, 1.0)
    assert report.depth == 2
    assert 0.0 <= report.deferral_rate <= 1.0
    assert expected_deferral_rate(report, 2) == pytest.approx(report.deferral_rate)


def test_zero_radius_certifies_any_strict_margin() -> None:
    scores = np.array([2.0, 1.1, 0.9, 0.8])
    result = certified_top_k("R", scores, _susceptibility(4), 2, 0.0, 1.0)
    assert result.certified is True


def test_depth_must_sit_inside_the_list() -> None:
    with pytest.raises(ValueError):
        certified_top_k("R", np.zeros(4), _susceptibility(4), 4, 0.1, 1.0)
    with pytest.raises(ValueError):
        certified_top_k("R", np.zeros(4), _susceptibility(4), 0, 0.1, 1.0)


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError):
        certified_top_k("R", np.zeros(4), _susceptibility(4), 2, 0.1, 1.0, "nonsense")


def test_flip_rate_is_zero_for_identical_scores() -> None:
    scores = np.random.default_rng(1).normal(size=(4, 6))
    assert flip_rate_under_perturbation(scores, scores, 3) == 0.0


def test_flip_rate_is_positive_when_the_order_changes() -> None:
    clean = np.zeros((1, 6))
    perturbed = np.zeros((1, 6))
    perturbed[0, 3] = 1.0
    assert flip_rate_under_perturbation(clean, perturbed, 1) == 1.0


def test_batch_requires_aligned_rows() -> None:
    with pytest.raises(ValueError):
        certified_batch(("R0",), np.zeros((2, 4)), _susceptibility(4), 2, 0.1, 1.0)
