"""Clinical-increment and calibration statistics."""

from __future__ import annotations

import numpy as np
import pytest

from rcrn.metrics.calibration import (
    calibration_in_the_large,
    calibration_intercept,
    calibration_ratio,
    calibration_slope,
    logistic_fit,
)
from rcrn.metrics.clinical import (
    auc,
    auc_variance,
    continuous_nri,
    decision_curve_table,
    incremental_auc,
    mcid_crossing_share,
    net_benefit,
    weighted_pool,
)


def _separable_but_imperfect() -> tuple[np.ndarray, np.ndarray]:
    predictions = np.array([0.9, 0.75, 0.62, 0.51, 0.42, 0.2, 0.15, 0.05])
    labels = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    return predictions, labels


def test_auc_matches_the_pairwise_count() -> None:
    predictions, labels = _separable_but_imperfect()
    positives = predictions[labels == 1]
    negatives = predictions[labels == 0]
    manual = sum(
        1.0 if value > other else 0.5 if value == other else 0.0
        for value in positives
        for other in negatives
    ) / (positives.size * negatives.size)
    assert auc(predictions, labels) == pytest.approx(manual, abs=1e-12)
    assert 0.0 < auc(predictions, labels) < 1.0


def test_auc_needs_both_classes() -> None:
    with pytest.raises(ValueError):
        auc(np.array([0.1, 0.2]), np.array([1, 1]))


def test_auc_variance_is_small_for_a_clean_signal() -> None:
    predictions, labels = _separable_but_imperfect()
    assert auc_variance(predictions, labels) > 0.0


def test_incremental_auc_reports_the_direction_of_the_gain() -> None:
    predictions, labels = _separable_but_imperfect()
    baseline = predictions - 0.6 * (labels - 0.5)
    gain = incremental_auc(baseline, predictions, labels)
    assert gain["increment"] > 0.0
    assert gain["standard_error"] >= 0.0
    assert gain["augmented_auc"] == pytest.approx(gain["baseline_auc"] + gain["increment"])


def test_incremental_auc_is_null_for_a_rank_preserving_shift() -> None:
    predictions, labels = _separable_but_imperfect()
    gain = incremental_auc(predictions - 0.3, predictions, labels)
    assert gain["increment"] == pytest.approx(0.0, abs=1e-12)


def test_net_benefit_of_a_perfect_test_at_a_low_threshold() -> None:
    labels = np.array([1, 1, 0, 0, 1])
    predictions = labels.astype(float)
    benefit = net_benefit(predictions, labels, 0.5)
    assert benefit == pytest.approx(0.6 - 0.0)


def test_net_benefit_rejects_an_out_of_range_threshold() -> None:
    with pytest.raises(ValueError):
        net_benefit(np.array([0.5]), np.array([1]), 1.5)


def test_nri_is_zero_for_an_unchanged_score() -> None:
    predictions, labels = _separable_but_imperfect()
    assert continuous_nri(predictions, predictions, labels) == 0.0


def test_nri_is_positive_when_cases_move_up_and_controls_down() -> None:
    labels = np.array([1, 0, 1, 0])
    baseline = np.array([0.4, 0.6, 0.4, 0.6])
    augmented = np.array([0.7, 0.3, 0.7, 0.3])
    assert continuous_nri(baseline, augmented, labels) > 0.0


def test_mcid_crossing_counts_only_reaching_records() -> None:
    increments = np.array([0.05, 0.07, 0.071, 0.09])
    assert mcid_crossing_share(increments, 0.07) == pytest.approx(0.75)


def test_weighted_pool_is_the_size_weighted_mean() -> None:
    assert weighted_pool(np.array([0.2, 0.6]), np.array([1.0, 3.0])) == pytest.approx(0.5)


def test_weighted_pool_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError):
        weighted_pool(np.array([0.2]), np.array([1.0, 2.0]))


def test_decision_curve_table_shape() -> None:
    predictions, labels = _separable_but_imperfect()
    thresholds = np.array([0.1, 0.2, 0.3])
    table = decision_curve_table(predictions, labels, thresholds)
    assert table.shape == (3,)


def test_calibration_slope_recovers_a_known_coefficient() -> None:
    rng = np.random.default_rng(4)
    latent = rng.normal(size=8000)
    slope = 1.2
    probability = 1.0 / (1.0 + np.exp(-(slope * latent)))
    labels = (rng.random(8000) < probability).astype(np.int64)
    assert calibration_slope(latent, labels) == pytest.approx(slope, abs=0.08)


def test_logistic_fit_recovers_an_intercept_shift() -> None:
    rng = np.random.default_rng(5)
    latent = rng.normal(size=6000)
    probability = 1.0 / (1.0 + np.exp(-(1.0 * latent - 0.7)))
    labels = (rng.random(6000) < probability).astype(np.int64)
    coefficients, errors = logistic_fit(latent.reshape(-1, 1), labels)
    assert coefficients[0] == pytest.approx(-0.7, abs=0.1)
    assert errors.shape == (2,)


def test_calibration_in_the_large_is_zero_for_a_centred_score() -> None:
    rng = np.random.default_rng(6)
    latent = rng.normal(size=3000)
    labels = (rng.random(3000) < 1.0 / (1.0 + np.exp(-latent))).astype(np.int64)
    assert abs(calibration_in_the_large(latent, labels)) < 0.05


def test_calibration_ratio_is_near_one_for_a_fitted_score() -> None:
    rng = np.random.default_rng(7)
    latent = rng.normal(size=3000)
    labels = (rng.random(3000) < 1.0 / (1.0 + np.exp(-latent))).astype(np.int64)
    assert 0.8 < calibration_ratio(latent, labels) < 1.25


def test_intercept_helper_agrees_with_the_fit() -> None:
    rng = np.random.default_rng(8)
    latent = rng.normal(size=2000)
    labels = (rng.random(2000) < 1.0 / (1.0 + np.exp(-latent))).astype(np.int64)
    coefficients, _ = logistic_fit(latent.reshape(-1, 1), labels)
    assert calibration_intercept(latent, labels) == pytest.approx(float(coefficients[0]), abs=1e-9)
