"""Rank-agreement metrics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from rcrn.metrics.ranking import (
    discordant_pairs,
    kendall_tau,
    ndcg_at_k,
    normalised_rank_agreement,
    rank_transform,
    relevance_from_order,
    spearman_rho,
    top_k_overlap,
)


def test_rank_transform_averages_ties() -> None:
    ranks = rank_transform(np.array([10.0, 20.0, 20.0, 30.0]))
    assert ranks.tolist() == [1.0, 2.5, 2.5, 4.0]


def test_spearman_matches_an_independent_library() -> None:
    rng = np.random.default_rng(0)
    left = rng.normal(size=30)
    right = rng.normal(size=30)
    assert spearman_rho(left, right) == pytest.approx(
        float(stats.spearmanr(left, right).statistic), abs=1e-12
    )


def test_spearman_is_one_for_a_monotone_transform() -> None:
    left = np.arange(1.0, 11.0)
    right = np.exp(left)
    assert spearman_rho(left, right) == pytest.approx(1.0, abs=1e-12)


def test_spearman_is_minus_one_when_reversed() -> None:
    left = np.arange(1.0, 11.0)
    assert spearman_rho(left, left[::-1]) == pytest.approx(-1.0, abs=1e-12)


def test_kendall_matches_the_brute_force_definition() -> None:
    left = np.array([1.0, 2.0, 3.0, 4.0])
    right = np.array([1.0, 3.0, 2.0, 4.0])
    assert kendall_tau(left, right) == pytest.approx(2 / 3, abs=1e-12)


def test_discordant_pair_count() -> None:
    assert discordant_pairs(np.array([1.0, 2.0]), np.array([2.0, 1.0])) == 1


def test_normalised_rank_agreement_bounds() -> None:
    order = np.arange(5.0)
    assert normalised_rank_agreement(order, order) == pytest.approx(1.0)
    assert normalised_rank_agreement(order, order[::-1]) == pytest.approx(0.0)


def test_ndcg_matches_the_gain_and_discount_definition() -> None:
    scores = np.array([0.2, 0.9, 0.4, 0.7])
    relevance = np.array([0.0, 3.0, 1.0, 2.0])
    ideal_order = np.argsort(-relevance)[:2]
    achieved_order = np.argsort(-scores)[:2]
    discounts = 1.0 / np.log2(np.arange(2, 4))
    achieved = float(np.sum((2.0 ** relevance[achieved_order] - 1.0) * discounts))
    ideal = float(np.sum((2.0 ** relevance[ideal_order] - 1.0) * discounts))
    assert ndcg_at_k(scores, relevance, 2) == pytest.approx(achieved / ideal, abs=1e-12)


def test_ndcg_is_zero_when_no_relevance_exists() -> None:
    assert ndcg_at_k(np.array([1.0, 2.0]), np.zeros(2), 1) == 0.0


def test_ndcg_rejects_a_non_positive_depth() -> None:
    with pytest.raises(ValueError):
        ndcg_at_k(np.array([1.0]), np.array([1.0]), 0)


def test_top_k_overlap_is_a_jaccard_index() -> None:
    assert top_k_overlap((0, 1, 2), (1, 2, 3)) == pytest.approx(2 / 4)


def test_relevance_from_order_grades_the_head_highest() -> None:
    order = np.array([[2, 0, 1]])
    grades = relevance_from_order(order)
    assert grades[0].tolist() == [1.0, 0.0, 2.0]


def test_misaligned_vectors_are_rejected() -> None:
    with pytest.raises(ValueError):
        spearman_rho(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError):
        kendall_tau(np.zeros(3), np.zeros(4))
