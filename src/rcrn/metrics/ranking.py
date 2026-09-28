"""Rank-agreement metrics.

Ref: Table 1 (rank agreement rho and NDCG@10 over the same site-stratified fold);
Figure 1 (agreement between the plaque-blind and the plaque-conditioned ordering of
the same candidate list).

Spearman's coefficient is computed from first principles - rank transform, then the
Pearson correlation of the ranks - rather than by delegating to a library, so the
verification pass can compare it against an independent implementation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def rank_transform(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties share the mean of the ranks they occupy."""

    flat = np.asarray(values, dtype=np.float64)
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.shape, dtype=np.float64)
    position = 0
    while position < order.size:
        end = position
        while end + 1 < order.size and flat[order[end + 1]] == flat[order[position]]:
            end += 1
        average = 0.5 * (position + end) + 1.0
        ranks[order[position : end + 1]] = average
        position = end + 1
    return ranks


def spearman_rho(left: Sequence[float], right: Sequence[float]) -> float:
    """Spearman rank correlation of two aligned score vectors."""

    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("score vectors must be aligned")
    if first.size < 2:
        return 0.0
    ranked_first = rank_transform(first)
    ranked_second = rank_transform(second)
    centred_first = ranked_first - ranked_first.mean()
    centred_second = ranked_second - ranked_second.mean()
    denominator = float(np.sqrt(np.sum(centred_first**2) * np.sum(centred_second**2)))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(centred_first * centred_second) / denominator)


def kendall_tau(left: Sequence[float], right: Sequence[float]) -> float:
    """Kendall tau-a over the concordant, discordant and tied pair counts."""

    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("score vectors must be aligned")
    count = first.size
    if count < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(count):
        for j in range(i + 1, count):
            product = (first[i] - first[j]) * (second[i] - second[j])
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total = count * (count - 1) / 2
    if total == 0:
        return 0.0
    return (concordant - discordant) / total


def normalised_rank_agreement(left: Sequence[float], right: Sequence[float]) -> float:
    """Agreement between two orderings on a scale where a full reversal is zero.

    The observed deviation is divided by the deviation a complete reversal would
    produce, so one means the orderings are identical and zero means every element
    sits at the opposite end of the list.
    """

    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("score vectors must be aligned")
    count = first.size
    if count < 2:
        return 1.0
    ranked_first = rank_transform(first)
    ranked_second = rank_transform(second)
    observed = float(np.sum(np.abs(ranked_first - ranked_second)))
    reversed_rank = ranked_first[::-1]
    extreme = float(np.sum(np.abs(ranked_first - reversed_rank)))
    if extreme == 0.0:
        return 1.0
    return float(max(0.0, 1.0 - observed / extreme))


def discordant_pairs(left: Sequence[float], right: Sequence[float]) -> int:
    """Number of pairs the two orderings place the other way round."""

    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("score vectors must be aligned")
    count = first.size
    discordant = 0
    for i in range(count):
        for j in range(i + 1, count):
            if (first[i] - first[j]) * (second[i] - second[j]) < 0:
                discordant += 1
    return discordant


def ndcg_at_k(scores: Sequence[float], relevance: Sequence[float], depth: int) -> float:
    """Graded NDCG at the given depth, with exponential gain and log discount."""

    if depth <= 0:
        raise ValueError("depth must be positive")
    score_array = np.asarray(scores, dtype=np.float64)
    relevance_array = np.asarray(relevance, dtype=np.float64)
    if score_array.shape != relevance_array.shape:
        raise ValueError("scores and relevance must be aligned")
    ranking = np.argsort(-score_array, kind="stable")[:depth]
    gains = np.power(2.0, relevance_array[ranking]) - 1.0
    discounts = 1.0 / np.log2(np.arange(2, ranking.size + 2, dtype=np.float64))
    achieved = float(np.sum(gains * discounts))
    ideal_ranking = np.argsort(-relevance_array, kind="stable")[:depth]
    ideal_gains = np.power(2.0, relevance_array[ideal_ranking]) - 1.0
    ideal = float(np.sum(ideal_gains * discounts))
    if ideal == 0.0:
        return 0.0
    return achieved / ideal


def top_k_overlap(first: Sequence[int], second: Sequence[int]) -> float:
    """Jaccard overlap of two top-k sets given as index sequences."""

    left = set(int(value) for value in first)
    right = set(int(value) for value in second)
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def relevance_from_order(observed_order: np.ndarray) -> np.ndarray:
    """Turn a best-to-worst ordering into graded relevance labels."""

    order = np.atleast_2d(np.asarray(observed_order))
    count = order.shape[-1]
    grades = np.zeros(order.shape, dtype=np.float64)
    values = np.linspace(count - 1, 0, count)
    for plaque in range(order.shape[0]):
        grades[plaque, order[plaque]] = values
    return grades
