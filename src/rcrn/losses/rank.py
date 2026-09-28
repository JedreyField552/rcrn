"""Agreement-weighted listwise rank objective.

Ref: Eq. (3); Methods, "Agreement-weighted listwise rank objective".

    L_rank = sum_p  w_p / |Q_p|  sum_{(i,j) in Q_p}  ell( (S(j|p) - S(i|p)) / epsilon )

with w_p the stratum agreement coefficient, ell a convex surrogate for the
concordance indicator, and epsilon a temperature. The pairwise form absorbs the
absolute scale mismatch between the two branches, so no decision threshold is set
at fitting time.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

SURROGATES: tuple[str, ...] = ("logistic", "square_hinge")


@dataclass(slots=True)
class RankPairSet:
    """Preferred and dispreferred candidate indices for each plaque."""

    preferred: Tensor
    dispreferred: Tensor
    plaque_index: Tensor
    weights: Tensor

    def size(self) -> int:
        return int(self.preferred.numel())


def build_pairs(
    scores: Tensor,
    observed_order: Tensor,
    weights: Tensor,
    budget: int,
    generator: torch.Generator | None = None,
) -> RankPairSet:
    """Enumerate or sample preferred/dispreferred pairs from each plaque's ordering.

    ``observed_order`` holds, per plaque, the candidate indices from best to worst.
    ``budget`` caps the number of pairs per plaque so the objective cost stays
    linear in the candidate list.
    """

    if scores.dim() != 2:
        raise ValueError("expected a (plaque, candidate) score matrix")
    if observed_order.shape != scores.shape:
        raise ValueError("the observed ordering must match the score matrix")
    plaque_count, candidate_count = scores.shape
    if budget <= 0:
        empty = torch.zeros(0, dtype=torch.long, device=scores.device)
        return RankPairSet(
            preferred=empty,
            dispreferred=empty,
            plaque_index=empty,
            weights=torch.zeros(0, dtype=scores.dtype, device=scores.device),
        )
    preferred: list[int] = []
    dispreferred: list[int] = []
    plaque_index: list[int] = []
    weight_values: list[float] = []
    for plaque in range(plaque_count):
        order = observed_order[plaque].tolist()
        pairs: list[tuple[int, int]] = []
        for left in range(candidate_count):
            for right in range(left + 1, candidate_count):
                pairs.append((order[left], order[right]))
                if len(pairs) >= budget:
                    break
            if len(pairs) >= budget:
                break
        if not pairs:
            continue
        for better, worse in pairs:
            preferred.append(better)
            dispreferred.append(worse)
            plaque_index.append(plaque)
            weight_values.append(float(weights[plaque]))
    if not preferred:
        empty = torch.zeros(0, dtype=torch.long, device=scores.device)
        return RankPairSet(
            preferred=empty,
            dispreferred=empty,
            plaque_index=empty,
            weights=torch.zeros(0, dtype=scores.dtype, device=scores.device),
        )
    return RankPairSet(
        preferred=torch.tensor(preferred, dtype=torch.long, device=scores.device),
        dispreferred=torch.tensor(dispreferred, dtype=torch.long, device=scores.device),
        plaque_index=torch.tensor(plaque_index, dtype=torch.long, device=scores.device),
        weights=torch.tensor(weight_values, dtype=scores.dtype, device=scores.device),
    )


def _surrogate(values: Tensor, kind: str) -> Tensor:
    if kind == "logistic":
        return torch.nn.functional.softplus(values)
    if kind == "square_hinge":
        return 0.5 * torch.clamp(values, min=-1.0) ** 2
    raise ValueError(f"unknown surrogate {kind}")


def agreement_weighted_rank_loss(
    pair_scores: Tensor,
    rank_pairs: RankPairSet,
    temperature: float,
    surrogate: str = "logistic",
) -> Tensor:
    """Evaluate Eq. (3) on a prepared pair set."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if rank_pairs.size() == 0:
        return pair_scores.sum() * 0.0
    better = pair_scores[rank_pairs.plaque_index, rank_pairs.preferred]
    worse = pair_scores[rank_pairs.plaque_index, rank_pairs.dispreferred]
    margins = (worse - better) / temperature
    terms = _surrogate(margins, surrogate) * rank_pairs.weights
    plaque_count = pair_scores.shape[0]
    totals = torch.zeros(plaque_count, dtype=terms.dtype, device=terms.device)
    totals = totals.index_add(0, rank_pairs.plaque_index, terms)
    counts = torch.zeros(plaque_count, dtype=terms.dtype, device=terms.device)
    counts = counts.index_add(0, rank_pairs.plaque_index, torch.ones_like(terms))
    return (totals / counts.clamp(min=1.0)).mean()


def concordance_rate(pair_scores: Tensor, rank_pairs: RankPairSet) -> Tensor:
    """Unweighted share of pairs the scorer orders correctly."""

    if rank_pairs.size() == 0:
        return pair_scores.sum() * 0.0
    better = pair_scores[rank_pairs.plaque_index, rank_pairs.preferred]
    worse = pair_scores[rank_pairs.plaque_index, rank_pairs.dispreferred]
    return (better > worse).to(pair_scores.dtype).mean()
