"""Gated pair scoring and per-plaque candidate orderings.

Ref: Eq. (1) (the gate chooses between the reaction-norm contributor and the
plaque-blind direct effect); Algorithm 1 steps 1-2 (score every candidate, sort
each plaque's list); Algorithm 2 step 6.

Deferred records fall back to the plaque-blind order of the same candidate list.
The fallback is recorded per record so the deferral share is an auditable cost
rather than a silent substitution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rcrn.models.coupling import CouplingOutput
from rcrn.models.gate import DeclineToConditionGate, GateDecision
from rcrn.models.interaction import InteractionSurfaces


@dataclass(slots=True)
class PlaqueOrdering:
    """One plaque's scored candidate list and the decision that produced it."""

    record_id: str
    scores: np.ndarray
    order: tuple[int, ...]
    ordering: tuple[str, ...]
    conditioned: bool
    statistic: float
    route_share: np.ndarray | None = None

    def top_k(self, depth: int) -> tuple[int, ...]:
        return self.order[:depth]

    def has_ties(self, tolerance: float = 1e-9) -> bool:
        return bool(np.ptp(self.scores) <= tolerance)


def _descending_order(scores: np.ndarray) -> tuple[int, ...]:
    return tuple(int(index) for index in np.argsort(-scores, kind="stable"))


class GatedScorer:
    """Turn a coupling output into one ordering per plaque record."""

    def __init__(
        self,
        candidate_ids: tuple[str, ...],
        gate: DeclineToConditionGate | None = None,
        blind_only: bool = False,
    ) -> None:
        self.candidate_ids = candidate_ids
        self.gate = gate
        self.blind_only = blind_only

    def order(
        self,
        record_ids: tuple[str, ...],
        coupling: CouplingOutput,
        surfaces: InteractionSurfaces | None = None,
        decision: GateDecision | None = None,
    ) -> tuple[PlaqueOrdering, ...]:
        conditioned_scores = coupling.pair_score.detach().cpu().numpy()
        plaque_blind = self.blind_scores(coupling).cpu().numpy()
        share = _route_share_numpy(coupling.route_contribution)
        orderings: list[PlaqueOrdering] = []
        statistics = (
            surfaces.statistic.detach().cpu().numpy()
            if surfaces is not None
            else np.zeros(len(record_ids), dtype=np.float64)
        )
        for position, record_id in enumerate(record_ids):
            if decision is None:
                conditioned = not self.blind_only
            else:
                conditioned = bool(decision.conditioned[position]) and not self.blind_only
            row = np.asarray(
                conditioned_scores[position] if conditioned else plaque_blind[position],
                dtype=np.float64,
            )
            order = _descending_order(row)
            orderings.append(
                PlaqueOrdering(
                    record_id=record_id,
                    scores=row,
                    order=order,
                    ordering=tuple(self.candidate_ids[index] for index in order),
                    conditioned=conditioned,
                    statistic=float(statistics[position]),
                    route_share=share[position] if conditioned else None,
                )
            )
        return tuple(orderings)

    def blind_scores(self, coupling: CouplingOutput) -> torch.Tensor:
        """The plaque-blind direct effect, repeated over the plaque axis."""

        direct = coupling.direct_effect.unsqueeze(0)
        return direct.expand(coupling.pair_score.shape[0], -1).contiguous()


def _route_share_numpy(route_contribution: torch.Tensor) -> np.ndarray:
    magnitude = np.abs(route_contribution.detach().cpu().numpy())
    total = magnitude.sum(axis=-1, keepdims=True)
    return magnitude / (total + 1e-9)


def degenerate_orderings(orderings: tuple[PlaqueOrdering, ...]) -> int:
    """Count the records whose candidate list carries no ordering information."""

    return sum(1 for ordering in orderings if ordering.has_ties())


def deferral_share(orderings: tuple[PlaqueOrdering, ...]) -> float:
    """Share of records that fell back to the plaque-blind order."""

    if not orderings:
        return 0.0
    return float(np.mean([not ordering.conditioned for ordering in orderings]))
