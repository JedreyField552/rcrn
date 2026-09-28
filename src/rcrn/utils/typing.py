"""Structural types shared across modules."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import torch

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@runtime_checkable
class Scorable(Protocol):
    """A scorer that ranks a candidate list against one plaque record."""

    def score(self, molecule_features: torch.Tensor, plaque_features: torch.Tensor) -> torch.Tensor:
        """Return one score per candidate molecule."""
        ...


@runtime_checkable
class RankingEstimator(Protocol):
    """A statistic that consumes two aligned orderings and returns a rank agreement."""

    def __call__(self, left: Sequence[float], right: Sequence[float]) -> float:
        """Return the agreement between two score vectors."""
        ...


@runtime_checkable
class CovarianceSource(Protocol):
    """Read-only view of the interaction covariance and its noise terms."""

    @property
    def covariance(self) -> np.ndarray:
        """Route-space interaction covariance."""
        ...

    @property
    def noise_variance(self) -> np.ndarray:
        """Per-record residual variance of the perturbation vector."""
        ...


class SplitLike(Protocol):
    """Anything that exposes index groups for fitting and scoring."""

    @property
    def fit_indices(self) -> Iterable[int]:
        """Index of records usable for fitting."""
        ...

    @property
    def score_indices(self) -> Iterable[int]:
        """Index of records reserved for scoring."""
        ...
