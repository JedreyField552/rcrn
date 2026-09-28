"""Auxiliary permeability head used before clinical supervision begins.

Ref: Methods, "The regime-route molecular encoder" (the branch is supervised on the
permeability-class targets {(gi, mi)} and that supervision regulates the route
coordinates); Algorithm 2 step 1.

The head exists only for the auxiliary stage. It supplies the direct effect and the
coupling state an initialisation the clinical stage then refines.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PropertyHead(nn.Module):
    """Predict the auxiliary permeability target from the coupling state."""

    def __init__(self, d_cap: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_cap)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_cap, 1)

    def forward(self, graph_state: Tensor) -> Tensor:
        normalized: Tensor = self.norm(graph_state)
        dropped: Tensor = self.dropout(normalized)
        prediction: Tensor = self.head(dropped).squeeze(-1)
        return prediction


def auxiliary_targets_available(targets: Tensor) -> Tensor:
    """Mask of molecules that actually carry an auxiliary label."""

    return torch.isfinite(targets)
