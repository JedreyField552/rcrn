"""Plaque composition head.

Ref: Methods, "The plaque composition encoder" (a degree of composition cp that
varies monotonically with necrotic-core damage, cap thickness and bleeding);
Table 2 Tier 3 (composition head replaced by a vulnerability classifier).

The default head emits a monotone scalar index. The named alternative emits a
vulnerability probability instead; it can shift every candidate's score but cannot
reorder candidates within a plaque, which is exactly the structural limitation the
manuscript records for the scalar-covariate substitute.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

COMPOSITION_KINDS: tuple[str, ...] = ("composition_index", "vulnerability_classifier")


@dataclass(slots=True)
class CompositionOutput:
    """Composition index and, for the alternative head, its probability form."""

    index: Tensor
    is_probability: bool


class CompositionHead(nn.Module):
    """Map plaque tokens onto the composition axis."""

    def __init__(
        self, hidden_dim: int, kind: str = "composition_index", dropout: float = 0.0
    ) -> None:
        super().__init__()
        if kind not in COMPOSITION_KINDS:
            raise ValueError(f"unknown composition head {kind}")
        self.kind = kind
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, pooled_tokens: Tensor) -> CompositionOutput:
        normalized: Tensor = self.norm(pooled_tokens)
        dropped: Tensor = self.dropout(normalized)
        logits: Tensor = self.head(dropped).squeeze(-1)
        if self.kind == "composition_index":
            return CompositionOutput(index=torch.sigmoid(logits), is_probability=False)
        return CompositionOutput(index=torch.sigmoid(logits), is_probability=True)
