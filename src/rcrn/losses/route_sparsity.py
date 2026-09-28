"""Route-coordinate sparsity term.

Ref: Eq. (4) (Lreg handles the sparsity contribution when the auxiliary endpoints
cannot separate the trajectory coordinates); Methods, "Complexity justification".
"""

from __future__ import annotations

import torch
from torch import Tensor


def group_sparsity_loss(route_matrix: Tensor, groups: int) -> Tensor:
    """Group-lasso over the route columns of a molecular field.

    Each group is one route, so the penalty shrinks a whole route direction towards
    zero instead of thinning individual molecules' coordinates.
    """

    if route_matrix.dim() != 2:
        raise ValueError("expected a (molecule, route) tensor")
    if groups != route_matrix.shape[-1]:
        raise ValueError("group count must equal the route count")
    column_norms = torch.linalg.norm(route_matrix, dim=0)
    return column_norms.sum()


def coordinate_sparsity_loss(route_matrix: Tensor) -> Tensor:
    """Elementwise L1 alternative, kept for the sparsity sweep."""

    return torch.abs(route_matrix).sum(dim=-1).mean()
