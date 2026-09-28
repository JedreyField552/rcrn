"""Auxiliary property-regression term.

Ref: Eq. (4) (Lprop acts as an auxiliary regression over molecular properties);
Methods, "The regime-route molecular encoder".
"""

from __future__ import annotations

import torch
from torch import Tensor

from rcrn.models.property_head import auxiliary_targets_available


def property_prior_loss(
    prediction: Tensor,
    target: Tensor,
    beta: float = 0.25,
) -> Tensor:
    """Smooth L1 over the molecules that carry an auxiliary label.

    Molecules without an auxiliary label contribute nothing rather than a zero
    target, which would pull the branch towards the mean of the labelled set.
    """

    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    mask = auxiliary_targets_available(target)
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    selected_prediction = prediction[mask]
    selected_target = target[mask]
    return torch.nn.functional.smooth_l1_loss(selected_prediction, selected_target, beta=beta)
