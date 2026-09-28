"""Exponential moving average of model weights.

Ref: Methods, "Training and inference" (the manuscript does not print an EMA decay;
the value is an engineering default exposed in configuration).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class EMAState:
    """How many updates the average has absorbed."""

    updates: int
    decay: float

    def as_record(self) -> dict[str, float]:
        return {"updates": float(self.updates), "decay": self.decay}


class ExponentialMovingAverage:
    """Track a shadow copy of the parameters and buffers of a model."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must lie in (0, 1)")
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the live parameters into the shadow copy."""

        self.updates += 1
        correction = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        live = dict(model.named_parameters())
        for name, parameter in self.shadow.named_parameters():
            parameter.mul_(correction).add_(live[name].detach(), alpha=1.0 - correction)
        live_buffers = dict(model.named_buffers())
        for name, buffer in self.shadow.named_buffers():
            source = live_buffers.get(name)
            if source is None:
                continue
            if buffer.dtype.is_floating_point:
                buffer.mul_(correction).add_(source.detach(), alpha=1.0 - correction)
            else:
                buffer.copy_(source.detach())

    def copy_to(self, model: nn.Module) -> nn.Module:
        """Copy the averaged weights into another model instance."""

        model.load_state_dict(self.shadow.state_dict())
        return model

    def state(self) -> EMAState:
        return EMAState(updates=self.updates, decay=self.decay)
