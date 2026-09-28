"""Optimiser construction.

Ref: Methods, "Training and inference"; Algorithm 2 step 5(c) (stochastic gradient
descent updates on the composite objective).
"""

from __future__ import annotations

import torch
from torch import nn

OPTIMIZERS: tuple[str, ...] = ("adamw", "adam", "sgd")


def parameter_groups(
    model: nn.Module, weight_decay: float, no_decay_terms: tuple[str, ...] = ("bias", "norm")
) -> list[dict[str, object]]:
    """Split parameters into a decayed and an undecayed group."""

    decayed: list[nn.Parameter] = []
    undecayed: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or any(term in name for term in no_decay_terms):
            undecayed.append(parameter)
        else:
            decayed.append(parameter)
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": undecayed, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: nn.Module,
    name: str,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Construct one of the declared optimisers over grouped parameters."""

    if name not in OPTIMIZERS:
        raise ValueError(f"unknown optimiser {name}")
    groups = parameter_groups(model, weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr, betas=betas, eps=eps)
    return torch.optim.SGD(groups, lr=lr, momentum=momentum)


def trainable_parameter_count(model: nn.Module) -> int:
    """Number of parameters the optimiser will actually update."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parameter_digest(model: nn.Module) -> dict[str, float]:
    """Cheap summary of parameter statistics, used to confirm an update happened."""

    with torch.no_grad():
        values = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
    return {
        "count": float(values.numel()),
        "mean": float(values.mean()),
        "l2": float(torch.linalg.vector_norm(values)),
    }
