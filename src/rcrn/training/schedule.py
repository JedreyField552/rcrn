"""Learning-rate schedules.

Ref: Methods, "Training and inference" (the manuscript does not print a schedule,
so the warmup fraction and the floor are engineering defaults exposed in config).

The warmup-plus-cosine factor is written as a pure function of the step so it can
be recomputed by hand in the verification pass, and the LambdaLR wrapper only
applies that function.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

SCHEDULES: tuple[str, ...] = ("cosine", "step", "constant")


def warmup_cosine_factor(
    step: int, total_steps: int, warmup_steps: int, min_factor: float = 0.0
) -> float:
    """Linear warmup followed by a cosine decay to the floor."""

    if total_steps <= 0:
        raise ValueError("total step count must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup steps cannot be negative")
    if step < warmup_steps:
        if warmup_steps == 0:
            return 1.0
        return max(min_factor, (step + 1) / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def warmup_step_factor(step: int, warmup_steps: int, decay_every: int, decay_gamma: float) -> float:
    """Linear warmup followed by a geometric decay every fixed number of steps."""

    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    exponent = (step - warmup_steps) // max(1, decay_every)
    return float(decay_gamma**exponent)


def build_schedule(
    optimizer: Optimizer,
    name: str,
    total_steps: int,
    warmup_fraction: float,
    min_factor: float = 0.0,
    decay_every: int = 10,
    decay_gamma: float = 0.5,
) -> LambdaLR:
    """Wrap the chosen factor in a LambdaLR, which owns ``get_lr``."""

    if name not in SCHEDULES:
        raise ValueError(f"unknown schedule {name}")
    warmup_steps = int(round(total_steps * warmup_fraction))

    def factor(step: int) -> float:
        if name == "cosine":
            return warmup_cosine_factor(step, total_steps, warmup_steps, min_factor)
        if name == "step":
            return warmup_step_factor(step, warmup_steps, decay_every, decay_gamma)
        return 1.0

    return LambdaLR(optimizer, lr_lambda=factor)


def learning_rates(optimizer: Optimizer) -> tuple[float, ...]:
    """Current learning rate of every parameter group."""

    return tuple(float(group["lr"]) for group in optimizer.param_groups)
