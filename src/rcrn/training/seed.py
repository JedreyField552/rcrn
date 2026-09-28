"""Deterministic seeding.

Ref: Results, Section 2.1 (each experiment is repeated with fixed seeds and the
reported values are averages across them); Methods, "Cohort construction and
auxiliary data".
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class SeedState:
    """The seed that was installed and the deterministic flag that accompanied it."""

    seed: int
    deterministic: bool

    def as_record(self) -> dict[str, int | bool]:
        return {"seed": self.seed, "deterministic": self.deterministic}


def set_seed(seed: int, deterministic: bool = True) -> SeedState:
    """Install one seed across the interpreter, NumPy and torch."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return SeedState(seed=seed, deterministic=deterministic)


def seed_sequence(base: int, count: int) -> tuple[int, ...]:
    """The reproducible ladder of seeds a multi-seed experiment uses."""

    if count < 1:
        raise ValueError("at least one seed is required")
    generator = np.random.default_rng(base)
    return tuple(int(value) for value in generator.integers(0, 2**31 - 1, size=count))
