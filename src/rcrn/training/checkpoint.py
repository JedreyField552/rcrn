"""Atomic checkpointing.

Ref: Methods, "Training and inference" (the molecular branch enters the joint stage
from an initialisation rather than from a random start, so the auxiliary stage has
to survive a restart).

A checkpoint carries the seed that produced it, so a resumed run restores the same
randomness rather than a fresh one. Writes go through a temporary file and
``os.replace``; the content digest hashes the tensor payloads rather than the
container bytes, because ``torch.save`` embeds metadata that changes every write.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from rcrn.utils.io import atomic_write_binary


@dataclass(slots=True)
class CheckpointState:
    """Everything a restart needs."""

    epoch: int
    step: int
    seed: int
    deterministic: bool
    config_fingerprint: str
    best_metric: float
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None
    property_head_state: dict[str, Any] | None = None
    history: tuple[dict[str, float], ...] = ()


def save_checkpoint(path: str | Path, state: CheckpointState) -> Path:
    """Write a checkpoint atomically and return its path."""

    payload = {
        "epoch": state.epoch,
        "step": state.step,
        "seed": state.seed,
        "deterministic": state.deterministic,
        "config_fingerprint": state.config_fingerprint,
        "best_metric": state.best_metric,
        "model": state.model_state,
        "optimizer": state.optimizer_state,
        "scheduler": state.scheduler_state,
        "property_head": state.property_head_state,
        "history": list(state.history),
    }
    return atomic_write_binary(path, lambda handle: torch.save(payload, handle))


def load_checkpoint(path: str | Path) -> CheckpointState:
    """Read a checkpoint written by :func:`save_checkpoint`."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return CheckpointState(
        epoch=int(payload["epoch"]),
        step=int(payload["step"]),
        seed=int(payload["seed"]),
        deterministic=bool(payload["deterministic"]),
        config_fingerprint=str(payload["config_fingerprint"]),
        best_metric=float(payload["best_metric"]),
        model_state=payload["model"],
        optimizer_state=payload.get("optimizer"),
        scheduler_state=payload.get("scheduler"),
        property_head_state=payload.get("property_head"),
        history=tuple(payload.get("history", ())),
    )


def model_state_digest(model: nn.Module) -> str:
    """Digest of the parameter payloads, independent of container metadata."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(bytes(str(tuple(tensor.shape)), "utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def restore_into(
    model: nn.Module,
    state: CheckpointState,
) -> nn.Module:
    """Load a checkpoint into a model instance in place."""

    model.load_state_dict(state.model_state)
    return model
