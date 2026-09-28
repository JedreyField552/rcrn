"""Distributed readiness.

Ref: Methods, "Problem formulation" (an insignificant incremental inference cost per
plaque in front of the imaging apparatus); Methods, "Training and inference".
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(slots=True)
class ProcessGroupInfo:
    """Rank layout of the current process."""

    rank: int
    world_size: int
    local_rank: int
    distributed: bool

    def is_primary(self) -> bool:
        return self.rank == 0


def initialise_process_group() -> ProcessGroupInfo:
    """Join the process group when the launcher set one up, otherwise stay local."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return ProcessGroupInfo(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        distributed=world_size > 1,
    )


def wrap_distributed(model: nn.Module, info: ProcessGroupInfo) -> nn.Module:
    """Wrap a model for data-parallel training when a process group exists."""

    if not info.distributed:
        return model
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda", info.local_rank))
    return DistributedDataParallel(
        model, device_ids=[info.local_rank] if torch.cuda.is_available() else None
    )


def unwrap(model: nn.Module) -> nn.Module:
    """Recover the underlying module from a distributed wrapper."""

    return model.module if isinstance(model, DistributedDataParallel) else model


def barrier(info: ProcessGroupInfo) -> None:
    """Synchronise ranks when a process group is active."""

    if info.distributed and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: float, info: ProcessGroupInfo, device: torch.device) -> float:
    """Average a scalar across ranks."""

    if not info.distributed or not dist.is_initialized():
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / info.world_size)


def effective_batch_size(batch_size: int, grad_accum: int, world_size: int) -> int:
    """The batch the optimiser actually sees, as the protocol defines it."""

    if min(batch_size, grad_accum, world_size) < 1:
        raise ValueError("batch, accumulation and world size must be positive")
    return batch_size * grad_accum * world_size


def destroy_process_group(info: ProcessGroupInfo) -> None:
    """Tear the process group down, if one was created."""

    if info.distributed and dist.is_initialized():
        dist.destroy_process_group()
