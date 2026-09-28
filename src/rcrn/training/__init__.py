"""Training utilities: seeding, optimisation, schedules, precision, checkpointing."""

from __future__ import annotations

from rcrn.training.amp import (
    PRECISIONS,
    autocast_context,
    build_grad_scaler,
    parameters_dtype,
    support_matrix,
)
from rcrn.training.checkpoint import (
    CheckpointState,
    load_checkpoint,
    model_state_digest,
    restore_into,
    save_checkpoint,
)
from rcrn.training.distributed import (
    ProcessGroupInfo,
    all_reduce_mean,
    barrier,
    destroy_process_group,
    effective_batch_size,
    initialise_process_group,
    unwrap,
    wrap_distributed,
)
from rcrn.training.ema import EMAState, ExponentialMovingAverage
from rcrn.training.optim import (
    OPTIMIZERS,
    build_optimizer,
    parameter_digest,
    parameter_groups,
    trainable_parameter_count,
)
from rcrn.training.schedule import (
    SCHEDULES,
    build_schedule,
    learning_rates,
    warmup_cosine_factor,
    warmup_step_factor,
)
from rcrn.training.seed import SeedState, seed_sequence, set_seed
from rcrn.training.trainer import (
    CohortView,
    GateFitting,
    Trainer,
    TrainingResult,
    build_view,
)

__all__ = [
    "OPTIMIZERS",
    "PRECISIONS",
    "SCHEDULES",
    "CheckpointState",
    "CohortView",
    "EMAState",
    "ExponentialMovingAverage",
    "GateFitting",
    "ProcessGroupInfo",
    "SeedState",
    "Trainer",
    "TrainingResult",
    "all_reduce_mean",
    "autocast_context",
    "barrier",
    "build_grad_scaler",
    "build_optimizer",
    "build_schedule",
    "build_view",
    "destroy_process_group",
    "effective_batch_size",
    "initialise_process_group",
    "learning_rates",
    "load_checkpoint",
    "model_state_digest",
    "parameter_digest",
    "parameter_groups",
    "parameters_dtype",
    "restore_into",
    "save_checkpoint",
    "seed_sequence",
    "set_seed",
    "support_matrix",
    "trainable_parameter_count",
    "unwrap",
    "warmup_cosine_factor",
    "warmup_step_factor",
    "wrap_distributed",
]
