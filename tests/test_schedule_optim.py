"""Optimisation utilities, schedules, precision and distributed helpers."""

from __future__ import annotations

import math

import pytest
import torch
from _paths import smoke_config

from rcrn.data.molecules import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from rcrn.models.rcrn import RCRNModel
from rcrn.training.amp import autocast_context, build_grad_scaler, parameters_dtype, support_matrix
from rcrn.training.distributed import (
    ProcessGroupInfo,
    all_reduce_mean,
    effective_batch_size,
    unwrap,
    wrap_distributed,
)
from rcrn.training.optim import (
    build_optimizer,
    parameter_digest,
    parameter_groups,
    trainable_parameter_count,
)
from rcrn.training.schedule import (
    build_schedule,
    learning_rates,
    warmup_cosine_factor,
    warmup_step_factor,
)
from rcrn.training.seed import seed_sequence, set_seed


def _model() -> RCRNModel:
    return RCRNModel(smoke_config(), ATOM_FEATURE_DIM, BOND_FEATURE_DIM)


def test_optimiser_groups_split_decayed_and_undecayed_parameters() -> None:
    model = _model()
    groups = parameter_groups(model, 0.1)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.1
    assert groups[1]["weight_decay"] == 0.0
    decayed = sum(parameter.numel() for parameter in groups[0]["params"])
    undecayed = sum(parameter.numel() for parameter in groups[1]["params"])
    assert decayed + undecayed == trainable_parameter_count(model)


def test_every_declared_optimiser_builds() -> None:
    for name in ("adamw", "adam", "sgd"):
        optimizer = build_optimizer(_model(), name, lr=1e-3, weight_decay=0.0)
        assert optimizer.param_groups


def test_unknown_optimiser_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_optimizer(_model(), "nope", 1e-3, 0.0)


def test_parameter_digest_reports_the_parameter_count() -> None:
    model = _model()
    digest = parameter_digest(model)
    assert digest["count"] == float(trainable_parameter_count(model))
    assert digest["l2"] >= 0.0


def test_warmup_cosine_factor_formula() -> None:
    total, warmup, floor = 100, 10, 0.1
    for step in (0, 5, 9):
        assert warmup_cosine_factor(step, total, warmup, floor) == pytest.approx(
            max(floor, (step + 1) / warmup)
        )
    step = 55
    progress = (step - warmup) / (total - warmup)
    expected = floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    assert warmup_cosine_factor(step, total, warmup, floor) == pytest.approx(expected)


def test_warmup_cosine_factor_endpoints() -> None:
    assert warmup_cosine_factor(0, 10, 0, 0.0) == pytest.approx(1.0)
    floor, progress = 0.2, 0.9
    assert warmup_cosine_factor(9, 10, 0, floor) == pytest.approx(
        floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)), abs=1e-12
    )


def test_step_schedule_decays_geometrically() -> None:
    assert warmup_step_factor(0, 4, 2, 0.5) == pytest.approx(0.25)
    assert warmup_step_factor(4, 0, 2, 0.5) == pytest.approx(0.25)
    assert warmup_step_factor(6, 0, 2, 0.5) == pytest.approx(0.125)


def test_schedule_wrapper_drives_the_learning_rate() -> None:
    optimizer = build_optimizer(_model(), "sgd", lr=1.0, weight_decay=0.0)
    scheduler = build_schedule(optimizer, "cosine", total_steps=10, warmup_fraction=0.2)
    rates = [learning_rates(optimizer)[0]]
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        rates.append(learning_rates(optimizer)[0])
    assert rates[0] < rates[1]
    assert rates[-1] <= rates[1]


def test_unknown_schedule_is_rejected() -> None:
    optimizer = build_optimizer(_model(), "sgd", 1.0, 0.0)
    with pytest.raises(ValueError):
        build_schedule(optimizer, "nope", 10, 0.1)


def test_precision_helpers() -> None:
    device = torch.device("cpu")
    assert autocast_context("fp32", device) is not None
    assert build_grad_scaler("fp16") is not None
    assert build_grad_scaler("bf16") is None
    assert parameters_dtype(_model()) in {"torch.float32", "float32"}
    assert set(support_matrix()) == {"fp32", "bf16", "fp16"}
    with pytest.raises(ValueError):
        autocast_context("float128", device)


def test_effective_batch_size_is_the_product() -> None:
    assert effective_batch_size(8, 4, 2) == 64
    with pytest.raises(ValueError):
        effective_batch_size(8, 0, 1)


def test_distributed_helpers_are_no_ops_without_a_group() -> None:
    info = ProcessGroupInfo(rank=0, world_size=1, local_rank=0, distributed=False)
    model = _model()
    assert wrap_distributed(model, info) is model
    assert unwrap(model) is model
    assert all_reduce_mean(0.5, info, torch.device("cpu")) == 0.5


def test_seed_sequence_is_deterministic_and_distinct() -> None:
    first = seed_sequence(7, 5)
    second = seed_sequence(7, 5)
    assert first == second
    assert len(set(first)) == 5


def test_set_seed_returns_the_installed_state() -> None:
    state = set_seed(123, deterministic=True)
    assert state.as_record() == {"seed": 123, "deterministic": True}
    drawing = torch.randn(1).item()
    set_seed(123, deterministic=True)
    assert torch.randn(1).item() == drawing
