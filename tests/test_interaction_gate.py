"""Interaction covariance, the gate statistic and threshold calibration."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rcrn.models.gate import (
    DeclineToConditionGate,
    calibrate_threshold,
    false_conditioning_rate,
)
from rcrn.models.interaction import (
    InteractionEstimator,
    InteractionSurfaces,
    interaction_statistic,
    perturbation_noise,
    shrinkage_covariance,
)


def test_shrinkage_interpolates_towards_a_scaled_identity() -> None:
    torch.manual_seed(0)
    susceptibility = torch.randn(20, 3, dtype=torch.float64)
    full = shrinkage_covariance(susceptibility, 0.0)
    shrunk = shrinkage_covariance(susceptibility, 1.0)
    trace = float(torch.trace(full)) / 3.0
    assert torch.allclose(shrunk, torch.eye(3, dtype=torch.float64) * trace, atol=1e-9)
    assert not torch.allclose(full, shrunk)


def test_shrinkage_of_a_single_sample_is_the_identity() -> None:
    covariance = shrinkage_covariance(torch.randn(1, 3, dtype=torch.float64), 0.2)
    assert torch.allclose(covariance, torch.eye(3, dtype=torch.float64) * 0.2)


def test_statistic_is_a_ratio_in_the_unit_interval() -> None:
    signal = torch.tensor([0.0, 1.0, 5.0])
    noise = torch.tensor([1.0, 1.0, 1.0])
    prior = torch.tensor([0.5, 0.5, 0.5])
    statistic = interaction_statistic(signal, noise, prior)
    assert bool(((statistic >= 0.0) & (statistic < 1.0)).all())
    assert statistic[2] > statistic[1] > statistic[0]


def test_noise_scales_inversely_with_the_plaque_volume() -> None:
    perturbation = torch.randn(4, 3)
    volumes = torch.tensor([10.0, 20.0, 40.0, 80.0])
    noise = perturbation_noise(perturbation, volumes, 20.0)
    assert bool((noise[:-1] > noise[1:]).all())


def test_estimator_surfaces_are_consistent() -> None:
    torch.manual_seed(1)
    susceptibility = torch.randn(15, 3, dtype=torch.float64)
    perturbation = torch.randn(6, 3, dtype=torch.float64)
    volumes = torch.full((6,), 30.0, dtype=torch.float64)
    estimator = InteractionEstimator()
    estimator.fit(susceptibility)
    surfaces = estimator.evaluate(perturbation, volumes, 30.0)
    manual = torch.einsum("pr,rs,ps->p", perturbation, surfaces.covariance, perturbation)
    assert torch.allclose(surfaces.signal, manual, atol=1e-9)
    assert torch.allclose(
        surfaces.statistic,
        interaction_statistic(surfaces.signal, surfaces.noise_variance, surfaces.prior),
        atol=1e-9,
    )


def test_false_conditioning_rate_counts_only_no_signal_records() -> None:
    statistic = np.array([0.9, 0.8, 0.1, 0.05])
    no_signal = np.array([False, False, True, True])
    assert false_conditioning_rate(statistic, no_signal, 0.5) == 0.0
    assert false_conditioning_rate(statistic, no_signal, 0.0) == 1.0
    assert false_conditioning_rate(statistic, np.zeros(4, dtype=bool), 0.5) == 0.0


def _surfaces(statistic: np.ndarray) -> InteractionSurfaces:
    values = torch.tensor(statistic, dtype=torch.float64)
    zeros = torch.zeros_like(values)
    return InteractionSurfaces(
        covariance=torch.eye(3, dtype=torch.float64),
        noise_variance=zeros,
        prior=zeros,
        signal=values,
        statistic=values,
    )


def test_calibration_respects_the_tolerance() -> None:
    statistic = np.linspace(0.0, 1.0, 50)
    reference = np.concatenate([np.zeros(10), np.ones(40)])
    calibration = calibrate_threshold(_surfaces(statistic), reference, tolerance=0.05)
    assert calibration.false_conditioning_rate <= 0.05
    assert calibration.threshold > 0.0


def test_gate_conditions_fewer_records_as_tolerance_tightens() -> None:
    statistic = np.linspace(0.0, 1.0, 60)
    surfaces = _surfaces(statistic)
    loose = DeclineToConditionGate()
    loose.calibrate(surfaces, None, 1.0)
    tight = DeclineToConditionGate()
    tight.calibrate(surfaces, None, 0.0)
    assert loose.threshold <= tight.threshold


def test_uncalibrated_gate_refuses_to_decide() -> None:
    with pytest.raises(RuntimeError):
        DeclineToConditionGate().decide(_surfaces(np.zeros(3)))
