"""Acquisition perturbation families and the admissible radius."""

from __future__ import annotations

import numpy as np
import pytest

from rcrn.data.augmentation import (
    DEGRADATION_FAMILIES,
    AcquisitionPerturber,
    admissible_radius,
    sweep_magnitudes,
)
from rcrn.data.schemas import ACQUISITION_CHANNELS, AcquisitionMeta


def _volume() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.abs(rng.normal(size=(4, 8, 12, 12))).astype(np.float32)


def test_every_family_produces_a_bounded_report() -> None:
    rng = np.random.default_rng(1)
    perturber = AcquisitionPerturber()
    volume = _volume()
    meta = AcquisitionMeta()
    for family in DEGRADATION_FAMILIES:
        perturbed, report = perturber.apply(volume, meta, family, 0.4, rng)
        assert report.family == family
        assert report.relative_l2 >= 0.0
        assert perturbed.shape[0] == volume.shape[0]


def test_channel_dropout_removes_a_channel() -> None:
    perturber = AcquisitionPerturber()
    perturbed, _ = perturber.apply(
        _volume(), AcquisitionMeta(), "contrast_channel_dropout", 1.0, np.random.default_rng(2)
    )
    assert np.any(np.all(perturbed == 0.0, axis=(1, 2, 3)))


def test_gaussian_family_is_bounded_by_the_declared_magnitude() -> None:
    perturber = AcquisitionPerturber(("gaussian_intensity_noise",))
    volume = _volume()
    _, report = perturber.apply(
        volume, AcquisitionMeta(), "gaussian_intensity_noise", 0.1, np.random.default_rng(3)
    )
    assert 0.0 < report.relative_l2 < 1.0


def test_admissible_radius_is_positive_and_deterministic() -> None:
    perturber = AcquisitionPerturber()
    volume = _volume()
    first = admissible_radius(perturber, volume, AcquisitionMeta(), 5)
    second = admissible_radius(perturber, volume, AcquisitionMeta(), 5)
    assert first > 0.0
    assert first == second


def test_unknown_family_is_rejected() -> None:
    with pytest.raises(ValueError):
        AcquisitionPerturber(("not_a_family",))
    with pytest.raises(ValueError):
        AcquisitionPerturber().apply(
            _volume(), AcquisitionMeta(), "nope", 0.1, np.random.default_rng(0)
        )


def test_magnitude_sweep_ladders_are_increasing() -> None:
    for family in DEGRADATION_FAMILIES:
        ladder = sweep_magnitudes(family)
        assert list(ladder) == sorted(ladder)
        assert all(value > 0.0 for value in ladder)


def test_slice_thickening_keeps_the_channel_count() -> None:
    perturber = AcquisitionPerturber(("slice_thickening",))
    volume = _volume()
    perturbed, report = perturber.apply(
        volume, AcquisitionMeta(), "slice_thickening", 0.5, np.random.default_rng(4)
    )
    assert perturbed.shape[0] == len(ACQUISITION_CHANNELS)
    assert report.relative_l2 > 0.0
