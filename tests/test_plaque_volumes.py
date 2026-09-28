"""Multi-contrast volume assembly and acquisition perturbation."""

from __future__ import annotations

import numpy as np
import pytest

from rcrn.data.plaque_volumes import (
    VolumeBank,
    apply_motion_degradation,
    assemble_volume,
    downsample_anisotropic,
    mask_missing_channels,
    resample_slice_thickness,
    volume_scale_factor,
)
from rcrn.data.schemas import ACQUISITION_CHANNELS, AcquisitionMeta


def test_assembly_places_channels_in_the_declared_order() -> None:
    first = np.full((2, 3, 3), 1.0, dtype=np.float32)
    second = np.full((2, 3, 3), 2.0, dtype=np.float32)
    volume = assemble_volume({"t1w": second, "tof": first})
    assert volume.array.shape[0] == len(ACQUISITION_CHANNELS)
    assert np.all(volume.array[0] == 1.0)
    assert np.all(volume.array[1] == 2.0)
    assert np.all(volume.array[2] == 0.0)


def test_missing_channel_is_a_zero_block() -> None:
    block = np.ones((4, 2, 2, 2), dtype=np.float32)
    masked = mask_missing_channels(block, ("tof", "t2w", "pd"))
    assert np.all(masked[1] == 0.0)
    assert np.all(masked[0] == 1.0)


def test_resampling_preserves_the_value_range() -> None:
    volume = np.linspace(0.0, 1.0, 4 * 4 * 2 * 2, dtype=np.float32).reshape(4, 4, 2, 2)
    thicker = resample_slice_thickness(volume, 1.0, 3.0)
    assert thicker.shape[1] < volume.shape[1]
    assert thicker.min() >= volume.min() - 1e-6
    assert thicker.max() <= volume.max() + 1e-6


def test_motion_degrades_by_a_bounded_amount() -> None:
    rng = np.random.default_rng(0)
    volume = np.abs(rng.normal(size=(2, 6, 8, 8))).astype(np.float32)
    clean = apply_motion_degradation(volume, 0.0, rng)
    assert np.allclose(clean, volume)
    degraded = apply_motion_degradation(volume, 0.4, rng)
    relative = np.linalg.norm((degraded - volume).reshape(-1)) / np.linalg.norm(volume.reshape(-1))
    assert 0.0 < relative < 5.0


def test_downsampling_keeps_the_depth() -> None:
    volume = np.arange(2 * 8 * 4 * 4, dtype=np.float32).reshape(2, 8, 4, 4)
    decimated = downsample_anisotropic(volume, 4)
    assert decimated.shape == volume.shape
    assert not np.allclose(decimated, volume)


def test_volume_scale_factor_grows_with_size() -> None:
    small = volume_scale_factor(20.0, 160.0)
    large = volume_scale_factor(640.0, 160.0)
    assert small < 1.0 < large


def test_volume_bank_reports_channel_statistics() -> None:
    volumes = [np.ones((2, 3, 3), dtype=np.float32), np.full((2, 3, 3), 3.0, dtype=np.float32)]
    bank = VolumeBank(volumes, ("A", "B"))
    means, deviations = bank.channel_statistics()
    assert means.shape == (2,)
    assert deviations[0] == pytest.approx(1.0, abs=1e-4)
    assert bank.index_of("B") == 1
    assert bank.stacked().shape == (2, 2, 3, 3)


def test_meta_declares_absent_channel() -> None:
    meta = AcquisitionMeta(channels=("tof", "t2w"))
    volume = assemble_volume({"tof": np.ones((1, 2, 2), dtype=np.float32)}, meta)
    assert volume.is_complete() is False
    assert np.all(volume.array[1] == 0.0)
