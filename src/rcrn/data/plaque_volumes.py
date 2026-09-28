"""Multi-contrast plaque volume assembly and acquisition perturbation.

Ref: Methods, "The plaque composition encoder"; Results, Table 3 (contrast protocol
without the T1-weighted channel, slice thickness of 3 mm or greater, motion-degraded
records, plaques below 40 mm3).

Channel order is fixed so that a missing channel is a zero block rather than a
shifted tensor, which is what keeps the contrast-dropout fold a genuine ablation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rcrn.data.schemas import ACQUISITION_CHANNELS, AcquisitionMeta


@dataclass(slots=True)
class PlaqueVolume:
    """One assembled multi-channel plaque volume with its acquisition record."""

    array: np.ndarray
    meta: AcquisitionMeta

    def shape(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.array.shape)

    def is_complete(self) -> bool:
        return self.meta.channels == ACQUISITION_CHANNELS

    def to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(self.array, dtype=np.float32))


def assemble_volume(
    channel_arrays: dict[str, np.ndarray], meta: AcquisitionMeta | None = None
) -> PlaqueVolume:
    """Stack per-contrast arrays into the canonical channel order, zero-filling absences."""

    if not channel_arrays:
        raise ValueError("no contrast channels supplied")
    shapes = {array.shape for array in channel_arrays.values()}
    if len(shapes) != 1:
        raise ValueError("contrast channels disagree on the grid")
    spatial = next(iter(shapes))
    if len(spatial) != 3:
        raise ValueError("expected a three-dimensional volume per channel")
    channels = meta.channels if meta is not None else tuple(channel_arrays)
    unknown = [name for name in channels if name not in ACQUISITION_CHANNELS]
    if unknown:
        raise ValueError(f"unknown contrast channel {unknown[0]}")
    block = np.zeros((len(ACQUISITION_CHANNELS), *spatial), dtype=np.float32)
    for name, array in channel_arrays.items():
        block[ACQUISITION_CHANNELS.index(name)] = array.astype(np.float32, copy=False)
    return PlaqueVolume(array=block, meta=meta or AcquisitionMeta(channels=channels))


def mask_missing_channels(volume: np.ndarray, present: tuple[str, ...]) -> np.ndarray:
    """Zero every channel that the acquisition did not record."""

    output = volume.copy()
    for index, name in enumerate(ACQUISITION_CHANNELS):
        if name not in present:
            output[index] = 0.0
    return output


def resample_slice_thickness(volume: np.ndarray, source_mm: float, target_mm: float) -> np.ndarray:
    """Resample along the slice axis by linear interpolation between slice centres."""

    if source_mm <= 0.0 or target_mm <= 0.0:
        raise ValueError("slice thickness must be positive")
    depth = volume.shape[1]
    if depth < 2:
        return volume.copy()
    target_depth = max(2, int(round(depth * source_mm / target_mm)))
    source_grid = np.arange(depth, dtype=np.float64) * source_mm
    target_grid = np.arange(target_depth, dtype=np.float64) * target_mm
    flat = volume.reshape(volume.shape[0], depth, -1)
    interpolated = np.empty((flat.shape[0], target_depth, flat.shape[2]), dtype=np.float32)
    for channel in range(flat.shape[0]):
        for column in range(flat.shape[2]):
            interpolated[channel, :, column] = np.interp(
                target_grid, source_grid, flat[channel, :, column]
            )
    return interpolated.reshape(volume.shape[0], target_depth, *volume.shape[2:])


def _blur_along_axis(volume: np.ndarray, axis: int, kernel: np.ndarray) -> np.ndarray:
    padding: list[tuple[int, int]] = [(0, 0)] * volume.ndim
    padding[axis] = (len(kernel) // 2, len(kernel) // 2)
    padded = np.moveaxis(np.pad(volume, padding, mode="edge"), axis, 0)
    moved = np.moveaxis(volume, axis, 0)
    accumulated = np.zeros(moved.shape, dtype=np.float64)
    for offset, weight in enumerate(kernel):
        accumulated += weight * padded[offset : offset + moved.shape[0]]
    return np.moveaxis(accumulated, 0, axis).astype(np.float32)


def apply_motion_degradation(
    volume: np.ndarray, severity: float, rng: np.random.Generator
) -> np.ndarray:
    """Apply a through-plane motion degradation as a phase ramp plus a directional blur."""

    if severity <= 0.0:
        return volume.copy()
    depth = volume.shape[1]
    ramp = np.linspace(0.0, 2.0 * np.pi * severity, depth, dtype=np.float64)
    ghost = (severity * np.sin(ramp)).astype(np.float32)
    shifted = np.roll(volume, 1, axis=1) - volume
    degraded = volume + ghost[None, :, None, None] * shifted
    sigma = max(1, int(round(severity * 3.0)))
    kernel = np.exp(-0.5 * (np.arange(-sigma, sigma + 1) / max(1.0, sigma / 2.0)) ** 2)
    kernel = (kernel / kernel.sum()).astype(np.float64)
    jitter = rng.normal(0.0, 0.01 * severity, size=degraded.shape).astype(np.float32)
    return _blur_along_axis(degraded, 1, kernel) + jitter


def downsample_anisotropic(volume: np.ndarray, factor: int) -> np.ndarray:
    """Decimate the slice axis and restore the original depth by nearest resampling."""

    if factor <= 1:
        return volume.copy()
    depth = volume.shape[1]
    coarse = volume[:, ::factor]
    if coarse.shape[1] < 2:
        return np.repeat(volume[:, :1], depth, axis=1)
    indices = np.linspace(0, coarse.shape[1] - 1, depth)
    lower = np.floor(indices).astype(np.int64)
    upper = np.clip(lower + 1, 0, coarse.shape[1] - 1)
    weight = (indices - lower).astype(np.float32)
    blended = (
        coarse[:, lower] * (1.0 - weight)[None, :, None, None]
        + coarse[:, upper] * weight[None, :, None, None]
    )
    return blended.astype(np.float32)


def volume_scale_factor(volume_mm3: float, reference_mm3: float) -> float:
    """Small plaques carry a noisier resistance read-out; expose the ratio as a covariate."""

    if reference_mm3 <= 0.0:
        raise ValueError("reference volume must be positive")
    return float(np.cbrt(max(volume_mm3, 1e-6) / reference_mm3))


class VolumeBank:
    """Holds the assembled volumes of one split as a single array for batching."""

    def __init__(self, volumes: list[np.ndarray], record_ids: tuple[str, ...]) -> None:
        if len(volumes) != len(record_ids):
            raise ValueError("volume and identifier counts differ")
        self._volumes = volumes
        self._record_ids = record_ids

    def __len__(self) -> int:
        return len(self._volumes)

    @property
    def record_ids(self) -> tuple[str, ...]:
        return self._record_ids

    def index_of(self, record_id: str) -> int:
        return self._record_ids.index(record_id)

    def stacked(self) -> np.ndarray:
        return np.stack(self._volumes, axis=0)

    def channel_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        """Pooled per-channel mean and standard deviation over foreground voxels."""

        stacked = self.stacked()
        means = np.zeros(stacked.shape[1], dtype=np.float64)
        deviations = np.zeros(stacked.shape[1], dtype=np.float64)
        for channel in range(stacked.shape[1]):
            foreground = stacked[:, channel][stacked[:, channel] > 0.0]
            if foreground.size == 0:
                foreground = stacked[:, channel].reshape(-1)
            means[channel] = float(np.mean(foreground))
            deviations[channel] = float(np.std(foreground) + 1e-6)
        return means, deviations
