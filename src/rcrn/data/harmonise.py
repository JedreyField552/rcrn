"""Site-blind intensity harmonisation.

Ref: Methods, "The plaque composition encoder" (intensity harmonisation without
exposing site identity); Methods, "Cohort construction and auxiliary data"
(normalisation statistics fitted on the training partition only).

The landmark map is the standard piecewise-linear intensity mapping between two
sets of robust intensity percentiles. It is fitted on the pooled training volumes
and therefore never sees a site label; a site can still be told apart afterwards
by its own outcome statistics, which is what the study design requires.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_LANDMARKS: tuple[float, ...] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98)


@dataclass(slots=True)
class ChannelScaling:
    """Robust location and scale of one channel, fitted on the training pool."""

    median: float
    interquartile_range: float

    def apply(self, values: np.ndarray) -> np.ndarray:
        scale = self.interquartile_range if self.interquartile_range > 0.0 else 1.0
        return (values - self.median) / scale


@dataclass(slots=True)
class LandmarkMap:
    """Piecewise-linear map between the reference landmarks and one volume's landmarks."""

    source: np.ndarray
    target: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        flat = values.astype(np.float64, copy=False)
        mapped = np.interp(flat, self.source, self.target)
        outside_low = flat < self.source[0]
        outside_high = flat > self.source[-1]
        if np.any(outside_low):
            mapped[outside_low] = self.target[0] + (flat[outside_low] - self.source[0])
        if np.any(outside_high):
            mapped[outside_high] = self.target[-1] + (flat[outside_high] - self.source[-1])
        return mapped.astype(values.dtype, copy=False)


def robust_landmarks(
    volume: np.ndarray, percentiles: tuple[float, ...] = DEFAULT_LANDMARKS
) -> np.ndarray:
    """Percentile landmarks of the foreground intensities of one volume."""

    foreground = volume[volume > 0.0]
    sample = foreground if foreground.size else volume.reshape(-1)
    if sample.size == 0:
        return np.zeros(len(percentiles), dtype=np.float64)
    return np.percentile(sample.astype(np.float64, copy=False), [p * 100.0 for p in percentiles])


class IntensityHarmoniser:
    """Fit a reference landmark set and map every volume onto it."""

    def __init__(self, percentiles: tuple[float, ...] = DEFAULT_LANDMARKS) -> None:
        self._percentiles = percentiles
        self._reference: np.ndarray | None = None
        self._scalings: dict[int, ChannelScaling] = {}

    @property
    def percentiles(self) -> tuple[float, ...]:
        return self._percentiles

    def fit(self, volumes: np.ndarray) -> IntensityHarmoniser:
        """Fit the reference landmarks and per-channel robust scaling on a training pool."""

        if volumes.ndim != 5:
            raise ValueError("expected a (N, C, Z, H, W) pool")
        channel_landmarks: list[np.ndarray] = []
        for channel in range(volumes.shape[1]):
            per_volume = [
                robust_landmarks(volumes[index, channel], self._percentiles)
                for index in range(volumes.shape[0])
            ]
            stacked = np.stack(per_volume, axis=0)
            channel_landmarks.append(np.median(stacked, axis=0))
            values = volumes[:, channel]
            foreground = values[values > 0.0]
            if foreground.size == 0:
                foreground = values.reshape(-1)
            lower, upper = np.percentile(foreground, [25.0, 75.0])
            self._scalings[channel] = ChannelScaling(
                median=float(np.median(foreground)), interquartile_range=float(upper - lower)
            )
        self._reference = np.stack(channel_landmarks, axis=0)
        return self

    def transform(self, volume: np.ndarray) -> np.ndarray:
        """Apply the landmark map and the robust scaling to one multi-channel volume."""

        if self._reference is None:
            raise RuntimeError("harmoniser is not fitted")
        if volume.ndim != 4:
            raise ValueError("expected a (C, Z, H, W) volume")
        if volume.shape[0] != self._reference.shape[0]:
            raise ValueError("channel count differs from the fitted reference")
        output = np.empty_like(volume)
        for channel in range(volume.shape[0]):
            landmarks = robust_landmarks(volume[channel], self._percentiles)
            mapping = LandmarkMap(source=landmarks, target=self._reference[channel])
            mapped = mapping.apply(volume[channel])
            output[channel] = self._scalings[channel].apply(mapped)
        return output

    def reference(self) -> np.ndarray:
        if self._reference is None:
            raise RuntimeError("harmoniser is not fitted")
        return self._reference
