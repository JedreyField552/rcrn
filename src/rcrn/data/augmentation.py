"""Acquisition perturbation families and the admissible-perturbation budget.

Ref: Results, Table 3 (the four imaging-side degradation classes); Methods,
"Certified top-k prioritisation", Eq. (5) (the bound on admissible plaque-input
variation).

Each family is expressed as a bounded operator on the input volume so the same
bound can be reused both as an evaluation fold and as the radius delta_adm of the
certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcrn.data.plaque_volumes import (
    apply_motion_degradation,
    downsample_anisotropic,
    mask_missing_channels,
    resample_slice_thickness,
)
from rcrn.data.schemas import AcquisitionMeta

DEGRADATION_FAMILIES: tuple[str, ...] = (
    "gaussian_intensity_noise",
    "contrast_channel_dropout",
    "slice_thickening",
    "motion",
    "small_plaque_blur",
)


@dataclass(slots=True)
class PerturbationReport:
    """Realised perturbation of one volume under one family."""

    family: str
    magnitude: float
    relative_l2: float


def _relative_l2(reference: np.ndarray, perturbed: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference.reshape(-1))) + 1e-9
    return float(np.linalg.norm((perturbed - reference).reshape(-1))) / denominator


class AcquisitionPerturber:
    """Apply one bounded acquisition perturbation family to a plaque volume."""

    def __init__(self, families: tuple[str, ...] = DEGRADATION_FAMILIES) -> None:
        unknown = [family for family in families if family not in DEGRADATION_FAMILIES]
        if unknown:
            raise ValueError(f"unknown perturbation family {unknown[0]}")
        self._families = families

    @property
    def families(self) -> tuple[str, ...]:
        return self._families

    def apply(
        self,
        volume: np.ndarray,
        meta: AcquisitionMeta,
        family: str,
        magnitude: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, PerturbationReport]:
        if family == "gaussian_intensity_noise":
            scale = float(np.std(volume) + 1e-6) * magnitude
            perturbed = volume + rng.normal(0.0, scale, size=volume.shape).astype(np.float32)
        elif family == "contrast_channel_dropout":
            drop_count = max(1, int(round(magnitude * len(meta.channels))))
            present = list(meta.channels)
            for _ in range(min(drop_count, len(present))):
                present.pop(int(rng.integers(0, len(present))))
            perturbed = mask_missing_channels(volume, tuple(present))
        elif family == "slice_thickening":
            target = max(meta.slice_thickness_mm, 3.0 * (1.0 + magnitude))
            perturbed = resample_slice_thickness(volume, meta.slice_thickness_mm, target)
            perturbed = _restore_depth(perturbed, volume.shape[1])
        elif family == "motion":
            perturbed = apply_motion_degradation(volume, magnitude, rng)
        elif family == "small_plaque_blur":
            factor = max(1, int(round(1.0 + magnitude * 3.0)))
            perturbed = downsample_anisotropic(volume, factor)
        else:
            raise ValueError(f"unknown perturbation family {family}")
        report = PerturbationReport(
            family=family, magnitude=magnitude, relative_l2=_relative_l2(volume, perturbed)
        )
        return perturbed.astype(np.float32, copy=False), report


def _restore_depth(volume: np.ndarray, depth: int) -> np.ndarray:
    if volume.shape[1] == depth:
        return volume
    indices = np.linspace(0, volume.shape[1] - 1, depth)
    lower = np.floor(indices).astype(np.int64)
    upper = np.clip(lower + 1, 0, volume.shape[1] - 1)
    weight = (indices - lower).astype(np.float32)
    return (
        volume[:, lower] * (1.0 - weight)[None, :, None, None]
        + volume[:, upper] * weight[None, :, None, None]
    ).astype(np.float32)


def admissible_radius(
    perturber: AcquisitionPerturber, base_volume: np.ndarray, meta: AcquisitionMeta, seed: int
) -> float:
    """Largest relative L2 displacement produced by the family at unit magnitude."""

    rng = np.random.default_rng(seed)
    largest = 0.0
    for family in perturber.families:
        _, report = perturber.apply(base_volume, meta, family, 1.0, rng)
        largest = max(largest, report.relative_l2)
    return float(largest)


def sweep_magnitudes(family: str) -> tuple[float, ...]:
    """The magnitude ladder used for the perturbation sweep of the discussion."""

    if family == "contrast_channel_dropout":
        return (0.25, 0.5, 0.75, 1.0)
    return (0.05, 0.1, 0.2, 0.35, 0.5)
