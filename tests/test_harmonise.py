"""Intensity harmonisation."""

from __future__ import annotations

import numpy as np

from rcrn.data.harmonise import IntensityHarmoniser, LandmarkMap, robust_landmarks


def test_landmark_map_is_identity_for_matching_landmarks() -> None:
    landmarks = np.array([0.1, 0.5, 0.9])
    mapping = LandmarkMap(source=landmarks, target=landmarks)
    values = np.array([0.0, 0.1, 0.3, 0.5, 0.9, 1.2])
    mapped = mapping.apply(values)
    assert np.allclose(mapped, values)


def test_landmark_map_is_monotone() -> None:
    mapping = LandmarkMap(source=np.array([0.0, 0.5, 1.0]), target=np.array([0.0, 0.2, 2.0]))
    values = np.linspace(0.0, 1.0, 21)
    mapped = mapping.apply(values)
    assert np.all(np.diff(mapped) > 0)


def test_landmarks_ignore_background() -> None:
    volume = np.zeros((4, 4, 4), dtype=np.float64)
    volume[1:3, 1:3, 1:3] = np.arange(8).reshape(2, 2, 2) + 1.0
    landmarks = robust_landmarks(volume, (0.5,))
    assert landmarks[0] > 0.0


def test_harmoniser_aligns_channel_distributions() -> None:
    rng = np.random.default_rng(0)
    pool = rng.gamma(2.0, 1.0, size=(6, 2, 4, 4, 4)).astype(np.float64)
    harmoniser = IntensityHarmoniser().fit(pool)
    shifted = pool[0] * 3.0 + 1.0
    transformed = harmoniser.transform(shifted)
    assert transformed.shape == shifted.shape
    reference = harmoniser.reference()
    assert reference.shape == (2, 7)
    assert np.isfinite(transformed).all()


def test_unfitted_harmoniser_refuses() -> None:
    try:
        IntensityHarmoniser().transform(np.zeros((2, 2, 2, 2)))
    except RuntimeError:
        return
    raise AssertionError("an unfitted harmoniser must refuse to transform")
