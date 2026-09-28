"""Tensor batching for molecules and volumes."""

from __future__ import annotations

import numpy as np
import pytest

from rcrn.data.batching import molecule_batch, subset_batch, volume_batch
from rcrn.data.molecules import featurise_smiles_list


def _records():
    return featurise_smiles_list(("A", "B", "C"), ("CCO", "c1ccccc1", "CC(=O)O"))


def test_molecule_batch_reports_the_graph_count() -> None:
    batch = molecule_batch(_records())
    assert batch.graph_count == 3
    assert batch.molecule_ids == ("A", "B", "C")
    assert batch.graph_index.shape[0] == batch.atom_features.shape[0]


def test_subset_batch_keeps_offsets_valid() -> None:
    batch = molecule_batch(_records())
    reduced = subset_batch(batch, (0, 2))
    assert reduced.graph_count == 2
    assert reduced.molecule_ids == ("A", "C")
    assert int(reduced.graph_index.max()) == 1
    if reduced.bond_index.numel():
        assert int(reduced.bond_index.max()) < reduced.atom_features.shape[0]


def test_volume_batch_respects_the_requested_order() -> None:
    volumes = {
        "R1": np.zeros((4, 2, 2, 2), dtype=np.float32),
        "R2": np.ones((4, 2, 2, 2), dtype=np.float32),
    }
    stacked = volume_batch(volumes, ("R2", "R1"))
    assert float(stacked[0].mean()) == 1.0
    assert float(stacked[1].mean()) == 0.0


def test_missing_volume_raises() -> None:
    with pytest.raises(KeyError):
        volume_batch({}, ("R1",))


def test_empty_molecule_batch_raises() -> None:
    with pytest.raises(ValueError):
        molecule_batch(())
