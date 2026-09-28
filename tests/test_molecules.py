"""Molecular graph parsing and featurisation."""

from __future__ import annotations

import numpy as np
import pytest

from rcrn.data.molecules import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    collate_molecules,
    featurise_graph,
    featurise_smiles_list,
    hydrogen_counts,
    parse_smiles,
)


@pytest.mark.parametrize(
    ("structure", "atoms", "bonds"),
    [
        ("CCO", 3, 2),
        ("c1ccccc1", 6, 6),
        ("CC(=O)O", 4, 3),
        ("ClCCl", 3, 2),
        ("[nH]1cccc1", 5, 5),
    ],
)
def test_hand_counted_graphs(structure: str, atoms: int, bonds: int) -> None:
    graph = parse_smiles(structure)
    assert graph.num_atoms() == atoms
    assert graph.num_bonds() == bonds


def test_benzene_is_one_ring_system() -> None:
    graph = parse_smiles("c1ccccc1")
    record = featurise_graph(graph)
    assert int(record.fragment_index.max()) + 1 == 1
    assert all(order == 1.5 for order in graph.bond_orders)


def test_ethanol_hydrogens_follow_valence() -> None:
    graph = parse_smiles("CCO")
    counts = hydrogen_counts(graph)
    assert counts.tolist() == [3, 2, 1]


def test_feature_blocks_have_declared_width() -> None:
    record = featurise_graph(parse_smiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O"))
    assert record.atom_features.shape[1] == ATOM_FEATURE_DIM
    assert record.bond_features.shape[1] == BOND_FEATURE_DIM
    assert record.bond_index.shape[1] == 2


def test_branch_and_ring_closure_recover_connectivity() -> None:
    graph = parse_smiles("CC(C)C")
    assert graph.num_atoms() == 4
    assert graph.num_bonds() == 3
    assert sorted(graph.bond_ends) == [(0, 1), (1, 2), (1, 3)]


def test_disconnected_components_are_separate_fragments() -> None:
    record = featurise_graph(parse_smiles("CCO.CC"))
    assert int(record.fragment_index.max()) + 1 == 2


def test_acyclic_molecule_is_one_fragment() -> None:
    record = featurise_graph(parse_smiles("CC(C)CC(=O)O"))
    assert int(record.fragment_index.max()) + 1 == 1


def test_linked_ring_systems_are_separate_fragments() -> None:
    record = featurise_graph(parse_smiles("c1ccccc1-c1ccccc1"))
    assert int(record.fragment_index.max()) + 1 == 2


def test_invalid_not_notation_raises() -> None:
    with pytest.raises(ValueError):
        parse_smiles("C1CC")
    with pytest.raises(ValueError):
        parse_smiles("C)")


def test_collate_offsets_bond_indices() -> None:
    records = featurise_smiles_list(("A", "B"), ("CCO", "CC(=O)O"))
    batch = collate_molecules(records)
    assert batch["atom_features"].shape[0] == 7
    assert batch["bond_index"].max() < 7
    assert batch["batch_index"].tolist() == [0, 0, 0, 1, 1, 1, 1]


def test_identifier_count_must_match() -> None:
    with pytest.raises(ValueError):
        featurise_smiles_list(("A",), ("CCO", "CC"))


def test_aromatic_flag_survives_featurisation() -> None:
    aromatic = featurise_graph(parse_smiles("c1ccccc1"))
    aliphatic = featurise_graph(parse_smiles("C1CCCCC1"))
    assert not np.allclose(aromatic.bond_features, aliphatic.bond_features)
