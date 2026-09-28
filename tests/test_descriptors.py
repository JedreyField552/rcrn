"""Descriptor families."""

from __future__ import annotations

import numpy as np

from rcrn.data.descriptors import (
    circular_fingerprint,
    descriptor_matrix,
    physicochemical_descriptors,
    topological_descriptors,
)
from rcrn.data.molecules import parse_smiles


def test_benzene_topological_quantities() -> None:
    """Cyclomatic number one and a Wiener index of twenty-seven for a six-cycle."""

    graph = parse_smiles("c1ccccc1")
    block = topological_descriptors(graph)
    atoms, bonds, components, cyclomatic, wiener = block[:5]
    assert atoms == 6.0
    assert bonds == 6.0
    assert components == 1.0
    assert cyclomatic == 1.0
    assert wiener == 27.0


def test_acyclic_chain_wiener_index() -> None:
    """A four-atom path has unordered distance sum 10, hence half of it for the index."""

    graph = parse_smiles("CCCC")
    block = topological_descriptors(graph)
    assert block[3] == 0.0
    assert block[4] == 10.0


def test_physicochemical_block_reports_element_fractions() -> None:
    graph = parse_smiles("CCO")
    block = physicochemical_descriptors(graph)
    fractions = block[:9]
    assert fractions[0] == 2 / 3
    assert fractions[2] == 1 / 3
    assert abs(float(block[9]) - (2 * 12.011 + 6 * 1.008 + 15.999)) < 1e-3


def test_fingerprint_is_deterministic_and_bounded() -> None:
    graph = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
    first = circular_fingerprint(graph)
    second = circular_fingerprint(graph)
    assert np.array_equal(first, second)
    assert first.sum() == 3 * graph.num_atoms()
    assert first.min() >= 0.0


def test_distinct_structures_give_distinct_fingerprints() -> None:
    first = circular_fingerprint(parse_smiles("CCO"))
    second = circular_fingerprint(parse_smiles("COC"))
    assert not np.array_equal(first, second)


def test_descriptor_matrix_keeps_the_candidate_order() -> None:
    identifiers = ("m1", "m2", "m3")
    matrix, kept = descriptor_matrix(identifiers, ("CCO", "c1ccccc1", "CC(=O)O"))
    assert kept == identifiers
    assert matrix.shape[0] == 3
    assert matrix.dtype == np.float32
