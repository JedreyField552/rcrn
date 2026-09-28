"""Physicochemical and topological descriptor families for the auxiliary prior.

Ref: Methods, "The regime-route molecular encoder" (auxiliary supervision
{(gi, mi)} that regulates route coordinates according to known descriptor-family
patterns); Results, Section 2.4 (the descriptor family as the assay comparator).

Every quantity here is computed exactly from the bond graph. Nothing is learned,
so the descriptor block can be recomputed by hand for any molecule in the
candidate list, which is what makes the assay comparison checkable.
"""

from __future__ import annotations

from collections import deque
from typing import Final

import numpy as np

from rcrn.data.molecules import MoleculeGraph, hydrogen_counts, parse_smiles

ATOMIC_MASS: Final[dict[str, float]] = {
    "H": 1.008,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.904,
}
DEFAULT_MASS: Final[float] = 12.0
TOPOLOGICAL_SLOTS: Final[int] = 12
FINGERPRINT_RADIUS: Final[int] = 2
FINGERPRINT_BITS: Final[int] = 1024
MASK_64: Final[int] = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    """Deterministic 64-bit mixing function, so fingerprints never depend on a seed."""

    state = (value + 0x9E3779B97F4A7C15) & MASK_64
    state = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & MASK_64
    state = ((state ^ (state >> 27)) * 0x94D049BB133111EB) & MASK_64
    return (state ^ (state >> 31)) & MASK_64


def _atom_code(
    graph: MoleculeGraph, atom: int, adjacency: list[list[tuple[int, int]]], hydrogens: np.ndarray
) -> int:
    element = sum(ord(character) for character in graph.elements[atom])
    packed = (
        (element << 24)
        | (len(adjacency[atom]) << 20)
        | ((graph.charges[atom] + 4) << 16)
        | (int(hydrogens[atom]) << 12)
        | (1 if any(graph.bond_aromatic[edge] for _, edge in adjacency[atom]) else 0)
    )
    return _splitmix64(packed)


def circular_fingerprint(
    graph: MoleculeGraph, radius: int = FINGERPRINT_RADIUS, bits: int = FINGERPRINT_BITS
) -> np.ndarray:
    """Counted circular fingerprint over the ordered neighbourhood of every atom."""

    adjacency = graph.adjacency()
    hydrogens = hydrogen_counts(graph)
    codes = [_atom_code(graph, atom, adjacency, hydrogens) for atom in range(graph.num_atoms())]
    counts = np.zeros(bits, dtype=np.float32)
    for code in codes:
        counts[_splitmix64(code) % bits] += 1.0
    for _ in range(radius):
        updated: list[int] = []
        for atom in range(graph.num_atoms()):
            neighbours = sorted(
                (int(round(graph.bond_orders[edge] * 2.0)), codes[neighbour])
                for neighbour, edge in adjacency[atom]
            )
            mixed = codes[atom]
            for bond_key, neighbour_code in neighbours:
                mixed = _splitmix64(mixed ^ _splitmix64(neighbour_code ^ (bond_key << 40)))
            updated.append(_splitmix64(mixed))
        codes = updated
        for code in codes:
            counts[_splitmix64(code) % bits] += 1.0
    return counts


def _all_pairs_distances(graph: MoleculeGraph) -> np.ndarray:
    adjacency = graph.adjacency()
    atom_count = graph.num_atoms()
    distances = np.full((atom_count, atom_count), np.inf, dtype=np.float64)
    for source in range(atom_count):
        distances[source, source] = 0.0
        frontier = deque([source])
        while frontier:
            node = frontier.popleft()
            for neighbour, _ in adjacency[node]:
                if distances[source, neighbour] == np.inf:
                    distances[source, neighbour] = distances[source, node] + 1.0
                    frontier.append(neighbour)
    return distances


def topological_descriptors(graph: MoleculeGraph) -> np.ndarray:
    """Wiener, Randic, Zagreb, Balaban and cyclomatic quantities of the bond graph."""

    adjacency = graph.adjacency()
    atom_count = graph.num_atoms()
    bond_count = graph.num_bonds()
    degrees = np.array([len(neighbours) for neighbours in adjacency], dtype=np.float64)
    components = _component_count(graph)
    cyclomatic = float(bond_count - atom_count + components)
    distances = _all_pairs_distances(graph)
    finite = distances[np.isfinite(distances)]
    wiener = float(np.sum(finite) / 2.0)
    randic = float(
        sum(
            1.0 / np.sqrt(degrees[left] * degrees[right])
            for left, right in graph.bond_ends
            if degrees[left] * degrees[right] > 0.0
        )
    )
    zagreb_m1 = float(np.sum(degrees**2))
    zagreb_m2 = float(sum(degrees[left] * degrees[right] for left, right in graph.bond_ends))
    distance_sums = np.where(np.isfinite(distances), distances, 0.0).sum(axis=1)
    balaban = 0.0
    if bond_count > 0 and cyclomatic >= 0.0:
        terms = [
            1.0 / np.sqrt(distance_sums[left] * distance_sums[right])
            for left, right in graph.bond_ends
            if distance_sums[left] * distance_sums[right] > 0.0
        ]
        balaban = float(bond_count) / (cyclomatic + 1.0) * float(sum(terms))
    diameter = float(finite.max()) if finite.size else 0.0
    mean_distance = float(finite.mean()) if finite.size else 0.0
    blocks = np.array(
        [
            float(atom_count),
            float(bond_count),
            float(components),
            cyclomatic,
            wiener,
            randic,
            zagreb_m1,
            zagreb_m2,
            balaban,
            diameter,
            mean_distance,
            float(np.sum(degrees == 1.0)),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(blocks, nan=0.0, posinf=0.0, neginf=0.0)


def _component_count(graph: MoleculeGraph) -> int:
    adjacency = graph.adjacency()
    seen: set[int] = set()
    components = 0
    for root in range(graph.num_atoms()):
        if root in seen:
            continue
        components += 1
        frontier = deque([root])
        seen.add(root)
        while frontier:
            node = frontier.popleft()
            for neighbour, _ in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    frontier.append(neighbour)
    return components


def physicochemical_descriptors(graph: MoleculeGraph) -> np.ndarray:
    """Element fractions, estimated molecular weight and heteroatom burden."""

    hydrogens = hydrogen_counts(graph)
    mass = sum(ATOMIC_MASS.get(element, DEFAULT_MASS) for element in graph.elements)
    mass += float(hydrogens.sum()) * ATOMIC_MASS["H"]
    atom_count = float(graph.num_atoms())
    heavy = {
        element: float(graph.elements.count(element))
        for element in ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I")
    }
    fractions = np.array([heavy[name] / atom_count for name in heavy], dtype=np.float32)
    burden = np.array(
        [
            mass,
            mass / max(atom_count, 1.0),
            float(hydrogens.sum()),
            float(sum(heavy[name] for name in ("N", "O", "S"))),
            float(heavy["C"]),
        ],
        dtype=np.float32,
    )
    return np.concatenate([fractions, burden])


def descriptor_vector(graph: MoleculeGraph) -> np.ndarray:
    """Concatenate the topological, physicochemical and fingerprint blocks."""

    return np.concatenate(
        [
            topological_descriptors(graph),
            physicochemical_descriptors(graph),
            circular_fingerprint(graph),
        ]
    ).astype(np.float32)


def descriptor_matrix(
    molecule_ids: tuple[str, ...], smiles: tuple[str, ...]
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Descriptor block for a candidate list, with the ordering made explicit."""

    if len(molecule_ids) != len(smiles):
        raise ValueError("identifier and structure counts differ")
    rows = [descriptor_vector(parse_smiles(structure)) for structure in smiles]
    return np.stack(rows, axis=0), molecule_ids


def descriptor_family_names() -> tuple[str, ...]:
    """Family labels in the order the descriptor block emits them."""

    return ("topological", "physicochemical", "circular_fingerprint")
