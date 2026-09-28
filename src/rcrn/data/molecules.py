"""Molecular graph construction and featurisation.

Ref: Methods, "The regime-route molecular encoder".

The branch consumes a bond graph rather than a scalar permeability. This module
turns a linear notation string into that graph, perceives the ring structure and
emits fixed-width atom and bond feature blocks. Element valences, the hydrogen
count estimate and the hybridisation assignment follow the usual organic-subset
conventions; the manuscript fixes neither, so they are engineering defaults.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Final

import numpy as np

from rcrn.data.schemas import MoleculeRecord

ORGANIC_SUBSET: Final[tuple[str, ...]] = ("B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I")
OTHER_ELEMENT_INDEX: Final[int] = len(ORGANIC_SUBSET)
ELEMENT_SLOTS: Final[int] = len(ORGANIC_SUBSET) + 1

DEGREE_SLOTS: Final[int] = 6
CHARGE_SLOTS: Final[int] = 5
HYDROGEN_SLOTS: Final[int] = 5
HYBRIDISATION_SLOTS: Final[int] = 6
RING_SIZE_SLOTS: Final[int] = 7
STEREO_SLOTS: Final[int] = 3
BOND_ORDER_SLOTS: Final[int] = 4
BOND_FLAG_SLOTS: Final[int] = 5
ATOM_RING_FLAGS: Final[int] = 2

ATOM_FEATURE_DIM: Final[int] = (
    ELEMENT_SLOTS
    + DEGREE_SLOTS
    + CHARGE_SLOTS
    + HYDROGEN_SLOTS
    + HYBRIDISATION_SLOTS
    + ATOM_RING_FLAGS
    + RING_SIZE_SLOTS
    + STEREO_SLOTS
)
BOND_FEATURE_DIM: Final[int] = BOND_ORDER_SLOTS + STEREO_SLOTS + BOND_FLAG_SLOTS

STEREOCHEMISTRY: Final[tuple[str, ...]] = ("", "E", "Z")
BOND_FLAG_NAMES: Final[tuple[str, ...]] = (
    "in_ring",
    "conjugated",
    "rotatable",
    "bridges_fragments",
    "same_fragment",
)

DEFAULT_VALENCE: Final[dict[str, int]] = {
    "B": 3,
    "C": 4,
    "N": 3,
    "O": 2,
    "P": 5,
    "S": 2,
    "F": 1,
    "Cl": 1,
    "Br": 1,
    "I": 1,
}


@dataclass(slots=True)
class MoleculeGraph:
    """Element labels, bond endpoints, bond orders, aromaticity and explicit hydrogens."""

    elements: tuple[str, ...]
    bond_ends: tuple[tuple[int, int], ...]
    bond_orders: tuple[float, ...]
    bond_aromatic: tuple[bool, ...]
    bond_stereo: tuple[str, ...]
    charges: tuple[int, ...]
    explicit_hydrogens: tuple[int, ...]

    def num_atoms(self) -> int:
        return len(self.elements)

    def num_bonds(self) -> int:
        return len(self.bond_ends)

    def adjacency(self) -> list[list[tuple[int, int]]]:
        adjacencies: list[list[tuple[int, int]]] = [[] for _ in range(self.num_atoms())]
        for index, (left, right) in enumerate(self.bond_ends):
            adjacencies[left].append((right, index))
            adjacencies[right].append((left, index))
        return adjacencies


def _tokenise(smiles: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(smiles):
        character = smiles[index]
        if character == "%" and index + 2 < len(smiles) and smiles[index + 1 : index + 3].isdigit():
            tokens.append(smiles[index : index + 3])
            index += 3
            continue
        if character == "[":
            closing = smiles.find("]", index)
            if closing < 0:
                raise ValueError("unclosed bracket atom")
            tokens.append(smiles[index : closing + 1])
            index = closing + 1
            continue
        if smiles.startswith("Cl", index) or smiles.startswith("Br", index):
            tokens.append(smiles[index : index + 2])
            index += 2
            continue
        tokens.append(character)
        index += 1
    return tokens


def _element_symbol(body: str) -> str:
    if not body or not body[0].isalpha():
        raise ValueError("bracket atom without an element")
    if len(body) >= 2 and body[1].islower():
        return body[0].upper() + body[1]
    return body[0].upper()


def _parse_bracket_atom(token: str) -> tuple[str, int, bool, int]:
    body = token[1:-1]
    if not body:
        raise ValueError("empty bracket atom")
    aromatic = body[0].islower()
    while body and body[0] == "@":
        body = body[1:]
    symbol = _element_symbol(body)
    remainder = body[len(symbol) :]
    positive = remainder.count("+")
    negative = remainder.count("-")
    charge = positive - negative
    digits = ""
    for index, character in enumerate(remainder):
        if character.isdigit():
            digits += character
            if index + 1 < len(remainder) and not remainder[index + 1].isdigit():
                break
    if digits:
        charge = int(digits) * (1 if positive else -1)
    explicit_hydrogens = 0
    if "H" in remainder:
        position = remainder.index("H") + 1
        explicit_hydrogens = 1
        if position < len(remainder) and remainder[position].isdigit():
            explicit_hydrogens = int(remainder[position])
    return symbol, charge, aromatic, explicit_hydrogens


def parse_smiles(smiles: str) -> MoleculeGraph:
    """Parse the organic subset of a linear notation string into a bond graph.

    A bond between two aromatic atoms is recorded as aromatic unless the string
    names an explicit order, which is what makes an aromatic ring distinguishable
    from its aliphatic counterpart.
    """

    tokens = _tokenise(smiles)
    elements: list[str] = []
    charges: list[int] = []
    explicit_hydrogens: list[int] = []
    atom_aromatic: list[bool] = []
    bond_ends: list[tuple[int, int]] = []
    bond_orders: list[float] = []
    bond_aromatic: list[bool] = []
    bond_stereo: list[str] = []

    branch_stack: list[int] = []
    ring_open: dict[str, tuple[int, float, bool, str, bool]] = {}
    previous: int | None = None
    pending_order = 1.0
    pending_aromatic = False
    pending_stereo = ""
    pending_explicit = False

    def reset_pending() -> tuple[float, bool, str, bool]:
        return 1.0, False, "", False

    for token in tokens:
        if token in "-=#:\\/":
            pending_explicit = True
            if token == "=":
                pending_order, pending_aromatic = 2.0, False
            elif token == "#":
                pending_order, pending_aromatic = 3.0, False
            elif token == ":":
                pending_order, pending_aromatic = 1.5, True
            elif token in "/\\":
                pending_stereo = "E" if token == "/" else "Z"
            else:
                pending_order, pending_aromatic = 1.0, False
            continue
        if token == "(":
            if previous is None:
                raise ValueError("branch without a preceding atom")
            branch_stack.append(previous)
            continue
        if token == ")":
            if not branch_stack:
                raise ValueError("unbalanced branch")
            previous = branch_stack.pop()
            continue
        if token == ".":
            previous = None
            pending_order, pending_aromatic, pending_stereo, pending_explicit = reset_pending()
            continue
        if token.isdigit() or token.startswith("%"):
            if previous is None:
                raise ValueError("ring closure without an atom")
            if token in ring_open:
                partner, order, aromatic, stereo, explicit = ring_open.pop(token)
                resolved_order, resolved_aromatic = _resolve_bond(
                    order,
                    aromatic,
                    explicit,
                    atom_aromatic[partner],
                    atom_aromatic[previous],
                )
                bond_ends.append((partner, previous))
                bond_orders.append(resolved_order)
                bond_aromatic.append(resolved_aromatic)
                bond_stereo.append(stereo)
            else:
                ring_open[token] = (
                    previous,
                    pending_order,
                    pending_aromatic,
                    pending_stereo,
                    pending_explicit,
                )
            pending_order, pending_aromatic, pending_stereo, pending_explicit = reset_pending()
            continue
        if token.startswith("["):
            symbol, charge, aromatic, hydrogens = _parse_bracket_atom(token)
        else:
            symbol = _element_symbol(token)
            charge = 0
            aromatic = token.islower()
            hydrogens = -1
        elements.append(symbol)
        charges.append(charge)
        explicit_hydrogens.append(hydrogens)
        atom_aromatic.append(aromatic)
        current = len(elements) - 1
        if previous is not None:
            resolved_order, resolved_aromatic = _resolve_bond(
                pending_order,
                pending_aromatic,
                pending_explicit,
                atom_aromatic[previous],
                atom_aromatic[current],
            )
            bond_ends.append((previous, current))
            bond_orders.append(resolved_order)
            bond_aromatic.append(resolved_aromatic)
            bond_stereo.append(pending_stereo)
        pending_order, pending_aromatic, pending_stereo, pending_explicit = reset_pending()
        previous = current
    if ring_open:
        raise ValueError("unclosed ring bond")
    return MoleculeGraph(
        elements=tuple(elements),
        bond_ends=tuple(bond_ends),
        bond_orders=tuple(bond_orders),
        bond_aromatic=tuple(bond_aromatic),
        bond_stereo=tuple(bond_stereo),
        charges=tuple(charges),
        explicit_hydrogens=tuple(explicit_hydrogens),
    )


def _resolve_bond(
    order: float,
    aromatic: bool,
    explicit: bool,
    left_aromatic: bool,
    right_aromatic: bool,
) -> tuple[float, bool]:
    """Aromatic atoms form an aromatic bond unless an order was named explicitly."""

    if aromatic:
        return order, True
    if not explicit and order == 1.0 and left_aromatic and right_aromatic:
        return 1.5, True
    return order, False


def _edge_in_cycle(graph: MoleculeGraph, adjacency: list[list[tuple[int, int]]]) -> np.ndarray:
    """An edge lies on a cycle exactly when its endpoints stay connected without it."""

    flags = np.zeros(graph.num_bonds(), dtype=bool)
    for skipped in range(graph.num_bonds()):
        left, right = graph.bond_ends[skipped]
        seen = {left}
        frontier = deque([left])
        while frontier:
            node = frontier.popleft()
            for neighbour, edge_index in adjacency[node]:
                if edge_index == skipped or neighbour in seen:
                    continue
                seen.add(neighbour)
                frontier.append(neighbour)
        flags[skipped] = right in seen
    return flags


def _shortest_cycle_lengths(
    graph: MoleculeGraph, adjacency: list[list[tuple[int, int]]], bond_in_cycle: np.ndarray
) -> np.ndarray:
    """Shortest cycle through each atom, or zero when the atom is acyclic."""

    lengths = np.zeros(graph.num_atoms(), dtype=np.int64)
    for atom in range(graph.num_atoms()):
        best = 0
        for neighbour, direct_edge in adjacency[atom]:
            if not bond_in_cycle[direct_edge]:
                continue
            seen = {atom}
            frontier: deque[tuple[int, int]] = deque([(neighbour, 1)])
            while frontier:
                node, distance = frontier.popleft()
                if node in seen:
                    continue
                seen.add(node)
                if node == atom:
                    best = distance if best == 0 else min(best, distance)
                    break
                for following, edge_index in adjacency[node]:
                    if edge_index == direct_edge:
                        continue
                    frontier.append((following, distance + 1))
        lengths[atom] = best
    return lengths


def _hydrogen_count(graph: MoleculeGraph, adjacency: list[list[tuple[int, int]]]) -> np.ndarray:
    counts = np.zeros(graph.num_atoms(), dtype=np.int64)
    for atom, element in enumerate(graph.elements):
        if graph.explicit_hydrogens[atom] >= 0:
            counts[atom] = graph.explicit_hydrogens[atom]
            continue
        valence = DEFAULT_VALENCE.get(element, 0)
        if valence == 0:
            continue
        used = sum(graph.bond_orders[edge] for _, edge in adjacency[atom])
        counts[atom] = max(0, int(round(valence - used + graph.charges[atom])))
    return np.clip(counts, 0, HYDROGEN_SLOTS - 1)


def _hybridisation(
    graph: MoleculeGraph, adjacency: list[list[tuple[int, int]]], atom_in_ring: np.ndarray
) -> np.ndarray:
    labels = np.full(graph.num_atoms(), 5, dtype=np.int64)
    for atom in range(graph.num_atoms()):
        edges = [edge for _, edge in adjacency[atom]]
        orders = [graph.bond_orders[edge] for edge in edges]
        if not orders:
            labels[atom] = 0
        elif any(order >= 3.0 for order in orders):
            labels[atom] = 1
        elif any(order == 2.0 for order in orders):
            labels[atom] = 2
        elif any(graph.bond_aromatic[edge] for edge in edges):
            labels[atom] = 3
        elif all(order == 1.0 for order in orders):
            labels[atom] = 4
        else:
            labels[atom] = 4
        if atom_in_ring[atom] and labels[atom] == 3:
            labels[atom] = 2
    return labels


def _one_hot(value: int, slots: int) -> np.ndarray:
    block = np.zeros(slots, dtype=np.float32)
    if 0 <= value < slots:
        block[value] = 1.0
    return block


def hydrogen_counts(graph: MoleculeGraph) -> np.ndarray:
    """Per-atom hydrogen count under the valence rules of the organic subset."""

    return _hydrogen_count(graph, graph.adjacency())


def _fragment_partition(
    graph: MoleculeGraph,
    adjacency: list[list[tuple[int, int]]],
    bond_in_cycle: np.ndarray,
    ring_membership: np.ndarray,
) -> np.ndarray:
    """Split a molecule at the acyclic single bonds that join two ring atoms.

    A bond inside a ring system is kept, and a substituent hanging off a ring stays
    with its ring, so a fragment is a ring system rather than a single atom. An
    acyclic molecule is therefore one fragment.
    """

    atom_count = graph.num_atoms()
    assignment = np.full(atom_count, -1, dtype=np.int64)
    fragment = 0
    for root in range(atom_count):
        if assignment[root] >= 0:
            continue
        assignment[root] = fragment
        frontier = deque([root])
        while frontier:
            node = frontier.popleft()
            for neighbour, edge_index in adjacency[node]:
                joint = (
                    graph.bond_orders[edge_index] == 1.0
                    and not bond_in_cycle[edge_index]
                    and bool(ring_membership[node])
                    and bool(ring_membership[neighbour])
                )
                if joint or assignment[neighbour] >= 0:
                    continue
                assignment[neighbour] = fragment
                frontier.append(neighbour)
        fragment += 1
    return assignment


def featurise_graph(graph: MoleculeGraph) -> MoleculeRecord:
    """Build the fixed-width atom, bond and fragment feature blocks of one molecule."""

    atom_count = graph.num_atoms()
    if atom_count == 0:
        raise ValueError("a molecule needs at least one atom")
    adjacency = graph.adjacency()
    bond_in_cycle = (
        _edge_in_cycle(graph, adjacency) if graph.num_bonds() else np.zeros(0, dtype=bool)
    )
    atom_in_ring = np.zeros(atom_count, dtype=bool)
    for index, (left, right) in enumerate(graph.bond_ends):
        if bond_in_cycle[index]:
            atom_in_ring[left] = True
            atom_in_ring[right] = True
    ring_sizes = (
        _shortest_cycle_lengths(graph, adjacency, bond_in_cycle)
        if graph.num_bonds()
        else np.zeros(atom_count, dtype=np.int64)
    )
    hydrogens = _hydrogen_count(graph, adjacency)
    hybridisation = _hybridisation(graph, adjacency, atom_in_ring)
    degrees = np.array([len(neighbours) for neighbours in adjacency], dtype=np.int64)

    atom_features = np.zeros((atom_count, ATOM_FEATURE_DIM), dtype=np.float32)
    for atom in range(atom_count):
        element_index = (
            ORGANIC_SUBSET.index(graph.elements[atom])
            if graph.elements[atom] in ORGANIC_SUBSET
            else OTHER_ELEMENT_INDEX
        )
        ring_bucket = 0 if ring_sizes[atom] == 0 else int(min(int(ring_sizes[atom]), 8)) - 1
        atom_features[atom] = np.concatenate(
            [
                _one_hot(element_index, ELEMENT_SLOTS),
                _one_hot(int(min(degrees[atom], DEGREE_SLOTS - 1)), DEGREE_SLOTS),
                _one_hot(int(np.clip(graph.charges[atom], -2, 2)) + 2, CHARGE_SLOTS),
                _one_hot(int(hydrogens[atom]), HYDROGEN_SLOTS),
                _one_hot(int(hybridisation[atom]), HYBRIDISATION_SLOTS),
                np.array(
                    [1.0 if atom_in_ring[atom] else 0.0, 1.0 if ring_sizes[atom] > 0 else 0.0],
                    dtype=np.float32,
                ),
                _one_hot(int(np.clip(ring_bucket, 0, RING_SIZE_SLOTS - 1)), RING_SIZE_SLOTS),
                _one_hot(0, STEREO_SLOTS),
            ]
        )

    bond_ends = np.array(graph.bond_ends, dtype=np.int64).reshape(-1, 2)
    fragment_index = _fragment_partition(graph, adjacency, bond_in_cycle, atom_in_ring)
    bond_features = np.zeros((graph.num_bonds(), BOND_FEATURE_DIM), dtype=np.float32)
    for index in range(graph.num_bonds()):
        order = graph.bond_orders[index]
        order_slot = BOND_ORDER_SLOTS - 1 if graph.bond_aromatic[index] else int(round(order)) - 1
        left, right = graph.bond_ends[index]
        stereo_slot = STEREOCHEMISTRY.index(graph.bond_stereo[index])
        flags = np.array(
            [
                1.0 if bond_in_cycle[index] else 0.0,
                1.0 if graph.bond_aromatic[index] or order < 2.0 and bond_in_cycle[index] else 0.0,
                1.0 if order == 1.0 and not bond_in_cycle[index] else 0.0,
                1.0 if order == 1.0 and not bond_in_cycle[index] else 0.0,
                1.0 if fragment_index[left] == fragment_index[right] else 0.0,
            ],
            dtype=np.float32,
        )
        bond_features[index] = np.concatenate(
            [
                _one_hot(int(np.clip(order_slot, 0, BOND_ORDER_SLOTS - 1)), BOND_ORDER_SLOTS),
                _one_hot(stereo_slot, STEREO_SLOTS),
                flags,
            ]
        )

    return MoleculeRecord(
        molecule_id="",
        atom_features=atom_features,
        bond_index=bond_ends,
        bond_features=bond_features,
        fragment_index=fragment_index,
    )


def featurise_smiles_list(
    molecule_ids: tuple[str, ...], smiles: tuple[str, ...]
) -> tuple[MoleculeRecord, ...]:
    """Featurise a candidate list, keeping the identifiers aligned."""

    if len(molecule_ids) != len(smiles):
        raise ValueError("identifier and structure counts differ")
    records: list[MoleculeRecord] = []
    for molecule_id, structure in zip(molecule_ids, smiles, strict=True):
        record = featurise_graph(parse_smiles(structure))
        records.append(
            MoleculeRecord(
                molecule_id=molecule_id,
                atom_features=record.atom_features,
                bond_index=record.bond_index,
                bond_features=record.bond_features,
                fragment_index=record.fragment_index,
            )
        )
    return tuple(records)


def collate_molecules(records: tuple[MoleculeRecord, ...]) -> dict[str, np.ndarray]:
    """Batch molecule graphs with a shared node offset and a batch index."""

    atom_blocks: list[np.ndarray] = []
    bond_blocks: list[np.ndarray] = []
    bond_index_blocks: list[np.ndarray] = []
    batch_index: list[np.ndarray] = []
    fragment_blocks: list[np.ndarray] = []
    offset = 0
    for position, record in enumerate(records):
        atom_blocks.append(record.atom_features)
        bond_index_blocks.append(record.bond_index + offset)
        bond_blocks.append(record.bond_features)
        batch_index.append(np.full(record.atom_features.shape[0], position, dtype=np.int64))
        fragment_blocks.append(record.fragment_index)
        offset += record.atom_features.shape[0]
    return {
        "atom_features": np.concatenate(atom_blocks, axis=0),
        "bond_index": np.concatenate(bond_index_blocks, axis=0),
        "bond_features": np.concatenate(bond_blocks, axis=0),
        "batch_index": np.concatenate(batch_index, axis=0),
        "fragment_index": np.concatenate(fragment_blocks, axis=0),
    }
