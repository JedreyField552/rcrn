"""Tensor batching for molecule graphs and plaque volumes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rcrn.data.molecules import collate_molecules
from rcrn.data.schemas import MoleculeRecord


@dataclass(slots=True)
class MoleculeBatch:
    """Flat node and bond tensors plus the graph index they belong to."""

    atom_features: torch.Tensor
    bond_index: torch.Tensor
    bond_features: torch.Tensor
    graph_index: torch.Tensor
    fragment_index: torch.Tensor
    graph_count: int
    molecule_ids: tuple[str, ...]


def molecule_batch(records: tuple[MoleculeRecord, ...]) -> MoleculeBatch:
    """Collate molecular graphs into one flat batch."""

    if not records:
        raise ValueError("no molecules to batch")
    arrays = collate_molecules(records)
    return MoleculeBatch(
        atom_features=torch.from_numpy(arrays["atom_features"]),
        bond_index=torch.from_numpy(arrays["bond_index"]),
        bond_features=torch.from_numpy(arrays["bond_features"]),
        graph_index=torch.from_numpy(arrays["batch_index"]),
        fragment_index=torch.from_numpy(arrays["fragment_index"]),
        graph_count=len(records),
        molecule_ids=tuple(record.molecule_id for record in records),
    )


def volume_batch(volumes: dict[str, np.ndarray], record_ids: tuple[str, ...]) -> torch.Tensor:
    """Stack the volumes of a record selection in the given order."""

    missing = [record_id for record_id in record_ids if record_id not in volumes]
    if missing:
        raise KeyError(missing[0])
    stacked = np.stack([volumes[record_id] for record_id in record_ids], axis=0)
    return torch.from_numpy(np.ascontiguousarray(stacked, dtype=np.float32))


def subset_batch(batch: MoleculeBatch, indices: tuple[int, ...]) -> MoleculeBatch:
    """Select a subset of molecules from an existing batch, keeping bond offsets."""

    if not indices:
        raise ValueError("empty subset")
    keep = set(indices)
    node_keep = [index for index, graph in enumerate(batch.graph_index.tolist()) if graph in keep]
    node_lookup = {node: position for position, node in enumerate(node_keep)}
    edge_keep = [
        edge
        for edge in range(batch.bond_index.shape[0])
        if int(batch.bond_index[edge, 0]) in node_lookup
        and int(batch.bond_index[edge, 1]) in node_lookup
    ]
    edge_index = torch.tensor(
        [
            [
                node_lookup[int(batch.bond_index[edge, 0])],
                node_lookup[int(batch.bond_index[edge, 1])],
            ]
            for edge in edge_keep
        ],
        dtype=batch.bond_index.dtype,
    ).reshape(-1, 2)
    remap = {graph: position for position, graph in enumerate(sorted(keep))}
    return MoleculeBatch(
        atom_features=batch.atom_features[node_keep],
        bond_index=edge_index,
        bond_features=batch.bond_features[edge_keep],
        graph_index=torch.tensor(
            [remap[int(batch.graph_index[node])] for node in node_keep],
            dtype=batch.graph_index.dtype,
        ),
        fragment_index=batch.fragment_index[node_keep],
        graph_count=len(remap),
        molecule_ids=tuple(batch.molecule_ids[graph] for graph in sorted(remap)),
    )
