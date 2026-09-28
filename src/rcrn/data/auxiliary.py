"""Public assay releases used for auxiliary supervision and endpoint comparison.

Ref: Methods, "Cohort construction and auxiliary data", Layer (ii); Data
availability, Layer (ii); Results, Section 2.4 (the assay endpoint and its
out-of-distribution release).

The tables are read from an on-disk release; nothing here downloads. The reader
takes a molecule identifier, a structure string, an optional cluster label and a
measured target. Cluster labels drive the grouped split, which keeps structural
analogues on one side of the partition.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rcrn.data.descriptors import descriptor_vector
from rcrn.data.molecules import parse_smiles

ASSAY_COLUMNS: tuple[str, ...] = ("molecule_id", "smiles", "target")
OPTIONAL_COLUMNS: tuple[str, ...] = ("cluster", "split")


@dataclass(slots=True)
class AssayTable:
    """One permeability release with a group label per compound."""

    molecule_ids: tuple[str, ...]
    smiles: tuple[str, ...]
    targets: np.ndarray
    clusters: np.ndarray
    source: str
    licence: str

    def __len__(self) -> int:
        return len(self.molecule_ids)

    def descriptor_block(self) -> np.ndarray:
        rows = [descriptor_vector(parse_smiles(structure)) for structure in self.smiles]
        return np.stack(rows, axis=0)


@dataclass(slots=True)
class GroupedSplit:
    """Indices of the fitting and held-out parts of an assay release."""

    fit_index: np.ndarray
    held_out_index: np.ndarray

    def sizes(self) -> tuple[int, int]:
        return int(self.fit_index.size), int(self.held_out_index.size)


def load_assay_table(path: str | Path, source: str, licence: str) -> AssayTable:
    """Read a permeability release from CSV."""

    table = Path(path)
    if not table.is_file():
        raise FileNotFoundError(table)
    with table.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        missing = [name for name in ASSAY_COLUMNS if name not in header]
        if missing:
            raise ValueError(f"assay table is missing column {missing[0]}")
        identifiers: list[str] = []
        structures: list[str] = []
        targets: list[float] = []
        clusters: list[str] = []
        for row in reader:
            identifiers.append(row["molecule_id"])
            structures.append(row["smiles"])
            targets.append(float(row["target"]))
            clusters.append(row.get("cluster") or row["molecule_id"])
    return AssayTable(
        molecule_ids=tuple(identifiers),
        smiles=tuple(structures),
        targets=np.array(targets, dtype=np.float64),
        clusters=np.array(clusters, dtype=object),
        source=source,
        licence=licence,
    )


def grouped_split(table: AssayTable, held_out_fraction: float, seed: int) -> GroupedSplit:
    """Hold out whole clusters, so no structural analogue crosses the partition."""

    if not 0.0 < held_out_fraction < 1.0:
        raise ValueError("held-out fraction must lie in (0, 1)")
    unique = np.array(sorted(set(table.clusters.tolist())), dtype=object)
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique.size)
    target_count = int(round(table.clusters.size * held_out_fraction))
    held_out_clusters: set[object] = set()
    running = 0
    for position in order:
        if running >= target_count:
            break
        label = unique[position]
        held_out_clusters.add(label)
        running += int(np.sum(table.clusters == label))
    mask = np.array([label in held_out_clusters for label in table.clusters], dtype=bool)
    return GroupedSplit(
        fit_index=np.flatnonzero(~mask),
        held_out_index=np.flatnonzero(mask),
    )


def mean_absolute_error(target: np.ndarray, prediction: np.ndarray) -> float:
    """Mean absolute error, the metric the assay comparison reports."""

    if target.shape != prediction.shape:
        raise ValueError("target and prediction shapes differ")
    if target.size == 0:
        raise ValueError("no observations")
    return float(np.mean(np.abs(target - prediction)))


def assay_penalty(in_distribution: float, out_of_distribution: float) -> float:
    """Inflation of the mean absolute error when the release moves off distribution."""

    if in_distribution <= 0.0:
        raise ValueError("in-distribution error must be positive")
    return out_of_distribution / in_distribution


def rank_families_by_error(
    errors: dict[str, float], descending: bool = False
) -> tuple[tuple[str, float], ...]:
    """Order descriptor and architecture families by endpoint error."""

    return tuple(sorted(errors.items(), key=lambda item: item[1], reverse=descending))
