"""Candidate-list loading.

Ref: Methods, "Problem formulation" (a list M of anti-inflammatory small molecules,
each associated with a graph gi).

The candidate list is an input artefact: an identifier and a structure string per
molecule, supplied by the study. This repository does not ship a structure table of
its own, because the list the study scores is part of the authorised material.
"""

from __future__ import annotations

import csv
from pathlib import Path

from rcrn.data.molecules import featurise_smiles_list
from rcrn.data.schemas import MoleculeRecord

REQUIRED_COLUMNS: tuple[str, ...] = ("molecule_id", "smiles")


def load_candidate_table(path: str | Path) -> tuple[MoleculeRecord, ...]:
    """Read a candidate list and featurise every structure."""

    table = Path(path)
    if not table.is_file():
        raise FileNotFoundError(table)
    identifiers: list[str] = []
    structures: list[str] = []
    with table.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        missing = [name for name in REQUIRED_COLUMNS if name not in header]
        if missing:
            raise ValueError(f"candidate table is missing column {missing[0]}")
        for row in reader:
            identifiers.append(row["molecule_id"].strip())
            structures.append(row["smiles"].strip())
    if not identifiers:
        raise ValueError("the candidate table is empty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate identifiers must be unique")
    return featurise_smiles_list(tuple(identifiers), tuple(structures))


def candidate_identifiers(records: tuple[MoleculeRecord, ...]) -> tuple[str, ...]:
    """Identifiers in the order the candidate list is scored."""

    return tuple(record.molecule_id for record in records)
