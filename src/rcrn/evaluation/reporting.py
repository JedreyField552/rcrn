"""Table assembly and JSON-safe reporting.

Ref: Tables 1-4; Figures 1-3. Every reported cell keeps the stratum, the arm or the
condition it belongs to, so a reader can never mistake a pooled value for a stratum
value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class TableRow:
    """One row of a reported table."""

    label: str
    cells: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {"label": self.label, **self.cells}


@dataclass(slots=True)
class ReportTable:
    """A titled table with a fixed column order."""

    title: str
    columns: tuple[str, ...]
    rows: tuple[TableRow, ...] = ()

    def as_record(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "columns": list(self.columns),
            "rows": [row.as_record() for row in self.rows],
        }

    def cell(self, label: str, column: str) -> Any:
        for row in self.rows:
            if row.label == label:
                return row.cells.get(column)
        raise KeyError(label)


def jsonable(value: Any) -> Any:
    """Convert NumPy and torch scalars into JSON-safe values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def ranked_table(
    rows: tuple[TableRow, ...], column: str, descending: bool = True
) -> tuple[TableRow, ...]:
    """Order rows by one numeric column."""

    def key(row: TableRow) -> float:
        value = row.cells.get(column)
        if value is None:
            return float("-inf")
        return float(value)

    return tuple(sorted(rows, key=key, reverse=descending))


def summarise(values: np.ndarray) -> dict[str, float]:
    """Mean, spread and count of a value vector."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "std": 0.0, "count": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "count": float(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def seed_summary(per_seed: dict[int, float]) -> dict[str, float]:
    """Mean and standard deviation of a metric across seeds."""

    values = np.array(list(per_seed.values()), dtype=np.float64)
    summary = summarise(values)
    summary["seeds"] = float(len(per_seed))
    return summary


def stratum_column_name(composition: str, phenotype: str) -> str:
    """The column a stratum's value belongs in."""

    return f"{composition}_{phenotype}"


def parity_gap(constrained: dict[str, float], twin: dict[str, float]) -> dict[str, float]:
    """Signed utility gap of the constrained variant against its unconstrained twin."""

    shared = sorted(set(constrained) & set(twin))
    return {metric: constrained[metric] - twin[metric] for metric in shared}


def mcid_crossing_column(individual_increments: np.ndarray, reference: float) -> float:
    """Share of records whose individual increment reaches the reference level."""

    array = np.asarray(individual_increments, dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.mean(array >= reference))
