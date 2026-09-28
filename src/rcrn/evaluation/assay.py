"""Assay-endpoint comparison.

Ref: Results, Section 2.4; Discussion ("the representational quality is a relation
between the architecture and the endpoint, not a trait of the architecture").

The comparison reports the mean absolute error per descriptor or architecture
family on the plaque-conditioned endpoint and on the assay endpoint, and states the
direction of the reversal rather than asserting a general superiority.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcrn.data.auxiliary import assay_penalty, rank_families_by_error

FAMILIES: tuple[str, ...] = (
    "descriptor_gradient_boosting",
    "graph_transformer",
    "state_space",
    "equivariant_geometry",
    "pretrained_language_model",
)


@dataclass(slots=True)
class FamilyError:
    """Mean absolute error of one family on one endpoint."""

    family: str
    endpoint: str
    mean_absolute_error: float

    def as_record(self) -> dict[str, float | str]:
        return {
            "family": self.family,
            "endpoint": self.endpoint,
            "mean_absolute_error": self.mean_absolute_error,
        }


def error_table(rows: tuple[FamilyError, ...], endpoint: str) -> tuple[FamilyError, ...]:
    """Rows of one endpoint, ordered from the lowest error upwards."""

    subset = [row for row in rows if row.endpoint == endpoint]
    ordered = rank_families_by_error({row.family: row.mean_absolute_error for row in subset})
    return tuple(FamilyError(family, endpoint, value) for family, value in ordered)


def best_family(rows: tuple[FamilyError, ...], endpoint: str) -> FamilyError:
    """The family with the lowest mean absolute error on an endpoint."""

    table = error_table(rows, endpoint)
    if not table:
        raise ValueError(f"no rows for endpoint {endpoint}")
    return table[0]


def endpoint_reversal(
    rows: tuple[FamilyError, ...],
    left_family: str,
    right_family: str,
    penetration_endpoint: str,
    assay_endpoint: str,
    penetration_lower_is_better: bool = False,
    assay_lower_is_better: bool = True,
) -> dict[str, float | bool]:
    """Whether a family that leads on one endpoint trails on the other.

    The two endpoints do not share an orientation: the penetration column is a rank
    agreement, where larger is better, while the assay column is an error, where
    smaller is better. The caller states the orientation of each endpoint rather
    than the function assuming one.
    """

    lookup = {(row.family, row.endpoint): row.mean_absolute_error for row in rows}
    left_penetration = lookup[(left_family, penetration_endpoint)]
    right_penetration = lookup[(right_family, penetration_endpoint)]
    left_assay = lookup[(left_family, assay_endpoint)]
    right_assay = lookup[(right_family, assay_endpoint)]

    def leads(left: float, right: float, lower_is_better: bool) -> bool:
        return left < right if lower_is_better else left > right

    penetration_order = leads(left_penetration, right_penetration, penetration_lower_is_better)
    assay_order = leads(left_assay, right_assay, assay_lower_is_better)
    return {
        "left_leads_penetration": penetration_order,
        "left_leads_assay": assay_order,
        "reversed": bool(penetration_order and not assay_order),
        "penetration_gap": abs(left_penetration - right_penetration),
        "assay_gap": abs(left_assay - right_assay),
    }


def assay_inflation(in_distribution: float, out_of_distribution: float) -> float:
    """Error inflation when the assay release moves off distribution."""

    return assay_penalty(in_distribution, out_of_distribution)


def loss_margin(family: FamilyError, comparator: FamilyError) -> float:
    """How much worse a family is than a comparator on the same endpoint."""

    if family.endpoint != comparator.endpoint:
        raise ValueError("comparisons must share an endpoint")
    return family.mean_absolute_error - comparator.mean_absolute_error


def family_rank_lookup(rows: tuple[FamilyError, ...], endpoint: str) -> dict[str, int]:
    """Family to rank position on one endpoint, starting at one."""

    return {row.family: position + 1 for position, row in enumerate(error_table(rows, endpoint))}


def error_spread(rows: tuple[FamilyError, ...], endpoint: str) -> float:
    """Difference between the worst and the best family on one endpoint."""

    table = error_table(rows, endpoint)
    if len(table) < 2:
        return 0.0
    values = np.array([row.mean_absolute_error for row in table], dtype=np.float64)
    return float(values.max() - values.min())
