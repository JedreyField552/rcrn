"""Design invariants the study depends on.

Ref: Methods, "Cohort construction and auxiliary data" (site-proxy features are on
the prohibited list, normalisation statistics are fitted on the training partition,
and the label pipeline is independent of the feature pipeline); Results, Section 2.4
(the composition index is derived from imaging alone, which is what stops the
composition strata from being self-fulfilling).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcrn.data.schemas import AcquisitionMeta


@dataclass(slots=True)
class InvariantOutcome:
    """Whether one invariant held, with the evidence that decided it."""

    name: str
    holds: bool
    evidence: str

    def as_record(self) -> dict[str, object]:
        return {"name": self.name, "holds": self.holds, "evidence": self.evidence}


def prohibited_features_absent(
    feature_names: tuple[str, ...], prohibited: tuple[str, ...]
) -> InvariantOutcome:
    """No site-identity or accrual proxy may reach the model."""

    present = [name for name in feature_names if any(token in name for token in prohibited)]
    return InvariantOutcome(
        name="prohibited_features_absent",
        holds=not present,
        evidence=f"checked {len(feature_names)} feature names",
    )


def normalisation_fitted_on_training(
    fitted_ids: frozenset[str], training_ids: frozenset[str]
) -> InvariantOutcome:
    """Normalisation statistics may only be fitted on the training partition."""

    leaked = fitted_ids - training_ids
    return InvariantOutcome(
        name="normalisation_fitted_on_training",
        holds=not leaked,
        evidence=f"{len(leaked)} of {len(fitted_ids)} fitted records lie outside the training partition",
    )


def composition_index_is_image_only(
    composition: np.ndarray, outcome: np.ndarray, tolerance: float = 0.999
) -> InvariantOutcome:
    """The composition index must not be a deterministic function of the outcome.

    A near-perfect correlation is treated as evidence that the index was built from
    the reporting table, which would make the composition strata self-fulfilling.
    """

    first = np.asarray(composition, dtype=np.float64)
    second = np.asarray(outcome, dtype=np.float64)
    if first.size < 3:
        return InvariantOutcome("composition_index_is_image_only", True, "too few records to test")
    correlation = float(np.corrcoef(first, second)[0, 1])
    holds = abs(correlation) < tolerance
    return InvariantOutcome(
        name="composition_index_is_image_only",
        holds=holds,
        evidence=f"|corr| with the outcome trend is {abs(correlation):.4f}",
    )


def label_pipeline_independent(
    feature_sources: tuple[str, ...], label_sources: tuple[str, ...]
) -> InvariantOutcome:
    """The label pipeline and the feature pipeline must share no input."""

    shared = sorted(set(feature_sources) & set(label_sources))
    return InvariantOutcome(
        name="label_pipeline_independent",
        holds=not shared,
        evidence=f"shared inputs: {shared if shared else 'none'}",
    )


def external_fold_unfitted(
    external_ids: frozenset[str], fitted_ids: frozenset[str]
) -> InvariantOutcome:
    """The held-out site and region must enter no fitting or selection step."""

    overlap = external_ids & fitted_ids
    return InvariantOutcome(
        name="external_fold_unfitted",
        holds=not overlap,
        evidence=f"{len(overlap)} records appear in both partitions",
    )


def acquisition_channels_declared(metas: tuple[AcquisitionMeta, ...]) -> InvariantOutcome:
    """Every record must declare which contrast channels it actually carries."""

    incomplete = [meta.channels for meta in metas if not meta.channels]
    return InvariantOutcome(
        name="acquisition_channels_declared",
        holds=not incomplete,
        evidence=f"{len(incomplete)} records carry no channel list",
    )


def collect_invariants(outcomes: tuple[InvariantOutcome, ...]) -> dict[str, object]:
    """Summarise a set of invariant outcomes."""

    failing = [outcome.name for outcome in outcomes if not outcome.holds]
    return {
        "checked": len(outcomes),
        "holding": len(outcomes) - len(failing),
        "failing": failing,
        "all_hold": not failing,
    }
