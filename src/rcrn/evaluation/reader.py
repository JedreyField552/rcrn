"""Reader-study statistics.

Ref: Results, Section 2.6; Methods, "Evaluation protocol and statistical analysis"
(six readers of declared expertise, two readings under identical blinded
conditions with a washout interval, and a second study comparing the model alone,
the reader alone and the reader with the model).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcrn.metrics.agreement import fleiss_kappa, weighted_kappa


@dataclass(slots=True)
class ReaderReading:
    """One reader's ordering of one record, with and without the prioritisation."""

    reader: str
    expertise: str
    record_id: str
    unassisted: np.ndarray
    assisted: np.ndarray


@dataclass(slots=True)
class ReaderStudy:
    """All readings of one human-factors study."""

    readings: tuple[ReaderReading, ...]
    depth: int

    def readers(self) -> tuple[str, ...]:
        return tuple(sorted({reading.reader for reading in self.readings}))

    def expertise_levels(self) -> tuple[str, ...]:
        return tuple(sorted({reading.expertise for reading in self.readings}))


def pairwise_agreement(readings: tuple[ReaderReading, ...], assisted: bool) -> np.ndarray:
    """Agreement matrix between every pair of readers on the shared records."""

    readers = sorted({reading.reader for reading in readings})
    lookup: dict[tuple[str, str], np.ndarray] = {}
    for reading in readings:
        order = reading.assisted if assisted else reading.unassisted
        lookup[(reading.reader, reading.record_id)] = top_k_levels(order)
    levels = max(2, max(len(value) for value in lookup.values()))
    matrix = np.full((len(readers), len(readers)), np.nan, dtype=np.float64)
    for left in range(len(readers)):
        for right in range(left + 1, len(readers)):
            shared = sorted(
                record
                for reader, record in lookup
                if reader == readers[left] and (readers[right], record) in lookup
            )
            if not shared:
                continue
            first = np.concatenate([lookup[(readers[left], record)] for record in shared])
            second = np.concatenate([lookup[(readers[right], record)] for record in shared])
            matrix[left, right] = weighted_kappa(first, second, levels=levels)
            matrix[right, left] = matrix[left, right]
    return matrix


def top_k_levels(order: np.ndarray) -> np.ndarray:
    """Bucket an ordering into rank levels, so a graded agreement can be computed."""

    ranks = np.empty_like(np.asarray(order), dtype=np.int64)
    for position, index in enumerate(np.asarray(order).tolist()):
        ranks[index] = position
    return ranks


def mean_pairwise_agreement(readings: tuple[ReaderReading, ...], assisted: bool) -> float:
    """Average inter-reader agreement, ignoring the self-pairs."""

    matrix = pairwise_agreement(readings, assisted)
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    finite = upper[np.isfinite(upper)]
    if finite.size == 0:
        return 0.0
    return float(finite.mean())


def agreement_ladder(readings: tuple[ReaderReading, ...]) -> tuple[float, float]:
    """Unassisted and assisted agreement, in that order."""

    return (
        mean_pairwise_agreement(readings, assisted=False),
        mean_pairwise_agreement(readings, assisted=True),
    )


def expertise_stratum_effect(readings: tuple[ReaderReading, ...], expertise: str) -> float:
    """Assisted-minus-unassisted agreement restricted to one expertise level."""

    subset = tuple(reading for reading in readings if reading.expertise == expertise)
    if not subset:
        return 0.0
    before, after = agreement_ladder(subset)
    return after - before


def ranking_study_comparison(
    consensus_ordering: np.ndarray,
    model_ordering: np.ndarray,
    reader_ordering: np.ndarray,
    target_ordering: np.ndarray,
) -> dict[str, float]:
    """Agreement with the outcome-implied order for the three arms of the study."""

    from rcrn.metrics.ranking import spearman_rho

    return {
        "multi_reader_consensus": spearman_rho(consensus_ordering, target_ordering),
        "model_alone": spearman_rho(model_ordering, target_ordering),
        "reader_with_model": spearman_rho(reader_ordering, target_ordering),
    }


def fleiss_from_readings(readings: tuple[ReaderReading, ...]) -> float:
    """Fleiss' kappa over the rank buckets the readers assigned."""

    records = sorted({reading.record_id for reading in readings})
    readers = sorted({reading.reader for reading in readings})
    levels = max(2, max(len(reading.unassisted) for reading in readings))
    table = np.zeros((len(records), levels), dtype=np.float64)
    for record_position, record in enumerate(records):
        for reading in readings:
            if reading.record_id != record:
                continue
            ranks = top_k_levels(reading.unassisted)
            bucket = int(np.clip(ranks.mean(), 0, levels - 1))
            table[record_position, bucket] += 1.0
    if not np.allclose(table.sum(axis=1), len(readers)):
        return 0.0
    return fleiss_kappa(table)
