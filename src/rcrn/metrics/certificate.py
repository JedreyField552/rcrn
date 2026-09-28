"""Certified top-k prioritisation.

Ref: Eq. (5); Algorithm 1; Methods, "Certified top-k prioritisation".

    min_{i in K, j in K^c} ( S(i|p) - S(j|p) )  >  2 L delta_adm

with L = L_enc ||s_i||_2 per molecule. The certificate is an arithmetic statement
about the returned set: whenever the margin clears the radius, no admissible
perturbation of the plaque input can change the set. Records whose margin falls
inside the radius are deferred to the plaque-blind order and counted in the
deferral rate, which is the honest cost of the guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFERRAL_POLICIES: tuple[str, ...] = ("plaque_blind_order", "drop_from_list")


@dataclass(slots=True)
class CertifiedSet:
    """The returned top-k set, its margin and the entries placed in deferral."""

    record_id: str
    top_k: tuple[int, ...]
    deferred: tuple[int, ...]
    margin: float
    radius: float
    certified: bool
    plaque_blind_order: tuple[int, ...]
    deferral_policy: str = "plaque_blind_order"

    def deferral_fraction(self, depth: int) -> float:
        if depth <= 0:
            return 0.0
        return len(self.deferred) / depth


@dataclass(slots=True)
class CertificateReport:
    """Aggregate certificate outcome over a cohort of plaque records."""

    sets: tuple[CertifiedSet, ...]
    depth: int
    deferral_rate: float
    certified_share: float
    admissible_perturbation: float

    def deferred_total(self) -> int:
        return sum(len(item.deferred) for item in self.sets)


def molecule_radius(susceptibility: np.ndarray, encoder_lipschitz: float) -> np.ndarray:
    """Per-molecule Lipschitz constant L = L_enc ||s_i||_2 of Algorithm 1 step 4."""

    if susceptibility.ndim != 2:
        raise ValueError("expected a (candidate, route) tensor")
    return float(encoder_lipschitz) * np.linalg.norm(susceptibility, axis=1)


def certified_top_k(
    record_id: str,
    scores: np.ndarray,
    susceptibility: np.ndarray,
    depth: int,
    admissible_perturbation: float,
    encoder_lipschitz: float,
    deferral_policy: str = "plaque_blind_order",
) -> CertifiedSet:
    """Run Algorithm 1 for one plaque record."""

    if deferral_policy not in DEFERRAL_POLICIES:
        raise ValueError(f"unknown deferral policy {deferral_policy}")
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("expected one score per candidate")
    if not 0 < depth < values.size:
        raise ValueError("depth must lie strictly inside the candidate list")
    order = tuple(int(index) for index in np.argsort(-values, kind="stable"))
    top = order[:depth]
    remainder = order[depth:]
    margin = float(min(values[i] - values[j] for i in top for j in remainder))
    radii = molecule_radius(susceptibility, encoder_lipschitz)
    radius = float(2.0 * admissible_perturbation * float(np.max(radii)))
    blind_order = tuple(int(index) for index in np.argsort(-values, kind="stable"))
    if margin > radius:
        return CertifiedSet(
            record_id=record_id,
            top_k=top,
            deferred=(),
            margin=margin,
            radius=radius,
            certified=True,
            plaque_blind_order=blind_order,
            deferral_policy=deferral_policy,
        )
    deferred = tuple(
        index for index in top if min(values[index] - values[j] for j in remainder) <= radius
    )
    return CertifiedSet(
        record_id=record_id,
        top_k=top,
        deferred=deferred,
        margin=margin,
        radius=radius,
        certified=False,
        plaque_blind_order=blind_order,
        deferral_policy=deferral_policy,
    )


def certified_batch(
    record_ids: tuple[str, ...],
    scores: np.ndarray,
    susceptibility: np.ndarray,
    depth: int,
    admissible_perturbation: float,
    encoder_lipschitz: float,
    deferral_policy: str = "plaque_blind_order",
) -> CertificateReport:
    """Run Algorithm 1 over a batch of plaque records and aggregate the cost."""

    if scores.shape[0] != len(record_ids):
        raise ValueError("score rows and identifiers differ")
    sets = tuple(
        certified_top_k(
            record_ids[position],
            scores[position],
            susceptibility,
            depth,
            admissible_perturbation,
            encoder_lipschitz,
            deferral_policy,
        )
        for position in range(len(record_ids))
    )
    record_count = max(len(record_ids), 1)
    deferral_rate = sum(len(item.deferred) for item in sets) / (depth * record_count)
    certified_share = sum(1 for item in sets if item.certified) / record_count
    return CertificateReport(
        sets=sets,
        depth=depth,
        deferral_rate=float(deferral_rate),
        certified_share=float(certified_share),
        admissible_perturbation=float(admissible_perturbation),
    )


def flip_rate_under_perturbation(
    clean_scores: np.ndarray,
    perturbed_scores: np.ndarray,
    depth: int,
) -> float:
    """Share of top-k entries that changed after an input perturbation.

    The sweep in the discussion compares this empirical flip rate against the
    certified radius; it is the quantity the certificate is supposed to bound.
    """

    if clean_scores.shape != perturbed_scores.shape:
        raise ValueError("score matrices must be aligned")
    if clean_scores.ndim != 2:
        raise ValueError("expected a (record, candidate) score matrix")
    flips = 0
    total = 0
    for row in range(clean_scores.shape[0]):
        clean_top = set(
            int(index) for index in np.argsort(-clean_scores[row], kind="stable")[:depth]
        )
        perturbed_top = set(
            int(index) for index in np.argsort(-perturbed_scores[row], kind="stable")[:depth]
        )
        flips += len(clean_top - perturbed_top)
        total += depth
    if total == 0:
        return 0.0
    return flips / total


def expected_deferral_rate(report: CertificateReport, depth: int) -> float:
    """Deferral rate recomputed from the returned sets, as Algorithm 1 step 7 defines it."""

    total_pairs = depth * max(len(report.sets), 1)
    return report.deferred_total() / total_pairs
