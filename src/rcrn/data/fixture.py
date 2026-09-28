"""Schema-compatible cohort fixture.

Ref: Methods, "Cohort construction and auxiliary data"; Methods, "The plaque
composition encoder"; Algorithm 1 (the pair score a_i - <s_i, r(x_p)>).

The private cohort is not distributed. This module builds a fixture that carries
the same schema and the same relationships the study design assumes, so every
mechanism in the release stays executable. The fixture serves execution checks
only, and no cohort-level value in the manuscript is produced from it.

The fixture exposes its own closed form. A plaque's route resistances drive the
channel intensities of its volume, a candidate's descriptor block drives its route
susceptibility, and the observed ordering is the reaction norm of the two plus a
recorded observation noise. That is what gives the execution checks a target they
can be wrong about; the noise is set so discrimination is well below its ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

from rcrn.data.descriptors import descriptor_vector
from rcrn.data.molecules import MoleculeGraph, featurise_graph
from rcrn.data.schemas import (
    ACQUISITION_CHANNELS,
    AcquisitionMeta,
    CerebrovascularOutcome,
    CompositionGrade,
    MoleculeRecord,
    Phenotype,
    PlaqueCohort,
    PlaqueRecord,
    Region,
    StudyArm,
)
from rcrn.utils.config import Config

ORGANIC_WEIGHTS: Final[tuple[str, ...]] = ("C", "C", "C", "C", "N", "O", "S", "F", "Cl")
ROUTE_LOADINGS: Final[np.ndarray] = np.array(
    [
        [0.62, 0.18, 0.20],
        [0.16, 0.66, 0.18],
        [0.20, 0.16, 0.64],
    ],
    dtype=np.float64,
)
SUSCEPTIBILITY_MAP: Final[np.ndarray] = np.array(
    [
        [0.9, -0.4, 0.2, 0.1],
        [-0.3, 0.8, 0.1, 0.35],
        [0.25, 0.15, -0.85, -0.2],
    ],
    dtype=np.float64,
)
DIRECT_MAP: Final[np.ndarray] = np.array([0.6, -0.35, 0.45, 0.2], dtype=np.float64)


@dataclass(slots=True)
class FixtureBundle:
    """The fixture cohort together with the closed form that produced it."""

    cohort: PlaqueCohort
    volumes: dict[str, np.ndarray]
    candidates: tuple[MoleculeRecord, ...]
    candidate_ids: tuple[str, ...]
    true_resistance: dict[str, np.ndarray] = field(default_factory=dict)
    true_composition: dict[str, float] = field(default_factory=dict)
    true_susceptibility: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    true_direct_effect: np.ndarray = field(default_factory=lambda: np.zeros(0))
    implied_order: dict[str, tuple[str, ...]] = field(default_factory=dict)
    observation_noise: float = 0.0

    def candidate_index(self) -> dict[str, int]:
        return {molecule_id: index for index, molecule_id in enumerate(self.candidate_ids)}

    def true_score(self, record_id: str) -> np.ndarray:
        resistance = self.true_resistance[record_id]
        return self.true_direct_effect - self.true_susceptibility @ resistance


def molecule_fixture_graph(index: int, rng: np.random.Generator) -> MoleculeGraph:
    """Grow a connected organic-subset graph, so the fixture needs no external structures."""

    atom_target = int(rng.integers(8, 22))
    elements: list[str] = ["C"]
    bond_ends: list[tuple[int, int]] = []
    bond_orders: list[float] = []
    aromatic_flags: list[bool] = []
    while len(elements) < atom_target:
        anchor = int(rng.integers(0, len(elements)))
        element = ORGANIC_WEIGHTS[int(rng.integers(0, len(ORGANIC_WEIGHTS)))]
        elements.append(element)
        bond_ends.append((anchor, len(elements) - 1))
        order = float(rng.choice([1.0, 1.0, 1.0, 2.0]))
        bond_orders.append(order)
        aromatic_flags.append(False)
    for _ in range(int(rng.integers(1, 3))):
        left = int(rng.integers(0, len(elements)))
        right = int(rng.integers(0, len(elements)))
        if left == right or (min(left, right), max(left, right)) in {
            (min(a, b), max(a, b)) for a, b in bond_ends
        }:
            continue
        bond_ends.append((min(left, right), max(left, right)))
        bond_orders.append(1.0)
        aromatic_flags.append(True)
    return MoleculeGraph(
        elements=tuple(elements),
        bond_ends=tuple(bond_ends),
        bond_orders=tuple(bond_orders),
        bond_aromatic=tuple(aromatic_flags),
        bond_stereo=tuple("" for _ in bond_ends),
        charges=tuple(0 for _ in elements),
        explicit_hydrogens=tuple(-1 for _ in elements),
    )


def _route_susceptibility(descriptors: np.ndarray) -> np.ndarray:
    reduced = np.array(
        [
            np.log1p(abs(descriptors[0])),
            descriptors[12],
            descriptors[4],
            descriptors[2] + descriptors[3],
        ],
        dtype=np.float64,
    )
    reduced = reduced / (np.linalg.norm(reduced) + 1e-9)
    raw = SUSCEPTIBILITY_MAP @ reduced
    return np.abs(raw) + 1e-3


def _direct_effect(descriptors: np.ndarray) -> float:
    reduced = np.array(
        [
            np.log1p(abs(descriptors[0])) / 4.0,
            descriptors[13],
            descriptors[5] / 4.0,
            descriptors[8] / 4.0,
        ],
        dtype=np.float64,
    )
    return float(DIRECT_MAP @ reduced)


def _volume_from_routes(
    resistance: np.ndarray,
    composition: float,
    shape: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Channel intensities are a monotone read-out of the route resistances."""

    _, depth, height, width = shape
    channels = len(ACQUISITION_CHANNELS)
    volume = np.zeros((channels, depth, height, width), dtype=np.float32)
    coords = np.linspace(-1.0, 1.0, height)
    rows = np.linspace(-1.0, 1.0, width)
    blob = np.exp(-3.0 * (coords[:, None] ** 2 + rows[None, :] ** 2))
    core = np.exp(-9.0 * (coords[:, None] ** 2 + rows[None, :] ** 2))
    for channel in range(channels):
        level = 0.6 + 0.9 * float(resistance[channel % resistance.shape[0]])
        profile = level * blob + 0.5 * composition * core
        depth_profile = np.exp(-2.0 * (np.linspace(-1.0, 1.0, depth) ** 2))
        volume[channel] = (
            profile[None, :, :] * depth_profile[:, None, None]
            + rng.normal(0.0, 0.02, size=(depth, height, width))
        ).astype(np.float32)
        volume[channel] = np.clip(volume[channel], 0.0, None)
    return volume


def build_cohort_fixture(config: Config) -> FixtureBundle:
    """Build the fixture cohort, its volumes, its candidate list and its closed form."""

    seed = int(config.require("data.fixture_seed"))
    record_count = int(config.require("data.records"))
    molecule_count = int(config.require("data.molecule_count"))
    shape = tuple(int(size) for size in config.require("data.volume_shape"))
    rng = np.random.default_rng(seed)
    sites = tuple(str(site) for site in config.require("data.sites"))
    resistance_lo, resistance_hi = (float(v) for v in config.require("data.route_resistance_scale"))
    composition_lo, composition_hi = (
        float(v) for v in config.require("data.composition_index_range")
    )
    kappa_lo, kappa_hi = (float(v) for v in config.require("data.reader_kappa_range"))
    hazard_scale = float(config.require("data.outcome_hazard_scale"))
    volume_lo, volume_hi = (float(v) for v in config.require("data.plaque_volume_mm3"))
    dispersion = float(config.require("data.molecule_dispersion"))

    candidate_ids = tuple(f"CM-{index:03d}" for index in range(molecule_count))
    candidates: list[MoleculeRecord] = []
    descriptor_rows: list[np.ndarray] = []
    for index in range(molecule_count):
        graph = molecule_fixture_graph(index, rng)
        record = featurise_graph(graph)
        rows = descriptor_vector(graph)
        candidates.append(
            MoleculeRecord(
                molecule_id=candidate_ids[index],
                atom_features=record.atom_features,
                bond_index=record.bond_index,
                bond_features=record.bond_features,
                fragment_index=record.fragment_index,
                descriptors=rows.astype(np.float32),
                auxiliary_permeability=_direct_effect(rows) + float(rng.normal(0.0, 0.25)),
            )
        )
        descriptor_rows.append(rows)
    descriptors = np.stack(descriptor_rows, axis=0)
    direct_effect = np.array([_direct_effect(row) for row in descriptors], dtype=np.float64)
    susceptibility = np.stack([_route_susceptibility(row) for row in descriptors], axis=0)
    susceptibility = susceptibility / np.linalg.norm(susceptibility, axis=1, keepdims=True)
    if dispersion < 1.0:
        anchor = susceptibility.mean(axis=0, keepdims=True)
        susceptibility = anchor + dispersion * (susceptibility - anchor)
        susceptibility = susceptibility / np.linalg.norm(susceptibility, axis=1, keepdims=True)

    records: list[PlaqueRecord] = []
    volumes: dict[str, np.ndarray] = {}
    true_resistance: dict[str, np.ndarray] = {}
    true_composition: dict[str, float] = {}
    implied_order: dict[str, tuple[str, ...]] = {}
    for index in range(record_count):
        site = sites[(index // 2) % len(sites)]
        region: Region = "II" if site == sites[-1] else "I"
        arm: StudyArm = "prospective" if index % 4 == 0 else "retrospective"
        composition = float(rng.uniform(composition_lo, composition_hi))
        grade: CompositionGrade = "rads_3_4" if composition > 0.45 else "rads_1_2"
        phenotypes: set[Phenotype] = set()
        if grade == "rads_3_4":
            phenotypes.add("iph_positive" if composition > 0.6 else "thin_fibrous_cap")
        else:
            phenotypes.add("lipid_rich_necrotic_core")
        phenotypes.add("symptomatic" if rng.random() < 0.5 else "asymptomatic")
        phenotype_score = 1.0 if "iph_positive" in phenotypes else 0.45
        site_shift = {"A": 0.0, "B": 0.12, "C": -0.16}.get(site, 0.0)
        route_load = ROUTE_LOADINGS @ np.array([composition, phenotype_score, site_shift + 0.5])
        resistance = np.clip(
            resistance_lo
            + (resistance_hi - resistance_lo)
            * (route_load / (route_load.sum() + 1e-9) * 3.0 + rng.normal(0.0, 0.05, size=3)),
            1e-3,
            None,
        )
        volume_mm3 = float(volume_lo + (volume_hi - volume_lo) * composition)
        channels = ACQUISITION_CHANNELS
        if site == sites[-1] and index % 5 == 0:
            channels = tuple(name for name in channels if name != "t1w")
        slice_thickness = 1.5 if index % 7 else 3.0
        motion = 0.0 if index % 6 else 0.35
        acquisition = AcquisitionMeta(
            channels=channels,
            slice_thickness_mm=slice_thickness,
            motion_grade=motion,
            vendor="A" if site != "B" else "B",
            accrual_index=index,
        )
        record_id = f"PLQ-{index:04d}"
        pathology = 0.0 if "iph_positive" not in phenotypes else 1.0
        hazard = float(hazard_scale * (composition + pathology) + rng.normal(0.0, 0.08))
        outcome = CerebrovascularOutcome(
            follow_up_days=float(rng.uniform(180.0, 1450.0)),
            event_observed=bool(hazard > 0.55),
            hazard=hazard,
            implied_rank=float(hazard),
        )
        record = PlaqueRecord(
            record_id=record_id,
            patient_id=f"PT-{index // 2:04d}",
            site=site,
            region=region,
            arm=arm,
            composition_grade=grade,
            phenotypes=frozenset(phenotypes),
            plaque_volume_mm3=volume_mm3,
            reader_kappa=float(rng.uniform(kappa_lo, kappa_hi)),
            acquisition=acquisition,
            outcome=outcome,
            composition_index=composition,
        )
        records.append(record)
        volumes[record_id] = _volume_from_routes(resistance, composition, shape, rng)
        true_resistance[record_id] = resistance
        true_composition[record_id] = composition
        clean = direct_effect - susceptibility @ resistance
        observed = clean + rng.normal(0.0, 0.55, size=clean.shape)
        implied_order[record_id] = tuple(
            candidate_ids[position] for position in np.argsort(-observed, kind="stable").tolist()
        )

    return FixtureBundle(
        cohort=PlaqueCohort(records=records),
        volumes=volumes,
        candidates=tuple(candidates),
        candidate_ids=candidate_ids,
        true_resistance=true_resistance,
        true_composition=true_composition,
        true_susceptibility=susceptibility,
        true_direct_effect=direct_effect,
        implied_order=implied_order,
        observation_noise=0.55,
    )


def bundle_from_authorised(
    cohort: PlaqueCohort,
    candidates: tuple[MoleculeRecord, ...],
    volumes: dict[str, np.ndarray],
    implied_order: dict[str, tuple[str, ...]],
) -> FixtureBundle:
    """Wrap an authorised cohort in the bundle the pipeline consumes.

    Nothing is built here: the volumes, the candidate graphs and the
    outcome-implied orderings all come from the authorised inputs, and the
    fixture-only fields are left empty so a cohort-level value can never be
    mistaken for a fixture derivation.
    """

    missing_orders = [
        record.record_id for record in cohort.records if record.record_id not in implied_order
    ]
    if missing_orders:
        raise ValueError(f"no implied ordering for record {missing_orders[0]}")
    identifier_set = {record.molecule_id for record in candidates}
    for record_id, order in implied_order.items():
        unknown = [molecule_id for molecule_id in order if molecule_id not in identifier_set]
        if unknown:
            raise ValueError(f"record {record_id} orders unknown candidate {unknown[0]}")
    return FixtureBundle(
        cohort=cohort,
        volumes={record.record_id: volumes[record.record_id] for record in cohort.records},
        candidates=candidates,
        candidate_ids=tuple(record.molecule_id for record in candidates),
        implied_order=implied_order,
    )


def fixture_candidate_structures(bundle: FixtureBundle) -> tuple[str, ...]:
    """Candidate identifiers in the fixed order used by every per-plaque ordering."""

    return bundle.candidate_ids


def route_share(resistance: np.ndarray) -> np.ndarray:
    """Share of the resistance carried by each transport route."""

    total = float(resistance.sum())
    if total <= 0.0:
        return np.full(resistance.shape, 1.0 / resistance.shape[0])
    return resistance / total
