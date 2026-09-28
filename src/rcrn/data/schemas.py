"""Cohort record types.

Ref: Methods, "Cohort construction and auxiliary data"; Methods, "Problem formulation";
Table 1 stratum definitions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

CompositionGrade = Literal["rads_1_2", "rads_3_4"]
Phenotype = Literal[
    "iph_positive",
    "thin_fibrous_cap",
    "lipid_rich_necrotic_core",
    "symptomatic",
    "asymptomatic",
]
StudyArm = Literal["prospective", "retrospective"]
Region = Literal["I", "II"]

ACQUISITION_CHANNELS: tuple[str, ...] = ("tof", "t1w", "t2w", "pd")
T1_CHANNEL_INDEX = ACQUISITION_CHANNELS.index("t1w")


@dataclass(slots=True)
class AcquisitionMeta:
    """Acquisition conditions that define the transport strata of Table 3."""

    channels: tuple[str, ...] = ACQUISITION_CHANNELS
    slice_thickness_mm: float = 2.0
    motion_grade: float = 0.0
    vendor: str = "A"
    accrual_index: int = 0

    def has_t1_weighted(self) -> bool:
        return "t1w" in self.channels

    def channel_indices(self) -> tuple[int, ...]:
        return tuple(ACQUISITION_CHANNELS.index(name) for name in self.channels)


@dataclass(slots=True)
class CerebrovascularOutcome:
    """Label side of a plaque record; built independently of any feature table."""

    follow_up_days: float
    event_observed: bool
    hazard: float
    implied_rank: float


@dataclass(slots=True)
class PlaqueRecord:
    """One plaque with its multi-contrast volume, composition and outcome trend."""

    record_id: str
    patient_id: str
    site: str
    region: Region
    arm: StudyArm
    composition_grade: CompositionGrade
    phenotypes: frozenset[Phenotype]
    plaque_volume_mm3: float
    reader_kappa: float
    acquisition: AcquisitionMeta
    outcome: CerebrovascularOutcome
    volume: np.ndarray | None = None
    composition_index: float | None = None

    def is_primary(self) -> bool:
        return self.composition_grade == "rads_3_4" and "iph_positive" in self.phenotypes

    def is_reference(self) -> bool:
        return self.composition_grade == "rads_1_2"

    def stratum_keys(self) -> tuple[str, ...]:
        keys = [self.composition_grade]
        if self.is_primary():
            keys.append("rads_3_4_iph")
        keys.extend(sorted(self.phenotypes))
        keys.append(f"site_{self.site}")
        keys.append(f"region_{self.region}")
        keys.append(self.arm)
        return tuple(keys)


@dataclass(slots=True)
class MoleculeRecord:
    """One candidate anti-inflammatory small molecule."""

    molecule_id: str
    atom_features: np.ndarray
    bond_index: np.ndarray
    bond_features: np.ndarray
    fragment_index: np.ndarray
    auxiliary_permeability: float | None = None
    descriptors: np.ndarray | None = None

    def num_atoms(self) -> int:
        return int(self.atom_features.shape[0])


@dataclass(slots=True)
class CandidateList:
    """The fixed candidate list scored once against every plaque record."""

    molecule_ids: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.molecule_ids)

    def index_of(self, molecule_id: str) -> int:
        return self.molecule_ids.index(molecule_id)


@dataclass(slots=True)
class PairObservation:
    """A scored (plaque, molecule) pair with its stratum agreement weight."""

    record_id: str
    molecule_id: str
    score: float
    weight: float


@dataclass(slots=True)
class PlaqueCohort:
    """A collection of plaque records indexed by identifier."""

    records: list[PlaqueRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def by_id(self, record_id: str) -> PlaqueRecord:
        for record in self.records:
            if record.record_id == record_id:
                return record
        raise KeyError(record_id)

    def select(self, predicate: Callable[[PlaqueRecord], bool]) -> list[PlaqueRecord]:
        return [record for record in self.records if predicate(record)]

    def primary_stratum(self) -> list[PlaqueRecord]:
        return [record for record in self.records if record.is_primary()]

    def reference_stratum(self) -> list[PlaqueRecord]:
        return [record for record in self.records if record.is_reference()]

    def arm(self, arm: StudyArm) -> list[PlaqueRecord]:
        return [record for record in self.records if record.arm == arm]

    def site(self, site: str) -> list[PlaqueRecord]:
        return [record for record in self.records if record.site == site]

    def region(self, region: Region) -> list[PlaqueRecord]:
        return [record for record in self.records if record.region == region]

    def patient_ids(self) -> tuple[str, ...]:
        return tuple(sorted({record.patient_id for record in self.records}))
