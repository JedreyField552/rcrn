"""Reader for the authorised cohort layer.

Ref: Methods, "Cohort construction and auxiliary data"; Data availability, Layer (i).

The cohort is held under a data-usage statement and is not redistributed. This
module reads a derived table that an authorised holder places on disk; it never
downloads, and it never writes a subject-level record back out. The field
contract below is what the manuscript's provenance record describes.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rcrn.data.schemas import (
    AcquisitionMeta,
    CerebrovascularOutcome,
    CompositionGrade,
    Phenotype,
    PlaqueCohort,
    PlaqueRecord,
    Region,
    StudyArm,
)

REQUIRED_FIELDS: tuple[str, ...] = (
    "record_id",
    "patient_id",
    "site",
    "region",
    "arm",
    "composition_grade",
    "phenotypes",
    "plaque_volume_mm3",
    "reader_kappa",
    "channels",
    "slice_thickness_mm",
    "motion_grade",
    "vendor",
    "accrual_index",
    "follow_up_days",
    "event_observed",
    "hazard",
)

PROHIBITED_FIELDS: tuple[str, ...] = (
    "patient_name",
    "medical_record_number",
    "date_of_birth",
    "re_identification_key",
    "site_name",
)


@dataclass(slots=True)
class CohortProvenance:
    """What the reader was given and under which terms."""

    source: str
    access_basis: str
    record_count: int
    sites: tuple[str, ...]
    regions: tuple[str, ...]
    volume_directory: str | None

    def as_record(self) -> dict[str, object]:
        return {
            "source": self.source,
            "access_basis": self.access_basis,
            "record_count": self.record_count,
            "sites": list(self.sites),
            "regions": list(self.regions),
            "volume_directory": self.volume_directory,
            "distributed": False,
        }


class CohortFieldError(ValueError):
    """Raised when a derived table carries a field the contract forbids or omits."""


def _validate_header(header: list[str]) -> None:
    missing = [name for name in REQUIRED_FIELDS if name not in header]
    if missing:
        raise CohortFieldError(f"missing required field {missing[0]}")
    leaked = [name for name in PROHIBITED_FIELDS if name in header]
    if leaked:
        raise CohortFieldError(f"prohibited identifying field present: {leaked[0]}")


def _parse_phenotypes(raw: str) -> frozenset[Phenotype]:
    parts = [part.strip() for part in raw.replace(";", "|").split("|") if part.strip()]
    allowed = {
        "iph_positive",
        "thin_fibrous_cap",
        "lipid_rich_necrotic_core",
        "symptomatic",
        "asymptomatic",
    }
    unknown = [part for part in parts if part not in allowed]
    if unknown:
        raise CohortFieldError(f"unknown phenotype {unknown[0]}")
    return frozenset(parts)


def _parse_record(row: dict[str, str]) -> PlaqueRecord:
    grade: CompositionGrade = (
        "rads_3_4" if row["composition_grade"] in {"3", "4", "rads_3_4"} else "rads_1_2"
    )
    region: Region = "II" if row["region"].upper() in {"II", "2"} else "I"
    arm: StudyArm = "prospective" if row["arm"].lower().startswith("pros") else "retrospective"
    channels = tuple(
        part.strip() for part in row["channels"].replace(";", "|").split("|") if part.strip()
    )
    acquisition = AcquisitionMeta(
        channels=channels,
        slice_thickness_mm=float(row["slice_thickness_mm"]),
        motion_grade=float(row["motion_grade"]),
        vendor=row["vendor"].strip(),
        accrual_index=int(row["accrual_index"]),
    )
    outcome = CerebrovascularOutcome(
        follow_up_days=float(row["follow_up_days"]),
        event_observed=row["event_observed"].strip().lower() in {"1", "true", "yes"},
        hazard=float(row["hazard"]),
        implied_rank=float(row["hazard"]),
    )
    return PlaqueRecord(
        record_id=row["record_id"],
        patient_id=row["patient_id"],
        site=row["site"].strip(),
        region=region,
        arm=arm,
        composition_grade=grade,
        phenotypes=_parse_phenotypes(row["phenotypes"]),
        plaque_volume_mm3=float(row["plaque_volume_mm3"]),
        reader_kappa=float(row["reader_kappa"]),
        acquisition=acquisition,
        outcome=outcome,
        composition_index=float(row["composition_index"]) if row.get("composition_index") else None,
    )


def load_authorised_cohort(
    path: str | Path, volume_directory: str | Path | None = None
) -> PlaqueCohort:
    """Read a derived cohort table. Absent file means no cohort is available."""

    table = Path(path)
    if not table.is_file():
        raise FileNotFoundError(table)
    with table.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        _validate_header(header)
        records = [_parse_record(row) for row in reader]
    if not records:
        raise CohortFieldError("cohort table carries no records")
    if volume_directory is not None:
        volumes = load_volume_directory(volume_directory)
        for record in records:
            record.volume = volumes.get(record.record_id)
    return PlaqueCohort(records=records)


def load_implied_orders(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Read the outcome-implied ordering of the candidate list for each record.

    The ordering is produced by the label and trajectory pipeline, which is a
    separate script from the feature pipeline; it is read here rather than derived
    from any feature the model sees.
    """

    table = Path(path)
    if not table.is_file():
        raise FileNotFoundError(table)
    orders: dict[str, tuple[str, ...]] = {}
    with table.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        missing = [name for name in ("record_id", "candidate_ids") if name not in header]
        if missing:
            raise CohortFieldError(f"ordering table is missing column {missing[0]}")
        for row in reader:
            candidates = tuple(
                part.strip()
                for part in row["candidate_ids"].replace(";", "|").split("|")
                if part.strip()
            )
            if not candidates:
                raise CohortFieldError(f"record {row['record_id']} carries an empty ordering")
            orders[row["record_id"]] = candidates
    return orders


def load_volume_directory(path: str | Path) -> dict[str, np.ndarray]:
    """Read assembled plaque volumes, one compressed array per record identifier."""

    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    volumes: dict[str, np.ndarray] = {}
    for candidate in sorted(directory.glob("*.npy")):
        volumes[candidate.stem] = np.load(candidate)
    return volumes


def provenance_of(
    cohort: PlaqueCohort, source: str, access_basis: str, volume_directory: str | None = None
) -> CohortProvenance:
    """Summarise the cohort without exposing any subject-level field."""

    return CohortProvenance(
        source=source,
        access_basis=access_basis,
        record_count=len(cohort.records),
        sites=tuple(sorted({record.site for record in cohort.records})),
        regions=tuple(sorted({record.region for record in cohort.records})),
        volume_directory=volume_directory,
    )


def write_provenance(provenance: CohortProvenance, destination: str | Path) -> None:
    """Persist the provenance record as JSON."""

    payload = provenance.as_record()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    Path(destination).write_text(text, encoding="utf-8")
