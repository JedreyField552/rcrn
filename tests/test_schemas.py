"""Cohort record types."""

from __future__ import annotations

import numpy as np

from rcrn.data.schemas import (
    ACQUISITION_CHANNELS,
    AcquisitionMeta,
    CerebrovascularOutcome,
    PlaqueCohort,
    PlaqueRecord,
)


def make_record(
    record_id: str = "R1", grade: str = "rads_3_4", phenotypes: tuple[str, ...] = ("iph_positive",)
) -> PlaqueRecord:
    return PlaqueRecord(
        record_id=record_id,
        patient_id="P1",
        site="A",
        region="I",
        arm="retrospective",
        composition_grade=grade,  # type: ignore[arg-type]
        phenotypes=frozenset(phenotypes),  # type: ignore[arg-type]
        plaque_volume_mm3=120.0,
        reader_kappa=0.6,
        acquisition=AcquisitionMeta(),
        outcome=CerebrovascularOutcome(
            follow_up_days=400.0, event_observed=True, hazard=0.8, implied_rank=0.8
        ),
        volume=np.zeros((1, 2, 2, 2), dtype=np.float32),
        composition_index=0.7,
    )


def test_channel_order_is_fixed() -> None:
    assert ACQUISITION_CHANNELS == ("tof", "t1w", "t2w", "pd")
    meta = AcquisitionMeta(channels=("tof", "t2w"))
    assert meta.has_t1_weighted() is False
    assert meta.channel_indices() == (0, 2)


def test_primary_and_reference_strata() -> None:
    primary = make_record()
    reference = make_record("R2", "rads_1_2", ("lipid_rich_necrotic_core",))
    assert primary.is_primary() is True
    assert primary.is_reference() is False
    assert reference.is_primary() is False
    assert reference.is_reference() is True


def test_stratum_keys_include_site_and_region() -> None:
    keys = make_record().stratum_keys()
    assert "rads_3_4_iph" in keys
    assert "site_A" in keys
    assert "region_I" in keys
    assert "iph_positive" in keys


def test_cohort_selection_helpers() -> None:
    cohort = PlaqueCohort(
        records=[
            make_record("R1"),
            make_record("R2", "rads_1_2", ("asymptomatic",)),
            make_record("R3"),
        ]
    )
    assert len(cohort) == 3
    assert [record.record_id for record in cohort.primary_stratum()] == ["R1", "R3"]
    assert [record.record_id for record in cohort.reference_stratum()] == ["R2"]
    assert cohort.by_id("R2").record_id == "R2"
    assert cohort.patient_ids() == ("P1",)
    assert len(cohort.select(lambda record: record.site == "A")) == 3
