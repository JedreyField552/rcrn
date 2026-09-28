"""Patient-level, site-stratified splitting."""

from __future__ import annotations

import numpy as np
from _paths import smoke_config

from rcrn.data.splits import patient_folds, plan_split, select_transport_folds

CONFIG_PATH = "configs/experiment/_smoke.yaml"


def _fixture():
    config = smoke_config()
    from rcrn.data.fixture import build_cohort_fixture

    return config, build_cohort_fixture(config)


def test_external_fold_covers_the_named_site_and_region() -> None:
    config, bundle = _fixture()
    plan = plan_split(
        bundle.cohort,
        external_site="C",
        external_region="II",
        val_fraction=0.25,
        seed=3,
        conditions=dict(config.get("data.transport_conditions", {})),
    )
    assert plan.external_ids
    for record_id in plan.external_ids:
        assert bundle.cohort.by_id(record_id).site == "C"


def test_partitions_are_disjoint_and_complete() -> None:
    _, bundle = _fixture()
    plan = plan_split(
        bundle.cohort, external_site="C", external_region="II", val_fraction=0.25, seed=5
    )
    fit = set(plan.fit_ids)
    validation = set(plan.validation_ids)
    external = set(plan.external_ids)
    assert not fit & validation
    assert not fit & external
    assert not validation & external
    assert len(fit | validation | external) == len(bundle.cohort)


def test_split_is_patient_disjoint() -> None:
    _, bundle = _fixture()
    plan = plan_split(
        bundle.cohort, external_site="C", external_region="II", val_fraction=0.25, seed=7
    )
    fit_patients = {bundle.cohort.by_id(record_id).patient_id for record_id in plan.fit_ids}
    validation_patients = {
        bundle.cohort.by_id(record_id).patient_id for record_id in plan.validation_ids
    }
    assert not fit_patients & validation_patients


def test_same_seed_reproduces_the_split() -> None:
    _, bundle = _fixture()
    first = plan_split(
        bundle.cohort, external_site="C", external_region="II", val_fraction=0.25, seed=11
    )
    second = plan_split(
        bundle.cohort, external_site="C", external_region="II", val_fraction=0.25, seed=11
    )
    assert first.fit_ids == second.fit_ids
    assert first.validation_ids == second.validation_ids


def test_transport_folds_use_acquisition_metadata_only() -> None:
    config, bundle = _fixture()
    conditions = dict(config.get("data.transport_conditions", {}))
    folds, patients = select_transport_folds(bundle.cohort, conditions, [])
    assert set(folds) == set(conditions)
    for record_id in folds["contrast_without_t1w"]:
        assert not bundle.cohort.by_id(record_id).acquisition.has_t1_weighted()
    for record_id in folds["slice_thickness_at_least_3mm"]:
        assert bundle.cohort.by_id(record_id).acquisition.slice_thickness_mm >= 3.0
    for record_id in folds["small_plaque_below_40mm3"]:
        assert bundle.cohort.by_id(record_id).plaque_volume_mm3 < 40.0
    assert set(patients) == set(conditions)


def test_patient_folds_are_disjoint() -> None:
    _, bundle = _fixture()
    record_ids = tuple(record.record_id for record in bundle.cohort.records)
    folds = patient_folds(bundle.cohort, record_ids, 3, seed=2)
    assert len(folds) == 3
    seen: set[str] = set()
    for fold in folds:
        patients = {bundle.cohort.by_id(record_id).patient_id for record_id in fold}
        assert not patients & seen
        seen |= patients
    assert sum(len(fold) for fold in folds) == len(record_ids)


def test_accrual_window_fold_is_defined() -> None:
    config, bundle = _fixture()
    conditions = dict(config.get("data.transport_conditions", {}))
    folds, _ = select_transport_folds(bundle.cohort, conditions, [])
    accrual = folds["later_accrual_window"]
    indices = np.array([bundle.cohort.by_id(rid).acquisition.accrual_index for rid in accrual])
    late = [record.acquisition.accrual_index for record in bundle.cohort.records]
    assert indices.size > 0
    assert indices.min() >= np.quantile(late, 0.5)
