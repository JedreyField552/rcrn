"""Patient-level, site-stratified splitting with an independent external fold.

Ref: Methods, "Cohort construction and auxiliary data"; Methods, "Evaluation protocol
and statistical analysis"; Results, Table 3 (every transport condition entered no
fitting step).

Splitting happens before any modelling step and never uses an outcome field. The
external fold is a whole site and region, so it takes no part in fitting, selection
or normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rcrn.data.schemas import PlaqueCohort, PlaqueRecord


@dataclass(slots=True)
class SplitPlan:
    """Record identifiers for the fitting, validation and external partitions."""

    fit_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    external_ids: tuple[str, ...]
    external_site: str
    external_region: str
    seed: int
    fold_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    fold_patient_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def all_fit(self) -> tuple[str, ...]:
        return tuple([*self.fit_ids, *self.validation_ids])

    def partitions(self) -> dict[str, tuple[str, ...]]:
        return {
            "fit": self.fit_ids,
            "validation": self.validation_ids,
            "external": self.external_ids,
        }


def _patients_by_site(cohort: PlaqueCohort) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = {}
    for record in cohort.records:
        grouped.setdefault(record.site, set()).add(record.patient_id)
    return {site: sorted(patients) for site, patients in grouped.items()}


def plan_split(
    cohort: PlaqueCohort,
    *,
    external_site: str,
    external_region: str,
    val_fraction: float,
    seed: int,
    conditions: dict[str, object] | None = None,
) -> SplitPlan:
    """Assign records to partitions at the patient level, stratified by site."""

    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("validation fraction must lie in [0, 1)")
    external_ids = [
        record.record_id
        for record in cohort.records
        if record.site == external_site or record.region == external_region
    ]
    external_record_set = set(external_ids)
    rng = np.random.default_rng(seed)
    fit_ids: list[str] = []
    validation_ids: list[str] = []
    for site, patients in sorted(_patients_by_site(cohort).items()):
        ordered = list(patients)
        rng.shuffle(ordered)
        holdout = int(round(len(ordered) * val_fraction))
        validation_patients = set(ordered[:holdout])
        for record in cohort.records:
            if record.record_id in external_record_set or record.site != site:
                continue
            if record.patient_id in validation_patients:
                validation_ids.append(record.record_id)
            else:
                fit_ids.append(record.record_id)
    plan = SplitPlan(
        fit_ids=tuple(sorted(fit_ids)),
        validation_ids=tuple(sorted(validation_ids)),
        external_ids=tuple(sorted(external_ids)),
        external_site=external_site,
        external_region=external_region,
        seed=seed,
    )
    if conditions:
        fold_ids, fold_patients = select_transport_folds(cohort, conditions, external_ids)
        plan.fold_ids = fold_ids
        plan.fold_patient_ids = fold_patients
    return plan


def _condition_matches(
    record: PlaqueRecord, condition: str, external_site: str, external_region: str
) -> bool:
    if condition == "held_out_site":
        return record.site == external_site
    if condition == "held_out_region":
        return record.region == external_region
    if condition == "scanner_vendor_shift":
        return record.acquisition.vendor != "A"
    if condition == "contrast_without_t1w":
        return not record.acquisition.has_t1_weighted()
    if condition == "slice_thickness_at_least_3mm":
        return record.acquisition.slice_thickness_mm >= 3.0
    if condition == "motion_degraded":
        return record.acquisition.motion_grade > 0.0
    if condition == "small_plaque_below_40mm3":
        return record.plaque_volume_mm3 < 40.0
    return False


def select_transport_folds(
    cohort: PlaqueCohort,
    conditions: dict[str, object],
    external_ids: list[str],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Group records by acquisition condition; no condition consults an outcome."""

    external_record_ids = sorted(external_ids)
    accrual_cut = _accrual_cut(cohort)
    fold_ids: dict[str, tuple[str, ...]] = {}
    fold_patients: dict[str, tuple[str, ...]] = {}
    for condition in conditions:
        if condition == "later_accrual_window":
            selected = [
                record.record_id
                for record in cohort.records
                if record.acquisition.accrual_index >= accrual_cut
            ]
        elif condition.startswith("held_out") and not external_record_ids:
            selected = []
        else:
            selected = [
                record.record_id
                for record in cohort.records
                if _condition_matches(record, condition, "C", "II")
            ]
        fold_ids[condition] = tuple(sorted(selected))
        fold_patients[condition] = tuple(
            sorted({cohort.by_id(record_id).patient_id for record_id in selected})
        )
    return fold_ids, fold_patients


def _accrual_cut(cohort: PlaqueCohort) -> int:
    indices = np.array(
        [record.acquisition.accrual_index for record in cohort.records], dtype=np.float64
    )
    if indices.size == 0:
        return 0
    return int(np.ceil(np.quantile(indices, 0.6)))


def patient_folds(
    cohort: PlaqueCohort, record_ids: tuple[str, ...], fold_count: int, seed: int
) -> list[tuple[str, ...]]:
    """Partition records into patient-disjoint folds for in-partition cross-validation."""

    if fold_count < 2:
        raise ValueError("at least two folds are required")
    grouped: dict[str, list[str]] = {}
    for record_id in record_ids:
        record = cohort.by_id(record_id)
        grouped.setdefault(record.patient_id, []).append(record_id)
    patients = sorted(grouped)
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    buckets: list[list[str]] = [[] for _ in range(fold_count)]
    for position, patient in enumerate(patients):
        buckets[position % fold_count].extend(grouped[patient])
    return [tuple(sorted(bucket)) for bucket in buckets]
