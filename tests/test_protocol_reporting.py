"""The pre-specified plan and the reporting helpers."""

from __future__ import annotations

import numpy as np
import pytest

from rcrn.evaluation.pipeline import COMPARATORS, comparator_names
from rcrn.evaluation.protocol import (
    AnalysisPlan,
    PlannedComparison,
    default_plan,
    evidence_footing,
    reachable_margin,
    unadjusted_after_evaluation,
)
from rcrn.evaluation.reporting import (
    ReportTable,
    TableRow,
    jsonable,
    mcid_crossing_column,
    parity_gap,
    ranked_table,
    seed_summary,
    stratum_column_name,
    summarise,
)


def _plan() -> AnalysisPlan:
    return default_plan(
        reference_stratum="rads_1_2",
        primary_stratum="rads_3_4_iph",
        rank_floor=0.05,
        reference_level=0.07,
        family_size=12,
        bootstrap_resamples=2000,
        seed_count=24,
        exploratory_fdr=0.05,
    )


def test_plan_pins_the_design_quantities() -> None:
    plan = _plan()
    assert plan.confirmatory_family_size == 12
    assert plan.bootstrap_resamples == 2000
    assert plan.seed_count == 24
    assert plan.exploratory_fdr == 0.05
    assert plan.multiplicity_method == "holm"
    assert plan.primary.stratum == "rads_3_4_iph"
    assert plan.primary.threshold == 0.05


def test_plan_evaluates_its_primary_criterion() -> None:
    plan = _plan()
    met = plan.evaluate({"primary_rank_agreement": 0.08})
    assert met["primary_met"] is True
    assert met["primary_margin"] == pytest.approx(0.03)
    missed = plan.evaluate({"primary_rank_agreement": 0.02})
    assert missed["primary_met"] is False
    assert missed["primary_margin"] == pytest.approx(-0.03)


def test_plan_reports_no_post_hoc_adjustment() -> None:
    plan = _plan()
    assert unadjusted_after_evaluation(plan) is True
    assert plan.as_record()["thresholds_adjusted_after_evaluation"] == 0


def test_plan_metric_names_cover_the_confirmatory_family() -> None:
    metrics = _plan().confirmatory_metrics()
    assert "primary_rank_agreement" in metrics
    assert metrics["incremental_auc"] == ("incremental_auc", 0.07)


def test_comparison_margin_is_signed() -> None:
    comparison = PlannedComparison(
        name="x", stratum="s", comparator="c", metric="m", threshold=0.05, family="confirmatory"
    )
    assert comparison.met(0.06) is True
    assert comparison.margin(0.06) == pytest.approx(0.01)


def test_reachable_margin_reports_headroom() -> None:
    reachable = reachable_margin(0.05, 0.61)
    assert reachable["headroom"] == pytest.approx(0.39)
    assert reachable["reachable"] is True
    assert reachable_margin(0.5, 0.61)["reachable"] is False


def test_evidence_footing_labels() -> None:
    assert evidence_footing(True, True) == "confirmatory"
    assert evidence_footing(True, False) == "primary_only"
    assert evidence_footing(False, True) == "not_confirmed"


def test_comparator_names_cover_the_tables() -> None:
    assert comparator_names() == COMPARATORS
    assert "rcrn" in COMPARATORS
    assert "clinical_nomogram" in COMPARATORS


def test_report_table_lookup() -> None:
    table = ReportTable(
        title="t",
        columns=("rho",),
        rows=(
            TableRow(label="RCRN", cells={"rho": 0.69}),
            TableRow(label="baseline", cells={"rho": 0.61}),
        ),
    )
    assert table.cell("RCRN", "rho") == 0.69
    with pytest.raises(KeyError):
        table.cell("missing", "rho")
    assert table.as_record()["columns"] == ["rho"]


def test_ranked_table_orders_by_a_column() -> None:
    rows = (
        TableRow(label="a", cells={"rho": 0.5}),
        TableRow(label="b", cells={"rho": 0.7}),
    )
    assert [row.label for row in ranked_table(rows, "rho")] == ["b", "a"]
    assert [row.label for row in ranked_table(rows, "rho", descending=False)] == ["a", "b"]


def test_jsonable_converts_numpy_scalars() -> None:
    payload = {"a": np.float32(0.5), "b": np.array([1, 2]), "c": (np.int64(3),)}
    converted = jsonable(payload)
    assert converted == {"a": 0.5, "b": [1, 2], "c": [3]}


def test_summarise_reports_spread_and_count() -> None:
    summary = summarise(np.array([1.0, 2.0, 3.0, 4.0]))
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["count"] == 4.0
    assert summary["min"] == 1.0 and summary["max"] == 4.0
    assert summarise(np.zeros(0))["count"] == 0.0


def test_seed_summary_reports_the_seed_count() -> None:
    summary = seed_summary({1: 0.6, 2: 0.7, 3: 0.8})
    assert summary["seeds"] == 3.0
    assert summary["mean"] == pytest.approx(0.7)
    assert summary["std"] > 0.0


def test_stratum_column_name_is_explicit() -> None:
    assert stratum_column_name("rads_3_4", "iph_positive") == "rads_3_4_iph_positive"


def test_parity_gap_is_signed_per_shared_metric() -> None:
    gap = parity_gap({"rho": 0.69, "auc": 0.71}, {"rho": 0.69, "auc": 0.72, "extra": 1.0})
    assert set(gap) == {"rho", "auc"}
    assert gap["rho"] == pytest.approx(0.0)
    assert gap["auc"] == pytest.approx(-0.01)


def test_mcid_crossing_column_ignores_an_empty_vector() -> None:
    assert mcid_crossing_column(np.zeros(0), 0.07) == 0.0
    assert mcid_crossing_column(np.array([0.08, 0.06]), 0.07) == pytest.approx(0.5)
