"""Reader agreement, assay endpoints and design invariants."""

from __future__ import annotations

import numpy as np
import pytest

from rcrn.data.auxiliary import assay_penalty, grouped_split, load_assay_table, mean_absolute_error
from rcrn.evaluation.assay import (
    FamilyError,
    endpoint_reversal,
    error_spread,
    error_table,
    family_rank_lookup,
    loss_margin,
)
from rcrn.evaluation.invariants import (
    collect_invariants,
    composition_index_is_image_only,
    external_fold_unfitted,
    label_pipeline_independent,
    normalisation_fitted_on_training,
    prohibited_features_absent,
)
from rcrn.evaluation.reader import (
    ReaderReading,
    agreement_ladder,
    expertise_stratum_effect,
    ranking_study_comparison,
)
from rcrn.metrics.agreement import (
    agreement_improvement,
    cohen_kappa,
    fleiss_kappa,
    weighted_kappa,
)


def test_cohen_kappa_matches_a_hand_value() -> None:
    first = np.array([0, 0, 1, 1])
    second = np.array([0, 0, 1, 1])
    assert cohen_kappa(first, second) == pytest.approx(1.0)
    thirds = np.array([0, 1, 0, 1])
    assert cohen_kappa(first, thirds) == pytest.approx(0.0)


def test_weighted_kappa_penalises_graded_disagreement() -> None:
    first = np.array([0, 1, 2, 3])
    near = np.array([0, 1, 2, 2])
    far = np.array([3, 2, 1, 0])
    assert weighted_kappa(first, near) > weighted_kappa(first, far)


def test_fleiss_kappa_is_one_for_unanimous_ratings() -> None:
    table = np.array([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    assert fleiss_kappa(table) == pytest.approx(1.0)


def test_fleiss_kappa_needs_a_constant_rater_count() -> None:
    with pytest.raises(ValueError):
        fleiss_kappa(np.array([[2.0, 0.0], [1.0, 0.0]]))


def test_agreement_helpers() -> None:
    assert agreement_improvement(0.431, 0.713) == pytest.approx(0.282)


def test_reader_ladder_improves_with_assistance() -> None:
    width = 6
    readings: list[ReaderReading] = []
    rng = np.random.default_rng(0)
    target = np.arange(width)
    for reader in range(4):
        expertise = "expert" if reader < 2 else "novice"
        for record in range(12):
            readings.append(
                ReaderReading(
                    reader=f"R{reader}",
                    expertise=expertise,
                    record_id=f"rec{record}",
                    unassisted=rng.permutation(width),
                    assisted=target if expertise == "expert" else np.roll(target, 1),
                )
            )
    before, after = agreement_ladder(tuple(readings))
    assert after > before
    assert expertise_stratum_effect(tuple(readings), "expert") > 0.0


def test_ranking_study_comparison_uses_spearman() -> None:
    target = np.arange(5.0)
    perfect = target.copy()
    reversed_order = target[::-1]
    table = ranking_study_comparison(perfect, perfect, reversed_order, target)
    assert table["model_alone"] == pytest.approx(1.0)
    assert table["reader_with_model"] == pytest.approx(-1.0)


def test_assay_reader_reads_a_table(tmp_path) -> None:
    table = tmp_path / "assay.csv"
    table.write_text(
        "molecule_id,smiles,target,cluster\nA,CCO,1.0,c1\nB,c1ccccc1,2.0,c2\nC,CC(=O)O,3.0,c2\n",
        encoding="utf-8",
    )
    release = load_assay_table(table, source="fixture release", licence="test")
    assert len(release) == 3
    assert release.descriptor_block().shape[0] == 3
    split = grouped_split(release, 0.5, seed=0)
    fit, held_out = split.sizes()
    assert fit + held_out == 3
    assert mean_absolute_error(np.array([1.0, 2.0]), np.array([1.0, 3.0])) == pytest.approx(0.5)
    assert assay_penalty(0.25, 0.30) == pytest.approx(1.2)


def test_assay_table_rejects_a_missing_column(tmp_path) -> None:
    table = tmp_path / "bad.csv"
    table.write_text("molecule_id,target\nA,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_assay_table(table, source="x", licence="y")


def _errors() -> tuple[FamilyError, ...]:
    return (
        FamilyError("descriptor", "assay", 0.2571),
        FamilyError("descriptor", "penetration", 0.54),
        FamilyError("state_space", "assay", 0.2655),
        FamilyError("state_space", "penetration", 0.59),
    )


def test_error_table_orders_by_error() -> None:
    rows = error_table(_errors(), "assay")
    assert [row.family for row in rows] == ["descriptor", "state_space"]
    assert family_rank_lookup(_errors(), "assay")["descriptor"] == 1
    assert error_spread(_errors(), "assay") == pytest.approx(0.0084, abs=1e-4)


def test_endpoint_reversal_detects_the_swap() -> None:
    outcome = endpoint_reversal(_errors(), "state_space", "descriptor", "penetration", "assay")
    assert outcome["left_leads_penetration"] is True
    assert outcome["left_leads_assay"] is False
    assert outcome["reversed"] is True


def test_loss_margin_needs_a_shared_endpoint() -> None:
    rows = _errors()
    assert loss_margin(rows[2], rows[0]) == pytest.approx(0.0084, abs=1e-4)
    with pytest.raises(ValueError):
        loss_margin(rows[0], rows[1])


def test_invariants_report_their_verdicts() -> None:
    assert prohibited_features_absent(("atom_features",), ("site_id",)).holds is True
    assert prohibited_features_absent(("site_id",), ("site_id",)).holds is False
    assert normalisation_fitted_on_training(frozenset({"a"}), frozenset({"a", "b"})).holds is True
    assert normalisation_fitted_on_training(frozenset({"c"}), frozenset({"a"})).holds is False
    assert external_fold_unfitted(frozenset({"x"}), frozenset({"y"})).holds is True
    assert label_pipeline_independent(("features",), ("labels",)).holds is True
    assert label_pipeline_independent(("shared",), ("shared",)).holds is False


def test_composition_index_rejects_a_deterministic_link() -> None:
    composition = np.linspace(0.0, 1.0, 20)
    assert composition_index_is_image_only(composition, composition * 2.0 + 1.0).holds is False
    rng = np.random.default_rng(0)
    assert composition_index_is_image_only(composition, rng.normal(size=20)).holds is True


def test_collect_invariants_names_the_failures() -> None:
    outcomes = (
        prohibited_features_absent(("atom_features",), ("site_id",)),
        prohibited_features_absent(("site_id",), ("site_id",)),
    )
    summary = collect_invariants(outcomes)
    assert summary["checked"] == 2
    assert summary["all_hold"] is False
    assert summary["failing"] == ["prohibited_features_absent"]
