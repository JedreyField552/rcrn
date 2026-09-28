"""Single-batch overfit and the end-to-end evaluation pipeline."""

from __future__ import annotations

import json

import numpy as np
import torch
from _paths import smoke_inputs

from rcrn.evaluation import build_report
from rcrn.evaluation.pipeline import (
    axis_attribution,
    certificate_summary,
    clinical_arm_table,
    clinical_increment_table,
    condition_dependence,
    design_invariants,
    multiplicity_table,
    ndcg_column,
    orderings_for,
    pooled_arm_consistency,
    ridge_regression,
    stratum_agreement,
    transport_summary_column,
)
from rcrn.evaluation.reporting import jsonable
from rcrn.training.optim import build_optimizer
from rcrn.training.seed import set_seed
from rcrn.training.trainer import Trainer, build_view


def _overfit_trainer(steps: int) -> tuple[list[float], Trainer]:
    config, bundle, split = smoke_inputs(
        ("train.lr=0.05", "train.warmup_fraction=0.0", "train.grad_clip=0.0")
    )
    set_seed(0, True)
    trainer = Trainer(config, bundle, split)
    view = build_view(bundle, split.all_fit())
    count = min(trainer.batch_size, view.size())
    batch = view.batch(torch.arange(count))
    optimizer = build_optimizer(trainer.model, "adamw", 0.05, 0.0)
    history = [trainer.step(batch, optimizer).total.item() for _ in range(steps)]
    return history, trainer


def test_single_batch_loss_decreases() -> None:
    history, _ = _overfit_trainer(40)
    assert history[-1] < history[0]
    assert np.isfinite(history).all()
    assert history[-1] < 0.9 * history[0]


def test_stratum_agreement_covers_the_named_strata() -> None:
    config, bundle, split = smoke_inputs()
    trainer = Trainer(config, bundle, split)
    trainer.run_steps(2)
    orderings, _ = orderings_for(trainer, split.all_fit())
    table = stratum_agreement(orderings, bundle, tuple(orderings))
    assert set(table) >= {"all_plaques", "rads_1_2", "rads_3_4", "rads_3_4_iph"}
    for entry in table.values():
        assert -1.0 <= entry["rho"] <= 1.0
        assert 0.0 <= entry["ndcg_at_10"] <= 1.0


def test_condition_dependence_reports_every_bin() -> None:
    config, bundle, split = smoke_inputs()
    trainer = Trainer(config, bundle, split)
    trainer.run_steps(2)
    orderings, _ = orderings_for(trainer, split.all_fit())
    table = condition_dependence(orderings, bundle, tuple(orderings), bin_count=3)
    assert set(table) == {"composition_quartile", "interaction_signal_bin"}
    for entry in table.values():
        assert len(entry) == 3
        for bucket in entry.values():
            assert bucket["records"] >= 0.0


def test_clinical_arm_table_lists_pooled_arms_and_sites() -> None:
    config, bundle, split = smoke_inputs()
    trainer = Trainer(config, bundle, split)
    trainer.run_steps(2)
    orderings, _ = orderings_for(trainer, split.all_fit())
    table = clinical_arm_table(orderings, bundle, split)
    assert {"pooled", "prospective", "retrospective"} <= set(table)
    consistency = pooled_arm_consistency(table)
    assert (
        consistency["records"]
        == table["prospective"]["records"] + table["retrospective"]["records"]
    )


def test_clinical_increment_is_bounded() -> None:
    _, bundle, _ = smoke_inputs()
    table = clinical_increment_table(bundle)
    assert 0.0 <= table["pooled"]["baseline_auc"] <= 1.0
    assert 0.0 <= table["pooled"]["augmented_auc"] <= 1.0
    assert table["pooled"]["increment_se"] >= 0.0


def test_transport_summary_counts_the_folds() -> None:
    config, bundle, split = smoke_inputs()
    trainer = Trainer(config, bundle, split)
    trainer.run_steps(2)
    orderings, _ = orderings_for(trainer, tuple([*split.all_fit(), *split.external_ids]))
    from rcrn.evaluation.pipeline import transport_table

    rows = transport_table(orderings, bundle, split, split.all_fit())
    summary = transport_summary_column(rows)
    populated = [cell for cell in rows.values() if cell.records > 0]
    assert summary["conditions"] == float(len(populated))
    attribution = axis_attribution(rows)
    assert set(attribution) == {"imaging_axis", "identity_or_time_axis"}


def test_certificate_summary_is_well_formed() -> None:
    config, bundle, split = smoke_inputs()
    trainer = Trainer(config, bundle, split)
    trainer.run_steps(2)
    orderings, _ = orderings_for(trainer, split.all_fit())
    summary = certificate_summary(orderings, trainer, 2, 0.05, 1.0)
    assert 0.0 <= summary["deferral_rate"] <= 1.0
    assert 0.0 <= summary["certified_share"] <= 1.0


def test_ndcg_column_reports_a_mean() -> None:
    config, bundle, split = smoke_inputs()
    trainer = Trainer(config, bundle, split)
    trainer.run_steps(2)
    orderings, _ = orderings_for(trainer, split.all_fit())
    column = ndcg_column(orderings, bundle, depth=3)
    assert 0.0 <= column["ndcg"] <= 1.0
    assert column["records"] == float(len(orderings))


def test_design_invariants_hold_on_the_shipped_artefacts() -> None:
    _, bundle, split = smoke_inputs()
    outcomes = design_invariants(bundle, split)
    assert len(outcomes) == 4
    assert all(outcome.holds for outcome in outcomes)


def test_ridge_regression_matches_the_normal_equations() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(20, 3))
    target = rng.normal(size=20)
    design = np.column_stack([np.ones(20), features])
    penalty = np.eye(4)
    penalty[0, 0] = 0.0
    manual = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    assert np.allclose(ridge_regression(features, target, 1.0), manual)


def test_multiplicity_table_adjusts_the_family() -> None:
    table = multiplicity_table((0.001, 0.02, 0.5), "holm", 0.05)
    assert table["family_size"] == 3.0
    assert table["rejections"] == 2.0


def test_full_pipeline_produces_a_serialisable_report() -> None:
    config, bundle, _ = smoke_inputs()
    report = build_report(config, bundle, steps=2)
    payload = jsonable(report.as_record())
    encoded = json.dumps(payload)
    assert "rank_agreement" in payload
    assert "in_distribution" in payload["rank_agreement"]
    assert payload["scale_note"]
    assert len(encoded) > 0
    assert payload["invariants"]


def test_full_pipeline_can_run_the_necessity_sweep() -> None:
    config, bundle, _ = smoke_inputs()
    variants = {
        "full": {"tier": "reference"},
        "no_gate": {
            "model.components.interaction_variance_gate": False,
            "tier": "tier_1_removal",
        },
        "joint_coupling_and_gate": {
            "model.components.reaction_norm_coupling": False,
            "model.coupling.kind": "plain_concat",
            "model.components.interaction_variance_gate": False,
            "tier": "tier_2_joint",
        },
    }
    report = build_report(config, bundle, variants=variants, steps=1)
    assert len(report.component_necessity) == 3
    assert report.component_necessity[0].variant == "full"
    assert report.component_necessity[0].delta == 0.0
    assert report.interaction_ratios or report.interaction_ratio_excluded
