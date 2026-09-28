"""Two-stage training, checkpointing and the minimal closed loop.

Ref: Algorithm 2; Methods, "Training and inference".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from _paths import smoke_inputs

from rcrn.training.checkpoint import (
    CheckpointState,
    load_checkpoint,
    model_state_digest,
    restore_into,
    save_checkpoint,
)
from rcrn.training.ema import ExponentialMovingAverage
from rcrn.training.optim import build_optimizer, parameter_digest
from rcrn.training.trainer import Trainer, TrainingResult, build_view


def _trainer() -> Trainer:
    config, bundle, split = smoke_inputs()
    return Trainer(config, bundle, split)


def test_minimal_training_loop_runs_two_steps() -> None:
    trainer = _trainer()
    result: TrainingResult = trainer.run_steps(2)
    assert result.steps == 2
    assert np.isfinite(result.final_loss())


def test_parameters_change_after_an_update() -> None:
    trainer = _trainer()
    before = parameter_digest(trainer.model)
    trainer.run_steps(1)
    after = parameter_digest(trainer.model)
    assert before["l2"] != after["l2"]
    assert before["count"] == after["count"]


def test_checkpoint_round_trip_is_exact(tmp_path: Path) -> None:
    trainer = _trainer()
    trainer.run_steps(2)
    digest = model_state_digest(trainer.model)
    path = trainer.save(tmp_path / "checkpoint.pt", best_metric=0.25, epoch=3)
    state = load_checkpoint(path)
    assert state.epoch == 3
    assert state.step == 2
    assert state.best_metric == pytest.approx(0.25)
    assert state.seed == int(trainer.config.get("train.seed_base", 0))
    assert state.property_head_state is not None
    trainer.load(path)
    assert model_state_digest(trainer.model) == digest


def test_checkpoint_carries_the_seed_and_fingerprint() -> None:
    trainer = _trainer()
    state = CheckpointState(
        epoch=0,
        step=0,
        seed=99,
        deterministic=True,
        config_fingerprint=trainer.config.fingerprint(),
        best_metric=0.0,
        model_state={key: value.detach() for key, value in trainer.model.state_dict().items()},
    )
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = save_checkpoint(Path(directory) / "plain.pt", state)
        recovered = load_checkpoint(path)
    assert recovered.seed == 99
    assert recovered.config_fingerprint == trainer.config.fingerprint()
    assert recovered.optimizer_state is None


def test_checkpoint_file_is_world_readable(tmp_path: Path) -> None:
    trainer = _trainer()
    path = trainer.save(tmp_path / "checkpoint.pt")
    assert oct(path.stat().st_mode & 0o777) == "0o644"


def test_restore_into_mutates_a_fresh_model() -> None:
    trainer = _trainer()
    trainer.run_steps(1)
    target = _trainer()
    before = model_state_digest(target.model)
    state = CheckpointState(
        epoch=0,
        step=1,
        seed=0,
        deterministic=True,
        config_fingerprint="x",
        best_metric=0.0,
        model_state={key: value.detach() for key, value in trainer.model.state_dict().items()},
    )
    restore_into(target.model, state)
    assert model_state_digest(target.model) != before
    assert model_state_digest(target.model) == model_state_digest(trainer.model)


def test_ema_shadow_tracks_the_live_model() -> None:
    trainer = _trainer()
    optimizer = build_optimizer(trainer.model, "adamw", 0.05, 0.0)
    view = build_view(trainer.bundle, trainer.split.all_fit())
    batch = view.batch(torch.arange(min(trainer.batch_size, view.size())))
    before = parameter_digest(trainer.ema.shadow)
    for _ in range(3):
        trainer.step(batch, optimizer)
    after = parameter_digest(trainer.ema.shadow)
    assert before["l2"] != after["l2"]
    assert trainer.ema.state().updates == 3
    assert trainer.ema.state().decay == trainer.decay


def test_ema_rejects_an_out_of_range_decay() -> None:
    trainer = _trainer()
    with pytest.raises(ValueError):
        ExponentialMovingAverage(trainer.model, 1.0)


def test_ema_copy_restores_the_average() -> None:
    trainer = _trainer()
    target = _trainer()
    trainer.ema.update(trainer.model)
    trainer.ema.copy_to(target.model)
    assert model_state_digest(target.model) == model_state_digest(trainer.ema.shadow)


def test_gate_fitting_returns_a_calibrated_threshold() -> None:
    trainer = _trainer()
    trainer.run_steps(2)
    fitting = trainer.fit_gate(tolerance=0.05)
    assert fitting.calibration.false_conditioning_rate <= 0.05
    assert fitting.reference_volume_mm3 > 0.0
    decision = fitting.gate.decide(
        fitting.estimator.evaluate(
            trainer.infer(trainer.split.all_fit()).resistance.perturbation.cpu(),
            torch.tensor(
                [
                    trainer.bundle.cohort.by_id(record_id).plaque_volume_mm3
                    for record_id in trainer.split.all_fit()
                ]
            ),
            fitting.reference_volume_mm3,
        )
    )
    assert decision.conditioning_share + decision.deferral_share == pytest.approx(1.0)


def test_pretrain_runs_on_the_auxiliary_labels() -> None:
    trainer = _trainer()
    steps = trainer.pretrain(max_steps=2)
    assert steps == 2


def test_pretrain_is_skipped_without_labels() -> None:
    from dataclasses import replace

    config, bundle, split = smoke_inputs()
    stripped = tuple(replace(record, auxiliary_permeability=None) for record in bundle.candidates)
    trainer = Trainer(config, replace(bundle, candidates=stripped), split)
    assert trainer.pretrain() == 0


def test_infer_produces_one_score_per_candidate() -> None:
    trainer = _trainer()
    output = trainer.infer(trainer.split.external_ids[:3])
    assert tuple(output.coupling.pair_score.shape) == (3, len(trainer.bundle.candidate_ids))


def test_view_batching_keeps_identifiers_aligned() -> None:
    _, bundle, split = smoke_inputs()
    view = build_view(bundle, split.all_fit()[:5])
    batch = view.batch(torch.tensor([0, 2]))
    assert batch.record_ids == (split.all_fit()[0], split.all_fit()[2])
    assert batch.volumes.shape[0] == 2
    assert batch.auxiliary_targets.shape[0] == len(bundle.candidate_ids)
