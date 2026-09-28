"""Two-stage trainer for the regime-coupled reaction-norm model.

Ref: Algorithm 2; Methods, "Training and inference"; Methods, "Evaluation protocol
and statistical analysis".

Stage one fits the molecular branch on the auxiliary permeability labels, which is
what gives the joint stage an initialisation rather than a random start. Stage two
fits both branches against the agreement-weighted listwise rank objective. The
interaction covariance, the per-record noise term and the gate threshold are
estimated inside the training partition only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from rcrn.data.batching import MoleculeBatch, molecule_batch, volume_batch
from rcrn.data.fixture import FixtureBundle
from rcrn.data.molecules import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from rcrn.data.splits import SplitPlan
from rcrn.losses.objective import (
    CompositeObjective,
    LossBreakdown,
    LossTargets,
    LossWeights,
    ObjectiveSettings,
)
from rcrn.losses.property_prior import property_prior_loss
from rcrn.losses.rank import build_pairs
from rcrn.models.gate import DeclineToConditionGate, GateCalibration
from rcrn.models.interaction import InteractionEstimator
from rcrn.models.property_head import PropertyHead
from rcrn.models.rcrn import RCRNModel, RCRNOutput
from rcrn.training.amp import autocast_context, build_grad_scaler
from rcrn.training.checkpoint import CheckpointState, load_checkpoint, save_checkpoint
from rcrn.training.distributed import (
    ProcessGroupInfo,
    initialise_process_group,
    unwrap,
    wrap_distributed,
)
from rcrn.training.ema import ExponentialMovingAverage
from rcrn.training.optim import build_optimizer
from rcrn.training.schedule import build_schedule
from rcrn.training.seed import set_seed
from rcrn.utils.config import Config
from rcrn.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class CohortView:
    """The tensor form of one partition of the cohort."""

    record_ids: tuple[str, ...]
    volumes: Tensor
    plaque_volume_mm3: Tensor
    agreement_weights: Tensor
    observed_order: Tensor
    auxiliary_targets: Tensor

    def batch(self, indices: Tensor) -> CohortView:
        return CohortView(
            record_ids=tuple(self.record_ids[int(index)] for index in indices.tolist()),
            volumes=self.volumes[indices],
            plaque_volume_mm3=self.plaque_volume_mm3[indices],
            agreement_weights=self.agreement_weights[indices],
            observed_order=self.observed_order[indices],
            auxiliary_targets=self.auxiliary_targets,
        )

    def size(self) -> int:
        return len(self.record_ids)


def build_view(bundle: FixtureBundle, record_ids: tuple[str, ...]) -> CohortView:
    """Assemble the tensor view of a record selection."""

    candidates = bundle.candidate_ids
    volumes = volume_batch(bundle.volumes, record_ids)
    volume_mm3 = torch.tensor(
        [bundle.cohort.by_id(record_id).plaque_volume_mm3 for record_id in record_ids],
        dtype=torch.float32,
    )
    weights = torch.tensor(
        [bundle.cohort.by_id(record_id).reader_kappa for record_id in record_ids],
        dtype=torch.float32,
    )
    observed = torch.tensor(
        [
            [candidates.index(molecule_id) for molecule_id in bundle.implied_order[record_id]]
            for record_id in record_ids
        ],
        dtype=torch.long,
    )
    auxiliary = torch.tensor(
        [
            (
                float(record.auxiliary_permeability)
                if record.auxiliary_permeability is not None
                else float("nan")
            )
            for record in bundle.candidates
        ],
        dtype=torch.float32,
    )
    return CohortView(
        record_ids=record_ids,
        volumes=volumes,
        plaque_volume_mm3=volume_mm3,
        agreement_weights=weights,
        observed_order=observed,
        auxiliary_targets=auxiliary,
    )


@dataclass(slots=True)
class TrainingResult:
    """What a training run produced."""

    epochs: int
    steps: int
    history: tuple[dict[str, float], ...]
    checkpoint_path: str | None = None
    pretrain_steps: int = 0
    auxiliary_targets_used: bool = False

    def final_loss(self) -> float:
        return self.history[-1]["total"] if self.history else float("nan")

    def first_loss(self) -> float:
        return self.history[0]["total"] if self.history else float("nan")


@dataclass(slots=True)
class GateFitting:
    """The estimated interaction surfaces and the calibrated gate."""

    estimator: InteractionEstimator
    gate: DeclineToConditionGate
    calibration: GateCalibration
    reference_volume_mm3: float


class Trainer:
    """Own the model, the two stages and the checkpoint path of one run."""

    def __init__(
        self,
        config: Config,
        bundle: FixtureBundle,
        split: SplitPlan,
        device: torch.device | None = None,
        process: ProcessGroupInfo | None = None,
    ) -> None:
        self.config = config
        self.bundle = bundle
        self.split = split
        self.device = device or torch.device("cpu")
        self.process = process or initialise_process_group()
        self.molecule_batch: MoleculeBatch = molecule_batch(bundle.candidates)
        self.model = RCRNModel(config, ATOM_FEATURE_DIM, BOND_FEATURE_DIM).to(self.device)
        self.property_head = PropertyHead(self.model.d_cap).to(self.device)
        self.weights = LossWeights(
            rank=float(config.get("objective.loss_weights.rank", 1.0)),
            property=float(config.get("objective.loss_weights.property", 0.5)),
            route_sparsity=float(config.get("objective.loss_weights.route_sparsity", 0.01)),
            consistency=float(config.get("objective.loss_weights.consistency", 0.2)),
        )
        self.settings = ObjectiveSettings(
            surrogate=str(config.get("objective.rank.surrogate", "logistic")),
            temperature=float(config.get("objective.rank.temperature", 0.5)),
            pair_budget=int(config.get("objective.rank.pair_budget_per_plaque", 256)),
            groups=self.model.route_count,
            property_beta=float(config.get("objective.property.beta", 0.25)),
            consistency_margin=float(config.get("objective.consistency.margin", 1.0)),
        )
        self.objective = CompositeObjective(self.weights, self.settings)
        self.precision = str(config.get("train.precision", "fp32"))
        self.grad_clip = float(config.get("train.grad_clip", 1.0))
        self.epochs = int(config.get("train.epochs", 1))
        self.pretrain_epochs = int(config.get("train.pretrain_epochs", 0))
        self.batch_size = int(config.get("train.batch_size", 4))
        self.decay = float(config.get("train.ema_decay", 0.999))
        self.history: list[dict[str, float]] = []
        self.step_count = 0
        self.ema = ExponentialMovingAverage(self.model, self.decay)
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: object | None = None
        self._scaler: object | None = None

    def _train_view(self) -> CohortView:
        return build_view(self.bundle, self.split.all_fit())

    def _optimisation_parts(self, total_steps: int) -> tuple[torch.optim.Optimizer, object]:
        holder = nn.ModuleList([self.model, self.property_head])
        optimizer = build_optimizer(
            holder,
            str(self.config.get("train.optimizer", "adamw")),
            float(self.config.get("train.lr", 3e-4)),
            float(self.config.get("train.weight_decay", 0.01)),
            tuple(float(value) for value in self.config.get("train.betas", [0.9, 0.999])),
            float(self.config.get("train.eps", 1e-8)),
        )
        scheduler = build_schedule(
            optimizer,
            str(self.config.get("train.scheduler", "cosine")),
            total_steps,
            float(self.config.get("train.warmup_fraction", 0.0)),
            float(self.config.get("train.min_lr_factor", 0.0)),
        )
        return optimizer, scheduler

    def _batches(self, view: CohortView) -> list[Tensor]:
        order = torch.randperm(view.size())
        return [
            order[start : start + self.batch_size]
            for start in range(0, view.size(), self.batch_size)
        ]

    def pretrain(self, max_steps: int | None = None) -> int:
        """Stage one: supervise the molecular branch on the auxiliary labels."""

        targets = self._auxiliary_targets()
        if not bool(torch.isfinite(targets).any()):
            LOGGER.info("no auxiliary labels on the candidate list; stage one is skipped")
            return 0
        optimizer, _ = self._optimisation_parts(max_steps or self.pretrain_epochs or 1)
        steps = 0
        budget = max(1, max_steps if max_steps is not None else self.pretrain_epochs)
        for _ in range(budget):
            optimizer.zero_grad(set_to_none=True)
            output = self.model(self.molecule_batch, self._probe_volumes())
            prediction = self.property_head(output.molecules.graph_state)
            loss = property_prior_loss(prediction, targets, self.settings.property_beta)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            steps += 1
        return steps

    def _auxiliary_targets(self) -> Tensor:
        return torch.tensor(
            [
                (
                    float(record.auxiliary_permeability)
                    if record.auxiliary_permeability is not None
                    else float("nan")
                )
                for record in self.bundle.candidates
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def _probe_volumes(self) -> Tensor:
        """One plaque volume used while the molecular stage runs alone."""

        first = self.split.all_fit()[0]
        return volume_batch(self.bundle.volumes, (first,)).to(self.device)

    def step(self, batch: CohortView, optimizer: torch.optim.Optimizer) -> LossBreakdown:
        """One composite-objective update over a plaque batch."""

        with autocast_context(self.precision, self.device):
            output = self.model(self.molecule_batch, batch.volumes.to(self.device))
            prediction = self.property_head(output.molecules.graph_state)
            targets = LossTargets(
                observed_order=batch.observed_order.to(self.device),
                agreement_weights=batch.agreement_weights.to(self.device),
                auxiliary_targets=self._auxiliary_targets(),
                rank_pairs=build_pairs(
                    output.coupling.pair_score.detach(),
                    batch.observed_order.to(self.device),
                    batch.agreement_weights.to(self.device),
                    self.settings.pair_budget,
                ),
            )
            loss = self.objective(output, targets, prediction)
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss.total)
        if self.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        optimizer.step()
        self.ema.update(self.model)
        self.step_count += 1
        return loss

    def run(self, seed: int | None = None) -> TrainingResult:
        """Run both stages and return the loss history."""

        resolved_seed = int(seed if seed is not None else self.config.get("train.seed_base", 0))
        set_seed(resolved_seed, bool(self.config.get("train.deterministic", True)))
        pretrain_steps = self.pretrain()
        view = self._train_view()
        batches = self._batches(view)
        total_steps = max(1, self.epochs * max(1, len(batches)))
        optimizer, scheduler = self._optimisation_parts(total_steps)
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._scaler = build_grad_scaler(self.precision)
        self.model = wrap_distributed(self.model, self.process)
        for epoch in range(self.epochs):
            for indices in batches:
                breakdown = self.step(view.batch(indices), optimizer)
                record = breakdown.as_dict()
                record["epoch"] = float(epoch)
                record["learning_rate"] = float(optimizer.param_groups[0]["lr"])
                self.history.append(record)
            scheduler.step()
        return TrainingResult(
            epochs=self.epochs,
            steps=len(self.history),
            history=tuple(self.history),
            pretrain_steps=pretrain_steps,
            auxiliary_targets_used=bool(torch.isfinite(self._auxiliary_targets()).any()),
        )

    def run_steps(self, steps: int) -> TrainingResult:
        """Run a fixed number of minibatch updates, for the minimal-loop check."""

        set_seed(int(self.config.get("train.seed_base", 0)), True)
        view = self._train_view()
        optimizer, scheduler = self._optimisation_parts(max(steps, 1))
        batches = self._batches(view)
        for index in range(steps):
            indices = batches[index % len(batches)]
            self.history.append(self.step(view.batch(indices), optimizer).as_dict())
            scheduler.step()
        return TrainingResult(epochs=0, steps=len(self.history), history=tuple(self.history))

    @torch.no_grad()
    def infer(self, record_ids: tuple[str, ...]) -> RCRNOutput:
        """Forward pass over a record selection, in evaluation mode."""

        self.model.eval()
        volumes = volume_batch(self.bundle.volumes, record_ids).to(self.device)
        output = self.model(self.molecule_batch, volumes)
        self.model.train()
        return output

    def fit_gate(self, tolerance: float) -> GateFitting:
        """Estimate the interaction surfaces on the fitting partition and calibrate the gate."""

        record_ids = self.split.all_fit()
        output = self.infer(record_ids)
        estimator = InteractionEstimator(
            prior_strength=float(self.config.get("objective.gate.prior_strength", 0.1)),
            shrinkage=0.1,
        )
        covariance = estimator.fit(output.coupling.susceptibility.cpu())
        del covariance
        volume_mm3 = torch.tensor(
            [self.bundle.cohort.by_id(record_id).plaque_volume_mm3 for record_id in record_ids],
            dtype=torch.float32,
        )
        reference = float(np.median(volume_mm3.numpy()))
        surfaces = estimator.evaluate(output.resistance.perturbation.cpu(), volume_mm3, reference)
        spread = output.coupling.pair_score.std(dim=1).cpu().numpy()
        gate = DeclineToConditionGate()
        calibration = gate.calibrate(surfaces, spread, tolerance)
        return GateFitting(
            estimator=estimator,
            gate=gate,
            calibration=calibration,
            reference_volume_mm3=reference,
        )

    def save(self, path: str | Path, best_metric: float = 0.0, epoch: int = 0) -> Path:
        """Write a checkpoint carrying the seed and the config fingerprint."""

        model = unwrap(self.model)
        state = CheckpointState(
            epoch=epoch,
            step=self.step_count,
            seed=int(self.config.get("train.seed_base", 0)),
            deterministic=bool(self.config.get("train.deterministic", True)),
            config_fingerprint=self.config.fingerprint(),
            best_metric=best_metric,
            model_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
            optimizer_state=self._optimizer.state_dict() if self._optimizer is not None else None,
            scheduler_state=self._scheduler.state_dict() if self._scheduler is not None else None,
            property_head_state={
                key: value.detach().cpu() for key, value in self.property_head.state_dict().items()
            },
            history=tuple(self.history),
        )
        return save_checkpoint(path, state)

    def load(self, path: str | Path) -> CheckpointState:
        """Restore a checkpoint, including the seed that produced it."""

        state = load_checkpoint(path)
        unwrap(self.model).load_state_dict(state.model_state)
        if state.property_head_state is not None:
            self.property_head.load_state_dict(state.property_head_state)
        if state.optimizer_state is not None and self._optimizer is not None:
            self._optimizer.load_state_dict(state.optimizer_state)
        if state.scheduler_state is not None and self._scheduler is not None:
            self._scheduler.load_state_dict(state.scheduler_state)
        self.step_count = state.step
        self.history = list(state.history)
        set_seed(state.seed, state.deterministic)
        return state
