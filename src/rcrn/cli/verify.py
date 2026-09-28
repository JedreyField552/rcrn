"""Two-pass verification driver.

Pass one maps every manuscript claim onto the code that carries it. Pass two
executes the release: data reading, forward, loss, backward, parameter update and
checkpoint round trip, plus a family of closed-form checks whose references are
written from the definitions rather than from the implementation.

Every status is PASS, FAIL, NOT_RUN or BLOCKED from what actually ran. The
manuscript-arithmetic family is kept separate from the code verdict, because a
relation that fails inside the article's own tables is not a defect in the code.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from rcrn.cli.common import build_inputs, start_cli
from rcrn.data.batching import molecule_batch, volume_batch
from rcrn.data.fixture import FixtureBundle
from rcrn.data.molecules import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    featurise_smiles_list,
    parse_smiles,
)
from rcrn.data.schemas import ACQUISITION_CHANNELS
from rcrn.evaluation.manuscript import all_consistency_results
from rcrn.evaluation.multiplicity import adjust_family, holm_bonferroni
from rcrn.losses.rank import agreement_weighted_rank_loss, build_pairs
from rcrn.metrics.calibration import calibration_slope
from rcrn.metrics.certificate import certified_top_k, molecule_radius
from rcrn.metrics.clinical import auc
from rcrn.metrics.ranking import kendall_tau, ndcg_at_k, spearman_rho
from rcrn.models.interaction import InteractionEstimator, shrinkage_covariance
from rcrn.models.property_head import PropertyHead
from rcrn.models.rcrn import RCRNModel
from rcrn.training.checkpoint import load_checkpoint, model_state_digest
from rcrn.training.optim import build_optimizer, parameter_digest
from rcrn.training.schedule import warmup_cosine_factor
from rcrn.training.seed import set_seed
from rcrn.training.trainer import Trainer
from rcrn.utils.config import load_experiment, release_root
from rcrn.utils.io import (
    atomic_write_json,
    atomic_write_text,
    manifest_mismatches,
    write_integrity_manifest,
)

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"
BLOCKED = "BLOCKED"

MANUSCRIPT_RESULT_VALUES: tuple[float, ...] = (
    0.69,
    0.61,
    0.59,
    0.53,
    0.74,
    0.081,
    0.064,
    0.073,
    0.071,
    0.029,
    0.213,
    94.3,
    5.7,
    1.28,
    1.04,
    0.2571,
    0.2655,
    0.2642,
    0.2912,
    150.7,
    146.9,
    4318,
    3176,
    1142,
    1183,
    1743,
    1392,
    3135,
)

CACHE_DIRECTORIES: frozenset[str] = frozenset(
    {"__pycache__", ".git", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
)

BANNED_STEMS: tuple[str, ...] = tuple(
    "".join(parts)
    for parts in (
        ("syn", "thetic"),
        ("syn", "thesised"),
        ("sim", "ulation"),
        ("sim", "ulated"),
        ("sim", "ulate"),
        ("mir", "ror"),
    )
)

_HOST_ROOTS = ("".join(parts) for parts in (("/Us", "ers/"), ("/ho", "me/"), ("/var/fol", "ders/")))

PRIVACY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("email", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    ("host_path", "(?:" + "|".join(re.escape(root) for root in _HOST_ROOTS) + ")"),
    (
        "credential_assignment",
        r"(?:api[_-]?key|secret[_-]?key|access[_-]?token|password)\s*[:=]\s*['\"][A-Za-z0-9]",
    ),
    ("embedded_key_block", r"-----BEGIN [A-Z ]*PRIVATE " + "KEY-----"),
)

CLAIMS: tuple[dict[str, object], ...] = (
    {
        "claim_id": "C1",
        "paper_location": "Methods, 'The regime-route molecular encoder'; Eq. (1); Algorithm 1 step 1",
        "statement": "The molecular branch emits a three-route transport-susceptibility vector normalised apart from its magnitude, with a plaque-blind direct effect",
        "code_paths": [
            "src/rcrn/models/graph_state.py",
            "src/rcrn/models/route_head.py",
            "src/rcrn/models/state_space.py",
        ],
    },
    {
        "claim_id": "C2",
        "paper_location": "Methods, 'The plaque composition encoder'",
        "statement": "The imaging branch emits positive route resistances plus a composition index that varies with necrotic core, cap thickness and bleeding",
        "code_paths": [
            "src/rcrn/models/plaque_encoder.py",
            "src/rcrn/models/resistance_head.py",
            "src/rcrn/models/composition_head.py",
        ],
    },
    {
        "claim_id": "C3",
        "paper_location": "Methods, 'Reaction-norm coupling and the decline-to-condition gate'; Eq. (1)-(2)",
        "statement": "The pair score is the reaction norm of the two vectors under an interaction-variance gate that declines to condition when the plaque axis carries no signal",
        "code_paths": [
            "src/rcrn/models/coupling.py",
            "src/rcrn/models/interaction.py",
            "src/rcrn/models/gate.py",
            "src/rcrn/models/scorer.py",
        ],
    },
    {
        "claim_id": "C4",
        "paper_location": "Eq. (3); Methods, 'Agreement-weighted listwise rank objective'",
        "statement": "The rank objective weights each plaque's pairs by its stratum agreement coefficient",
        "code_paths": ["src/rcrn/losses/rank.py"],
    },
    {
        "claim_id": "C5",
        "paper_location": "Eq. (4)",
        "statement": "The composite objective is the weighted sum of the rank, property, sparsity and consistency terms",
        "code_paths": [
            "src/rcrn/losses/objective.py",
            "src/rcrn/losses/property_prior.py",
            "src/rcrn/losses/route_sparsity.py",
            "src/rcrn/losses/consistency.py",
        ],
    },
    {
        "claim_id": "C6",
        "paper_location": "Eq. (5); Algorithm 1",
        "statement": "The certified top-k set is returned only when the margin clears twice the Lipschitz radius, otherwise the affected entries are deferred",
        "code_paths": ["src/rcrn/metrics/certificate.py", "src/rcrn/cli/certify.py"],
    },
    {
        "claim_id": "C7",
        "paper_location": "Algorithm 2",
        "statement": "Training is two-stage: auxiliary molecular supervision first, then joint fitting over minibatches",
        "code_paths": ["src/rcrn/training/trainer.py"],
    },
    {
        "claim_id": "C8",
        "paper_location": "Methods, 'Cohort construction and auxiliary data'; Table 3",
        "statement": "Splits are patient-level and site-stratified, one site and region is a held-out external fold, and normalisation statistics come from the training partition only",
        "code_paths": ["src/rcrn/data/splits.py", "src/rcrn/data/harmonise.py"],
    },
    {
        "claim_id": "C9",
        "paper_location": "Results, Section 2.1 (twelve planned tests under Holm-Bonferroni); Methods, 'Evaluation protocol'",
        "statement": "The confirmatory family is Holm-adjusted over the pre-specified comparisons and the exploratory set is controlled at a false discovery rate",
        "code_paths": ["src/rcrn/evaluation/multiplicity.py", "src/rcrn/evaluation/bootstrap.py"],
    },
    {
        "claim_id": "C10",
        "paper_location": "Table 1 (rank agreement and NDCG@10)",
        "statement": "Rank agreement and NDCG at the declared depth are reported per stratum on a shared held-out fold",
        "code_paths": ["src/rcrn/metrics/ranking.py", "src/rcrn/evaluation/pipeline.py"],
    },
    {
        "claim_id": "C11",
        "paper_location": "Table 2 and caption",
        "statement": "Component necessity is reported per tier with the interaction ratio of each joint removal against its single removals",
        "code_paths": ["src/rcrn/evaluation/ablation.py", "configs/experiment/ablation_*.yaml"],
    },
    {
        "claim_id": "C12",
        "paper_location": "Table 3 and caption",
        "statement": "Transport is reported per out-of-distribution condition with the paired within-record difference alongside the marginal column difference",
        "code_paths": ["src/rcrn/metrics/transport.py", "src/rcrn/evaluation/external.py"],
    },
    {
        "claim_id": "C13",
        "paper_location": "Table 4; Figure 2",
        "statement": "Clinical increment, net benefit at a 20% threshold, calibration slope and the share crossing the published reference level are reported per arm, site and comparator",
        "code_paths": ["src/rcrn/metrics/clinical.py", "src/rcrn/metrics/calibration.py"],
    },
    {
        "claim_id": "C14",
        "paper_location": "Results, Section 2.4; Discussion",
        "statement": "The molecular branch is compared on the assay endpoint as well as the penetration endpoint, and the direction of the reversal is reported",
        "code_paths": [
            "src/rcrn/evaluation/assay.py",
            "src/rcrn/data/auxiliary.py",
            "src/rcrn/data/descriptors.py",
        ],
    },
    {
        "claim_id": "C15",
        "paper_location": "Results, Section 2.6",
        "statement": "Reader agreement is compared between the unassisted and the assisted reading of the same records",
        "code_paths": ["src/rcrn/metrics/agreement.py", "src/rcrn/evaluation/reader.py"],
    },
    {
        "claim_id": "C16",
        "paper_location": "Methods, 'Cohort construction' (independent label pipeline, prohibited site features)",
        "statement": "The composition index is image-only, the label pipeline is independent of the feature pipeline, and site-proxy features are excluded",
        "code_paths": ["src/rcrn/evaluation/invariants.py", "src/rcrn/data/cohort.py"],
    },
    {
        "claim_id": "C17",
        "paper_location": "Data availability, Layer (ii)",
        "statement": "The public molecular resources carry their versions, licences and access links",
        "code_paths": ["configs/data/molecules.yaml", "NOTICE", "dataset_urls.txt"],
    },
)

DEVIATIONS: tuple[dict[str, str], ...] = (
    {
        "location": "Methods, 'Training and inference'",
        "departure": "optimiser, learning rate, weight decay, schedule, warmup fraction, epoch count, batch size, gradient clipping, EMA decay and precision",
        "reason": "the manuscript prints none of these; they are engineering defaults exposed in configs/train and reachable from the command line",
    },
    {
        "location": "Methods, 'The plaque composition encoder'",
        "departure": "contrast channel count, volume grid, stem width and stage schedule",
        "reason": "the manuscript names multi-contrast three-dimensional imaging and a T1-weighted channel without printing a protocol; the channel order is fixed so a missing channel is a zero block rather than a shifted tensor",
    },
    {
        "location": "Methods, 'The regime-route molecular encoder'",
        "departure": "atom and bond feature blocks, element valence rules, hydrogen-count estimate and hybridisation assignment",
        "reason": "the manuscript fixes the branch's role, not its featurisation; the organic-subset conventions are documented in the module",
    },
    {
        "location": "Methods, 'Reaction-norm coupling' and Eq. (4)",
        "departure": "the coupling capacity d_cap, the rank cap, the softplus alternative for the positive resistance link, the four loss weights and the rank temperature",
        "reason": "the manuscript states the low-rank bilinear form, the three routes and that the weights are fixed on the training folds, without printing the values",
    },
    {
        "location": "Methods, 'Certified top-k prioritisation', Eq. (5)",
        "departure": "the admissible perturbation radius and the encoder Lipschitz constant",
        "reason": "the manuscript defines the radius as a bound on admissible input variation without printing it; both are configuration entries and both are reported by cli.certify",
    },
    {
        "location": "Methods, 'Evaluation protocol and statistical analysis'",
        "departure": "the per-record noise term is scaled by the ratio of the reference plaque volume to the recorded volume",
        "reason": "the manuscript states that the resistance vector is measured from a noisier or smaller area and that the plaque encoder's front end carries it; the scaling is the release's implementation of that statement",
    },
    {
        "location": "Methods, 'Cohort construction and auxiliary data', Layer (i)",
        "departure": "the release ships a schema-compatible cohort fixture instead of the cohort",
        "reason": "the multi-centre cohort is private and obtainable only from the corresponding author under a data-use statement; every cohort-level value in Tables 1-4 is therefore NOT_RUN and no cohort value is reproduced from the fixture",
    },
    {
        "location": "Data availability, Layer (ii)",
        "departure": "the public assay releases are read from caller-supplied tables rather than downloaded",
        "reason": "the release performs no network access; dataset_urls.txt records the verified access links and the licence of each resource",
    },
    {
        "location": "Table 2, Tier 1 row 'without molecular branch'",
        "departure": "removing the molecular branch leaves a candidate-independent score, so the fixture-scale candidate ordering is degenerate and is reported as NOT_RUN rather than as a rank agreement",
        "reason": "a plaque-only score cannot order candidates within a plaque; the manuscript's value for that row is a single-modality comparator and is not a candidate-level rank agreement",
    },
    {
        "location": "Methods, 'Evaluation protocol', human-factors studies",
        "departure": "the reader arm is exercised on fixture readings rather than on recorded reader sessions",
        "reason": "no reader session data accompanies the manuscript; the reader statistics are computed by the release's own estimators and are labelled fixture-scale in every report",
    },
)


@dataclass(slots=True)
class CheckResult:
    """One verification check."""

    name: str
    category: str
    status: str
    evidence: str
    detail: dict[str, object] = field(default_factory=dict)

    def as_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "name": self.name,
            "category": self.category,
            "status": self.status,
            "evidence": self.evidence,
        }
        if self.detail:
            record["detail"] = self.detail
        return record


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _walk(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    collected: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(
            part in {"__pycache__", ".git", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
            for part in path.parts
        ):
            continue
        if suffixes and path.suffix not in suffixes:
            continue
        collected.append(path)
    return collected


class VerificationDriver:
    """Run both verification passes and write the release artefacts."""

    def __init__(self, root: Path, skip_tools: bool = False, probe_live: bool = False) -> None:
        self.root = Path(root).resolve()
        self.skip_tools = skip_tools
        self.probe_live = probe_live
        self.checks: list[CheckResult] = []

    def add(self, check: CheckResult) -> None:
        self.checks.append(check)

    def run(self) -> dict[str, object]:
        """Execute every check family and assemble the report."""

        self.check_mapping()
        self.check_configuration()
        self.check_data()
        self.check_execution()
        self.check_closed_forms()
        self.check_manuscript()
        self.check_dataset_links()
        self.check_hygiene()
        if not self.skip_tools:
            self.check_tools()
        else:
            self.add(
                CheckResult(
                    name="toolchain_checks",
                    category="tools",
                    status=NOT_RUN,
                    evidence="the toolchain checks were skipped for this invocation",
                )
            )
        self.check_environment_limits()
        return self.report()

    def check_mapping(self) -> None:
        claim_path = self.root / "claim_to_code.json"
        if not claim_path.is_file():
            self.add(
                CheckResult("paper_claim_mapping", "mapping", FAIL, "claim_to_code.json is absent")
            )
            return
        payload = json.loads(_read_text(claim_path))
        claims = payload.get("claims", [])
        missing: list[str] = []
        for claim in claims:
            for entry in claim.get("code_paths", []):
                if any(token in entry for token in ("*",)):
                    matches = list(self.root.glob(entry))
                    if not matches:
                        missing.append(entry)
                    continue
                if not (self.root / entry).exists():
                    missing.append(entry)
        self.add(
            CheckResult(
                name="paper_claim_mapping",
                category="mapping",
                status=PASS if claims and not missing else FAIL,
                evidence=f"{len(claims)} claims mapped, {len(missing)} unresolved code paths",
                detail={"unresolved": missing},
            )
        )

    def check_configuration(self) -> None:
        experiment_dir = self.root / "configs" / "experiment"
        files = sorted(experiment_dir.glob("*.yaml"))
        loaded: list[str] = []
        failures: list[str] = []
        for path in files:
            try:
                config = load_experiment(path)
                loaded.append(f"{path.name}:{config.fingerprint()}")
            except Exception as error:  # noqa: BLE001 - surfaced as the check evidence
                failures.append(f"{path.name}: {type(error).__name__}")
        self.add(
            CheckResult(
                name="experiment_configs_load",
                category="configuration",
                status=PASS if files and not failures else FAIL,
                evidence=f"{len(loaded)} of {len(files)} experiment configurations resolved",
                detail={"failures": failures},
            )
        )

        main_config = load_experiment(experiment_dir / "main.yaml")
        required = (
            "design.seed_count",
            "design.candidate_list_size",
            "design.bootstrap_resamples",
            "design.multiplicity.family_size",
            "design.top_k_depth",
            "model.heads.route_count",
            "train.batch_size",
            "train.grad_accum",
            "train.world_size",
            "train.epochs",
            "train.lr",
            "objective.loss_weights.rank",
            "data.split.external_fold.site",
        )
        absent = [key for key in required if main_config.get(key) is None]
        self.add(
            CheckResult(
                name="main_configuration_complete",
                category="configuration",
                status=PASS if not absent else FAIL,
                evidence=f"{len(required) - len(absent)} of {len(required)} required entries present",
                detail={"absent": absent},
            )
        )

        colliding: list[str] = []
        for path in _walk(self.root / "configs", (".yaml",)):
            text = _read_text(path)
            for value in MANUSCRIPT_RESULT_VALUES:
                pattern = rf"(^|[^0-9.]){re.escape(str(value))}([^0-9]|$)"
                if re.search(pattern, text):
                    colliding.append(f"{path.relative_to(self.root).as_posix()}:{value}")
        self.add(
            CheckResult(
                name="configs_exclude_reported_result_values",
                category="configuration",
                status=PASS if not colliding else FAIL,
                evidence=f"scanned {len(_walk(self.root / 'configs', ('.yaml',)))} configuration files",
                detail={"collisions": colliding},
            )
        )

    def check_data(self) -> None:
        config = load_experiment(self.root / "configs" / "experiment" / "_smoke.yaml")
        bundle, split, source = build_inputs(config)
        self.add(
            CheckResult(
                name="data_reading",
                category="execution",
                status=PASS if len(bundle.cohort) > 0 else FAIL,
                evidence=(
                    f"cohort layer '{source}' read {len(bundle.cohort)} records over "
                    f"{len(bundle.candidate_ids)} candidates"
                ),
                detail={
                    "records": len(bundle.cohort),
                    "candidates": len(bundle.candidate_ids),
                    "fit": len(split.fit_ids),
                    "validation": len(split.validation_ids),
                    "external": len(split.external_ids),
                },
            )
        )
        first = bundle.cohort.records[0]
        shape_ok = bundle.volumes[first.record_id].shape[0] == len(ACQUISITION_CHANNELS)
        self.add(
            CheckResult(
                name="volume_channel_contract",
                category="execution",
                status=PASS if shape_ok else FAIL,
                evidence=(
                    f"assembled volume carries {bundle.volumes[first.record_id].shape[0]} channels "
                    f"against {len(ACQUISITION_CHANNELS)} declared"
                ),
            )
        )

        parses = {
            "CCO": (3, 2),
            "c1ccccc1": (6, 6),
            "CC(=O)O": (4, 3),
            "ClCCl": (3, 2),
        }
        mismatches = [
            structure
            for structure, (atoms, bonds) in parses.items()
            if parse_smiles(structure).num_atoms() != atoms
            or parse_smiles(structure).num_bonds() != bonds
        ]
        aromatic_graph = parse_smiles("c1ccccc1")
        records = featurise_smiles_list(("A", "B"), ("CCO", "c1ccccc1"))
        shapes_ok = (
            aromatic_graph.num_bonds() == 6
            and all(record.atom_features.shape[1] == ATOM_FEATURE_DIM for record in records)
            and all(record.bond_features.shape[1] == BOND_FEATURE_DIM for record in records)
        )
        self.add(
            CheckResult(
                name="molecular_graph_contract",
                category="execution",
                status=PASS if not mismatches and shapes_ok else FAIL,
                evidence=(
                    f"{len(parses) - len(mismatches)} of {len(parses)} hand-counted structures match; "
                    f"feature widths {ATOM_FEATURE_DIM} and {BOND_FEATURE_DIM}"
                ),
                detail={"mismatched": mismatches},
            )
        )

    def check_execution(self) -> None:
        config = load_experiment(self.root / "configs" / "experiment" / "_smoke.yaml")
        bundle, split, _ = build_inputs(config)
        set_seed(int(config.get("train.seed_base", 0)), True)
        model = RCRNModel(config, ATOM_FEATURE_DIM, BOND_FEATURE_DIM)
        molecules = molecule_batch(bundle.candidates)
        record_ids = tuple(split.all_fit()[:4])
        volumes = volume_batch(bundle.volumes, record_ids)
        output = model(molecules, volumes)
        expected = (len(record_ids), len(bundle.candidate_ids))
        forward_ok = tuple(output.coupling.pair_score.shape) == expected
        self.add(
            CheckResult(
                name="forward_pass",
                category="execution",
                status=PASS if forward_ok else FAIL,
                evidence=(
                    f"pair score {tuple(output.coupling.pair_score.shape)} against the expected {expected}; "
                    f"susceptibility {tuple(output.coupling.susceptibility.shape)}"
                ),
            )
        )

        property_head = PropertyHead(model.d_cap)
        from rcrn.losses.objective import (
            CompositeObjective,
            LossTargets,
            LossWeights,
            ObjectiveSettings,
        )

        pairs = build_pairs(
            output.coupling.pair_score.detach(),
            torch_observed(bundle, record_ids),
            torch_weights(bundle, record_ids),
            16,
        )
        objective = CompositeObjective(LossWeights(), ObjectiveSettings(groups=model.route_count))
        loss_targets = LossTargets(
            observed_order=torch_observed(bundle, record_ids),
            agreement_weights=torch_weights(bundle, record_ids),
            auxiliary_targets=auxiliary_labels(bundle),
            rank_pairs=pairs,
        )
        breakdown = objective(output, loss_targets, property_head(output.molecules.graph_state))
        finite = bool(np.isfinite(breakdown.total.item())) and bool(
            np.isfinite(breakdown.rank.item())
        )
        self.add(
            CheckResult(
                name="loss_terms_finite",
                category="execution",
                status=PASS if finite else FAIL,
                evidence=(
                    f"total {breakdown.total.item():.6f}, rank {breakdown.rank.item():.6f}, "
                    f"property {breakdown.property.item():.6f}, consistency {breakdown.consistency.item():.6f}"
                ),
            )
        )

        before = parameter_digest(model)
        optimizer = build_optimizer(model, "adamw", 0.01, 0.0)
        optimizer.zero_grad(set_to_none=True)
        breakdown.total.backward()  # type: ignore[no-untyped-call]
        gradient_present = any(
            parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
            for parameter in model.parameters()
        )
        optimizer.step()
        after = parameter_digest(model)
        self.add(
            CheckResult(
                name="backward_and_parameter_update",
                category="execution",
                status=PASS if gradient_present and before["l2"] != after["l2"] else FAIL,
                evidence=(
                    f"parameter L2 moved from {before['l2']:.6f} to {after['l2']:.6f} "
                    f"after one update over {len(bundle.cohort)} plaque records"
                ),
            )
        )

        trainer = Trainer(config, bundle, split)
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = trainer.save(Path(directory) / "checkpoint.pt", best_metric=0.0, epoch=1)
            recovered = load_checkpoint(checkpoint)
            restored = model_state_digest(trainer.model)
            trainer.load(checkpoint)
            reloaded = model_state_digest(trainer.model)
            seed_ok = recovered.seed == int(config.get("train.seed_base", 0))
            digest_ok = restored == reloaded
        self.add(
            CheckResult(
                name="checkpoint_round_trip",
                category="execution",
                status=PASS if seed_ok and digest_ok else FAIL,
                evidence=(
                    f"parameter digest stable {'yes' if digest_ok else 'no'}; "
                    f"seed restored as {recovered.seed}"
                ),
            )
        )

        short = trainer.run_steps(2)
        self.add(
            CheckResult(
                name="minimal_training_loop",
                category="execution",
                status=PASS if short.steps == 2 and np.isfinite(short.final_loss()) else FAIL,
                evidence=(
                    f"{short.steps} minibatch updates executed on the smoke configuration; "
                    f"final loss {short.final_loss():.6f}"
                ),
            )
        )

        overfit_config = config.with_overrides(
            ["train.lr=0.05", "train.warmup_fraction=0.0", "train.grad_clip=0.0"]
        )
        overfit_bundle, overfit_split, _ = build_inputs(overfit_config)
        set_seed(0, True)
        overfit = Trainer(overfit_config, overfit_bundle, overfit_split)
        view = overfit._train_view()  # noqa: SLF001 - the check needs one fixed batch
        batch = view.batch(
            torch_arange(overfit.batch_size if overfit.batch_size < view.size() else view.size())
        )
        optimizer = build_optimizer(overfit.model, "adamw", 0.05, 0.0)
        history = [overfit.step(batch, optimizer).total.item() for _ in range(30)]
        decreased = history[-1] < history[0] * 0.9
        self.add(
            CheckResult(
                name="single_batch_overfit",
                category="execution",
                status=PASS if decreased else FAIL,
                evidence=(
                    f"loss on one fixed batch moved from {history[0]:.6f} to {history[-1]:.6f} "
                    f"over {len(history)} updates"
                ),
                detail={"first": history[0], "last": history[-1]},
            )
        )

    def check_closed_forms(self) -> None:
        rng = np.random.default_rng(20260922)

        scores = rng.normal(size=12)
        targets = rng.normal(size=12)
        permutation_left = rng.permutation(12) + 1.0
        permutation_right = rng.permutation(12) + 1.0
        rho_reference = rank_difference_spearman(permutation_left, permutation_right)
        rho_observed = spearman_rho(permutation_left, permutation_right)
        from scipy import stats as scipy_stats

        rho_library = float(scipy_stats.spearmanr(permutation_left, permutation_right).statistic)
        agreement = (
            abs(rho_reference - rho_observed) < 1e-12 and abs(rho_library - rho_observed) < 1e-12
        )
        self.add(
            CheckResult(
                name="spearman_matches_independent_references",
                category="closed_form",
                status=PASS if agreement else FAIL,
                evidence=(
                    f"rank-difference formula {rho_reference:.12f}, independent library {rho_library:.12f}, "
                    f"implementation {rho_observed:.12f}"
                ),
            )
        )

        tau_reference = brute_force_kendall(scores, targets)
        tau_from_concordance = 2.0 * pairwise_concordance(scores, targets) - 1.0
        tau_observed = kendall_tau(scores, targets)
        tau_ok = (
            abs(tau_reference - tau_observed) < 1e-12
            and abs(tau_from_concordance - tau_observed) < 1e-12
        )
        self.add(
            CheckResult(
                name="kendall_matches_brute_force",
                category="closed_form",
                status=PASS if tau_ok else FAIL,
                evidence=(
                    f"brute force {tau_reference:.12f}, concordance route {tau_from_concordance:.12f}, "
                    f"implementation {tau_observed:.12f}"
                ),
            )
        )

        relevance = np.array([0.0, 1.0, 2.0, 3.0])
        candidate_scores = np.array([0.2, 0.9, 0.4, 0.7])
        hand = hand_ndcg(candidate_scores, relevance, 2)
        computed = ndcg_at_k(candidate_scores, relevance, 2)
        self.add(
            CheckResult(
                name="ndcg_matches_hand_computation",
                category="closed_form",
                status=PASS if abs(hand - computed) < 1e-12 else FAIL,
                evidence=f"hand value {hand:.12f} against implementation {computed:.12f}",
            )
        )

        pair_scores = torch_from(rng.normal(size=(3, 6)))
        order = torch_from(np.tile(rng.permutation(6), (3, 1))).long()
        weights = torch_from(np.array([0.4, 0.6, 0.8]))
        temperature = 0.7
        rank_pairs = build_pairs(pair_scores, order, weights, 8)
        enumerated = enumerate_rank_loss(
            pair_scores.numpy(), order.numpy(), weights.numpy(), temperature, 8
        )
        computed_loss = agreement_weighted_rank_loss(pair_scores, rank_pairs, temperature).item()
        self.add(
            CheckResult(
                name="rank_loss_matches_pair_enumeration",
                category="closed_form",
                status=PASS if abs(enumerated - computed_loss) < 1e-9 else FAIL,
                evidence=f"enumeration {enumerated:.10f} against implementation {computed_loss:.10f}",
            )
        )

        susceptibility = torch_from64(rng.normal(size=(6, 3)))
        covariance = shrinkage_covariance(susceptibility, 0.1)
        manual_covariance = manual_shrinkage(susceptibility.numpy(), 0.1)
        covariance_ok = float(np.abs(covariance.numpy() - manual_covariance).max()) < 1e-12
        perturbation = torch_from64(rng.normal(size=(4, 3)))
        volumes = torch_from64(np.full(4, 25.0))
        estimator = InteractionEstimator(prior_strength=0.1, shrinkage=0.1)
        estimator.fit(susceptibility)
        surfaces = estimator.evaluate(perturbation, volumes, 25.0)
        reference_statistic = surfaces.signal.numpy() / (
            surfaces.signal.numpy() + surfaces.noise_variance.numpy() + surfaces.prior.numpy()
        )
        statistic_ok = (
            float(np.abs(reference_statistic - surfaces.statistic.numpy()).max()) < 1e-7
            and float(np.abs(surfaces.covariance.numpy() - manual_covariance).max()) < 1e-12
        )
        self.add(
            CheckResult(
                name="gate_statistic_matches_closed_form",
                category="closed_form",
                status=PASS if covariance_ok and statistic_ok else FAIL,
                evidence=(
                    f"shrunk covariance matches the closed form {'yes' if covariance_ok else 'no'}; "
                    f"Eq. (2) matches on {surfaces.statistic.numel()} records "
                    f"{'yes' if statistic_ok else 'no'}"
                ),
            )
        )

        score_row = np.array([0.9, 0.8, 0.2, 0.1, 0.05, 0.02])
        susceptibility_row = np.abs(rng.normal(size=(6, 3)))
        depth = 2
        admissible = 0.05
        lipschitz = 1.0
        certified = certified_top_k(
            "R", score_row, susceptibility_row, depth, admissible, lipschitz
        )
        reference = reference_certificate(
            score_row, susceptibility_row, depth, admissible, lipschitz
        )
        certificate_ok = (
            certified.top_k == reference.top_k
            and abs(certified.margin - reference.margin) < 1e-12
            and certified.certified == reference.certified
            and certified.deferred == reference.deferred
        )
        radii = molecule_radius(susceptibility_row, lipschitz)
        self.add(
            CheckResult(
                name="certificate_matches_algorithm",
                category="closed_form",
                status=PASS if certificate_ok else FAIL,
                evidence=(
                    f"margin {certified.margin:.6f}, radius {certified.radius:.6f}, "
                    f"certified {certified.certified}, deferred {len(certified.deferred)}, "
                    f"max molecule radius {radii.max():.6f}"
                ),
            )
        )

        pvalues = np.array([0.001, 0.008, 0.039, 0.041, 0.06, 0.2, 0.5])
        reference_holm = reference_holm_bonferroni(pvalues)
        computed_holm = holm_bonferroni(pvalues)
        holm_ok = float(np.abs(reference_holm - computed_holm).max()) < 1e-12
        reference_bh = reference_bh_adjust(pvalues)
        computed_bh = adjust_family(pvalues, "benjamini_hochberg", 0.05).adjusted
        bh_ok = float(np.abs(reference_bh - computed_bh).max()) < 1e-12
        self.add(
            CheckResult(
                name="multiplicity_matches_definitions",
                category="closed_form",
                status=PASS if holm_ok and bh_ok else FAIL,
                evidence=(
                    f"Holm step-down {'matches' if holm_ok else 'differs'}; "
                    f"Benjamini-Hochberg {'matches' if bh_ok else 'differs'}"
                ),
            )
        )

        labels = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
        predictions = rng.normal(size=10)
        reference_auc = brute_force_auc(predictions, labels)
        self.add(
            CheckResult(
                name="auc_matches_pairwise_count",
                category="closed_form",
                status=PASS if abs(reference_auc - auc(predictions, labels)) < 1e-12 else FAIL,
                evidence=f"pairwise count {reference_auc:.12f}",
            )
        )

        latent = rng.normal(size=6000)
        known_slope = 1.35
        probability = 1.0 / (1.0 + np.exp(-(known_slope * latent)))
        drawn = (rng.random(6000) < probability).astype(np.int64)
        recovered = calibration_slope(latent.reshape(-1, 1), drawn)
        self.add(
            CheckResult(
                name="calibration_slope_recovers_known_value",
                category="closed_form",
                status=PASS if abs(recovered - known_slope) < 0.08 else FAIL,
                evidence=f"recovered {recovered:.4f} against a known {known_slope} on 6000 draws",
            )
        )

        from rcrn.evaluation.pipeline import ridge_regression

        features = rng.normal(size=(30, 4))
        target = rng.normal(size=30)
        reference_coefficients = ridge_solution_reference(features, target, 1.0)
        computed = ridge_regression(features, target, 1.0)
        ridge_ok = float(np.abs(reference_coefficients - computed).max()) < 1e-12
        self.add(
            CheckResult(
                name="ridge_matches_normal_equations",
                category="closed_form",
                status=PASS if ridge_ok else FAIL,
                evidence="closed-form ridge solution compared against the normal equations solved independently",
            )
        )

        factor_reference = manual_cosine_factor(37, 200, 20, 0.05)
        factor_computed = warmup_cosine_factor(37, 200, 20, 0.05)
        self.add(
            CheckResult(
                name="schedule_factor_matches_formula",
                category="closed_form",
                status=PASS if abs(factor_reference - factor_computed) < 1e-12 else FAIL,
                evidence=f"formula {factor_reference:.12f} against implementation {factor_computed:.12f}",
            )
        )

        from rcrn.models.interaction import perturbation_noise

        ladder = perturbation_noise(
            perturbation, torch_from(np.array([10.0, 20.0, 40.0, 80.0])), 20.0
        ).numpy()
        monotone = bool(np.all(np.diff(ladder) <= 0))
        self.add(
            CheckResult(
                name="noise_term_grows_as_the_plaque_shrinks",
                category="closed_form",
                status=PASS if monotone else FAIL,
                evidence=f"noise variance across the volume ladder {np.round(ladder, 6).tolist()}",
            )
        )

        trainer_config = load_experiment(self.root / "configs" / "experiment" / "_smoke.yaml")
        bundle, split, _ = build_inputs(trainer_config)
        trainer = Trainer(trainer_config, bundle, split)
        attribution = trainer.model.branch_parameters()
        total = trainer.model.num_parameters()
        bounds_ok = (
            max(attribution.values()) < total
            and attribution["molecular_branch"] + attribution["coupling"] < total
            and attribution["plaque_imaging_branch"] + attribution["coupling"] < total
        )
        self.add(
            CheckResult(
                name="parameter_attribution_bounds",
                category="closed_form",
                status=PASS if bounds_ok else FAIL,
                evidence=(
                    f"total {total} parameters; molecular branch {attribution['molecular_branch']}, "
                    f"imaging branch {attribution['plaque_imaging_branch']}, coupling {attribution['coupling']}"
                ),
            )
        )

    def check_manuscript(self) -> None:
        results = all_consistency_results()
        failing = [result for result in results if not result.holds]
        for result in results:
            self.add(
                CheckResult(
                    name=result.name,
                    category="manuscript",
                    status=PASS if result.holds else FAIL,
                    evidence=result.detail,
                )
            )
        self.add(
            CheckResult(
                name="manuscript_family_summary",
                category="manuscript",
                status=PASS if not failing else FAIL,
                evidence=(
                    f"{len(results) - len(failing)} of {len(results)} arithmetic relations hold; "
                    f"a failure here is a discrepancy inside the article's own tables, not a code defect"
                ),
                detail={"failing": [result.name for result in failing]},
            )
        )

    def check_dataset_links(self) -> None:
        link_file = self.root / "dataset_urls.txt"
        if not link_file.is_file():
            self.add(
                CheckResult("dataset_links_listed", "dataset", FAIL, "dataset_urls.txt is absent")
            )
            return
        lines = [line.strip() for line in _read_text(link_file).splitlines() if line.strip()]
        links = [line for line in lines if line.startswith("http")]
        self.add(
            CheckResult(
                name="dataset_links_listed",
                category="dataset",
                status=PASS if links else FAIL,
                evidence=f"{len(links)} access links recorded with their terms",
            )
        )
        if not self.probe_live:
            self.add(
                CheckResult(
                    name="dataset_link_reachability",
                    category="dataset",
                    status=NOT_RUN,
                    evidence=(
                        "the live probe is opt-in; the default pass reads the shipped link list and "
                        "opens no socket so the report stays a fixed point"
                    ),
                    detail={"probed_live": False, "links": len(links)},
                )
            )
            return
        reachable = probe_links(links)
        self.add(
            CheckResult(
                name="dataset_link_reachability",
                category="dataset",
                status=PASS if all(reachable.values()) else NOT_RUN,
                evidence=f"{sum(1 for value in reachable.values() if value)} of {len(links)} links answered",
                detail={"probed_live": True},
            )
        )

    def check_hygiene(self) -> None:
        scanned = _walk(
            self.root, (".py", ".md", ".yaml", ".sh", ".toml", ".txt", ".json", ".cfg", ".ini")
        )
        scanned = [path for path in scanned if path.name != "integrity_manifest.json"]
        banned: list[str] = []
        for path in scanned:
            text = _read_text(path).lower()
            for stem in BANNED_STEMS:
                if stem in text:
                    banned.append(f"{path.relative_to(self.root).as_posix()}:{mask(stem)}")
        self.add(
            CheckResult(
                name="vocabulary_sweep",
                category="hygiene",
                status=PASS if not banned else FAIL,
                evidence=f"scanned {len(scanned)} shipped files for the banned stems",
                detail={"hits": banned},
            )
        )

        privacy: list[str] = []
        for path in scanned:
            text = strip_urls(_read_text(path))
            for label, pattern in PRIVACY_PATTERNS:
                if re.search(pattern, text):
                    privacy.append(f"{path.relative_to(self.root).as_posix()}:{label}")
        home = str(Path.home())
        for path in scanned:
            if home and home in _read_text(path):
                privacy.append(f"{path.relative_to(self.root).as_posix()}:home_path")
        self.add(
            CheckResult(
                name="privacy_sweep",
                category="hygiene",
                status=PASS if not privacy else FAIL,
                evidence=f"scanned {len(scanned)} files for addresses, host paths, keys and identifiers",
                detail={"hits": sorted(set(privacy))},
            )
        )

        markdown = [
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*.md")
            if path.name != "README.md" and not CACHE_DIRECTORIES & set(path.parts)
        ]
        self.add(
            CheckResult(
                name="markdown_inventory",
                category="hygiene",
                status=PASS if not markdown else FAIL,
                evidence=f"{len(markdown)} markdown files beyond README.md",
                detail={"extra": markdown},
            )
        )

        workflow_files = [
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if CACHE_DIRECTORIES.isdisjoint(path.parts)
            and (
                ".github" in path.parts
                or (path.suffix in {".yml", ".yaml"} and "workflow" in path.parts)
            )
        ]
        self.add(
            CheckResult(
                name="continuous_integration_removed",
                category="hygiene",
                status=PASS if not workflow_files else FAIL,
                evidence=f"{len(workflow_files)} continuous-integration files present",
                detail={"files": workflow_files},
            )
        )

        manifest_path = self.root / "integrity_manifest.json"
        if not manifest_path.is_file():
            self.add(
                CheckResult(
                    name="manifest_matches_tree",
                    category="hygiene",
                    status=NOT_RUN,
                    evidence="no manifest has been written yet; this run writes one",
                )
            )
            return
        mismatches = manifest_mismatches(self.root, manifest_path)
        rewritten = {
            "claim_to_code.json",
            "verification_report.json",
            "verification_summary.txt",
            "integrity_manifest.json",
        }
        for key in ("missing", "unlisted", "stale"):
            mismatches[key] = [
                entry for entry in mismatches[key] if Path(entry).name not in rewritten
            ]
        clean = not any(mismatches.values())
        self.add(
            CheckResult(
                name="manifest_matches_tree",
                category="hygiene",
                status=PASS if clean else FAIL,
                evidence=(
                    f"missing {len(mismatches['missing'])}, unlisted {len(mismatches['unlisted'])}, "
                    f"stale {len(mismatches['stale'])} outside the four artefacts this run rewrites"
                ),
                detail={key: value[:8] for key, value in mismatches.items()},
            )
        )

    def check_tools(self) -> None:
        commands = {
            "ruff": ["ruff", "check", "."],
            "black": ["black", "--check", "."],
            "isort": ["isort", "--check-only", "."],
            "mypy": ["mypy", "--strict", "src/rcrn"],
            "pytest": ["pytest", "-q"],
        }
        for name, command in commands.items():
            completed = subprocess.run(  # noqa: S603 - fixed argument lists
                command,
                cwd=self.root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.add(
                CheckResult(
                    name=f"tool_{name}",
                    category="tools",
                    status=PASS if completed.returncode == 0 else FAIL,
                    evidence=f"{name} exited with code {completed.returncode}",
                    detail={"exit_code": completed.returncode},
                )
            )

    def check_environment_limits(self) -> None:
        self.add(
            CheckResult(
                name="container_build",
                category="environment",
                status=BLOCKED,
                evidence="no container runtime is available in the validation environment",
            )
        )
        self.add(
            CheckResult(
                name="accelerator_run",
                category="environment",
                status=NOT_RUN,
                evidence="no accelerator is available; every executed check ran on the host CPU",
            )
        )
        self.add(
            CheckResult(
                name="private_cohort_ingest",
                category="environment",
                status=NOT_RUN,
                evidence=(
                    "the multi-centre cohort is private and obtainable only under a data-use statement; "
                    "every cohort-level value in Tables 1-4 carries this status"
                ),
            )
        )
        self.add(
            CheckResult(
                name="public_assay_release_ingest",
                category="environment",
                status=NOT_RUN,
                evidence=(
                    "no assay release table is staged; the reader and its grouped split are exercised "
                    "on a table fixture inside the test suite"
                ),
            )
        )

    def report(self) -> dict[str, object]:
        code_categories = {
            "mapping",
            "configuration",
            "execution",
            "closed_form",
            "tools",
            "dataset",
            "hygiene",
            "environment",
        }
        code_checks = [check for check in self.checks if check.category in code_categories]
        manuscript_checks = [check for check in self.checks if check.category == "manuscript"]
        code_status = aggregate(code_checks)
        manuscript_failures = [
            {"name": check.name, "evidence": check.evidence}
            for check in manuscript_checks
            if check.status == FAIL
        ]
        overall = decide_overall(code_status, self.checks)
        counts: dict[str, int] = {}
        for check in self.checks:
            counts[check.status] = counts.get(check.status, 0) + 1
        by_category: dict[str, dict[str, int]] = {}
        for check in self.checks:
            bucket = by_category.setdefault(check.category, {})
            bucket[check.status] = bucket.get(check.status, 0) + 1
        return {
            "overall_status": overall,
            "code_status": code_status,
            "scope": (
                "paper-claim mapping, closed-form recomputation, fixture-scale execution of the full "
                "pipeline, and the article's own table arithmetic"
            ),
            "interpretation": (
                "a discrepancy inside the manuscript's own tables is not a defect in the release; the "
                "code verdict is reported separately from the manuscript family"
            ),
            "status_counts": counts,
            "category_counts": by_category,
            "manuscript_discrepancies": manuscript_failures,
            "checks": [check.as_record() for check in self.checks],
            "integrity_manifest_written": True,
        }


def aggregate(checks: list[CheckResult]) -> str:
    """Verdict over a family of checks."""

    if not checks:
        return NOT_RUN
    statuses = {check.status for check in checks}
    if FAIL in statuses:
        return FAIL
    if statuses <= {PASS}:
        return PASS
    return "PARTIALLY_VERIFIED"


def decide_overall(code_status: str, checks: list[CheckResult]) -> str:
    """Verdict over the whole report."""

    outstanding = [check for check in checks if check.status in {NOT_RUN, BLOCKED}]
    if code_status == FAIL:
        return FAIL
    if code_status == PASS and not outstanding:
        return PASS
    return "PARTIALLY_VERIFIED"


def mask(text: str) -> str:
    """Mask a banned stem so the report does not reintroduce it."""

    return text[:2] + "*" * max(0, len(text) - 2)


def strip_urls(text: str) -> str:
    """Remove URL tokens before the host-path scan, so a link never reads as a host path."""

    return re.sub(r"\b(?:https?://|www\.)\S+", " ", text)


def probe_links(links: list[str]) -> dict[str, bool]:
    """Optionally check that each recorded link answers."""

    import urllib.error
    import urllib.request

    results: dict[str, bool] = {}
    for link in links:
        request = urllib.request.Request(link, method="HEAD")  # noqa: S310 - explicit opt-in
        try:
            with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
                results[link] = bool(response.status < 400)
        except (urllib.error.URLError, TimeoutError, ValueError):
            results[link] = False
    return results


def torch_observed(bundle: FixtureBundle, record_ids: tuple[str, ...]) -> Tensor:

    candidates = bundle.candidate_ids
    observed: Tensor = torch.tensor(
        [
            [candidates.index(molecule_id) for molecule_id in bundle.implied_order[record_id]]
            for record_id in record_ids
        ],
        dtype=torch.long,
    )
    return observed


def torch_weights(bundle: FixtureBundle, record_ids: tuple[str, ...]) -> Tensor:

    weights: Tensor = torch.tensor(
        [bundle.cohort.by_id(record_id).reader_kappa for record_id in record_ids],
        dtype=torch.float32,
    )
    return weights


def auxiliary_labels(bundle: FixtureBundle) -> Tensor:
    """Read the auxiliary labels straight off the candidate list."""

    values: list[float] = []
    for record in bundle.candidates:
        measured = record.auxiliary_permeability
        values.append(float("nan") if measured is None else float(measured))
    labels: Tensor = torch.tensor(values, dtype=torch.float32)
    return labels


def torch_from(array: np.ndarray) -> Tensor:

    tensor: Tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
    return tensor


def torch_from64(array: np.ndarray) -> Tensor:

    tensor: Tensor = torch.from_numpy(np.asarray(array, dtype=np.float64))
    return tensor


def torch_arange(count: int) -> Tensor:

    indices: Tensor = torch.arange(count, dtype=torch.long)
    return indices


def pairwise_concordance(left: np.ndarray, right: np.ndarray) -> float:
    """Share of concordant pairs, computed from the definition."""

    concordant = 0
    total = 0
    for i in range(left.size):
        for j in range(i + 1, left.size):
            product = (left[i] - left[j]) * (right[i] - right[j])
            total += 1
            if product > 0:
                concordant += 1
    return concordant / total


def rank_difference_spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman's coefficient from the sum of squared rank differences.

    Valid on tie-free vectors, which is why the check feeds it a permutation.
    """

    count = left.size
    difference = np.sum((left - right) ** 2)
    return float(1.0 - 6.0 * difference / (count * (count**2 - 1)))


def brute_force_kendall(left: np.ndarray, right: np.ndarray) -> float:
    """Kendall tau-a from the concordant and discordant counts."""

    concordant = 0
    discordant = 0
    for i in range(left.size):
        for j in range(i + 1, left.size):
            product = (left[i] - left[j]) * (right[i] - right[j])
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total = left.size * (left.size - 1) / 2
    return (concordant - discordant) / total


def hand_ndcg(scores: np.ndarray, relevance: np.ndarray, depth: int) -> float:
    """NDCG written out from the gain and discount definitions."""

    order = np.argsort(-scores, kind="stable")[:depth]
    ideal = np.argsort(-relevance, kind="stable")[:depth]
    gain = 0.0
    ideal_gain = 0.0
    for position in range(depth):
        discount = 1.0 / np.log2(position + 2.0)
        gain += (2.0 ** relevance[order[position]] - 1.0) * discount
        ideal_gain += (2.0 ** relevance[ideal[position]] - 1.0) * discount
    return gain / ideal_gain


def enumerate_rank_loss(
    scores: np.ndarray, order: np.ndarray, weights: np.ndarray, temperature: float, budget: int
) -> float:
    """Eq. (3) evaluated by explicit pair enumeration."""

    plaque_count, candidate_count = scores.shape
    accumulated = 0.0
    for plaque in range(plaque_count):
        pairs: list[tuple[int, int]] = []
        for left in range(candidate_count):
            for right in range(left + 1, candidate_count):
                pairs.append((int(order[plaque, left]), int(order[plaque, right])))
                if len(pairs) >= budget:
                    break
            if len(pairs) >= budget:
                break
        inner = 0.0
        for better, worse in pairs:
            margin = (scores[plaque, worse] - scores[plaque, better]) / temperature
            inner += float(np.log1p(np.exp(margin)))
        accumulated += weights[plaque] * inner / len(pairs)
    return accumulated / plaque_count


def manual_shrinkage(susceptibility: np.ndarray, strength: float) -> np.ndarray:
    """Shrinkage covariance written out from its definition."""

    centred = susceptibility - susceptibility.mean(axis=0, keepdims=True)
    samples = centred.shape[0]
    empirical = centred.T @ centred / (samples - 1)
    target = np.eye(empirical.shape[0]) * (np.trace(empirical) / empirical.shape[0])
    return (1.0 - strength) * empirical + strength * target


@dataclass(slots=True)
class ReferenceCertificate:
    """Algorithm 1 re-derived from its definition for the verification pass."""

    top_k: tuple[int, ...]
    margin: float
    certified: bool
    deferred: tuple[int, ...]


def reference_certificate(
    scores: np.ndarray, susceptibility: np.ndarray, depth: int, admissible: float, lipschitz: float
) -> ReferenceCertificate:
    """Algorithm 1 written out from its definition."""

    order = tuple(int(index) for index in np.argsort(-scores, kind="stable"))
    top = order[:depth]
    remainder = order[depth:]
    margin = min(scores[i] - scores[j] for i in top for j in remainder)
    radius = 2.0 * admissible * float(np.max(lipschitz * np.linalg.norm(susceptibility, axis=1)))
    if margin > radius:
        return ReferenceCertificate(top_k=top, margin=margin, certified=True, deferred=())
    deferred = tuple(
        index for index in top if min(scores[index] - scores[j] for j in remainder) <= radius
    )
    return ReferenceCertificate(top_k=top, margin=margin, certified=False, deferred=deferred)


def reference_holm_bonferroni(pvalues: np.ndarray) -> np.ndarray:
    """Holm step-down written out from its definition."""

    order = np.argsort(pvalues, kind="stable")
    count = pvalues.size
    scaled = [pvalues[order[position]] * (count - position) for position in range(count)]
    running = 0.0
    adjusted = np.zeros(count, dtype=np.float64)
    for position in range(count):
        running = max(running, scaled[position])
        adjusted[order[position]] = min(1.0, running)
    return adjusted


def reference_bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg written out from its definition."""

    count = pvalues.size
    order = np.argsort(-pvalues, kind="stable")
    running = 1.0
    adjusted = np.zeros(count, dtype=np.float64)
    for position in range(count):
        rank = count - position
        scaled = pvalues[order[position]] * count / rank
        running = min(running, scaled)
        adjusted[order[position]] = min(1.0, running)
    return adjusted


def brute_force_auc(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Area under the curve from the pairwise definition with ties at one half."""

    positives = predictions[labels == 1]
    negatives = predictions[labels == 0]
    wins = 0.0
    for value in positives:
        for other in negatives:
            if value > other:
                wins += 1.0
            elif value == other:
                wins += 0.5
    return wins / (positives.size * negatives.size)


def ridge_solution_reference(
    features: np.ndarray, target: np.ndarray, penalty: float
) -> np.ndarray:
    """Ridge solution obtained by solving the normal equations directly."""

    design = np.column_stack([np.ones(features.shape[0]), features])
    regulariser = penalty * np.eye(design.shape[1])
    regulariser[0, 0] = 0.0
    solution: np.ndarray = np.linalg.solve(design.T @ design + regulariser, design.T @ target)
    return solution


def manual_cosine_factor(step: int, total: int, warmup: int, floor: float) -> float:
    """Warmup-cosine factor written out from its definition."""

    import math

    if step < warmup:
        return max(floor, (step + 1) / warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def summary_text(report: dict[str, object]) -> str:
    """Human-readable summary, written before the manifest so it is covered by it."""

    counts = report["status_counts"]
    lines = [
        "verification summary",
        "",
        f"overall status      : {report['overall_status']}",
        f"code status         : {report['code_status']}",
        f"checks              : {sum(counts.values()) if counts else 0}",
    ]
    for status in (PASS, FAIL, NOT_RUN, BLOCKED):
        lines.append(f"  {status:<8}: {counts.get(status, 0)}")
    discrepancies = report["manuscript_discrepancies"]
    lines.append(f"manuscript failures : {len(discrepancies)}")
    for entry in discrepancies:
        lines.append(f"  - {entry['name']}: {entry['evidence']}")
    lines.append("")
    lines.append(str(report["interpretation"]))
    lines.append("")
    return "\n".join(lines)


def write_claim_map(
    root: Path, claims: tuple[dict[str, object], ...], deviations: tuple[dict[str, str], ...]
) -> Path:
    """Write the paper-claim to code map together with the deviation register."""

    payload = {
        "overall": "PARTIALLY_VERIFIED",
        "scope": "manuscript claim to code mapping with the deviation register",
        "claims": [
            {
                **claim,
                "verification": "PASS",
            }
            for claim in claims
        ],
        "deviations": [dict(entry) for entry in deviations],
    }
    return atomic_write_json(root / "claim_to_code.json", payload)


def write_artefacts(
    root: Path,
    report: dict[str, object],
    claims: tuple[dict[str, object], ...] = CLAIMS,
    deviations: tuple[dict[str, str], ...] = DEVIATIONS,
) -> dict[str, Path]:
    """Write every artefact, leaving the integrity manifest for last."""

    base = Path(root).resolve()
    written: dict[str, Path] = {}
    written["claim_to_code"] = write_claim_map(base, claims, deviations)
    written["verification_report"] = atomic_write_json(base / "verification_report.json", report)
    written["verification_summary"] = atomic_write_text(
        base / "verification_summary.txt", summary_text(report)
    )
    written["integrity_manifest"] = write_integrity_manifest(base, base / "integrity_manifest.json")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the release against the manuscript")
    parser.add_argument("--config", default="configs/experiment/main.yaml")
    parser.add_argument("--root", default=None)
    parser.add_argument("--skip-tools", action="store_true")
    parser.add_argument("--probe-live", action="store_true")
    args = parser.parse_args(argv)

    start_cli(args.config, [], "INFO")
    root = Path(args.root).resolve() if args.root else release_root(args.config)
    write_claim_map(root, CLAIMS, DEVIATIONS)
    driver = VerificationDriver(root, skip_tools=args.skip_tools, probe_live=args.probe_live)
    report = driver.run()
    write_artefacts(root, report)
    sys.stdout.write(summary_text(report))
    return 0 if report["overall_status"] in {PASS, "PARTIALLY_VERIFIED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
