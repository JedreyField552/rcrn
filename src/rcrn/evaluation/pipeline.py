"""Evaluation pipeline over the fixture cohort.

Ref: Table 1 (rank agreement and NDCG@10 per stratum); Table 2 (component
necessity); Figure 1 (condition dependence); Table 3 (transport); Table 4 (clinical
arms and comparators); Results, Section 2.6 (reader agreement); Section 2.4 (assay
endpoint).

Every number this module produces is computed on the cohort fixture at the scale the
configuration declares. It reproduces the *mechanisms* of each table, and the
manuscript's own values stay reference values reported alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from rcrn.data.fixture import FixtureBundle, route_share
from rcrn.data.splits import SplitPlan, plan_split
from rcrn.evaluation.ablation import (
    AblationRow,
    interaction_ratio,
    parameter_share,
    retained_share,
)
from rcrn.evaluation.assay import FamilyError, error_table
from rcrn.evaluation.external import acquisition_axis_attribution, fold_comparison
from rcrn.evaluation.invariants import (
    InvariantOutcome,
    composition_index_is_image_only,
    external_fold_unfitted,
    label_pipeline_independent,
    prohibited_features_absent,
)
from rcrn.evaluation.multiplicity import adjust_family
from rcrn.evaluation.reader import ReaderReading, agreement_ladder, expertise_stratum_effect
from rcrn.metrics.certificate import certified_batch, flip_rate_under_perturbation
from rcrn.metrics.clinical import incremental_auc, mcid_crossing_share, net_benefit, weighted_pool
from rcrn.metrics.ranking import (
    ndcg_at_k,
    normalised_rank_agreement,
    relevance_from_order,
    spearman_rho,
)
from rcrn.metrics.transport import transport_row, transport_summary
from rcrn.models.interaction import InteractionSurfaces
from rcrn.models.rcrn import RCRNModel
from rcrn.models.scorer import GatedScorer
from rcrn.training.trainer import GateFitting, Trainer
from rcrn.utils.config import Config

COMPARATORS: tuple[str, ...] = (
    "rcrn",
    "unconstrained_twin",
    "intermediate_fusion",
    "imaging_encoder",
    "state_space_encoder",
    "clinical_nomogram",
)


@dataclass(slots=True)
class TransportCell:
    """One fold's record count, agreement pair and degradation class."""

    records: int
    in_distribution: float
    out_of_distribution: float
    paired_delta_pp: float
    marginal_delta_pp: float
    degradation: str

    def as_record(self) -> dict[str, object]:
        return {
            "records": float(self.records),
            "in_distribution": self.in_distribution,
            "out_of_distribution": self.out_of_distribution,
            "paired_delta_pp": self.paired_delta_pp,
            "marginal_delta_pp": self.marginal_delta_pp,
            "degradation": self.degradation,
        }


@dataclass(slots=True)
class EvaluationReport:
    """Everything the evaluation produced, ready to serialise."""

    rank_agreement: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    component_necessity: tuple[AblationRow, ...] = ()
    interaction_ratios: dict[str, float] = field(default_factory=dict)
    interaction_ratio_excluded: dict[str, str] = field(default_factory=dict)
    condition_dependence: dict[str, dict[str, float]] = field(default_factory=dict)
    transport: dict[str, TransportCell] = field(default_factory=dict)
    clinical: dict[str, dict[str, float]] = field(default_factory=dict)
    reader: dict[str, float] = field(default_factory=dict)
    assay: dict[str, dict[str, float]] = field(default_factory=dict)
    certificate: dict[str, float] = field(default_factory=dict)
    invariants: tuple[InvariantOutcome, ...] = ()
    multiplicity: dict[str, float] = field(default_factory=dict)
    scale_note: str = ""

    def as_record(self) -> dict[str, object]:
        return {
            "rank_agreement": self.rank_agreement,
            "component_necessity": [
                {
                    "tier": row.tier,
                    "variant": row.variant,
                    "metric": row.metric,
                    "delta": row.delta,
                    "parameters_m": row.parameters_m,
                    "linked_equation": row.linked_equation,
                }
                for row in self.component_necessity
            ],
            "interaction_ratios": self.interaction_ratios,
            "interaction_ratio_excluded": self.interaction_ratio_excluded,
            "condition_dependence": self.condition_dependence,
            "transport": {name: cell.as_record() for name, cell in self.transport.items()},
            "clinical": self.clinical,
            "reader": self.reader,
            "assay": self.assay,
            "certificate": self.certificate,
            "invariants": [outcome.as_record() for outcome in self.invariants],
            "multiplicity": self.multiplicity,
            "scale_note": self.scale_note,
        }


def orderings_for(
    trainer: Trainer, record_ids: tuple[str, ...]
) -> tuple[dict[str, tuple[int, ...]], dict[str, float]]:
    """Score every record and return its candidate ordering plus the gate statistic."""

    output = trainer.infer(record_ids)
    fitting = trainer.fit_gate(
        float(trainer.config.get("objective.gate.false_conditioning_tolerance", 0.05))
    )
    decision = fitting.gate.decide(_surfaces_for(trainer, fitting, record_ids), None)
    scorer = GatedScorer(trainer.bundle.candidate_ids, fitting.gate)
    orderings = scorer.order(record_ids, output.coupling, None, decision)
    return (
        {ordering.record_id: ordering.order for ordering in orderings},
        {ordering.record_id: ordering.statistic for ordering in orderings},
    )


def _surfaces_for(
    trainer: Trainer, fitting: GateFitting, record_ids: tuple[str, ...]
) -> InteractionSurfaces:
    """Recompute the interaction surfaces for a record selection."""

    output = trainer.infer(record_ids)
    volume_mm3 = np.array(
        [trainer.bundle.cohort.by_id(record_id).plaque_volume_mm3 for record_id in record_ids],
        dtype=np.float32,
    )
    return fitting.estimator.evaluate(
        output.resistance.perturbation.cpu(),
        _as_tensor(volume_mm3),
        fitting.reference_volume_mm3,
    )


def _as_tensor(values: np.ndarray) -> Tensor:
    return torch.from_numpy(values)


def stratum_agreement(
    orderings: dict[str, tuple[int, ...]],
    bundle: FixtureBundle,
    record_ids: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Rank agreement and NDCG@10 per stratum on the fixture scale."""

    strata: dict[str, list[str]] = {"all_plaques": list(record_ids)}
    for spec in ("rads_1_2", "rads_3_4", "rads_3_4_iph", "iph_positive"):
        strata[spec] = [
            record_id
            for record_id in record_ids
            if spec in bundle.cohort.by_id(record_id).stratum_keys()
        ]
    lookup = {molecule_id: position for position, molecule_id in enumerate(bundle.candidate_ids)}
    table: dict[str, dict[str, float]] = {}
    for name, members in strata.items():
        if not members:
            table[name] = {"records": 0.0, "rho": 0.0, "ndcg_at_10": 0.0}
            continue
        rho_values: list[float] = []
        ndcg_values: list[float] = []
        for record_id in members:
            order = orderings[record_id]
            observed = np.array(
                [lookup[molecule_id] for molecule_id in bundle.implied_order[record_id]],
                dtype=np.float64,
            )
            relevance = np.array(
                [len(observed) - lookup[molecule_id] for molecule_id in bundle.candidate_ids],
                dtype=np.float64,
            )
            score_vector = np.empty(len(bundle.candidate_ids), dtype=np.float64)
            for position, index in enumerate(order):
                score_vector[index] = len(order) - position
            rho_values.append(spearman_rho(score_vector, relevance))
            ndcg_values.append(
                ndcg_at_k(score_vector, relevance, min(10, len(bundle.candidate_ids)))
            )
        table[name] = {
            "records": float(len(members)),
            "rho": float(np.mean(rho_values)),
            "rho_std": float(np.std(rho_values, ddof=1)) if len(rho_values) > 1 else 0.0,
            "ndcg_at_10": float(np.mean(ndcg_values)),
        }
    return table


def condition_dependence(
    orderings: dict[str, tuple[int, ...]],
    bundle: FixtureBundle,
    record_ids: tuple[str, ...],
    bin_count: int = 4,
) -> dict[str, dict[str, float]]:
    """Conditioning gain by composition quartile and by interaction-signal bin.

    Ref: Figure 1. The left panel groups records by the composition index, which is
    derived from imaging alone; the right panel groups them by the share of the
    score change the extracellular-matrix route carries.
    """

    composition = np.array(
        [float(bundle.true_composition[record_id]) for record_id in record_ids], dtype=np.float64
    )
    quartile_edges = np.quantile(composition, np.linspace(0.0, 1.0, bin_count + 1))
    route_share_matrix = np.vstack(
        [route_share(bundle.true_resistance[record_id]) for record_id in record_ids]
    )
    matrix_share = route_share_matrix[:, 0]
    share_edges = np.quantile(matrix_share, np.linspace(0.0, 1.0, bin_count + 1))
    lookup = {molecule_id: position for position, molecule_id in enumerate(bundle.candidate_ids)}
    output: dict[str, dict[str, float]] = {
        "composition_quartile": {},
        "interaction_signal_bin": {},
    }
    for label, edges, values in (
        ("composition_quartile", quartile_edges, composition),
        ("interaction_signal_bin", share_edges, matrix_share),
    ):
        for bucket in range(bin_count):
            lower, upper = edges[bucket], edges[bucket + 1]
            members = [
                record_id
                for position, record_id in enumerate(record_ids)
                if lower <= values[position] <= upper
            ]
            if not members:
                output[label][f"bin_{bucket}"] = {"records": 0.0, "gain": 0.0, "matrix_share": 0.0}
                continue
            gains: list[float] = []
            blind_agreement: list[float] = []
            for record_id in members:
                order = orderings[record_id]
                predicted = np.empty(len(bundle.candidate_ids), dtype=np.float64)
                for position, index in enumerate(order):
                    predicted[index] = len(order) - position
                blind = np.array(
                    [-lookup[molecule_id] for molecule_id in bundle.candidate_ids], dtype=np.float64
                )
                observed = np.array(
                    [lookup[molecule_id] for molecule_id in bundle.implied_order[record_id]],
                    dtype=np.float64,
                )
                gains.append(spearman_rho(predicted, observed) - spearman_rho(blind, observed))
                blind_agreement.append(normalised_rank_agreement(predicted, blind))
            output[label][f"bin_{bucket}"] = {
                "records": float(len(members)),
                "gain": float(np.mean(gains)),
                "blind_vs_conditioned_agreement": float(np.mean(blind_agreement)),
                "matrix_share": float(
                    np.mean([matrix_share[record_ids.index(rid)] for rid in members])
                ),
            }
    return output


def transport_table(
    orderings: dict[str, tuple[int, ...]],
    bundle: FixtureBundle,
    split: SplitPlan,
    reference_ids: tuple[str, ...],
) -> dict[str, TransportCell]:
    """Degradation of the ordering on each transport fold.

    Ref: Table 3 and its caption. The fold memberships come from acquisition
    metadata only. The reported change is the mean paired difference, taken against
    each record's own site-matched reference rather than against the pooled
    reference; that is why it can differ from the difference of the two marginal
    columns, and both are carried.
    """

    lookup = {molecule_id: position for position, molecule_id in enumerate(bundle.candidate_ids)}
    reference_rho = {
        record_id: spearman_rho(
            _score_vector(orderings[record_id], len(bundle.candidate_ids)),
            _relevance(bundle, lookup, record_id),
        )
        for record_id in reference_ids
    }
    site_values: dict[str, list[float]] = {}
    for record_id, value in reference_rho.items():
        site_values.setdefault(bundle.cohort.by_id(record_id).site, []).append(value)
    site_mean = {site: float(np.mean(values)) for site, values in site_values.items()}
    pooled_reference = float(np.mean(list(reference_rho.values()))) if reference_rho else 0.0
    rows: dict[str, TransportCell] = {}
    for condition, members in split.fold_ids.items():
        shared = [record_id for record_id in members if record_id in orderings]
        if not shared:
            rows[condition] = TransportCell(
                records=0,
                in_distribution=pooled_reference,
                out_of_distribution=pooled_reference,
                paired_delta_pp=0.0,
                marginal_delta_pp=0.0,
                degradation="not_run",
            )
            continue
        fold_values = [
            spearman_rho(
                _score_vector(orderings[record_id], len(bundle.candidate_ids)),
                _relevance(bundle, lookup, record_id),
            )
            for record_id in shared
        ]
        deltas = np.array(
            [
                value - site_mean.get(bundle.cohort.by_id(record_id).site, pooled_reference)
                for record_id, value in zip(shared, fold_values, strict=True)
            ],
            dtype=np.float64,
        )
        row = transport_row(
            condition,
            len(shared),
            pooled_reference,
            float(np.mean(fold_values)),
            deltas,
        )
        rows[condition] = TransportCell(
            records=row.records,
            in_distribution=row.in_distribution,
            out_of_distribution=row.out_of_distribution,
            paired_delta_pp=row.paired_delta_pp,
            marginal_delta_pp=row.marginal_delta_pp,
            degradation=row.degradation,
        )
    return rows


def _score_vector(order: tuple[int, ...], width: int) -> np.ndarray:
    vector = np.empty(width, dtype=np.float64)
    for position, index in enumerate(order):
        vector[index] = width - position
    return vector


def _relevance(bundle: FixtureBundle, lookup: dict[str, int], record_id: str) -> np.ndarray:
    return np.array(
        [len(lookup) - lookup[molecule_id] for molecule_id in bundle.candidate_ids],
        dtype=np.float64,
    )


def clinical_arm_table(
    orderings: dict[str, tuple[int, ...]],
    bundle: FixtureBundle,
    split: SplitPlan,
) -> dict[str, dict[str, float]]:
    """Per-arm and per-site rank agreement on the fixture scale."""

    lookup = {molecule_id: position for position, molecule_id in enumerate(bundle.candidate_ids)}
    groups: dict[str, list[str]] = {"pooled": list(orderings)}
    for arm in ("prospective", "retrospective"):
        groups[arm] = [
            record_id for record_id in orderings if bundle.cohort.by_id(record_id).arm == arm
        ]
    for site in sorted({record.site for record in bundle.cohort.records}):
        groups[f"site_{site}"] = [
            record_id for record_id in orderings if bundle.cohort.by_id(record_id).site == site
        ]
    for region in ("I", "II"):
        groups[f"region_{region}"] = [
            record_id for record_id in orderings if bundle.cohort.by_id(record_id).region == region
        ]
    table: dict[str, dict[str, float]] = {}
    for name, members in groups.items():
        if not members:
            table[name] = {"records": 0.0, "rho": 0.0}
            continue
        values = [
            spearman_rho(
                _score_vector(orderings[record_id], len(bundle.candidate_ids)),
                _relevance(bundle, lookup, record_id),
            )
            for record_id in members
        ]
        table[name] = {"records": float(len(members)), "rho": float(np.mean(values))}
    del split
    return table


def clinical_increment_table(bundle: FixtureBundle, seed: int = 7) -> dict[str, dict[str, float]]:
    """Incremental area under the curve and net benefit on a fixture endpoint.

    The baseline is the composition index alone and the augmented score adds the
    matrix-route share; both are read out of the fixture, so the mechanism the
    manuscript describes is exercised end to end.
    """

    rng = np.random.default_rng(seed)
    record_ids = tuple(record.record_id for record in bundle.cohort.records)
    composition = np.array([float(bundle.true_composition[r]) for r in record_ids])
    matrix_share = np.array([float(route_share(bundle.true_resistance[r])[0]) for r in record_ids])
    labels = np.array(
        [1 if bundle.cohort.by_id(r).outcome.event_observed else 0 for r in record_ids],
        dtype=np.int64,
    )
    noise = rng.normal(0.0, 0.05, size=composition.shape)
    baseline = composition + noise
    augmented = baseline + 0.35 * matrix_share
    increment = incremental_auc(baseline, augmented, labels)
    return {
        "pooled": {
            "baseline_auc": increment["baseline_auc"],
            "augmented_auc": increment["augmented_auc"],
            "increment": increment["increment"],
            "increment_se": increment["standard_error"],
            "net_benefit_20": net_benefit(1.0 / (1.0 + np.exp(-augmented)), labels, 0.2),
        }
    }


def pooled_arm_consistency(arms: dict[str, dict[str, float]]) -> dict[str, float]:
    """Size-weighted pool of the arm values, recomputed from the arm entries."""

    sizes = np.array([arms[name]["records"] for name in ("prospective", "retrospective")])
    values = np.array([arms[name]["rho"] for name in ("prospective", "retrospective")])
    return {"pooled_rho": weighted_pool(values, sizes), "records": float(sizes.sum())}


def necessity_sweep(
    config: Config,
    bundle: FixtureBundle,
    split: SplitPlan,
    variants: dict[str, dict[str, object]],
    steps: int,
) -> tuple[tuple[AblationRow, ...], dict[str, float], dict[str, str]]:
    """Train each variant briefly and record its fixture-scale agreement.

    Ref: Table 2. The sweep exercises the removals and substitutions; the values it
    returns are fixture-scale and are labelled as such in the report. The full model
    is always evaluated first so every delta is taken against the same reference.
    """

    ordered_names = ["full"] + [name for name in variants if name != "full"]
    evaluation_ids = (split.external_ids or split.all_fit())[:32]
    metrics: dict[str, float] = {}
    rows: list[AblationRow] = []
    for name in ordered_names:
        overrides = variants[name]
        variant_config = config.with_overrides(_flatten_overrides(overrides))
        trainer = Trainer(variant_config, bundle, split)
        trainer.run_steps(steps)
        orderings, _ = orderings_for(trainer, evaluation_ids)
        table = stratum_agreement(orderings, bundle, tuple(orderings))
        metric = float(table.get("rads_3_4_iph", table["all_plaques"])["rho"])
        metrics[name] = metric
        rows.append(
            AblationRow(
                tier=str(overrides.get("tier", "tier_1_removal")),
                variant=name,
                metric=metric,
                delta=metric - metrics["full"],
                parameters_m=trainer.model.num_parameters() / 1e6,
                linked_equation=str(overrides.get("linked_equation", "eq_objective")),
            )
        )
    ratios, excluded = _interaction_ratios(tuple(rows))
    return tuple(rows), ratios, excluded


def _flatten_overrides(overrides: dict[str, object]) -> list[str]:
    pairs: list[str] = []
    for key, value in overrides.items():
        if key in {"tier", "linked_equation"}:
            continue
        if isinstance(value, bool):
            pairs.append(f"{key}={str(value).lower()}")
        elif isinstance(value, (int, float)):
            pairs.append(f"{key}={value}")
        else:
            pairs.append(f"{key}={value}")
    return pairs


INTERACTION_PAIRS: dict[str, tuple[str, str]] = {
    "coupling_and_gate": ("no_reaction_norm_coupling", "no_gate"),
    "coupling_and_composition": ("no_reaction_norm_coupling", "no_composition_head"),
    "route_and_composition": ("scalar_susceptibility", "no_composition_head"),
    "route_and_rank": ("scalar_susceptibility", "uniform_rank_weight"),
}


def _interaction_ratios(
    rows: tuple[AblationRow, ...],
) -> tuple[dict[str, float], dict[str, str]]:
    """Joint loss over the sum of the two separate losses, per Tier 2 row.

    A ratio is only defined when both separate removals cost something. A pair
    whose single loss is null is reported as excluded with its reason rather than
    silently dropped, because that null is the Tier 1 result the manuscript prints.
    """

    by_variant = {row.variant: row for row in rows}
    ratios: dict[str, float] = {}
    excluded: dict[str, str] = {}
    for name, members in INTERACTION_PAIRS.items():
        joint = by_variant.get(f"joint_{name}")
        first = by_variant.get(members[0])
        second = by_variant.get(members[1])
        if joint is None or first is None or second is None:
            excluded[name] = "the joint row or one of its single rows was not evaluated"
            continue
        parts = (-first.delta, -second.delta)
        if min(parts) <= 0.0:
            excluded[name] = (
                "a single removal costs nothing at this scale, so the ratio is undefined"
            )
            continue
        ratios[name] = float(interaction_ratio(-joint.delta, parts))
    return ratios, excluded


def diversity_metrics(routes: np.ndarray) -> dict[str, float]:
    """Dispersion of the route coordinates, used by the outlook analysis."""

    return {
        "mean_route_dispersion": float(np.std(routes, axis=1).mean()),
        "max_route_dispersion": float(np.std(routes, axis=1).max()),
    }


def certificate_summary(
    orderings: dict[str, tuple[int, ...]],
    trainer: Trainer,
    depth: int,
    admissible: float,
    lipschitz: float,
) -> dict[str, float]:
    """Deferral rate and certified share under the declared perturbation radius."""

    record_ids = tuple(orderings)
    output = trainer.infer(record_ids)
    scores = np.vstack(
        [
            _score_vector(orderings[record_id], len(trainer.bundle.candidate_ids))
            for record_id in record_ids
        ]
    )
    susceptibility = output.coupling.susceptibility.detach().cpu().numpy()
    width = len(trainer.bundle.candidate_ids)
    resolved_depth = max(1, min(depth, width - 1))
    report = certified_batch(
        record_ids, scores, susceptibility, resolved_depth, admissible, lipschitz
    )
    return {
        "records": float(len(record_ids)),
        "deferral_rate": report.deferral_rate,
        "certified_share": report.certified_share,
        "admissible_perturbation": admissible,
        "depth": float(resolved_depth),
    }


def perturbation_sweep_summary(
    clean: np.ndarray, perturbed: np.ndarray, depth: int
) -> dict[str, float]:
    """Empirical top-k flip rate after an input perturbation."""

    return {"flip_rate": flip_rate_under_perturbation(clean, perturbed, depth)}


def reader_table(
    orderings: dict[str, tuple[int, ...]], bundle: FixtureBundle, seed: int = 11
) -> dict[str, float]:
    """Reader agreement on fixture readings with and without the ordering.

    Six readers of three expertise levels order a subset of the records; the
    assisted reading follows the model ordering with a reader-specific amount of
    noise drawn from the record's agreement coefficient.
    """

    rng = np.random.default_rng(seed)
    selected = tuple(sorted(orderings))[: min(60, len(orderings))]
    expertise = ("expert", "intermediate", "novice")
    readings: list[ReaderReading] = []
    width = len(bundle.candidate_ids)
    for reader_index in range(6):
        level = expertise[reader_index % len(expertise)]
        for record_id in selected:
            kappa = bundle.cohort.by_id(record_id).reader_kappa
            unassisted_scores = rng.normal(0.0, 1.0 / max(kappa, 1e-3), size=width)
            unassisted = np.argsort(-unassisted_scores, kind="stable")
            model_order = np.array(orderings[record_id], dtype=np.int64)
            noise = rng.normal(0.0, 1.0 - min(0.95, kappa), size=width)
            assisted = np.argsort(
                -(np.array([width - position for position in np.argsort(model_order)]) + noise),
                kind="stable",
            )
            readings.append(
                ReaderReading(
                    reader=f"R{reader_index + 1}",
                    expertise=level,
                    record_id=record_id,
                    unassisted=np.asarray(unassisted, dtype=np.int64),
                    assisted=np.asarray(assisted, dtype=np.int64),
                )
            )
    before, after = agreement_ladder(tuple(readings))
    return {
        "readers": 6.0,
        "records": float(len(selected)),
        "agreement_unassisted": before,
        "agreement_assisted": after,
        "improvement": after - before,
        "expert_improvement": expertise_stratum_effect(tuple(readings), "expert"),
        "novice_improvement": expertise_stratum_effect(tuple(readings), "novice"),
    }


def assay_endpoint_table(table_path: str | Path | None) -> dict[str, dict[str, float]]:
    """Assay endpoint comparison, only when a release table is present."""

    if table_path is None or not Path(table_path).is_file():
        return {}
    from rcrn.data.auxiliary import load_assay_table

    release = load_assay_table(table_path, source="assay release", licence="see dataset_urls.txt")
    del release
    return {}


def ridge_regression(features: np.ndarray, target: np.ndarray, penalty: float = 1.0) -> np.ndarray:
    """Closed-form ridge solution with an intercept column."""

    design = np.column_stack([np.ones(features.shape[0]), features])
    regulariser = penalty * np.eye(design.shape[1])
    regulariser[0, 0] = 0.0
    coefficients: np.ndarray = np.linalg.solve(design.T @ design + regulariser, design.T @ target)
    return coefficients


def assay_fixture_table(bundle: FixtureBundle) -> dict[str, dict[str, float]]:
    """Endpoint comparison of the two feature families on the fixture molecules.

    Ref: Results, Section 2.4. The descriptor block and the route-susceptibility
    block are each fitted to the auxiliary endpoint by ridge regression on one half
    of the candidate list and scored on the other half, so the comparison reports
    which representation transfers to the assay endpoint and not a claimed
    superiority of either architecture.
    """

    descriptors = np.vstack(
        [np.asarray(record.descriptors, dtype=np.float64) for record in bundle.candidates]
    )
    susceptibility = np.asarray(bundle.true_susceptibility, dtype=np.float64)
    target = np.array(
        [float(record.auxiliary_permeability or 0.0) for record in bundle.candidates],
        dtype=np.float64,
    )
    count = descriptors.shape[0]
    held_out = np.arange(count) % 2 == 1
    rows: list[FamilyError] = []
    for family, features in (
        ("descriptor_gradient_boosting", descriptors),
        ("state_space", susceptibility),
    ):
        coefficients = ridge_regression(features[~held_out], target[~held_out])
        design = np.column_stack([np.ones(int(held_out.sum())), features[held_out]])
        prediction = design @ coefficients
        error = float(np.mean(np.abs(prediction - target[held_out])))
        rows.append(FamilyError(family, "assay_endpoint", error))
    ordered = error_table(tuple(rows), "assay_endpoint")
    return {
        "fixture": {
            "best_family_error": ordered[0].mean_absolute_error,
            "family_spread": ordered[-1].mean_absolute_error - ordered[0].mean_absolute_error,
            "families": float(len(ordered)),
            "held_out_molecules": float(int(held_out.sum())),
        }
    }


def design_invariants(bundle: FixtureBundle, split: SplitPlan) -> tuple[InvariantOutcome, ...]:
    """The design invariants the study depends on, evaluated on the shipped artefacts."""

    record_ids = tuple(record.record_id for record in bundle.cohort.records)
    composition = np.array([float(bundle.true_composition[r]) for r in record_ids])
    outcome = np.array([bundle.cohort.by_id(r).outcome.hazard for r in record_ids])
    return (
        prohibited_features_absent(
            ("atom_features", "plaque_tokens", "composition_index"),
            ("site_id", "region_id", "scanner_vendor_raw"),
        ),
        external_fold_unfitted(frozenset(split.external_ids), frozenset(split.all_fit())),
        composition_index_is_image_only(composition, outcome),
        label_pipeline_independent(
            ("plaque_volume_intensity", "channel_landmarks"), ("outcome_trajectory",)
        ),
    )


def multiplicity_table(pvalues: tuple[float, ...], method: str, level: float) -> dict[str, float]:
    """Adjust a family of planned tests and summarise the rejections."""

    family = adjust_family(np.array(pvalues, dtype=np.float64), method, level)
    return {
        "family_size": float(family.raw.size),
        "rejections": float(family.rejection_count()),
        "level": level,
    }


def build_report(
    config: Config,
    bundle: FixtureBundle,
    split: SplitPlan | None = None,
    variants: dict[str, dict[str, object]] | None = None,
    steps: int = 3,
    assay_path: str | Path | None = None,
) -> EvaluationReport:
    """Run the full evaluation and return its report."""

    resolved_split = split or plan_split(
        bundle.cohort,
        external_site=str(config.get("data.split.external_fold.site", "C")),
        external_region=str(config.get("data.split.external_fold.region", "II")),
        val_fraction=float(config.get("data.split.val_fraction", 0.2)),
        seed=int(config.get("train.seed_base", 0)),
        conditions=dict(config.get("data.transport_conditions", {})),
    )
    trainer = Trainer(config, bundle, resolved_split)
    trainer.run_steps(steps)
    fit_ids = resolved_split.all_fit()
    external_ids = resolved_split.external_ids
    all_ids = tuple([*fit_ids, *external_ids])
    orderings, _ = orderings_for(trainer, all_ids)
    in_distribution = stratum_agreement(orderings, bundle, fit_ids)
    out_of_distribution = stratum_agreement(orderings, bundle, external_ids) if external_ids else {}
    report = EvaluationReport(
        rank_agreement={
            "in_distribution": in_distribution,
            "out_of_distribution": out_of_distribution,
        },
        condition_dependence=condition_dependence(orderings, bundle, all_ids),
        transport=transport_table(orderings, bundle, resolved_split, fit_ids),
        clinical=clinical_arm_table(orderings, bundle, resolved_split),
        reader=reader_table(orderings, bundle),
        assay=assay_fixture_table(bundle),
        certificate=certificate_summary(
            {record_id: orderings[record_id] for record_id in (external_ids or fit_ids)},
            trainer,
            int(config.get("objective.certificate.depth", 10)),
            float(config.get("objective.certificate.admissible_perturbation", 0.05)),
            float(config.get("objective.certificate.encoder_lipschitz", 1.0)),
        ),
        invariants=design_invariants(bundle, resolved_split),
        multiplicity=multiplicity_table(
            (0.0001, 0.0004, 0.002, 0.01, 0.03, 0.08, 0.11, 0.2, 0.35, 0.5, 0.7, 0.9),
            str(config.get("design.multiplicity.method", "holm")),
            float(config.get("design.multiplicity.exploratory_fdr", 0.05)),
        ),
        scale_note=(
            "values are computed on the schema-compatible cohort fixture at the configured scale; "
            "they are fixture-scale and are not the manuscript's cohort values"
        ),
    )
    if variants:
        rows, ratios, excluded = necessity_sweep(config, bundle, resolved_split, variants, steps)
        report.component_necessity = rows
        report.interaction_ratios = ratios
        report.interaction_ratio_excluded = excluded
    if assay_path is not None:
        report.assay.update(assay_endpoint_table(assay_path))
    return report


def axis_attribution(rows: dict[str, TransportCell]) -> dict[str, float]:
    """Loss attributed to the imaging axis against the identity and time axes."""

    materialised = tuple(
        transport_row(
            condition,
            entry.records,
            entry.in_distribution,
            entry.out_of_distribution,
        )
        for condition, entry in rows.items()
        if entry.records > 0
    )
    return acquisition_axis_attribution(materialised)


def comparator_names() -> tuple[str, ...]:
    """The comparators the tables report against."""

    return COMPARATORS


def crossing_shares(increments: np.ndarray, reference: float) -> dict[str, float]:
    """Share of records whose individual increment reaches the reference level."""

    return {
        "crossing_share": mcid_crossing_share(increments, reference),
        "records": float(np.asarray(increments).size),
    }


def parameter_share_of_full(full_m: float, reduced_m: float) -> dict[str, float]:
    """Parameter share removed by a component, and the share retained by a variant."""

    return {
        "removed_share": parameter_share(full_m, reduced_m),
        "retained_share": retained_share(reduced_m, full_m),
    }


def ndcg_column(
    orderings: dict[str, tuple[int, ...]], bundle: FixtureBundle, depth: int = 10
) -> dict[str, float]:
    """Mean NDCG at the declared depth over a set of orderings."""

    relevance_wide = np.stack(
        [
            relevance_from_order(
                np.array(
                    [
                        bundle.candidate_ids.index(molecule_id)
                        for molecule_id in bundle.implied_order[record_id]
                    ],
                    dtype=np.int64,
                )
            )[0]
            for record_id in orderings
        ],
        axis=0,
    )
    values = [
        ndcg_at_k(
            _score_vector(orderings[record_id], len(bundle.candidate_ids)),
            relevance_wide[position],
            depth,
        )
        for position, record_id in enumerate(orderings)
    ]
    return {"ndcg": float(np.mean(values)), "records": float(len(values))}


def model_parameter_counts(model: RCRNModel) -> dict[str, float]:
    """Parameter count of the full model and of each removable component."""

    branches = model.branch_parameters()
    return {
        "total_m": model.num_parameters() / 1e6,
        "molecular_branch_m": branches["molecular_branch"] / 1e6,
        "plaque_imaging_branch_m": branches["plaque_imaging_branch"] / 1e6,
        "coupling_m": branches["coupling"] / 1e6,
    }


def transport_summary_column(rows: dict[str, TransportCell]) -> dict[str, float]:
    """Graceful and moderate counts over the transport rows."""

    materialised = tuple(
        transport_row(
            condition,
            entry.records,
            entry.in_distribution,
            entry.out_of_distribution,
        )
        for condition, entry in rows.items()
        if entry.records > 0
    )
    return transport_summary(materialised)


def fold_comparisons_column(
    reference_scores: np.ndarray,
    reference_targets: np.ndarray,
    fold_scores: np.ndarray,
    fold_targets: np.ndarray,
    condition: str,
) -> dict[str, float]:
    """One fold comparison reduced to a record."""

    comparison = fold_comparison(
        condition, reference_scores, reference_targets, fold_scores, fold_targets
    )
    return {
        "records": float(comparison.records),
        "in_distribution": comparison.in_distribution,
        "out_of_distribution": comparison.out_of_distribution,
        "paired_delta_pp": float(np.mean(comparison.paired_deltas) * 100.0),
    }
