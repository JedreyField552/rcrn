"""Evaluate the fitted model against every reported table.

Ref: Tables 1-4; Figures 1-3; Results, Sections 2.1-2.8.
"""

from __future__ import annotations

import argparse
import sys

from rcrn.cli.common import base_parser, build_inputs, output_directory, start_cli
from rcrn.evaluation.pipeline import build_report, model_parameter_counts
from rcrn.evaluation.protocol import default_plan
from rcrn.evaluation.reporting import jsonable
from rcrn.training.trainer import Trainer
from rcrn.utils.io import atomic_write_json, atomic_write_text


def _variant_spec() -> dict[str, dict[str, object]]:
    return {
        "full": {"tier": "reference"},
        "no_reaction_norm_coupling": {
            "model.components.reaction_norm_coupling": False,
            "model.coupling.kind": "plain_concat",
            "tier": "tier_1_removal",
            "linked_equation": "eq_reaction_norm",
        },
        "no_gate": {
            "model.components.interaction_variance_gate": False,
            "model.coupling.gate": "always_condition",
            "tier": "tier_1_removal",
            "linked_equation": "eq_gate",
        },
        "no_composition_head": {
            "model.components.plaque_composition_head": False,
            "tier": "tier_1_removal",
            "linked_equation": "eq_pair_model",
        },
        "scalar_susceptibility": {
            "model.components.regime_route_head": False,
            "model.heads.route_count": 1,
            "tier": "tier_1_removal",
            "linked_equation": "eq_objective",
        },
        "uniform_rank_weight": {
            "model.components.agreement_weighted_rank": False,
            "model.ranking.weighting": "linear",
            "tier": "tier_1_removal",
            "linked_equation": "eq_objective",
        },
        "joint_coupling_and_gate": {
            "model.components.reaction_norm_coupling": False,
            "model.coupling.kind": "plain_concat",
            "model.components.interaction_variance_gate": False,
            "tier": "tier_2_joint",
            "linked_equation": "eq_reaction_norm",
        },
        "joint_coupling_and_composition": {
            "model.components.reaction_norm_coupling": False,
            "model.coupling.kind": "plain_concat",
            "model.components.plaque_composition_head": False,
            "tier": "tier_2_joint",
            "linked_equation": "eq_pair_model",
        },
        "joint_route_and_composition": {
            "model.components.regime_route_head": False,
            "model.heads.route_count": 1,
            "model.components.plaque_composition_head": False,
            "tier": "tier_2_joint",
            "linked_equation": "eq_objective",
        },
        "joint_route_and_rank": {
            "model.components.regime_route_head": False,
            "model.heads.route_count": 1,
            "model.components.agreement_weighted_rank": False,
            "tier": "tier_2_joint",
            "linked_equation": "eq_objective",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser: argparse.ArgumentParser = base_parser("Evaluate the reaction-norm model")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--variants", action="store_true", help="also run the necessity sweep")
    parser.add_argument("--assay-table", default=None)
    args = parser.parse_args(argv)

    config, root = start_cli(args.config, args.set, args.log_level)
    bundle, split, source = build_inputs(config)
    variants = _variant_spec() if args.variants else None
    report = build_report(
        config,
        bundle,
        split,
        variants=variants,
        steps=args.steps,
        assay_path=args.assay_table,
    )
    trainer = Trainer(config, bundle, split)
    plan = default_plan(
        reference_stratum="rads_1_2",
        primary_stratum="rads_3_4_iph",
        rank_floor=float(config.get("design.rank_floor", 0.05)),
        reference_level=0.07,
        family_size=int(config.get("design.multiplicity.family_size", 12)),
        bootstrap_resamples=int(config.get("design.bootstrap_resamples", 2000)),
        seed_count=int(config.get("design.seed_count", 24)),
        exploratory_fdr=float(config.get("design.multiplicity.exploratory_fdr", 0.05)),
    )
    payload = {
        "source": source,
        "config_fingerprint": config.fingerprint(),
        "analysis_plan": plan.as_record(),
        "parameters": model_parameter_counts(trainer.model),
        "report": report.as_record(),
    }
    run_directory = output_directory(root, args.out, str(config.get("name", "evaluation")))
    atomic_write_json(run_directory / "evaluation_report.json", jsonable(payload))
    atomic_write_text(
        run_directory / "evaluation_summary.txt",
        "\n".join(
            [
                f"source: {source}",
                f"records: {len(bundle.cohort)}  candidates: {len(bundle.candidate_ids)}",
                f"partitions: fit {len(split.fit_ids)}  validation {len(split.validation_ids)}  "
                f"external {len(split.external_ids)}",
                report.scale_note,
                "manuscript cohort values are reported as reference values, not as run results",
                "",
            ]
        ),
    )
    sys.stdout.write(f"wrote {run_directory / 'evaluation_report.json'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
