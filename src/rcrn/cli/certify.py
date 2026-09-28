"""Certified top-k prioritisation.

Ref: Eq. (5); Algorithm 1; Methods, "Certified top-k prioritisation".
"""

from __future__ import annotations

import argparse
import sys

from rcrn.cli.common import base_parser, build_inputs, output_directory, start_cli
from rcrn.evaluation.pipeline import certificate_summary, orderings_for
from rcrn.training.trainer import Trainer
from rcrn.utils.io import atomic_write_json, atomic_write_text


def main(argv: list[str] | None = None) -> int:
    parser: argparse.ArgumentParser = base_parser("Certify the returned top-k set")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--depth", type=int, default=None)
    args = parser.parse_args(argv)

    config, root = start_cli(args.config, args.set, args.log_level)
    bundle, split, source = build_inputs(config)
    trainer = Trainer(config, bundle, split)
    trainer.run_steps(args.steps)
    record_ids = split.external_ids or split.all_fit()
    orderings, statistics = orderings_for(trainer, record_ids)
    depth = args.depth if args.depth is not None else int(config.get("design.top_k_depth", 10))
    admissible = float(config.get("objective.certificate.admissible_perturbation", 0.05))
    lipschitz = float(config.get("objective.certificate.encoder_lipschitz", 1.0))
    summary = certificate_summary(orderings, trainer, depth, admissible, lipschitz)
    payload = {
        "source": source,
        "records": len(record_ids),
        "gate_statistics": {record_id: float(statistics[record_id]) for record_id in record_ids},
        "certificate": summary,
        "deferral_policy": str(
            config.get("objective.certificate.deferral_policy", "plaque_blind_order")
        ),
    }
    run_directory = output_directory(root, args.out, "certification")
    atomic_write_json(run_directory / "certification_report.json", payload)
    atomic_write_text(
        run_directory / "certification_summary.txt",
        "\n".join(
            [
                f"records: {len(record_ids)}",
                f"depth: {int(summary['depth'])}",
                f"certified share: {summary['certified_share']:.4f}",
                f"deferral rate: {summary['deferral_rate']:.4f}",
                f"admissible perturbation: {admissible}",
                "",
            ]
        ),
    )
    sys.stdout.write(f"certified share {summary['certified_share']:.4f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
