"""Report the transport degradation across the out-of-distribution folds.

Ref: Table 3; Figure 3; Results, Section 2.5.
"""

from __future__ import annotations

import argparse
import json
import sys

from rcrn.cli.common import base_parser, build_inputs, output_directory, start_cli
from rcrn.evaluation.pipeline import (
    axis_attribution,
    orderings_for,
    transport_summary_column,
    transport_table,
)
from rcrn.training.trainer import Trainer
from rcrn.utils.io import atomic_write_json, atomic_write_text


def main(argv: list[str] | None = None) -> int:
    parser: argparse.ArgumentParser = base_parser("Report transport degradation")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args(argv)

    config, root = start_cli(args.config, args.set, args.log_level)
    bundle, split, source = build_inputs(config)
    trainer = Trainer(config, bundle, split)
    trainer.run_steps(args.steps)
    orderings, _ = orderings_for(trainer, tuple([*split.all_fit(), *split.external_ids]))
    rows = transport_table(orderings, bundle, split, split.all_fit())
    payload = {
        "source": source,
        "folds": {name: cell.as_record() for name, cell in rows.items()},
        "summary": transport_summary_column(rows),
        "axis_attribution": axis_attribution(rows),
        "paired_vs_marginal_note": (
            "the reported change is the mean paired difference within records against each "
            "record's site-matched reference; the marginal column difference is carried alongside"
        ),
    }
    run_directory = output_directory(root, args.out, "transport")
    atomic_write_json(run_directory / "transport_report.json", payload)
    atomic_write_text(
        run_directory / "transport_summary.txt",
        "\n".join(
            [f"source: {source}"]
            + [
                f"{name}: records {entry.records}  paired {entry.paired_delta_pp:+.2f} pp  "
                f"marginal {entry.marginal_delta_pp:+.2f} pp  {entry.degradation}"
                for name, entry in rows.items()
            ]
            + [""]
        ),
    )
    sys.stdout.write(json.dumps(payload["summary"], sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
