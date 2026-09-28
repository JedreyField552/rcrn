"""Train the regime-coupled reaction-norm model.

Ref: Algorithm 2; Methods, "Training and inference".
"""

from __future__ import annotations

import json
import sys

from rcrn.cli.common import base_parser, build_inputs, output_directory, start_cli
from rcrn.training.trainer import Trainer
from rcrn.utils.io import atomic_write_json, atomic_write_text


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("Train the regime-coupled reaction-norm model")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args(argv)

    config, root = start_cli(args.config, args.set, args.log_level)
    bundle, split, source = build_inputs(config)
    trainer = Trainer(config, bundle, split)
    result = trainer.run_steps(args.steps) if args.steps is not None else trainer.run(args.seed)
    run_directory = output_directory(root, args.out, str(config.get("name", "run")))
    checkpoint_path = run_directory / "checkpoint.pt"
    trainer.save(checkpoint_path, best_metric=result.final_loss(), epoch=result.epochs)
    payload = {
        "name": str(config.get("name", "run")),
        "source": source,
        "epochs": result.epochs,
        "steps": result.steps,
        "pretrain_steps": result.pretrain_steps,
        "auxiliary_targets_used": result.auxiliary_targets_used,
        "first_loss": result.first_loss(),
        "final_loss": result.final_loss(),
        "config_fingerprint": config.fingerprint(),
        "partitions": {
            "fit": len(split.fit_ids),
            "validation": len(split.validation_ids),
            "external": len(split.external_ids),
        },
        "parameters": trainer.model.num_parameters(),
    }
    atomic_write_json(run_directory / "training_report.json", payload)
    atomic_write_text(
        run_directory / "training_summary.txt",
        "\n".join(
            [
                f"name: {payload['name']}",
                f"source: {source}",
                f"epochs: {result.epochs}  steps: {result.steps}",
                f"first loss: {result.first_loss():.6f}",
                f"final loss: {result.final_loss():.6f}",
                f"parameters: {payload['parameters']}",
                f"checkpoint: {checkpoint_path.name}",
                "manuscript values are not reported by this command",
                "",
            ]
        ),
    )
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
