"""Command-line entry points, exercised as a user would run them."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from _paths import ROOT, experiment

SMOKE = "configs/experiment/_smoke.yaml"
ENVIRONMENT = {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def run(module: str, *arguments: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed interpreter and module name
        [sys.executable, "-m", module, *arguments],
        cwd=str(cwd or ROOT),
        env=ENVIRONMENT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_train_writes_a_checkpoint_and_a_report(tmp_path: Path) -> None:
    completed = run(
        "rcrn.cli.train",
        "--config",
        SMOKE,
        "--steps",
        "2",
        "--out",
        str(tmp_path),
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads((tmp_path / "training_report.json").read_text(encoding="utf-8"))
    assert payload["steps"] == 2
    assert payload["parameters"] > 0
    assert (tmp_path / "checkpoint.pt").is_file()
    summary = (tmp_path / "training_summary.txt").read_text(encoding="utf-8")
    assert "manuscript values are not reported" in summary


def test_evaluate_writes_a_report(tmp_path: Path) -> None:
    completed = run(
        "rcrn.cli.evaluate",
        "--config",
        SMOKE,
        "--steps",
        "2",
        "--out",
        str(tmp_path),
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads((tmp_path / "evaluation_report.json").read_text(encoding="utf-8"))
    assert payload["analysis_plan"]["confirmatory_family_size"] == 12
    assert "in_distribution" in payload["report"]["rank_agreement"]
    assert payload["report"]["scale_note"]


def test_evaluate_can_run_the_necessity_sweep(tmp_path: Path) -> None:
    completed = run(
        "rcrn.cli.evaluate",
        "--config",
        SMOKE,
        "--steps",
        "1",
        "--variants",
        "--out",
        str(tmp_path),
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads((tmp_path / "evaluation_report.json").read_text(encoding="utf-8"))
    assert payload["report"]["component_necessity"]
    formed = payload["report"]["interaction_ratios"]
    excluded = payload["report"]["interaction_ratio_excluded"]
    assert len(formed) + len(excluded) == 4
    assert all(reason for reason in excluded.values())


def test_certify_reports_a_deferral_rate(tmp_path: Path) -> None:
    completed = run(
        "rcrn.cli.certify",
        "--config",
        SMOKE,
        "--steps",
        "2",
        "--out",
        str(tmp_path),
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads((tmp_path / "certification_report.json").read_text(encoding="utf-8"))
    certificate = payload["certificate"]
    assert 0.0 <= certificate["certified_share"] <= 1.0
    assert 0.0 <= certificate["deferral_rate"] <= 1.0
    assert payload["deferral_policy"] == "plaque_blind_order"


def test_transport_lists_every_fold(tmp_path: Path) -> None:
    completed = run(
        "rcrn.cli.transport",
        "--config",
        SMOKE,
        "--steps",
        "2",
        "--out",
        str(tmp_path),
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads((tmp_path / "transport_report.json").read_text(encoding="utf-8"))
    assert len(payload["folds"]) == 8
    assert payload["summary"]["conditions"] >= 0.0
    assert "paired_vs_marginal_note" in payload


def test_verify_honours_the_root_argument(tmp_path: Path) -> None:
    """The driver writes to the root it is given, never to the shipped tree."""

    scratch = tmp_path / "release"
    for layer in ("model", "data", "train", "experiment"):
        (scratch / "configs" / layer).mkdir(parents=True)
        for source in (ROOT / "configs" / layer).glob("*.yaml"):
            (scratch / "configs" / layer / source.name).write_text(
                source.read_text(encoding="utf-8"), encoding="utf-8"
            )
    dispatched = run(
        "rcrn.cli.verify",
        "--config",
        str(scratch / "configs" / "experiment" / "main.yaml"),
        "--root",
        str(scratch),
        "--skip-tools",
    )
    assert dispatched.returncode in {0, 1}, dispatched.stderr
    for name in (
        "claim_to_code.json",
        "verification_report.json",
        "verification_summary.txt",
        "integrity_manifest.json",
    ):
        assert (scratch / name).is_file(), name
    report = json.loads((scratch / "verification_report.json").read_text(encoding="utf-8"))
    assert report["checks"]
    assert report["scope"]


def test_cli_reports_a_missing_configuration() -> None:
    completed = run(
        "rcrn.cli.train", "--config", "configs/experiment/does_not_exist.yaml", "--steps", "1"
    )
    assert completed.returncode != 0


def test_override_reaches_the_configuration(tmp_path: Path) -> None:
    completed = run(
        "rcrn.cli.train",
        "--config",
        SMOKE,
        "--steps",
        "1",
        "--set",
        "train.batch_size=8",
        "--out",
        str(tmp_path),
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads((tmp_path / "training_report.json").read_text(encoding="utf-8"))
    assert payload["steps"] == 1


def test_main_configuration_is_reachable() -> None:
    assert experiment("main.yaml").is_file()
    completed = run(
        "rcrn.cli.train",
        "--config",
        "configs/experiment/main.yaml",
        "--steps",
        "1",
        "--out",
        "/tmp/rcrn-main-check",
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "module",
    [
        "rcrn.cli.train",
        "rcrn.cli.evaluate",
        "rcrn.cli.certify",
        "rcrn.cli.transport",
        "rcrn.cli.verify",
    ],
)
def test_help_is_available(module: str) -> None:
    completed = run(module, "--help")
    assert completed.returncode == 0
    assert "usage" in completed.stdout.lower()
