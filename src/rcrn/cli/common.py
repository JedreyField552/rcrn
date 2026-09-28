"""Shared helpers for the command-line entry points."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from rcrn.data.candidates import load_candidate_table
from rcrn.data.cohort import (
    load_authorised_cohort,
    load_implied_orders,
    load_volume_directory,
)
from rcrn.data.fixture import FixtureBundle, build_cohort_fixture, bundle_from_authorised
from rcrn.data.splits import SplitPlan, plan_split
from rcrn.utils.config import Config, load_experiment, release_root
from rcrn.utils.logging import configure_logging


@dataclass(slots=True)
class PreparedInputs:
    """Config, cohort layer and split resolved from one experiment file."""

    config: Config
    bundle: FixtureBundle
    split: SplitPlan
    root: Path
    source: str


def base_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser shared by the entry points."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/experiment/main.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def prepare(config_path: str, overrides: list[str]) -> tuple[Config, Path]:
    """Load a configuration and resolve the release root."""

    config = load_experiment(config_path, overrides)
    return config, release_root(config_path)


def build_inputs(config: Config) -> tuple[FixtureBundle, SplitPlan, str]:
    """Resolve the cohort layer and the split the configuration declares."""

    source = str(config.get("data.source", "fixture"))
    bundle = _authorised_bundle(config) if source == "cohort" else build_cohort_fixture(config)
    split = plan_split(
        bundle.cohort,
        external_site=str(config.get("data.split.external_fold.site", "C")),
        external_region=str(config.get("data.split.external_fold.region", "II")),
        val_fraction=float(config.get("data.split.val_fraction", 0.2)),
        seed=int(config.get("train.seed_base", 0)),
        conditions=dict(config.get("data.transport_conditions", {})),
    )
    return bundle, split, source


def _authorised_bundle(config: Config) -> FixtureBundle:
    """Assemble the bundle from the authorised derived tables."""

    table = config.get("data.cohort.table")
    candidate_table = config.get("data.cohort.candidates")
    volume_dir = config.get("data.cohort.volumes")
    order_table = config.get("data.cohort.implied_order")
    missing = [
        name
        for name, value in (
            ("data.cohort.table", table),
            ("data.cohort.candidates", candidate_table),
            ("data.cohort.volumes", volume_dir),
            ("data.cohort.implied_order", order_table),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"{missing[0]} is required when data.source is 'cohort'")
    cohort = load_authorised_cohort(str(table), str(volume_dir))
    candidates = load_candidate_table(str(candidate_table))
    volumes = load_volume_directory(str(volume_dir))
    orders = load_implied_orders(str(order_table))
    return bundle_from_authorised(cohort, candidates, volumes, orders)


def output_directory(root: Path, requested: str | None, name: str) -> Path:
    """Resolve the run directory, defaulting under the ignored runs tree."""

    target = Path(requested) if requested else root / "runs" / name
    target.mkdir(parents=True, exist_ok=True)
    return target


def start_cli(config_path: str, overrides: list[str], level: str) -> tuple[Config, Path]:
    """Load the configuration and install logging."""

    configure_logging(level=getattr(__import__("logging"), level.upper(), 20))
    return prepare(config_path, overrides)
