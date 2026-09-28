"""Shared paths and helpers for the test suite."""

from __future__ import annotations

from pathlib import Path

from rcrn.data.fixture import FixtureBundle, build_cohort_fixture
from rcrn.data.splits import SplitPlan, plan_split
from rcrn.utils.config import Config, load_experiment

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "configs" / "experiment"


def experiment(name: str) -> Path:
    return EXPERIMENT_DIR / name


def smoke_config(overrides: tuple[str, ...] = ()) -> Config:
    return load_experiment(experiment("_smoke.yaml"), overrides)


def smoke_inputs(overrides: tuple[str, ...] = ()) -> tuple[Config, FixtureBundle, SplitPlan]:
    config = smoke_config(overrides)
    bundle = build_cohort_fixture(config)
    split = plan_split(
        bundle.cohort,
        external_site=str(config.get("data.split.external_fold.site", "C")),
        external_region=str(config.get("data.split.external_fold.region", "II")),
        val_fraction=float(config.get("data.split.val_fraction", 0.25)),
        seed=int(config.get("train.seed_base", 0)),
        conditions=dict(config.get("data.transport_conditions", {})),
    )
    return config, bundle, split
