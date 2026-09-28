"""Configuration layering and atomic IO."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _paths import ROOT, experiment

from rcrn.utils.config import Config, deep_merge, load_experiment, release_root
from rcrn.utils.io import (
    atomic_write_json,
    atomic_write_text,
    manifest_mismatches,
    read_json,
    sha256_file,
    write_integrity_manifest,
)


def test_experiment_resolves_its_inherited_layers() -> None:
    config = load_experiment(experiment("main.yaml"))
    assert config.get("model.heads.route_count") == 3
    assert config.get("train.batch_size") == 8
    assert config.get("objective.loss_weights.rank") == 1.0
    assert config.get("design.candidate_list_size") == 32


def test_experiment_overrides_are_applied() -> None:
    config = load_experiment(
        experiment("_smoke.yaml"), ["train.batch_size=2", "model.heads.route_count=1"]
    )
    assert config.get("train.batch_size") == 2
    assert config.get("model.heads.route_count") == 1


def test_fingerprint_tracks_the_effective_payload() -> None:
    base = load_experiment(experiment("_smoke.yaml"))
    changed = load_experiment(experiment("_smoke.yaml"), ["train.lr=0.5"])
    assert base.fingerprint() != changed.fingerprint()
    assert base.fingerprint() == load_experiment(experiment("_smoke.yaml")).fingerprint()


def test_require_reports_a_missing_key() -> None:
    config = load_experiment(experiment("_smoke.yaml"))
    with pytest.raises(KeyError):
        config.require("no.such.key")


def test_section_returns_a_nested_view() -> None:
    config = load_experiment(experiment("main.yaml"))
    section = config.section("model.heads")
    assert section.require("route_count") == 3


def test_deep_merge_replaces_non_mappings() -> None:
    merged = deep_merge({"a": {"b": 1, "c": 2}, "d": [1]}, {"a": {"b": 9}, "d": [2, 3]})
    assert merged == {"a": {"b": 9, "c": 2}, "d": [2, 3]}


def test_release_root_requires_a_configs_experiment_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        release_root(tmp_path / "main.yaml")


def test_release_root_resolves_from_the_experiment_file() -> None:
    assert release_root(experiment("main.yaml")) == ROOT


def test_cyclic_inheritance_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "configs" / "experiment").mkdir(parents=True)
    (tmp_path / "configs" / "experiment" / "a.yaml").write_text(
        "extends: [configs/experiment/b.yaml]\n", encoding="utf-8"
    )
    (tmp_path / "configs" / "experiment" / "b.yaml").write_text(
        "extends: [configs/experiment/a.yaml]\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_experiment(tmp_path / "configs" / "experiment" / "a.yaml")


def test_override_without_an_equals_is_rejected() -> None:
    config = Config({"a": 1}, ())
    with pytest.raises(ValueError):
        config.with_overrides(["justakey"])


def test_atomic_text_and_json_round_trip(tmp_path: Path) -> None:
    text_path = atomic_write_text(tmp_path / "note.txt", "hello\n")
    assert text_path.read_text(encoding="utf-8") == "hello\n"
    json_path = atomic_write_json(tmp_path / "payload.json", {"b": 1, "a": 2})
    assert read_json(json_path) == {"a": 2, "b": 1}
    assert oct(json_path.stat().st_mode & 0o777) == "0o644"


def test_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"payload")
    second.write_bytes(b"payload")
    assert sha256_file(first) == sha256_file(second)
    second.write_bytes(b"payload2")
    assert sha256_file(first) != sha256_file(second)


def test_manifest_excludes_itself_and_skips_caches(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "module.pyc").write_bytes(b"cache")
    (tmp_path / "configs" / "data").mkdir(parents=True)
    (tmp_path / "configs" / "data" / "cohorts.yaml").write_text("a: 1\n", encoding="utf-8")
    manifest = write_integrity_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    paths = {entry["path"] for entry in payload["files"]}
    assert "integrity_manifest.json" not in paths
    assert "src/module.py" in paths
    assert "configs/data/cohorts.yaml" in paths
    assert not any("__pycache__" in path for path in paths)
    assert payload["algorithm"] == "SHA-256"


def test_manifest_mismatch_detection(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("one\n", encoding="utf-8")
    write_integrity_manifest(tmp_path)
    assert all(not value for value in manifest_mismatches(tmp_path).values())
    (tmp_path / "file.txt").write_text("two\n", encoding="utf-8")
    assert manifest_mismatches(tmp_path)["stale"] == ["file.txt"]
    (tmp_path / "extra.txt").write_text("three\n", encoding="utf-8")
    assert "extra.txt" in manifest_mismatches(tmp_path)["unlisted"]
    (tmp_path / "file.txt").unlink()
    mismatches = manifest_mismatches(tmp_path)
    assert "file.txt" in mismatches["missing"]


def test_manifest_directory_argument_must_exist() -> None:
    with pytest.raises(NotADirectoryError):
        from rcrn.utils.io import iter_manifest_paths

        list(iter_manifest_paths(Path("/nowhere/at/all")))
