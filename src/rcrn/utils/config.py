"""Layered YAML configuration with recursive inheritance and CLI overrides."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

_MISSING = object()


class Config:
    """Immutable view over a deep-merged YAML document."""

    def __init__(self, payload: dict[str, Any], sources: Sequence[Path]) -> None:
        self._payload = payload
        self._sources = tuple(sources)

    @property
    def sources(self) -> tuple[Path, ...]:
        return self._sources

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._payload)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._payload
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def require(self, dotted: str) -> Any:
        value = self.get(dotted, _MISSING)
        if value is _MISSING:
            raise KeyError(dotted)
        return value

    def section(self, dotted: str) -> Config:
        value = self.require(dotted)
        if not isinstance(value, dict):
            raise TypeError(dotted)
        return Config(value, self._sources)

    def with_overrides(self, pairs: Iterable[str]) -> Config:
        payload = self.as_dict()
        for pair in pairs:
            if "=" not in pair:
                raise ValueError(pair)
            dotted, _, raw = pair.partition("=")
            _assign(payload, dotted.strip(), _parse_scalar(raw.strip()))
        return Config(payload, self._sources)

    def fingerprint(self) -> str:
        """Stable short digest of the effective configuration."""

        import hashlib

        blob = json.dumps(self._payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def _parse_scalar(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _assign(payload: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = payload
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; every other type replaces wholesale."""

    merged = copy.deepcopy(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def release_root(experiment_path: str | Path) -> Path:
    """Resolve the release root from a configs/experiment file location."""

    path = Path(experiment_path).resolve()
    if path.parent.name != "experiment" or path.parent.parent.name != "configs":
        raise ValueError(f"expected a configs/experiment/*.yaml path, received {path.name}")
    return path.parents[2]


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(path)
    return loaded


def _resolve_chain(path: Path, root: Path, seen: list[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"cyclic extends chain at {resolved.name}")
    payload = _read_yaml(resolved)
    parents = payload.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = (root / parent).resolve()
        merged = deep_merge(merged, _resolve_chain(parent_path, root, [*seen, resolved]))
    return deep_merge(merged, payload)


def load_experiment(experiment_path: str | Path, overrides: Iterable[str] = ()) -> Config:
    """Load an experiment file together with its inherited configuration layers."""

    path = Path(experiment_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    root = release_root(path)
    payload = _resolve_chain(path, root, [])
    sources = tuple(
        (root / entry).resolve()
        for entry in ([*_collect_extends(path, root, [])] or [path.resolve()])
    )
    config = Config(payload, sources)
    override_list = list(overrides)
    return config.with_overrides(override_list) if override_list else config


def _collect_extends(path: Path, root: Path, seen: list[Path]) -> list[Path]:
    resolved = path.resolve()
    if resolved in seen:
        return []
    payload = _read_yaml(resolved)
    parents = payload.get("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    collected: list[Path] = []
    for parent in parents:
        parent_path = (root / parent).resolve()
        collected.extend(_collect_extends(parent_path, root, [*seen, resolved]))
    collected.append(resolved)
    return collected
