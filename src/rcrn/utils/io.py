"""Atomic filesystem helpers and the content-digest manifest writer."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

_SKIP_DIRS = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".git",
        ".venv",
    }
)
_SKIP_TOP_DIRS = frozenset({"data", "runs", "models", "scratch", "build", "dist"})


def sha256_file(path: str | Path) -> str:
    """Digest a file in bounded chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(
    path: str | Path,
    write: Callable[[IO[Any]], None],
    mode: str,
    encoding: str | None = None,
) -> Path:
    """Write through a mode-0644 temporary file and rename it into place."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, mode, encoding=encoding) as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return destination


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write text atomically."""

    return _atomic_write(path, lambda handle: handle.write(text), "w", "utf-8")


def atomic_write_binary(path: str | Path, write: Callable[[IO[Any]], None]) -> Path:
    """Write a binary payload atomically, for serialisers that take a file handle."""

    return _atomic_write(path, write, "wb")


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Serialise a payload with a stable key order and a trailing newline."""

    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return atomic_write_text(path, text)


def read_json(path: str | Path) -> Any:
    """Load a JSON document."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_manifest_paths(root: str | Path) -> list[Path]:
    """List every file below a release root that belongs in the manifest."""

    base = Path(root)
    if not base.is_dir():
        raise NotADirectoryError(base)
    collected: list[Path] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(base)
        if any(part in _SKIP_DIRS for part in relative.parts):
            continue
        if relative.parts and relative.parts[0] in _SKIP_TOP_DIRS and len(relative.parts) > 1:
            continue
        collected.append(relative)
    return collected


def write_integrity_manifest(root: str | Path, destination: str | Path | None = None) -> Path:
    """Digest the shipped tree, excluding only the manifest itself."""

    base = Path(root).resolve()
    target = Path(destination) if destination is not None else base / "integrity_manifest.json"
    target_resolved = target.resolve()
    files: list[dict[str, Any]] = []
    for relative in iter_manifest_paths(base):
        absolute = base / relative
        if absolute.resolve() == target_resolved:
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(absolute),
                "bytes": absolute.stat().st_size,
            }
        )
    payload = {
        "algorithm": "SHA-256",
        "root_scope": "release tree, excluding the manifest file and build or cache directories",
        "file_count": len(files),
        "files": files,
    }
    return atomic_write_json(target, payload)


def manifest_mismatches(
    root: str | Path, destination: str | Path | None = None
) -> dict[str, list[str]]:
    """Report missing, extra and stale entries against the live tree."""

    base = Path(root).resolve()
    target = Path(destination) if destination is not None else base / "integrity_manifest.json"
    payload = read_json(target)
    listed = {entry["path"]: entry["sha256"] for entry in payload["files"]}
    live: dict[str, str] = {}
    for relative in iter_manifest_paths(base):
        absolute = base / relative
        if absolute.resolve() == target.resolve():
            continue
        live[relative.as_posix()] = sha256_file(absolute)
    return {
        "missing": sorted(set(listed) - set(live)),
        "unlisted": sorted(set(live) - set(listed)),
        "stale": sorted(path for path in set(listed) & set(live) if listed[path] != live[path]),
    }
