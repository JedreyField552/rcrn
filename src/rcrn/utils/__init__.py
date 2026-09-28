"""Shared utilities: configuration, logging, atomic IO and structural types."""

from __future__ import annotations

from rcrn.utils.config import Config, deep_merge, load_experiment, release_root
from rcrn.utils.io import (
    atomic_write_json,
    atomic_write_text,
    manifest_mismatches,
    read_json,
    sha256_file,
    write_integrity_manifest,
)
from rcrn.utils.logging import configure_logging, get_logger

__all__ = [
    "Config",
    "atomic_write_json",
    "atomic_write_text",
    "configure_logging",
    "deep_merge",
    "get_logger",
    "load_experiment",
    "manifest_mismatches",
    "read_json",
    "release_root",
    "sha256_file",
    "write_integrity_manifest",
]
