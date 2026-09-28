"""The shipped verification artefacts.

The writer helpers are exercised against a scratch directory so a test run can never
replace the artefacts this release ships.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from _paths import ROOT

from rcrn.cli.verify import (
    CLAIMS,
    DEVIATIONS,
    PASS,
    VerificationDriver,
    mask,
    summary_text,
    write_artefacts,
    write_claim_map,
)
from rcrn.utils.io import manifest_mismatches, write_integrity_manifest

SHIPPED = (
    "claim_to_code.json",
    "verification_report.json",
    "verification_summary.txt",
    "integrity_manifest.json",
)


def test_shipped_artefacts_exist() -> None:
    for name in SHIPPED:
        assert (ROOT / name).is_file(), name
    assert (ROOT / "dataset_urls.txt").is_file()
    assert (ROOT / "NOTICE").is_file()


def test_shipped_report_is_complete() -> None:
    report = json.loads((ROOT / "verification_report.json").read_text(encoding="utf-8"))
    assert report["overall_status"] in {PASS, "PARTIALLY_VERIFIED", "FAIL"}
    assert report["code_status"] in {PASS, "PARTIALLY_VERIFIED", "FAIL"}
    names = {check["name"] for check in report["checks"]}
    assert len(report["checks"]) >= 70
    assert "paper_claim_mapping" in names
    assert "single_batch_overfit" in names
    assert "checkpoint_round_trip" in names
    assert "manuscript_family_summary" in names
    assert sum(report["status_counts"].values()) == len(report["checks"])
    assert "manuscript_discrepancies" in report
    assert report["integrity_manifest_written"] is True


def test_shipped_report_has_no_volatile_fields() -> None:
    text = (ROOT / "verification_report.json").read_text(encoding="utf-8")
    assert "seconds" not in text
    assert "timestamp" not in text
    for parts in (("/Us", "ers/"), ("/var/fol", "ders/")):
        assert _assembled_stem(parts) not in text
    assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", text)


def test_shipped_claim_map_resolves_its_paths() -> None:
    payload = json.loads((ROOT / "claim_to_code.json").read_text(encoding="utf-8"))
    assert payload["claims"]
    assert payload["deviations"]
    for claim in payload["claims"]:
        assert claim["code_paths"]
        for entry in claim["code_paths"]:
            if "*" in entry:
                assert list(ROOT.glob(entry))
            else:
                assert (ROOT / entry).exists(), entry


def test_shipped_claim_map_records_every_deviation_reason() -> None:
    payload = json.loads((ROOT / "claim_to_code.json").read_text(encoding="utf-8"))
    for entry in payload["deviations"]:
        assert entry["location"]
        assert entry["departure"]
        assert entry["reason"]


def test_shipped_summary_matches_the_report() -> None:
    report = json.loads((ROOT / "verification_report.json").read_text(encoding="utf-8"))
    text = (ROOT / "verification_summary.txt").read_text(encoding="utf-8")
    assert summary_text(report).strip() == text.strip()


def test_dataset_links_are_plain_and_reachable_by_name() -> None:
    lines = [
        line.strip()
        for line in (ROOT / "dataset_urls.txt").read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("http")
    ]
    assert lines
    for line in lines:
        assert " " in line, line
        assert line.split(" ")[0].startswith("http")
    joined = "\n".join(lines)
    assert "tdcommons.ai" in joined
    assert "zenodo.org/record/6481870" in joined
    assert "drugdesign.unistra.fr" not in joined
    assert "doi.org" not in joined


def test_manifest_excludes_itself_and_skips_caches() -> None:
    payload = json.loads((ROOT / "integrity_manifest.json").read_text(encoding="utf-8"))
    paths = {entry["path"] for entry in payload["files"]}
    assert "integrity_manifest.json" not in paths
    assert not any("__pycache__" in path for path in paths)
    assert "src/rcrn/models/rcrn.py" in paths
    assert "configs/experiment/main.yaml" in paths
    assert "dataset_urls.txt" in paths
    assert payload["file_count"] == len(payload["files"])


def test_manifest_matches_the_live_tree_outside_the_rewritten_artefacts() -> None:
    mismatches = manifest_mismatches(ROOT, ROOT / "integrity_manifest.json")
    rewritten = set(SHIPPED)
    for key in ("missing", "unlisted", "stale"):
        remaining = [entry for entry in mismatches[key] if Path(entry).name not in rewritten]
        assert remaining == [], (key, remaining)


def test_vocabulary_and_privacy_sweep_are_clean() -> None:
    driver = VerificationDriver(ROOT, skip_tools=True)
    driver.check_hygiene()
    by_name = {check.name: check for check in driver.checks}
    assert by_name["vocabulary_sweep"].status == PASS, by_name["vocabulary_sweep"].detail
    assert by_name["privacy_sweep"].status == PASS, by_name["privacy_sweep"].detail
    assert by_name["markdown_inventory"].status == PASS, by_name["markdown_inventory"].detail
    assert by_name["continuous_integration_removed"].status == PASS


CACHE_DIRECTORIES = {"__pycache__", ".git", ".mypy_cache", ".ruff_cache", ".pytest_cache"}


def test_only_the_readme_is_a_markdown_file() -> None:
    markdown = [path.name for path in ROOT.rglob("*.md") if not CACHE_DIRECTORIES & set(path.parts)]
    assert markdown == ["README.md"]


def _assembled_stem(parts: tuple[str, ...]) -> str:
    """Build a banned stem from fragments so this file does not carry it literally."""

    return "".join(parts)


def test_masked_stems_do_not_reintroduce_the_word() -> None:
    for parts in (("syn", "thetic"), ("sim", "ulation"), ("mir", "ror")):
        stem = _assembled_stem(parts)
        masked = mask(stem)
        assert masked != stem
        assert "*" in masked
        assert stem[:2] == masked[:2]


def test_writers_leave_the_shipped_artefacts_untouched(tmp_path: Path) -> None:
    before = {name: (ROOT / name).read_bytes() for name in SHIPPED}
    report = {
        "overall_status": PASS,
        "code_status": PASS,
        "status_counts": {},
        "checks": [],
        "manuscript_discrepancies": [],
        "interpretation": "scratch",
    }
    (tmp_path / "configs" / "experiment").mkdir(parents=True)
    (tmp_path / "configs" / "experiment" / "main.yaml").write_text(
        "name: scratch\n", encoding="utf-8"
    )
    written = write_artefacts(tmp_path, report)
    assert set(written) == {
        "claim_to_code",
        "verification_report",
        "verification_summary",
        "integrity_manifest",
    }
    for name, payload in before.items():
        assert (ROOT / name).read_bytes() == payload
    scratch = json.loads((tmp_path / "claim_to_code.json").read_text(encoding="utf-8"))
    assert scratch["claims"]
    assert scratch["deviations"]


def test_writer_writes_the_manifest_last(tmp_path: Path) -> None:
    payload = {
        "overall_status": PASS,
        "code_status": PASS,
        "status_counts": {},
        "checks": [],
        "manuscript_discrepancies": [],
        "interpretation": "scratch",
    }
    written = write_artefacts(tmp_path, payload)
    manifest = json.loads((tmp_path / "integrity_manifest.json").read_text(encoding="utf-8"))
    paths = {entry["path"] for entry in manifest["files"]}
    assert "verification_report.json" in paths
    assert "verification_summary.txt" in paths
    assert (
        written["integrity_manifest"].stat().st_mtime_ns
        >= written["verification_report"].stat().st_mtime_ns
    )


def test_claim_map_writer_is_idempotent(tmp_path: Path) -> None:
    first = write_claim_map(tmp_path, CLAIMS, DEVIATIONS)
    initial = first.read_bytes()
    second = write_claim_map(tmp_path, CLAIMS, DEVIATIONS)
    assert second.read_bytes() == initial


def test_manifest_round_trip_on_a_scratch_tree(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    write_integrity_manifest(tmp_path)
    assert all(not value for value in manifest_mismatches(tmp_path).values())


def test_claim_ids_are_unique() -> None:
    identifiers = [claim["claim_id"] for claim in CLAIMS]
    assert len(identifiers) == len(set(identifiers))


@pytest.mark.parametrize("name", [claim["claim_id"] for claim in CLAIMS])
def test_every_claim_names_a_location_and_a_path(name: str) -> None:
    claim = next(entry for entry in CLAIMS if entry["claim_id"] == name)
    assert claim["paper_location"]
    assert claim["statement"]
    assert claim["code_paths"]
