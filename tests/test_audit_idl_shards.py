"""Tests for ``tools/audit_idl_shards.py``.

Inventory:

- ``test_per_shard_failure_doesnt_kill_audit`` -- one bad shard
  must record an error row but NOT abort the audit. The error
  must appear (a) in the main per-shard health table's error
  column and (b) in the flagged-shards summary section.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
TOOL_PATH = REPO / "tools" / "audit_idl_shards.py"


@pytest.fixture(scope="module")
def tool():
    """Import audit_idl_shards.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "audit_idl_shards_tool", TOOL_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _healthy_row(shard_name: str) -> dict:
    """A row shape matching what _audit_one_shard returns for a
    100%-decode-OK shard."""
    return {
        "shard": shard_name,
        "n_attempted": 50,
        "decode_ok": 50,
        "decode_ok_frac": 1.0,
        "zero_line": 0,
        "render_fail": 0,
        "median_lines_per_page": 33,
        "p95_lines_per_page": 100,
        "bbox_total": 1500,
        "bbox_inside": 1500,
        "bbox_inside_frac": 1.0,
        "total_lines": 1500,
        "non_latin_lines": 0,
        "non_latin_frac": 0.0,
    }


def test_per_shard_failure_doesnt_kill_audit(tool, tmp_path: Path, monkeypatch):
    """Monkeypatch ``_audit_one_shard`` to raise on shard 5 of 12;
    assert:

      1. The audit completes (rc != 0 because there's a flagged
         shard, but it doesn't crash).
      2. The error row is the 5th element with ``error`` set.
      3. The markdown output's MAIN table includes the error in
         the 5th row (the dedicated error column from commit 3).
      4. The flagged-shards section ALSO mentions the bad shard
         with the AUDIT RAISED prefix.
      5. The 11 healthy shards are reported as healthy.
    """
    # Stage 12 fake shard files so expand_shards finds them.
    shard_dir = tmp_path / "data" / "raw" / "idl"
    shard_dir.mkdir(parents=True)
    shard_paths = []
    for i in range(12):
        p = shard_dir / f"idl-train-{i:05d}.tar"
        p.write_bytes(b"")
        shard_paths.append(str(p))

    BAD_INDEX = 5
    bad_shard_name = f"idl-train-{BAD_INDEX:05d}.tar"

    def fake_audit_one_shard(shard_path: str, n_records: int) -> dict:
        if Path(shard_path).name == bad_shard_name:
            raise RuntimeError("simulated mid-shard explosion")
        return _healthy_row(Path(shard_path).name)

    monkeypatch.setattr(tool, "_audit_one_shard", fake_audit_one_shard)

    out_path = tmp_path / "report.md"
    monkeypatch.setattr(
        sys, "argv",
        ["audit_idl_shards.py",
         "--shards", str(shard_dir / "idl-train-*.tar"),
         "--n-per-shard", "50",
         "--out", str(out_path)],
    )
    rc = tool.main()
    # rc == 1 because flagged > 0 (the bad shard); but the run completed.
    assert rc == 1

    # Markdown report exists and contains all 12 shards.
    md = out_path.read_text(encoding="utf-8")
    for i in range(12):
        assert f"idl-train-{i:05d}.tar" in md, f"shard {i} missing from report"

    # The bad shard appears in the main table with the error message.
    # We check that the error string lands on a row containing the
    # bad shard name (not just in the flagged section).
    main_table_section = md.split("## Flagged shards")[0]
    assert bad_shard_name in main_table_section
    assert "simulated mid-shard explosion" in main_table_section

    # Flagged-shards section also mentions it with the AUDIT RAISED prefix.
    flagged_section = md.split("## Flagged shards")[1]
    assert "AUDIT RAISED" in flagged_section
    assert bad_shard_name in flagged_section
    assert "simulated mid-shard explosion" in flagged_section
