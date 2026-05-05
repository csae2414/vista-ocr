"""Tests for ``vista_ocr.utils.shard_glob.expand_shards``.

The same brace-range expansion logic was previously duplicated in
``tools/synth_target_distributions.py`` and ``tools/audit_tokenizer.py``
with no shared test coverage; this file is the canonical regression
suite for the helper. Each test imports ``expand_shards`` directly
from the package; tests do NOT import from each other.
"""
from __future__ import annotations

from pathlib import Path

from vista_ocr.utils.shard_glob import expand_shards


def test_brace_range_expands(tmp_path: Path):
    """Single-range brace pattern must expand to all matching files,
    sorted, when those files exist on disk."""
    for i in (1, 2, 3):
        (tmp_path / f"x-{i:04d}.tar").write_bytes(b"")
    pattern = str(tmp_path / "x-{0001..0003}.tar")
    out = expand_shards(pattern)
    assert [Path(p).name for p in out] == ["x-0001.tar", "x-0002.tar", "x-0003.tar"]


def test_brace_range_skips_missing_files(tmp_path: Path):
    """The pattern's range exceeds what's on disk; expander must
    silently drop missing files, not raise. Mirrors ``glob.glob``
    semantics."""
    (tmp_path / "x-0001.tar").write_bytes(b"")
    (tmp_path / "x-0003.tar").write_bytes(b"")
    pattern = str(tmp_path / "x-{0001..0005}.tar")
    out = expand_shards(pattern)
    assert [Path(p).name for p in out] == ["x-0001.tar", "x-0003.tar"]


def test_plain_glob_fallback(tmp_path: Path):
    """No brace range -> falls back to plain ``glob.glob``."""
    (tmp_path / "x-0001.tar").write_bytes(b"")
    (tmp_path / "x-0002.tar").write_bytes(b"")
    out = expand_shards(str(tmp_path / "x-*.tar"))
    assert [Path(p).name for p in out] == ["x-0001.tar", "x-0002.tar"]


def test_inverted_brace_range_normalises(tmp_path: Path):
    """``{0003..0001}`` should expand the same way as ``{0001..0003}``
    (matches what bash does with brace ranges in either direction)."""
    for i in (1, 2, 3):
        (tmp_path / f"x-{i:04d}.tar").write_bytes(b"")
    pattern = str(tmp_path / "x-{0003..0001}.tar")
    out = expand_shards(pattern)
    assert [Path(p).name for p in out] == ["x-0001.tar", "x-0002.tar", "x-0003.tar"]
