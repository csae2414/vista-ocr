"""Tests for :class:`vista_ocr.data.pdfa_shard.PdfaShardReader`.

Builds synthetic tarballs in tmp_path so the tests don't need real
PDFA shards.
"""
from __future__ import annotations

import io
import json
import tarfile

import pytest

from vista_ocr.data.pdfa_shard import PdfaShardReader


def _make_shard(path, payloads: list[dict | bytes]) -> None:
    """Build a tar at ``path`` containing ``payloads`` as ``{i}.json``.

    Bytes payloads are written as-is (so we can include malformed
    entries); dicts are JSON-encoded.
    """
    with tarfile.open(path, "w") as tar:
        for i, p in enumerate(payloads):
            data = p if isinstance(p, bytes) else json.dumps(p).encode("utf-8")
            info = tarfile.TarInfo(name=f"sample{i}.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_missing_shard_raises_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        PdfaShardReader([tmp_path / "does-not-exist.tar"])


def test_iter_pages_across_two_shards(tmp_path):
    s1 = tmp_path / "a.tar"
    s2 = tmp_path / "b.tar"
    _make_shard(s1, [{"pages": [{"id": 1}, {"id": 2}]}])
    _make_shard(s2, [{"pages": [{"id": 3}]}])
    reader = PdfaShardReader([s1, s2])
    page_ids = [p["id"] for p in reader.iter_pages()]
    assert page_ids == [1, 2, 3]


def test_iter_line_texts(tmp_path):
    s = tmp_path / "a.tar"
    _make_shard(s, [{
        "pages": [
            {"lines": {"text": ["hello", "world"]}},
            {"lines": {"text": ["foo"]}},
        ],
    }])
    reader = PdfaShardReader([s])
    assert list(reader.iter_line_texts()) == ["hello", "world", "foo"]


def test_skips_non_string_text_entries(tmp_path):
    s = tmp_path / "a.tar"
    _make_shard(s, [{
        "pages": [{"lines": {"text": ["hello", 123, None, "world"]}}],
    }])
    reader = PdfaShardReader([s])
    assert list(reader.iter_line_texts()) == ["hello", "world"]


def test_skips_pages_without_lines_block(tmp_path):
    s = tmp_path / "a.tar"
    _make_shard(s, [{
        "pages": [
            {"id": 1},                                 # no lines block
            {"lines": {"text": ["alpha"]}},
            {"lines": None},                            # malformed
        ],
    }])
    reader = PdfaShardReader([s])
    assert list(reader.iter_line_texts()) == ["alpha"]


def test_malformed_json_is_skipped_not_fatal(tmp_path, caplog):
    """Tolerant by design: corrupt entries log + skip; later valid
    entries still flow through."""
    s = tmp_path / "a.tar"
    _make_shard(s, [
        b"this is not json",
        {"pages": [{"lines": {"text": ["recoverable"]}}]},
    ])
    reader = PdfaShardReader([s])
    import logging
    with caplog.at_level(logging.WARNING):
        out = list(reader.iter_line_texts())
    assert out == ["recoverable"]
    assert any("malformed json" in r.message for r in caplog.records)


def test_ignores_non_json_tar_members(tmp_path):
    """PDFA shards contain ``.pdf`` siblings; reader must skip them."""
    s = tmp_path / "a.tar"
    with tarfile.open(s, "w") as tar:
        # A fake .pdf entry alongside one .json entry.
        pdf_bytes = b"%PDF-1.4 fake"
        info = tarfile.TarInfo(name="sample0.pdf")
        info.size = len(pdf_bytes)
        tar.addfile(info, io.BytesIO(pdf_bytes))
        payload = json.dumps({"pages": [{"lines": {"text": ["x"]}}]}).encode()
        info = tarfile.TarInfo(name="sample0.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    reader = PdfaShardReader([s])
    assert list(reader.iter_line_texts()) == ["x"]
