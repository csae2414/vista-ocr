"""Tests for the rewritten IDL loader (notes/plan_iter_idl_fix.md).

Inventory:

- ``test_decode_idl_record_with_realistic_payload`` -- happy path:
  pdf+json record yields a single Sample with task / source / lines
  shape per the IDL schema.
- ``test_decode_idl_record_drops_low_score_lines`` -- score below
  min_line_score is filtered out.
- ``test_decode_idl_record_normalized_xywh_to_pixel_xyxy`` -- bbox
  conversion contract end-to-end on a known input.
- ``test_decode_idl_record_legacy_png_record_yields_nothing`` -- a
  record with the old png+json layout (which is what the broken
  loader expected) yields no Samples; the iterator returns ``[]``,
  not ``None``.
- ``test_iter_idl_empty_check_false`` -- WebDataset is built with
  ``empty_check=False`` so a per-worker empty shard slice can't
  raise; matches PDFA.
- ``test_iter_idl_skips_malformed_record_with_warning`` -- one bad
  record (truncated JSON) does not abort the iterator; a WARNING
  surfaces in caplog.
- ``test_idl_config_defaults_mirror_pdfa_config`` -- dpi /
  min_line_score / flatten_multi_page default to the same values as
  PdfaConfig, locking the symmetry the loader rewrite assumes.
- ``test_iter_idl_yields_real_samples`` -- gated real-shard smoke
  (skip when ``data/raw/idl/idl-train-00000.tar`` absent). Pre-fix
  yielded 0; post-fix should yield > 10.

All unit tests monkeypatch ``vista_ocr.data.idl._render_pdf_page``
to return a fixed PIL image, keeping them fast and deterministic
across pypdfium2 versions. Real PDF rendering is exercised by the
gated real-shard smoke.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from PIL import Image

from vista_ocr.data.idl import IdlConfig, _decode_idl_record, iter_idl
from vista_ocr.data.pdfa import PdfaConfig


REPO = Path(__file__).resolve().parent.parent
REAL_SHARD = REPO / "data" / "raw" / "idl" / "idl-train-00000.tar"


def _payload(text=("hello world", "foo bar baz"),
             bbox=([0.1, 0.2, 0.3, 0.05], [0.1, 0.3, 0.4, 0.05]),
             score=(0.99, 0.95)) -> bytes:
    """Build a minimal IDL JSON sidecar matching the on-disk schema."""
    return json.dumps({
        "pages": [{
            "text": list(text),
            "bbox": [list(b) for b in bbox],
            "score": list(score),
        }],
    }).encode("utf-8")


def _record(payload_bytes: bytes | None = None,
            pdf: bytes | None = b"<dummy pdf bytes>",
            key: str = "fake_key_001",
            **extra) -> dict:
    rec: dict = {"__key__": key}
    if pdf is not None:
        rec["pdf"] = pdf
    if payload_bytes is not None:
        rec["json"] = payload_bytes
    rec.update(extra)
    return rec


@pytest.fixture
def patched_render(monkeypatch):
    """Replace _render_pdf_page with a deterministic 1000x800 raster.

    Paints a chunk of black pixels so the rendered page does NOT
    trip ``is_blank_image`` (which drops uniformly-white pages by
    design). Without this, every test sample would be dropped at
    the drop_blank gate."""
    img = Image.new("L", (1000, 800), 255)
    # Paint a 600x500 black block in the middle (~37% of page area)
    # -- well above any blank-image threshold.
    from PIL import ImageDraw
    ImageDraw.Draw(img).rectangle((100, 100, 700, 600), fill=0)
    monkeypatch.setattr(
        "vista_ocr.data.idl._render_pdf_page",
        lambda pdf_bytes, dpi: iter([img]),
    )
    return img


# ---------------------------------------------------------------------------
# _decode_idl_record
# ---------------------------------------------------------------------------

def test_decode_idl_record_with_realistic_payload(patched_render):
    cfg = IdlConfig(shards=[])
    samples = list(_decode_idl_record(_record(_payload()), cfg))
    assert len(samples) == 1
    s = samples[0]
    assert s.task == "ocr_layout"
    assert s.source == "idl:fake_key_001"
    assert len(s.lines) == 2
    assert s.lines[0].text == "hello world"
    assert s.image.size == (1000, 800)


def test_decode_idl_record_drops_low_score_lines(patched_render):
    """One line at score=0.1 (< default min_line_score=0.5) must be
    dropped; the other survives."""
    cfg = IdlConfig(shards=[])
    payload = _payload(
        text=("kept", "dropped"),
        bbox=([0.1, 0.2, 0.3, 0.05], [0.5, 0.6, 0.2, 0.05]),
        score=(0.99, 0.1),
    )
    samples = list(_decode_idl_record(_record(payload), cfg))
    assert len(samples) == 1
    assert [ln.text for ln in samples[0].lines] == ["kept"]


def test_decode_idl_record_normalized_xywh_to_pixel_xyxy(patched_render):
    """Pin the bbox conversion contract: normalised xywh
    [0.1, 0.2, 0.3, 0.05] on a 1000x800 image -> pixel xyxy
    (100, 160, 400, 200)."""
    cfg = IdlConfig(shards=[])
    payload = _payload(
        text=("only line",),
        bbox=([0.1, 0.2, 0.3, 0.05],),
        score=(0.99,),
    )
    samples = list(_decode_idl_record(_record(payload), cfg))
    assert len(samples) == 1
    assert samples[0].lines[0].bbox == (100, 160, 400, 200)


def test_decode_idl_record_legacy_png_record_yields_nothing(patched_render):
    """A record with only the old png+json layout (what the broken
    pre-fix loader expected) yields nothing now: the iterator
    returns []. NOT None -- _decode_idl_record is an iterator."""
    cfg = IdlConfig(shards=[])
    legacy = {
        "__key__": "legacy",
        "png": b"<fake png bytes>",
        "json": _payload(),
    }
    assert list(_decode_idl_record(legacy, cfg)) == []


def test_decode_idl_record_drops_non_latin(patched_render):
    """drop_non_latin is the caller's responsibility; verify it
    actually fires post-extract."""
    cfg = IdlConfig(shards=[], drop_non_latin=True)
    payload = _payload(
        text=("english line", "中文 line"),
        bbox=([0.1, 0.2, 0.3, 0.05], [0.1, 0.3, 0.3, 0.05]),
        score=(0.99, 0.99),
    )
    samples = list(_decode_idl_record(_record(payload), cfg))
    assert len(samples) == 1
    # is_latin_text drops the CJK line; English survives.
    assert all("中" not in ln.text for ln in samples[0].lines)


def test_decode_idl_record_missing_pdf_yields_nothing(patched_render):
    cfg = IdlConfig(shards=[])
    assert list(_decode_idl_record(_record(_payload(), pdf=None), cfg)) == []


def test_decode_idl_record_missing_json_yields_nothing(patched_render):
    cfg = IdlConfig(shards=[])
    assert list(_decode_idl_record(_record(payload_bytes=None), cfg)) == []


# ---------------------------------------------------------------------------
# iter_idl
# ---------------------------------------------------------------------------

def test_iter_idl_empty_check_false(monkeypatch):
    """WebDataset must be built with empty_check=False to match
    iter_pdfa; without it, an empty per-worker shard slice raises."""
    captured: dict = {}

    class _FakePipeline:
        def __init__(self, shards, **kwargs):
            captured["shards"] = shards
            captured["kwargs"] = kwargs
        def __iter__(self):
            return iter([])

    monkeypatch.setattr("webdataset.WebDataset", _FakePipeline)
    cfg = IdlConfig(shards=["a.tar"])
    list(iter_idl(cfg))
    assert captured["kwargs"].get("empty_check") is False


def test_iter_idl_skips_malformed_record_with_warning(
    monkeypatch, caplog, patched_render,
):  # patched_render fixture must come before the WebDataset patch below.
    """One malformed record (truncated JSON) does not abort the
    iterator; a WARNING surfaces. The good record is still
    emitted."""
    good = _record(_payload(), key="good_key")
    bad = _record(b"not valid json", key="bad_key")
    monkeypatch.setattr(
        "webdataset.WebDataset",
        lambda *a, **kw: iter([good, bad]),
    )
    with caplog.at_level(logging.WARNING, logger="vista_ocr.data.idl"):
        samples = list(iter_idl(IdlConfig(shards=["a.tar"])))
    assert [s.source for s in samples] == ["idl:good_key"]
    assert any(
        "skipping malformed record" in rec.message and "bad_key" in rec.message
        for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# IdlConfig <-> PdfaConfig default parity
# ---------------------------------------------------------------------------

def test_idl_config_defaults_mirror_pdfa_config():
    """The IDL loader rewrite assumes IdlConfig defaults match PdfaConfig
    for the fields they share. Locking this symmetry prevents a future
    drift where (say) PdfaConfig.dpi changes and IdlConfig silently
    keeps the old value, producing skewed mixed-stream pages."""
    icfg = IdlConfig(shards=[])
    pcfg = PdfaConfig(shards=[])
    assert icfg.dpi == pcfg.dpi
    assert icfg.min_line_score == pcfg.min_line_score
    assert icfg.flatten_multi_page == pcfg.flatten_multi_page


# ---------------------------------------------------------------------------
# Gated real-shard smoke (the regression test for the rewrite itself)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not REAL_SHARD.exists(),
    reason=f"real IDL shard not present: {REAL_SHARD}; gated smoke skipped",
)
def test_iter_idl_yields_real_samples():
    """Pre-fix this returned 0; post-fix this should return many.
    Limited to the first 25 samples so the test stays under a
    second on the L40S."""
    import itertools
    cfg = IdlConfig(shards=[str(REAL_SHARD)])
    samples = list(itertools.islice(iter_idl(cfg), 25))
    assert len(samples) >= 10
    for s in samples:
        assert s.task == "ocr_layout"
        assert s.source.startswith("idl:")
        assert s.lines, "empty Sample.lines on real IDL data"
        assert s.image.size[0] > 0 and s.image.size[1] > 0
