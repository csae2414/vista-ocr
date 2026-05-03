"""Unit tests for the PDFA loader's pure-Python helpers.

The PDF-render path requires pypdfium2 + a real PDF blob; the integration
test that exercises that path runs only when an actual shard is present
(``--pdfa-shard <path>``)."""
from __future__ import annotations

from vista_ocr.data.pdfa import _lines_for_page, _norm_bbox_to_pixels


def test_norm_bbox_to_pixels_basic():
    bbox = _norm_bbox_to_pixels([0.1, 0.2, 0.3, 0.4], img_w=1000, img_h=500)
    assert bbox == (100, 100, 400, 300)


def test_norm_bbox_to_pixels_clamps():
    bbox = _norm_bbox_to_pixels([0.9, 0.9, 0.5, 0.5], img_w=100, img_h=100)
    assert bbox == (90, 90, 100, 100)


def test_norm_bbox_to_pixels_rejects_zero_area():
    assert _norm_bbox_to_pixels([0.5, 0.5, 0.0, 0.1], 100, 100) is None
    assert _norm_bbox_to_pixels([0.5, 0.5, 0.1, 0.0], 100, 100) is None


def test_lines_for_page_filters_low_score():
    page = {
        "lines": {
            "text": ["hi", "lo", "ok"],
            "bbox": [
                [0.0, 0.0, 0.2, 0.05],
                [0.0, 0.1, 0.2, 0.05],
                [0.0, 0.2, 0.2, 0.05],
            ],
            "score": [1.0, 0.1, 0.9],
        }
    }
    lines = _lines_for_page(page, img_w=1000, img_h=1000, min_score=0.5)
    texts = [ln.text for ln in lines]
    assert texts == ["hi", "ok"]


def test_lines_for_page_handles_missing_block():
    assert _lines_for_page({}, 100, 100, 0.5) == []
    assert _lines_for_page({"lines": {}}, 100, 100, 0.5) == []


def test_pdfa_config_default_outlier_filters_disabled():
    """B4: drop_above_lines / drop_above_words default to None."""
    from vista_ocr.data.pdfa import PdfaConfig
    cfg = PdfaConfig(shards=["x"])
    assert cfg.drop_above_lines is None
    assert cfg.drop_above_words is None


# ---- Phase 1: shard cycling + reshuffle ----

class _StubPipeline:
    """Fake WebDataset pipeline that records the calls made on it.

    Lets us assert that ``iter_pdfa`` configures cycling correctly
    without spinning up a real tar + pypdfium2.
    """

    def __init__(self, **kwargs):
        self.init_kwargs = dict(kwargs)
        self.shuffle_buffer: int | None = None
        self.repeated = False

    def shuffle(self, n):
        self.shuffle_buffer = n
        return self

    def repeat(self):
        self.repeated = True
        return self

    def __iter__(self):
        return iter([])


def _patched_iter_pdfa_capture(monkeypatch):
    """Helper: patch wds.WebDataset to a stub and return the constructed pipeline."""
    import webdataset as wds
    from vista_ocr.data import pdfa as pdfa_mod

    captured: list[_StubPipeline] = []

    def _factory(*args, **kwargs):
        p = _StubPipeline(**{**({"shards": args[0]} if args else {}), **kwargs})
        captured.append(p)
        return p

    monkeypatch.setattr(wds, "WebDataset", _factory)
    return captured


def test_iter_pdfa_default_does_not_cycle(monkeypatch):
    captured = _patched_iter_pdfa_capture(monkeypatch)
    from vista_ocr.data.pdfa import PdfaConfig, iter_pdfa
    list(iter_pdfa(PdfaConfig(shards=["x.tar"])))
    assert len(captured) == 1
    p = captured[0]
    assert p.init_kwargs.get("shardshuffle") is False
    assert p.init_kwargs.get("seed") is None
    assert p.shuffle_buffer is None
    assert p.repeated is False


def test_iter_pdfa_cycle_enables_wds_primitives(monkeypatch):
    captured = _patched_iter_pdfa_capture(monkeypatch)
    from vista_ocr.data.pdfa import PdfaConfig, iter_pdfa
    list(iter_pdfa(PdfaConfig(
        shards=["x.tar", "y.tar"],
        cycle=True,
        cycle_shuffle_buffer=500,
        cycle_seed=42,
    )))
    p = captured[0]
    assert p.init_kwargs["shardshuffle"] is True
    assert p.init_kwargs["seed"] == 42
    assert p.shuffle_buffer == 500
    assert p.repeated is True


def test_iter_pdfa_cycle_buffer_default(monkeypatch):
    """Default sample-shuffle buffer is 1000 when cycle=True."""
    captured = _patched_iter_pdfa_capture(monkeypatch)
    from vista_ocr.data.pdfa import PdfaConfig, iter_pdfa
    list(iter_pdfa(PdfaConfig(shards=["x.tar"], cycle=True)))
    p = captured[0]
    assert p.shuffle_buffer == 1000
    assert p.repeated is True


def test_pdfa_config_cycle_defaults():
    from vista_ocr.data.pdfa import PdfaConfig
    cfg = PdfaConfig(shards=["x"])
    assert cfg.cycle is False
    assert cfg.cycle_shuffle_buffer == 1000
    assert cfg.cycle_seed == 0
