"""Tests for vista_ocr.data.single_line."""
from __future__ import annotations

from unittest.mock import patch

from PIL import Image

from vista_ocr.data.single_line import (
    SingleLineConfig,
    crop_line,
    iter_single_line_samples,
)
from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.tokenizer import Line


def test_crop_line_emits_trivial_bbox():
    img = Image.new("L", (200, 100), 255)
    line = Line(text="hello", bbox=(20, 30, 120, 60))
    sample = crop_line(img, line, SingleLineConfig())
    assert sample is not None
    assert sample.image.size == (108, 38)  # +pad on each side
    assert sample.lines[0].text == "hello"
    cw, ch = sample.image.size
    assert sample.lines[0].bbox == (0, 0, cw, ch)


def test_crop_line_drops_too_short():
    img = Image.new("L", (200, 100), 255)
    line = Line(text="x", bbox=(0, 0, 10, 10))   # h=10 < min_line_h
    assert crop_line(img, line, SingleLineConfig()) is None


def test_crop_line_drops_too_tall():
    img = Image.new("L", (200, 200), 255)
    line = Line(text="x", bbox=(0, 0, 100, 150))  # h=150 > max_line_h
    assert crop_line(img, line, SingleLineConfig()) is None


def test_crop_line_clamps_to_image_bounds():
    img = Image.new("L", (200, 100), 255)
    # bbox close to right edge; pad_px must clamp.
    line = Line(text="hello", bbox=(190, 30, 199, 60))
    sample = crop_line(img, line, SingleLineConfig())
    # Width below min_line_w (32) -> dropped.
    assert sample is None


def test_crop_line_min_line_w_enforced():
    img = Image.new("L", (200, 100), 255)
    line = Line(text="hi", bbox=(0, 30, 20, 60))   # w=20 < min_line_w
    assert crop_line(img, line, SingleLineConfig()) is None


def _fake_pages():
    """Two synthetic pages: page 1 has 2 keepable lines + 1 too-short;
    page 2 has 1 keepable line. iter_single_line_samples should yield 3."""
    img1 = Image.new("L", (300, 200), 255)
    img2 = Image.new("L", (300, 200), 255)
    page1 = Sample(
        image=img1,
        lines=[
            Line(text="alpha", bbox=(10, 20, 110, 50)),
            Line(text="beta",  bbox=(10, 80, 110, 110)),
            Line(text="x",     bbox=(0, 0, 10, 10)),    # too short -> drop
        ],
        task="ocr_layout",
        source="pdfa",
    )
    page2 = Sample(
        image=img2,
        lines=[Line(text="gamma", bbox=(10, 20, 110, 50))],
        task="ocr_layout",
        source="pdfa",
    )
    return [page1, page2]


def test_iter_single_line_samples_yields_one_per_keepable_line():
    """C2: the public iterator emits one Sample per keepable line and
    drops degenerate ones. Tested by mocking iter_pdfa with synthetic
    pages so we don't need a real shard."""
    fake_pages = _fake_pages()
    with patch("vista_ocr.data.single_line.iter_pdfa", return_value=iter(fake_pages)):
        from vista_ocr.data.pdfa import PdfaConfig
        samples = list(iter_single_line_samples(PdfaConfig(shards=["unused"])))

    assert len(samples) == 3
    assert [s.lines[0].text for s in samples] == ["alpha", "beta", "gamma"]
    # Each emitted sample carries a single trivial bbox covering the crop.
    for s in samples:
        cw, ch = s.image.size
        assert s.lines[0].bbox == (0, 0, cw, ch)
        assert s.task == "ocr_layout"
        assert s.source == "single_line:pdfa"
