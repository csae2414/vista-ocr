"""Tests for vista_ocr.data.single_line."""
from __future__ import annotations

from PIL import Image

from vista_ocr.data.single_line import SingleLineConfig, crop_line
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
