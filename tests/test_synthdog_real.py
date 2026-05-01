"""Tests for the realistic SynthDOG generator."""
from __future__ import annotations

from pathlib import Path

import pytest

from vista_ocr.data.synth.synthdog_real import (
    SynthDocConfig,
    _scan_fonts,
    _wrap_words_to_lines,
    generate_sample,
    iter_synthdoc,
)


def _have_fonts() -> bool:
    return len(_scan_fonts()) > 0


pytestmark = pytest.mark.skipif(
    not _have_fonts(),
    reason="No system .ttf fonts available; run `apt install fonts-dejavu`",
)


def test_word_wrap_respects_width():
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("L", (200, 200), 255)
    draw = ImageDraw.Draw(img)
    fonts = _scan_fonts()
    font = ImageFont.truetype(str(fonts[0]), 16)
    text = "the quick brown fox jumps over the lazy dog " * 5
    out = _wrap_words_to_lines(draw, font, text, max_width=180)
    assert len(out) > 1
    for line in out:
        bbox = draw.textbbox((0, 0), line, font=font)
        assert bbox[2] - bbox[0] <= 180


def test_generate_sample_returns_lines_with_real_bboxes():
    cfg = SynthDocConfig(canvas_h=400, canvas_w=300, seed=0, blur_prob=0.0, rotate_deg=0.0)
    sample = generate_sample(
        ["the quick brown fox jumps over the lazy dog. " * 5],
        cfg,
    )
    assert sample.lines, "Renderer produced no lines"
    for ln in sample.lines:
        x1, y1, x2, y2 = ln.bbox
        assert 0 <= x1 < x2 <= cfg.canvas_w
        assert 0 <= y1 < y2 <= cfg.canvas_h


def test_iter_synthdoc_yields_multiple_pages():
    cfg = SynthDocConfig(canvas_h=300, canvas_w=300, seed=1, blur_prob=0.0, rotate_deg=0.0)
    paragraphs = ["alpha bravo charlie delta echo foxtrot golf hotel. " * 3] * 8
    out = list(iter_synthdoc(iter(paragraphs), cfg, paragraphs_per_page=2))
    assert len(out) == 4
    for s in out:
        assert s.lines
        assert s.task == "ocr_layout"


def test_scan_fonts_finds_at_least_one():
    fonts = _scan_fonts()
    assert len(fonts) > 0
