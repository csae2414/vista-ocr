"""Unit tests for the PDFA loader's pure-Python helpers.

The PDF-render path requires pypdfium2 + a real PDF blob; the integration
test that exercises that path runs only when an actual shard is present
(``--pdfa-shard <path>``)."""
from __future__ import annotations

import pytest

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
