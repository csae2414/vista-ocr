"""HierText loader tests using fabricated annotations.

The HuggingFace ``datasets`` library and the actual download are *not*
exercised here -- those would need network and ~12 GB of cache. We only
test the pure-Python decode helpers.
"""
from __future__ import annotations

import json

from vista_ocr.data.hiertext import (
    HierTextConfig,
    _decode_annotations,
    _lines_from_annotations,
    _quad_to_aabb,
)


def _fab(line: dict) -> dict:
    """Wrap a single ``line`` dict in the paragraphs->lines schema."""
    return {"paragraphs": [{"lines": [line]}]}


def test_decode_annotations_accepts_dict_and_json_str():
    payload = {"paragraphs": []}
    assert _decode_annotations(payload) is payload
    assert _decode_annotations(json.dumps(payload)) == payload
    assert _decode_annotations(json.dumps(payload).encode("utf-8")) == payload


def test_quad_to_aabb_basic():
    assert _quad_to_aabb([[10, 20], [30, 25], [28, 50], [12, 45]]) == (10, 20, 30, 50)


def test_quad_to_aabb_rejects_zero_area():
    assert _quad_to_aabb([[5, 5], [5, 5], [5, 5], [5, 5]]) is None
    assert _quad_to_aabb([]) is None


def test_lines_from_annotations_keeps_legible_horizontal():
    ann = _fab({
        "vertices": [[10, 20], [30, 20], [30, 50], [10, 50]],
        "text": "hello",
        "legible": True,
        "handwritten": False,
        "vertical": False,
    })
    lines = _lines_from_annotations(ann, HierTextConfig())
    assert len(lines) == 1
    assert lines[0].text == "hello"
    assert lines[0].bbox == (10, 20, 30, 50)


def test_lines_from_annotations_drops_illegible():
    ann = _fab({"vertices": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "text": "x", "legible": False})
    assert _lines_from_annotations(ann, HierTextConfig()) == []


def test_lines_from_annotations_drops_vertical_when_configured():
    ann = _fab({"vertices": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "text": "x", "legible": True, "vertical": True})
    assert _lines_from_annotations(ann, HierTextConfig(skip_vertical=True)) == []
    assert len(_lines_from_annotations(ann, HierTextConfig(skip_vertical=False))) == 1


def test_lines_from_annotations_handwritten_filter():
    hw = _fab({"vertices": [[0, 0], [10, 0], [10, 10], [0, 10]],
               "text": "hw", "legible": True, "handwritten": True})
    pr = _fab({"vertices": [[0, 0], [10, 0], [10, 10], [0, 10]],
               "text": "pr", "legible": True, "handwritten": False})

    assert len(_lines_from_annotations(hw, HierTextConfig(include_handwritten=False))) == 0
    assert len(_lines_from_annotations(hw, HierTextConfig(include_handwritten=True))) == 1
    assert len(_lines_from_annotations(pr, HierTextConfig(include_printed=False))) == 0
    assert len(_lines_from_annotations(pr, HierTextConfig(include_printed=True))) == 1


def test_lines_from_annotations_skips_blank_text():
    ann = _fab({"vertices": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "text": "   ", "legible": True})
    assert _lines_from_annotations(ann, HierTextConfig()) == []


def test_lines_from_annotations_aggregates_across_paragraphs():
    ann = {
        "paragraphs": [
            {"lines": [{"vertices": [[0, 0], [10, 0], [10, 10], [0, 10]],
                        "text": "para1-line1", "legible": True}]},
            {"lines": [
                {"vertices": [[0, 20], [10, 20], [10, 30], [0, 30]],
                 "text": "para2-line1", "legible": True},
                {"vertices": [[0, 40], [10, 40], [10, 50], [0, 50]],
                 "text": "para2-line2", "legible": True},
            ]},
        ]
    }
    out = _lines_from_annotations(ann, HierTextConfig())
    assert [ln.text for ln in out] == ["para1-line1", "para2-line1", "para2-line2"]
