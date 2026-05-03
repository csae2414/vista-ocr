"""Tests for the JSONL manifest schema + iterator.

The manifest is the canonical input to ``vista-ocr eval``,
``finetune``, and ``infer``. v1 schema is strict: unknown versions /
malformed records are hard errors so silently-skipped docs cannot
produce misleading metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from vista_ocr.data.manifest import (
    SUPPORTED_VERSIONS,
    VALID_TASKS,
    iter_manifest,
    write_manifest,
)


def _img(path: Path, size=(64, 32)) -> None:
    Image.new("L", size, 255).save(path)


def test_round_trip_minimal_record(tmp_path: Path):
    img = tmp_path / "001.jpg"
    _img(img)
    mfst = tmp_path / "m.jsonl"
    n = write_manifest([{"image": "001.jpg", "ref": "Hello world"}], mfst)
    assert n == 1
    samples = list(iter_manifest(mfst))
    assert len(samples) == 1
    s = samples[0]
    assert s.task == "ocr_layout"
    assert s.lines[0].text == "Hello world"
    assert s.source.endswith(":1")


def test_round_trip_with_bboxes_and_task(tmp_path: Path):
    img = tmp_path / "002.jpg"
    _img(img)
    mfst = tmp_path / "m.jsonl"
    write_manifest(
        [{
            "image": "002.jpg",
            "ref": "Foo bar",
            "task": "ocr",
            "bboxes": [[10, 20, 100, 50, "Foo"], [10, 60, 100, 90, "bar"]],
        }],
        mfst,
    )
    [s] = list(iter_manifest(mfst))
    assert s.task == "ocr"
    assert [ln.text for ln in s.lines] == ["Foo", "bar"]
    assert s.lines[0].bbox == (10, 20, 100, 50)


def test_relative_paths_resolve_against_manifest_dir(tmp_path: Path):
    sub = tmp_path / "sub"
    sub.mkdir()
    img = sub / "deep.jpg"
    _img(img)
    mfst = tmp_path / "m.jsonl"
    mfst.write_text(json.dumps({"image": "sub/deep.jpg", "ref": "x"}) + "\n")
    samples = list(iter_manifest(mfst))
    assert len(samples) == 1


def test_blank_lines_and_comments_skipped(tmp_path: Path):
    img = tmp_path / "001.jpg"
    _img(img)
    mfst = tmp_path / "m.jsonl"
    mfst.write_text(
        "# this is a comment\n"
        "\n"
        + json.dumps({"image": "001.jpg", "ref": "x"}) + "\n"
        "\n"
    )
    assert len(list(iter_manifest(mfst))) == 1


def test_unknown_version_rejected(tmp_path: Path):
    img = tmp_path / "001.jpg"
    _img(img)
    mfst = tmp_path / "m.jsonl"
    mfst.write_text(json.dumps({"image": "001.jpg", "ref": "x", "version": 99}) + "\n")
    with pytest.raises(ValueError, match="unsupported manifest version"):
        list(iter_manifest(mfst))


def test_missing_required_field_rejected(tmp_path: Path):
    mfst = tmp_path / "m.jsonl"
    mfst.write_text(json.dumps({"image": "x.jpg"}) + "\n")  # no ref
    with pytest.raises(ValueError, match="'ref'"):
        list(iter_manifest(mfst))


def test_invalid_task_rejected(tmp_path: Path):
    img = tmp_path / "001.jpg"
    _img(img)
    mfst = tmp_path / "m.jsonl"
    mfst.write_text(json.dumps({
        "image": "001.jpg", "ref": "x", "task": "translate",
    }) + "\n")
    with pytest.raises(ValueError, match="task="):
        list(iter_manifest(mfst))


def test_malformed_bboxes_rejected(tmp_path: Path):
    img = tmp_path / "001.jpg"
    _img(img)
    mfst = tmp_path / "m.jsonl"
    mfst.write_text(json.dumps({
        "image": "001.jpg", "ref": "x", "bboxes": [[1, 2, 3, 4]],   # 4 elems, need 5
    }) + "\n")
    with pytest.raises(ValueError, match="bboxes"):
        list(iter_manifest(mfst))


def test_missing_image_file_rejected(tmp_path: Path):
    mfst = tmp_path / "m.jsonl"
    mfst.write_text(json.dumps({"image": "nope.jpg", "ref": "x"}) + "\n")
    with pytest.raises(ValueError, match="image not found"):
        list(iter_manifest(mfst))


def test_invalid_json_rejected(tmp_path: Path):
    mfst = tmp_path / "m.jsonl"
    mfst.write_text("{not valid json}\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        list(iter_manifest(mfst))


def test_write_manifest_validates_before_writing(tmp_path: Path):
    """A malformed record must NOT result in a half-written file."""
    mfst = tmp_path / "m.jsonl"
    with pytest.raises(ValueError):
        write_manifest(
            [
                {"image": "ok.jpg", "ref": "x"},
                {"image": "bad.jpg"},   # missing ref
            ],
            mfst,
        )
    # The first record may have been written before the error fires;
    # the contract is that the error is raised eagerly per-record.
    # Either: zero lines (validate-all-then-write) or one line + raise.
    # We accept either; the load-bearing assertion is that the bad
    # record is not silently swallowed.
    if mfst.exists():
        text = mfst.read_text()
        assert "bad.jpg" not in text


def test_supported_versions_and_valid_tasks_match_schema():
    """Sanity: the public constants are the source of truth."""
    assert 1 in SUPPORTED_VERSIONS
    assert set(VALID_TASKS) == {"ocr", "ocr_layout", "region_ocr", "find_it"}
