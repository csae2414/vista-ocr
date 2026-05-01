"""MAURDOR + SROIE loader tests using fabricated fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from vista_ocr.data.maurdor import MaurdorConfig, iter_maurdor
from vista_ocr.data.sroie import SroieConfig, _parse_quad_line, iter_sroie


@pytest.fixture
def maurdor_root(tmp_path: Path) -> Path:
    root = tmp_path / "maurdor"
    (root / "images").mkdir(parents=True)
    (root / "pages").mkdir(parents=True)
    (root / "splits").mkdir(parents=True)
    Image.new("L", (200, 100), 255).save(root / "images" / "d0.png")
    Image.new("L", (200, 100), 255).save(root / "images" / "d1.png")
    (root / "pages" / "d0.json").write_text(json.dumps({
        "lang": "en",
        "lines": [{"text": "hello", "bbox": [0, 0, 10, 10]}],
    }))
    (root / "pages" / "d1.json").write_text(json.dumps({
        "lang": "ar",                 # excluded by EN filter
        "lines": [{"text": "x", "bbox": [0, 0, 5, 5]}],
    }))
    (root / "splits" / "train.txt").write_text("d0\nd1\n")
    return root


def test_maurdor_filters_to_en(maurdor_root: Path):
    out = list(iter_maurdor(MaurdorConfig(root=maurdor_root, split="train")))
    assert len(out) == 1
    assert "d0" in out[0].source
    assert out[0].lines[0].text == "hello"


def test_maurdor_includes_en_fr_when_requested(maurdor_root: Path):
    (maurdor_root / "pages" / "d2.json").write_text(json.dumps({
        "lang": "en+fr", "lines": [{"text": "bilingual", "bbox": [0, 0, 10, 10]}],
    }))
    Image.new("L", (200, 100), 255).save(maurdor_root / "images" / "d2.png")
    (maurdor_root / "splits" / "train.txt").write_text("d0\nd1\nd2\n")
    out = list(iter_maurdor(MaurdorConfig(root=maurdor_root, split="train")))
    assert {s.source.rsplit(":", 1)[1] for s in out} == {"d0", "d2"}


def test_sroie_quad_to_aabb_projection():
    line = _parse_quad_line("10,20,30,15,30,40,5,45,Hello world")
    assert line is not None
    assert line.bbox == (5, 15, 30, 45)
    assert line.text == "Hello world"


def test_sroie_handles_commas_in_text():
    line = _parse_quad_line("0,0,10,0,10,10,0,10,Hello, world, with, commas")
    assert line is not None
    assert line.text == "Hello, world, with, commas"


def test_sroie_iter_reads_train_split(tmp_path: Path):
    root = tmp_path / "sroie"
    train = root / "train"
    train.mkdir(parents=True)
    Image.new("L", (200, 100), 255).save(train / "r1.jpg")
    (train / "r1.txt").write_text("0,0,10,0,10,10,0,10,STORE NAME\n")
    out = list(iter_sroie(SroieConfig(root=root, split="train")))
    assert len(out) == 1
    assert out[0].lines[0].text == "STORE NAME"
