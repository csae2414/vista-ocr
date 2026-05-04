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


def test_sroie_drops_zero_width_bbox():
    """Real SROIE files contain occasional degenerate annotations where
    the quad corners collapse to a vertical line (x1 == x2). Such
    bboxes crash Albumentations mid-training (x_max <= x_min). Filter
    at parse time."""
    line = _parse_quad_line("100,200,100,200,100,210,100,210,18")
    assert line is None


def test_sroie_drops_zero_height_bbox():
    """Same shape, horizontal-line collapse (y1 == y2)."""
    line = _parse_quad_line("100,200,150,200,150,200,100,200,total")
    assert line is None


def test_sroie_drops_point_bbox():
    """All four corners coincide -> degenerate point."""
    line = _parse_quad_line("100,200,100,200,100,200,100,200,x")
    assert line is None


def test_sroie_to_manifest_skips_degenerate_quads(tmp_path):
    """The manifest emitter must mirror iter_sroie's filter so a
    downstream `vista-ocr finetune --train-manifest` can never receive
    a zero-area bbox."""
    import json
    import subprocess
    import sys
    from PIL import Image

    REPO = Path(__file__).resolve().parent.parent
    EMITTER = REPO / "scripts" / "datasets" / "sroie_to_manifest.py"

    root = tmp_path / "sroie"
    test = root / "test"
    test.mkdir(parents=True)
    Image.new("L", (200, 100), 255).save(test / "doc.jpg")
    (test / "doc.txt").write_text(
        "10,20,100,20,100,50,10,50,GOOD\n"     # well-formed
        "100,200,100,200,100,210,100,210,18\n"  # zero width
        "200,300,250,300,250,300,200,300,X\n"   # zero height
    )
    out = tmp_path / "test.jsonl"
    r = subprocess.run(
        [sys.executable, str(EMITTER),
         "--root", str(root), "--split", "test", "--out", str(out)],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stderr
    [rec] = [json.loads(line) for line in out.read_text().splitlines()]
    # Only the well-formed line survives.
    assert rec["bboxes"] == [[10, 20, 100, 50, "GOOD"]]


def test_sroie_iter_reads_train_split(tmp_path: Path):
    root = tmp_path / "sroie"
    train = root / "train"
    train.mkdir(parents=True)
    Image.new("L", (200, 100), 255).save(train / "r1.jpg")
    (train / "r1.txt").write_text("0,0,10,0,10,10,0,10,STORE NAME\n")
    out = list(iter_sroie(SroieConfig(root=root, split="train")))
    assert len(out) == 1
    assert out[0].lines[0].text == "STORE NAME"


def test_sroie_val_batches_yields_batch_and_ref(tmp_path: Path):
    """SROIE finetune chain wires val on the test split via
    ``sroie_val_batches`` (mirrors ``pdfa_val_batches``). Smoke-test
    that each yielded item is a (Batch, ref_str) pair where ``ref_str``
    is the whitespace-joined line text."""
    from vista_ocr.data.preprocess import PreprocessConfig
    from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
    from vista_ocr.tokenizer.build_spm import train_spm
    from vista_ocr.tokenizer.tokenizer import (
        VistaTokenizer,
        list_special_and_spatial_tokens,
    )
    from vista_ocr.training.val_helpers import sroie_val_batches

    root = tmp_path / "sroie"
    test = root / "test"
    test.mkdir(parents=True)
    Image.new("L", (128, 64), 255).save(test / "doc.jpg")
    (test / "doc.txt").write_text(
        "0,0,40,0,40,10,0,10,FOO\n"
        "0,20,40,20,40,30,0,30,BAR\n"
    )

    grid = SpatialGrid(canvas_h=64, canvas_w=64, quantizer_px=4, scheme="original")
    spm_dir = tmp_path / "spm"
    spm_dir.mkdir()
    corpus = spm_dir / "c.txt"
    corpus.write_text(
        (
            "hello world FOO BAR baz qux quux corge grault garply\n"
            "abc def ghi jkl mno pqr stu vwx yz\n"
            "the quick brown fox jumps over the lazy dog\n"
            "Sphinx of black quartz judge my vow\n"
        ) * 200,
        encoding="utf-8",
    )
    train_spm(corpus, spm_dir / "tr", vocab_size=120,
              user_symbols=list_special_and_spatial_tokens(grid))
    tokenizer = VistaTokenizer(spm_dir / "tr.model", grid)
    pre_cfg = PreprocessConfig(target_h=64, target_w=64, pad_multiple=32)

    items = list(sroie_val_batches(root, tokenizer, pre_cfg, split="test"))
    assert len(items) == 1
    batch, ref = items[0]
    assert ref == "FOO BAR"
    # batch is a real Batch, not just a tensor -- it has the canonical fields.
    assert hasattr(batch, "images") and hasattr(batch, "labels")
