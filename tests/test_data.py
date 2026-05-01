"""Tests for data preprocessing, collation, mixture sampler, and the
synthetic generators."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vista_ocr.data.collate import build_target_ids, collate
from vista_ocr.data.mixture import MixedTaskStream, TaskMix
from vista_ocr.data.preprocess import (
    PreprocessConfig,
    is_blank_image,
    is_latin_text,
    pad_to_multiple,
    resize_to_canvas,
)
from vista_ocr.data.synth.sroie_synth import SroieSynthConfig, generate_sample as gen_sroie
from vista_ocr.data.synth.synthdog_bbox import SynthDogConfig, generate_sample as gen_synthdog
from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.build_spm import train_spm
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import (
    Line,
    VistaTokenizer,
    list_special_and_spatial_tokens,
)


# ---------- shared tokenizer fixture ----------

CORPUS = (
    "The quick brown fox jumps over the lazy dog.\n"
    "Pack my box with five dozen liquor jugs.\n"
    "Receipt total: $42.00. Subtotal: $39.99. Tax: $2.01.\n"
    "Read at 10,20,100,40 and find_it returns boxes.\n"
    "Digits: 0 1 2 3 4 5 6 7 8 9.\n"
)


@pytest.fixture(scope="module")
def grid() -> SpatialGrid:
    return SpatialGrid(canvas_h=1024, canvas_w=768, quantizer_px=10, scheme="original")


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory: pytest.TempPathFactory, grid: SpatialGrid) -> VistaTokenizer:
    tmp = tmp_path_factory.mktemp("spm_data")
    corpus_path = tmp / "tiny.txt"
    corpus_path.write_text(CORPUS * 200, encoding="utf-8")
    out = tmp / "data_spm"
    train_spm(corpus_path, out, vocab_size=400, user_symbols=list_special_and_spatial_tokens(grid))
    return VistaTokenizer(out.with_suffix(".model"), grid)


# ---------- preprocess ----------

def test_is_latin_text():
    assert is_latin_text("Hello world 123")
    assert not is_latin_text("こんにちは")


def test_resize_smaller_image_no_downscale():
    from PIL import Image

    img = Image.new("L", (100, 100), 255)
    out, scale, _ = resize_to_canvas(img, PreprocessConfig(target_h=3508, target_w=2480))
    assert scale == 1.0
    assert out.size == (100, 100)


def test_resize_oversize_image_downscales():
    from PIL import Image

    img = Image.new("L", (5000, 5000), 255)
    out, scale, (h, w) = resize_to_canvas(img, PreprocessConfig(target_h=2000, target_w=2000))
    assert scale < 1.0
    assert out.size == (w, h)
    assert max(h, w) <= 2000


def test_pad_to_multiple():
    from PIL import Image

    img = Image.new("L", (50, 70), 255)
    out, (dh, dw) = pad_to_multiple(img, multiple=32)
    assert out.size[0] % 32 == 0
    assert out.size[1] % 32 == 0
    assert dh > 0 and dw > 0


# ---------- collate ----------

def test_build_target_ids_layout(tokenizer: VistaTokenizer):
    sample = Sample(
        image=None,
        lines=[Line("hello", (10, 20, 100, 40))],
        task="ocr_layout",
    )
    seq, plen = build_target_ids(tokenizer, sample)
    assert seq[0] == tokenizer.bos_id
    assert seq[-1] == tokenizer.eos_id
    assert plen == 2  # bos + <task=ocr_layout>


def test_build_target_ids_region_ocr_requires_bbox(tokenizer: VistaTokenizer):
    s = Sample(image=None, lines=[Line("x", (0, 0, 1, 1))], task="region_ocr")
    with pytest.raises(ValueError):
        build_target_ids(tokenizer, s)


def test_build_target_ids_find_it_no_match_emits_no_bbox(tokenizer: VistaTokenizer):
    s = Sample(
        image=None,
        lines=[Line("hello", (10, 20, 30, 40))],
        task="find_it",
        query_text="absent",
    )
    seq, _ = build_target_ids(tokenizer, s)
    spatial_count = sum(1 for i in seq if tokenizer.is_spatial_id(i))
    assert spatial_count == 0


def test_collate_pads_to_max_in_batch(tokenizer: VistaTokenizer):
    from PIL import Image

    s1 = Sample(
        image=Image.new("L", (32, 64), 255),
        lines=[Line("a", (0, 0, 8, 16))],
        task="ocr",
    )
    s2 = Sample(
        image=Image.new("L", (48, 32), 255),
        lines=[Line("ab cd", (0, 0, 16, 16))],
        task="ocr_layout",
    )
    batch = collate([s1, s2], tokenizer, PreprocessConfig(pad_multiple=16, target_h=3508, target_w=2480))
    assert batch.images.shape[0] == 2
    assert batch.decoder_input_ids.shape == batch.labels.shape
    assert batch.images.shape[-2] % 16 == 0 and batch.images.shape[-1] % 16 == 0
    # Both samples should have prompt-mask True at position 0 (bos).
    assert batch.prompt_mask[:, 0].all()


# ---------- mixture ----------

def test_mixture_relabels_to_chosen_task():
    base = [
        Sample(image=None, lines=[Line("hello world foo", (0, 0, 50, 20))], task="ocr_layout"),
        Sample(image=None, lines=[Line("the quick brown fox", (0, 0, 80, 20))], task="ocr_layout"),
    ]
    mix = TaskMix({"ocr": 0.0, "ocr_layout": 0.0, "region_ocr": 1.0, "find_it": 0.0})
    stream = MixedTaskStream(iter(base), mix, seed=0)
    out = list(stream)
    assert all(s.task == "region_ocr" for s in out)
    assert all(s.query_bbox is not None for s in out)


# ---------- synth ----------

def test_synthdog_emits_bboxes_within_canvas():
    sample = gen_synthdog(["hello", "world"], SynthDogConfig(canvas_h=128, canvas_w=128, seed=1))
    assert len(sample.lines) <= 2
    for line in sample.lines:
        x1, y1, x2, y2 = line.bbox
        assert 0 <= x1 <= x2 <= 128
        assert 0 <= y1 <= y2 <= 128


def test_sroie_synth_produces_typical_receipt_lines():
    sample = gen_sroie(SroieSynthConfig(
        canvas_h=512, canvas_w=384, seed=42,
        # Disable augmentations so the bbox layout stays predictable.
        blur_prob=0.0, background_markup_prob=0.0, slant_prob=0.0,
        shadow_prob=0.0, poor_resolution_prob=0.0,
    ))
    texts = [ln.text for ln in sample.lines]
    assert any("TOTAL" in t for t in texts)
    assert any("SUBTOTAL" in t for t in texts)


def test_sroie_synth_with_full_augmentations():
    """Augmentations must not crash, but they may shift bboxes. Just make
    sure we still get some lines."""
    sample = gen_sroie(SroieSynthConfig(
        canvas_h=512, canvas_w=384, seed=7,
        blur_prob=1.0, background_markup_prob=1.0, slant_prob=1.0,
        shadow_prob=1.0, poor_resolution_prob=1.0,
    ))
    assert len(sample.lines) > 3
