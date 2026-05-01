"""Inference-API tests using a tiny untrained model.

These don't check OCR quality (the model is random) — they verify that the
prompt → generation → output-parser pipeline is wired correctly and that
each task returns the right Python type."""
from __future__ import annotations

import pytest
import torch

from vista_ocr.data.synth.synthdog_bbox import SynthDogConfig, generate_sample
from vista_ocr.inference.generate import (
    InferenceConfig,
    find_it,
    ocr_with_layout,
    region_ocr,
)
from vista_ocr.models.decoder import small_random_decoder
from vista_ocr.models.encoder import FCNEncoderWidther
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.build_spm import train_spm
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import (
    VistaTokenizer,
    list_special_and_spatial_tokens,
)


@pytest.fixture(scope="module")
def grid() -> SpatialGrid:
    return SpatialGrid(canvas_h=128, canvas_w=128, quantizer_px=8, scheme="original")


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory, grid):
    tmp = tmp_path_factory.mktemp("inf")
    corpus = tmp / "c.txt"
    corpus.write_text(
        (
            "hello world\nthe quick brown fox jumps over the lazy dog\n"
            "Sphinx of black quartz judge my vow\n"
            "Pack my box with five dozen liquor jugs\n"
            "abc def ghi jkl mno pqr stu vwx yz\n"
            "0 1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90\n"
        ) * 400,
        encoding="utf-8",
    )
    out = tmp / "inf"
    train_spm(corpus, out, vocab_size=140, user_symbols=list_special_and_spatial_tokens(grid))
    return VistaTokenizer(out.with_suffix(".model"), grid)


@pytest.fixture(scope="module")
def model(tokenizer: VistaTokenizer) -> VistaOCR:
    torch.manual_seed(0)
    enc = FCNEncoderWidther(input_channels=1, dropout=0.0)
    dec = small_random_decoder(
        vocab_size=tokenizer.vocab_size, d_model=1024, n_layers=1, n_heads=4, ffn_dim=128
    )
    return VistaOCR(enc, dec).eval()


def test_ocr_with_layout_returns_list(model, tokenizer):
    sample = generate_sample(["hi"], SynthDogConfig(canvas_h=64, canvas_w=64))
    cfg = InferenceConfig(max_new_tokens=8, target_h=128, target_w=128, pad_multiple=32)
    out = ocr_with_layout(model, sample.image, tokenizer, cfg)
    assert isinstance(out, list)


def test_region_ocr_returns_string(model, tokenizer):
    sample = generate_sample(["hi"], SynthDogConfig(canvas_h=64, canvas_w=64))
    cfg = InferenceConfig(max_new_tokens=8, target_h=128, target_w=128, pad_multiple=32)
    out = region_ocr(model, sample.image, (0, 0, 30, 30), tokenizer, cfg)
    assert isinstance(out, str)


def test_find_it_returns_box_list(model, tokenizer):
    sample = generate_sample(["hi"], SynthDogConfig(canvas_h=64, canvas_w=64))
    cfg = InferenceConfig(max_new_tokens=8, target_h=128, target_w=128, pad_multiple=32)
    out = find_it(model, sample.image, "hi", tokenizer, cfg)
    assert isinstance(out, list)
    for b in out:
        assert len(b) == 4
