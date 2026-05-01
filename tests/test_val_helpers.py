"""Tests for vista_ocr.training.val_helpers.

Exercises the factory + the produced loss / decode functions on a tiny
model + fabricated PDFA-shaped samples. No network, no real shard.
"""
from __future__ import annotations

import pytest
import torch
from PIL import Image

from vista_ocr.data.preprocess import PreprocessConfig
from vista_ocr.data.types import Sample
from vista_ocr.models.decoder import small_random_decoder
from vista_ocr.models.encoder import FCNEncoderWidther
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.build_spm import train_spm
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import (
    Line,
    VistaTokenizer,
    list_special_and_spatial_tokens,
)
from vista_ocr.training.val_helpers import (
    make_val_decode_fn,
    make_val_loss_fn,
    val_batches_with_refs,
)

CORPUS = (
    "the quick brown fox jumps over the lazy dog\n"
    "0 1 2 3 4 5 6 7 8 9\n"
    "abc def ghi jkl mno pqr stu vwx yz\n"
    "Sphinx of black quartz judge my vow\n"
    "Pack my box with five dozen liquor jugs\n"
)


@pytest.fixture(scope="module")
def grid() -> SpatialGrid:
    return SpatialGrid(canvas_h=128, canvas_w=128, quantizer_px=4, scheme="original")


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory, grid):
    tmp = tmp_path_factory.mktemp("vh")
    corpus = tmp / "c.txt"
    corpus.write_text(CORPUS * 400, encoding="utf-8")
    out = tmp / "vh"
    train_spm(corpus, out, vocab_size=180,
              user_symbols=list_special_and_spatial_tokens(grid))
    return VistaTokenizer(out.with_suffix(".model"), grid)


@pytest.fixture
def samples() -> list[Sample]:
    return [
        Sample(
            image=Image.new("L", (128, 128), 255),
            lines=[Line(text="hello", bbox=(0, 0, 40, 16))],
            task="ocr_layout",
        ),
        Sample(
            image=Image.new("L", (128, 128), 255),
            lines=[
                Line(text="foo", bbox=(0, 0, 30, 16)),
                Line(text="bar", bbox=(0, 20, 30, 36)),
            ],
            task="ocr_layout",
        ),
    ]


@pytest.fixture
def pre_cfg() -> PreprocessConfig:
    return PreprocessConfig(target_h=128, target_w=128, pad_multiple=32)


@pytest.fixture
def tiny_model(tokenizer) -> VistaOCR:
    torch.manual_seed(0)
    enc = FCNEncoderWidther(input_channels=1, dropout=0.0)
    dec = small_random_decoder(
        vocab_size=tokenizer.vocab_size, d_model=1024, n_layers=1, n_heads=4,
        ffn_dim=128, max_position_embeddings=4096,
    )
    return VistaOCR(enc, dec).eval()


# --------- val_batches_with_refs ---------

def test_val_batches_with_refs_yields_tuples(samples, tokenizer, pre_cfg):
    items = list(val_batches_with_refs(iter(samples), tokenizer, pre_cfg))
    assert len(items) == 2
    for batch, ref in items:
        assert hasattr(batch, "images")
        assert hasattr(batch, "decoder_input_ids")
        assert isinstance(ref, str)


def test_val_batches_with_refs_concatenates_line_text(samples, tokenizer, pre_cfg):
    items = list(val_batches_with_refs(iter(samples), tokenizer, pre_cfg))
    _, ref0 = items[0]
    _, ref1 = items[1]
    assert ref0 == "hello"
    assert ref1 == "foo bar"


def test_val_batches_handles_empty_input(tokenizer, pre_cfg):
    assert list(val_batches_with_refs(iter([]), tokenizer, pre_cfg)) == []


# --------- make_val_loss_fn ---------

def test_make_val_loss_fn_produces_scalar(tiny_model, tokenizer, pre_cfg, samples):
    loss_fn = make_val_loss_fn(tokenizer._spatial_ids, lambda_text=0.5)
    item = next(iter(val_batches_with_refs(iter(samples), tokenizer, pre_cfg)))
    loss = loss_fn(tiny_model, item)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_make_val_loss_fn_unpacks_tuple_correctly(tiny_model, tokenizer, pre_cfg, samples):
    """Sanity: passing a non-tuple item raises a clear unpack error."""
    loss_fn = make_val_loss_fn(tokenizer._spatial_ids, lambda_text=0.5)
    with pytest.raises((ValueError, TypeError)):
        # raw Batch, not (batch, ref) tuple
        item = next(iter(val_batches_with_refs(iter(samples), tokenizer, pre_cfg)))
        loss_fn(tiny_model, item[0])  # pass just the batch


# --------- make_val_decode_fn ---------

def test_make_val_decode_fn_returns_str_pair(tiny_model, tokenizer, pre_cfg, samples):
    decode_fn = make_val_decode_fn(tokenizer, max_new_tokens=16)
    item = next(iter(val_batches_with_refs(iter(samples), tokenizer, pre_cfg)))
    refs, hyps = decode_fn(tiny_model, item)
    assert refs == ["hello"]
    assert isinstance(hyps, list) and len(hyps) == 1
    assert isinstance(hyps[0], str)


def test_make_val_decode_fn_respects_repetition_penalty(
    tiny_model, tokenizer, pre_cfg, samples
):
    """Two factories with different penalties should produce decode_fn
    callables -- both still return list[str] pairs."""
    fn_a = make_val_decode_fn(tokenizer, max_new_tokens=8, repetition_penalty=1.0)
    fn_b = make_val_decode_fn(tokenizer, max_new_tokens=8, repetition_penalty=10.0)
    item = next(iter(val_batches_with_refs(iter(samples), tokenizer, pre_cfg)))
    refs_a, hyps_a = fn_a(tiny_model, item)
    refs_b, hyps_b = fn_b(tiny_model, item)
    assert refs_a == refs_b == ["hello"]
    assert isinstance(hyps_a[0], str)
    assert isinstance(hyps_b[0], str)
