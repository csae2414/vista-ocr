"""Training-loop and schedule tests.

The overfit-on-one-image test is the smoke gate for task #13: it verifies
that gradient flow, tokenizer, loss and model wiring are all correct on
CPU before we touch the GPU VM.
"""
from __future__ import annotations

from itertools import cycle, islice

import pytest
import torch
from PIL import Image

from vista_ocr.data.synth.synthdog_bbox import SynthDogConfig, generate_sample
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
from vista_ocr.training.schedules import exponential_dropout, linear_warmup_cosine
from vista_ocr.training.train_loop import TrainConfig, train


def test_linear_warmup_cosine_shape():
    cfg = dict(warmup_steps=10, total_steps=100, base_lr=1e-3)
    assert linear_warmup_cosine(0, **cfg) < 1e-3
    assert linear_warmup_cosine(9, **cfg) <= 1e-3
    assert linear_warmup_cosine(10, **cfg) == pytest.approx(1e-3)
    # After total_steps it stays at the floor.
    assert linear_warmup_cosine(200, **cfg) == 0.0


def test_exponential_dropout_grows_to_one():
    assert exponential_dropout(0) == pytest.approx(0.0)
    assert exponential_dropout(int(1e6)) > 0.999


@pytest.fixture(scope="module")
def grid() -> SpatialGrid:
    return SpatialGrid(canvas_h=128, canvas_w=128, quantizer_px=4, scheme="original")


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory, grid):
    tmp = tmp_path_factory.mktemp("spm_tr")
    corpus = tmp / "c.txt"
    corpus.write_text(
        (
            "hello world\nfoo bar\nthe quick brown fox jumps over the lazy dog\n"
            "abc def ghi jkl mno pqr stu vwx yz\n"
            "Sphinx of black quartz judge my vow\n"
            "Pack my box with five dozen liquor jugs\n"
            "0 1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100\n"
        ) * 400,
        encoding="utf-8",
    )
    out = tmp / "tr"
    train_spm(corpus, out, vocab_size=180, user_symbols=list_special_and_spatial_tokens(grid))
    return VistaTokenizer(out.with_suffix(".model"), grid)


@pytest.fixture(scope="module")
def tiny_model(tokenizer: VistaTokenizer) -> VistaOCR:
    torch.manual_seed(0)
    enc = FCNEncoderWidther(input_channels=1, dropout=0.0)
    dec = small_random_decoder(
        vocab_size=tokenizer.vocab_size,
        d_model=1024,
        n_layers=1,
        n_heads=4,
        ffn_dim=128,
    )
    return VistaOCR(encoder=enc, decoder=dec)


def test_one_optim_step_runs(tiny_model: VistaOCR, tokenizer: VistaTokenizer):
    sample = generate_sample(["hi"], SynthDogConfig(canvas_h=64, canvas_w=64, line_height=20))
    cfg = TrainConfig(
        base_lr=1e-3,
        warmup_steps=1,
        total_steps=4,
        micro_batch_size=1,
        grad_accum_steps=1,
        target_h=128,
        target_w=128,
        pad_multiple=32,
    )
    history = train(tiny_model, [sample], tokenizer, cfg, max_steps=1)
    assert len(history) == 1
    assert history[0].lr > 0


def test_overfit_single_sample_loss_decreases(
    tiny_model: VistaOCR, tokenizer: VistaTokenizer
):
    """Smoke test (task #13). On a 1-sample dataset, the loss should drop
    monotonically over a few dozen steps."""
    sample = generate_sample(
        ["abc"], SynthDogConfig(canvas_h=64, canvas_w=64, line_height=24)
    )
    sample.task = "ocr"

    cfg = TrainConfig(
        base_lr=3e-3,
        warmup_steps=2,
        total_steps=60,
        micro_batch_size=1,
        grad_accum_steps=1,
        target_h=128,
        target_w=128,
        pad_multiple=32,
        lambda_text=1.0,             # OCR-only sample has no spatial tokens.
    )
    stream = cycle([sample])
    history = train(tiny_model, islice(stream, 60), tokenizer, cfg, max_steps=60)

    first_loss = sum(h.loss for h in history[:3]) / 3
    last_loss = sum(h.loss for h in history[-3:]) / 3
    assert last_loss < first_loss * 0.6, (first_loss, last_loss)


def test_optimizer_excludes_frozen_params(tiny_model: VistaOCR):
    from vista_ocr.training.train_loop import make_optimizer
    tiny_model.freeze_decoder(True)
    cfg = TrainConfig(base_lr=1e-4)
    opt = make_optimizer(tiny_model, cfg)
    opt_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    decoder_param_ids = {id(p) for p in tiny_model.decoder.parameters()}
    assert decoder_param_ids.isdisjoint(opt_param_ids)
    tiny_model.freeze_decoder(False)


def test_optimizer_uses_mbart_betas_and_eps(tiny_model: VistaOCR):
    from vista_ocr.training.train_loop import make_optimizer
    cfg = TrainConfig(adam_betas=(0.9, 0.98), adam_eps=1e-6)
    opt = make_optimizer(tiny_model, cfg)
    for g in opt.param_groups:
        assert g["betas"] == (0.9, 0.98)
        assert g["eps"] == 1e-6


def test_freeze_decoder_disables_decoder_grads(
    tiny_model: VistaOCR, tokenizer: VistaTokenizer
):
    sample = generate_sample(["hi"], SynthDogConfig(canvas_h=64, canvas_w=64))
    sample.task = "ocr"
    cfg = TrainConfig(
        base_lr=1e-4,
        warmup_steps=1,
        total_steps=4,
        target_h=128,
        target_w=128,
        pad_multiple=32,
        freeze_decoder=True,
        lambda_text=1.0,
    )
    train(tiny_model, [sample], tokenizer, cfg, max_steps=1)
    assert all(not p.requires_grad for p in tiny_model.decoder.parameters())
    # Restore.
    tiny_model.freeze_decoder(False)
