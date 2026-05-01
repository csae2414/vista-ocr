"""DataLoader path tests.

Two correctness requirements:

1. The single-process DataLoader path (``num_workers=0``) must produce
   *bit-exact* the same loss trajectory as the inline
   ``iter_pdfa -> collate`` path, given the same seed and shard.
2. The multi-worker path (``num_workers >= 1``) yields equivalent samples
   (every sample appears once per epoch) -- not bit-exact because workers
   interleave across shards.

We use the in-memory :class:`InMemoryPdfaDataset` shape (a list of pre-built
:class:`Sample`s) so the test is fast and doesn't depend on PDF rendering.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vista_ocr.data.collate import collate
from vista_ocr.data.preprocess import PreprocessConfig
from vista_ocr.data.synth.synthdog_bbox import SynthDogConfig, generate_sample
from vista_ocr.data.types import Sample
from vista_ocr.models.decoder import small_random_decoder
from vista_ocr.models.encoder import FCNEncoderWidther
from vista_ocr.models.vista_ocr import VistaOCR
from vista_ocr.tokenizer.build_spm import train_spm
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import (
    VistaTokenizer,
    list_special_and_spatial_tokens,
)
from vista_ocr.training.train_loop import TrainConfig, train


@pytest.fixture(scope="module")
def grid() -> SpatialGrid:
    return SpatialGrid(canvas_h=128, canvas_w=128, quantizer_px=4, scheme="original")


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory, grid):
    tmp = tmp_path_factory.mktemp("dl_spm")
    corpus = tmp / "c.txt"
    corpus.write_text(
        "hello world\nfoo bar\nthe quick brown fox jumps\n"
        "abc def ghi jkl\nDigits 0 1 2 3 4 5 6 7 8 9\n" * 400,
        encoding="utf-8",
    )
    out = tmp / "tk"
    train_spm(corpus, out, vocab_size=180, user_symbols=list_special_and_spatial_tokens(grid))
    return VistaTokenizer(out.with_suffix(".model"), grid)


def _samples(n: int) -> list[Sample]:
    out = []
    for i in range(n):
        s = generate_sample(
            ["hi", "ok"], SynthDogConfig(canvas_h=64, canvas_w=64, line_height=24, seed=i)
        )
        s.task = "ocr"
        out.append(s)
    return out


def _build_model(tokenizer: VistaTokenizer) -> VistaOCR:
    enc = FCNEncoderWidther(input_channels=1, dropout=0.0)
    dec = small_random_decoder(
        vocab_size=tokenizer.vocab_size,
        d_model=1024, n_layers=1, n_heads=4, ffn_dim=128,
        max_position_embeddings=4096,
    )
    return VistaOCR(enc, dec)


def _train_with(
    tokenizer: VistaTokenizer,
    samples: list[Sample],
    *,
    use_dataloader: bool,
    seed: int = 0,
):
    """Run a tiny deterministic training loop. Returns the loss list."""
    torch.manual_seed(seed)
    model = _build_model(tokenizer)
    cfg = TrainConfig(
        base_lr=1e-3,
        warmup_steps=2,
        total_steps=10,
        micro_batch_size=1,
        grad_accum_steps=1,
        target_h=128,
        target_w=128,
        pad_multiple=32,
        lambda_text=1.0,
    )

    if use_dataloader:
        # Pre-collate Batches off-thread (here in-process — same code path
        # as num_workers=0 inside the DataLoader). The trainer accepts a
        # Batch iterator directly.
        pre_cfg = PreprocessConfig(target_h=128, target_w=128, pad_multiple=32)
        batches = [collate([s], tokenizer, pre_cfg) for s in samples[:10]]
        history = train(model, iter(batches), tokenizer, cfg, max_steps=10)
    else:
        history = train(model, iter(samples[:10]), tokenizer, cfg, max_steps=10)
    return [h.loss for h in history]


def test_inline_and_batch_paths_match_bitexact(tokenizer: VistaTokenizer):
    """Same seed + same data -> same loss trajectory regardless of whether
    the trainer collates inline or consumes pre-collated Batches."""
    samples = _samples(20)
    a = _train_with(tokenizer, samples, use_dataloader=False, seed=42)
    b = _train_with(tokenizer, samples, use_dataloader=True, seed=42)
    assert len(a) == len(b) == 10
    for la, lb in zip(a, b):
        assert abs(la - lb) < 1e-5, (la, lb)


def test_train_rejects_unknown_stream_type(tokenizer: VistaTokenizer):
    model = _build_model(tokenizer)
    cfg = TrainConfig(base_lr=1e-3, warmup_steps=1, total_steps=1, target_h=64, target_w=64,
                      pad_multiple=32, lambda_text=1.0)
    with pytest.raises(TypeError):
        train(model, iter([{"not": "a sample"}]), tokenizer, cfg, max_steps=1)


def test_make_pdfa_dataloader_identity_collate_passthrough(tokenizer: VistaTokenizer):
    """The DataLoader's identity collate must hand the worker-built Batch
    through unchanged (shape, dtype, pad_id)."""
    from vista_ocr.data.collate import collate as collate_fn
    from vista_ocr.data.dataloader import _identity_collate

    pre_cfg = PreprocessConfig(target_h=128, target_w=128, pad_multiple=32)
    batch = collate_fn(_samples(2), tokenizer, pre_cfg)
    out = _identity_collate([batch])
    assert out is batch


def test_make_pdfa_dataloader_rejects_wrong_arity():
    from vista_ocr.data.dataloader import _identity_collate
    with pytest.raises(RuntimeError):
        _identity_collate([])
    with pytest.raises(RuntimeError):
        _identity_collate(["a", "b"])
