"""Encoder tests: shape, gradients, dropout scheduler, parameter count."""
from __future__ import annotations

import pytest
import torch

from vista_ocr.models.encoder import (
    ConvBlock,
    DSCBlock,
    FCNEncoderWidther,
    Factorized2DPositionalEmbedding,
    _MixDropout,
)


@pytest.fixture(scope="module")
def encoder() -> FCNEncoderWidther:
    torch.manual_seed(0)
    return FCNEncoderWidther(input_channels=1, dropout=0.0)


def test_convblock_preserves_shape_with_stride_1():
    cb = ConvBlock(8, 16, stride=(1, 1), dropout=0.0)
    x = torch.randn(1, 8, 32, 32)
    y = cb(x)
    assert y.shape == (1, 16, 32, 32)


def test_convblock_downsamples_with_stride_2():
    cb = ConvBlock(8, 16, stride=(2, 2), dropout=0.0)
    x = torch.randn(1, 8, 32, 32)
    y = cb(x)
    assert y.shape == (1, 16, 16, 16)


def test_dscblock_residual_when_shapes_match():
    dsc = DSCBlock(16, 16, stride=(1, 1), dropout=0.0)
    x = torch.randn(1, 16, 8, 8)
    y = dsc(x)
    assert y.shape == x.shape


def test_factorized_pe_adds_position_information():
    pe = Factorized2DPositionalEmbedding(d_model=4, h_max=10, w_max=10)
    x = torch.zeros(1, 4, 4, 4)
    y = pe(x)
    # Different spatial positions must yield different PE contributions.
    assert not torch.allclose(y[0, :, 0, 0], y[0, :, 1, 0])
    assert not torch.allclose(y[0, :, 0, 0], y[0, :, 0, 1])


def test_factorized_pe_rejects_oversize():
    pe = Factorized2DPositionalEmbedding(d_model=4, h_max=2, w_max=2)
    with pytest.raises(ValueError):
        pe(torch.zeros(1, 4, 8, 8))


def test_encoder_output_shape_64x64(encoder: FCNEncoderWidther):
    x = torch.randn(2, 1, 64, 64)
    y = encoder(x)
    # Stride (32, 8): H=64 -> 2, W=64 -> 8 → 2 * 8 = 16 spatial tokens.
    assert y.shape == (2, 2 * 8, 1024)


def test_encoder_output_shape_rectangular(encoder: FCNEncoderWidther):
    x = torch.randn(1, 1, 96, 128)
    y = encoder(x)
    assert y.shape == (1, 3 * 16, 1024)


def test_encoder_backward(encoder: FCNEncoderWidther):
    x = torch.randn(1, 1, 64, 64, requires_grad=True)
    y = encoder(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_param_count_in_expected_range(encoder: FCNEncoderWidther):
    n = sum(p.numel() for p in encoder.parameters())
    # Measured ~21.5M with paper-default PE (h_max=500, w_max=1000):
    # ConvBlocks ~14.9M + DSCBlocks ~5.1M + PE ~1.5M.
    assert 18e6 < n < 25e6, n


def test_set_dropout_propagates(encoder: FCNEncoderWidther):
    encoder.set_dropout(0.25)
    for m in encoder.modules():
        if isinstance(m, _MixDropout):
            assert m.dropout.p == 0.25
            assert m.dropout2d.p == 0.125
    encoder.set_dropout(0.0)
