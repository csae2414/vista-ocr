"""DANIEL-style FCN encoder with factorized 2-D positional embedding.

Architecture transcribed from
``Shulk97/daniel/basic/encoders.py::FCN_Encoder_Widther`` (Constum, Tranouez,
Paquet — *DANIEL*, IJDAR 2025; ref [14] in the VISTA-OCR paper). This is the
exact encoder VISTA-OCR is "inspired by".

Pipeline
--------
1. ``init_blocks`` — six :class:`ConvBlock` stages with strides
   ``(1,1)(2,2)(2,2)(2,2)(2,1)(2,1)`` widening 1→32→64→128→256→512→512
   channels. Total stride after this stage: ``(32, 8)``.
2. ``blocks`` — four :class:`DSCBlock` (depthwise-separable + residual)
   refining at the same spatial scale and widening to ``1024`` channels.
3. :class:`Factorized2DPositionalEmbedding` — separate learned ``H`` and
   ``W`` embeddings (``h_max=500``, ``w_max=1000`` by default), summed and
   added to the feature map. Cheap and resolution-flexible.
4. The result is flattened to ``(B, H'·W', 1024)`` to feed cross-attention
   in the decoder.
"""
from __future__ import annotations

import logging
import random

import torch
from torch import Tensor, nn

LOG = logging.getLogger(__name__)


class _MixDropout(nn.Module):
    """Match DANIEL's MixDropout: randomly pick 1-D dropout or 2-D dropout
    each call. Keeps the same interface as :class:`torch.nn.Dropout`."""

    def __init__(self, p: float = 0.4, p2d: float | None = None) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.dropout2d = nn.Dropout2d(p2d if p2d is not None else p / 2)

    def forward(self, x: Tensor) -> Tensor:
        if random.random() < 0.5:
            return self.dropout(x)
        return self.dropout2d(x)


class _DepthSepConv2D(nn.Module):
    """Depthwise-separable 3×3 convolution as in DANIEL ``layers.py``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.depth = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            groups=in_channels,
        )
        self.point = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.point(self.depth(x))


class ConvBlock(nn.Module):
    """Three 3×3 convolutions, InstanceNorm, ReLU, MixDropout inserted at
    one of three positions per forward (matches DANIEL's stochastic
    placement). The third conv carries the configurable ``stride``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int] = (1, 1),
        kernel_size: int = 3,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=pad)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.norm = nn.InstanceNorm2d(out_channels, eps=1e-3, momentum=0.99, track_running_stats=False)
        self.act = nn.ReLU(inplace=True)
        self.dropout = _MixDropout(p=dropout, p2d=dropout / 2)

    def forward(self, x: Tensor) -> Tensor:
        pos = random.randint(1, 3)
        x = self.act(self.conv1(x))
        if pos == 1:
            x = self.dropout(x)
        x = self.act(self.conv2(x))
        if pos == 2:
            x = self.dropout(x)
        x = self.norm(x)
        x = self.act(self.conv3(x))
        if pos == 3:
            x = self.dropout(x)
        return x


class DSCBlock(nn.Module):
    """Depthwise-separable variant of :class:`ConvBlock` with residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int] = (1, 1),
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.conv1 = _DepthSepConv2D(in_channels, out_channels)
        self.conv2 = _DepthSepConv2D(out_channels, out_channels)
        self.conv3 = _DepthSepConv2D(out_channels, out_channels, stride=stride)
        self.norm = nn.InstanceNorm2d(out_channels, eps=1e-3, momentum=0.99, track_running_stats=False)
        self.act = nn.ReLU(inplace=True)
        self.dropout = _MixDropout(p=dropout, p2d=dropout / 2)

    def forward(self, x_in: Tensor) -> Tensor:
        pos = random.randint(1, 3)
        x = self.act(self.conv1(x_in))
        if pos == 1:
            x = self.dropout(x)
        x = self.act(self.conv2(x))
        if pos == 2:
            x = self.dropout(x)
        x = self.norm(x)
        x = self.conv3(x)
        if pos == 3:
            x = self.dropout(x)
        return x + x_in if x.shape == x_in.shape else x


class Factorized2DPositionalEmbedding(nn.Module):
    """Learned positional embedding factorized into separate ``H`` and ``W``
    tables. For a feature map of shape ``(B, C, H, W)`` we add
    ``pe_h[h] + pe_w[w]`` at every spatial location.

    This is the cheapest learned 2-D variant and matches DANIEL's defaults
    (``h_max=500``, ``w_max=1000``)."""

    def __init__(self, d_model: int, h_max: int = 500, w_max: int = 1000) -> None:
        super().__init__()
        self.pe_h = nn.Embedding(h_max, d_model)
        self.pe_w = nn.Embedding(w_max, d_model)
        self.h_max = h_max
        self.w_max = w_max
        nn.init.normal_(self.pe_h.weight, std=0.02)
        nn.init.normal_(self.pe_w.weight, std=0.02)

    def forward(self, feat: Tensor) -> Tensor:
        b, c, h, w = feat.shape
        if h > self.h_max or w > self.w_max:
            raise ValueError(
                f"Feature map {h}x{w} exceeds positional table {self.h_max}x{self.w_max}"
            )
        pe_h = self.pe_h(torch.arange(h, device=feat.device))   # (H, C)
        pe_w = self.pe_w(torch.arange(w, device=feat.device))   # (W, C)
        pe = pe_h[:, None, :] + pe_w[None, :, :]                # (H, W, C)
        pe = pe.permute(2, 0, 1).unsqueeze(0)                    # (1, C, H, W)
        return feat + pe


class FCNEncoderWidther(nn.Module):
    """Faithful re-implementation of DANIEL's ``FCN_Encoder_Widther``.

    Output shape is ``(B, 1024, H/32, W/8)``.
    """

    OUT_CHANNELS: int = 1024
    STRIDE_H: int = 32
    STRIDE_W: int = 8

    def __init__(
        self,
        input_channels: int = 1,
        dropout: float = 0.5,
        pe_h_max: int = 500,
        pe_w_max: int = 1000,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.dropout_p = dropout

        self.init_blocks = nn.Sequential(
            ConvBlock(input_channels, 32, stride=(1, 1), dropout=dropout),
            ConvBlock(32, 64, stride=(2, 2), dropout=dropout),
            ConvBlock(64, 128, stride=(2, 2), dropout=dropout),
            ConvBlock(128, 256, stride=(2, 2), dropout=dropout),
            ConvBlock(256, 512, stride=(2, 1), dropout=dropout),
            ConvBlock(512, 512, stride=(2, 1), dropout=dropout),
        )
        self.blocks = nn.Sequential(
            DSCBlock(512, 512, stride=(1, 1), dropout=dropout),
            DSCBlock(512, 512, stride=(1, 1), dropout=dropout),
            DSCBlock(512, 512, stride=(1, 1), dropout=dropout),
            DSCBlock(512, 1024, stride=(1, 1), dropout=dropout),
        )
        self.pe = Factorized2DPositionalEmbedding(self.OUT_CHANNELS, pe_h_max, pe_w_max)

        n_params = sum(p.numel() for p in self.parameters())
        LOG.info("FCNEncoderWidther initialised: %.2fM params", n_params / 1e6)

    def forward(self, x: Tensor) -> Tensor:
        """Returns ``(B, H/32 * W/8, 1024)`` flattened-token features ready
        for cross-attention."""
        feat = self.blocks(self.init_blocks(x))    # (B, 1024, H/32, W/8)
        feat = self.pe(feat)
        b, c, h, w = feat.shape
        return feat.flatten(2).transpose(1, 2)     # (B, H'*W', C)

    def set_dropout(self, p: float) -> None:
        """Update dropout for all internal :class:`_MixDropout` modules.
        Used by the exponential dropout scheduler from DANIEL."""
        for m in self.modules():
            if isinstance(m, _MixDropout):
                m.dropout.p = p
                m.dropout2d.p = p / 2


__all__ = [
    "ConvBlock",
    "DSCBlock",
    "Factorized2DPositionalEmbedding",
    "FCNEncoderWidther",
]
