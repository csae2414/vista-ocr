"""Spatial token grid for VISTA-OCR.

The paper uses two disjoint sets of positional tokens for the X and Y axes
("Original" encoding scheme, Table 7). Tokens form a grid by quantizing
pixel coordinates on a fixed canvas (page size).

Defaults in `configs/base.yaml` give a 248×351 grid (≈600 tokens) at 10 px
quantization on a 2480×3508 canvas — close to the paper's "10 pixel
quantizer" setting.

Three encoding schemes from Table 7 (`scheme=`):
  - "original":  ``<x_i><y_j> w1 w2 ... wm <x_k><y_l>``  (interleaved)
  - "segmented": ``w1 w2 ... wm </text> <x_i><y_j><x_k><y_l> </location>``
  - "unified":   one shared XY token set instead of separate X and Y sets

Implementation lives here so it has zero dependencies; tokenizer.py wraps
this together with the SentencePiece subword model.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpatialGrid:
    canvas_h: int
    canvas_w: int
    quantizer_px: int
    scheme: str = "original"

    @property
    def n_x(self) -> int:
        return (self.canvas_w + self.quantizer_px - 1) // self.quantizer_px

    @property
    def n_y(self) -> int:
        return (self.canvas_h + self.quantizer_px - 1) // self.quantizer_px

    def x_token(self, px: int) -> str:
        idx = min(self.n_x - 1, max(0, px // self.quantizer_px))
        return f"<x_{idx}>"

    def y_token(self, px: int) -> str:
        idx = min(self.n_y - 1, max(0, px // self.quantizer_px))
        return f"<y_{idx}>"

    def all_tokens(self) -> list[str]:
        if self.scheme == "unified":
            n = max(self.n_x, self.n_y)
            return [f"<xy_{i}>" for i in range(n)]
        return [f"<x_{i}>" for i in range(self.n_x)] + [f"<y_{j}>" for j in range(self.n_y)]
