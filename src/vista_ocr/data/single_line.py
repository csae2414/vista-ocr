"""Single-line curriculum data source (C2).

Generates :class:`vista_ocr.data.types.Sample` objects where each
sample is a tightly cropped *single line* taken from a real PDFA page.
Used as an optional stage-0 to get the text head out of the early-
training n-gram-collapse regime before exposing it to full-page
distractor noise.

Per the design notes the cropped sample still emits trivial spatial
tokens (``<x_0><y_0> ... <x_W><y_H>``) so the decoder continues to
learn the spatial vocabulary. Without this the spatial token vocab
would atrophy in stage-0 and have to be re-learned in stage-1.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

from PIL import Image

from vista_ocr.data.pdfa import PdfaConfig, iter_pdfa
from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.tokenizer import Line

LOG = logging.getLogger(__name__)


@dataclass
class SingleLineConfig:
    """Crop knobs for the curriculum source."""

    pad_px: int = 4
    min_line_h: int = 12
    max_line_h: int = 96
    min_line_w: int = 32


def crop_line(image: Image.Image, line: Line, cfg: SingleLineConfig) -> Sample | None:
    """Return a Sample carrying just one line, or None if it's degenerate."""
    x1, y1, x2, y2 = line.bbox
    h = y2 - y1
    w = x2 - x1
    if h < cfg.min_line_h or h > cfg.max_line_h or w < cfg.min_line_w:
        return None
    pad = cfg.pad_px
    img_w, img_h = image.size
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(img_w, x2 + pad)
    cy2 = min(img_h, y2 + pad)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = image.crop((cx1, cy1, cx2, cy2)).convert("L")
    cw, ch = crop.size
    # Trivial bbox = the entire crop. Decoder still emits <x><y>...<x><y>.
    new_line = Line(text=line.text, bbox=(0, 0, cw, ch))
    return Sample(
        image=crop, lines=[new_line], task="ocr_layout", source="single_line:pdfa",
    )


def iter_single_line_samples(
    pdfa_cfg: PdfaConfig,
    cfg: SingleLineConfig | None = None,
) -> Iterator[Sample]:
    """Stream single-line crops out of a PDFA shard set."""
    cfg = cfg or SingleLineConfig()
    for page in iter_pdfa(pdfa_cfg):
        for line in page.lines:
            sample = crop_line(page.image, line, cfg)
            if sample is not None:
                yield sample
