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

from vista_ocr.data.bbox import BBox
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
    src = BBox.from_xyxy(*line.bbox)
    if (
        src.height < cfg.min_line_h
        or src.height > cfg.max_line_h
        or src.width < cfg.min_line_w
    ):
        return None
    img_w, img_h = image.size
    crop_box = src.pad(cfg.pad_px).clip_to(img_w=img_w, img_h=img_h)
    if crop_box.is_degenerate():
        return None
    crop = image.crop(crop_box.to_xyxy()).convert("L")
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
