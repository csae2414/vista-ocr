"""Minimal SynthDOG-bbox generator.

Renders text on a (mostly white) page with line-level bounding boxes that
match what the encoder + tokenizer expect downstream.

The original SynthDOG (Donut, NAVER 2022) and its bbox-emitting fork
(``Veason-silverbullet/ViTLP``) are far more sophisticated — they support
real document templates, paragraph layouting, photo backgrounds, and
multilingual rendering. This module covers what VISTA-OCR's *output* needs
(line transcription + line bbox) without dragging in those dependencies.
A more faithful renderer can be plugged in later behind the same
:func:`generate_sample` interface.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.tokenizer import Line

LOG = logging.getLogger(__name__)


@dataclass
class SynthDogConfig:
    canvas_h: int = 1024
    canvas_w: int = 768
    margin: int = 32
    line_height: int = 28
    font_size: int = 18
    font_path: Path | None = None         # None → PIL default font
    seed: int | None = None


def _choose_font(cfg: SynthDogConfig) -> ImageFont.ImageFont:
    if cfg.font_path is not None:
        try:
            return ImageFont.truetype(str(cfg.font_path), cfg.font_size)
        except OSError:
            LOG.warning("Could not load %s; falling back to default", cfg.font_path)
    return ImageFont.load_default()


def generate_sample(
    text_lines: list[str],
    cfg: SynthDogConfig | None = None,
) -> Sample:
    """Render ``text_lines`` onto a blank canvas, one line per row, and
    return a :class:`Sample` with line-level bboxes."""
    cfg = cfg or SynthDogConfig()
    rng = random.Random(cfg.seed)

    img = Image.new("L", (cfg.canvas_w, cfg.canvas_h), color=255)
    draw = ImageDraw.Draw(img)
    font = _choose_font(cfg)

    lines: list[Line] = []
    y = cfg.margin
    for text in text_lines:
        if y + cfg.line_height > cfg.canvas_h - cfg.margin:
            break
        x = cfg.margin + rng.randint(0, 8)
        # Pillow's textbbox returns (l, t, r, b) anchored at the draw point.
        l, t, r, b = draw.textbbox((x, y), text, font=font)
        draw.text((x, y), text, fill=0, font=font)
        lines.append(Line(text=text, bbox=(int(l), int(t), int(r), int(b))))
        y += cfg.line_height

    return Sample(image=img, lines=lines, task="ocr_layout", source="synth_synthdog")
