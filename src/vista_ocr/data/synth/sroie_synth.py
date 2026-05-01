"""Faker-based synthetic SROIE-style receipt generator.

SROIE's real receipts are short, store-name + items + totals layouts. We
emit a reasonable approximation: header (store name + address), a list of
items and prices, and a total. Paper appendix 0.A.4 mentions augmentation
(background markup, slanted text, shadow, poor resolution) — a minimal
slant + Gaussian-blur knob is exposed here, with everything else best
done at the augmentation layer (see TODO at the bottom).
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from vista_ocr.data.synth.synthdog_bbox import _choose_font
from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.tokenizer import Line

LOG = logging.getLogger(__name__)


@dataclass
class SroieSynthConfig:
    canvas_h: int = 1024
    canvas_w: int = 384
    margin: int = 24
    line_height: int = 22
    font_size: int = 16
    item_count_range: tuple[int, int] = (3, 8)
    blur_prob: float = 0.3
    seed: int | None = None


def _faker():
    from faker import Faker  # noqa: PLC0415

    return Faker("en_US")


def _build_lines(rng: random.Random) -> list[str]:
    fk = _faker()
    fk.seed_instance(rng.randint(0, 2**31))
    lines = [fk.company().upper(), fk.street_address(), f"{fk.city()}, {fk.zipcode()}"]
    n_items = rng.randint(3, 8)
    subtotal = 0.0
    for _ in range(n_items):
        name = fk.word().capitalize()
        price = round(rng.uniform(0.99, 49.99), 2)
        lines.append(f"{name:<14}${price:>6.2f}")
        subtotal += price
    tax = round(subtotal * 0.07, 2)
    total = round(subtotal + tax, 2)
    lines.append(f"SUBTOTAL      ${subtotal:>6.2f}")
    lines.append(f"TAX 7%        ${tax:>6.2f}")
    lines.append(f"TOTAL         ${total:>6.2f}")
    lines.append(f"Receipt #: {fk.bothify('????-#####').upper()}")
    return lines


def generate_sample(cfg: SroieSynthConfig | None = None) -> Sample:
    cfg = cfg or SroieSynthConfig()
    rng = random.Random(cfg.seed)
    text_lines = _build_lines(rng)

    img = Image.new("L", (cfg.canvas_w, cfg.canvas_h), 255)
    draw = ImageDraw.Draw(img)
    font = _choose_font(_FontProxy(cfg.font_size))  # see helper below

    lines: list[Line] = []
    y = cfg.margin
    for text in text_lines:
        if y + cfg.line_height > cfg.canvas_h - cfg.margin:
            break
        x = cfg.margin
        l, t, r, b = draw.textbbox((x, y), text, font=font)
        draw.text((x, y), text, fill=0, font=font)
        lines.append(Line(text=text, bbox=(int(l), int(t), int(r), int(b))))
        y += cfg.line_height

    if rng.random() < cfg.blur_prob:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.7))

    return Sample(image=img, lines=lines, task="ocr_layout", source="synth_sroie")


class _FontProxy:
    """Tiny shim so we can reuse :func:`synthdog_bbox._choose_font`'s
    contract (it takes a config object with ``font_size`` + ``font_path``)."""

    def __init__(self, font_size: int) -> None:
        self.font_size = font_size
        self.font_path = None
