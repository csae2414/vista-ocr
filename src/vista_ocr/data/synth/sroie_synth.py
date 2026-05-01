"""Faker-based synthetic SROIE-style receipts with paper-appendix augmentations.

Paper Appendix 0.A.4 lists four augmentation families for the synthetic
SROIE pipeline:

1. **background markup** - light noise / lines / smudges behind the text
2. **slanted text**      - small rotation, +/- a few degrees
3. **shadow effects**    - one-sided radial gradient overlay
4. **poor resolution**   - downsample then upsample, or JPEG compression

Each receipt is a layout of header lines + items + subtotal/total, rendered
with a real TTF font at variable size. Augmentations are sampled
independently per receipt with configurable probabilities.
"""
from __future__ import annotations

import io
import logging
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from vista_ocr.data.synth.synthdog_real import _scan_fonts
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
    # Augmentation probabilities (paper Appendix 0.A.4)
    blur_prob: float = 0.30
    background_markup_prob: float = 0.30
    slant_prob: float = 0.30
    shadow_prob: float = 0.25
    poor_resolution_prob: float = 0.25
    slant_max_deg: float = 3.0
    seed: int | None = None


def _faker():
    from faker import Faker  # noqa: PLC0415

    return Faker("en_US")


def _build_text_lines(rng: random.Random) -> list[str]:
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


def _add_background_markup(img: Image.Image, rng: random.Random) -> Image.Image:
    """Paper-appendix style: random thin lines + speckle noise behind the
    text. Subtle so the OCR target is still legible."""
    w, h = img.size
    arr = np.asarray(img, dtype=np.int16)
    # Salt-and-pepper noise on ~1% of pixels.
    n_noise = (w * h) // 100
    ys = np.random.randint(0, h, n_noise)
    xs = np.random.randint(0, w, n_noise)
    arr[ys, xs] = np.where(np.random.rand(n_noise) < 0.5, 0, 255)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")

    draw = ImageDraw.Draw(img)
    n_lines = rng.randint(1, 3)
    for _ in range(n_lines):
        x1, y1 = rng.randint(0, w), rng.randint(0, h)
        x2, y2 = rng.randint(0, w), rng.randint(0, h)
        gray = rng.randint(180, 230)
        draw.line((x1, y1, x2, y2), fill=gray, width=1)
    return img


def _add_shadow(img: Image.Image, rng: random.Random) -> Image.Image:
    """One-sided radial-ish gradient that darkens one edge of the page."""
    w, h = img.size
    side = rng.choice(("left", "right", "top", "bottom"))
    yy, xx = np.indices((h, w), dtype=np.float32)
    if side == "left":
        d = xx / w
    elif side == "right":
        d = 1 - xx / w
    elif side == "top":
        d = yy / h
    else:
        d = 1 - yy / h
    strength = rng.uniform(0.3, 0.55)
    factor = 1.0 - (1.0 - d) * strength
    arr = np.asarray(img, dtype=np.float32) * factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")


def _add_poor_resolution(img: Image.Image, rng: random.Random) -> Image.Image:
    """Downsample by a factor then upsample (and a shot of JPEG)."""
    if rng.random() < 0.5:
        w, h = img.size
        f = rng.uniform(2.0, 3.5)
        small = img.resize((int(w / f), int(h / f)), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)
    # JPEG compression artefact
    buf = io.BytesIO()
    img.convert("L").save(buf, format="JPEG", quality=rng.randint(20, 50))
    buf.seek(0)
    return Image.open(buf).convert("L")


def _slant(img: Image.Image, lines: list[Line], rng: random.Random,
           max_deg: float) -> tuple[Image.Image, list[Line]]:
    """Small rotation. For small angles (<5 degrees) the AABB approximation
    of each line bbox is tight enough; we rotate corner points and project
    back to AABB."""
    angle = rng.uniform(-max_deg, max_deg)
    img2 = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)

    cx, cy = img.size[0] / 2.0, img.size[1] / 2.0
    cos_a = np.cos(np.deg2rad(-angle))
    sin_a = np.sin(np.deg2rad(-angle))
    new_lines = []
    for ln in lines:
        x1, y1, x2, y2 = ln.bbox
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        rotated = []
        for cx0, cy0 in corners:
            dx, dy = cx0 - cx, cy0 - cy
            rx = cx + dx * cos_a - dy * sin_a
            ry = cy + dx * sin_a + dy * cos_a
            rotated.append((rx, ry))
        rxs = [p[0] for p in rotated]
        rys = [p[1] for p in rotated]
        new_lines.append(
            Line(text=ln.text, bbox=(int(min(rxs)), int(min(rys)),
                                     int(max(rxs)), int(max(rys))))
        )
    return img2, new_lines


def generate_sample(cfg: SroieSynthConfig | None = None) -> Sample:
    cfg = cfg or SroieSynthConfig()
    rng = random.Random(cfg.seed)
    text_lines = _build_text_lines(rng)

    fonts = _scan_fonts()
    if fonts:
        font_path = rng.choice(fonts)
        font = ImageFont.truetype(str(font_path), cfg.font_size)
    else:
        font = ImageFont.load_default()

    img = Image.new("L", (cfg.canvas_w, cfg.canvas_h), 255)
    draw = ImageDraw.Draw(img)

    lines: list[Line] = []
    y = cfg.margin
    for text in text_lines:
        if y + cfg.line_height > cfg.canvas_h - cfg.margin:
            break
        x = cfg.margin
        bx1, by1, bx2, by2 = draw.textbbox((x, y), text, font=font)
        draw.text((x, y), text, fill=0, font=font)
        lines.append(Line(text=text, bbox=(int(bx1), int(by1), int(bx2), int(by2))))
        y += cfg.line_height

    if rng.random() < cfg.background_markup_prob:
        img = _add_background_markup(img, rng)
    if rng.random() < cfg.slant_prob:
        img, lines = _slant(img, lines, rng, cfg.slant_max_deg)
    if rng.random() < cfg.shadow_prob:
        img = _add_shadow(img, rng)
    if rng.random() < cfg.poor_resolution_prob:
        img = _add_poor_resolution(img, rng)
    if rng.random() < cfg.blur_prob:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.0)))

    return Sample(image=img, lines=lines, task="ocr_layout", source="synth_sroie")
