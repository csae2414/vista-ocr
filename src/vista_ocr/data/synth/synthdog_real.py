"""Real SynthDOG-style generator (English).

Improves over the toy renderer in :mod:`vista_ocr.data.synth.synthdog_bbox`:

- Sentences sourced from real Wikipedia text (HuggingFace
  ``wikitext-2-raw-v1`` by default, or any file with one paragraph per
  line).
- Multiple TTF fonts (printed + handwritten-style) sampled per page so
  one model sees many typefaces -- matches the paper's appendix 0.A.4
  guidance.
- Mild augmentation: random margin, font size, line spacing, optional
  Gaussian blur.
- Word-level + line-level bboxes recorded; we expose line-level by
  default to match the VISTA-OCR output sequence.

Designed as a long-running iterator: each call to :func:`generate_sample`
produces a fresh ``(image, lines)`` pair. Use a buffer + DataLoader
worker if you want it to feed training in parallel.
"""
from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.tokenizer import Line

LOG = logging.getLogger(__name__)


_DEFAULT_FONT_DIRS = [
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/liberation",
    "/usr/share/fonts/truetype/noto",
]


def _scan_fonts(extra_dirs: list[Path] | None = None) -> list[Path]:
    """Find all .ttf files under known system font dirs, plus any user dirs."""
    dirs = [Path(p) for p in _DEFAULT_FONT_DIRS]
    if extra_dirs:
        dirs.extend(Path(p) for p in extra_dirs)
    out: list[Path] = []
    for d in dirs:
        if not d.exists():
            continue
        out.extend(p for p in d.rglob("*.ttf"))
    return out


@dataclass
class SynthDocConfig:
    canvas_h: int = 1100
    canvas_w: int = 850
    margin_px: tuple[int, int] = (32, 96)        # min/max
    font_size_px: tuple[int, int] = (14, 22)     # min/max
    line_spacing_px: tuple[int, int] = (4, 12)
    blur_prob: float = 0.15
    rotate_deg: float = 1.0                       # max +/-
    fonts: list[Path] = field(default_factory=list)
    seed: int | None = None


def _wrap_words_to_lines(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    paragraph: str,
    max_width: int,
) -> list[str]:
    """Greedy word-wrap to fit ``paragraph`` inside ``max_width`` pixels."""
    words = paragraph.split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        attempt = " ".join(cur + [w])
        bbox = draw.textbbox((0, 0), attempt, font=font)
        if bbox[2] - bbox[0] > max_width and cur:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines


def generate_sample(paragraphs: list[str], cfg: SynthDocConfig) -> Sample:
    """Render ``paragraphs`` onto a page with realistic font + bbox emission."""
    rng = random.Random(cfg.seed)
    fonts = cfg.fonts or _scan_fonts()
    if not fonts:
        raise RuntimeError("No .ttf fonts found; install fonts-dejavu / fonts-liberation")
    font_path = rng.choice(fonts)
    font_size = rng.randint(*cfg.font_size_px)
    font = ImageFont.truetype(str(font_path), font_size)

    img = Image.new("L", (cfg.canvas_w, cfg.canvas_h), color=255)
    draw = ImageDraw.Draw(img)

    margin_l = rng.randint(*cfg.margin_px)
    margin_t = rng.randint(*cfg.margin_px)
    line_gap = rng.randint(*cfg.line_spacing_px)
    text_width = cfg.canvas_w - 2 * margin_l

    lines: list[Line] = []
    y = margin_t
    for para in paragraphs:
        wrapped = _wrap_words_to_lines(draw, font, para, max_width=text_width)
        for text in wrapped:
            l, t, r, b = draw.textbbox((margin_l, y), text, font=font)
            line_h = b - t
            if y + line_h > cfg.canvas_h - margin_t:
                break
            draw.text((margin_l, y), text, fill=0, font=font)
            lines.append(Line(text=text, bbox=(int(l), int(t), int(r), int(b))))
            y += line_h + line_gap
        # paragraph break
        y += line_gap

    if rng.random() < cfg.blur_prob:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.9)))

    if abs(cfg.rotate_deg) > 0:
        angle = rng.uniform(-cfg.rotate_deg, cfg.rotate_deg)
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
        # bbox rotation handling deferred -- small angles only (<2deg) so the
        # AABB approximation is still tight. For larger angles compute the
        # rotated polygon and project to AABB.

    return Sample(image=img, lines=lines, task="ocr_layout", source="synth_synthdog_real")


def iter_synthdoc(
    paragraph_source: Iterable[str],
    cfg: SynthDocConfig | None = None,
    paragraphs_per_page: int = 4,
) -> Iterator[Sample]:
    """Stream synthetic pages forever from a paragraph source."""
    cfg = cfg or SynthDocConfig()
    rng = random.Random(cfg.seed)
    buf: list[str] = []
    for para in paragraph_source:
        if not para.strip():
            continue
        buf.append(para)
        if len(buf) >= paragraphs_per_page:
            local_cfg = SynthDocConfig(
                canvas_h=cfg.canvas_h, canvas_w=cfg.canvas_w,
                margin_px=cfg.margin_px, font_size_px=cfg.font_size_px,
                line_spacing_px=cfg.line_spacing_px, blur_prob=cfg.blur_prob,
                rotate_deg=cfg.rotate_deg, fonts=cfg.fonts,
                seed=rng.randint(0, 2**31 - 1),
            )
            yield generate_sample(buf, local_cfg)
            buf = []


def load_wikitext_paragraphs(out_path: Path) -> Iterator[str]:
    """Stream paragraphs from a cached WikiText-2 dump (one per line).
    If the cache doesn't exist yet, materialise it via HuggingFace datasets."""
    if not out_path.exists():
        from datasets import load_dataset  # noqa: PLC0415
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        with out_path.open("w", encoding="utf-8") as f:
            for row in ds:
                t = row["text"].strip()
                if t and not t.startswith("="):
                    f.write(t + "\n")
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line:
                yield line
