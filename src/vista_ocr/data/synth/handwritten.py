"""License-clean synthetic handwritten-line generator.

NOT a paper-equivalent reproduction of VISTA-OCR's synthetic IAM/RIMES
corpora — the paper uses an undisclosed font set + sentence corpus
shape. This module is a *license-clean approximation* designed to
expand the model's distribution coverage on handwriting-shaped pages
without scraping IAM itself. Per-writer variation is **not** modeled.

The class is parameterised on language so it can drive both English
(IAM-shaped) and French (RIMES-shaped, gated on tokenizer coverage,
see ``notes/plan_phase_j.md`` §0b) without paper-name leakage in the
API. Output ``Sample`` is drop-in compatible with
:class:`vista_ocr.data.mixture_stream.MixedStream` for stage 2/3.

Determinism contract
--------------------

A constructed :class:`HandwrittenLineSynth` with a given ``seed``
yields a deterministic sequence of *line metadata* (selected font id,
sampled text, mask-derived bboxes, line composition order). Image
pixels are deterministic up to PIL antialiasing; tests should compare
metadata exactly and image pixels via perceptual hash. Augmentation
must run with deterministic state if downstream tests want full
byte-equality, otherwise keep aug off in determinism tests.

Bbox derivation
---------------

Bboxes come from the rendered line's non-zero pixel mask, not from
``draw.textbbox()``. This is robust under stroke-thickness jitter,
dilation/erosion, and any geometric per-line transform we add later
without each transform having to know how to update a parallel bbox
estimate.
"""
from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.tokenizer import Line

LOG = logging.getLogger(__name__)

DEFAULT_FONT_RESOURCE = "vista_ocr.data.synth.fonts.handwritten"

# French charset the J0 tokenizer audit + J1a charset test enforce.
# Lowercase + uppercase accents + ligatures + curly apostrophe.
FRENCH_CHARSET = set("éèêëàâäçôöïîùüÿæœÉÈÊËÀÂÄÇÔÖÏÎÙÜŸÆŒ’")


@dataclass
class TextSource:
    """Local-file text source. NEVER downloads; HF dataset names live
    in ``scripts/datasets/setup_synth_corpus.sh``.

    :param path: One sentence (or paragraph) per line. UTF-8.
    :param mode: ``"sentence"`` samples one line per draw.
        ``"paragraph"`` joins consecutive lines until a blank.
    :param tag: Free-form label used in ``Sample.source`` /
        ``Sample.meta`` for diagnostics.
    """

    path: Path
    mode: str = "sentence"
    tag: str = "unknown"
    _lines: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"TextSource path does not exist: {self.path}. "
                f"Synthetic corpora must be staged locally; run "
                f"scripts/datasets/setup_synth_corpus.sh."
            )
        if self.mode not in ("sentence", "paragraph"):
            raise ValueError(f"TextSource mode must be sentence|paragraph, got {self.mode!r}")
        raw = self.path.read_text(encoding="utf-8").splitlines()
        if self.mode == "sentence":
            self._lines = [s.strip() for s in raw if s.strip()]
        else:
            paras: list[str] = []
            buf: list[str] = []
            for ln in raw:
                if ln.strip():
                    buf.append(ln.strip())
                elif buf:
                    paras.append(" ".join(buf))
                    buf = []
            if buf:
                paras.append(" ".join(buf))
            self._lines = paras
        if not self._lines:
            raise ValueError(f"TextSource produced no lines: {self.path}")

    def sample(self, rng: random.Random) -> str:
        return rng.choice(self._lines)


@dataclass
class HandwrittenLineSynthConfig:
    """:class:`HandwrittenLineSynth` configuration.

    :param language: ``"en"`` or ``"fr"``. Drives French-specific
        glyph-coverage validation in :meth:`HandwrittenLineSynth._validate_fonts`.
    :param font_paths: Explicit list of TTF paths, or ``None`` to use
        the packaged default font dir (resolved via
        ``importlib.resources``).
    :param canvas_size: ``(H, W)`` of the rendered page.
    :param min_lines / max_lines: Per-page line count range.
    :param min_line_height / max_line_height: Per-line height in px.
    :param min_chars_per_line / max_chars_per_line: Target text length
        per rendered line.
    :param line_spacing_min / line_spacing_max: Multiple of line
        height between consecutive line baselines.
    :param margin_left_min / margin_left_max: Random left margin in px.
    :param task: ``"ocr_layout"`` (default) or ``"ocr"``. The
        OCR-vs-layout ablation (notes/plan_phase_j.md §10b #4) flips
        this without changing rendering.
    """

    language: str = "en"
    font_paths: list[Path] | None = None
    canvas_size: tuple[int, int] = (1050, 1400)        # (H, W) -- matches PAGE_PRESET=large
    min_lines: int = 1
    max_lines: int = 12
    min_line_height: int = 32
    max_line_height: int = 72
    min_chars_per_line: int = 20
    max_chars_per_line: int = 60
    line_spacing_min: float = 1.0
    line_spacing_max: float = 1.6
    margin_left_min: int = 8
    margin_left_max: int = 32
    margin_top: int = 20
    task: str = "ocr_layout"

    def __post_init__(self) -> None:
        if self.language not in ("en", "fr"):
            raise ValueError(f"language must be 'en' or 'fr', got {self.language!r}")
        if self.task not in ("ocr", "ocr_layout"):
            raise ValueError(f"task must be 'ocr' or 'ocr_layout', got {self.task!r}")
        if self.min_lines < 1 or self.max_lines < self.min_lines:
            raise ValueError("invalid line-count range")
        if self.min_line_height < 8 or self.max_line_height < self.min_line_height:
            raise ValueError("invalid line-height range")


def _resolve_font_paths(cfg: HandwrittenLineSynthConfig) -> list[Path]:
    if cfg.font_paths is not None:
        paths = [Path(p) for p in cfg.font_paths]
    else:
        try:
            font_dir = resources.files(DEFAULT_FONT_RESOURCE)
        except (ModuleNotFoundError, FileNotFoundError) as e:
            raise FileNotFoundError(
                f"No fonts directory packaged at {DEFAULT_FONT_RESOURCE}. "
                f"Either bundle fonts or pass font_paths explicitly."
            ) from e
        paths = sorted(Path(str(p)) for p in font_dir.iterdir() if str(p).endswith(".ttf"))
    if not paths:
        raise FileNotFoundError("HandwrittenLineSynth: no .ttf files found")
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"font path does not exist: {p}")
    return paths


def _font_supports_chars(font: ImageFont.FreeTypeFont, chars: set[str]) -> set[str]:
    """Return the subset of ``chars`` the font has glyphs for.

    Uses the font's underlying ``getmask`` -- a missing glyph renders as
    a zero-area mask. Robust across Pillow versions.
    """
    have: set[str] = set()
    for c in chars:
        try:
            mask = font.getmask(c)
            if mask.size[0] > 0 and mask.size[1] > 0:
                have.add(c)
        except Exception:  # noqa: BLE001 -- per-char survey, robust on bad fonts
            continue
    return have


def _mask_bbox(arr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bbox of non-background pixels. Returns None if the mask is empty."""
    nz = np.argwhere(arr < 250)  # near-white = background
    if nz.size == 0:
        return None
    y0, x0 = nz.min(axis=0)
    y1, x1 = nz.max(axis=0) + 1
    return int(x0), int(y0), int(x1), int(y1)


class HandwrittenLineSynth:
    """Per-sample multi-line handwritten page generator.

    Iteration emits :class:`vista_ocr.data.types.Sample` values with
    ``task="ocr_layout"`` (or ``"ocr"`` per config), language-tagged
    ``source``, and ``meta`` populated with font + text-source ids
    for downstream per-source diagnostics
    (notes/plan_phase_j.md §10b #5/#6).
    """

    def __init__(
        self,
        text_sources: list[TextSource],
        cfg: HandwrittenLineSynthConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        if not text_sources:
            raise ValueError("HandwrittenLineSynth requires at least one TextSource")
        self.cfg = cfg or HandwrittenLineSynthConfig()
        self.text_sources = text_sources
        self.font_paths = _resolve_font_paths(self.cfg)
        self._rng = random.Random(seed)
        self._validate_fonts()

    def _validate_fonts(self) -> None:
        if self.cfg.language != "fr":
            return
        for fp in self.font_paths:
            font = ImageFont.truetype(str(fp), 40)
            missing = FRENCH_CHARSET - _font_supports_chars(font, FRENCH_CHARSET)
            if missing:
                raise ValueError(
                    f"Font {fp.name} is missing French glyphs: "
                    f"{''.join(sorted(missing))}. Drop the font or use --language=en."
                )

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

    def _truncate(self, text: str) -> str:
        target = self._rng.randint(self.cfg.min_chars_per_line, self.cfg.max_chars_per_line)
        if len(text) <= target:
            return text
        cut = text[:target]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        return cut

    def _render_line(self, text: str, font: ImageFont.FreeTypeFont, line_h: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
        """Render a single line in a tight raster and derive bbox from mask.

        The line image is the size needed to hold the text at the chosen
        font size; the caller composites it onto the page canvas.
        """
        # Generous initial canvas, cropped via mask bbox after render.
        ascent, descent = font.getmetrics()
        h = line_h + descent + 8
        # Width estimate from PIL; just for canvas allocation, not bbox.
        try:
            est_w = int(font.getlength(text)) + 16
        except AttributeError:
            est_w = font.getmask(text).size[0] + 16
        line_img = Image.new("L", (max(est_w, 8), h), 255)
        draw = ImageDraw.Draw(line_img)
        draw.text((4, 4), text, fill=0, font=font)
        arr = np.array(line_img)
        bb = _mask_bbox(arr)
        if bb is None:
            # Whitespace-only or unrenderable: emit a thin valid stub so
            # the caller can drop it, rather than raising.
            return line_img, (0, 0, 1, 1)
        return line_img, bb

    def __iter__(self) -> Iterator[Sample]:
        return self

    def __next__(self) -> Sample:
        return self.generate()

    def generate(self) -> Sample:
        cfg = self.cfg
        H, W = cfg.canvas_size
        page = Image.new("L", (W, H), 255)
        n_lines = self._rng.randint(cfg.min_lines, cfg.max_lines)
        line_h = self._rng.randint(cfg.min_line_height, cfg.max_line_height)
        font_path = self._rng.choice(self.font_paths)
        font = ImageFont.truetype(str(font_path), max(line_h - 4, 8))
        text_source = self._rng.choice(self.text_sources)
        spacing = self._rng.uniform(cfg.line_spacing_min, cfg.line_spacing_max)
        margin_l = self._rng.randint(cfg.margin_left_min, cfg.margin_left_max)

        lines: list[Line] = []
        y = cfg.margin_top
        used_text_sources: list[str] = []
        for _ in range(n_lines):
            advance = int(line_h * spacing)
            if y + line_h > H - cfg.margin_top:
                break
            text = self._truncate(text_source.sample(self._rng))
            if not text:
                y += advance
                continue
            line_img, (lx0, ly0, lx1, ly1) = self._render_line(text, font, line_h)
            crop = line_img.crop((lx0, ly0, lx1, ly1))
            # Place the cropped, mask-tight raster on the page.
            paste_x = margin_l
            paste_y = y
            if paste_x + crop.width > W or paste_y + crop.height > H:
                break
            page.paste(crop, (paste_x, paste_y))
            page_bbox = (paste_x, paste_y, paste_x + crop.width, paste_y + crop.height)
            lines.append(Line(text=text, bbox=page_bbox))
            used_text_sources.append(text_source.tag)
            y += advance

        if not lines:
            # All lines clipped or empty -- still emit a valid Sample
            # with one minimal line so MixedStream doesn't get a None.
            lines = [Line(text="", bbox=(0, 0, 1, 1))]
            used_text_sources = [text_source.tag]

        source = f"synth_handwritten:{cfg.language}:{text_source.tag}"
        meta = {
            "source_family": "synth_handwritten",
            "language": cfg.language,
            "font": font_path.name,
            "text_source": text_source.tag,
            "n_lines": len(lines),
            "line_height": line_h,
        }
        return Sample(
            image=page,
            lines=lines,
            task=cfg.task,                # type: ignore[arg-type]
            source=source,
            meta=meta,
        )
