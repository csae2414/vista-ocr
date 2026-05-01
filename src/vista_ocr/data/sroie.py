"""SROIE 2019 (ICDAR 2019 Robust Reading Challenge, Task 1+2+3) loader.

Paper Table 2: VISTA-OCR reaches F1 = 93.95 on SROIE. The dataset ships
as a flat directory of ``.jpg`` receipts plus a ``.txt`` file per receipt
in the format::

    x1,y1,x2,y2,x3,y3,x4,y4,transcription

(Eight coordinates: a quadrilateral in clockwise order.)  We project to
an axis-aligned bbox by taking ``(min_x, min_y, max_x, max_y)`` -- this
matches the paper's line-level interface.

Expected layout::

    sroie_root/
      train/
        X51005230625.jpg
        X51005230625.txt
        ...
      test/
        ...
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.tokenizer import Line

LOG = logging.getLogger(__name__)


@dataclass
class SroieConfig:
    root: Path
    split: str = "train"             # "train" or "test"
    drop_zero_lines: bool = True


def _parse_quad_line(s: str) -> Line | None:
    parts = s.rstrip("\n").split(",", 8)
    if len(parts) < 9:
        return None
    try:
        coords = [int(float(p)) for p in parts[:8]]
    except ValueError:
        return None
    text = parts[8].strip()
    if not text:
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    return Line(text=text, bbox=(min(xs), min(ys), max(xs), max(ys)))


def iter_sroie(cfg: SroieConfig) -> Iterator[Sample]:
    split_dir = cfg.root / cfg.split
    if not split_dir.exists():
        raise FileNotFoundError(split_dir)
    for txt_path in sorted(split_dir.glob("*.txt")):
        img_path = txt_path.with_suffix(".jpg")
        if not img_path.exists():
            continue
        lines: list[Line] = []
        for raw in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            ln = _parse_quad_line(raw)
            if ln is not None:
                lines.append(ln)
        if cfg.drop_zero_lines and not lines:
            continue
        img = Image.open(img_path).convert("L")
        yield Sample(image=img, lines=lines, task="ocr_layout",
                     source=f"sroie:{cfg.split}:{txt_path.stem}")
