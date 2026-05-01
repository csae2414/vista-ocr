"""MAURDOR loader (English subset).

Paper Table 4: VISTA-OCR reaches Area-F1=87.02 on MAURDOR. The dataset
is multilingual; per appendix 0.A.3 the authors keep documents tagged as
``en`` / ``fr`` / ``en+fr``. We're EN-only so we keep ``en`` and the
English half of ``en+fr``.

MAURDOR is licence-restricted (LIMSI/CNRS); users obtain it directly.
The expected on-disk layout from ``prepare_maurdor.py``::

    maurdor_root/
      images/
        doc_001.tif
      pages/
        doc_001.json     # { "lang": "en", "lines": [{"text", "bbox": [x,y,x,y]}] }
      splits/
        train.txt
        val.txt
        test.txt
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.tokenizer import Line

LOG = logging.getLogger(__name__)


@dataclass
class MaurdorConfig:
    root: Path
    split: str = "train"
    languages: tuple[str, ...] = ("en", "en+fr")


def _read_ids(cfg: MaurdorConfig) -> list[str]:
    p = cfg.root / "splits" / f"{cfg.split}.txt"
    if not p.exists():
        raise FileNotFoundError(p)
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def _decode_page(payload: dict) -> list[Line]:
    out: list[Line] = []
    for entry in payload.get("lines", []):
        text = entry.get("text", "").strip()
        bbox = entry.get("bbox") or entry.get("box")
        if not text or not bbox or len(bbox) != 4:
            continue
        out.append(Line(text=text, bbox=tuple(int(v) for v in bbox)))
    return out


def iter_maurdor(cfg: MaurdorConfig) -> Iterator[Sample]:
    image_dir = cfg.root / "images"
    pages_dir = cfg.root / "pages"
    for doc_id in _read_ids(cfg):
        json_path = pages_dir / f"{doc_id}.json"
        if not json_path.exists():
            LOG.warning("MAURDOR page missing: %s", doc_id)
            continue
        payload = json.loads(json_path.read_text())
        if payload.get("lang") not in cfg.languages:
            continue
        for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
            img_path = image_dir / f"{doc_id}{ext}"
            if img_path.exists():
                break
        else:
            LOG.warning("MAURDOR image missing for %s", doc_id)
            continue
        lines = _decode_page(payload)
        if not lines:
            continue
        img = Image.open(img_path).convert("L")
        yield Sample(image=img, lines=lines, task="ocr_layout",
                     source=f"maurdor:{cfg.split}:{doc_id}")
