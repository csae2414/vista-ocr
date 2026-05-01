"""WebDataset loader for ``pixparse/pdfa-eng-wds``.

The pixparse PDFA shards ship one image-per-page plus a JSON sidecar with
the OCR ground truth. The schema is large; here we extract only the bits
VISTA-OCR needs: rasterised page (PIL) and a list of line-level
``(text, bbox)`` tuples.

This module yields :class:`vista_ocr.data.types.Sample` objects so the
collate code is shared with synthetic data.

For unit tests we provide :class:`InMemoryPdfaDataset`, which produces
fully-synthetic PDFA-shaped samples without touching the network.
"""
from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from vista_ocr.data.preprocess import is_blank_image, is_latin_text
from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.tokenizer import Line

LOG = logging.getLogger(__name__)


@dataclass
class PdfaConfig:
    shards: list[str]                 # local glob, http, or s3:// urls
    drop_non_latin: bool = True
    drop_blank: bool = True
    max_lines_per_page: int = 200


def _decode_pdfa_record(record: dict) -> Sample | None:
    """Best-effort decoder for one webdataset record.

    Two pixparse PDFA variants exist; both expose ``__key__``, an image
    blob (key ending ``.png`` / ``.jpg``), and a JSON blob (``.json``)
    containing per-line annotations under ``lines`` or ``words`` arrays.
    We prefer ``lines`` for line-level boxes (paper-faithful)."""
    img_bytes = None
    payload: dict | None = None
    for k, v in record.items():
        if k.endswith((".png", ".jpg", ".jpeg")):
            img_bytes = v
        elif k.endswith(".json"):
            payload = json.loads(v.decode("utf-8") if isinstance(v, bytes) else v)
    if img_bytes is None or payload is None:
        return None

    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    raw_lines = payload.get("lines") or payload.get("text_lines") or []
    lines: list[Line] = []
    for entry in raw_lines:
        text = entry.get("text") or entry.get("transcription") or ""
        bbox = entry.get("bbox") or entry.get("box")
        if not text or bbox is None or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (int(v) for v in bbox)
        lines.append(Line(text=text, bbox=(x1, y1, x2, y2)))

    if not lines:
        return None
    return Sample(image=img, lines=lines, task="ocr_layout", source="pdfa")


def iter_pdfa(cfg: PdfaConfig) -> Iterator[Sample]:
    """Stream PDFA samples through WebDataset.

    Imported lazily so unit tests don't require ``webdataset``.
    """
    import webdataset as wds  # noqa: PLC0415

    pipeline = wds.WebDataset(cfg.shards, shardshuffle=False)
    for raw in pipeline:
        sample = _decode_pdfa_record(raw)
        if sample is None:
            continue
        if cfg.drop_blank and is_blank_image(sample.image):
            continue
        if cfg.drop_non_latin:
            sample.lines = [ln for ln in sample.lines if is_latin_text(ln.text)]
            if not sample.lines:
                continue
        if len(sample.lines) > cfg.max_lines_per_page:
            sample.lines = sample.lines[: cfg.max_lines_per_page]
        yield sample


class InMemoryPdfaDataset:
    """Fixture-friendly in-memory dataset that yields PDFA-shaped samples
    without any network or webdataset dependency."""

    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __iter__(self) -> Iterator[Sample]:
        yield from self.samples

    def __len__(self) -> int:
        return len(self.samples)
