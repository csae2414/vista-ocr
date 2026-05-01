"""WebDataset loader for ``pixparse/pdfa-eng-wds``.

Shard schema (verified 2026-05-01 on shard ``pdfa-eng-train-0000.tar``):

* per-sample files: ``{key}.pdf`` (raw PDF) + ``{key}.json``
* JSON top-level: ``{"pages": [...]}``
* Each page: ``{"words": {...}, "lines": {...}, "images_bbox": [...],
  "images_bbox_no_text_overlap": [...]}``
* ``lines.text`` -- list of line strings
* ``lines.bbox`` -- list of ``[x_norm, y_norm, w_norm, h_norm]`` in
  page-relative coordinates ``[0, 1]``
* ``lines.score`` -- per-line confidence (1.0 for clean PDFs)
* ``lines.word_slice`` -- index range into ``words.text``

We render the PDF at 200 dpi (paper Appendix 0.A.1) and convert the
normalized line bboxes to pixel coordinates against the rendered image.
Multi-page PDFs are flattened: each page becomes its own ``Sample``.
"""
from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass

from PIL import Image

from vista_ocr.data.preprocess import is_blank_image, is_latin_text
from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.tokenizer import Line

LOG = logging.getLogger(__name__)

PAPER_DPI = 200


@dataclass
class PdfaConfig:
    shards: list[str]
    drop_non_latin: bool = True
    drop_blank: bool = True
    max_lines_per_page: int = 200
    dpi: int = PAPER_DPI
    min_line_score: float = 0.5
    flatten_multi_page: bool = True


def _norm_bbox_to_pixels(
    bbox: list[float], img_w: int, img_h: int
) -> tuple[int, int, int, int] | None:
    """Convert normalized ``[x, y, w, h]`` (page-relative, 0..1) to integer
    pixel ``[x1, y1, x2, y2]``."""
    if len(bbox) != 4:
        return None
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    x1 = max(0, int(round(x * img_w)))
    y1 = max(0, int(round(y * img_h)))
    x2 = min(img_w, int(round((x + w) * img_w)))
    y2 = min(img_h, int(round((y + h) * img_h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _render_pdf_page(pdf_bytes: bytes, dpi: int) -> Iterator[Image.Image]:
    """Render every page of ``pdf_bytes`` to a grayscale PIL image at the
    requested DPI. Uses pypdfium2 (no system dependencies)."""
    import pypdfium2 as pdfium  # noqa: PLC0415

    pdf = pdfium.PdfDocument(pdf_bytes)
    scale = dpi / 72.0
    try:
        for page in pdf:
            try:
                img = page.render(scale=scale, grayscale=True).to_pil()
                yield img.convert("L")
            finally:
                page.close()
    finally:
        pdf.close()


def _lines_for_page(page: dict, img_w: int, img_h: int, min_score: float) -> list[Line]:
    block = page.get("lines") or {}
    texts = block.get("text") or []
    bboxes = block.get("bbox") or []
    scores = block.get("score") or [1.0] * len(texts)

    out: list[Line] = []
    for text, bbox, score in zip(texts, bboxes, scores):
        if not text or score < min_score:
            continue
        px = _norm_bbox_to_pixels(bbox, img_w, img_h)
        if px is None:
            continue
        out.append(Line(text=text, bbox=px))
    return out


def _samples_from_record(
    pdf_bytes: bytes, payload: dict, cfg: PdfaConfig, source_key: str
) -> Iterator[Sample]:
    pages = payload.get("pages") or []
    rendered = list(_render_pdf_page(pdf_bytes, cfg.dpi))
    if len(rendered) != len(pages):
        # Some PDFs render fewer pages than the JSON describes (e.g. when
        # the OCR pipeline upstream dropped truly blank pages). Take the
        # min and align by index.
        n = min(len(rendered), len(pages))
        rendered = rendered[:n]
        pages = pages[:n]

    for img, page in zip(rendered, pages):
        w, h = img.size
        lines = _lines_for_page(page, w, h, cfg.min_line_score)
        if cfg.drop_non_latin:
            lines = [ln for ln in lines if is_latin_text(ln.text)]
        if not lines:
            continue
        if len(lines) > cfg.max_lines_per_page:
            lines = lines[: cfg.max_lines_per_page]
        if cfg.drop_blank and is_blank_image(img):
            continue
        yield Sample(
            image=img, lines=lines, task="ocr_layout", source=f"pdfa:{source_key}"
        )
        if not cfg.flatten_multi_page:
            return  # only emit the first page


def _decode_pdfa_record(record: dict, cfg: PdfaConfig) -> Iterator[Sample]:
    # webdataset emits records keyed by the bare extension ("pdf", "json")
    # without the leading dot, plus internal "__key__" / "__url__".
    pdf_bytes = record.get("pdf")
    json_blob = record.get("json")
    if pdf_bytes is None or json_blob is None:
        return
    payload = json.loads(json_blob.decode("utf-8") if isinstance(json_blob, bytes) else json_blob)
    source_key = record.get("__key__", "?")
    yield from _samples_from_record(pdf_bytes, payload, cfg, source_key)


def iter_pdfa(cfg: PdfaConfig) -> Iterator[Sample]:
    """Stream :class:`vista_ocr.data.types.Sample` from the configured PDFA shards.

    ``webdataset`` is imported lazily so unit tests don't need it.
    """
    import webdataset as wds  # noqa: PLC0415

    # empty_check=False so a per-worker shard slice that happens to be
    # empty (workers > shards) does not raise; the worker just yields
    # nothing and the DataLoader keeps draining the others.
    pipeline = wds.WebDataset(cfg.shards, shardshuffle=False, empty_check=False)
    for raw in pipeline:
        try:
            yield from _decode_pdfa_record(raw, cfg)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Failed to decode PDFA record %s: %s", raw.get("__key__"), exc)


class InMemoryPdfaDataset:
    """Fixture-friendly in-memory dataset that yields PDFA-shaped samples
    without webdataset / network."""

    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __iter__(self) -> Iterator[Sample]:
        yield from self.samples

    def __len__(self) -> int:
        return len(self.samples)
