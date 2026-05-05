"""WebDataset loader for ``pixparse/idl-wds`` (UCSF Industry Documents Library).

Schema as actually shipped in ``data/raw/idl/idl-train-*.tar``
(verified 2026-05-05 against shards 00000 / 00005 / 00011):

- Each record's tar entry-set is keyed by a stem (e.g. ``klpb0135``)
  with extensions ``pdf`` (single-page document), ``tif`` (1-bit
  bilevel raster), ``json`` (layout sidecar), ``ocr`` (flat text
  dump). WebDataset surfaces these in ``record`` keyed without the
  leading dot.
- The JSON sidecar shape is ``{"pages": [{"text": [...], "bbox": [...],
  "poly": [...], "score": [...]}]}``. Per-page ``text`` / ``bbox`` /
  ``score`` are parallel arrays. ``bbox`` is normalised xywh in
  ``[0, 1]``; ``poly`` (4 ``{X, Y}`` corners) is provided too but
  unused -- the rectangle from xywh suffices for our line-level
  layout target.

We re-use PDFA's render + bbox-conversion helpers via private import
since both modules consume the same sort of normalised-xywh
parallel-arrays payload (PDFA nests them under ``page["lines"]``;
IDL has them at ``page`` top level).

Pre-fix history: an earlier draft of this file looked for
``record["png"|"jpg"|"jpeg"]`` and ``payload["blocks"|"lines"]``
expecting per-block ``{text, bbox}`` with bbox as pixel xyxy. None of
those keys exist on the actual shards; ``iter_idl`` produced zero
samples and stage-2 launches with ``DATA_MIX=pdfa+idl`` crashed at
WebDataset's ``check_empty`` boundary. See
``notes/plan_iter_idl_fix.md`` for the verification that closed the
fix.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass

from vista_ocr.data.pdfa import _norm_bbox_to_pixels, _render_pdf_page
from vista_ocr.data.preprocess import is_blank_image, is_latin_text
from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.tokenizer import Line

LOG = logging.getLogger(__name__)


@dataclass
class IdlConfig:
    shards: list[str]
    drop_non_latin: bool = True
    drop_blank: bool = True
    dpi: int = 200
    min_line_score: float = 0.5
    # Mirrors :class:`vista_ocr.data.pdfa.PdfaConfig.flatten_multi_page`:
    # True yields one Sample per rendered page; False emits only the
    # first page. IDL records sampled in the verification pass were
    # uniformly single-page so the flag is empirically inert, but
    # mirroring PDFA's default keeps the two loaders semantically
    # symmetric for any future multi-page record.
    flatten_multi_page: bool = True
    # Mirrors :class:`vista_ocr.data.pdfa.PdfaConfig.cycle`: when True
    # the pipeline reshuffles shards on each pass, runs a sample-level
    # shuffle buffer, and repeats indefinitely. Use it when this loader
    # is one source in a long-running mixture so the mixture's weight
    # ratio stays honoured (without cycling, IDL exhausts first and
    # the mixture silently collapses to PDFA-only).
    cycle: bool = False
    cycle_shuffle_buffer: int = 1000
    cycle_seed: int = 0


def _extract_lines(
    text_list: list[str],
    bbox_list: list[list[float]],
    score_list: list[float],
    img_w: int,
    img_h: int,
    min_score: float,
) -> list[Line]:
    """Parallel-arrays -> list[Line], dropping low-score and
    out-of-bounds bboxes. Does NOT filter by language; that's the
    caller's job (mirrors ``iter_pdfa``'s outer post-filter)."""
    out: list[Line] = []
    for text, bbox, score in zip(text_list, bbox_list, score_list, strict=False):
        if not text or score < min_score:
            continue
        px = _norm_bbox_to_pixels(bbox, img_w, img_h)
        if px is None:
            continue
        out.append(Line(text=text, bbox=px))
    return out


def _decode_idl_record(record: dict, cfg: IdlConfig) -> Iterator[Sample]:
    """Decode one WebDataset record into 0+ :class:`Sample` values.

    Yields nothing (rather than raising) on records missing the
    required ``pdf`` + ``json`` keys; ``iter_idl``'s outer
    try/except catches anything else. Each yielded ``Sample`` has
    ``task="ocr_layout"`` and ``source=f"idl:{record['__key__']}"``."""
    pdf_bytes = record.get("pdf")
    json_blob = record.get("json")
    if pdf_bytes is None or json_blob is None:
        return
    payload = json.loads(
        json_blob.decode("utf-8") if isinstance(json_blob, bytes) else json_blob
    )
    pages = payload.get("pages") or []
    if not pages:
        return
    rendered = list(_render_pdf_page(pdf_bytes, cfg.dpi))
    if not rendered:
        return
    n = min(len(rendered), len(pages))
    source_key = record.get("__key__", "?")
    for img, page in zip(rendered[:n], pages[:n], strict=False):
        w, h = img.size
        texts = page.get("text") or []
        bboxes = page.get("bbox") or []
        scores = page.get("score") or [1.0] * len(texts)
        lines = _extract_lines(texts, bboxes, scores, w, h, cfg.min_line_score)
        if cfg.drop_non_latin:
            lines = [ln for ln in lines if is_latin_text(ln.text)]
        if not lines:
            continue
        if cfg.drop_blank and is_blank_image(img):
            continue
        yield Sample(
            image=img, lines=lines, task="ocr_layout", source=f"idl:{source_key}"
        )
        if not cfg.flatten_multi_page:
            return


def iter_idl(cfg: IdlConfig) -> Iterator[Sample]:
    """Stream :class:`Sample` from the configured IDL shards.

    Uses ``empty_check=False`` so a per-worker shard slice that
    happens to be empty (workers > shards) does not raise; matches
    :func:`vista_ocr.data.pdfa.iter_pdfa`. Per-record exceptions are
    logged at WARNING and skipped -- one malformed record (bad PDF,
    truncated JSON, etc.) cannot abort a multi-day training chain.
    """
    import webdataset as wds  # noqa: PLC0415

    pipeline = wds.WebDataset(
        cfg.shards,
        shardshuffle=cfg.cycle,
        empty_check=False,
        seed=cfg.cycle_seed if cfg.cycle else None,
    )
    if cfg.cycle:
        pipeline = pipeline.shuffle(cfg.cycle_shuffle_buffer).repeat()
    for raw in pipeline:
        try:
            yield from _decode_idl_record(raw, cfg)
        except Exception as e:  # noqa: BLE001 -- mirror iter_pdfa
            LOG.warning(
                "iter_idl: skipping malformed record %r: %s",
                raw.get("__key__", "?"), e,
            )
            continue
