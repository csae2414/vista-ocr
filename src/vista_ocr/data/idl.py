"""WebDataset loader for ``pixparse/idl-wds`` (UCSF Industry Documents Library).

Same shape as :mod:`vista_ocr.data.pdfa`; the schema differs only in the
JSON sidecar field names. We re-use the PDFA decoder by aliasing the
sidecar keys before decoding."""
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


@dataclass
class IdlConfig:
    shards: list[str]
    drop_non_latin: bool = True
    drop_blank: bool = True
    # Mirrors :class:`vista_ocr.data.pdfa.PdfaConfig.cycle`: when True
    # the pipeline reshuffles shards on each pass, runs a sample-level
    # shuffle buffer, and repeats indefinitely. Use it when this loader
    # is one source in a long-running mixture so the mixture's weight
    # ratio stays honoured (without cycling, IDL exhausts first and
    # the mixture silently collapses to PDFA-only).
    cycle: bool = False
    cycle_shuffle_buffer: int = 1000
    cycle_seed: int = 0


def _decode_idl_record(record: dict) -> Sample | None:
    # webdataset strips the leading dot from extensions.
    img_bytes = record.get("png") or record.get("jpg") or record.get("jpeg")
    json_blob = record.get("json")
    if img_bytes is None or json_blob is None:
        return None
    payload = json.loads(json_blob.decode("utf-8") if isinstance(json_blob, bytes) else json_blob)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    blocks = payload.get("blocks") or payload.get("lines") or []
    lines: list[Line] = []
    for blk in blocks:
        text = blk.get("text") or ""
        bbox = blk.get("bbox") or blk.get("polygon")
        if not text or bbox is None:
            continue
        if len(bbox) == 4:
            x1, y1, x2, y2 = (int(v) for v in bbox)
        elif len(bbox) >= 8:                        # polygon → enclosing rect
            xs = [int(bbox[i]) for i in range(0, len(bbox), 2)]
            ys = [int(bbox[i]) for i in range(1, len(bbox), 2)]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            continue
        lines.append(Line(text=text, bbox=(x1, y1, x2, y2)))
    if not lines:
        return None
    return Sample(image=img, lines=lines, task="ocr_layout", source="idl")


def iter_idl(cfg: IdlConfig) -> Iterator[Sample]:
    import webdataset as wds  # noqa: PLC0415

    pipeline = wds.WebDataset(
        cfg.shards,
        shardshuffle=cfg.cycle,
        seed=cfg.cycle_seed if cfg.cycle else None,
    )
    if cfg.cycle:
        pipeline = pipeline.shuffle(cfg.cycle_shuffle_buffer).repeat()
    for raw in pipeline:
        sample = _decode_idl_record(raw)
        if sample is None:
            continue
        if cfg.drop_blank and is_blank_image(sample.image):
            continue
        if cfg.drop_non_latin:
            sample.lines = [ln for ln in sample.lines if is_latin_text(ln.text)]
            if not sample.lines:
                continue
        yield sample
