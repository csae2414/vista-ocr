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


def _decode_idl_record(record: dict) -> Sample | None:
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

    pipeline = wds.WebDataset(cfg.shards, shardshuffle=False)
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
