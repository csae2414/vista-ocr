"""HierText loader (Google Research, ICDAR 2023).

HierText combines printed and handwritten text in the **same** scenes,
with a paragraph -> line -> word hierarchy and explicit ``handwritten``
flags per line. That makes it the cleanest single-source choice for a
mixed printed/handwritten pretraining corpus.

Dataset card: https://huggingface.co/datasets/google-research-datasets/hiertext
Original release: https://github.com/google-research-datasets/hiertext
Paper: Long et al., *Towards End-to-End Unified Scene Text Detection and
Layout Analysis*, CVPR 2022 (ICDAR-2023 challenge).

Schema (per item)::

    {
      "image_id": "abc",
      "image": <PIL.Image>,
      "annotations": {
        "paragraphs": [
          {
            "vertices": [[x,y]*4],
            "lines": [
              {
                "vertices": [[x,y]*4],
                "text": "...",
                "legible": true,
                "handwritten": false,
                "vertical": false,
                "words": [...]
              }
            ]
          }
        ]
      }
    }

We project the line-level 4-point quad polygons to axis-aligned bboxes
(min/max of vertices) to match the rest of the loaders. Vertical lines
and illegible lines are dropped. ``include_handwritten`` / ``include_printed``
let callers fold in only one half if they want.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass

from PIL import Image

from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.tokenizer import Line

LOG = logging.getLogger(__name__)


@dataclass
class HierTextConfig:
    """Loader configuration.

    :param hf_dataset: HuggingFace dataset id; overridden by ``cache_dir``
        when ``cache_dir`` points at a local snapshot.
    :param split: ``"train"`` | ``"validation"`` | ``"test"``.
    :param include_handwritten: keep handwritten lines.
    :param include_printed: keep printed lines.
    :param skip_vertical: drop lines flagged ``vertical=True`` (the paper
        and our serialization assume horizontal reading order).
    :param min_lines_per_image: drop images with fewer than this many
        kept lines after filtering.
    """

    hf_dataset: str = "google-research-datasets/hiertext"
    split: str = "train"
    include_handwritten: bool = True
    include_printed: bool = True
    skip_vertical: bool = True
    min_lines_per_image: int = 1
    streaming: bool = False
    cache_dir: str | None = None


def _decode_annotations(raw) -> dict:
    """Annotations field is sometimes a dict, sometimes a JSON string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, bytes | bytearray):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _quad_to_aabb(vertices: list[list[float]]) -> tuple[int, int, int, int] | None:
    """Project a 4-point polygon to (x1, y1, x2, y2) axis-aligned bbox."""
    if not vertices or len(vertices) < 3:
        return None
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _lines_from_annotations(
    annotations: dict, cfg: HierTextConfig
) -> list[Line]:
    out: list[Line] = []
    for para in annotations.get("paragraphs", []):
        for line_dict in para.get("lines", []):
            if cfg.skip_vertical and line_dict.get("vertical"):
                continue
            if not line_dict.get("legible", True):
                continue
            is_hw = bool(line_dict.get("handwritten", False))
            if is_hw and not cfg.include_handwritten:
                continue
            if (not is_hw) and not cfg.include_printed:
                continue
            text = (line_dict.get("text") or "").strip()
            if not text:
                continue
            bbox = _quad_to_aabb(line_dict.get("vertices") or [])
            if bbox is None:
                continue
            out.append(Line(text=text, bbox=bbox))
    return out


def iter_hiertext(cfg: HierTextConfig) -> Iterator[Sample]:
    """Yield :class:`vista_ocr.data.types.Sample` from HierText.

    ``datasets`` is imported lazily so unit tests don't require it.
    """
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(
        cfg.hf_dataset,
        split=cfg.split,
        streaming=cfg.streaming,
        cache_dir=cfg.cache_dir,
    )
    for item in ds:
        ann = _decode_annotations(item.get("annotations"))
        lines = _lines_from_annotations(ann, cfg)
        if len(lines) < cfg.min_lines_per_image:
            continue
        img = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        yield Sample(
            image=img.convert("L"),
            lines=lines,
            task="ocr_layout",
            source=f"hiertext:{cfg.split}:{item.get('image_id', '?')}",
        )
