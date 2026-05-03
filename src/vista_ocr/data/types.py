"""Shared dataclasses for VISTA-OCR data pipelines.

``Line`` is defined alongside the tokenizer (it is the layout primitive
both halves of the codebase round-trip through), and re-exported here
so data modules import their types from a data module rather than
reaching into ``vista_ocr.tokenizer``. The canonical definition stays
in ``vista_ocr.tokenizer.tokenizer`` to avoid circular imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from PIL import Image

from vista_ocr.tokenizer.tokenizer import Line

__all__ = ["Line", "Sample", "TaskName"]

TaskName = Literal["ocr", "ocr_layout", "region_ocr", "find_it"]


@dataclass
class Sample:
    """A single training sample.

    :param image: Grayscale image as a PIL.Image (L mode) or a 1-channel
        float tensor in ``[0, 1]``.
    :param lines: Layout-aware ground-truth lines with pixel bboxes.
    :param task: Which prompt/output formatting to use.
    :param query_text: Required iff ``task == "find_it"``.
    :param query_bbox: Required iff ``task == "region_ocr"``.
    :param source: Free-form provenance tag (e.g. ``"pdfa"``).
    """

    image: Image.Image | torch.Tensor
    lines: list[Line]
    task: TaskName = "ocr_layout"
    query_text: str | None = None
    query_bbox: tuple[int, int, int, int] | None = None
    source: str = ""
    meta: dict = field(default_factory=dict)
