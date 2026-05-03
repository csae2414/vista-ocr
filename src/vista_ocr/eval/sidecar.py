"""Single source of truth for the ``vista-ocr eval`` JSON sidecar shape.

The eval verb populates an :class:`EvalSidecar` dataclass and dumps it
via :func:`dataclasses.asdict`. Optional metric blocks (``detection``,
``cer_ap``, ...) default to ``None`` and are dropped from the dumped
JSON when not applicable, so a manifest without ``bboxes`` produces a
sidecar with only the recognition keys -- back-compat with consumers
of the pre-G2 sidecar (no schema break).

Future readers (``vista-ocr report``, dashboard tools) import this
dataclass to know what to expect.
"""
from __future__ import annotations

import dataclasses as _dc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectionBlock:
    """Detection metrics aggregated across the manifest. Populated when
    the source manifest carries ``bboxes`` per record."""

    deteval_precision: float
    deteval_recall: float
    deteval_f1: float
    area_f1: float
    ap_at_iou: dict[str, float]   # keys are str(threshold) for JSON portability
    iou_threshold: float
    bbox_expand_px: int


@dataclass
class CerApBlock:
    """AP at CER thresholds. Populated when the operator passes
    ``--cer-ap-thresholds`` (paper §4.2 region-OCR row)."""

    ap: dict[str, float]
    thresholds: list[float]


@dataclass
class EvalSidecar:
    """``vista-ocr eval`` JSON sidecar schema.

    Required (always present):
      ckpt, ckpt_step, manifest, n_docs, n_empty,
      cer, wer, precision, recall, word_f1,
      elapsed_s, max_new_tokens, repetition_penalty, no_repeat_ngram_size

    Optional blocks:
      detection -- populated when manifest has bboxes
      cer_ap    -- populated when operator opts in via flag
    """

    ckpt: str
    ckpt_step: int
    manifest: str
    n_docs: int
    n_empty: int
    cer: float
    wer: float
    precision: float
    recall: float
    word_f1: float
    elapsed_s: float
    max_new_tokens: int
    repetition_penalty: float
    no_repeat_ngram_size: int
    detection: DetectionBlock | None = None
    cer_ap: CerApBlock | None = None

    def to_jsonable(self) -> dict[str, Any]:
        """Drop ``None`` blocks so the JSON sidecar stays minimal and
        back-compatible with pre-G2 readers."""
        d = _dc.asdict(self)
        return {k: v for k, v in d.items() if v is not None}
