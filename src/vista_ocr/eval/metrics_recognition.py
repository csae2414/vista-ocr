"""Text-recognition metrics.

* :func:`compute_cer` and :func:`compute_wer` use ``jiwer``.
* :func:`word_exact_prf` is the SROIE-style word-set P/R/F1 used in the
  paper's Table 2 (treats prediction and ground truth as multi-sets of
  whitespace-separated tokens).
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import jiwer

LOG = logging.getLogger(__name__)


@dataclass
class RecognitionMetrics:
    cer: float
    wer: float
    precision: float
    recall: float
    f1: float


def compute_cer(refs: list[str], hyps: list[str]) -> float:
    """Character Error Rate aggregated over all examples."""
    if not refs:
        return 0.0
    return float(jiwer.cer(refs, hyps))


def compute_wer(refs: list[str], hyps: list[str]) -> float:
    """Word Error Rate aggregated over all examples."""
    if not refs:
        return 0.0
    return float(jiwer.wer(refs, hyps))


def per_doc_cer(ref: str, hyp: str) -> float:
    """Character Error Rate for a single (ref, hyp) pair.

    Single source of truth for per-document CER -- callers in
    :func:`cer_ap` and the eval verb's per-doc predictions JSONL all
    go through this so a future jiwer pin or normalization tweak
    lands in one place.

    Edge-case convention (matches OCR literature):

    * Both empty -> 0.0 (nothing to score, treat as match).
    * One side empty, other non-empty -> 1.0 (worst). jiwer raises
      on empty refs after whitespace-tokenisation; we short-circuit.
    """
    if not ref and not hyp:
        return 0.0
    if not ref or not hyp:
        return 1.0
    return float(jiwer.cer(ref, hyp))


def word_exact_prf(refs: list[str], hyps: list[str]) -> tuple[float, float, float]:
    """Word-multiset Precision / Recall / F1.

    Paper Table 2 reports these for SROIE 2019. Implementation: for each
    document, intersect the multi-set of whitespace-tokenized words; sum
    TP / FP / FN across the whole eval set, then return micro P, R, F1.
    """
    tp = fp = fn = 0
    for ref, hyp in zip(refs, hyps, strict=False):
        ref_c = Counter(ref.split())
        hyp_c = Counter(hyp.split())
        for w, n in hyp_c.items():
            match = min(n, ref_c.get(w, 0))
            tp += match
            fp += n - match
        for w, n in ref_c.items():
            match = min(n, hyp_c.get(w, 0))
            fn += n - match
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def cer_ap(
    refs: list[str],
    hyps: list[str],
    thresholds: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3),
) -> dict[float, float]:
    """Average Precision at CER thresholds (paper §4.2 region-OCR row).

    For each (ref, hyp) pair, compute per-doc CER. AP at threshold ``t``
    is the fraction of pairs with ``CER <= t``. Returns
    ``{threshold: AP}``.

    Lower-is-better (mirror of ``ap_at_iou_thresholds`` in
    ``metrics_detection``, which is higher-is-better). The ``0.0``
    threshold reports exact-match rate; ``0.3`` is a generous "close
    enough" bucket for the long-form region-OCR rows. Greedy decoding
    has no per-prediction confidence, so callers don't pass scores --
    if a future beam-search caller wants to weight by confidence,
    extend the signature there.
    """
    if not refs:
        return {t: 0.0 for t in thresholds}
    cers = [per_doc_cer(r, h) for r, h in zip(refs, hyps, strict=False)]
    n = len(cers)
    return {
        float(t): sum(1 for c in cers if c <= t) / n
        for t in thresholds
    }


def recognition_metrics(refs: list[str], hyps: list[str]) -> RecognitionMetrics:
    p, r, f = word_exact_prf(refs, hyps)
    return RecognitionMetrics(
        cer=compute_cer(refs, hyps),
        wer=compute_wer(refs, hyps),
        precision=p,
        recall=r,
        f1=f,
    )
