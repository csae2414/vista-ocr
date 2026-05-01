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


def word_exact_prf(refs: list[str], hyps: list[str]) -> tuple[float, float, float]:
    """Word-multiset Precision / Recall / F1.

    Paper Table 2 reports these for SROIE 2019. Implementation: for each
    document, intersect the multi-set of whitespace-tokenized words; sum
    TP / FP / FN across the whole eval set, then return micro P, R, F1.
    """
    tp = fp = fn = 0
    for ref, hyp in zip(refs, hyps):
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


def recognition_metrics(refs: list[str], hyps: list[str]) -> RecognitionMetrics:
    p, r, f = word_exact_prf(refs, hyps)
    return RecognitionMetrics(
        cer=compute_cer(refs, hyps),
        wer=compute_wer(refs, hyps),
        precision=p,
        recall=r,
        f1=f,
    )
