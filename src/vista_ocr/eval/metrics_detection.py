"""Text-detection metrics for VISTA-OCR.

Three primary protocols are implemented:

* **DetEval** (Wolf & Jolion, IJDAR 2006 — paper ref [37]). Object-count /
  area-graph evaluation: a prediction matches a ground-truth box if their
  intersection covers at least ``tr`` of the GT area and at least ``tp``
  of the prediction's area; many-to-one matches are counted at half
  weight.
* **Area-F1** — pixel-level F1 over the union of all boxes (paper's
  MAURDOR metric).
* **AP @ IoU** — COCO-style average precision at multiple IoU thresholds.

All functions take axis-aligned boxes as ``(x1, y1, x2, y2)`` tuples.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

LOG = logging.getLogger(__name__)

Box = tuple[float, float, float, float]


def _box_area(b: Box) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_area(a: Box, b: Box) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def iou(a: Box, b: Box) -> float:
    inter = _intersection_area(a, b)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def expand_box(box: Box, px: int) -> Box:
    """Expand a bbox by ``px`` pixels on each side (negative contracts).

    The eval verb uses this to apply the paper's §4.1.1 +1 / +2 px
    expansion to PREDICTED boxes only. Never call this on ground
    truth -- doing so would inflate detection F1 above what's
    achievable. The asymmetry is enforced by tests, not by this
    module (the helper itself is symmetric; the caller wires the
    asymmetry).
    """
    if px == 0:
        return box
    x1, y1, x2, y2 = box
    return (x1 - px, y1 - px, x2 + px, y2 + px)


# -------------------- DetEval (Wolf & Jolion 2006) -----------------------

@dataclass
class DetEvalResult:
    precision: float
    recall: float
    f1: float


def deteval(
    gt: list[Box],
    pred: list[Box],
    *,
    tr: float = 0.8,
    tp: float = 0.4,
) -> DetEvalResult:
    """Compute Wolf & Jolion's DetEval P/R/F1.

    A prediction ``p`` is considered to match a ground-truth ``g`` when
    both ``area(p ∩ g) / area(g) >= tr`` *and* ``area(p ∩ g) / area(p) >= tp``.
    Many-to-one matches contribute 0.5; one-to-one matches contribute 1.0
    (the original paper's heuristic). The metric returns
    ``precision = matched_preds / |pred|`` and
    ``recall = matched_gt / |gt|``.
    """
    n_gt, n_pred = len(gt), len(pred)
    if n_gt == 0 and n_pred == 0:
        return DetEvalResult(1.0, 1.0, 1.0)
    if n_gt == 0 or n_pred == 0:
        return DetEvalResult(0.0, 0.0, 0.0)

    inter = np.zeros((n_gt, n_pred))
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            inter[i, j] = _intersection_area(g, p)
    g_area = np.array([_box_area(g) for g in gt])
    p_area = np.array([_box_area(p) for p in pred])

    rec_mat = inter / np.maximum(g_area[:, None], 1e-9)
    prec_mat = inter / np.maximum(p_area[None, :], 1e-9)
    match = (rec_mat >= tr) & (prec_mat >= tp)

    matched_gt = 0.0
    matched_pred = 0.0
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    # one-to-one
    for i in range(n_gt):
        for j in range(n_pred):
            if match[i, j] and i not in used_gt and j not in used_pred:
                matched_gt += 1
                matched_pred += 1
                used_gt.add(i)
                used_pred.add(j)
                break
    # many-to-one (one GT split across many preds, or vice versa)
    for i in range(n_gt):
        if i in used_gt:
            continue
        candidates = [j for j in range(n_pred) if match[i, j]]
        if len(candidates) > 1:
            matched_gt += 0.5
            for j in candidates:
                if j not in used_pred:
                    matched_pred += 0.5 / max(1, len(candidates))
                    used_pred.add(j)
    for j in range(n_pred):
        if j in used_pred:
            continue
        candidates = [i for i in range(n_gt) if match[i, j]]
        if len(candidates) > 1:
            matched_pred += 0.5
            for i in candidates:
                if i not in used_gt:
                    matched_gt += 0.5 / max(1, len(candidates))
                    used_gt.add(i)

    precision = matched_pred / n_pred
    recall = matched_gt / n_gt
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return DetEvalResult(precision=precision, recall=recall, f1=f1)


# -------------------- Area F1 -------------------------------------------

def _rasterize_union(boxes: list[Box], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        x1c, y1c = max(0, int(x1)), max(0, int(y1))
        x2c, y2c = min(w, int(x2)), min(h, int(y2))
        if x2c > x1c and y2c > y1c:
            mask[y1c:y2c, x1c:x2c] = True
    return mask


def area_f1(gt: list[Box], pred: list[Box], page_shape: tuple[int, int]) -> float:
    """Pixel-level F1 over the union of all boxes (paper's MAURDOR metric)."""
    g_mask = _rasterize_union(gt, page_shape)
    p_mask = _rasterize_union(pred, page_shape)
    inter = np.logical_and(g_mask, p_mask).sum()
    n_g = g_mask.sum()
    n_p = p_mask.sum()
    if n_g == 0 and n_p == 0:
        return 1.0
    precision = inter / n_p if n_p > 0 else 0.0
    recall = inter / n_g if n_g > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


# -------------------- Average Precision @ IoU ---------------------------

def _ap_at_iou(gt: list[Box], pred: list[Box], scores: list[float], iou_thr: float) -> float:
    """COCO-style AP at a single IoU threshold (no interpolation)."""
    if not gt and not pred:
        return 1.0
    if not gt:
        return 0.0
    if not pred:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    pred_sorted = [pred[i] for i in order]
    matched = [False] * len(gt)

    tp = np.zeros(len(pred_sorted))
    fp = np.zeros(len(pred_sorted))
    for k, p in enumerate(pred_sorted):
        best_iou, best_i = 0.0, -1
        for i, g in enumerate(gt):
            if matched[i]:
                continue
            v = iou(p, g)
            if v > best_iou:
                best_iou = v
                best_i = i
        if best_iou >= iou_thr and best_i >= 0:
            tp[k] = 1
            matched[best_i] = True
        else:
            fp[k] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / max(1, len(gt))
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    ap = 0.0
    prev_r = 0.0
    for r, p in zip(recall, precision, strict=False):
        ap += (r - prev_r) * p
        prev_r = r
    return float(ap)


def ap_at_iou_thresholds(
    gt: list[Box],
    pred: list[Box],
    scores: list[float] | None = None,
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8),
) -> dict[float, float]:
    """Returns ``{iou_threshold: AP}``."""
    if scores is None:
        scores = [1.0] * len(pred)
    return {t: _ap_at_iou(gt, pred, scores, t) for t in thresholds}
