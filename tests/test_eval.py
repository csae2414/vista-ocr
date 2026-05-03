"""Eval-metric tests.

Inventory:

* ``test_cer_*`` / ``test_wer_*`` / ``test_word_prf_*`` -- recognition
  metrics; sparse coverage, the per-doc helpers below extend it.
* ``test_per_doc_cer_*`` -- the single-pair CER helper used by both
  the eval verb's per-doc predictions writer and ``cer_ap``.
* ``test_cer_ap_*`` -- AP-at-CER-threshold (paper §4.2 region-OCR row).
* ``test_iou_*`` -- IoU primitive: identical, disjoint, partial,
  degenerate boxes.
* ``test_expand_box_*`` -- bbox-expand helper. Asymmetry (predictions
  expand, GT does not) is enforced at the eval-verb layer, NOT here;
  this module just verifies the math.
* ``test_deteval_*`` -- Wolf-Jolion DetEval: perfect match, empty,
  one-to-many (one GT split across multiple preds), many-to-one
  (multiple GT lines merged into one pred).
* ``test_area_f1_*`` -- pixel-rasterised F1.
* ``test_ap_at_iou_*`` -- AP@IoU sweep with multi-pred docs and
  intermediate-IoU cases.
"""
from __future__ import annotations

import pytest

from vista_ocr.eval.metrics_detection import (
    ap_at_iou_thresholds,
    area_f1,
    deteval,
    expand_box,
    iou,
)
from vista_ocr.eval.metrics_recognition import (
    cer_ap,
    compute_cer,
    compute_wer,
    per_doc_cer,
    word_exact_prf,
)

# ---------- recognition ----------

def test_cer_zero_for_perfect_match():
    assert compute_cer(["hello world"], ["hello world"]) == 0.0


def test_wer_zero_for_perfect_match():
    assert compute_wer(["hello world"], ["hello world"]) == 0.0


def test_word_prf_perfect_recovery():
    p, r, f = word_exact_prf(["a b c"], ["a b c"])
    assert (p, r, f) == (1.0, 1.0, 1.0)


def test_word_prf_handles_duplicates():
    p, r, f = word_exact_prf(["a a b"], ["a b b"])
    # TP = min(2,1) for 'a' + min(1,2) for 'b' = 1 + 1 = 2.
    # FP = 1 (extra b), FN = 1 (missing a).
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(2 / 3)


# ---------- detection ----------

def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (10, 10, 20, 20)) == 0.0
    assert iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)


def test_deteval_perfect_match():
    boxes = [(0.0, 0.0, 10.0, 10.0), (20.0, 0.0, 30.0, 10.0)]
    res = deteval(boxes, boxes)
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1 == 1.0


def test_deteval_empty_predictions_gives_zero():
    res = deteval([(0.0, 0.0, 10.0, 10.0)], [])
    assert res.precision == 0.0
    assert res.recall == 0.0


def test_area_f1_full_overlap():
    g = [(0.0, 0.0, 10.0, 10.0)]
    p = [(0.0, 0.0, 10.0, 10.0)]
    assert area_f1(g, p, page_shape=(20, 20)) == pytest.approx(1.0)


def test_area_f1_no_overlap():
    g = [(0.0, 0.0, 5.0, 5.0)]
    p = [(10.0, 10.0, 15.0, 15.0)]
    assert area_f1(g, p, page_shape=(20, 20)) == 0.0


def test_ap_at_iou_perfect_predictions():
    boxes = [(0.0, 0.0, 10.0, 10.0), (20.0, 0.0, 30.0, 10.0)]
    aps = ap_at_iou_thresholds(boxes, boxes)
    for t, v in aps.items():
        assert v == pytest.approx(1.0), (t, v)


def test_ap_at_iou_drops_with_threshold():
    g = [(0.0, 0.0, 10.0, 10.0)]
    p = [(2.0, 2.0, 12.0, 12.0)]   # IoU ≈ 0.47
    aps = ap_at_iou_thresholds(g, p)
    assert aps[0.5] == 0.0          # below 0.5 IoU after rounding
    assert aps[0.8] == 0.0


# ---------- per_doc_cer / cer_ap ----------

def test_per_doc_cer_perfect_pair():
    assert per_doc_cer("hello world", "hello world") == 0.0


def test_per_doc_cer_handles_empty_pair():
    """Both empty: CER 0.0 (the loop has no characters to score)."""
    assert per_doc_cer("", "") == 0.0


def test_per_doc_cer_handles_one_side_empty():
    """jiwer rejects a fully empty ref; the helper substitutes a
    space sentinel so the call survives. Either side empty against a
    non-empty other side yields CER >= 1.0 (worst)."""
    assert per_doc_cer("hello", "") >= 1.0
    assert per_doc_cer("", "hello") >= 1.0


def test_cer_ap_exact_match_at_zero_threshold():
    """All pairs perfect -> AP=1.0 at every threshold."""
    refs = ["hello world", "foo bar"]
    hyps = ["hello world", "foo bar"]
    aps = cer_ap(refs, hyps)
    for t, v in aps.items():
        assert v == pytest.approx(1.0), (t, v)


def test_cer_ap_partial_match_distributes_correctly():
    """One perfect, one bad. AP@0.0 = 0.5 (only the perfect one);
    AP@0.3 may include the bad one if its CER <= 0.3."""
    refs = ["hello", "hello"]
    hyps = ["hello", "xxxxx"]
    aps = cer_ap(refs, hyps, thresholds=(0.0, 1.0))
    assert aps[0.0] == pytest.approx(0.5)
    assert aps[1.0] == pytest.approx(1.0)   # all pairs have CER <= 1.0


def test_cer_ap_thresholds_are_monotonic():
    """For any (refs, hyps), AP must be non-decreasing in threshold:
    a more lenient threshold accepts at least as many pairs."""
    refs = ["foo", "bar", "baz"]
    hyps = ["foo", "bxr", "qux"]
    aps = cer_ap(refs, hyps, thresholds=(0.0, 0.1, 0.2, 0.3, 0.5, 1.0))
    keys = sorted(aps.keys())
    for a, b in zip(keys, keys[1:], strict=False):
        assert aps[a] <= aps[b], f"non-monotonic at {a}->{b}: {aps[a]} > {aps[b]}"


def test_cer_ap_empty_inputs():
    aps = cer_ap([], [])
    for v in aps.values():
        assert v == 0.0


# ---------- IoU edge cases ----------

def test_iou_degenerate_zero_area_box():
    """Zero-area box (point or line) returns 0.0, never NaN/raise."""
    assert iou((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 10.0, 10.0)) == 0.0
    assert iou((5.0, 5.0, 5.0, 10.0), (0.0, 0.0, 10.0, 10.0)) == 0.0


def test_iou_one_inside_the_other():
    """Small box fully inside large: IoU = small_area / large_area."""
    assert iou((0.0, 0.0, 5.0, 5.0), (0.0, 0.0, 10.0, 10.0)) == pytest.approx(25 / 100)


# ---------- expand_box ----------

def test_expand_box_zero_is_noop():
    b = (10, 20, 100, 50)
    assert expand_box(b, 0) == b


def test_expand_box_grows_each_side():
    """+2 px on each side: width grows by 4, height grows by 4."""
    out = expand_box((10, 20, 100, 50), 2)
    assert out == (8, 18, 102, 52)


def test_expand_box_negative_contracts():
    out = expand_box((10, 20, 100, 50), -2)
    assert out == (12, 22, 98, 48)


# ---------- DetEval many-to-one and one-to-many ----------

def test_deteval_split_prediction_at_strict_tr_misses():
    """One GT line, model emits two predictions covering halves of it.
    Each pred individually covers 50% of GT, which fails the default
    strict tr=0.8. Pinning this behaviour: at default thresholds the
    paper's metric does NOT credit a split detection."""
    gt = [(0.0, 0.0, 100.0, 10.0)]
    pred = [(0.0, 0.0, 50.0, 10.0), (50.0, 0.0, 100.0, 10.0)]
    res = deteval(gt, pred)
    assert res.recall == 0.0


def test_deteval_split_prediction_credited_at_lenient_tr():
    """Same split case but with a lenient tr -- each pred passes the
    GT-recall threshold individually, so the algorithm one-to-one
    matches one of them and recall is non-zero."""
    gt = [(0.0, 0.0, 100.0, 10.0)]
    pred = [(0.0, 0.0, 50.0, 10.0), (50.0, 0.0, 100.0, 10.0)]
    res = deteval(gt, pred, tr=0.4)
    assert res.recall > 0.0


def test_deteval_merged_prediction_credited():
    """Two GT lines, one merged pred that covers both. The pred is
    >= tp of its area on each GT, but each GT individually has only
    50% coverage by the pred (which fails default tr=0.8). This is
    the symmetric case to the split test above."""
    gt = [(0.0, 0.0, 50.0, 10.0), (50.0, 0.0, 100.0, 10.0)]
    pred = [(0.0, 0.0, 100.0, 10.0)]
    res = deteval(gt, pred, tr=0.4)
    # With tr=0.4, the merged pred passes both GT recall checks ->
    # one one-to-one match between the pred and one of the GT lines.
    assert res.recall > 0.0


def test_deteval_disjoint_match_zero():
    """Predictions completely outside GT regions -> precision/recall
    are both zero."""
    gt = [(0.0, 0.0, 10.0, 10.0)]
    pred = [(100.0, 100.0, 110.0, 110.0)]
    res = deteval(gt, pred)
    assert res.precision == 0.0
    assert res.recall == 0.0


# ---------- AP@IoU edge cases ----------

def test_ap_at_iou_with_multiple_predictions_per_doc():
    """Multi-pred case: 2 GT + 2 pred, perfect alignment ->
    AP=1.0 at every threshold."""
    gt = [(0.0, 0.0, 10.0, 10.0), (20.0, 0.0, 30.0, 10.0)]
    pred = list(gt)
    aps = ap_at_iou_thresholds(gt, pred)
    for v in aps.values():
        assert v == pytest.approx(1.0)


def test_ap_at_iou_passes_at_low_thresh_fails_at_high_thresh():
    """A pred with IoU in [0.7, 0.8) should pass AP@0.5/0.6/0.7 but
    fail @0.8."""
    gt = [(0.0, 0.0, 100.0, 100.0)]
    # 100x100 boxes shifted vertically by 17 -> intersection 100*83
    # = 8300, union 11700, IoU ~= 0.709.
    pred = [(0.0, 17.0, 100.0, 117.0)]
    iou_v = iou(gt[0], pred[0])
    assert 0.7 <= iou_v < 0.8, f"fixture iou is {iou_v}, expected in [0.7, 0.8)"
    aps = ap_at_iou_thresholds(gt, pred, thresholds=(0.5, 0.6, 0.7, 0.8))
    assert aps[0.5] == pytest.approx(1.0)
    assert aps[0.7] == pytest.approx(1.0)
    assert aps[0.8] == 0.0


def test_ap_at_iou_accepts_explicit_scores():
    """The signature already accepts ``scores`` (forward-compat with
    beam search). Passing scores=[1.0, 1.0] for a greedy caller is
    equivalent to omitting them."""
    gt = [(0.0, 0.0, 10.0, 10.0), (20.0, 0.0, 30.0, 10.0)]
    pred = list(gt)
    no_scores = ap_at_iou_thresholds(gt, pred)
    with_scores = ap_at_iou_thresholds(gt, pred, scores=[1.0, 1.0])
    assert no_scores == with_scores
