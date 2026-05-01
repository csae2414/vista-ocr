"""Eval-metric tests."""
from __future__ import annotations

import pytest

from vista_ocr.eval.metrics_detection import (
    ap_at_iou_thresholds,
    area_f1,
    deteval,
    iou,
)
from vista_ocr.eval.metrics_recognition import (
    compute_cer,
    compute_wer,
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
