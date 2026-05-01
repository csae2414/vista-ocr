"""Tests for :mod:`vista_ocr.ablation.base`.

Pure-Python tests: a fake :class:`Ablation` subclass plus a stub
``train`` function (monkey-patched at the import site) lets us exercise
the loop, summarisation, and reporting without spinning up a real
model. Closes the C1 gap (no tests on the ablation skeleton).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

from vista_ocr.ablation import Ablation, AblationVariant, summarise_history


@dataclass
class _StepRecord:
    loss: float


def test_summarise_history_empty_returns_nan():
    out = summarise_history("v", [], elapsed=0.5)
    assert out["name"] == "v"
    assert out["n_steps"] == 0
    assert out["loss_first"] != out["loss_first"]   # NaN
    assert out["loss_last"] != out["loss_last"]


def test_summarise_history_first_last_windowed():
    history = [_StepRecord(loss=float(i)) for i in range(100)]
    out = summarise_history("v", history, elapsed=10.0, window=10)
    # mean of 0..9 = 4.5; mean of 90..99 = 94.5
    assert out["loss_first"] == pytest.approx(4.5)
    assert out["loss_last"] == pytest.approx(94.5)
    assert out["delta"] == pytest.approx(-90.0)
    assert out["n_steps"] == 100
    assert out["per_step"] == pytest.approx(0.1)


class _StubAblation(Ablation):
    """Pretends to train; returns a fixed loss curve per variant."""

    def __init__(self, curves: dict[str, list[float]]):
        self.curves = curves
        self.calls: list[str] = []

    def variants(self):
        return [AblationVariant(name=n, extra={"tag": n.upper()})
                for n in self.curves]

    def build(self, variant):  # never reached -- we monkey-patch _train_one
        raise NotImplementedError

    def _train_one(self, variant, *, max_steps):
        self.calls.append(variant.name)
        history = [_StepRecord(loss=v) for v in self.curves[variant.name][:max_steps]]
        row = summarise_history(variant.name, history, elapsed=0.1,
                                window=self.summary_window)
        row.update(variant.extra)
        return row


def test_run_executes_each_variant_in_order():
    abl = _StubAblation({"a": [10.0] * 10, "b": [5.0] * 10})
    results = abl.run(max_steps=10)
    assert abl.calls == ["a", "b"]
    assert [r["name"] for r in results] == ["a", "b"]
    assert all(r["loss_last"] == r["loss_first"] for r in results)


def test_extra_metadata_flows_into_results():
    abl = _StubAblation({"alpha": [1.0] * 5})
    results = abl.run(max_steps=5)
    assert results[0]["tag"] == "ALPHA"


def test_report_prints_chosen_columns(caplog):
    abl = _StubAblation({"a": list(range(20))})
    results = abl.run(max_steps=20)
    with caplog.at_level(logging.INFO):
        Ablation.report(results, columns=["name", "loss_first", "loss_last"])
    msgs = " ".join(r.message for r in caplog.records)
    assert "name" in msgs
    assert "loss_first" in msgs
    assert "loss_last" in msgs
    # delta column was NOT requested -> not in output
    assert "delta" not in msgs.lower()
