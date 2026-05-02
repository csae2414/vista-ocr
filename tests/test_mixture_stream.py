"""Tests for :class:`vista_ocr.data.mixture_stream.MixedStream`."""
from __future__ import annotations

import logging

import pytest

from vista_ocr.data.mixture_stream import MixedStream, MixedStreamSource


def _src(name: str, items, weight: float = 1.0) -> MixedStreamSource:
    return MixedStreamSource(name=name, weight=weight, stream=iter(list(items)))


def test_deterministic_with_seed():
    a = MixedStream([
        _src("x", range(10000)),
        _src("y", range(10000, 20000)),
    ], seed=0)
    b = MixedStream([
        _src("x", range(10000)),
        _src("y", range(10000, 20000)),
    ], seed=0)
    assert [next(a) for _ in range(50)] == [next(b) for _ in range(50)]


def test_seeds_diverge():
    a = MixedStream([
        _src("x", range(10000)),
        _src("y", range(10000, 20000)),
    ], seed=0)
    b = MixedStream([
        _src("x", range(10000)),
        _src("y", range(10000, 20000)),
    ], seed=1)
    seq_a = [next(a) for _ in range(50)]
    seq_b = [next(b) for _ in range(50)]
    assert seq_a != seq_b


def test_weight_distribution_within_tolerance():
    """Weights of 7:2:1 -> ~70%/20%/10% over 1000 draws (±10 % each)."""
    stream = MixedStream([
        _src("a", range(1_000_000), weight=7),
        _src("b", range(1_000_000), weight=2),
        _src("c", range(1_000_000), weight=1),
    ], seed=42)
    [next(stream) for _ in range(1000)]
    counts = stream.counts
    assert 0.60 < counts["a"] / 1000 < 0.80, counts
    assert 0.10 < counts["b"] / 1000 < 0.30, counts
    assert 0.04 < counts["c"] / 1000 < 0.16, counts


def test_zero_weight_source_never_drawn():
    stream = MixedStream([
        _src("on",  range(1_000_000), weight=1),
        _src("off", range(1_000_000), weight=0),
    ], seed=0)
    [next(stream) for _ in range(200)]
    assert stream.counts["off"] == 0
    assert stream.counts["on"] == 200


def test_single_source_yields_all_items():
    stream = MixedStream([_src("only", [10, 20, 30])], seed=0)
    assert list(stream) == [10, 20, 30]


def test_exhausted_source_removed_from_rotation(caplog):
    stream = MixedStream([
        _src("short", [1, 2]),
        _src("long",  range(100, 200)),
    ], seed=0)
    with caplog.at_level(logging.INFO):
        out = [next(stream) for _ in range(20)]
    # short was exhausted at some point; counts should reflect.
    assert stream.counts["short"] == 2
    assert stream.counts["long"] == 18
    assert any("exhausted" in r.message for r in caplog.records)
    # Continuing past short exhaustion still works -- only long remains.
    more = [next(stream) for _ in range(10)]
    assert all(v >= 100 for v in more)


def test_full_exhaustion_raises_stop_iteration():
    stream = MixedStream([
        _src("a", [1, 2]),
        _src("b", [3, 4]),
    ], seed=0)
    items = list(stream)
    assert sorted(items) == [1, 2, 3, 4]


def test_validation_errors():
    with pytest.raises(ValueError, match="at least one source"):
        MixedStream([])
    with pytest.raises(ValueError, match="non-negative"):
        MixedStream([_src("x", [], weight=-1.0)])
    with pytest.raises(ValueError, match="positive weight"):
        MixedStream([
            _src("a", [], weight=0),
            _src("b", [], weight=0),
        ])
    with pytest.raises(ValueError, match="unique"):
        MixedStream([
            _src("dup", [1]),
            _src("dup", [2]),
        ])
