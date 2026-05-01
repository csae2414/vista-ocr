"""Tests for the VRAM-aware page-resolution helper."""
from __future__ import annotations

import pytest

from vista_ocr.training import resolution as R


def test_preset_for_vram_thresholds():
    assert R.preset_for_vram(8) == "tiny"
    assert R.preset_for_vram(12) == "tiny"
    assert R.preset_for_vram(14) == "small"
    assert R.preset_for_vram(16) == "small"
    assert R.preset_for_vram(20) == "medium"   # exactly the 3090's 24GB tier
    assert R.preset_for_vram(24) == "medium"
    assert R.preset_for_vram(40) == "large"
    assert R.preset_for_vram(80) == "large"


def test_preset_for_vram_below_floor_is_tiny():
    assert R.preset_for_vram(0) == "tiny"
    assert R.preset_for_vram(0.5) == "tiny"


def test_resolve_named_preset():
    assert R.resolve("tiny") == R.PRESETS["tiny"]
    assert R.resolve("medium") == R.PRESETS["medium"]
    assert R.resolve("large") == R.PRESETS["large"]


def test_resolve_none_defaults_to_medium():
    assert R.resolve(None) == R.PRESETS["medium"]


def test_resolve_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown page preset"):
        R.resolve("huge")


def test_resolve_auto_falls_back_to_tiny_without_cuda(monkeypatch, caplog):
    """When no CUDA device is visible the auto path must not crash; it
    picks the conservative floor and logs a WARNING."""
    monkeypatch.setattr(R, "detect_vram_gb", lambda: None)
    import logging
    with caplog.at_level(logging.WARNING):
        out = R.resolve("auto")
    assert out == R.PRESETS["tiny"]
    assert any("auto-detect" in r.message for r in caplog.records)


def test_resolve_auto_uses_detected_vram(monkeypatch):
    """Stub VRAM detection at 24 GB -> medium; at 48 GB -> large."""
    monkeypatch.setattr(R, "detect_vram_gb", lambda: 24.0)
    assert R.resolve("auto") == R.PRESETS["medium"]
    monkeypatch.setattr(R, "detect_vram_gb", lambda: 48.0)
    assert R.resolve("auto") == R.PRESETS["large"]


def test_3090_24gb_picks_medium():
    """Pin the operator-relevant case: a 3090 (24 GB) must resolve to
    medium = 1100x850 so the pretraining run launches with the same
    canvas the rest of the pipeline was tested on."""
    assert R.preset_for_vram(24.0) == "medium"
    assert R.PRESETS["medium"].height == 1100
    assert R.PRESETS["medium"].width == 850
