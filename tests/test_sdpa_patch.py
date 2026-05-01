"""Tests for the optional SDPA monkey-patch.

The numerical-equivalence test is the ship gate: per the C3 entry in
the notes, ``apply()`` must not be flipped on in production unless
``numerical_equivalence`` passes on the bf16 (1, 16, 2048, 64) shape.
The CPU CI path runs a smaller fp32 shape; the bf16-on-cuda check
runs only on a CUDA box.
"""
from __future__ import annotations

import pytest
import torch

from vista_ocr.models.sdpa_patch import apply, numerical_equivalence


def test_disabled_via_env(monkeypatch):
    monkeypatch.setenv("VISTA_NO_SDPA", "1")
    # Re-import to ensure clean state for this test
    from vista_ocr.models import sdpa_patch
    sdpa_patch._PATCHED = False
    assert apply() is False
    assert sdpa_patch._PATCHED is False


def test_numerical_equivalence_fp32_cpu():
    """Tiny fp32 case: SDPA and eager should match within 1e-3."""
    max_diff, mean_diff, passes = numerical_equivalence(
        batch=1, heads=4, seq=16, head_dim=8, dtype=torch.float32,
    )
    assert passes, f"fp32 max={max_diff:.6f} mean={mean_diff:.6f}"
    assert max_diff < 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 CUDA only")
def test_numerical_equivalence_bf16_cuda_at_paper_shape():
    """C3 ship gate: bf16 on (1, 16, 2048, 64) must pass tolerance."""
    max_diff, mean_diff, passes = numerical_equivalence(
        batch=1, heads=16, seq=2048, head_dim=64, dtype=torch.bfloat16,
        rtol=1e-3, atol=1e-3,
    )
    assert passes, f"bf16 max={max_diff:.4f} mean={mean_diff:.4f}"


def test_apply_is_idempotent(monkeypatch):
    """Calling apply() twice doesn't double-patch (the original forward
    isn't lost)."""
    # Ensure we're starting clean
    monkeypatch.delenv("VISTA_NO_SDPA", raising=False)
    from vista_ocr.models import sdpa_patch
    if sdpa_patch._PATCHED:
        sdpa_patch.revert()
    try:
        first = apply()
        second = apply()
        assert first is True
        assert second is True
    finally:
        sdpa_patch.revert()
