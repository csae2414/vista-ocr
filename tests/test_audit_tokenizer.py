"""Tests for ``tools/audit_tokenizer.py``'s truncation audit path.

Inventory (notes/plan_audit_tools_polish.md commit 2):

- ``test_audit_pages_uses_build_target_ids_untruncated`` -- the
  audit reports the same length ``build_target_ids(...,
  truncate=False)`` returns. Pins the contract that the audit
  uses the production helper, not a re-implemented formula.
- ``test_audit_pages_truncation_count_uses_max_target_tokens`` --
  feed one short Sample + one long Sample whose pre-truncation
  length exceeds ``MAX_TARGET_TOKENS``; assert the long Sample's
  reported target length is the *pre-truncation* length (NOT
  capped at 2048) and the truncation count is 1. This is the
  regression catch for a future caller forgetting
  ``truncate=False``.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

from vista_ocr.data.collate import MAX_TARGET_TOKENS, build_target_ids
from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.build_spm import train_spm
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import (
    Line,
    VistaTokenizer,
    list_special_and_spatial_tokens,
)


REPO = Path(__file__).resolve().parent.parent
TOOL_PATH = REPO / "tools" / "audit_tokenizer.py"


@pytest.fixture(scope="module")
def tool():
    """Import audit_tokenizer.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "audit_tokenizer_tool", TOOL_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def grid() -> SpatialGrid:
    return SpatialGrid(canvas_h=128, canvas_w=128, quantizer_px=4, scheme="original")


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory: pytest.TempPathFactory, grid: SpatialGrid) -> VistaTokenizer:
    tmp = tmp_path_factory.mktemp("audit_tok_spm")
    corpus_path = tmp / "c.txt"
    corpus_path.write_text(
        "hello world\nfoo bar\nthe quick brown fox jumps over\n"
        "abc def ghi jkl mno pqr stu vwx yz\n"
        "Digits 0 1 2 3 4 5 6 7 8 9 ten eleven twelve\n" * 200,
        encoding="utf-8",
    )
    out = tmp / "tk"
    train_spm(corpus_path, out, vocab_size=180,
              user_symbols=list_special_and_spatial_tokens(grid))
    return VistaTokenizer(out.with_suffix(".model"), grid)


def _short_sample() -> Sample:
    return Sample(
        image=None,
        lines=[Line("hello world", (10, 20, 80, 40))],
        task="ocr_layout",
    )


def _long_sample(n_lines: int = 400) -> Sample:
    return Sample(
        image=None,
        lines=[
            Line(
                f"line {i} fox jumps over lazy dog",
                (10 + (i % 4) * 12, 20 + i * 2, 80, 40 + i * 2),
            )
            for i in range(n_lines)
        ],
        task="ocr_layout",
    )


def test_audit_pages_uses_build_target_ids_untruncated(tool, tokenizer):
    """The audit must report exactly what
    ``build_target_ids(truncate=False)`` returns. NOT a
    re-implemented formula. Pin the contract."""
    samples = [_short_sample(), _long_sample(n_lines=50)]
    expected_lengths = [
        len(build_target_ids(tokenizer, s, truncate=False)[0])
        for s in samples
    ]

    audit = tool._audit_pages_truncation(
        tokenizer=tokenizer,
        samples_iter=iter(samples),
        n=10,
        task="ocr_layout",
        max_target_tokens=MAX_TARGET_TOKENS,
    )
    # Reported percentile distribution should be over exactly those
    # two lengths, so min == shortest, max == longest.
    assert audit["n_pages"] == 2
    assert audit["target_tokens"]["min"] == float(min(expected_lengths))
    assert audit["target_tokens"]["max"] == float(max(expected_lengths))


def test_audit_pages_truncation_count_uses_max_target_tokens(tool, tokenizer):
    """Feed one short page and one synthesised long page whose
    pre-truncation length exceeds ``MAX_TARGET_TOKENS``. The audit
    must:

      1. Report the long page's pre-truncation target length as
         strictly > MAX_TARGET_TOKENS (NOT capped at 2048 -- that
         would mean a future caller forgot ``truncate=False``).
      2. Count exactly 1 page as truncated.
    """
    short = _short_sample()
    # 400 lines @ ~10-12 tokens/line + 4 spatial tokens/line in
    # ocr_layout = > MAX_TARGET_TOKENS. Verify the assumption
    # holds before relying on it.
    long_sample = _long_sample(n_lines=400)
    long_seq, _ = build_target_ids(tokenizer, long_sample, truncate=False)
    assert len(long_seq) > MAX_TARGET_TOKENS, (
        f"long_sample pre-truncation length {len(long_seq)} <= MAX_TARGET_TOKENS "
        f"{MAX_TARGET_TOKENS}; bump n_lines"
    )

    audit = tool._audit_pages_truncation(
        tokenizer=tokenizer,
        samples_iter=iter([short, long_sample]),
        n=10,
        task="ocr_layout",
        max_target_tokens=MAX_TARGET_TOKENS,
    )
    assert audit["n_pages"] == 2
    assert audit["truncated_pages"] == 1
    assert audit["truncated_frac"] == 0.5
    # Headline regression catch: the reported max length must be
    # the long sample's PRE-truncation length, not 2048.
    assert audit["target_tokens"]["max"] > MAX_TARGET_TOKENS, (
        f"audit reported target_tokens max {audit['target_tokens']['max']} "
        f"<= MAX_TARGET_TOKENS {MAX_TARGET_TOKENS} -- did the audit lose "
        f"truncate=False on build_target_ids?"
    )
