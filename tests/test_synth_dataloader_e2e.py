"""End-to-end synth → make_mixed_loader → collate → Batch tests.

Inventory (notes/plan_phase_j_followup.md §Fix 4):

- ``test_handwritten_synth_through_make_mixed_loader_collates`` --
  pure collate-invariant test on the actual Batch fields.
- ``test_synth_only_supported_via_pdfa_weight_zero`` -- the
  zero-PDFA contract test (synth-only loader works without PDFA
  shards). Verified at plan-time; this asserts it stays working.
- ``test_handwritten_synth_through_loader_with_workers`` -- same
  as the first test but with num_workers=2, tightly bounded to
  one batch. Catches DataLoader pickle / start-method regressions.
- ``test_synth_task_ocr_collates_correctly`` -- OCR-vs-layout
  ablation collate path (--synth-task=ocr). Asserts no spatial
  tokens leak into labels and the sequence tokenises cleanly.

Coverage hole: these tests are effectively Linux-CI-only because
they require ``/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf``
(uncommon on Linux dev/CI; common on macOS). macOS-only synth
bugs (Pillow rasterisation drift, Darwin spawn semantics, HF font
cache paths) won't be caught here. If macOS bugs surface in
operator use, bundle a tiny test font in-tree and re-enable.
Decision recorded in plan §Fix 4 "Acknowledged coverage hole".
"""
from __future__ import annotations

import signal
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from vista_ocr.data.dataloader import DataLoaderConfig, make_mixed_loader
from vista_ocr.data.preprocess import PreprocessConfig
from vista_ocr.data.synth.factory import HandwrittenSynthFactory
from vista_ocr.tokenizer.build_spm import train_spm
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import (
    VistaTokenizer,
    list_special_and_spatial_tokens,
)


SYSTEM_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
SYSTEM_FONT_AVAILABLE = SYSTEM_FONT.exists()

pytestmark = pytest.mark.skipif(
    not SYSTEM_FONT_AVAILABLE,
    reason="DejaVu test font not available; Phase J e2e tests are Linux-CI-only",
)


@pytest.fixture(scope="module")
def grid() -> SpatialGrid:
    return SpatialGrid(canvas_h=128, canvas_w=128, quantizer_px=4, scheme="original")


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory, grid):
    """Mirror the small-SPM fixture from tests/test_dataloader.py."""
    tmp = tmp_path_factory.mktemp("synth_e2e_spm")
    corpus = tmp / "c.txt"
    corpus.write_text(
        "hello world\nfoo bar\nthe quick brown fox jumps\n"
        "abc def ghi jkl\nDigits 0 1 2 3 4 5 6 7 8 9\n" * 400,
        encoding="utf-8",
    )
    out = tmp / "tk"
    train_spm(
        corpus, out, vocab_size=180,
        user_symbols=list_special_and_spatial_tokens(grid),
    )
    return VistaTokenizer(out.with_suffix(".model"), grid)


@pytest.fixture
def text_corpus(tmp_path: Path) -> Path:
    p = tmp_path / "synth_corpus.txt"
    p.write_text(
        "hello world\nfoo bar baz\nthe quick brown fox jumps\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def factory(text_corpus: Path):
    """Tiny synth factory tuned to the 128x128 fixture grid: line
    heights kept small so render output fits the test canvas."""
    return HandwrittenSynthFactory(
        text_corpus_paths=[text_corpus],
        language="en",
        font_paths=[SYSTEM_FONT],
        canvas_size=(120, 120),
        task="ocr_layout",
    )


def _pre_cfg() -> PreprocessConfig:
    """Match the canvas/grid the SPM was built against."""
    return PreprocessConfig(target_h=128, target_w=128, pad_multiple=32)


@contextmanager
def _alarm_timeout(seconds: int, message: str):
    """SIGALRM-based hard timeout so a hung DataLoader cannot
    deadlock the test runner. Linux/macOS only (Windows lacks
    SIGALRM); this whole test file is Linux-CI-only anyway via
    the DejaVu skipif at module level.
    """
    def _handler(signum, frame):
        raise TimeoutError(message)
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------------------
# Pure collate invariants
# ---------------------------------------------------------------------------

def _assert_batch_invariants(batch, pad_id: int):
    assert batch.images.dim() == 4
    assert batch.images.shape[1] == 1, f"expected 1 channel, got {batch.images.shape}"
    assert batch.decoder_input_ids.dim() == 2
    assert batch.decoder_input_ids.shape == batch.labels.shape
    assert batch.decoder_input_ids.shape == batch.prompt_mask.shape
    assert batch.labels.dtype == torch.long
    assert batch.prompt_mask.dtype == torch.bool
    assert batch.pad_id == pad_id
    # At least one non-pad token AND at least one prompt position.
    assert (batch.labels != pad_id).any().item()
    assert batch.prompt_mask.any().item()


def test_handwritten_synth_through_make_mixed_loader_collates(tokenizer, factory):
    """The headline e2e test: synth Sample → MixedStream → collate
    → valid Batch. With pdfa_weight=0, synth_weight=1, every sample
    must be synth -- no source channel needed to verify that."""
    loader = make_mixed_loader(
        pdfa_shards=[],
        idl_shards=None,
        synth_factory=factory,
        pdfa_weight=0.0,
        idl_weight=0.0,
        synth_weight=1.0,
        tokenizer=tokenizer,
        pre_cfg=_pre_cfg(),
        dl_cfg=DataLoaderConfig(
            micro_batch_size=1, num_workers=0, prefetch_factor=2,
            persistent_workers=False,
        ),
    )
    batch = next(iter(loader))
    _assert_batch_invariants(batch, pad_id=tokenizer.pad_id)


def test_synth_only_supported_via_pdfa_weight_zero(tokenizer, factory):
    """Standalone test: synth-only loader (no PDFA shards at all)
    constructs and yields a Batch without raising. The dataloader
    docstring claims this is supported; this test pins the contract
    so a future regression toward 'must have PDFA' surfaces with a
    clear message."""
    loader = make_mixed_loader(
        pdfa_shards=[],
        synth_factory=factory,
        pdfa_weight=0.0,
        synth_weight=1.0,
        tokenizer=tokenizer,
        pre_cfg=_pre_cfg(),
        dl_cfg=DataLoaderConfig(
            micro_batch_size=1, num_workers=0, prefetch_factor=2,
            persistent_workers=False,
        ),
    )
    batch = next(iter(loader))
    assert batch.images.shape[0] == 1


# ---------------------------------------------------------------------------
# Multi-worker pickle path
# ---------------------------------------------------------------------------

def test_handwritten_synth_through_loader_with_workers(tokenizer, factory):
    """``num_workers=1`` exercises the full DataLoader plumbing:
    the IterableDataset (and the embedded HandwrittenSynthFactory)
    crosses the worker process boundary, ``get_worker_info()``
    returns a real per-worker slice, and one batch reaches the
    main process.

    Speed budget. Earlier versions used ``num_workers=2`` +
    ``prefetch_factor=2``, which combined with
    ``persistent_workers=False`` reliably hung at > 30 s on some
    boxes during DataLoader teardown (workers blocked on the
    queue + slow ``join()``). We narrow to a single worker with a
    single prefetched batch, then wrap the entire dance in a
    SIGALRM hard timeout so a future regression surfaces as a
    test FAIL, not as a runner deadlock.

    Linux fork-vs-spawn note: this runs under PyTorch's default
    ``fork`` start method on Linux (the only platform where this
    file's module-level skip lets it run). Fork copies process
    state without pickling, so a non-picklable closure on the
    factory would NOT surface here; the dedicated picklability
    backstop is ``test_synth_factory.test_factory_is_picklable``.
    """
    loader = make_mixed_loader(
        pdfa_shards=[],
        synth_factory=factory,
        pdfa_weight=0.0,
        synth_weight=1.0,
        tokenizer=tokenizer,
        pre_cfg=_pre_cfg(),
        dl_cfg=DataLoaderConfig(
            micro_batch_size=1, num_workers=1, prefetch_factor=1,
            persistent_workers=False,
        ),
    )
    with _alarm_timeout(20, "DataLoader fork+join exceeded 20 s; regression in worker shutdown semantics?"):
        it = iter(loader)
        try:
            batch = next(it)
            _assert_batch_invariants(batch, pad_id=tokenizer.pad_id)
        finally:
            # Force worker shutdown explicitly so the SIGALRM doesn't
            # fire on a slow GC if the assertion above raised.
            del it
            del loader


# ---------------------------------------------------------------------------
# OCR-vs-layout ablation (synth_task=ocr)
# ---------------------------------------------------------------------------

def test_synth_task_ocr_collates_correctly(tokenizer, text_corpus):
    """Synth emits task='ocr' (no spatial tokens in the target
    sequence). Catches a regression where the OCR-only path ships
    but produces malformed sequences for synth specifically.

    Asserts:
    1. NO spatial-token IDs leak into batch.labels.
    2. <unk> rate on the non-pad / non-prompt label tokens stays
       low (< 5%); a high rate would mean the synth text is being
       routed through some path that drops chars to <unk>.
    """
    factory = HandwrittenSynthFactory(
        text_corpus_paths=[text_corpus],
        language="en",
        font_paths=[SYSTEM_FONT],
        canvas_size=(120, 120),
        task="ocr",
    )
    loader = make_mixed_loader(
        pdfa_shards=[],
        synth_factory=factory,
        pdfa_weight=0.0,
        synth_weight=1.0,
        tokenizer=tokenizer,
        pre_cfg=_pre_cfg(),
        dl_cfg=DataLoaderConfig(
            micro_batch_size=1, num_workers=0, prefetch_factor=2,
            persistent_workers=False,
        ),
    )
    batch = next(iter(loader))

    # Spatial-token IDs must NOT appear in labels for task=ocr.
    spatial_ids = tokenizer._spatial_ids
    label_ids = batch.labels.flatten().tolist()
    spatial_in_labels = [tid for tid in label_ids if tid in spatial_ids]
    assert not spatial_in_labels, (
        f"spatial tokens in task=ocr labels: {spatial_in_labels[:10]}"
    )

    # <unk> rate on supervised positions (non-pad, non-prompt) is low.
    pad_id = tokenizer.pad_id
    unk_id = tokenizer.unk_id
    supervised = (batch.labels != pad_id) & (~batch.prompt_mask)
    if supervised.any():
        sup_ids = batch.labels[supervised].tolist()
        unk_count = sum(1 for tid in sup_ids if tid == unk_id)
        unk_rate = unk_count / len(sup_ids)
        # SPM is tiny + tuned to 'hello world / abc def / digits';
        # synth text uses the same vocabulary, so <unk> rate should
        # be near zero. The 5% threshold is generous; flag a
        # regression if it ever climbs.
        assert unk_rate < 0.05, f"<unk> rate too high on task=ocr labels: {unk_rate:.3f}"
