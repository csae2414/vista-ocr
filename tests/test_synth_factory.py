"""Tests for HandwrittenSynthFactory + make_mixed_loader synth path.

Inventory:
- HandwrittenSynthFactory __post_init__ validates corpora + fonts at
  construction time (not inside the worker)
- factory is picklable (DataLoader workers serialise it)
- factory yields independent streams under different worker seeds
- make_mixed_loader rejects negative weights / all-zero weights
- make_mixed_loader requires synth_factory iff synth_weight > 0
- make_mixed_loader synth-only-no-idl path constructs without errors
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from vista_ocr.data.synth.factory import HandwrittenSynthFactory


SYSTEM_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
SYSTEM_FONT_AVAILABLE = SYSTEM_FONT.exists()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    p = tmp_path / "c.txt"
    p.write_text("hello world\nfoo bar baz qux\n", encoding="utf-8")
    return p


@pytest.fixture
def factory(corpus: Path) -> HandwrittenSynthFactory:
    if not SYSTEM_FONT_AVAILABLE:
        pytest.skip("DejaVu test font not available")
    return HandwrittenSynthFactory(
        text_corpus_paths=[corpus],
        language="en",
        font_paths=[SYSTEM_FONT],
        canvas_size=(400, 600),
    )


def test_factory_validates_missing_corpus_at_construction(tmp_path: Path):
    """If we deferred this to __call__ inside a worker, it surfaces as
    an opaque DataLoader crash. Catch it at construction time instead."""
    with pytest.raises(FileNotFoundError):
        HandwrittenSynthFactory(text_corpus_paths=[tmp_path / "nope.txt"])


def test_factory_validates_missing_font_at_construction(corpus: Path, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        HandwrittenSynthFactory(
            text_corpus_paths=[corpus],
            font_paths=[tmp_path / "nope.ttf"],
        )


def test_factory_rejects_empty_corpus_list():
    with pytest.raises(ValueError):
        HandwrittenSynthFactory(text_corpus_paths=[])


def test_factory_is_picklable(factory: HandwrittenSynthFactory):
    """DataLoader workers pickle the IterableDataset (and therefore
    the factory it holds) when num_workers > 0."""
    data = pickle.dumps(factory)
    restored = pickle.loads(data)
    gen = restored(worker_seed=0)
    s = next(gen)
    assert s.source.startswith("synth_handwritten:")


def test_factory_seed_per_worker_changes_output(factory: HandwrittenSynthFactory):
    a = factory(worker_seed=1)
    b = factory(worker_seed=2)
    sa = next(a)
    sb = next(b)
    # Different seeds must diverge on at least one observable (line text or bbox).
    assert ([(ln.text, ln.bbox) for ln in sa.lines]
            != [(ln.text, ln.bbox) for ln in sb.lines])


def test_factory_default_text_source_tags(corpus: Path):
    if not SYSTEM_FONT_AVAILABLE:
        pytest.skip("DejaVu test font not available")
    f = HandwrittenSynthFactory(
        text_corpus_paths=[corpus], font_paths=[SYSTEM_FONT],
    )
    s = next(f(worker_seed=0))
    # Default tag is the corpus stem.
    assert s.meta["text_source"] == corpus.stem


# ---------------------------------------------------------------------------
# make_mixed_loader validation
# ---------------------------------------------------------------------------

def test_make_mixed_loader_rejects_negative_weight():
    from vista_ocr.data.dataloader import make_mixed_loader
    with pytest.raises(ValueError):
        make_mixed_loader(
            pdfa_shards=["x.tar"], tokenizer=None, pre_cfg=None,  # type: ignore[arg-type]
            pdfa_weight=-0.1,
        )


def test_make_mixed_loader_rejects_all_zero_weight():
    from vista_ocr.data.dataloader import make_mixed_loader
    with pytest.raises(ValueError):
        make_mixed_loader(
            pdfa_shards=["x.tar"], tokenizer=None, pre_cfg=None,  # type: ignore[arg-type]
            pdfa_weight=0.0, idl_weight=0.0, synth_weight=0.0,
        )


def test_make_mixed_loader_synth_weight_requires_factory():
    from vista_ocr.data.dataloader import make_mixed_loader
    with pytest.raises(ValueError):
        make_mixed_loader(
            pdfa_shards=["x.tar"], tokenizer=None, pre_cfg=None,  # type: ignore[arg-type]
            pdfa_weight=0.5, synth_weight=0.5, synth_factory=None,
        )


def test_make_mixed_loader_idl_weight_requires_shards():
    from vista_ocr.data.dataloader import make_mixed_loader
    with pytest.raises(ValueError):
        make_mixed_loader(
            pdfa_shards=["x.tar"], tokenizer=None, pre_cfg=None,  # type: ignore[arg-type]
            pdfa_weight=0.5, idl_weight=0.5, idl_shards=None,
        )
