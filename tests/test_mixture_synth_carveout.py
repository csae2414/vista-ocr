"""Tests for the synth-task carve-out in MixedTaskStream (Fix 1).

Inventory (notes/plan_phase_j_followup.md §Fix 1):

- ``test_mixedtaskstream_preserves_synth_task`` -- synth-tagged
  samples retain their emission-time task regardless of TaskMix
  weights.
- ``test_mixedtaskstream_relabels_pdfa_normally`` -- NEGATIVE case:
  non-synth samples STILL go through the four-task relabelling.
  Without this, the test would only prove synth preservation, not
  that the four-task mix still works for real data.
- ``test_is_fixed_task_source_helper`` -- direct truth-table unit
  test of the helper.

The carve-out exists because the synth generator is parameterised
on a single task per run (--synth-task drives the OCR-vs-layout
ablation in plan §10b #4). Silently relabelling synth would
invalidate that ablation. See mixture.py module docstring for the
distributional consequence.
"""
from __future__ import annotations

from PIL import Image

from vista_ocr.data.mixture import (
    MixedTaskStream,
    TaskMix,
    is_fixed_task_source,
)
from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.tokenizer import Line


def _synth_sample(task: str = "ocr") -> Sample:
    img = Image.new("L", (200, 100), 255)
    return Sample(
        image=img,
        lines=[Line(text="hello world", bbox=(10, 10, 100, 40))],
        task=task,           # type: ignore[arg-type]
        source=f"synth_handwritten:en:fake:{task}",
        meta={"source_family": "synth_handwritten",
              "language": "en",
              "font": "fake.ttf",
              "text_source": "fake"},
    )


def _pdfa_sample(task: str = "ocr_layout") -> Sample:
    img = Image.new("L", (200, 100), 255)
    return Sample(
        image=img,
        lines=[Line(text="hello world", bbox=(10, 10, 100, 40))],
        task=task,           # type: ignore[arg-type]
        source="pdfa:fake",
        meta={},
    )


# ---------------------------------------------------------------------------
# is_fixed_task_source helper
# ---------------------------------------------------------------------------

def test_is_fixed_task_source_helper_truth_table():
    assert is_fixed_task_source(_synth_sample("ocr")) is True
    assert is_fixed_task_source(_synth_sample("ocr_layout")) is True
    assert is_fixed_task_source(_pdfa_sample()) is False
    # Sample with empty meta dict (default for non-synth iter_*).
    s_no_meta = Sample(
        image=Image.new("L", (10, 10), 255),
        lines=[],
        task="ocr_layout",
        source="other:fake",
        meta={},
    )
    assert is_fixed_task_source(s_no_meta) is False
    # Some other future fixed-source family that hasn't opted in.
    s_other_family = Sample(
        image=Image.new("L", (10, 10), 255),
        lines=[],
        task="ocr_layout",
        source="other:fake",
        meta={"source_family": "something_else"},
    )
    assert is_fixed_task_source(s_other_family) is False


# ---------------------------------------------------------------------------
# MixedTaskStream behaviour
# ---------------------------------------------------------------------------

def test_mixedtaskstream_preserves_synth_task():
    """Synth-tagged samples must retain their emission-time task
    regardless of the TaskMix weights — the load-bearing fix."""
    base = [_synth_sample("ocr") for _ in range(20)]
    # Mix that, if applied, would assign find_it/region_ocr most of
    # the time. The carve-out must dodge it.
    mix = TaskMix(weights={"ocr": 0.0, "ocr_layout": 0.0,
                           "region_ocr": 0.5, "find_it": 0.5})
    stream = MixedTaskStream(iter(base), mix, seed=0)
    out = list(stream)
    assert len(out) == 20
    for s in out:
        assert s.task == "ocr", \
            f"synth sample relabelled to {s.task}; carve-out broken"
        # Provenance preserved.
        assert s.source.startswith("synth_handwritten:")
        # Synth samples have query_bbox=None / query_text=None;
        # relabel_for_task to region_ocr/find_it would have set them.
        assert s.query_bbox is None
        assert s.query_text is None


def test_mixedtaskstream_preserves_synth_task_layout_too():
    """Same as above but with task='ocr_layout' (the default
    synth emission). Belt-and-braces."""
    base = [_synth_sample("ocr_layout") for _ in range(10)]
    mix = TaskMix(weights={"ocr": 1.0, "ocr_layout": 0.0,
                           "region_ocr": 0.0, "find_it": 0.0})
    out = list(MixedTaskStream(iter(base), mix, seed=0))
    for s in out:
        assert s.task == "ocr_layout"


def test_mixedtaskstream_relabels_pdfa_normally():
    """NEGATIVE CASE: non-synth (PDFA-shape) samples MUST still go
    through the four-task relabelling. Without this assertion the
    carve-out test only proves synth preservation, not that the
    rest of stage 3's task mix still works."""
    base = [_pdfa_sample("ocr_layout") for _ in range(200)]
    # Force everything to ocr so the assertion is unambiguous.
    mix = TaskMix(weights={"ocr": 1.0, "ocr_layout": 0.0,
                           "region_ocr": 0.0, "find_it": 0.0})
    out = list(MixedTaskStream(iter(base), mix, seed=0))
    assert len(out) == 200
    # Every PDFA sample relabelled to ocr.
    assert all(s.task == "ocr" for s in out)
    # Meta untouched (relabel_for_task should not synthesise meta).
    assert all(s.meta == {} for s in out)


def test_mixedtaskstream_distributes_across_four_tasks_for_pdfa():
    """Slightly stronger negative case: with uniform weights and
    enough samples, all four tasks should appear on PDFA samples."""
    base = [_pdfa_sample("ocr_layout") for _ in range(1000)]
    mix = TaskMix(weights={"ocr": 0.25, "ocr_layout": 0.25,
                           "region_ocr": 0.25, "find_it": 0.25})
    out = list(MixedTaskStream(iter(base), mix, seed=0))
    seen = {s.task for s in out}
    # All four tasks visible at n=1000 with seed=0 + uniform weights.
    assert seen == {"ocr", "ocr_layout", "region_ocr", "find_it"}


def test_mixedtaskstream_mixed_stream_carves_out_only_synth():
    """End-to-end: feed an interleaved stream (synth, pdfa, synth,
    pdfa, …) and assert each item is treated according to its
    family."""
    base = []
    for _ in range(50):
        base.append(_synth_sample("ocr"))
        base.append(_pdfa_sample("ocr_layout"))
    mix = TaskMix(weights={"ocr": 0.0, "ocr_layout": 0.0,
                           "region_ocr": 1.0, "find_it": 0.0})
    out = list(MixedTaskStream(iter(base), mix, seed=0))
    synth_out = [s for s in out if s.source.startswith("synth_handwritten:")]
    pdfa_out  = [s for s in out if s.source == "pdfa:fake"]
    assert len(synth_out) == 50
    assert len(pdfa_out)  == 50
    # Synth: emission-time task preserved.
    assert all(s.task == "ocr" for s in synth_out)
    # PDFA: relabelled to region_ocr per the mix.
    assert all(s.task == "region_ocr" for s in pdfa_out)
