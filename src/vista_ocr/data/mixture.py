"""Multi-task / multi-source mixed sampler.

Stage-3 pretraining (PLAN §6) draws batches from four task buckets:
``ocr``, ``ocr_layout``, ``region_ocr``, ``find_it``. This sampler also
mixes multiple data *sources* with their own weights — paper appendix
mentions IDL+PDFA + synthetic data.

Stage-3 task mix carve-out for fixed-task sources
-------------------------------------------------

Stage 3 introduces all four tasks **for real-data samples**;
synthetic-handwritten samples carry their emission-time task end-
to-end and are NOT relabelled. Reason: the synth generator is
parameterised on a single task per run (the ``--synth-task`` flag,
which drives the OCR-vs-layout ablation in
``notes/plan_phase_j.md`` §10b #4). Silently overwriting that task
inside ``MixedTaskStream`` would invalidate the ablation and is the
bug Fix 1 of the Phase J followup closes.

The carve-out predicate is :func:`is_fixed_task_source`, which
checks ``meta["source_family"] == "synth_handwritten"`` (a robust
programmatic hook; ``Sample.source`` is operator-facing provenance
and could change shape without notice).

**Distributional consequence (load-bearing).** When synth is on,
the effective stage-3 task histogram shifts:

- synth off → 25 / 25 / 25 / 25 across the four tasks.
- ``SYNTH_WEIGHT=0.2``, ``SYNTH_TASK=ocr_layout`` →
  ocr_layout = 0.40, the other three = 0.20 each.
- ``SYNTH_WEIGHT=0.2``, ``SYNTH_TASK=ocr`` →
  ocr = 0.40, the other three = 0.20 each.

The post-J A/B protocol must NOT conflate "synth was on" with
"stage-3 task histogram was rebalanced." If the operator wants
synth-on-vs-off comparable independent of task-shape change, pass
non-default ``--w-*`` weights so both arms produce the same
effective task histogram.
"""
from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass, replace

from vista_ocr.data.types import Sample, TaskName

LOG = logging.getLogger(__name__)


@dataclass
class TaskMix:
    weights: dict[TaskName, float]

    def sample_task(self, rng: random.Random) -> TaskName:
        items = list(self.weights.items())
        names, ws = zip(*items, strict=False)
        return rng.choices(names, weights=ws, k=1)[0]


def is_fixed_task_source(sample: Sample) -> bool:
    """True iff this sample's ``task`` should NOT be overwritten by
    stage-3 multitask relabelling.

    Currently fires only on ``meta["source_family"] == "synth_handwritten"``.
    Future fixed-task sources can opt in by setting the same field.

    Why ``meta["source_family"]`` and not ``sample.source``: the
    ``source`` string is operator-facing provenance ("synth_handwritten:en:pg19")
    and could change shape without notice; ``meta["source_family"]``
    is a programmatic hook stable across language tags + text-source
    permutations.
    """
    return sample.meta.get("source_family") == "synth_handwritten"


def relabel_for_task(sample: Sample, task: TaskName, rng: random.Random) -> Sample:
    """Return a shallow copy of ``sample`` re-tagged with ``task`` and any
    extra fields needed by that task (random query bbox / text)."""
    if task == "region_ocr":
        if not sample.lines:
            return sample
        line = rng.choice(sample.lines)
        return replace(sample, task=task, query_bbox=line.bbox)
    if task == "find_it":
        if not sample.lines:
            return sample
        line = rng.choice(sample.lines)
        words = line.text.split()
        if not words:
            return sample
        # Paper Table 6 splits 2-5 / 5-8 / 8-11 word queries; sample uniformly.
        n = max(1, min(len(words), rng.randint(2, 11)))
        start = rng.randint(0, max(0, len(words) - n))
        query = " ".join(words[start : start + n])
        return replace(sample, task=task, query_text=query)
    return replace(sample, task=task)


class MixedTaskStream:
    """Wrap a base iterable of :class:`Sample`s and randomly relabel each
    sample with a task drawn from ``mix``."""

    def __init__(
        self,
        base: Iterator[Sample],
        mix: TaskMix,
        seed: int = 0,
    ) -> None:
        self.base = base
        self.mix = mix
        self.rng = random.Random(seed)

    def __iter__(self) -> Iterator[Sample]:
        for s in self.base:
            if is_fixed_task_source(s):
                # Carve-out for synth-handwritten (and any other
                # source that opts into fixed-task semantics via
                # meta["source_family"]). See module docstring for
                # the design + the distributional consequence.
                yield s
                continue
            task = self.mix.sample_task(self.rng)
            yield relabel_for_task(s, task, self.rng)
