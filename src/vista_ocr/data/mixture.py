"""Multi-task / multi-source mixed sampler.

Stage-3 pretraining (PLAN §6) draws batches from four task buckets:
``ocr``, ``ocr_layout``, ``region_ocr``, ``find_it``. This sampler also
mixes multiple data *sources* with their own weights — paper appendix
mentions IDL+PDFA + synthetic data.
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
        names, ws = zip(*items)
        return rng.choices(names, weights=ws, k=1)[0]


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
            task = self.mix.sample_task(self.rng)
            yield relabel_for_task(s, task, self.rng)
