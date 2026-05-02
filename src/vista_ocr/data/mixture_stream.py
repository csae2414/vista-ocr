"""Weighted mixture across multiple sample iterators.

VISTA-OCR's pretrain blend is roughly 120 K synthetic + 170 K real
samples. We don't pre-shuffle and pre-mix on disk; instead, this
class draws each next sample from one of N source iterators
according to a weight vector. Reasonable for streaming setups
(WebDataset PDFA, IDL, on-the-fly synthesis).

This is **not** a replacement for
:class:`vista_ocr.data.mixture.TaskMix` (which decides the *task*
within a single sample). They compose: ``MixedStream`` decides which
*dataset* the sample comes from; ``TaskMix`` decides which *task*
the model should solve on that sample.

Exhaustion semantics
--------------------

An exhausted source is removed from the rotation. The remaining
sources are renormalised. When all sources are exhausted, the
iterator stops cleanly. Rationale: WebDataset shards are typically
re-iterated by the user via outer loops, so per-iteration exhaustion
is a real edge that should not crash a training run.

Why hand-rolled vs. ``torch.utils.data.WeightedRandomSampler``
--------------------------------------------------------------

That sampler picks indices into a finite dataset. We sample
*iterators* yielding different shapes from different IO paths
(WebDataset, generators). Different problem.
"""
from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

LOG = logging.getLogger(__name__)


@dataclass
class MixedStreamSource:
    """One source in the mixture: a label, an iterable, and a weight.

    The weight is relative; the sampler normalises across all live
    sources at every draw.
    """

    name: str
    weight: float
    stream: Iterable[Any]


def _validate_sources(sources: list[MixedStreamSource]) -> None:
    if not sources:
        raise ValueError("MixedStream requires at least one source")
    if any(s.weight < 0 for s in sources):
        raise ValueError("MixedStream weights must be non-negative")
    if all(s.weight == 0 for s in sources):
        raise ValueError("At least one source must have a positive weight")
    names = [s.name for s in sources]
    if len(set(names)) != len(names):
        raise ValueError(f"MixedStream source names must be unique: {names}")


class MixedStream:
    """Draw samples from multiple iterators by weighted random choice.

    :param sources: list of :class:`MixedStreamSource`. Each ``stream``
        is treated as a one-shot iterable; the class wraps it in
        ``iter(...)`` once and consumes it.
    :param seed: integer seed for reproducibility. Two ``MixedStream``
        instances built with the same sources, same seed, and the
        sources yielding the same items in the same order produce
        identical sample sequences.
    """

    def __init__(
        self,
        sources: list[MixedStreamSource],
        *,
        seed: int = 0,
    ) -> None:
        _validate_sources(sources)
        self._iterators: dict[str, Iterator[Any]] = {
            s.name: iter(s.stream) for s in sources
        }
        self._weights: dict[str, float] = {s.name: float(s.weight) for s in sources}
        self._rng = random.Random(seed)
        self._counts: dict[str, int] = {s.name: 0 for s in sources}

    @property
    def counts(self) -> dict[str, int]:
        """Per-source sample counts emitted so far. Useful for
        end-of-epoch logs / sanity checks."""
        return dict(self._counts)

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        while self._iterators:
            names = [n for n, w in self._weights.items()
                     if w > 0 and n in self._iterators]
            if not names:
                raise StopIteration
            weights = [self._weights[n] for n in names]
            chosen = self._rng.choices(names, weights=weights, k=1)[0]
            try:
                item = next(self._iterators[chosen])
                self._counts[chosen] += 1
                return item
            except StopIteration:
                LOG.info("MixedStream: source %r exhausted (after %d samples); "
                         "removing from rotation", chosen, self._counts[chosen])
                del self._iterators[chosen]
        raise StopIteration


__all__ = ["MixedStream", "MixedStreamSource"]
