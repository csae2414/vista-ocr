"""Picklable synth-source factories for use inside DataLoader workers.

The DataLoader pickles its IterableDataset across worker boundaries.
A bare ``HandwrittenLineSynth`` instance is not picklable (RNG state
+ pre-loaded text-source contents), so we keep the *configuration*
on a dataclass and instantiate the generator inside each worker via
``__call__``. Each worker's call passes its own seed so the synth
streams are independent across workers.

The factory is the seam between the train-loop's "sample stream"
abstraction and the synth generator's "next page" loop. It does
NOT mix; mixing happens in :mod:`vista_ocr.data.dataloader` via
``MixedStream``.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from vista_ocr.data.synth.handwritten import (
    HandwrittenLineSynth,
    HandwrittenLineSynthConfig,
    TextSource,
)
from vista_ocr.data.types import Sample


@dataclass
class HandwrittenSynthFactory:
    """Build a :class:`HandwrittenLineSynth` iterator inside a worker.

    Held as ``synth_factory`` on the mixed dataloader's IterableDataset;
    the dataloader serialises this dataclass into the worker, which
    then calls it with a worker-local seed.

    :param text_corpus_paths: One or more local UTF-8 corpus files.
        Sampled uniformly at runtime.
    :param language: ``"en"`` or ``"fr"``.
    :param font_paths: Explicit list of TTF paths. If ``None``, the
        generator falls back to the packaged default font dir.
    :param task: ``"ocr_layout"`` (default) or ``"ocr"`` (the
        OCR-vs-layout ablation, see notes/plan_phase_j.md §10b #4).
    :param canvas_size: ``(H, W)``. Must match the page preset the
        rest of the pipeline uses, otherwise the rendered pages get
        rescaled by ``resize_to_canvas`` and the bboxes shift.
    """

    text_corpus_paths: list[Path]
    language: str = "en"
    font_paths: list[Path] | None = None
    task: str = "ocr_layout"
    canvas_size: tuple[int, int] = (1050, 1400)
    text_source_mode: str = "sentence"
    text_source_tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.text_corpus_paths = [Path(p) for p in self.text_corpus_paths]
        if not self.text_corpus_paths:
            raise ValueError("HandwrittenSynthFactory: text_corpus_paths empty")
        if not self.text_source_tags:
            self.text_source_tags = [p.stem for p in self.text_corpus_paths]
        if len(self.text_source_tags) != len(self.text_corpus_paths):
            raise ValueError("text_source_tags must align with text_corpus_paths")
        # Fail loud at construction time, NOT on first __call__ inside
        # a worker (a worker-side FileNotFoundError surfaces as an
        # opaque DataLoader crash).
        for p in self.text_corpus_paths:
            if not p.exists():
                raise FileNotFoundError(f"synth corpus path does not exist: {p}")
        if self.font_paths is not None:
            self.font_paths = [Path(p) for p in self.font_paths]
            for p in self.font_paths:
                if not p.exists():
                    raise FileNotFoundError(f"synth font path does not exist: {p}")

    def __call__(self, worker_seed: int) -> Iterator[Sample]:
        sources = [
            TextSource(path=p, mode=self.text_source_mode, tag=tag)
            for p, tag in zip(self.text_corpus_paths, self.text_source_tags, strict=True)
        ]
        cfg = HandwrittenLineSynthConfig(
            language=self.language,
            font_paths=self.font_paths,
            canvas_size=self.canvas_size,
            task=self.task,                # type: ignore[arg-type]
        )
        return iter(HandwrittenLineSynth(sources, cfg, seed=worker_seed))
