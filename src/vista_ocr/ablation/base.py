"""Shared :class:`Ablation` base class for the operator comparison scripts.

A subclass implements :meth:`variants` and :meth:`build`; the base
handles the ``for variant in variants: train, summarise, append`` loop
and the comparison-table print. Adding a new ablation is now a class
plus an ``argparse`` shim, not a copied skeleton.

Reproducibility: ``run()`` reseeds ``torch.manual_seed(seed)`` before
building each variant, so two ablations with the same data + same
steps produce comparable losses.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

LOG = logging.getLogger(__name__)


@dataclass
class AblationVariant:
    """Lightweight description of one ablation cell.

    * ``name``: short label printed in the comparison table.
    * ``overrides``: free-form dict the subclass's :meth:`build` reads.
    * ``extra``: arbitrary metadata propagated into the result row
      (useful for columns the base class doesn't know about, e.g.
      "spatial_token_count" for the encoding-scheme ablation).
    """

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def summarise_history(
    name: str,
    history: list,
    elapsed: float,
    *,
    window: int = 50,
) -> dict[str, Any]:
    """Extract first-``window`` / last-``window`` loss + elapsed + steps.

    Pulled out as a free function so a subclass can reuse it when
    overriding :meth:`Ablation.run_variant` for non-default reporting.
    """
    if not history:
        return {
            "name": name, "elapsed": elapsed, "n_steps": 0,
            "loss_first": float("nan"), "loss_last": float("nan"),
            "delta": float("nan"),
            "per_step": float("nan"),
        }
    first = sum(h.loss for h in history[:window]) / max(1, len(history[:window]))
    last = sum(h.loss for h in history[-window:]) / max(1, len(history[-window:]))
    return {
        "name": name, "elapsed": elapsed, "n_steps": len(history),
        "loss_first": first, "loss_last": last, "delta": first - last,
        "per_step": elapsed / max(1, len(history)),
    }


class Ablation(ABC):
    """Common loop for the comparison scripts.

    Subclass contract:
      * :meth:`variants` returns the cells to run, in display order.
      * :meth:`build` takes one variant and returns ``(model,
        tokenizer, train_cfg, sample_stream)`` -- the four arguments
        :func:`vista_ocr.training.train_loop.train` needs.
    """

    seed: int = 0
    summary_window: int = 50

    @abstractmethod
    def variants(self) -> list[AblationVariant]: ...

    @abstractmethod
    def build(self, variant: AblationVariant):
        """Return ``(model, tokenizer, train_cfg, sample_stream_iterable)``."""

    # ------------------------------------------------------------------

    def _train_one(self, variant: AblationVariant, *, max_steps: int) -> dict[str, Any]:
        from vista_ocr.training.train_loop import train

        torch.manual_seed(self.seed)
        model, tokenizer, cfg, sample_stream = self.build(variant)
        t0 = time.perf_counter()
        history = train(
            model=model, sample_stream=iter(sample_stream),
            tokenizer=tokenizer, cfg=cfg, max_steps=max_steps,
        )
        elapsed = time.perf_counter() - t0
        row = summarise_history(
            variant.name, history, elapsed, window=self.summary_window,
        )
        row.update(variant.extra)
        return row

    def run(self, *, max_steps: int) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        variants = self.variants()
        for v in variants:
            LOG.info("=" * 60)
            LOG.info("Variant: %s", v.name)
            results.append(self._train_one(v, max_steps=max_steps))
        return results

    # ------------------------------------------------------------------

    @staticmethod
    def report(
        results: list[dict[str, Any]],
        *,
        columns: list[str] | None = None,
        widths: dict[str, int] | None = None,
        log: logging.Logger = LOG,
    ) -> None:
        """Print a single comparison table.

        Caller picks the columns; the base only knows how to format the
        always-present rows (``name``, ``loss_first``, ``loss_last``,
        ``delta``, ``per_step``, ``elapsed``, ``n_steps``).
        """
        cols = columns or [
            "name", "n_steps", "loss_first", "loss_last", "delta",
            "per_step", "elapsed",
        ]
        widths = widths or {}
        default_w = 12
        log.info("=" * 60)
        header = "".join(c.ljust(widths.get(c, default_w)) for c in cols)
        log.info(header)
        for r in results:
            cells: list[str] = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, float):
                    cells.append(f"{v:.3f}".ljust(widths.get(c, default_w)))
                elif isinstance(v, int):
                    cells.append(str(v).ljust(widths.get(c, default_w)))
                else:
                    cells.append(str(v).ljust(widths.get(c, default_w)))
            log.info("".join(cells))
