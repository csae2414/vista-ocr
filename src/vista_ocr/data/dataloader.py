"""Multi-worker DataLoader for PDFA shards.

The slow CPU work in our pipeline -- PDF rasterization at 200 dpi,
deskew/rectify, tokenization, batch collation -- runs single-threaded in
the default :func:`vista_ocr.training.train_loop._iter_batches`. On the
GPU VM that left the GPU at ~50% utilisation while the CPU did the work
serially. This module wraps ``iter_pdfa`` in a
:class:`torch.utils.data.IterableDataset` and uses a standard PyTorch
:class:`~torch.utils.data.DataLoader` with workers + prefetch so the GPU
is fed continuously.

Worker sharding strategy:

- Worker ``i`` of ``N`` reads ``shards[i::N]``.
- If there's only one shard or ``num_workers == 0``, falls back to the
  single-process iterator.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from vista_ocr.data.collate import Batch, collate
from vista_ocr.data.pdfa import PdfaConfig, iter_pdfa
from vista_ocr.data.preprocess import PreprocessConfig
from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.tokenizer import VistaTokenizer

LOG = logging.getLogger(__name__)


@dataclass
class DataLoaderConfig:
    micro_batch_size: int = 1
    num_workers: int = 4
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


class _IterablePdfaDataset(IterableDataset):
    """Per-worker shard split + on-the-fly collation.

    Each worker rebuilds its own ``iter_pdfa`` over its shard subset and
    emits already-collated :class:`Batch` objects -- so the slow PDF
    render + tokenize + collate cost is paid by the worker, not the main
    GPU process."""

    def __init__(
        self,
        shards: list[str],
        tokenizer: VistaTokenizer,
        pre_cfg: PreprocessConfig,
        micro_batch_size: int,
        pdfa_cfg_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.shards = shards
        self.tokenizer = tokenizer
        self.pre_cfg = pre_cfg
        self.micro_batch_size = micro_batch_size
        self.pdfa_cfg_kwargs = pdfa_cfg_kwargs or {}

    def _shards_for_this_worker(self) -> list[str]:
        info = get_worker_info()
        if info is None:
            return list(self.shards)
        wid = info.id
        n = info.num_workers
        return self.shards[wid::n]

    def __iter__(self) -> Iterator[Batch]:
        shards = self._shards_for_this_worker()
        if not shards:
            return
        cfg = PdfaConfig(shards=shards, **self.pdfa_cfg_kwargs)
        buf: list[Sample] = []
        for sample in iter_pdfa(cfg):
            buf.append(sample)
            if len(buf) == self.micro_batch_size:
                yield collate(buf, self.tokenizer, self.pre_cfg)
                buf = []
        if buf:
            yield collate(buf, self.tokenizer, self.pre_cfg)


def _identity_collate(batches: list[Batch]) -> Batch:
    """The dataset already collates one Batch per item; the DataLoader
    just unpacks the (single) item from the worker's queue."""
    if len(batches) != 1:
        raise RuntimeError(f"Expected exactly 1 pre-collated Batch, got {len(batches)}")
    return batches[0]


def make_pdfa_dataloader(
    shards: list[str],
    tokenizer: VistaTokenizer,
    pre_cfg: PreprocessConfig,
    *,
    dl_cfg: DataLoaderConfig | None = None,
    pdfa_cfg_kwargs: dict | None = None,
) -> DataLoader:
    """Build a multi-worker DataLoader over PDFA shards yielding :class:`Batch`.

    Use this in place of ``iter_pdfa`` when training so that PDF render,
    tokenization and collate run in worker processes."""
    dl_cfg = dl_cfg or DataLoaderConfig()
    ds = _IterablePdfaDataset(
        shards=shards,
        tokenizer=tokenizer,
        pre_cfg=pre_cfg,
        micro_batch_size=dl_cfg.micro_batch_size,
        pdfa_cfg_kwargs=pdfa_cfg_kwargs,
    )
    # IterableDataset already groups into Batch objects; we want batch_size=1
    # at the DataLoader level so the queue carries one Batch per item.
    return DataLoader(
        ds,
        batch_size=1,
        collate_fn=_identity_collate,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor if dl_cfg.num_workers > 0 else None,
        pin_memory=dl_cfg.pin_memory,
        persistent_workers=dl_cfg.persistent_workers and dl_cfg.num_workers > 0,
    )


__all__ = [
    "DataLoaderConfig",
    "make_pdfa_dataloader",
]
