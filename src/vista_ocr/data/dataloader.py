"""Multi-worker DataLoader for PDFA shards (and PDFA + IDL mixtures).

The slow CPU work in our pipeline -- PDF rasterization at 200 dpi,
deskew/rectify, tokenization, batch collation -- runs single-threaded in
the default :func:`vista_ocr.training.train_loop._iter_batches`. On the
GPU VM that left the GPU at ~50% utilisation while the CPU did the work
serially. This module wraps ``iter_pdfa`` in a
:class:`torch.utils.data.IterableDataset` and uses a standard PyTorch
:class:`~torch.utils.data.DataLoader` with workers + prefetch so the GPU
is fed continuously.

Worker sharding strategy
------------------------

The PDFA-only loader (:class:`_IterablePdfaDataset`) gives worker ``i``
of ``N`` the slice ``shards[i::N]``. When there's only one shard or
``num_workers == 0``, falls back to the single-process iterator.

The mixed PDFA + IDL loader (:class:`_IterableMixedDataset`) uses the
same per-source slice but adds an explicit policy for the case when
one source has fewer shards than ``num_workers`` -- e.g. 12 IDL shards
across 16 workers. Without intervention, the high-id workers see an
empty IDL slice and fall back to PDFA-only, which **silently distorts
the global mix ratio**.

Two policies are supported via the ``on_short_shards`` flag on
:func:`make_mixed_pdfa_idl_loader`:

- ``"broadcast"`` (default): every worker reads ALL shards of the
  under-sourced side. Per-worker prime-hashed seeds keep their sample
  draws independent. Throughput is preserved, but each unique sample
  appears ``N`` times across the worker pool per pass; gradient noise
  reduction is altered relative to a non-broadcast setup. A WARNING
  is logged when this kicks in.
- ``"cap"``: silently downgrade ``num_workers`` for the affected
  source. Workers above the shard count get an empty slice and yield
  nothing. Preserves "each shard read once per worker per epoch"
  semantics at the cost of pipeline parallelism.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from vista_ocr.data.collate import Batch, collate
from vista_ocr.data.idl import IdlConfig, iter_idl
from vista_ocr.data.mixture_stream import MixedStream, MixedStreamSource
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


def _per_worker_cycle_seed(base_seed: int, worker_id: int) -> int:
    """Prime-hashed per-worker seed.

    A naive ``base_seed + worker_id`` produces consecutive integers
    that some downstream RNGs treat as correlated draws. Two large
    primes mixed in give workers seeds that look independent under
    any reasonable ``random.Random(seed)``-style consumer.
    """
    # 100003 and 31 are both prime; the offset by ``base_seed`` is
    # mixed in *first* so callers can still reseed an entire run by
    # changing only the base.
    return (base_seed * 100003) ^ (worker_id * 31)


def _slice_with_policy(
    shards: list[str],
    *,
    worker_id: int | None,
    num_workers: int | None,
    on_short_shards: str,
) -> tuple[list[str], bool]:
    """Compute one worker's shard slice under the chosen short-shards policy.

    Returns ``(shards_for_this_worker, did_broadcast)``.

    Single-process (``worker_id is None``) always returns the full
    list. The interesting case is ``len(shards) < num_workers``:

    * ``broadcast``: every worker reads ALL shards.
    * ``cap``: workers with ``id >= len(shards)`` get an empty list.
    """
    if worker_id is None or num_workers is None:
        return list(shards), False
    if not shards:
        return [], False
    if len(shards) >= num_workers:
        return shards[worker_id::num_workers], False
    if on_short_shards == "broadcast":
        return list(shards), True
    if on_short_shards == "cap":
        if worker_id >= len(shards):
            return [], False
        return shards[worker_id::num_workers], False
    raise ValueError(
        f"on_short_shards must be 'broadcast' or 'cap', got {on_short_shards!r}",
    )


class _IterableMixedDataset(IterableDataset):
    """Per-worker mixed-source iterator yielding pre-collated batches.

    Mirrors :class:`_IterablePdfaDataset` but draws from two sources
    (PDFA + IDL) via :class:`MixedStream`. Each worker gets a slice of
    BOTH shard lists; mixing happens at the per-worker level so the
    weight ratio is preserved within each worker's stream.

    See module docstring for the ``on_short_shards`` policy that
    governs what happens when one source has fewer shards than
    ``num_workers``.
    """

    def __init__(
        self,
        pdfa_shards: list[str],
        idl_shards: list[str],
        pdfa_weight: float,
        idl_weight: float,
        tokenizer: VistaTokenizer,
        pre_cfg: PreprocessConfig,
        micro_batch_size: int,
        pdfa_cfg_kwargs: dict | None = None,
        idl_cfg_kwargs: dict | None = None,
        seed: int = 0,
        on_short_shards: str = "broadcast",
        synth_factory: Callable[[int], Iterator[Sample]] | None = None,
        synth_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.pdfa_shards = pdfa_shards
        self.idl_shards = idl_shards
        self.pdfa_weight = pdfa_weight
        self.idl_weight = idl_weight
        self.tokenizer = tokenizer
        self.pre_cfg = pre_cfg
        self.micro_batch_size = micro_batch_size
        self.pdfa_cfg_kwargs = pdfa_cfg_kwargs or {}
        self.idl_cfg_kwargs = idl_cfg_kwargs or {}
        self.seed = seed
        self.on_short_shards = on_short_shards
        self.synth_factory = synth_factory
        self.synth_weight = synth_weight

    def __iter__(self) -> Iterator[Batch]:
        info = get_worker_info()
        wid = info.id if info is not None else None
        nworkers = info.num_workers if info is not None else None

        pdfa_slice, pdfa_bcast = _slice_with_policy(
            self.pdfa_shards, worker_id=wid, num_workers=nworkers,
            on_short_shards=self.on_short_shards,
        )
        idl_slice, idl_bcast = _slice_with_policy(
            self.idl_shards, worker_id=wid, num_workers=nworkers,
            on_short_shards=self.on_short_shards,
        )

        if (pdfa_bcast or idl_bcast) and (wid == 0 or wid is None):
            # Log once per epoch (only worker 0 emits) so the operator
            # sees the gradient-noise consequence is in effect.
            LOG.warning(
                "Mixed loader: num_workers (%d) exceeds shard count for "
                "an under-sourced side (pdfa=%d, idl=%d). Broadcasting "
                "the under-sourced shards to all workers; effective "
                "gradient noise structure differs from a non-broadcast "
                "setup. Consider reducing num_workers or downloading "
                "more shards if this matters for your experiment.",
                nworkers, len(self.pdfa_shards), len(self.idl_shards),
            )

        worker_seed = (
            _per_worker_cycle_seed(self.seed, wid) if wid is not None
            else self.seed
        )

        def _maybe_per_worker_seed(cfg_kwargs: dict, broadcast: bool) -> dict:
            """When broadcasting, override cycle_seed so workers reading
            the same shard list don't yield identical sequences. The
            non-broadcast slice path keeps the caller's cycle_seed."""
            if not broadcast:
                return cfg_kwargs
            out = dict(cfg_kwargs)
            out["cycle_seed"] = worker_seed
            return out

        sources: list[MixedStreamSource] = []
        if pdfa_slice and self.pdfa_weight > 0:
            pdfa_cfg = PdfaConfig(
                shards=pdfa_slice,
                **_maybe_per_worker_seed(self.pdfa_cfg_kwargs, pdfa_bcast),
            )
            sources.append(MixedStreamSource(
                name="pdfa", weight=self.pdfa_weight, stream=iter_pdfa(pdfa_cfg),
            ))
        if idl_slice and self.idl_weight > 0:
            idl_cfg = IdlConfig(
                shards=idl_slice,
                **_maybe_per_worker_seed(self.idl_cfg_kwargs, idl_bcast),
            )
            sources.append(MixedStreamSource(
                name="idl", weight=self.idl_weight, stream=iter_idl(idl_cfg),
            ))
        if self.synth_factory is not None and self.synth_weight > 0:
            sources.append(MixedStreamSource(
                name="synth", weight=self.synth_weight,
                stream=self.synth_factory(worker_seed),
            ))
        if not sources:
            return
        mix = MixedStream(sources, seed=worker_seed)

        buf: list[Sample] = []
        for sample in mix:
            buf.append(sample)
            if len(buf) == self.micro_batch_size:
                yield collate(buf, self.tokenizer, self.pre_cfg)
                buf = []
        if buf:
            yield collate(buf, self.tokenizer, self.pre_cfg)


def make_mixed_pdfa_idl_loader(
    pdfa_shards: list[str],
    idl_shards: list[str],
    tokenizer: VistaTokenizer,
    pre_cfg: PreprocessConfig,
    *,
    pdfa_weight: float = 0.7,
    idl_weight: float = 0.3,
    dl_cfg: DataLoaderConfig | None = None,
    pdfa_cfg_kwargs: dict | None = None,
    idl_cfg_kwargs: dict | None = None,
    seed: int = 0,
    on_short_shards: str = "broadcast",
) -> DataLoader:
    """Multi-worker DataLoader over a weighted PDFA + IDL mixture.

    Default weights ``(0.7, 0.3)`` follow paper §3.4 ("real-world
    distribution skew toward clean PDFA"). Pass ``pdfa_cfg_kwargs={
    'cycle': True}`` (and same for ``idl_cfg_kwargs``) for long
    training runs that would otherwise exhaust one source mid-mix
    and silently collapse to the other.

    :param on_short_shards: policy when one source has fewer shards
        than ``num_workers``.

        * ``"broadcast"`` (default): every worker reads ALL shards of
          the under-sourced side. Throughput preserved; per-worker
          seed differentiation handled internally; one WARNING line
          logged per epoch.
        * ``"cap"``: workers above the shard count get an empty slice
          for that source. Preserves the "each shard read once per
          worker per epoch" semantics at the cost of pipeline
          parallelism.

        See the module docstring for the trade-off.
    """
    dl_cfg = dl_cfg or DataLoaderConfig()
    ds = _IterableMixedDataset(
        pdfa_shards=pdfa_shards,
        idl_shards=idl_shards,
        pdfa_weight=pdfa_weight,
        idl_weight=idl_weight,
        tokenizer=tokenizer,
        pre_cfg=pre_cfg,
        micro_batch_size=dl_cfg.micro_batch_size,
        pdfa_cfg_kwargs=pdfa_cfg_kwargs,
        idl_cfg_kwargs=idl_cfg_kwargs,
        seed=seed,
        on_short_shards=on_short_shards,
    )
    return DataLoader(
        ds,
        batch_size=1,
        collate_fn=_identity_collate,
        num_workers=dl_cfg.num_workers,
        prefetch_factor=dl_cfg.prefetch_factor if dl_cfg.num_workers > 0 else None,
        pin_memory=dl_cfg.pin_memory,
        persistent_workers=dl_cfg.persistent_workers and dl_cfg.num_workers > 0,
    )


def make_mixed_loader(
    pdfa_shards: list[str],
    tokenizer: VistaTokenizer,
    pre_cfg: PreprocessConfig,
    *,
    idl_shards: list[str] | None = None,
    synth_factory: Callable[[int], Iterator[Sample]] | None = None,
    pdfa_weight: float = 1.0,
    idl_weight: float = 0.0,
    synth_weight: float = 0.0,
    dl_cfg: DataLoaderConfig | None = None,
    pdfa_cfg_kwargs: dict | None = None,
    idl_cfg_kwargs: dict | None = None,
    seed: int = 0,
    on_short_shards: str = "broadcast",
) -> DataLoader:
    """4-mode mixed loader: ``pdfa | pdfa+idl | pdfa+synth | pdfa+idl+synth``.

    Sources with weight 0 are dropped. PDFA is always present; the
    other two are optional. Validates that weights are non-negative
    and sum > 0.

    :param synth_factory: Picklable callable taking ``worker_seed`` and
        returning a ``Iterator[Sample]``. Use
        :class:`vista_ocr.data.synth.factory.HandwrittenSynthFactory`.
    """
    if pdfa_weight < 0 or idl_weight < 0 or synth_weight < 0:
        raise ValueError(
            f"weights must be non-negative; got pdfa={pdfa_weight}, "
            f"idl={idl_weight}, synth={synth_weight}"
        )
    if pdfa_weight + idl_weight + synth_weight <= 0:
        raise ValueError("at least one source must have positive weight")
    if idl_weight > 0 and not idl_shards:
        raise ValueError("idl_weight > 0 but idl_shards is empty/None")
    if synth_weight > 0 and synth_factory is None:
        raise ValueError("synth_weight > 0 but synth_factory is None")

    dl_cfg = dl_cfg or DataLoaderConfig()
    ds = _IterableMixedDataset(
        pdfa_shards=pdfa_shards,
        idl_shards=list(idl_shards or []),
        pdfa_weight=pdfa_weight,
        idl_weight=idl_weight,
        tokenizer=tokenizer,
        pre_cfg=pre_cfg,
        micro_batch_size=dl_cfg.micro_batch_size,
        pdfa_cfg_kwargs=pdfa_cfg_kwargs,
        idl_cfg_kwargs=idl_cfg_kwargs,
        seed=seed,
        on_short_shards=on_short_shards,
        synth_factory=synth_factory,
        synth_weight=synth_weight,
    )
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
    "make_mixed_pdfa_idl_loader",
    "make_mixed_loader",
]
