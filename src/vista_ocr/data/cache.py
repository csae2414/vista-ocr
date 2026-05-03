"""On-disk sample cache: pre-resized image (PNG) + per-sample JSON.

Many of our data sources (PDFA via pypdfium2, future PageXML / Alto
loaders) pay a non-trivial CPU cost on every epoch to render or parse
the source format. For long training runs at fixed resolution that
cost dominates the GPU's wait time. This module materialises samples
as a fixed-resolution image cache so the hot loop becomes pure I/O.

Cache layout
------------

::

    <cache_dir>/
        manifest.json           -- (target_h, target_w, dpi, source, ...)
        sample_<N>.png          -- pre-resized grayscale image
        sample_<N>.json         -- {text, bbox, task, source} per sample

Geometry binding
----------------

The cache is tied to a single ``(target_h, target_w, dpi,
score_threshold, source)``. ``open_cache`` raises a hard error on
mismatch. To use a different preset, rebuild the cache.

Atomic write + resume
---------------------

:class:`CacheWriter` writes to ``.tmp`` files and renames atomically.
:meth:`CacheWriter.next_index` resumes from the highest existing
sample index so a partial render can be re-launched without losing
work.

Producers
---------

This module is **format-agnostic**. To populate it from a new source,
write a small loop that produces ``Sample`` objects and feeds them
through :meth:`CacheWriter.write`. PDFA + IDL adapters live in the
caller (``scripts/cache_dataset.py``).
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from vista_ocr.data.types import Line, Sample

LOG = logging.getLogger(__name__)


_MANIFEST_NAME = "manifest.json"


@dataclass
class CacheManifest:
    """The geometry/source binding of a cache. Two caches built with
    different fields here are NOT interchangeable.

    Geometry fields (``target_h``, ``target_w``, ``dpi``,
    ``score_threshold``, ``source``) are checked on load with HARD
    error.  ``loader_version`` and ``vista_ocr_commit`` are recorded
    for forensics but only WARN on mismatch -- caching semantics may
    change without invalidating geometry.
    """

    target_h: int
    target_w: int
    dpi: int
    score_threshold: float
    source: str
    loader_version: str = "1"
    vista_ocr_commit: str = ""
    n_samples_written: int = 0

    def to_disk(self, path: Path) -> None:
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def from_disk(cls, path: Path) -> CacheManifest:
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(**d)


class CacheMismatchError(RuntimeError):
    """Raised when an existing cache's manifest disagrees with what
    the caller is configured to read/write."""


def _check_geometry_compat(
    have: CacheManifest, want: CacheManifest, *, ignore_loader_version: bool = False,
) -> None:
    """Hard-error on geometry/source mismatch; WARN on loader version."""
    for field_name in ("target_h", "target_w", "dpi", "score_threshold", "source"):
        h = getattr(have, field_name)
        w = getattr(want, field_name)
        if h != w:
            raise CacheMismatchError(
                f"Cache manifest mismatch on {field_name!r}: "
                f"on disk={h!r}, requested={w!r}. Rebuild the cache "
                f"with the new setting or point at a different cache_dir."
            )
    if not ignore_loader_version and have.loader_version != want.loader_version:
        LOG.warning(
            "Cache loader_version differs (on disk=%s, current=%s). "
            "Reading anyway; pass ignore_loader_version=True to silence.",
            have.loader_version, want.loader_version,
        )


# ---------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------

class CacheWriter:
    """Write samples to ``cache_dir`` atomically with resume support.

    Usage::

        manifest = CacheManifest(target_h=1100, target_w=850, dpi=200,
                                 score_threshold=0.5, source="pdfa")
        with CacheWriter(cache_dir, manifest) as w:
            for sample in iter_pdfa(cfg):
                w.write(sample)

    On entry, if a manifest exists the writer loads it and resumes from
    the highest existing index. Geometry mismatch raises
    :class:`CacheMismatchError`.
    """

    def __init__(self, cache_dir: Path, manifest: CacheManifest) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.cache_dir / _MANIFEST_NAME
        if manifest_path.exists():
            on_disk = CacheManifest.from_disk(manifest_path)
            _check_geometry_compat(on_disk, manifest)
            # Carry forward the on-disk count so we resume from there.
            manifest.n_samples_written = on_disk.n_samples_written
        self.manifest = manifest
        # Determine resume index: scan for the highest existing
        # sample_<N>.png to be robust if the manifest count drifted.
        self._next_index = self._scan_next_index()

    def _scan_next_index(self) -> int:
        max_idx = -1
        for p in self.cache_dir.glob("sample_*.png"):
            try:
                idx = int(p.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            max_idx = max(max_idx, idx)
        return max_idx + 1

    def next_index(self) -> int:
        return self._next_index

    def write(self, sample: Sample) -> int:
        """Write one sample. Skips silently if the index is already
        present (so re-running the producer pipeline is idempotent).
        Returns the index assigned."""
        idx = self._next_index
        png_path = self.cache_dir / f"sample_{idx:08d}.png"
        json_path = self.cache_dir / f"sample_{idx:08d}.json"
        # Skip if both exist (resume happy path).
        if png_path.exists() and json_path.exists():
            self._next_index += 1
            return idx
        # Atomic: write to .tmp then rename.
        png_tmp = png_path.with_suffix(".png.tmp")
        json_tmp = json_path.with_suffix(".json.tmp")
        sample.image.save(png_tmp, format="PNG", optimize=False)
        meta = {
            "lines": [
                {"text": ln.text, "bbox": list(ln.bbox)} for ln in sample.lines
            ],
            "task": sample.task,
            "source": sample.source,
        }
        json_tmp.write_text(
            json.dumps(meta, sort_keys=True), encoding="utf-8",
        )
        png_tmp.replace(png_path)
        json_tmp.replace(json_path)
        self._next_index += 1
        self.manifest.n_samples_written = self._next_index
        # Persist the manifest periodically (every 100 samples) and on
        # close so a kill leaves a defensible count.
        if self._next_index % 100 == 0:
            self.manifest.to_disk(self.cache_dir / _MANIFEST_NAME)
        return idx

    def close(self) -> None:
        self.manifest.to_disk(self.cache_dir / _MANIFEST_NAME)

    def __enter__(self) -> CacheWriter:
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN001
        self.close()


# ---------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------

def open_cache(
    cache_dir: Path,
    expected: CacheManifest,
    *,
    ignore_loader_version: bool = False,
) -> CacheManifest:
    """Verify the on-disk manifest matches ``expected`` (geometry
    only). Returns the on-disk manifest. Raises
    :class:`CacheMismatchError` on geometry mismatch.
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / _MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"No cache manifest at {path}")
    on_disk = CacheManifest.from_disk(path)
    _check_geometry_compat(
        on_disk, expected, ignore_loader_version=ignore_loader_version,
    )
    return on_disk


def iter_cached_samples(
    cache_dir: Path,
    expected: CacheManifest,
    *,
    ignore_loader_version: bool = False,
) -> Iterator[Sample]:
    """Yield :class:`Sample` from a cache directory.

    Reads ``manifest.json`` first; raises on geometry mismatch with
    ``expected``. Iterates ``sample_*.png`` in lexicographic order,
    pairing each with its ``sample_*.json`` sidecar.
    """
    cache_dir = Path(cache_dir)
    open_cache(cache_dir, expected, ignore_loader_version=ignore_loader_version)
    for png_path in sorted(cache_dir.glob("sample_*.png")):
        json_path = png_path.with_suffix(".json")
        if not json_path.exists():
            LOG.warning("Skipping cached %s without sidecar %s",
                        png_path.name, json_path.name)
            continue
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        img = Image.open(png_path).convert("L")
        lines = [Line(text=d["text"], bbox=tuple(d["bbox"]))
                 for d in meta.get("lines", [])]
        yield Sample(
            image=img,
            lines=lines,
            task=meta.get("task", "ocr_layout"),
            source=meta.get("source", "cache"),
        )


__all__ = [
    "CacheManifest",
    "CacheMismatchError",
    "CacheWriter",
    "iter_cached_samples",
    "open_cache",
]
