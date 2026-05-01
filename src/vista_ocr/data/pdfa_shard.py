"""Lightweight tar reader for PDFA shards.

Used by helper scripts that need to inspect raw PDFA JSON sidecars
*without* paying the cost of WebDataset + PDF rendering. The
production training loader (:mod:`vista_ocr.data.pdfa`) renders pages
through pypdfium2; this reader skips PDFs entirely and only walks the
JSON sidecars, which is what every measurement / corpus-extraction
script actually needs.

Two scripts duplicated this logic before extraction:

* ``scripts/measure_pdfa_distribution.py`` -- needs page records to
  count lines/words per page.
* ``scripts/retrain_spm_on_pdfa.py`` -- needs the line-text stream
  to feed SentencePiece training.

One reader, two helpers (``iter_pages`` and ``iter_line_texts``), zero
duplication.
"""
from __future__ import annotations

import json
import logging
import tarfile
from collections.abc import Iterator
from pathlib import Path

LOG = logging.getLogger(__name__)


class PdfaShardReader:
    """Walk one or more PDFA shard tarballs and yield JSON sidecar payloads.

    Tolerant by design: a malformed ``.json`` entry is logged and
    skipped, not fatal. Yields nothing for missing shards (after a
    warning) -- callers that want hard-fail wrap the iterator with an
    explicit ``Path.exists()`` check (the constructor does this).
    """

    def __init__(self, shards: list[Path] | list[str]) -> None:
        self._shards: list[Path] = [Path(s) for s in shards]
        for s in self._shards:
            if not s.exists():
                raise FileNotFoundError(s)

    @property
    def shards(self) -> list[Path]:
        return list(self._shards)

    # ------------------------------------------------------------------
    # core: yield raw JSON payloads (the building block for everything)
    # ------------------------------------------------------------------

    def iter_payloads(self) -> Iterator[dict]:
        """Yield every successfully-decoded JSON sidecar across all shards."""
        for shard in self._shards:
            with tarfile.open(shard, "r") as tar:
                for member in tar:
                    if not member.name.endswith(".json"):
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    try:
                        yield json.loads(f.read().decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        LOG.warning("skip malformed json %s: %s",
                                    member.name, exc)
                        continue

    # ------------------------------------------------------------------
    # convenience helpers (the duplicated ones)
    # ------------------------------------------------------------------

    def iter_pages(self) -> Iterator[dict]:
        """Yield each page dict from every payload."""
        for payload in self.iter_payloads():
            yield from payload.get("pages", [])

    def iter_line_texts(self) -> Iterator[str]:
        """Yield each ``pages[].lines.text`` string from every payload.

        Skips non-string entries silently -- malformed wire data
        shouldn't take down a corpus build.
        """
        for page in self.iter_pages():
            block = page.get("lines") or {}
            for text in block.get("text", []):
                if isinstance(text, str):
                    yield text


__all__ = ["PdfaShardReader"]
