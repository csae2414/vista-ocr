"""Bash-style brace-range glob expansion for WebDataset shard patterns.

The chain script (``scripts/pretrain_chain.sh``) and the runbook use
patterns like ``data/raw/pdfa/pdfa-eng-train-{0000..0117}.tar`` -- a
zero-padded numeric range that bash expands natively. Python's
``glob.glob`` does NOT understand braces; passing the raw pattern to
``glob.glob`` matches zero files silently.

This helper recognises a single ``{start..stop}`` numeric range with
zero-padding preserved, expands it to a list of literal globs, and
returns the sorted union of matching files. Multi-range patterns
(``{a..b}-{c..d}``) are not supported; callers in this repo only use
the single-range form.

Patterns without braces fall through to plain ``glob.glob``.
"""
from __future__ import annotations

import glob
import re

_BRACE_RANGE_RE = re.compile(r"\{(\d+)\.\.(\d+)\}")


def expand_shards(pattern: str) -> list[str]:
    """Expand a bash-style brace-range glob to sorted matching files.

    :param pattern: e.g. ``"data/raw/pdfa/pdfa-eng-train-{0000..0117}.tar"``
        or a plain glob like ``"data/raw/idl/idl-train-*.tar"``.
    :returns: Sorted list of matching file paths. Missing files
        within the range are silently dropped (matches ``glob.glob``
        semantics).

    Single-range only: a pattern with multiple ``{a..b}`` ranges
    raises no error but only the first range is expanded; the rest
    pass through to ``glob.glob`` as literal text and likely match
    nothing.
    """
    m = _BRACE_RANGE_RE.search(pattern)
    if m is None:
        return sorted(glob.glob(pattern))
    start_s, stop_s = m.group(1), m.group(2)
    width = max(len(start_s), len(stop_s))
    start, stop = int(start_s), int(stop_s)
    lo, hi = (start, stop) if start <= stop else (stop, start)
    out: list[str] = []
    for i in range(lo, hi + 1):
        literal = pattern[: m.start()] + str(i).zfill(width) + pattern[m.end():]
        out.extend(glob.glob(literal))
    return sorted(out)
