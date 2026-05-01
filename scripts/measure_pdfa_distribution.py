"""Print the empirical lines-per-page and words-per-page distributions
for one or more PDFA shards.

Use this BEFORE setting any value on
``PdfaConfig.drop_above_lines`` / ``drop_above_words`` so the cutoff is
data-driven rather than guessed.

Reads ``pages[].lines.text`` directly from the tar's JSON sidecars; no
PDF rendering, no model. Fast.

Examples::

    python scripts/measure_pdfa_distribution.py \\
        --shards data/raw/pdfa/pdfa-eng-train-0000.tar

    python scripts/measure_pdfa_distribution.py \\
        --shards data/raw/pdfa/pdfa-eng-train-000{0,1,2,3}.tar \\
        --percentiles 50 75 90 95 99
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from collections.abc import Iterator
from pathlib import Path

from vista_ocr.logging_config import setup_logging


def _iter_pages(shard: Path) -> Iterator[dict]:
    if not shard.exists():
        raise FileNotFoundError(shard)
    with tarfile.open(shard, "r") as tar:
        for member in tar:
            if not member.name.endswith(".json"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            try:
                payload = json.loads(f.read().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            yield from payload.get("pages", [])


def _percentile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    k = int(round((q / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(len(sorted_vals) - 1, k))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", type=Path, required=True)
    ap.add_argument("--percentiles", nargs="+", type=float,
                    default=[50, 75, 90, 95, 99, 99.5])
    args = ap.parse_args()
    setup_logging(level="INFO")

    lines_per_page: list[int] = []
    words_per_page: list[int] = []
    for shard in args.shards:
        print(f"reading {shard}", file=sys.stderr)
        for page in _iter_pages(shard):
            block = page.get("lines") or {}
            texts = block.get("text") or []
            lines_per_page.append(len(texts))
            words_per_page.append(sum(len(t.split()) for t in texts))

    if not lines_per_page:
        sys.exit("no pages found")

    lines_per_page.sort()
    words_per_page.sort()

    def _summarise(name: str, vals: list[int]) -> None:
        print(f"\n=== {name} ===")
        print(f"  pages   : {len(vals)}")
        print(f"  min/max : {vals[0]} / {vals[-1]}")
        print(f"  mean    : {sum(vals) / len(vals):.1f}")
        for q in args.percentiles:
            print(f"  p{q:>5}: {_percentile(vals, q)}")

    _summarise("lines per page", lines_per_page)
    _summarise("words per page", words_per_page)
    print(
        "\nUse the p95 (or p99) value on PdfaConfig.drop_above_lines / "
        "drop_above_words to clip outliers without losing the bulk of "
        "the distribution.",
    )


if __name__ == "__main__":
    main()
