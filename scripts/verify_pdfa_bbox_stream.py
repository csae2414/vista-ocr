"""Step-5 snapshot-verify: hash the PDFA bbox stream pre vs post BBox refactor.

The bit-for-bit equivalence is already unit-tested in
``tests/test_bbox.py::TestFromNormalisedXywh::test_matches_legacy``,
so this script is belt-and-braces: it walks one or more real shards,
hashes the (text, bbox) stream, and prints the hash. Run it BEFORE
the refactor lands and AFTER on the same shards; identical hash means
the conversion is empirically unchanged at production scale.

Usage::

    python scripts/verify_pdfa_bbox_stream.py \\
        --shards data/raw/pdfa/pdfa-eng-train-0000.tar
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from vista_ocr.data.pdfa import PdfaConfig, iter_pdfa
from vista_ocr.logging_config import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=200,
                    help="Max pages to hash (default: 200).")
    args = ap.parse_args()
    setup_logging(level="WARNING")

    h = hashlib.sha256()
    cfg = PdfaConfig(shards=[str(p) for p in args.shards])
    pages = 0
    lines = 0
    for sample in iter_pdfa(cfg):
        for line in sample.lines:
            x1, y1, x2, y2 = line.bbox
            h.update(line.text.encode("utf-8"))
            h.update(f"|{x1},{y1},{x2},{y2}\n".encode())
            lines += 1
        pages += 1
        if pages >= args.limit:
            break

    print(f"pages={pages} lines={lines}")
    print(f"sha256={h.hexdigest()}")


if __name__ == "__main__":
    main()
