"""J0 #3: distribution-baseline tool.

Reads a stream of :class:`Sample` (PDFA / IDL / IAM, etc.) and reports
sanity-band distributions used in the J1a generator's diagnostics
(notes/plan_phase_j.md §0/J0 + §10b #1).

CAVEAT: PDFA/IDL are PRINTED-document distributions; IAM is
HANDWRITING. Forcing handwritten synth to MATCH PDFA/IDL is wrong;
these are SANITY BANDS to catch pathological collapse, not fitting
targets. If IAM is accessible, capture IAM stats too with
``--source iam`` and prefer those for fitting.

Usage:
    python tools/synth_target_distributions.py \\
        --source pdfa --shards 'pdfa-eng-train-{0000..0117}.tar' \\
        --out notes/synth_target_distributions.json \\
        --n 5000

The output JSON is what HandwrittenLineSynth's diagnostics report
compares against in J1a.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, choices=["pdfa", "idl", "iam"],
                   help="Which iter_<source> to drive.")
    p.add_argument("--shards", default=None,
                   help="WebDataset shard glob (pdfa/idl) -- ignored for iam.")
    p.add_argument("--root", type=Path, default=None,
                   help="Dataset root (iam).")
    p.add_argument("--n", type=int, default=5000,
                   help="Number of samples to ingest (default 5000).")
    p.add_argument("--out", type=Path, required=True,
                   help="Path to write the JSON report.")
    return p.parse_args()


def _percentiles(xs: list[float], qs: tuple[int, ...] = (50, 90, 95, 99)) -> dict[str, float]:
    if not xs:
        return {f"p{q}": 0.0 for q in qs}
    xs = sorted(xs)
    out: dict[str, float] = {}
    for q in qs:
        idx = min(len(xs) - 1, int(q * len(xs) / 100))
        out[f"p{q}"] = float(xs[idx])
    out["mean"] = float(statistics.mean(xs))
    out["min"] = float(xs[0])
    out["max"] = float(xs[-1])
    return out


def _build_iter(args: argparse.Namespace):
    if args.source == "pdfa":
        from vista_ocr.data.pdfa import iter_pdfa
        if not args.shards:
            raise SystemExit("--shards required for pdfa")
        return iter_pdfa(args.shards)
    if args.source == "idl":
        from vista_ocr.data.idl import iter_idl
        if not args.shards:
            raise SystemExit("--shards required for idl")
        return iter_idl(args.shards)
    if args.source == "iam":
        from vista_ocr.data.iam import IamConfig, iter_iam
        if args.root is None:
            raise SystemExit("--root required for iam")
        return iter_iam(IamConfig(root=args.root))
    raise SystemExit(f"unknown source: {args.source}")


def main() -> int:
    args = parse_args()

    iterator = _build_iter(args)

    line_heights: list[float] = []
    words_per_line: list[float] = []
    chars_per_line: list[float] = []
    aspect_ratios: list[float] = []
    text_density: list[float] = []
    seq_len_chars: list[float] = []
    n = 0
    for sample in iterator:
        if n >= args.n:
            break
        n += 1
        img_w, img_h = sample.image.size if hasattr(sample.image, "size") else (0, 0)
        page_area = max(img_w * img_h, 1)
        bbox_area_total = 0
        page_chars = 0
        for ln in sample.lines:
            x1, y1, x2, y2 = ln.bbox
            w, h = max(x2 - x1, 0), max(y2 - y1, 0)
            if h > 0:
                line_heights.append(h)
            if h > 0 and w > 0:
                aspect_ratios.append(w / h)
            words_per_line.append(len(ln.text.split()))
            chars_per_line.append(len(ln.text))
            bbox_area_total += w * h
            page_chars += len(ln.text)
        text_density.append(bbox_area_total / page_area)
        seq_len_chars.append(page_chars)

    report: dict[str, Any] = {
        "source": args.source,
        "n_samples": n,
        "line_height_px": _percentiles(line_heights),
        "words_per_line": _percentiles(words_per_line),
        "chars_per_line": _percentiles(chars_per_line),
        "bbox_aspect_ratio": _percentiles(aspect_ratios),
        "page_text_density": _percentiles(text_density),
        "page_total_chars": _percentiles(seq_len_chars),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
