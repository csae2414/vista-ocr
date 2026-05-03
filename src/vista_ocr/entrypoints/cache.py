"""``vista-ocr cache`` -- pre-render dataset cache builder.

Thin CLI mirror of ``scripts/cache_dataset.py``. The behavioural-
equivalence test in ``tests/test_entrypoints_equivalence.py`` asserts
the two parsers expose the same flags, defaults, and types.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="vista-ocr cache",
        description="Pre-render dataset samples to a geometry-bound on-disk cache.",
        add_help=add_help,
    )
    ap.add_argument("--source", required=True, choices=("pdfa", "idl"))
    ap.add_argument("--shards", required=True, nargs="+")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--page-h", type=int, required=True)
    ap.add_argument("--page-w", type=int, required=True)
    ap.add_argument("--dpi", type=int, default=200,
                    help="Render dpi; only meaningful for PDFA.")
    ap.add_argument("--score-threshold", type=float, default=0.5)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Cap total samples written. 0 = no cap.")
    ap.add_argument("--log-every", type=int, default=500)
    return ap


def run(args: argparse.Namespace) -> int:
    # Defer heavy imports until execution.
    import time

    from vista_ocr.data.cache import CacheManifest, CacheWriter
    from vista_ocr.data.idl import IdlConfig, iter_idl
    from vista_ocr.data.pdfa import PdfaConfig, iter_pdfa
    from vista_ocr.data.preprocess import PreprocessConfig, resize_to_canvas, pad_to_multiple
    from vista_ocr.logging_config import setup_logging
    import logging

    LOG = logging.getLogger("cache")

    setup_logging(level="INFO")
    pre_cfg = PreprocessConfig(
        target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
    )
    manifest = CacheManifest(
        target_h=args.page_h, target_w=args.page_w,
        dpi=args.dpi, score_threshold=args.score_threshold,
        source=args.source,
    )

    if args.source == "pdfa":
        samples = iter_pdfa(PdfaConfig(
            shards=args.shards, dpi=args.dpi,
            score_threshold=args.score_threshold,
        ))
    else:
        samples = iter_idl(IdlConfig(
            shards=args.shards, score_threshold=args.score_threshold,
        ))

    LOG.info(
        "Caching to %s  source=%s shards=%d  geometry=%dx%d  dpi=%d  score>=%.2f",
        args.out_dir, args.source, len(args.shards),
        args.page_h, args.page_w, args.dpi, args.score_threshold,
    )

    written = 0
    t0 = time.perf_counter()
    with CacheWriter(args.out_dir, manifest) as w:
        if w.next_index() > 0:
            LOG.info("Resuming from index %d (already cached).", w.next_index())
        for sample in samples:
            img, _, _ = resize_to_canvas(sample.image, pre_cfg)
            img, _ = pad_to_multiple(img, pre_cfg.pad_multiple)
            sample.image = img
            w.write(sample)
            written += 1
            if written % args.log_every == 0:
                rate = written / max(time.perf_counter() - t0, 1e-9)
                LOG.info("  ...%d samples written (%.1f/s)", written, rate)
            if args.max_samples and written >= args.max_samples:
                break

    LOG.info("DONE: wrote %d samples to %s", written, args.out_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
