"""Pre-render a dataset to the on-disk sample cache.

Run once before a long training run on a fixed page resolution. The
cache lives at one fixed ``(target_h, target_w, dpi, source,
score_threshold)``; rebuilding for a different preset is required.

Sources currently supported: ``pdfa``, ``idl``. Adding another source
(PageXML, Alto, etc.) is a small adapter that produces ``Sample``
objects and a few lines below.

Examples::

    # PDFA at the L40S 'large' preset.
    python scripts/cache_dataset.py \
        --source pdfa \
        --shards data/raw/pdfa/pdfa-eng-train-*.tar \
        --out-dir /tmp/vista-ocr-data/cache/pdfa-large \
        --page-h 1050 --page-w 1400

    # IDL at the same preset.
    python scripts/cache_dataset.py \
        --source idl \
        --shards data/raw/idl/idl-train-*.tar \
        --out-dir /tmp/vista-ocr-data/cache/idl-large \
        --page-h 1050 --page-w 1400

The script is **idempotent**: re-running on a partially-rendered cache
resumes from the highest existing index. Kill it any time; restart
without losing work.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from vista_ocr.data.cache import CacheManifest, CacheWriter
from vista_ocr.data.idl import IdlConfig, iter_idl
from vista_ocr.data.pdfa import PdfaConfig, iter_pdfa
from vista_ocr.data.preprocess import PreprocessConfig, resize_to_canvas
from vista_ocr.data.types import Sample
from vista_ocr.logging_config import setup_logging
from vista_ocr.tokenizer.tokenizer import Line

LOG = logging.getLogger("cache_dataset")


def _resize_sample(sample: Sample, pre_cfg: PreprocessConfig) -> Sample:
    """Resize image + scale bboxes proportionally so cached samples
    are bit-exact ready for collate's no-op resize path."""
    if sample.image is None:
        return sample
    img, scale, _ = resize_to_canvas(sample.image, pre_cfg)
    if scale != 1.0:
        from vista_ocr.data.bbox import BBox  # noqa: PLC0415
        new_lines = []
        for line in sample.lines:
            bb = BBox.from_xyxy(*line.bbox).scale(sx=scale, sy=scale)
            new_lines.append(Line(text=line.text, bbox=bb.to_xyxy()))
        from dataclasses import replace as _replace  # noqa: PLC0415
        return _replace(sample, image=img, lines=new_lines)
    from dataclasses import replace as _replace  # noqa: PLC0415
    return _replace(sample, image=img)


def _produce_samples(
    source: str, shards: list[str], score_threshold: float, dpi: int,
):
    if source == "pdfa":
        cfg = PdfaConfig(shards=shards, min_line_score=score_threshold, dpi=dpi)
        return iter_pdfa(cfg)
    if source == "idl":
        cfg = IdlConfig(shards=shards)
        return iter_idl(cfg)
    raise ValueError(f"Unknown --source {source!r}; supported: pdfa, idl")


def main() -> None:
    ap = argparse.ArgumentParser()
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
    args = ap.parse_args()

    setup_logging(level="INFO")

    manifest = CacheManifest(
        target_h=args.page_h, target_w=args.page_w,
        dpi=args.dpi, score_threshold=args.score_threshold,
        source=args.source,
    )
    pre_cfg = PreprocessConfig(
        target_h=args.page_h, target_w=args.page_w, pad_multiple=32,
    )

    LOG.info("Caching to %s", args.out_dir)
    LOG.info("  source=%s shards=%d  geometry=%dx%d  dpi=%d  score>=%.2f",
             args.source, len(args.shards), args.page_h, args.page_w,
             args.dpi, args.score_threshold)

    written = 0
    t0 = time.perf_counter()
    samples = _produce_samples(
        args.source, args.shards, args.score_threshold, args.dpi,
    )
    with CacheWriter(args.out_dir, manifest) as w:
        if w.next_index() > 0:
            LOG.info("Resuming from index %d (already cached).",
                     w.next_index())
        for sample in samples:
            sample = _resize_sample(sample, pre_cfg)
            w.write(sample)
            written += 1
            if written % args.log_every == 0:
                rate = written / max(time.perf_counter() - t0, 1e-9)
                LOG.info("  ...%d samples written (%.1f/s)", written, rate)
            if args.max_samples and written >= args.max_samples:
                break

    LOG.info("DONE: wrote %d samples to %s", written, args.out_dir)


if __name__ == "__main__":
    main()
