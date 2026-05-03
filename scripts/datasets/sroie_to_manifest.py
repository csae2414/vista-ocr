"""SROIE flat-layout -> JSONL manifest.

Walks ``<root>/<split>/<id>.txt`` (the format produced by
``setup_sroie.sh``), emits one JSONL line per doc with the doc's
image path, whitespace-joined ref text, and the per-line quadrilateral
projected to an axis-aligned bbox.

Usage::

    python scripts/datasets/sroie_to_manifest.py \\
        --root data/raw/sroie --split train \\
        --out data/raw/sroie/manifests/train.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

LOG = logging.getLogger("sroie-to-manifest")


def _parse_quad_line(s: str):
    parts = s.rstrip("\n").split(",", 8)
    if len(parts) < 9:
        return None
    try:
        coords = [int(float(p)) for p in parts[:8]]
    except ValueError:
        return None
    text = parts[8].strip()
    if not text:
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    return [min(xs), min(ys), max(xs), max(ys), text]


def _emit_record(image_rel: str, lines: list[list]) -> dict:
    ref = " ".join(line[4] for line in lines)
    rec: dict = {"image": image_rel, "ref": ref}
    if lines:
        rec["bboxes"] = lines
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="SROIE flat layout root (with <split>/<id>.{jpg,txt}).")
    ap.add_argument("--split", required=True, choices=("train", "test"))
    ap.add_argument("--out", required=True, type=Path,
                    help="JSONL output path. Image paths are written "
                         "relative to this file's parent dir.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    split_dir = args.root / args.split
    if not split_dir.exists():
        raise SystemExit(f"split dir missing: {split_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for txt_path in sorted(split_dir.glob("*.txt")):
            jpg_path = txt_path.with_suffix(".jpg")
            if not jpg_path.exists():
                n_skipped += 1
                continue
            lines: list[list] = []
            for raw in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                ln = _parse_quad_line(raw)
                if ln is not None:
                    lines.append(ln)
            if not lines:
                n_skipped += 1
                continue
            try:
                # Image path written relative to the manifest's dir so
                # the JSONL is portable.
                image_rel = str(jpg_path.resolve().relative_to(args.out.parent.resolve()))
            except ValueError:
                # Fall back to absolute path when the image isn't under
                # the manifest dir (common when the data root is on a
                # different mount than the manifests).
                image_rel = str(jpg_path.resolve())
            rec = _emit_record(image_rel, lines)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    LOG.info("Wrote %d records to %s (skipped %d empty/missing)",
             n_written, args.out, n_skipped)


if __name__ == "__main__":
    main()
