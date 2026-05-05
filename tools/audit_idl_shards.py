"""IDL shard audit: walk every available IDL shard, sample N
records, and report per-shard health metrics.

Per ``notes/run_e_intent.md`` §12 priority 3, the operator-side
smoke (``data/raw/idl/idl-train-00000.tar``, 100 samples) was
clean, but the assumption that all 12 shards resemble shard 00000
hasn't been validated. This tool drives ``iter_idl`` over every
shard and reports:

- **decode-OK rate**: fraction of records that produced a Sample.
- **zero-line rate**: Samples with empty ``lines`` (post-filter).
- **median lines/page**: distribution-shape signal across shards.
- **bbox-in-image rate**: fraction of bboxes that pass the
  resize/clamp pipeline cleanly.
- **non-Latin/drop rate**: fraction of records dropped by the
  ``drop_non_latin`` filter.

Threshold: any shard with ``decode_ok < 0.9`` is flagged red --
that shard would silently distort a long training run if mixed in.

Usage::

    python tools/audit_idl_shards.py \\
        --shards 'data/raw/idl/idl-train-*.tar' \\
        --n-per-shard 50 \\
        --out notes/idl_shards_audit.md
"""
from __future__ import annotations

import argparse
import glob
import itertools
import statistics
import sys
from pathlib import Path


def _expand_shards(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern))


def _audit_one_shard(shard_path: str, n_records: int) -> dict:
    """Drive ``iter_idl`` over a single shard with all default
    filters off, then re-apply the filters manually so we can count
    each drop reason separately."""
    from vista_ocr.data.idl import IdlConfig, _decode_idl_record
    import webdataset as wds  # noqa: PLC0415

    cfg = IdlConfig(shards=[shard_path], drop_non_latin=False, drop_blank=False)
    pipeline = wds.WebDataset([shard_path], shardshuffle=False, empty_check=False)

    n_attempted = 0
    n_decode_ok = 0
    n_zero_line = 0
    n_render_fail = 0
    lines_per_page: list[int] = []
    bbox_total = 0
    bbox_inside = 0
    non_latin_lines = 0
    total_lines = 0

    from vista_ocr.data.preprocess import is_latin_text

    for raw in itertools.islice(pipeline, n_records):
        n_attempted += 1
        try:
            samples = list(_decode_idl_record(raw, cfg))
        except Exception:  # noqa: BLE001
            n_render_fail += 1
            continue
        if not samples:
            n_zero_line += 1
            continue
        n_decode_ok += 1
        s = samples[0]
        lines_per_page.append(len(s.lines))
        img_w, img_h = s.image.size if hasattr(s.image, "size") else (0, 0)
        for ln in s.lines:
            total_lines += 1
            x1, y1, x2, y2 = ln.bbox
            if 0 <= x1 < x2 <= img_w and 0 <= y1 < y2 <= img_h:
                bbox_inside += 1
            bbox_total += 1
            if not is_latin_text(ln.text):
                non_latin_lines += 1

    return {
        "shard": Path(shard_path).name,
        "n_attempted": n_attempted,
        "decode_ok": n_decode_ok,
        "decode_ok_frac": n_decode_ok / max(n_attempted, 1),
        "zero_line": n_zero_line,
        "render_fail": n_render_fail,
        "median_lines_per_page": (
            statistics.median(lines_per_page) if lines_per_page else 0
        ),
        "p95_lines_per_page": (
            sorted(lines_per_page)[int(0.95 * len(lines_per_page))]
            if lines_per_page else 0
        ),
        "bbox_total": bbox_total,
        "bbox_inside": bbox_inside,
        "bbox_inside_frac": bbox_inside / max(bbox_total, 1),
        "total_lines": total_lines,
        "non_latin_lines": non_latin_lines,
        "non_latin_frac": non_latin_lines / max(total_lines, 1),
    }


def _md_table(rows: list[dict]) -> str:
    cols = [
        ("shard", "shard"),
        ("decode_ok_frac", "decode-OK"),
        ("zero_line", "zero-line"),
        ("render_fail", "render-fail"),
        ("median_lines_per_page", "med lines/page"),
        ("p95_lines_per_page", "p95 lines/page"),
        ("bbox_inside_frac", "bbox-in-image"),
        ("non_latin_frac", "non-latin lines"),
    ]
    out = ["| " + " | ".join(label for _, label in cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        cells = []
        for key, _ in cols:
            v = row.get(key, "")
            if isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", required=True,
                   help="Glob, e.g. 'data/raw/idl/idl-train-*.tar'")
    p.add_argument("--n-per-shard", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--decode-ok-threshold", type=float, default=0.9)
    args = p.parse_args()

    shards = _expand_shards(args.shards)
    if not shards:
        print(f"no shards matched: {args.shards}", file=sys.stderr)
        return 1
    print(f"auditing {len(shards)} IDL shards", file=sys.stderr)

    rows: list[dict] = []
    for sp in shards:
        print(f"  {Path(sp).name}", file=sys.stderr)
        rows.append(_audit_one_shard(sp, args.n_per_shard))

    flagged = [r for r in rows if r["decode_ok_frac"] < args.decode_ok_threshold]

    md = ["# IDL shard audit", ""]
    md.append(f"- Shards walked: {len(rows)}")
    md.append(f"- Records sampled per shard: {args.n_per_shard}")
    md.append(f"- decode-OK threshold: {args.decode_ok_threshold:.2f}")
    md.append(f"- Flagged shards (decode-OK < threshold): **{len(flagged)}**")
    md.append("")
    md.append("## Per-shard health")
    md.append("")
    md.append(_md_table(rows))
    md.append("")
    if flagged:
        md.append("## Flagged shards")
        md.append("")
        for r in flagged:
            md.append(f"- `{r['shard']}` -- decode-OK={r['decode_ok_frac']:.3f}, "
                      f"zero-line={r['zero_line']}, render-fail={r['render_fail']}")
        md.append("")
        md.append("Recommendation: exclude flagged shards from the IDL_SHARDS_GLOB ")
        md.append("for Run F, or re-download/re-derive the affected shards before relying ")
        md.append("on them in a long run.")
    else:
        md.append("All shards passed the decode-OK threshold. IDL set is uniform.")
    md.append("")
    md.append("## Reading the table")
    md.append("")
    md.append("- **`decode-OK`**: fraction of records that produced a non-empty Sample. ")
    md.append("  Anything < 0.9 is red.")
    md.append("- **`zero-line`**: records that decoded but produced 0 lines (filtered out). ")
    md.append("  High counts here can indicate JSON-schema drift on that shard.")
    md.append("- **`render-fail`**: pypdfium2 raised on the embedded PDF. Should be near zero.")
    md.append("- **`bbox-in-image`**: fraction of emitted bboxes that fit inside the rendered ")
    md.append("  image. Should be ~1.000; lower means the normalised xywh -> pixel xyxy ")
    md.append("  conversion drifted, or the shard's coordinates are mis-scaled.")
    md.append("- **`non-latin lines`**: fraction of lines that ``drop_non_latin`` would filter ")
    md.append("  out. Material non-zero values mean we may be silently throwing away usable ")
    md.append("  multilingual signal in production.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
