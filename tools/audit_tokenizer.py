"""Tokenizer audit: <unk> rate, fragmentation, truncation pressure
across PDFA / IDL / SROIE corpora.

Per ``notes/run_e_intent.md`` §12, the production SPM
(``data/processed/vocab/sp_en_16k.model``) was bootstrapped from
WikiText-2 (clean prose). OCR labels are heavy on addresses, table
cells, SKUs, dates, prices -- non-word strings that prose-trained
subword vocabulary fragments aggressively. The risk:

- Inflated target sequence lengths -> ``MAX_TARGET_TOKENS=2048``
  truncation -> attention bottleneck and silent label loss.
- Non-zero ``<unk>`` rate on training labels -> model learns to
  PREDICT ``<unk>``, not real characters.

This tool quantifies the risk before Run F is planned. Cheap,
read-only, runs in minutes against a small sample of each corpus.

Usage::

    python tools/audit_tokenizer.py \\
        --spm data/processed/vocab/sp_en_16k.model \\
        --pdfa-shards 'data/raw/pdfa/pdfa-eng-train-{0000..0005}.tar' \\
        --idl-shards  'data/raw/idl/idl-train-*.tar' \\
        --sroie-manifest data/sroie/manifests/train.jsonl \\
        --n-per-corpus 500 \\
        --max-target-tokens 2048 \\
        --out notes/tokenizer_audit.md

Any of ``--pdfa-shards``, ``--idl-shards``, ``--sroie-manifest``
may be omitted; the corpus is skipped if absent.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import re
import statistics
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

_BRACE_RANGE_RE = re.compile(r"\{(\d+)\.\.(\d+)\}")


def _expand_shards(pattern: str) -> list[str]:
    """Bash-style brace-range glob expansion (mirrors
    tools/synth_target_distributions.py)."""
    m = _BRACE_RANGE_RE.search(pattern)
    if m is None:
        return sorted(glob.glob(pattern))
    start_s, stop_s = m.group(1), m.group(2)
    width = max(len(start_s), len(stop_s))
    lo, hi = sorted([int(start_s), int(stop_s)])
    out: list[str] = []
    for i in range(lo, hi + 1):
        literal = pattern[: m.start()] + str(i).zfill(width) + pattern[m.end():]
        out.extend(glob.glob(literal))
    return sorted(out)


def _percentiles(xs: list[float], qs: tuple[int, ...] = (50, 90, 95, 99)) -> dict[str, float]:
    if not xs:
        return {f"p{q}": 0.0 for q in qs} | {"mean": 0.0, "min": 0.0, "max": 0.0}
    xs_sorted = sorted(xs)
    out: dict[str, float] = {}
    for q in qs:
        idx = min(len(xs_sorted) - 1, int(q * len(xs_sorted) / 100))
        out[f"p{q}"] = float(xs_sorted[idx])
    out["mean"] = float(statistics.mean(xs))
    out["min"] = float(xs_sorted[0])
    out["max"] = float(xs_sorted[-1])
    return out


def _audit_lines(
    sp,
    unk_id: int,
    lines_iter: Iterable[str],
    n: int,
) -> dict:
    """Tokenize an iterable of plain-text lines; report on
    fragmentation (chars/token, tokens/line) and ``<unk>`` rate."""
    chars_per_token: list[float] = []
    tokens_per_line: list[float] = []
    unk_lines = 0
    total_lines = 0
    total_tokens = 0
    total_unk = 0
    for line in itertools.islice(lines_iter, n):
        line = line.strip()
        if not line:
            continue
        ids = sp.EncodeAsIds(line)
        total_lines += 1
        total_tokens += len(ids)
        if ids:
            chars_per_token.append(len(line) / len(ids))
            tokens_per_line.append(len(ids))
        n_unk = sum(1 for i in ids if i == unk_id)
        total_unk += n_unk
        if n_unk > 0:
            unk_lines += 1
    return {
        "n_lines": total_lines,
        "total_tokens": total_tokens,
        "unk_lines": unk_lines,
        "unk_lines_frac": unk_lines / max(total_lines, 1),
        "unk_tokens": total_unk,
        "unk_tokens_frac": total_unk / max(total_tokens, 1),
        "chars_per_token": _percentiles(chars_per_token),
        "tokens_per_line": _percentiles(tokens_per_line),
    }


def _audit_pages_truncation(
    sp,
    pages_iter: Iterator[list[str]],
    n: int,
    max_target_tokens: int,
) -> dict:
    """For each page (list of line texts), compute the serialized
    target length the trainer would see (lines joined with newline)
    and report the fraction that exceed ``max_target_tokens``.

    The actual production target is more elaborate (BOS + prompt +
    output tokens + EOS, with spatial tokens for ocr_layout). We
    approximate with raw text-token count + a small constant; the
    relative shape across corpora is what matters.
    """
    raw_lengths: list[int] = []
    truncated = 0
    total = 0
    for page_lines in itertools.islice(pages_iter, n):
        if not page_lines:
            continue
        joined = "\n".join(line.strip() for line in page_lines if line.strip())
        if not joined:
            continue
        ids = sp.EncodeAsIds(joined)
        # Approximate prompt+sentinels overhead; not exact but stable
        # across corpora so the relative truncation rate is meaningful.
        approx_target_len = len(ids) + 8
        raw_lengths.append(approx_target_len)
        total += 1
        if approx_target_len > max_target_tokens:
            truncated += 1
    return {
        "n_pages": total,
        "approx_target_tokens": _percentiles([float(x) for x in raw_lengths]),
        "truncated_pages": truncated,
        "truncated_frac": truncated / max(total, 1),
        "max_target_tokens": max_target_tokens,
    }


def _iter_pdfa_lines(shards: list[str], n: int) -> tuple[Iterator[str], Iterator[list[str]]]:
    from vista_ocr.data.pdfa import PdfaConfig, iter_pdfa
    samples = itertools.islice(iter_pdfa(PdfaConfig(shards=shards)), n)
    samples = list(samples)
    line_iter = (ln.text for s in samples for ln in s.lines)
    page_iter = ([ln.text for ln in s.lines] for s in samples)
    return line_iter, page_iter


def _iter_idl_lines(shards: list[str], n: int) -> tuple[Iterator[str], Iterator[list[str]]]:
    from vista_ocr.data.idl import IdlConfig, iter_idl
    samples = list(itertools.islice(iter_idl(IdlConfig(shards=shards)), n))
    line_iter = (ln.text for s in samples for ln in s.lines)
    page_iter = ([ln.text for ln in s.lines] for s in samples)
    return line_iter, page_iter


def _iter_sroie_lines(manifest_path: Path, n: int) -> tuple[Iterator[str], Iterator[list[str]]]:
    import json
    samples: list[list[str]] = []
    with manifest_path.open() as f:
        for raw in itertools.islice(f, n):
            rec = json.loads(raw)
            ref = rec.get("ref", "")
            if not ref:
                continue
            samples.append(ref.splitlines())
    line_iter = (ln for page in samples for ln in page)
    page_iter = (page for page in samples)
    return line_iter, page_iter


def _md_table(rows: list[tuple[str, dict]], cols: list[str]) -> str:
    out = ["| Corpus | " + " | ".join(cols) + " |", "|" + "|".join(["---"] * (len(cols) + 1)) + "|"]
    for name, row in rows:
        cells = [name]
        for col in cols:
            v = row.get(col, "")
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            elif isinstance(v, dict) and "p50" in v:
                cells.append(f"p50={v['p50']:.1f} p95={v['p95']:.1f} p99={v['p99']:.1f} mean={v['mean']:.1f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _resolve_locked_pdfa_shards(pattern: str) -> list[str]:
    """PDFA train/val/test split is locked at 0118/0119; refuse those
    in audit input mirroring tools/synth_target_distributions.py."""
    from vista_ocr.data.split import TEST_SHARD_BASENAME, VAL_SHARD_BASENAME
    paths = _expand_shards(pattern)
    locked = {VAL_SHARD_BASENAME, TEST_SHARD_BASENAME}
    return [p for p in paths if Path(p).name not in locked]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spm", type=Path, required=True)
    p.add_argument("--pdfa-shards", default=None,
                   help="Bash-style glob, e.g. 'data/raw/pdfa/pdfa-eng-train-{0000..0005}.tar'")
    p.add_argument("--idl-shards", default=None)
    p.add_argument("--sroie-manifest", type=Path, default=None)
    p.add_argument("--n-per-corpus", type=int, default=500,
                   help="Sample this many pages per corpus.")
    p.add_argument("--max-target-tokens", type=int, default=2048,
                   help="Same as vista_ocr.data.collate.MAX_TARGET_TOKENS.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    try:
        import sentencepiece as spm
    except ImportError:
        print("sentencepiece not installed", file=sys.stderr)
        return 1

    if not args.spm.exists():
        print(f"SPM model not found: {args.spm}", file=sys.stderr)
        return 1
    sp = spm.SentencePieceProcessor()
    sp.Load(str(args.spm))
    unk_id = sp.piece_to_id("<unk>")
    vocab_size = sp.GetPieceSize()

    line_rows: list[tuple[str, dict]] = []
    page_rows: list[tuple[str, dict]] = []

    if args.pdfa_shards:
        shards = _resolve_locked_pdfa_shards(args.pdfa_shards)
        if shards:
            print(f"PDFA: {len(shards)} shards", file=sys.stderr)
            line_iter, page_iter = _iter_pdfa_lines(shards, args.n_per_corpus)
            line_rows.append(("pdfa", _audit_lines(sp, unk_id, line_iter, args.n_per_corpus * 50)))
            page_rows.append(("pdfa", _audit_pages_truncation(sp, page_iter, args.n_per_corpus, args.max_target_tokens)))

    if args.idl_shards:
        shards = _expand_shards(args.idl_shards)
        if shards:
            print(f"IDL: {len(shards)} shards", file=sys.stderr)
            line_iter, page_iter = _iter_idl_lines(shards, args.n_per_corpus)
            line_rows.append(("idl", _audit_lines(sp, unk_id, line_iter, args.n_per_corpus * 50)))
            page_rows.append(("idl", _audit_pages_truncation(sp, page_iter, args.n_per_corpus, args.max_target_tokens)))

    if args.sroie_manifest and args.sroie_manifest.exists():
        print(f"SROIE: {args.sroie_manifest}", file=sys.stderr)
        line_iter, page_iter = _iter_sroie_lines(args.sroie_manifest, args.n_per_corpus)
        line_rows.append(("sroie", _audit_lines(sp, unk_id, line_iter, args.n_per_corpus * 50)))
        page_rows.append(("sroie", _audit_pages_truncation(sp, page_iter, args.n_per_corpus, args.max_target_tokens)))

    if not line_rows:
        print("No corpora resolved; nothing to audit.", file=sys.stderr)
        return 1

    md = ["# Tokenizer audit", ""]
    md.append(f"- SPM model: `{args.spm}`")
    md.append(f"- Vocab size: {vocab_size}")
    md.append(f"- Sample size: ~{args.n_per_corpus} pages per corpus")
    md.append(f"- Truncation threshold: `MAX_TARGET_TOKENS={args.max_target_tokens}`")
    md.append("")
    md.append("## Per-line tokenization")
    md.append("")
    md.append(_md_table(line_rows, ["n_lines", "unk_lines_frac", "unk_tokens_frac",
                                     "chars_per_token", "tokens_per_line"]))
    md.append("")
    md.append("## Per-page target length (truncation pressure)")
    md.append("")
    md.append(_md_table(page_rows, ["n_pages", "approx_target_tokens",
                                     "truncated_pages", "truncated_frac"]))
    md.append("")
    md.append("## Reading the table")
    md.append("")
    md.append("- **`unk_tokens_frac`**: fraction of emitted token IDs that are `<unk>`. ")
    md.append("  Anything above ~0.5% is a label-quality red flag (the model learns to predict `<unk>`).")
    md.append("- **`chars_per_token` mean**: 4.0+ on prose-like text is healthy. ")
    md.append("  < 2.0 indicates heavy fragmentation (typical for digit-rich, address-rich OCR labels).")
    md.append("- **`truncated_frac`**: fraction of pages whose serialized target exceeds ")
    md.append(f"  `MAX_TARGET_TOKENS={args.max_target_tokens}`. Anything above ~5% silently drops training signal at collate.")
    md.append("- **`approx_target_tokens` p99**: if this is close to or exceeds the truncation threshold, ")
    md.append("  the long-page tail is paying compute and getting cut off.")
    md.append("")
    md.append("## Recommended actions")
    md.append("")
    md.append("- If `unk_tokens_frac` > 0.5% on any corpus -> retrain SPM with that corpus mixed in.")
    md.append("- If `chars_per_token` mean < 2.5 -> consider a larger SPM vocab (32k) or domain-mixed retrain.")
    md.append("- If `truncated_frac` > 5% on any corpus -> consider raising `MAX_TARGET_TOKENS` ")
    md.append("  (cost: more attention compute) OR filtering long-tail pages at the loader.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
