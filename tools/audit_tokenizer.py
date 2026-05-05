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

Truncation pressure is measured against the production helper
:func:`vista_ocr.data.collate.build_target_ids` called with
``truncate=False`` so the audit observes the **pre-truncation**
target length. A naive call would silently cap at
``MAX_TARGET_TOKENS`` and report a meaningless truncation rate.

SROIE is read via ``--sroie-root <path>`` driving
:func:`vista_ocr.data.sroie.iter_sroie`. The manifest path is NOT
supported because manifest records may lack layout bboxes; without
bboxes the ``ocr_layout`` task length is undefined.

Usage::

    python tools/audit_tokenizer.py \\
        --spm data/processed/vocab/sp_en_16k.model \\
        --pdfa-shards 'data/raw/pdfa/pdfa-eng-train-{0000..0005}.tar' \\
        --idl-shards  'data/raw/idl/idl-train-*.tar' \\
        --sroie-root  data/raw/SROIE2019 \\
        --task ocr_layout \\
        --n-per-corpus 500 \\
        --out notes/tokenizer_audit.md

Any of ``--pdfa-shards``, ``--idl-shards``, ``--sroie-root`` may
be omitted; the corpus is skipped if absent.
"""
from __future__ import annotations

import argparse
import itertools
import statistics
import sys
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path

from vista_ocr.data.collate import MAX_TARGET_TOKENS, build_target_ids
from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import VistaTokenizer
from vista_ocr.utils.shard_glob import expand_shards


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
) -> dict:
    """Tokenize an iterable of plain-text lines; report on
    fragmentation (chars/token, tokens/line) and ``<unk>`` rate.

    The input iterator is page-capped by the caller (one of the
    ``_iter_*`` helpers, each of which slices to ``n_per_corpus``
    samples). No internal cap is applied here -- adding one
    creates two layers of capping that disagree silently. The
    page cap is the single source of truth.
    """
    chars_per_token: list[float] = []
    tokens_per_line: list[float] = []
    unk_lines = 0
    total_lines = 0
    total_tokens = 0
    total_unk = 0
    for line in lines_iter:
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
    tokenizer: VistaTokenizer,
    samples_iter: Iterator[Sample],
    n: int,
    task: str,
    max_target_tokens: int,
) -> dict:
    """For each Sample, compute the **exact pre-truncation** target
    length the trainer would see (BOS + prompt + output + EOS,
    including spatial tokens for ``ocr_layout``) by calling
    :func:`build_target_ids` with ``truncate=False``. Report the
    fraction whose pre-truncation length exceeds
    ``max_target_tokens``.

    Returns target_tokens percentiles + truncated count + frac.
    The ``truncate=False`` keyword on ``build_target_ids`` is the
    contract this audit depends on; a regression catch lives in
    ``tests/test_data.py::test_build_target_ids_returns_full_length_when_truncate_false``.
    """
    target_lengths: list[int] = []
    truncated = 0
    total = 0
    for sample in itertools.islice(samples_iter, n):
        if not sample.lines:
            continue
        # Override the sample's task so the audit reports the
        # task-specific truncation rate the operator selected.
        s = sample if sample.task == task else replace(sample, task=task)
        try:
            seq, _prompt_len = build_target_ids(tokenizer, s, truncate=False)
        except ValueError:
            # Some tasks need a query_text/query_bbox the audit
            # doesn't synthesise (region_ocr, find_it). Skip cleanly.
            continue
        target_len = len(seq)
        target_lengths.append(target_len)
        total += 1
        if target_len > max_target_tokens:
            truncated += 1
    return {
        "n_pages": total,
        "target_tokens": _percentiles([float(x) for x in target_lengths]),
        "truncated_pages": truncated,
        "truncated_frac": truncated / max(total, 1),
        "max_target_tokens": max_target_tokens,
        "task": task,
    }


def _materialise_samples(samples: Iterator[Sample], n: int) -> list[Sample]:
    """Pull at most ``n`` Samples eagerly so we can produce two
    independent views (line iter + page iter) without re-driving
    the (slow) underlying loader."""
    return list(itertools.islice(samples, n))


def _iter_pdfa(shards: list[str], n: int) -> tuple[Iterator[str], Iterator[Sample]]:
    from vista_ocr.data.pdfa import PdfaConfig, iter_pdfa
    samples = _materialise_samples(iter_pdfa(PdfaConfig(shards=shards)), n)
    return ((ln.text for s in samples for ln in s.lines), iter(samples))


def _iter_idl(shards: list[str], n: int) -> tuple[Iterator[str], Iterator[Sample]]:
    from vista_ocr.data.idl import IdlConfig, iter_idl
    samples = _materialise_samples(iter_idl(IdlConfig(shards=shards)), n)
    return ((ln.text for s in samples for ln in s.lines), iter(samples))


def _iter_sroie(root: Path, n: int) -> tuple[Iterator[str], Iterator[Sample]]:
    from vista_ocr.data.sroie import SroieConfig, iter_sroie
    samples = _materialise_samples(iter_sroie(SroieConfig(root=root)), n)
    return ((ln.text for s in samples for ln in s.lines), iter(samples))


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
    paths = expand_shards(pattern)
    locked = {VAL_SHARD_BASENAME, TEST_SHARD_BASENAME}
    return [p for p in paths if Path(p).name not in locked]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spm", type=Path, required=True)
    p.add_argument("--pdfa-shards", default=None,
                   help="Bash-style glob, e.g. 'data/raw/pdfa/pdfa-eng-train-{0000..0005}.tar'")
    p.add_argument("--idl-shards", default=None)
    p.add_argument("--sroie-root", type=Path, default=None,
                   help="SROIE 2019 raw tree containing img/ + box/ subdirs (drives iter_sroie). "
                        "The manifest path is NOT supported -- manifest records may lack bboxes "
                        "and ocr_layout target length is undefined without them.")
    p.add_argument("--task", default="ocr_layout", choices=("ocr", "ocr_layout"),
                   help="Task to use for the truncation audit. ocr_layout is the production "
                        "stage-2/3 path and includes spatial tokens around each line; ocr "
                        "produces a pure text target.")
    p.add_argument("--n-per-corpus", type=int, default=500,
                   help="Sample this many pages per corpus.")
    p.add_argument("--max-target-tokens", type=int, default=MAX_TARGET_TOKENS,
                   help="Defaults to vista_ocr.data.collate.MAX_TARGET_TOKENS.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if not args.spm.exists():
        print(f"SPM model not found: {args.spm}", file=sys.stderr)
        return 1

    # Build the production VistaTokenizer; the per-line <unk> audit
    # reaches through to the underlying SentencePieceProcessor via
    # ``tokenizer.sp``. The truncation audit calls build_target_ids
    # which requires the full tokenizer + spatial grid.
    grid = SpatialGrid(canvas_h=3508, canvas_w=2480, quantizer_px=10, scheme="original")
    tokenizer = VistaTokenizer(spm_model_path=str(args.spm), grid=grid)
    sp = tokenizer.sp
    unk_id = tokenizer.unk_id
    vocab_size = sp.GetPieceSize()

    line_rows: list[tuple[str, dict]] = []
    page_rows: list[tuple[str, dict]] = []

    if args.pdfa_shards:
        shards = _resolve_locked_pdfa_shards(args.pdfa_shards)
        if shards:
            print(f"PDFA: {len(shards)} shards", file=sys.stderr)
            line_iter, samples_iter = _iter_pdfa(shards, args.n_per_corpus)
            line_rows.append(("pdfa", _audit_lines(sp, unk_id, line_iter)))
            page_rows.append(("pdfa", _audit_pages_truncation(
                tokenizer, samples_iter, args.n_per_corpus, args.task, args.max_target_tokens,
            )))

    if args.idl_shards:
        shards = expand_shards(args.idl_shards)
        if shards:
            print(f"IDL: {len(shards)} shards", file=sys.stderr)
            line_iter, samples_iter = _iter_idl(shards, args.n_per_corpus)
            line_rows.append(("idl", _audit_lines(sp, unk_id, line_iter)))
            page_rows.append(("idl", _audit_pages_truncation(
                tokenizer, samples_iter, args.n_per_corpus, args.task, args.max_target_tokens,
            )))

    if args.sroie_root and args.sroie_root.exists():
        print(f"SROIE: {args.sroie_root}", file=sys.stderr)
        line_iter, samples_iter = _iter_sroie(args.sroie_root, args.n_per_corpus)
        line_rows.append(("sroie", _audit_lines(sp, unk_id, line_iter)))
        page_rows.append(("sroie", _audit_pages_truncation(
            tokenizer, samples_iter, args.n_per_corpus, args.task, args.max_target_tokens,
        )))

    if not line_rows:
        print("No corpora resolved; nothing to audit.", file=sys.stderr)
        return 1

    md = ["# Tokenizer audit", ""]
    md.append(f"- SPM model: `{args.spm}`")
    md.append(f"- Vocab size: {vocab_size}")
    md.append(f"- Sample size: ~{args.n_per_corpus} pages per corpus")
    md.append(f"- Truncation threshold: `MAX_TARGET_TOKENS={args.max_target_tokens}`")
    md.append(f"- Task: `{args.task}` (target length includes BOS + prompt + output + EOS, "
              f"plus spatial tokens for ocr_layout). Computed via "
              f"`build_target_ids(truncate=False)` so reported lengths are pre-truncation.")
    md.append("")
    md.append("## Per-line tokenization")
    md.append("")
    md.append(_md_table(line_rows, ["n_lines", "unk_lines_frac", "unk_tokens_frac",
                                     "chars_per_token", "tokens_per_line"]))
    md.append("")
    md.append(f"## Per-page target length (truncation pressure, task={args.task})")
    md.append("")
    md.append(_md_table(page_rows, ["n_pages", "target_tokens",
                                     "truncated_pages", "truncated_frac"]))
    md.append("")
    md.append("## Reading the table")
    md.append("")
    md.append("- **`unk_tokens_frac`**: fraction of emitted token IDs that are `<unk>`. ")
    md.append("  Anything above ~0.5% is a label-quality red flag (the model learns to predict `<unk>`).")
    md.append("- **`chars_per_token` mean**: 4.0+ on prose-like text is healthy. ")
    md.append("  < 2.0 indicates heavy fragmentation (typical for digit-rich, address-rich OCR labels).")
    md.append("- **`truncated_frac`**: fraction of pages whose pre-truncation target exceeds ")
    md.append(f"  `MAX_TARGET_TOKENS={args.max_target_tokens}`. Anything above ~5% silently drops training signal at collate.")
    md.append("- **`target_tokens` p99**: if this is close to or exceeds the truncation threshold, ")
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
