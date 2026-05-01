"""Retrain a SentencePiece tokenizer on real PDFA line text.

The default SPM (``data/processed/vocab/sp_en_16k.model``) was trained
on WikiText-2 -- generic English. PDFA contains receipts, forms,
academic papers, etc. with different word distributions and many short
domain-specific tokens (codes, dates, prices). A domain-matched SPM
typically reduces text loss by 5-15 % on its own.

This script is a **tool**; it does not change project defaults. The
retrained model is written to a separate file so an operator can
explicitly opt in via ``--spm`` on the stage scripts.

**Important constraint**: SentencePiece vocabulary IDs change with the
training corpus. Switching SPM means retraining the model from scratch
(any saved checkpoint trained against the old vocab cannot be loaded).
The script does NOT touch existing checkpoints.

Filter policy (per the design notes):
- drop lines shorter than 3 characters (single tokens / artefacts)
- drop lines whose alpha-character fraction is below 0.5
  (mostly-numeric / punctuation lines distort the BPE statistics)

Example::

    python scripts/retrain_spm_on_pdfa.py \\
        --shards data/raw/pdfa/pdfa-eng-train-{0000,0001,0002,0003}.tar \\
        --vocab-size 16000 \\
        --out-prefix data/processed/vocab/sp_pdfa_16k

The output is a SentencePiece ``.model`` + ``.vocab`` file. Use it via::

    python scripts/stage1_run.py --spm data/processed/vocab/sp_pdfa_16k.model ...
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tarfile
from collections.abc import Iterable, Iterator
from pathlib import Path

from vista_ocr.logging_config import setup_logging
from vista_ocr.tokenizer.build_spm import train_spm
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import list_special_and_spatial_tokens

LOG = logging.getLogger("retrain_spm")


def is_useful_line(text: str, min_chars: int = 3, min_alpha_frac: float = 0.5) -> bool:
    """True if ``text`` carries enough English signal to feed BPE.

    Filters out single-character noise, mostly-numeric lines, and
    punctuation-only artefacts. Tested directly so the policy is
    locked-in regardless of the calling context.
    """
    if len(text) < min_chars:
        return False
    if not text:
        return False
    alpha = sum(1 for c in text if c.isalpha())
    return alpha / len(text) >= min_alpha_frac


def iter_lines_from_shard(shard: Path) -> Iterator[str]:
    """Stream every ``pages[].lines.text`` entry out of a PDFA shard.

    Reads the tar without depending on webdataset; each json is parsed
    independently so a malformed entry is skipped, not fatal.
    """
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
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                LOG.warning("skip malformed json %s: %s", member.name, exc)
                continue
            for page in payload.get("pages", []):
                lines = page.get("lines") or {}
                for text in lines.get("text", []):
                    if isinstance(text, str):
                        yield text


def write_corpus(
    line_iter: Iterable[str],
    out_path: Path,
    *,
    min_chars: int = 3,
    min_alpha_frac: float = 0.5,
) -> tuple[int, int]:
    """Write filtered lines to ``out_path``. Returns ``(kept, dropped)``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = 0
    with out_path.open("w", encoding="utf-8") as f:
        for raw in line_iter:
            text = raw.strip()
            if is_useful_line(text, min_chars, min_alpha_frac):
                f.write(text + "\n")
                kept += 1
            else:
                dropped += 1
    return kept, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", type=Path, required=True,
                    help="PDFA shard tarballs to extract line text from.")
    ap.add_argument("--corpus", type=Path,
                    default=Path("data/processed/corpora/pdfa_lines.txt"),
                    help="Where the filtered corpus is written.")
    ap.add_argument("--out-prefix", type=Path,
                    default=Path("data/processed/vocab/sp_pdfa_16k"),
                    help="SentencePiece output prefix; .model + .vocab written next to it.")
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--canvas-h", type=int, default=3508)
    ap.add_argument("--canvas-w", type=int, default=2480)
    ap.add_argument("--quantizer-px", type=int, default=10)
    ap.add_argument("--scheme", default="original",
                    choices=("original", "segmented", "unified"))
    ap.add_argument("--min-chars", type=int, default=3)
    ap.add_argument("--min-alpha-frac", type=float, default=0.5)
    ap.add_argument("--max-lines", type=int, default=0,
                    help="Cap total kept lines (0 = no cap).")
    args = ap.parse_args()

    setup_logging(level="INFO")

    # ---------- 1. extract + filter ----------
    LOG.info("Extracting lines from %d shards", len(args.shards))

    def stream() -> Iterator[str]:
        seen = 0
        for shard in args.shards:
            LOG.info("  shard %s", shard)
            for text in iter_lines_from_shard(shard):
                yield text
                seen += 1
                if args.max_lines and seen >= args.max_lines:
                    return

    kept, dropped = write_corpus(
        stream(), args.corpus,
        min_chars=args.min_chars, min_alpha_frac=args.min_alpha_frac,
    )
    LOG.info("Corpus: kept=%d, dropped=%d, ratio=%.3f -> %s",
             kept, dropped, kept / max(1, kept + dropped), args.corpus)
    if kept == 0:
        LOG.error("No lines kept; refusing to train SentencePiece on empty corpus.")
        sys.exit(1)

    # ---------- 2. train SPM with the same special + spatial tokens ----------
    grid = SpatialGrid(
        canvas_h=args.canvas_h, canvas_w=args.canvas_w,
        quantizer_px=args.quantizer_px, scheme=args.scheme,
    )
    user_syms = list_special_and_spatial_tokens(grid)
    LOG.info(
        "Training SentencePiece: vocab=%d, user_symbols=%d (specials + spatial)",
        args.vocab_size, len(user_syms),
    )
    model_path = train_spm(
        corpus_path=args.corpus,
        out_prefix=args.out_prefix,
        vocab_size=args.vocab_size,
        user_symbols=user_syms,
    )
    LOG.info("DONE: %s", model_path)


if __name__ == "__main__":
    main()
