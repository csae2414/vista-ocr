"""Bootstrap a SentencePiece tokenizer on a small generic English corpus.

This is a placeholder corpus to unblock model + loss work before the PDFA /
IDL downloads finish. Once those are ready, retrain SPM on the real corpus
via the same `train_spm` function.

Default corpus: WikiText-2 (small, ~2M tokens, license-free).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from vista_ocr.tokenizer.build_spm import train_spm
from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
from vista_ocr.tokenizer.tokenizer import list_special_and_spatial_tokens


def _materialize_corpus(out_path: Path) -> None:
    from datasets import load_dataset

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    n_lines = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            text = row["text"].strip()
            if not text:
                continue
            f.write(text + "\n")
            n_lines += 1
    print(f"Wrote {n_lines} lines to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus", type=Path, default=Path("data/processed/corpora/en_bootstrap.txt")
    )
    ap.add_argument(
        "--out_prefix", type=Path, default=Path("data/processed/vocab/sp_en_16k")
    )
    ap.add_argument("--vocab_size", type=int, default=16000)
    ap.add_argument("--canvas_h", type=int, default=3508)
    ap.add_argument("--canvas_w", type=int, default=2480)
    ap.add_argument("--quantizer_px", type=int, default=10)
    ap.add_argument("--scheme", type=str, default="original")
    ap.add_argument("--skip_corpus_download", action="store_true")
    args = ap.parse_args()

    if not args.skip_corpus_download and not args.corpus.exists():
        _materialize_corpus(args.corpus)

    grid = SpatialGrid(
        canvas_h=args.canvas_h,
        canvas_w=args.canvas_w,
        quantizer_px=args.quantizer_px,
        scheme=args.scheme,
    )
    user_syms = list_special_and_spatial_tokens(grid)
    print(f"Training SPM with vocab_size={args.vocab_size}, "
          f"user_symbols={len(user_syms)} (specials + spatial)...")
    model_path = train_spm(
        corpus_path=args.corpus,
        out_prefix=args.out_prefix,
        vocab_size=args.vocab_size,
        user_symbols=user_syms,
    )
    print(f"Wrote: {model_path}")


if __name__ == "__main__":
    main()
