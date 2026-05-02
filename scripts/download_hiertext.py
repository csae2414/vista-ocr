"""Download the HierText dataset from HuggingFace.

The canonical Google-Research HierText repo is not hosted on the
HuggingFace Hub, so this script defaults to the working community
mirror ``Berzerker/ocr_hiertext`` which carries pre-rendered page
images plus a derivative annotation field (``output_json_dumpsed``).
The dataset weighs ~1.6 GB train (single parquet).

Override ``--repo`` if you have a private mirror that exposes the
canonical schema.

Examples::

    # Download the default mirror into data/raw/hiertext/.
    python scripts/download_hiertext.py

    # Just the train split.
    python scripts/download_hiertext.py --splits train
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=Path("data/raw/hiertext"),
                    help="Local cache directory. Same value goes into HierTextConfig.")
    ap.add_argument("--splits", nargs="+",
                    default=["train"],
                    choices=["train", "validation", "test"],
                    help="Splits to materialise. The default mirror only "
                         "carries 'train'; pass 'validation'/'test' only "
                         "with a --repo that has them.")
    ap.add_argument("--repo", default="Berzerker/ocr_hiertext")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("`datasets` not installed. Run `pip install -e .` first.")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Caching {args.repo} -> {args.cache_dir}")
    for split in args.splits:
        t0 = time.perf_counter()
        ds = load_dataset(args.repo, split=split, cache_dir=str(args.cache_dir))
        elapsed = time.perf_counter() - t0
        print(f"  split={split:10s} N={len(ds):>6d}  ({elapsed:.1f}s)")
    print("DONE")


if __name__ == "__main__":
    main()
