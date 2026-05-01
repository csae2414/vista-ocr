"""Download the HierText dataset from HuggingFace.

HierText is ~12 GB total (train + validation + test). The HuggingFace
``datasets`` library handles the actual download and caching. We just
materialise the splits into a known location so the loader can read
them without re-downloading.

Examples::

    # Download all three splits into the default HF cache + a local
    # snapshot at data/raw/hiertext/.
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
                    default=["train", "validation", "test"],
                    choices=["train", "validation", "test"])
    ap.add_argument("--repo", default="google-research-datasets/hiertext")
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
