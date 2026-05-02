"""Download IDL-WDS shards from HuggingFace ``pixparse/idl-wds``.

IDL (Industry Documents Library) is the noisy real-world OCR partner
to the cleaner PDFA dataset; the paper uses both. The full dataset is
3000 tar shards (~5 TB) -- we pull a small contiguous range for
parity with the PDFA shard count.

Examples::

    # Default 12 shards (~10-20 GB) into data/raw/idl/.
    python scripts/download_idl.py

    # 24 shards.
    python scripts/download_idl.py --num-shards 24

    # Specific range.
    python scripts/download_idl.py --start 12 --end 24
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw/idl"),
                    help="Local directory to download into.")
    ap.add_argument("--start", type=int, default=0,
                    help="First shard index (inclusive).")
    ap.add_argument("--end", type=int, default=None,
                    help="Last shard index (exclusive). Overrides --num-shards.")
    ap.add_argument("--num-shards", type=int, default=12,
                    help="How many shards to fetch starting at --start. "
                         "Ignored when --end is given.")
    ap.add_argument("--repo", default="pixparse/idl-wds")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub not installed. Run `pip install -e .` first.")

    end = args.end if args.end is not None else args.start + args.num_shards
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading shards {args.start}..{end - 1} from {args.repo} -> {args.out}")
    total_mb = 0.0
    for i in range(args.start, end):
        fname = f"idl-train-{i:05d}.tar"
        target = args.out / fname
        if target.exists():
            size_mb = target.stat().st_size / 1e6
            print(f"  {fname}: already present ({size_mb:.0f} MB)")
            total_mb += size_mb
            continue
        t0 = time.perf_counter()
        path = hf_hub_download(
            args.repo, filename=fname, repo_type="dataset",
            local_dir=str(args.out),
        )
        elapsed = time.perf_counter() - t0
        size_mb = Path(path).stat().st_size / 1e6
        total_mb += size_mb
        print(f"  {fname}: {size_mb:.0f} MB in {elapsed:.1f}s "
              f"({size_mb / max(elapsed, 1e-3):.1f} MB/s)")
    print(f"DONE: {end - args.start} shards, {total_mb / 1024:.2f} GB total -> {args.out}")


if __name__ == "__main__":
    main()
