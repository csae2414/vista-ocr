"""Download PDFA shards from HuggingFace ``pixparse/pdfa-eng-wds``.

The dataset is split into ~1800 shards of ~800 MB each. You typically want
just enough to keep the DataLoader fed across all 200K + steps -- 12-20
shards is plenty for an end-to-end run on a single GPU.

Examples::

    # Default: shards 0..11 (12 shards, ~10 GB) into data/raw/pdfa/
    python scripts/download_pdfa.py

    # 24 shards
    python scripts/download_pdfa.py --num-shards 24

    # Specific range
    python scripts/download_pdfa.py --start 4 --end 16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw/pdfa"),
                    help="Local directory to download into.")
    ap.add_argument("--start", type=int, default=0,
                    help="First shard index (inclusive).")
    ap.add_argument("--end", type=int, default=None,
                    help="Last shard index (exclusive). Overrides --num-shards.")
    ap.add_argument("--num-shards", type=int, default=12,
                    help="How many shards to fetch starting at --start. "
                         "Ignored when --end is given.")
    ap.add_argument("--repo", default="pixparse/pdfa-eng-wds")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub not installed. Run `pip install -e .` first.")

    end = args.end if args.end is not None else args.start + args.num_shards
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading shards {args.start}..{end - 1} from {args.repo} -> {args.out}")
    for i in range(args.start, end):
        fname = f"pdfa-eng-train-{i:04d}.tar"
        target = args.out / fname
        if target.exists():
            print(f"  {fname}: already present ({target.stat().st_size / 1e6:.0f} MB)")
            continue
        t0 = time.perf_counter()
        path = hf_hub_download(
            args.repo, filename=fname, repo_type="dataset",
            local_dir=str(args.out),
        )
        elapsed = time.perf_counter() - t0
        size_mb = Path(path).stat().st_size / 1e6
        print(f"  {fname}: {size_mb:.0f} MB in {elapsed:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
