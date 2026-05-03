"""PDFA shard split policy.

A single source of truth for which shard is held out for which purpose.
Two shards are reserved at the dataset boundary:

* ``VAL_SHARD_BASENAME`` -- used by ``train()`` for ``ckpt_best``
  selection (val_loss / val_word_f1).
* ``TEST_SHARD_BASENAME`` -- touched **only** by ``scripts/eval_run.sh``
  post-hoc. Training scripts refuse to accept it via
  ``assert_not_test_shard``.

This split is what makes the "PDFA test" rows in ``BENCHMARKS.md``
held-out in the strict sense -- ckpt_best selection has not seen them.
The dynamic "highest-indexed shard is val" policy that preceded this
allowed silent leakage every time ``NUM_PDFA_SHARDS`` changed.
"""
from __future__ import annotations

from pathlib import Path

VAL_SHARD_BASENAME = "pdfa-eng-train-0118.tar"
TEST_SHARD_BASENAME = "pdfa-eng-train-0119.tar"


def assert_not_test_shard(path: Path | str) -> None:
    """Refuse to use the locked test shard for any training-time path.

    Raises ``SystemExit`` (not a generic exception) so a chain script's
    ``set -e`` propagates the failure cleanly.
    """
    name = Path(path).name
    if name == TEST_SHARD_BASENAME:
        raise SystemExit(
            f"refusing: {name} is the locked PDFA test shard "
            f"(see vista_ocr.data.split). Training-time paths must "
            f"never touch it. Use {VAL_SHARD_BASENAME} for val."
        )


def is_val_shard(path: Path | str) -> bool:
    return Path(path).name == VAL_SHARD_BASENAME


def is_test_shard(path: Path | str) -> bool:
    return Path(path).name == TEST_SHARD_BASENAME
