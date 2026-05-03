"""Test the locked PDFA val/test split.

Three things this test enforces:

1. ``assert_not_test_shard`` raises ``SystemExit`` on the locked test
   basename and is a no-op on every other path.
2. ``pretrain_chain.sh`` excludes both the val and test basenames from
   ``TRAIN_SHARDS``.
3. ``pretrain_chain.sh`` never passes the test shard to a stage script
   as ``--val-shard`` (or as a train shard).

A static-text scan is sufficient -- the chain script is short and the
basenames are constants. We do *not* execute the script (it requires
shards on disk + a conda env).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from vista_ocr.data.split import (
    TEST_SHARD_BASENAME,
    VAL_SHARD_BASENAME,
    assert_not_test_shard,
    is_test_shard,
    is_val_shard,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAIN_SH = REPO_ROOT / "scripts" / "pretrain_chain.sh"


def test_assert_not_test_shard_blocks_locked_basename():
    with pytest.raises(SystemExit) as exc:
        assert_not_test_shard(f"data/raw/pdfa/{TEST_SHARD_BASENAME}")
    assert TEST_SHARD_BASENAME in str(exc.value)


def test_assert_not_test_shard_passes_val_and_train_shards():
    # Val shard is fine. So is any other train shard.
    assert_not_test_shard(f"data/raw/pdfa/{VAL_SHARD_BASENAME}")
    assert_not_test_shard("data/raw/pdfa/pdfa-eng-train-0042.tar")
    assert_not_test_shard(Path("/tmp/nope.tar"))


def test_is_val_and_is_test_helpers_are_basename_scoped():
    # Path components other than the basename are ignored.
    assert is_val_shard(f"/abs/path/to/{VAL_SHARD_BASENAME}")
    assert is_test_shard(Path("data/raw/pdfa") / TEST_SHARD_BASENAME)
    assert not is_val_shard("pdfa-eng-train-0042.tar")
    assert not is_test_shard("pdfa-eng-train-0042.tar")


def test_pretrain_chain_excludes_test_shard_from_train():
    """The chain script's ``TRAIN_SHARDS`` build must skip both
    locked basenames. A regression here is exactly the leak D1 was
    meant to prevent."""
    src = CHAIN_SH.read_text()
    assert VAL_SHARD_BASENAME in src, (
        "chain script must reference the locked val shard basename"
    )
    assert TEST_SHARD_BASENAME in src, (
        "chain script must reference the locked test shard basename"
    )
    # The exclusion pattern: both basenames appear in a `continue`
    # block that filters the train list.
    assert (
        '"$base" == "$VAL_BASE"' in src
        and '"$base" == "$TEST_BASE"' in src
    ), (
        "chain script must filter both VAL_BASE and TEST_BASE out of "
        "TRAIN_SHARDS"
    )


def test_pretrain_chain_passes_val_shard_not_test_to_stage_scripts():
    """Every ``--val-shard`` passed to a stage_run script must point at
    ``$VAL`` (the locked val), never ``$TEST``."""
    src = CHAIN_SH.read_text()
    # No stage_run.py invocation may pass the TEST var as --val-shard.
    assert '--val-shard "$TEST"' not in src
    # Every stage_run.py invocation must carry --val-shard "$VAL".
    # (We require all three; the chain has stages 1/2/3.)
    assert src.count('--val-shard "$VAL"') >= 3
