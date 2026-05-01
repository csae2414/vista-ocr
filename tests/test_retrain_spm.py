"""Tests for scripts/retrain_spm_on_pdfa.py.

Exercises the pure-Python helpers (filter, shard reader, corpus writer)
on fabricated tar fixtures. The full SPM-training step is exercised on a
tiny synthetic corpus to keep CI fast.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

# Load the script as a module by path -- it lives in scripts/ which is
# not a Python package.
_THIS = Path(__file__).resolve()
_SCRIPT = _THIS.parent.parent / "scripts" / "retrain_spm_on_pdfa.py"
_spec = importlib.util.spec_from_file_location("retrain_spm_on_pdfa", _SCRIPT)
retrain = importlib.util.module_from_spec(_spec)
sys.modules["retrain_spm_on_pdfa"] = retrain
_spec.loader.exec_module(retrain)


# ---------- filter ----------

def test_is_useful_line_drops_short():
    assert not retrain.is_useful_line("ab")
    assert retrain.is_useful_line("abc")


def test_is_useful_line_drops_mostly_numeric():
    assert not retrain.is_useful_line("12345 67890")
    assert not retrain.is_useful_line("$$$ $$$ $$$")


def test_is_useful_line_keeps_normal_english():
    assert retrain.is_useful_line("hello world")
    assert retrain.is_useful_line("Receipt total: $42")


def test_is_useful_line_threshold_strict():
    """Exactly at 0.5 should pass. Below should fail."""
    text = "ab12"  # 2/4 = 0.5
    assert retrain.is_useful_line(text, min_alpha_frac=0.5)
    text2 = "a123"  # 1/4 = 0.25
    assert not retrain.is_useful_line(text2, min_alpha_frac=0.5)


def test_is_useful_line_handles_empty():
    assert not retrain.is_useful_line("")


# ---------- iter_lines_from_shard ----------

def _make_fake_pdfa_shard(path: Path, payloads: list[dict]) -> None:
    """Write a tar containing ``{i}.json`` files with the given payloads."""
    with tarfile.open(path, "w") as tar:
        for i, p in enumerate(payloads):
            data = json.dumps(p).encode("utf-8")
            info = tarfile.TarInfo(name=f"{i}.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_iter_lines_from_shard_reads_pages_lines_text(tmp_path: Path):
    shard = tmp_path / "fake.tar"
    _make_fake_pdfa_shard(shard, [
        {"pages": [{"lines": {"text": ["hello", "world"]}}]},
        {"pages": [
            {"lines": {"text": ["foo"]}},
            {"lines": {"text": ["bar", "baz"]}},
        ]},
    ])
    out = list(retrain.iter_lines_from_shard(shard))
    assert out == ["hello", "world", "foo", "bar", "baz"]


def test_iter_lines_from_shard_skips_malformed_json(tmp_path: Path):
    shard = tmp_path / "fake.tar"
    with tarfile.open(shard, "w") as tar:
        # one good json
        good = json.dumps({"pages": [{"lines": {"text": ["ok"]}}]}).encode()
        info = tarfile.TarInfo(name="0.json")
        info.size = len(good)
        tar.addfile(info, io.BytesIO(good))
        # one broken json
        bad = b"{ not valid json"
        info = tarfile.TarInfo(name="1.json")
        info.size = len(bad)
        tar.addfile(info, io.BytesIO(bad))
    out = list(retrain.iter_lines_from_shard(shard))
    assert out == ["ok"]


def test_iter_lines_from_shard_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        list(retrain.iter_lines_from_shard(tmp_path / "nope.tar"))


def test_iter_lines_from_shard_handles_missing_keys(tmp_path: Path):
    """Pages without 'lines', or lines without 'text', must not crash."""
    shard = tmp_path / "fake.tar"
    _make_fake_pdfa_shard(shard, [
        {},
        {"pages": [{}, {"lines": {}}]},
        {"pages": [{"lines": {"text": ["only"]}}]},
    ])
    out = list(retrain.iter_lines_from_shard(shard))
    assert out == ["only"]


# ---------- write_corpus ----------

def test_write_corpus_filters_and_counts(tmp_path: Path):
    out = tmp_path / "corpus.txt"
    lines = [
        "the quick brown fox",
        "ab",                     # too short -> drop
        "1 2 3 4 5 6",            # mostly numeric -> drop
        "hello world",
        "",                       # empty -> drop
    ]
    kept, dropped = retrain.write_corpus(iter(lines), out)
    assert kept == 2
    assert dropped == 3
    contents = out.read_text(encoding="utf-8").splitlines()
    assert contents == ["the quick brown fox", "hello world"]


def test_write_corpus_creates_parent_dir(tmp_path: Path):
    out = tmp_path / "deep" / "nest" / "corpus.txt"
    retrain.write_corpus(iter(["hello world"]), out)
    assert out.exists()


# ---------- end-to-end SPM train on tiny corpus ----------

def test_train_spm_on_filtered_corpus(tmp_path: Path):
    """Smoke test: filter -> corpus -> SPM train -> file exists."""
    from vista_ocr.tokenizer.build_spm import train_spm
    from vista_ocr.tokenizer.spatial_tokens import SpatialGrid
    from vista_ocr.tokenizer.tokenizer import (
        VistaTokenizer,
        list_special_and_spatial_tokens,
    )

    # Build a tiny corpus that survives the filter.
    raw = [
        "the quick brown fox jumps over the lazy dog",
        "Sphinx of black quartz judge my vow",
        "Pack my box with five dozen liquor jugs",
        "Receipt total: forty two dollars",
        "Hello world how are you today",
    ] * 200
    corpus = tmp_path / "c.txt"
    retrain.write_corpus(iter(raw), corpus)

    grid = SpatialGrid(canvas_h=128, canvas_w=128, quantizer_px=4, scheme="original")
    out_prefix = tmp_path / "test_spm"
    model_path = train_spm(
        corpus_path=corpus,
        out_prefix=out_prefix,
        vocab_size=180,
        user_symbols=list_special_and_spatial_tokens(grid),
    )
    assert model_path.exists()
    # The resulting SPM model should load and have the spatial tokens
    # registered as user_defined_symbols.
    tk = VistaTokenizer(model_path, grid)
    for tok in grid.all_tokens()[:5]:
        assert tk.piece_to_id(tok) != tk.unk_id
