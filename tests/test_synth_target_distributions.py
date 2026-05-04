"""Tests for the J0 #3 distribution-baseline tool.

Inventory (notes/plan_phase_j_followup.md §Fix 3):

- ``test_pdfa_path_calls_iter_pdfa_with_config`` -- iter_pdfa is
  invoked with a PdfaConfig, NOT a bare list/string, and the tool
  emits the schema we expect from in-memory Sample fakes.
- ``test_idl_path_calls_iter_idl_with_config`` -- mirror.
- ``test_brace_expansion_resolves_pattern`` -- direct unit test
  of _expand_shards on a tmp dir with three matching files.
- ``test_locked_val_test_shards_rejected`` -- the val/test reject
  step refuses to proceed when 0118 / 0119 are in the glob match.

Tests use monkeypatch on the tool module's iter_pdfa / iter_idl
references (NOT empty fake .tar shards). Empty WebDataset tar
files fail in WebDataset's reader for unrelated reasons; a green
test would prove nothing about this tool.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from vista_ocr.data.pdfa import PdfaConfig
from vista_ocr.data.idl import IdlConfig
from vista_ocr.data.types import Sample
from vista_ocr.tokenizer.tokenizer import Line


REPO = Path(__file__).resolve().parent.parent
TOOL_PATH = REPO / "tools" / "synth_target_distributions.py"


@pytest.fixture(scope="module")
def tool():
    """Import the tool from its file path; ``tools/`` is not a
    Python package so the standard ``import`` won't reach it."""
    spec = importlib.util.spec_from_file_location(
        "synth_target_distributions_tool", TOOL_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_samples(n: int = 3):
    """Yield n in-memory Samples. Lines have non-degenerate bboxes
    so the percentile computation has something to work with."""
    img = Image.new("L", (200, 100), 255)
    for i in range(n):
        lines = [
            Line(text=f"hello world {i}", bbox=(10, 10 + i * 20, 100, 25 + i * 20)),
        ]
        yield Sample(image=img, lines=lines, task="ocr_layout", source="fake_pdfa")


# ---------------------------------------------------------------------------
# brace expansion
# ---------------------------------------------------------------------------

def test_brace_expansion_resolves_pattern(tool, tmp_path: Path):
    """Single-range brace pattern must expand to all matching files,
    sorted, when those files exist on disk."""
    for i in (1, 2, 3):
        (tmp_path / f"x-{i:04d}.tar").write_bytes(b"")
    pattern = str(tmp_path / "x-{0001..0003}.tar")
    out = tool._expand_shards(pattern)
    assert [Path(p).name for p in out] == ["x-0001.tar", "x-0002.tar", "x-0003.tar"]


def test_brace_expansion_matches_only_existing_files(tool, tmp_path: Path):
    """The pattern's range exceeds what's on disk; expander must
    silently drop missing files, not raise."""
    (tmp_path / "x-0001.tar").write_bytes(b"")
    (tmp_path / "x-0003.tar").write_bytes(b"")
    pattern = str(tmp_path / "x-{0001..0005}.tar")
    out = tool._expand_shards(pattern)
    assert [Path(p).name for p in out] == ["x-0001.tar", "x-0003.tar"]


def test_plain_glob_still_works(tool, tmp_path: Path):
    """No brace range → falls back to plain glob.glob()."""
    (tmp_path / "x-0001.tar").write_bytes(b"")
    (tmp_path / "x-0002.tar").write_bytes(b"")
    out = tool._expand_shards(str(tmp_path / "x-*.tar"))
    assert [Path(p).name for p in out] == ["x-0001.tar", "x-0002.tar"]


# ---------------------------------------------------------------------------
# locked val/test rejection
# ---------------------------------------------------------------------------

def test_locked_val_test_shards_rejected(tool, tmp_path: Path):
    """A glob that pulls in 0118 / 0119 must hard-fail rather than
    silently filter. The locked split documented in
    vista_ocr.data.split is load-bearing for BENCHMARKS hold-out
    semantics; a tool that touches it is a regression."""
    for name in (
        "pdfa-eng-train-0117.tar",
        "pdfa-eng-train-0118.tar",   # locked val
        "pdfa-eng-train-0119.tar",   # locked test
    ):
        (tmp_path / name).write_bytes(b"")
    pattern = str(tmp_path / "pdfa-eng-train-*.tar")
    paths = tool._expand_shards(pattern)
    with pytest.raises(SystemExit) as exc:
        tool._reject_locked_shards(paths)
    msg = str(exc.value)
    assert "refusing" in msg
    assert "pdfa-eng-train-0118.tar" in msg
    assert "pdfa-eng-train-0119.tar" in msg


def test_no_locked_shards_passes_through(tool, tmp_path: Path):
    """Pattern that matches only train shards passes the rejector
    unchanged (does not raise)."""
    (tmp_path / "pdfa-eng-train-0001.tar").write_bytes(b"")
    (tmp_path / "pdfa-eng-train-0002.tar").write_bytes(b"")
    paths = tool._expand_shards(str(tmp_path / "pdfa-eng-train-*.tar"))
    out = tool._reject_locked_shards(paths)
    assert [Path(p).name for p in out] == [
        "pdfa-eng-train-0001.tar", "pdfa-eng-train-0002.tar",
    ]


# ---------------------------------------------------------------------------
# iter_pdfa / iter_idl wrapping in {Pdfa,Idl}Config
# ---------------------------------------------------------------------------

def test_pdfa_path_calls_iter_pdfa_with_config(tool, tmp_path: Path, monkeypatch):
    """The bug we're fixing: tool used to call iter_pdfa(args.shards)
    with a bare string. Verify it now passes a PdfaConfig."""
    received: dict[str, object] = {}

    def fake_iter_pdfa(cfg):
        received["cfg"] = cfg
        yield from _fake_samples(3)

    monkeypatch.setattr("vista_ocr.data.pdfa.iter_pdfa", fake_iter_pdfa)
    # Also stage at least one matching shard so the rejector + glob pass.
    (tmp_path / "pdfa-eng-train-0001.tar").write_bytes(b"")
    out_json = tmp_path / "out.json"
    monkeypatch.setattr(
        sys, "argv",
        ["synth_target_distributions.py",
         "--source", "pdfa",
         "--shards", str(tmp_path / "pdfa-eng-train-{0001..0001}.tar"),
         "--out", str(out_json),
         "--n", "3"],
    )
    rc = tool.main()
    assert rc == 0
    assert isinstance(received["cfg"], PdfaConfig)
    assert [Path(s).name for s in received["cfg"].shards] == ["pdfa-eng-train-0001.tar"]
    report = json.loads(out_json.read_text())
    assert report["source"] == "pdfa"
    assert report["n_samples"] == 3
    for key in ("line_height_px", "words_per_line", "chars_per_line",
                "bbox_aspect_ratio", "page_text_density", "page_total_chars"):
        assert key in report
        assert "p50" in report[key]


def test_idl_path_calls_iter_idl_with_config(tool, tmp_path: Path, monkeypatch):
    received: dict[str, object] = {}

    def fake_iter_idl(cfg):
        received["cfg"] = cfg
        yield from _fake_samples(2)

    monkeypatch.setattr("vista_ocr.data.idl.iter_idl", fake_iter_idl)
    (tmp_path / "idl-train-0001.tar").write_bytes(b"")
    out_json = tmp_path / "out.json"
    monkeypatch.setattr(
        sys, "argv",
        ["synth_target_distributions.py",
         "--source", "idl",
         "--shards", str(tmp_path / "idl-train-*.tar"),
         "--out", str(out_json),
         "--n", "2"],
    )
    rc = tool.main()
    assert rc == 0
    assert isinstance(received["cfg"], IdlConfig)
    assert [Path(s).name for s in received["cfg"].shards] == ["idl-train-0001.tar"]
    report = json.loads(out_json.read_text())
    assert report["source"] == "idl"
    assert report["n_samples"] == 2
