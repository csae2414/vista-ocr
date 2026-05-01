"""Tests for scripts/measure_pdfa_distribution.py.

Closes the B4 gap: prior to this file the script had zero tests.
We import the script's helpers as plain Python (no subprocess) and
exercise them on synthetic data.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def measure_module():
    """Import the script as a module without running its argparse."""
    repo = Path(__file__).resolve().parent.parent
    path = repo / "scripts" / "measure_pdfa_distribution.py"
    spec = importlib.util.spec_from_file_location("_measure", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_measure"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_percentile_empty(measure_module):
    assert measure_module._percentile([], 50) == 0


def test_percentile_known_values(measure_module):
    """Percentile uses int(round(q/100 * (n-1))) — pin the formula."""
    vals = [1, 2, 3, 4, 5]
    assert measure_module._percentile(vals, 0) == 1
    assert measure_module._percentile(vals, 50) == 3
    assert measure_module._percentile(vals, 100) == 5


def test_percentile_clamps_out_of_range(measure_module):
    vals = [1, 2, 3]
    assert measure_module._percentile(vals, 150) == 3
    assert measure_module._percentile(vals, -5) == 1


def _make_shard(path: Path, pages: list[dict]) -> None:
    payload = json.dumps({"pages": pages}).encode("utf-8")
    with tarfile.open(path, "w") as tar:
        info = tarfile.TarInfo("rec.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))


def test_main_counts_lines_and_words_per_page(tmp_path, measure_module, capsys):
    """End-to-end: synthetic shard with 2 pages -> percentiles are computed."""
    shard = tmp_path / "synthetic.tar"
    _make_shard(shard, [
        {"lines": {"text": ["alpha beta", "gamma"]}},        # 2 lines, 3 words
        {"lines": {"text": ["one", "two three", "four"]}},   # 3 lines, 4 words
    ])
    sys.argv = ["measure", "--shards", str(shard), "--percentiles", "50", "100"]
    measure_module.main()
    out = capsys.readouterr().out

    # pages count + percentile rows show in the printed output.
    assert "pages   : 2" in out
    assert "lines per page" in out
    assert "words per page" in out
    # p100 of {2, 3} = 3; p100 of {3, 4} = 4.
    assert "p100.0:" in out  # rendering of percentile rows
