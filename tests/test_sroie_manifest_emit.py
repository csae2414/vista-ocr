"""SROIE flat-layout -> JSONL manifest emitter tests.

The emitter is the bridge between the operator's data prep
(``setup_sroie.sh``) and the manifest-driven CLI verbs
(``vista-ocr finetune`` / ``vista-ocr eval``). The tests verify that
the produced manifest is consumable by ``vista_ocr.data.manifest.iter_manifest``
and that the SROIE quad coordinates project to the expected AABB.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
EMITTER = REPO / "scripts" / "datasets" / "sroie_to_manifest.py"


def _make_sroie_split(root: Path, split: str, docs: list[tuple[str, list[str]]]) -> None:
    """Write a tiny SROIE flat layout. ``docs`` is a list of
    ``(id, [quad_line, ...])`` pairs."""
    sd = root / split
    sd.mkdir(parents=True, exist_ok=True)
    for doc_id, quad_lines in docs:
        Image.new("L", (200, 100), 255).save(sd / f"{doc_id}.jpg")
        (sd / f"{doc_id}.txt").write_text("\n".join(quad_lines) + "\n")


def _run_emitter(root: Path, split: str, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(EMITTER),
         "--root", str(root), "--split", split, "--out", str(out)],
        capture_output=True, text=True, timeout=30,
    )


def test_emit_minimal_manifest(tmp_path: Path):
    root = tmp_path / "sroie"
    _make_sroie_split(root, "test", [
        ("X51005230625", ["0,0,40,0,40,10,0,10,STORE NAME"]),
    ])
    manifest = tmp_path / "manifests" / "test.jsonl"
    r = _run_emitter(root, "test", manifest)
    assert r.returncode == 0, r.stderr

    [rec_str] = manifest.read_text().splitlines()
    rec = json.loads(rec_str)
    assert rec["ref"] == "STORE NAME"
    # SROIE quad (0,0,40,0,40,10,0,10) projects to AABB (0,0,40,10).
    assert rec["bboxes"] == [[0, 0, 40, 10, "STORE NAME"]]


def test_emit_concats_lines_into_ref(tmp_path: Path):
    root = tmp_path / "sroie"
    _make_sroie_split(root, "test", [
        ("doc1", [
            "0,0,40,0,40,10,0,10,FOO",
            "0,20,40,20,40,30,0,30,BAR",
        ]),
    ])
    manifest = tmp_path / "test.jsonl"
    r = _run_emitter(root, "test", manifest)
    assert r.returncode == 0, r.stderr
    [rec] = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert rec["ref"] == "FOO BAR"
    assert len(rec["bboxes"]) == 2


def test_emit_skips_empty_or_malformed(tmp_path: Path):
    root = tmp_path / "sroie"
    _make_sroie_split(root, "test", [
        ("good", ["0,0,40,0,40,10,0,10,OK"]),
        ("empty", [""]),                             # no valid lines
        ("malformed", ["not,a,quad"]),               # too few coords
    ])
    manifest = tmp_path / "test.jsonl"
    r = _run_emitter(root, "test", manifest)
    assert r.returncode == 0, r.stderr
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["ref"] == "OK"


def test_emitted_manifest_is_consumable_by_iter_manifest(tmp_path: Path):
    """The cross-component contract: emitted manifest must be a valid
    v1 input for ``vista_ocr.data.manifest.iter_manifest``."""
    from vista_ocr.data.manifest import iter_manifest

    root = tmp_path / "sroie"
    _make_sroie_split(root, "test", [
        ("doc1", ["10,20,100,20,100,50,10,50,FOO"]),
        ("doc2", ["0,0,40,0,40,10,0,10,BAR"]),
    ])
    manifest = tmp_path / "test.jsonl"
    r = _run_emitter(root, "test", manifest)
    assert r.returncode == 0, r.stderr

    samples = list(iter_manifest(manifest))
    assert len(samples) == 2
    # Reference text matches the SROIE-style whitespace-joined ref.
    refs = sorted(" ".join(ln.text for ln in s.lines) for s in samples)
    assert refs == ["BAR", "FOO"]
    # AABB projection round-tripped through the manifest.
    s_foo = next(s for s in samples if s.lines[0].text == "FOO")
    assert s_foo.lines[0].bbox == (10, 20, 100, 50)


def test_setup_sroie_sh_help_lists_emit_flag():
    """The shell wrapper grew a --emit-manifests flag; smoke that the
    new flag is documented in the script header."""
    setup_sh = REPO / "scripts" / "datasets" / "setup_sroie.sh"
    text = setup_sh.read_text()
    assert "--emit-manifests" in text
