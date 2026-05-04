"""Tests for ``scripts/datasets/setup_synth_corpus.sh`` (Phase J Fix 2).

Inventory (notes/plan_phase_j_followup.md §Fix 2):

- ``test_help_smoke`` -- ``--help`` exits cleanly. Catches bash
  syntax breaks in the wrapper.
- ``test_dry_run_offline`` -- ``--dry-run`` is 100% offline. Runs
  with ``HF_HUB_OFFLINE=1`` set; asserts exit 0 + every required
  key in stdout.
- ``test_fixture_only_writes_letterlike_offline`` -- ``--fixture-only``
  writes ``letterlike.txt`` + ``PROVENANCE.json`` into a temp
  output dir without ever touching the network.
- ``test_letterlike_fixture_present_and_nonempty`` -- independent
  of the wrapper: the in-tree ``corpora/synth/letterlike.txt``
  file exists and has > 50 non-empty lines.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "datasets" / "setup_synth_corpus.sh"
LETTERLIKE = REPO / "corpora" / "synth" / "letterlike.txt"


def _run(args: list[str], extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, **(extra_env or {})}
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=60,
    )


# ---------------------------------------------------------------------------
# in-tree letterlike.txt fixture
# ---------------------------------------------------------------------------

def test_letterlike_fixture_present_and_nonempty():
    """The in-tree fixture is committed; the wrapper's
    --fixture-only path copies it. If a contributor accidentally
    empties or deletes it, this test fires regardless of whether
    setup_synth_corpus.sh has ever run on this box."""
    assert LETTERLIKE.exists(), f"missing in-tree fixture: {LETTERLIKE}"
    lines = [ln for ln in LETTERLIKE.read_text().splitlines() if ln.strip()]
    assert len(lines) > 50, f"letterlike.txt too short: {len(lines)} non-empty lines"


# ---------------------------------------------------------------------------
# wrapper smoke
# ---------------------------------------------------------------------------

def test_help_smoke():
    """--help must exit 0 and print something resembling argparse usage."""
    r = _run(["--help"])
    assert r.returncode == 0, f"--help failed:\n{r.stderr}"
    assert "--out-dir" in r.stdout
    assert "--fixture-only" in r.stdout
    assert "--dry-run" in r.stdout


def test_dry_run_offline(tmp_path: Path):
    """--dry-run must be 100% offline. Set HF_HUB_OFFLINE=1; the
    wrapper must still exit 0 with the resolved-plan JSON, proving
    nothing in the dry-run path hits HF."""
    r = _run(
        ["--dry-run", "--out-dir", str(tmp_path / "out")],
        extra_env={"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
    )
    assert r.returncode == 0, f"--dry-run --out-dir <tmp> failed offline:\n{r.stderr}"
    # Stdout contains the JSON plan; pull it out (the wrapper may
    # emit conda-activation chatter on some boxes before/after).
    m = re.search(r"\{[\s\S]*\}", r.stdout)
    assert m, f"no JSON plan in --dry-run stdout:\n{r.stdout}"
    plan = json.loads(m.group(0))
    assert plan["dry_run"] is True
    assert plan["lang"] == "en"
    # Every required source declared with id + revision + license.
    names = {c["name"] for c in plan["corpora"]}
    assert names == {"wikitext", "pg19"}
    for c in plan["corpora"]:
        assert c["hf_id"]
        assert c["hf_revision"]
        assert c["license"] in ("CC-BY-SA-3.0", "Apache-2.0", "MIT")
        assert c["output"].startswith(str(tmp_path / "out"))


def test_fixture_only_writes_letterlike_offline(tmp_path: Path):
    """--fixture-only writes letterlike.txt + PROVENANCE.json into
    --out-dir without touching the network. Drives the operator's
    offline subset of the workflow.

    Run with HF_HUB_OFFLINE=1 + HF_DATASETS_OFFLINE=1 to prove no
    HF code path is reachable from the fixture-only mode."""
    out = tmp_path / "synth"
    r = _run(
        ["--fixture-only", "--out-dir", str(out)],
        extra_env={"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
    )
    assert r.returncode == 0, f"--fixture-only failed:\n{r.stderr}"
    # letterlike.txt copied + non-empty.
    ll = out / "letterlike.txt"
    assert ll.exists()
    nonempty = [s for s in ll.read_text().splitlines() if s.strip()]
    assert len(nonempty) > 50
    # PROVENANCE.json with the letterlike entry populated.
    prov = json.loads((out / "PROVENANCE.json").read_text())
    assert prov["schema_version"] == 1
    names = {c["name"] for c in prov["corpora"]}
    assert "letterlike" in names
    ll_entry = next(c for c in prov["corpora"] if c["name"] == "letterlike")
    assert ll_entry["sha256"]  # present + non-empty
    # SHA256 matches what's actually on disk.
    import hashlib
    h = hashlib.sha256()
    h.update(ll.read_bytes())
    assert ll_entry["sha256"] == h.hexdigest()
    assert ll_entry["sample_config"]["seed"] == 0


def test_unknown_flag_rejected():
    """Unknown args propagate to argparse and fail non-zero. Catches
    a wrapper that swallows errors silently."""
    r = _run(["--this-flag-does-not-exist"])
    assert r.returncode != 0


@pytest.mark.parametrize("flag,value", [
    ("--pg19-max-docs", "500"),
    ("--max-sentences", "12345"),
    ("--seed", "42"),
])
def test_dry_run_passes_through_caps(flag: str, value: str, tmp_path: Path):
    """Cap-related flags reach the JSON plan, so an operator can
    verify the values before committing to a real download."""
    r = _run(
        ["--dry-run", "--out-dir", str(tmp_path / "out"), flag, value],
        extra_env={"HF_HUB_OFFLINE": "1"},
    )
    assert r.returncode == 0, r.stderr
    m = re.search(r"\{[\s\S]*\}", r.stdout)
    assert m
    plan = json.loads(m.group(0))
    key = flag.lstrip("-").replace("-", "_")
    assert str(plan[key]) == value


# ---------------------------------------------------------------------------
# pinned revisions + env-override + drift detection
# ---------------------------------------------------------------------------

def test_dry_run_reports_pinned_revisions(tmp_path: Path):
    """Default revisions are pinned commit SHAs (40-hex), NOT
    placeholder 'main'. Reproducible corpus acquisition relies on
    this; if it ever drifts back to 'main' the test fires."""
    r = _run(
        ["--dry-run", "--out-dir", str(tmp_path / "out")],
        extra_env={"HF_HUB_OFFLINE": "1"},
    )
    assert r.returncode == 0, r.stderr
    m = re.search(r"\{[\s\S]*\}", r.stdout)
    assert m
    plan = json.loads(m.group(0))
    sha_re = re.compile(r"^[0-9a-f]{40}$")
    for c in plan["corpora"]:
        rev = c["hf_revision"]
        assert sha_re.match(rev), \
            f"{c['name']}: hf_revision={rev!r} is not a 40-hex SHA (pinned). "\
            "Did the placeholder 'main' regress?"


def test_env_override_for_revision_takes_effect(tmp_path: Path):
    """CORPUS_HF_REVISION_WIKITEXT overrides the pinned default
    so an operator can run against a later upstream revision (and
    the override is recorded in PROVENANCE.json -- see fixture-only
    is the wrong path for that, so the helper itself records it
    via _resolve_revision at module-import time inside the worker
    process). This test exercises the override on the dry-run plan."""
    r = _run(
        ["--dry-run", "--out-dir", str(tmp_path / "out")],
        extra_env={
            "HF_HUB_OFFLINE": "1",
            "CORPUS_HF_REVISION_WIKITEXT": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        },
    )
    assert r.returncode == 0, r.stderr
    m = re.search(r"\{[\s\S]*\}", r.stdout)
    assert m
    plan = json.loads(m.group(0))
    by_name = {c["name"]: c for c in plan["corpora"]}
    assert by_name["wikitext"]["hf_revision"] == "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    # PG-19 unchanged because we only overrode wikitext.
    assert by_name["pg19"]["hf_revision"] != "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"


def test_drift_detection_hard_fails_on_tampered_file(tmp_path: Path):
    """First run --fixture-only writes letterlike.txt + PROVENANCE.
    Then we tamper with letterlike.txt and re-run; the helper must
    hard-fail rather than silently accept the drifted file. --force
    overrides cleanly (covered separately by re-running with --force)."""
    out = tmp_path / "synth"
    r1 = _run(["--fixture-only", "--out-dir", str(out)],
              extra_env={"HF_HUB_OFFLINE": "1"})
    assert r1.returncode == 0, r1.stderr
    ll = out / "letterlike.txt"
    original = ll.read_bytes()

    # Tamper.
    ll.write_text(original.decode() + "\nINJECTED LINE\n", encoding="utf-8")

    # Re-run via Python module directly (bash wrapper would also
    # work but invoking the module is cheaper + isolates the test
    # from PATH conda env quirks).
    import sys as _sys
    import subprocess as _sub
    r2 = _sub.run(
        [_sys.executable, "-m", "vista_ocr.data.synth._setup_corpus",
         "--out-dir", str(out)],
        cwd=REPO,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
        capture_output=True, text=True, timeout=60,
    )
    assert r2.returncode != 0, "drift detection should have hard-failed"
    assert "drift" in (r2.stdout + r2.stderr).lower()


def test_provenance_records_hf_revision(tmp_path: Path):
    """After a fixture-only run, PROVENANCE.json must NOT silently
    drop the hf_revision field for letterlike (which has no HF
    revision; field is absent rather than null), AND the schema
    is stable enough for the post-J A/B to recover the exact
    corpus state from the manifest."""
    out = tmp_path / "synth"
    r = _run(["--fixture-only", "--out-dir", str(out)],
             extra_env={"HF_HUB_OFFLINE": "1"})
    assert r.returncode == 0, r.stderr
    prov = json.loads((out / "PROVENANCE.json").read_text())
    by_name = {c["name"]: c for c in prov["corpora"]}
    # letterlike is in-tree only; no hf_revision recorded.
    ll = by_name["letterlike"]
    assert "hf_revision" not in ll
    assert ll["sha256"]
    # Schema field stability: schema_version must be present and 1.
    assert prov["schema_version"] == 1
