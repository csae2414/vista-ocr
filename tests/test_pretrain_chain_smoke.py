"""Smoke tests for ``scripts/pretrain_chain.sh``.

Three static-text tests + one runtime ``DRY_RUN=1`` test.

Inventory:

* ``test_chain_forwards_distinct_select_on_per_stage`` -- each of the
  three stage_run.py invocations references its own per-stage env
  variable (catches drift back to a single SELECT_ON for all stages,
  which is the bug Run C hit at step 10500).
* ``test_chain_defaults_stage1_calibration_settings`` -- defaults
  pin stage 1 to (val_loss, patience=40, init from ckpt_final.pt).
  Pins the AI-engineer reasoning behind those numbers.
* ``test_chain_auto_cap_present`` -- the AUTO-CAP block exists for
  all three stages. Forces a future "fix" that removes auto-capping
  to do so deliberately (and update this test).
* ``test_chain_dry_run_resolves_correct_per_stage_args`` -- runtime
  test invoking the chain with ``DRY_RUN=1`` and asserting the
  resolved arglist for each stage carries the expected
  ``--select-on`` and ``--early-stop-patience`` flags. Catches
  wiring bugs (env-var typos, shell quoting) that static-text
  scans cannot.

The runtime test needs the locked val + test PDFA shards on disk;
on a CI box without them, the chain exits before printing the
arglists. Test fakes those shards via temp dir + ``cd`` so we can
drive the chain to its DRY_RUN print without needing real data.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CHAIN_SH = REPO / "scripts" / "pretrain_chain.sh"


def test_chain_forwards_distinct_select_on_per_stage():
    """Stage 1 must use STAGE1_SELECT_ON; stage 2 STAGE2_SELECT_ON;
    stage 3 STAGE3_SELECT_ON. A single SELECT_ON forwarded to all
    three is the Run-C-stage-1-stuck-at-step-10500 bug."""
    src = CHAIN_SH.read_text()
    assert '--select-on "$STAGE1_SELECT_ON"' in src
    assert '--select-on "$STAGE2_SELECT_ON"' in src
    assert '--select-on "$STAGE3_SELECT_ON"' in src
    # The legacy whole-chain SELECT_ON is allowed to exist (back-compat
    # shim that overrides the per-stage values), but must not be passed
    # directly to any stage's --select-on flag.
    assert '--select-on "$SELECT_ON"' not in src


def test_chain_defaults_stage1_calibration_settings():
    """Stage 1's defaults differ from stages 2/3 in three places:

    * select_on = val_loss (frozen decoder; word_f1 is structurally 0)
    * patience  = 40       (val_loss is bouncier than word_f1)
    * stage 2 init source is ckpt_final.pt, not ckpt_best.pt
      (stage 1's val_loss minimum can be a transient; final state
      is the fully-calibrated encoder we want)
    """
    src = CHAIN_SH.read_text()
    assert 'STAGE1_SELECT_ON="${STAGE1_SELECT_ON:-val_loss}"' in src
    assert 'STAGE2_SELECT_ON="${STAGE2_SELECT_ON:-val_word_f1}"' in src
    assert 'STAGE3_SELECT_ON="${STAGE3_SELECT_ON:-val_word_f1}"' in src
    assert 'STAGE1_PATIENCE="${STAGE1_PATIENCE:-40}"' in src
    assert 'STAGE2_PATIENCE="${STAGE2_PATIENCE:-10}"' in src
    assert 'STAGE3_PATIENCE="${STAGE3_PATIENCE:-15}"' in src
    # Stage 2 inits from ckpt_final.pt (default) with ckpt_best.pt
    # fallback when save_final wasn't on.
    assert 'STAGE1_INIT_CKPT_DEFAULT="checkpoints/stage1/ckpt_final.pt"' in src
    assert 'STAGE1_INIT_CKPT_FALLBACK="checkpoints/stage1/ckpt_best.pt"' in src


def test_chain_auto_cap_present():
    """Auto-cap STAGE_N_STEPS when EARLY_STOP=1 would fire first.
    Without this, an operator who sets STAGE1_STEPS=50000 with
    EARLY_STOP=1 + patience=40 sees the chain stop at step ~22500
    with no clear log signal -- silent budget truncation."""
    src = CHAIN_SH.read_text()
    assert "AUTO-CAP:" in src
    # All three stages get capped.
    assert "STAGE1_STEPS=$s1_cap" in src
    assert "STAGE2_STEPS=$s2_cap" in src
    assert "STAGE3_STEPS=$s3_cap" in src


# ---------- Runtime DRY_RUN test ----------------------------------------


def _setup_fake_repo_for_dry_run(tmp: Path) -> Path:
    """Build a minimal repo skeleton sufficient for the chain's
    pre-stage validation to pass:

    * ``data/raw/pdfa/pdfa-eng-train-{0001,0118,0119}.tar`` -- empty
      tar files just to exist (chain checks ``-f``).
    * ``data/processed/vocab/sp_en_16k.model`` -- empty file.
    * ``scripts/pretrain_chain.sh`` -- copied from the real repo.

    The chain needs at least 1 train shard + the locked val + test
    shards, so we ship 3.
    """
    (tmp / "data/raw/pdfa").mkdir(parents=True)
    (tmp / "data/processed/vocab").mkdir(parents=True)
    (tmp / "scripts").mkdir(parents=True)
    (tmp / "checkpoints").mkdir(parents=True)
    (tmp / "logs").mkdir(parents=True)

    for shard in ("pdfa-eng-train-0001.tar",
                  "pdfa-eng-train-0118.tar",
                  "pdfa-eng-train-0119.tar"):
        (tmp / "data/raw/pdfa" / shard).write_bytes(b"")
    (tmp / "data/processed/vocab/sp_en_16k.model").write_bytes(b"")

    # Phase H: optional IDL shard fixtures, only created when the test
    # asks for the pdfa+idl mix. Tests that don't need them just pass
    # DATA_MIX=pdfa (the default).
    (tmp / "data/raw/idl").mkdir(parents=True)
    for shard in ("idl-train-0001.tar", "idl-train-0002.tar"):
        (tmp / "data/raw/idl" / shard).write_bytes(b"")

    shutil.copy(CHAIN_SH, tmp / "scripts/pretrain_chain.sh")
    os.chmod(tmp / "scripts/pretrain_chain.sh", 0o755)
    return tmp


def _run_dry(env_overrides: dict) -> str:
    """Invoke the chain in DRY_RUN=1 mode in a fake repo; return stdout."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        _setup_fake_repo_for_dry_run(tmp)
        # The conda activation block in the chain looks for
        # /opt/miniconda3 etc.; if present on the CI box, it'll
        # source it. That's a no-op for DRY_RUN.
        env = {**os.environ, "DRY_RUN": "1", **env_overrides}
        r = subprocess.run(
            ["bash", "scripts/pretrain_chain.sh"],
            cwd=tmp, env=env,
            capture_output=True, text=True, timeout=30,
        )
        return r.stdout + "\n--STDERR--\n" + r.stderr


def test_chain_dry_run_resolves_correct_per_stage_args():
    """End-to-end: run the chain script with DRY_RUN=1 + EARLY_STOP=1
    and verify each stage's arglist has the right per-stage
    --select-on and --early-stop-patience."""
    out = _run_dry({"EARLY_STOP": "1"})

    # Stage 1: val_loss + patience 40
    s1_match = re.search(r"^DRY_RUN \[stage1\]: (.+)$", out, re.MULTILINE)
    assert s1_match, f"stage1 DRY_RUN line missing in:\n{out}"
    s1_args = s1_match.group(1)
    assert "--select-on val_loss" in s1_args
    assert "--early-stop-patience 40" in s1_args
    assert "--early-stop" in s1_args

    # Stage 2: val_word_f1 + patience 10 + init from ckpt_final
    s2_match = re.search(r"^DRY_RUN \[stage2\]: (.+)$", out, re.MULTILINE)
    assert s2_match, f"stage2 DRY_RUN line missing in:\n{out}"
    s2_args = s2_match.group(1)
    assert "--select-on val_word_f1" in s2_args
    assert "--early-stop-patience 10" in s2_args
    # In DRY_RUN we never run stage 1, so ckpt_final.pt doesn't exist;
    # but the DRY_RUN code path skips the file existence check and
    # uses the default. Test for the default path.
    assert "checkpoints/stage1/ckpt_final.pt" in s2_args

    # Stage 3: val_word_f1 + patience 15
    s3_match = re.search(r"^DRY_RUN \[stage3\]: (.+)$", out, re.MULTILINE)
    assert s3_match, f"stage3 DRY_RUN line missing in:\n{out}"
    s3_args = s3_match.group(1)
    assert "--select-on val_word_f1" in s3_args
    assert "--early-stop-patience 15" in s3_args


def test_chain_dry_run_legacy_select_on_overrides_all_stages():
    """Setting SELECT_ON=val_loss whole-chain must propagate to every
    stage's --select-on flag (back-compat with pre-fix scripts)."""
    out = _run_dry({"EARLY_STOP": "0", "SELECT_ON": "val_loss"})
    for stage in ("stage1", "stage2", "stage3"):
        m = re.search(rf"^DRY_RUN \[{stage}\]: (.+)$", out, re.MULTILINE)
        assert m, f"{stage} DRY_RUN line missing in:\n{out}"
        assert "--select-on val_loss" in m.group(1), (
            f"{stage}: SELECT_ON whole-chain override didn't propagate"
        )


def test_chain_dry_run_auto_cap_fires_when_steps_exceeds_cap():
    """STAGE1_STEPS=99999 with EARLY_STOP=1 + patience=40 has cap
    500*(5+40) = 22500. Chain should print AUTO-CAP and the stage 1
    --steps arg should be 22500, not 99999."""
    out = _run_dry({
        "EARLY_STOP": "1",
        "STAGE1_STEPS": "99999",
        "STAGE1_PATIENCE": "40",
    })
    assert "AUTO-CAP: STAGE1_STEPS=99999" in out
    s1_match = re.search(r"^DRY_RUN \[stage1\]: (.+)$", out, re.MULTILINE)
    assert s1_match
    # The arglist should have --steps 22500 (the cap), not 99999.
    assert "--steps 22500" in s1_match.group(1)
    assert "--steps 99999" not in s1_match.group(1)


# ---------- Phase H: DATA_MIX + paper preset --------------------------


def test_phase_h_default_data_mix_is_pdfa_only():
    """Default DATA_MIX=pdfa keeps the legacy invocation: stages 2+3
    do NOT receive --idl-shards."""
    out = _run_dry({"DATA_MIX": "pdfa"})
    for stage in ("stage1", "stage2", "stage3"):
        m = re.search(rf"^DRY_RUN \[{stage}\]: (.+)$", out, re.MULTILINE)
        assert m
        assert "--idl-shards" not in m.group(1), (
            f"{stage}: DATA_MIX=pdfa leaked --idl-shards"
        )


def test_phase_h_pdfa_plus_idl_propagates_to_stages_2_and_3_only():
    """DATA_MIX=pdfa+idl: stage 1 stays PDFA-only (frozen decoder
    calibration); stages 2+3 get --idl-shards."""
    out = _run_dry({"DATA_MIX": "pdfa+idl"})

    s1 = re.search(r"^DRY_RUN \[stage1\]: (.+)$", out, re.MULTILINE)
    assert s1
    assert "--idl-shards" not in s1.group(1), (
        "stage 1 must not receive --idl-shards (frozen decoder calibration "
        "is PDFA-only by design; mix would add noise without benefit)"
    )

    for stage in ("stage2", "stage3"):
        m = re.search(rf"^DRY_RUN \[{stage}\]: (.+)$", out, re.MULTILINE)
        assert m, f"{stage} arglist missing"
        assert "--idl-shards" in m.group(1), (
            f"{stage}: DATA_MIX=pdfa+idl didn't propagate --idl-shards"
        )
        # Two IDL fixture shards in the test environment.
        assert "idl-train-0001.tar" in m.group(1)
        assert "idl-train-0002.tar" in m.group(1)
        # Default IDL_WEIGHT=0.6 (paper-comparable; majority real).
        assert "--idl-weight 0.6" in m.group(1)


def test_phase_h_unknown_data_mix_rejected():
    """An unknown DATA_MIX value (typo) must hard-fail rather than
    silently fall back to a default."""
    out = _run_dry({"DATA_MIX": "pdfa+iddl"})    # typo
    assert "Unknown DATA_MIX=" in out


def test_phase_h_paper_preset_propagates():
    """PAGE_PRESET=paper: each stage gets --page-preset paper. The
    stage scripts will resolve to (2200, 1700) at run time; the chain
    just forwards the preset name."""
    out = _run_dry({"PAGE_PRESET": "paper"})
    for stage in ("stage1", "stage2", "stage3"):
        m = re.search(rf"^DRY_RUN \[{stage}\]: (.+)$", out, re.MULTILINE)
        assert m
        assert "--page-preset paper" in m.group(1)


def test_phase_h_idl_weight_override():
    """IDL_WEIGHT can be overridden from the chain (e.g. for an
    ablation comparing 0.6 vs 0.7)."""
    out = _run_dry({"DATA_MIX": "pdfa+idl", "IDL_WEIGHT": "0.7"})
    s2 = re.search(r"^DRY_RUN \[stage2\]: (.+)$", out, re.MULTILINE)
    assert s2
    assert "--idl-weight 0.7" in s2.group(1)


def test_phase_h_missing_idl_shards_fails_loudly():
    """When DATA_MIX=pdfa+idl is set but no IDL shards match the glob,
    chain exits with a clear error rather than silently running a
    PDFA-only training that the operator thought was pdfa+idl."""
    out = _run_dry({
        "DATA_MIX": "pdfa+idl",
        "IDL_SHARDS_GLOB": "data/raw/idl/idl-doesnotexist-*.tar",
    })
    assert "no IDL shards matched" in out
